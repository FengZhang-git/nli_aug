import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from collections import defaultdict
import time
import math

from sklearn import metrics



def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototype_loss(support_em, query_em, nway, kshot, qshot):

    support_em = support_em.view(nway, kshot, support_em.shape[1])
    prototypes = support_em.mean(dim=1)
    
    dists = euclidean_dist(query_em, prototypes)
    
   

    log_p_y = F.log_softmax(-dists, dim=1).view(nway, qshot, -1)

    
    target_inds = torch.arange(0, nway)
    target_inds = target_inds.view(nway, 1, 1)
    target_inds = target_inds.expand(nway, qshot, 1).long()
    target_inds = target_inds.cuda()
    

    loss_pn = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_pn = y_hat.eq(target_inds.squeeze()).float().mean()

    logs = F.softmax(-dists,dim=1) # query size, nway

    return loss_pn, acc_pn, logs

def get_similarity_matrix(x, rbf_scale):
    b, c = x.size()
    sq_dist = ((x.view(b, 1, c) - x.view(1, b, c))**2).sum(-1) / np.sqrt(c)
    mask = sq_dist != 0
    sq_dist = sq_dist / sq_dist[mask].std()
    weights = torch.exp(-sq_dist * rbf_scale)
    mask = torch.eye(weights.size(1), dtype=torch.bool, device=weights.device)
    weights = weights * (~mask).float()
    return weights

def pairwise_loss(support_em, nway, kshot):
   
    support_em = support_em.view(nway, kshot, support_em.shape[1])
    prototypes = support_em.mean(dim=1)

    weights = get_similarity_matrix(prototypes, rbf_scale=1)
    loss = weights.sum()/(nway * nway - nway)
    return loss

    

def get_weights(alpha, logits, mask, probality, labels):
    ones = torch.ones_like(mask)
    neg_mask = ones - mask
    sim_prob = torch.exp(logits) * neg_mask
    sim_prob = sim_prob / sim_prob.sum(1, keepdim=True)

    class_prob = []
    bsz = logits.shape[0]
    for i in range(bsz):
        tmp = []
        each_prob = probality[i] # n-way
        each_label = labels[i]
        for j in range(bsz):
            tmp.append(each_prob[labels[j]])
        tmp = torch.stack(tmp)
        class_prob.append(tmp)
    class_prob = torch.stack(class_prob)
    class_prob = class_prob * neg_mask

    weights = alpha * sim_prob + (1 - alpha) * class_prob

    # positive
    weights = weights + mask

    return weights


def get_prob(support_em, query_em, nway, kshot):

    features = torch.cat((support_em, query_em), dim=0)

    support_em = support_em.view(nway, kshot, support_em.shape[1])
    prototypes = support_em.mean(dim=1)
    dists = euclidean_dist(features, prototypes)
    
    logs = F.softmax(-dists,dim=1) # query size, nway

    return logs


def contrastive_loss(integrate_support_em, integrate_query_em, nway, kshot, qshot, prob, alpha, temperature=0.1):
    
    features = torch.cat((integrate_support_em, integrate_query_em), dim=0)
    features = F.normalize(features, p=2, dim=1)
 

    target_inds = torch.arange(0, nway).view(nway, 1)
    target_inds = target_inds.repeat(1, kshot).long()
    target_inds = target_inds.view(-1)
    test_inds = torch.arange(0, nway).view(nway, 1)
    test_inds = test_inds.repeat(1, qshot).long()
    test_inds = test_inds.view(-1)
    labels = torch.cat((target_inds, test_inds), dim=0)
    
    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().cuda()

  
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T),
        temperature)
   
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    weights = get_weights(alpha, logits, mask, prob, labels)

   
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).cuda(),
        0
    )
    
    mask = mask * logits_mask
 
   
    exp_logits = torch.exp(logits) * (logits_mask * weights)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    if torch.any(torch.isnan(log_prob)):
        raise ValueError("Log_prob has nan!")
    

   
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1))

    if torch.any(torch.isnan(mean_log_prob_pos)):
        raise ValueError("mean_log_prob_pos has nan!")

   
    loss = - mean_log_prob_pos
   
    if torch.any(torch.isnan(loss)):
            raise ValueError("loss has nan!")
    loss = loss.mean()

    
    return loss

class MixedLoss(nn.Module):
    def __init__(self, beta=0.1, temperature=0.1, gamma=0.1, alpha=0.5):
        super(MixedLoss, self).__init__()
        
        
        self.beta = beta
        self.temperature = temperature
        self.gamma = gamma
        self.alpha = alpha
    

    def forward(self,  tasks_em, nway, kshot, qshot, repeat, model):
        
       
        support_size = nway * kshot
        support_em = tasks_em[:, :support_size, :]
        query_em = tasks_em[:, support_size:, :]

        original_support_em = support_em[0]
        original_query_em = query_em[0]

       
        integrate_support_em = model.integrate(support_em) # task_nember, support_size, dim
        integrate_query_em = model.integrate(query_em)

        

        integrate_loss, acc, integrate_prob = prototype_loss(integrate_support_em, integrate_query_em, nway, kshot, qshot)
        

       
        pair_loss = pairwise_loss(integrate_support_em, nway, kshot)
        
        prob = get_prob(integrate_support_em, integrate_query_em, nway, kshot)
        con_loss = contrastive_loss(integrate_support_em, integrate_query_em, nway, kshot, qshot, prob, self.alpha, self.temperature)

        print("Integrated loss: %f, Integrated Acc: %f, Pair loss: %f, Con loss: %f" % (integrate_loss, acc, pair_loss, con_loss))
        
        loss = integrate_loss + self.beta * pair_loss + self.gamma * con_loss
       

        

        return loss, acc
       
    
