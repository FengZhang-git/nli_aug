

import math
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F


from transformers import BertTokenizer, BertConfig, BertModel



class BertEncoder(nn.Module):

    def __init__(self, args):
        super(BertEncoder, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(args.fileVocab, do_basic_tokenize=True)
        config = BertConfig.from_json_file(args.fileModelConfig)   
        self.bert = BertModel.from_pretrained(args.fileModel,config=config)
        

        self.device = torch.device('cuda', args.numDevice)
        torch.cuda.set_device(self.device)

        if args.numFreeze > 0:
            self.freeze_layers(args.numFreeze)

        self.bert.cuda()


    def freeze_layers(self, numFreeze):
        unfreeze_layers = ["pooler"]
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer."+str(i))

        for name ,param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
    
    def forward(self, text):

        tokenizer = self.tokenizer(
        text,
        padding = True,
        truncation = True,
        max_length = 250,
        return_tensors='pt' 
        )
       
        input_ids = tokenizer['input_ids'].to(self.device)
        token_type_ids = tokenizer['token_type_ids'].to(self.device)
        attention_mask = tokenizer['attention_mask'].to(self.device)

        outputs = self.bert(
              input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
              )
        token_word_tensor = outputs[0]
        text_em = token_word_tensor.mean(dim=1)
       
       
        return text_em
      



class ModelManager(nn.Module):

    def __init__(self, args):
        super(ModelManager, self).__init__()

        self.encoder = BertEncoder(args)
        self.hidden_dim = 128
        

        self.pos_emb = nn.Parameter(torch.Tensor(args.max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim,
                                                   nhead=args.num_heads, activation='gelu')
        self.inte_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_trans_layers)
        
    def forward(self, text, class_labels):
        text_em = self.encoder(text)
        
        return text_em
    
   

   # visiualization
    def get_word_embeddings(self, text):
         word_tensor = self.encoder(text)

         return word_tensor

    def integrate(self, text_em):
        # text_em: T, bsz, dim
        bsz = text_em.shape[1]
       
        bsz_pos_emb = self.pos_emb.unsqueeze(1)
        bsz_pos_emb = bsz_pos_emb.repeat(1, bsz, 1)

        x = text_em + bsz_pos_emb
       

        x = self.inte_encoder(x)
        x = x.permute(1, 0, 2)  # bsz, T, dim
        
        # average
        feat = x.mean(dim=1)
        
       
       
        # bsz, dim
        return feat
