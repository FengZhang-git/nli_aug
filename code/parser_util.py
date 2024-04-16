# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataFile',
                        type=str,
                        help='path to dataset')
    parser.add_argument('--allFile',
                        type=str,
                        help='path to all aug text')

    parser.add_argument('--fileVocab',
                        type=str,
                        help='path to pretrained model vocab')
        
    parser.add_argument('--fileModelConfig',
                        type=str,
                        help='path to pretrained model config')

    parser.add_argument('--fileModel',
                        type=str,
                        help='path to pretrained model')

    parser.add_argument('--fileModelSave',
                        type=str,
                        help='path to save model')
    

    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)
    parser.add_argument('--numNWay',
                        type=int,
                        help='number of classes per episode',
                        default=5)
    parser.add_argument('--numKShot',
                        type=int,
                        help='number of instances per class',
                        default=5)

    parser.add_argument('--numQShot',
                        type=int,
                        help='number of querys per class',
                        default=5)
    
    parser.add_argument('--episodeTrain',
                        type=int,
                        help='number of tasks per epoch in training process',
                        default=200)

    parser.add_argument('--episodeTest',
                        type=int,
                        help='number of tasks per epoch in testing process',
                        default=100)

    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

  

    parser.add_argument('--numFreeze',
                        type=int,
                        help='number of freezed layers in pretrained model, default=12',
                        default=12)

    parser.add_argument('--numDevice',
                        type=int,
                        help='id of gpu ',
                        default=0)

    parser.add_argument('--warmup_steps',
                        type=int,
                        help='num of warmup_steps',
                        default=100)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='ratio of decay',
                        default=0.2)

    parser.add_argument('--dropout_rate',
                        type=float,
                        help='ratio of dropout',
                        default=0.1)

    parser.add_argument('--max_posts',
                        type=int,
                        help='number of maxsium augmented text',
                        default=9)

    parser.add_argument('--num_heads',
                        type=int,
                        help='number of transformer encoder head',
                        default=8)
    
    parser.add_argument('--num_trans_layers',
                        type=int,
                        help='number of transformer layers',
                        default=2)

    parser.add_argument('--beta',
                        type=float,
                        help='ratio of pair loss',
                        default=0.1)

    parser.add_argument('--gamma',
                        type=float,
                        help='ratio of contrastive loss',
                        default=0.3)           

    parser.add_argument('--tolerate',
                        type=int,
                        help='number of early stop',
                        default=10)

    return parser
