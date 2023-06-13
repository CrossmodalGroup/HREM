import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from lib.loss import loss_select
from lib.mca import AdjacencyModel


def extra_parameters(parser):

    # loss function
    parser.add_argument('--base_loss', default='trip', type=str, help='the loss function for the initial embeddings.')
    parser.add_argument('--gnn_loss', default='trip', type=str, help='the loss function for the enhanced embeddings.')

    # warmup for training
    parser.add_argument('--warmup', default=8000, type=int, help='warmup iteration for instance-level interaction network')

    # Graph modelling on fragment-level 
    parser.add_argument('--residual_weight', default=0.8, type=float, help='the weight of residual operation for pooling')

    # Graph modelling on instance-level 
    parser.add_argument('--num_layers_enc', default=1, type=int, help='the num_layers of Transformer encoder')   
    parser.add_argument('--nhead', default=16, type=int, help='the num_head for Transformer encoder')
    parser.add_argument('--dropout', default=0.1, type=float, help='the dropout rate for Transformer encoder')
    parser.add_argument('--graph_lr_factor', default=1., type=float, help='the learning rate factor for the interaction model')

    # connection and relevance relation 
    parser.add_argument('--mask_weight', default=1.0, type=float, help='use extra weight for the attention matrix')  
    parser.add_argument('--threshold', default=0.5, type=float, help='give a threshold for the mask proportion, 0-1')
    parser.add_argument('--topk', default=10, type=int, help='the topk for the region-word pair selection')
    parser.add_argument('--reg_loss_weight', default=10, type=float, help='the values for the regularization loss')
    parser.add_argument('--norm_input', default=1, type=int, help='if use L2-norm embeddings as input')
    
    # loss function
    parser.add_argument('--cross_loss', default=1, type=int, help='if compute the loss for cross embeddings')

    return parser


# instance-level relation modeling
class GraphLoss(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.iter_count = 0
        self.embed_size = opt.embed_size

        # Initialize the dml objective function for embeddings learning.
        self.base_loss = loss_select(opt, loss_type=opt.base_loss)
        self.gnn_loss = loss_select(opt, loss_type=opt.gnn_loss)

        # the fusion interaction mechanism
        encoder_layer = nn.TransformerEncoderLayer(d_model=opt.embed_size, nhead=opt.nhead, 
                                                    dim_feedforward=opt.embed_size, dropout=opt.dropout)          
        self.gnn = nn.TransformerEncoder(encoder_layer, num_layers=opt.num_layers_enc)

        # construct the cross-embedding graph
        self.adj_model = AdjacencyModel(hidden_size=opt.embed_size, threshold=opt.threshold, topk=opt.topk, detach=True)
            

    def forward(self, img_emb, cap_emb, img_ids):
        
        # get latent features and embeddings
        # include the pre-pooling and after-pooling features
        img_feat, img_len, img_emb, img_emb_notnorm, img_emb_pre_pool = img_emb
        cap_feat, cap_len, cap_emb, cap_emb_notnorm, cap_emb_pre_pool = cap_emb

        bs = img_emb_notnorm.shape[0]
        assert img_emb_notnorm.shape[0] == cap_emb_notnorm.shape[0]

        num_loss = 0

        # basic matching loss
        base_loss = self.base_loss(img_emb, cap_emb, img_ids)
        num_loss += 1

        if self.iter_count >= self.opt.warmup:

            # get the connection relation and the relevance relation
            mask_weight = self.opt.mask_weight
            batch_c, batch_r, reg_loss = self.adj_model(img_emb, cap_emb, 
                                                        img_regions=img_emb_pre_pool, 
                                                        cap_words=cap_emb_pre_pool, 
                                                        img_len=img_len,
                                                        cap_len=cap_len,)
            # connection relation
            connect_mask = torch.cat((torch.cat((batch_c['i2i'], batch_c['i2t']), dim=1), 
                                    torch.cat((batch_c['t2i'], batch_c['t2t']), dim=1)), dim=0)

            # relevance relation
            relation_mask = torch.cat((torch.cat((batch_r['i2i'], batch_r['i2t']), dim=1), 
                                    torch.cat((batch_r['t2i'], batch_r['t2t']), dim=1)), dim=0)

            mask = mask_weight * relation_mask.masked_fill_(~connect_mask, float('-inf'))                                                                

            # concat mbeddings, batch as the dim=1
            if self.opt.norm_input:
                all_embs = torch.cat((img_emb, cap_emb), dim=0)
            else:
                all_embs = torch.cat((img_emb_notnorm, cap_emb_notnorm), dim=0)

            # get the instance-level relation modeling 
            all_embs_gnn = self.gnn(all_embs.unsqueeze(1), mask).squeeze(1)

            # get the enhanced embeddings
            img_emb_gnn, cap_emb_gnn = torch.split(all_embs_gnn, bs, dim=0)              

            # L2 normalization for the relation-enhanced embeddings
            img_emb_gnn = F.normalize(img_emb_gnn)
            cap_emb_gnn = F.normalize(cap_emb_gnn)

            # compute loss
            if self.opt.cross_loss:
                gnn_loss1 = self.gnn_loss(img_emb, cap_emb_gnn, img_ids)
                gnn_loss2 = self.gnn_loss(img_emb_gnn, cap_emb, img_ids)
                num_loss += 3
            else:
                gnn_loss1 = 0.
                gnn_loss2 = 0.
                num_loss += 1

            gnn_loss3 = self.gnn_loss(img_emb_gnn, cap_emb_gnn, img_ids)
            gnn_loss = gnn_loss1 + gnn_loss2 + gnn_loss3               
                
        else:
            gnn_loss = 0.
            reg_loss = 0.

        loss = (base_loss + gnn_loss) 
        
        loss += self.opt.reg_loss_weight * reg_loss

        self.iter_count += 1

        return loss


if __name__ == '__main__':

    pass

