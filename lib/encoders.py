import os
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
import logging

from lib.mlp import FC_MLP

logger = logging.getLogger(__name__)


# 'True' represents to be masked （Do not participate in the calculation of attention）
# 'False' represents not to be masked
def padding_mask(embs, lengths):

    mask = torch.ones(len(lengths), embs.shape[1], device=lengths.device)
    for i in range(mask.shape[0]):
        end = int(lengths[i])
        mask[i, :end] = 0.

    return mask.bool()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    _x, index = x.topk(k, dim=dim)
    return _x

    
# uncertain length
def maxk_pool1d_var(x, dim, k, lengths):
    # k >= 1
    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):
        # keep use all number of features
        k = min(k, int(lengths[idx].item()))

        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim-1)[0]

        max_k_i = maxk_pool1d(tmp, dim-1, k)
        results.append(max_k_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


def avg_pool1d_var(x, dim, lengths):

    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):

        # keep use all number of features
        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim-1)[0]
        avg_i = tmp.mean(dim-1)

        results.append(avg_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


class Maxk_Pooling_Variable(nn.Module):
    def __init__(self, dim=1, k=2):
        super(Maxk_Pooling_Variable, self).__init__()

        self.dim = dim
        self.k = k

    def forward(self, features, lengths):

        pool_weights = None
        pooled_features = maxk_pool1d_var(features, dim=self.dim, k=self.k, lengths=lengths)
        
        return pooled_features, pool_weights


class Avg_Pooling_Variable(nn.Module):
    def __init__(self, dim=1):
        super(Avg_Pooling_Variable, self).__init__()
        
        self.dim = dim

    def forward(self, features, lengths):

        pool_weights = None
        pooled_features = avg_pool1d_var(features, dim=self.dim, lengths=lengths)
        
        return pooled_features, pool_weights


def get_text_encoder(opt, embed_size, no_txtnorm=False): 
    
    text_encoder = EncoderText_BERT(opt, embed_size, no_txtnorm=no_txtnorm)
    
    return text_encoder


def get_image_encoder(opt, img_dim, embed_size, no_imgnorm=False):
    
    img_enc = EncoderImageAggr(opt, img_dim, embed_size, no_imgnorm)
    
    return img_enc


class EncoderImageAggr(nn.Module):
    def __init__(self, opt, img_dim=2048, embed_size=1024, no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()

        self.opt = opt

        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        
        # B * N * 2048 -> B * N * 1024
        # N = 36 for region features
        self.fc = FC_MLP(img_dim, embed_size // 2, embed_size, 2, bn=True)           
        self.fc.apply(init_weights)

        # fragment-level relation modeling (for local features)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=opt.nhead,
                                                   dim_feedforward=embed_size, dropout=opt.dropout)
        self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

        # pooling function
        self.graph_pool = Avg_Pooling_Variable()
        self.gpool = Maxk_Pooling_Variable()

    def forward(self, images, image_lengths, graph=False):

        img_emb = self.fc(images)

        # initial visual embedding
        img_emb_res, _ = self.gpool(img_emb, image_lengths)

        img_emb_pre_pool = img_emb

        # fragment-level relation modeling for region features

        # get padding mask
        src_key_padding_mask = padding_mask(img_emb, image_lengths)

        # switch the dim
        img_emb = img_emb.transpose(1, 0)
        img_emb = self.aggr(img_emb, src_key_padding_mask=src_key_padding_mask)
        img_emb = img_emb.transpose(1, 0)

        # enhanced visual embedding
        img_emb, _  = self.graph_pool(img_emb, image_lengths)

        # the final global embedding
        img_emb =  self.opt.residual_weight * img_emb_res + (1-self.opt.residual_weight) * img_emb

        img_emb_notnorm = img_emb
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        if graph:
            return images, image_lengths, img_emb, img_emb_notnorm, img_emb_pre_pool
        else:
            return img_emb


# Language Model with BERT backbone
class EncoderText_BERT(nn.Module):
    def __init__(self, opt, embed_size=1024, no_txtnorm=False):
        super(EncoderText_BERT, self).__init__()

        self.opt = opt

        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # self.bert = BertModel.from_pretrained(opt.bert_path)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # backbone features -> embbedings
        self.linear = nn.Linear(768, embed_size)
        
        # relation modeling for local feature
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=opt.nhead,
                                                   dim_feedforward=embed_size, dropout=opt.dropout)
        self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

        # pooling function
        self.graph_pool = Avg_Pooling_Variable()
        self.gpool = Maxk_Pooling_Variable()


    def forward(self, x, lengths, graph=False):

        # Embed word ids to vectors
        # pad 0 for redundant tokens in previous process
        bert_attention_mask = (x != 0).float()

        # all hidden features, D=768 in bert-base model
        # attention_mask： Mask to avoid performing attention on padding token indices.
        # bert_output[0] is the last/final hidden states of all tokens
        # bert_output[1] is the hidden state of [CLS] + one fc layer + Tanh, can be used for classification tasks.

        # N = max_cap_lengths, D = 768
        bert_emb = self.bert(input_ids=x, attention_mask=bert_attention_mask)[0]  # B x N x D
        cap_len = lengths

        # B x N x embed_size
        cap_emb = self.linear(bert_emb)

        # initial textual embedding
        cap_emb_res, _ = self.gpool(cap_emb, cap_len)

        cap_emb_pre_pool = cap_emb

        # fragment-level relation modeling for word features
        
        # get padding mask
        src_key_padding_mask = padding_mask(cap_emb, cap_len)
        
        # switch the dim
        cap_emb = cap_emb.transpose(1, 0)
        cap_emb = self.aggr(cap_emb, src_key_padding_mask=src_key_padding_mask)
        cap_emb = cap_emb.transpose(1, 0)

        # enhanced textual embedding
        cap_emb, _ = self.graph_pool(cap_emb, cap_len)

        cap_emb = self.opt.residual_weight * cap_emb_res + (1-self.opt.residual_weight) * cap_emb 

        # the final global embedding
        cap_emb_notnorm = cap_emb
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        if graph:
            return bert_emb, cap_len, cap_emb, cap_emb_notnorm, cap_emb_pre_pool
        else:
            return cap_emb


if __name__ == '__main__':

    pass
