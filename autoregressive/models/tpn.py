#-------------------------------------
# Project: Transductive Propagation Network for Few-shot Learning
# Date: 2019.1.11
# Author: Yanbin Liu
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        # 生成正弦位置编码矩阵 [max_len, dim]
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer("pe", pe, persistent=False)  # 不训练，不存 ckpt 里

    def forward(self, x):
        """
        x: [bs, seq_len, dim]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

"""
class CrossAttentionBlock(nn.Module): 
    def __init__(self, dim, num_heads): 
        super().__init__() 
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True) 
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True) 
        self.ffn = nn.Sequential( nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim) ) 
        self.norm1 = nn.LayerNorm(dim) 
        self.norm2 = nn.LayerNorm(dim) 
        self.norm3 = nn.LayerNorm(dim) 
    def forward(self, x, context): 
        # Self-Attention 
        h, self_attn_w = self.self_attn(x, x, x) 
        x = self.norm1(x + h) 
        # Cross-Attention 
        h, cross_attn_weights = self.cross_attn(x, context, context) 
        x = self.norm2(x + h) 
        # FFN 
        h = self.ffn(x) 
        x = self.norm3(x + h)
        return x, cross_attn_weights, self_attn_w


class CrossAttentionTransformer(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_heads, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # 先把 logits/probabilities 投影到 hidden_dim
        #self.logits_proj = nn.Linear(vocab_size, hidden_dim)

        # 加可学习的位置嵌入
        #self.pos_embedding = nn.Embedding(256, hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)
        # 原来的 cross-attention transformer
        self.layers = nn.ModuleList([ CrossAttentionBlock(self.hidden_dim, num_heads) for _ in range(num_layers) ]) 

        # 再投影回 vocab_size
        #self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, context):
        # 投影到 hidden_dim
        #x = self.logits_proj(x)  # [bs, seq_len, hidden_dim]
        bs, seq_len, _ = x.shape


        #pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(bs, 1)

        #x = x + self.pos_embedding(pos_ids)

        x = self.pos_encoding(x)
        # cross-attention 融合上下文
        for layer in self.layers: 

            x , cross_attn_weights, self_attn_w= layer(x, context) 
            print(cross_attn_weights)
            print(self_attn_w)
        return x
"""

class CrossAttentionTransformer(nn.Module):
    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        #self.pos_encoding = SinusoidalPositionalEncoding(dim)

    def forward(self, query, context):
        # query: [bs, Lq, dim]   (先验 embedding)
        # context: [bs, Lc, dim] (hidden_states)
        #query = self.pos_encoding(query)
        out, attn_weights = self.cross_attn(query, context, context,
                                            need_weights=True,
                                            average_attn_weights=False)
        out = self.norm(query + out)  # 残差 + norm 更稳定
        return out, attn_weights



class RelationNetwork1D(nn.Module):
    """
    输入 x: [bs, seq_len, feat_dim]
    输出 sigma: [bs, seq_len]  （每个位置一个正数）
    """
    def __init__(self, feat_dim, proj_dim=256, hidden_dim=128, use_projection=True):
        super().__init__()
        self.use_projection = use_projection
        if use_projection:
            # 把 feat_dim 投影到较小的通道数，避免 Conv1d 的 in_channels 太大
            self.input_proj = nn.Linear(feat_dim, proj_dim)
            in_ch = proj_dim
        else:
            self.input_proj = nn.Identity()
            in_ch = feat_dim

        # Conv1d 流水线：in_ch -> hidden_dim -> hidden_dim -> 1
        self.conv1 = nn.Conv1d(in_ch, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 最终用 1x1 conv 输出每位置的 scalar
        self.out_conv = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x, rn):
        """
        x: [bs, seq_len, feat_dim]
        return: sigma: [bs, seq_len] （正数）
        """
        bs, seq_len, feat_dim = x.shape

        # 1) 可选投影： [bs, seq_len, proj_dim]
        if self.use_projection:
            x_proj = self.input_proj(x)   # [bs, seq_len, proj_dim]
        else:
            x_proj = x

        # 2) 转为 Conv1d 要求的格式 [bs, channels, seq_len]
        x_conv = x_proj.transpose(1, 2)  # [bs, in_ch, seq_len]

        # 3) conv 层
        h = F.relu(self.bn1(self.conv1(x_conv)))   # [bs, hidden_dim, seq_len]
        h = F.relu(self.bn2(self.conv2(h)))        # [bs, hidden_dim, seq_len]

        # 4) 输出每位置的 scalar -> [bs, 1, seq_len]
        out = self.out_conv(h)                     # [bs, 1, seq_len]

        # 5) 转回 [bs, seq_len]
        sigma = out.squeeze(1)                       # [bs, seq_len]

        return sigma



class Prototypical(nn.Module):
    """Main Module for prototypical networlks"""
    def __init__(self, args):
        super(Prototypical, self).__init__()
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.h_dim, self.z_dim = args['h_dim'], args['z_dim']

        self.args = args
        self.encoder = CNNEncoder(args)

    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1]
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)

        inp   = torch.cat((support,query), 0)
        emb   = self.encoder(inp) # 80x64x5x5
        emb_s, emb_q = torch.split(emb, [num_classes*num_support, num_classes*num_queries], 0)
        emb_s = emb_s.view(num_classes, num_support, 1600).mean(1)
        emb_q = emb_q.view(-1, 1600)
        emb_s = torch.unsqueeze(emb_s,0)     # 1xNxD
        emb_q = torch.unsqueeze(emb_q,1)     # Nx1xD
        dist  = ((emb_q-emb_s)**2).mean(2)   # NxNxD -> NxN

        ce = nn.CrossEntropyLoss().cuda(0)
        loss = ce(-dist, torch.argmax(q_labels,1))
        ## acc
        pred = torch.argmax(-dist,1)
        gt   = torch.argmax(q_labels,1)
        correct = (pred==gt).sum()
        total   = num_queries*num_classes
        acc = 1.0 * correct.float() / float(total)

        return loss, acc


class LabelPropagation(nn.Module):
    """Label Propagation"""
    def __init__(self, args, device):
        super(LabelPropagation, self).__init__()

        self.args = args

        self.relation = RelationNetwork1D(self.args.feat_dim)

        #self.cross_attention = CrossAttentionTransformer(self.args.feat_dim, self.args.vocab_size, 12, num_layers=1)

        self.cross_attention = CrossAttentionTransformer(self.args.feat_dim, 12)
        self.device = device
        self.pos_encoding = SinusoidalPositionalEncoding(self.args.feat_dim)

        if   self.args.rn == 300:   # learned sigma, fixed alpha
            self.alpha = torch.tensor([args.alpha], requires_grad=False).to(self.device)
        elif self.args.rn == 30:    # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([args.alpha]).to(self.device), requires_grad=True)

    def forward(self, generated_embeddings, ungenerated_embeddings, hidden_states, labels):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init  构建embedding
        eps = 1e-6

        N = labels.shape[1]

        idx = generated_embeddings.shape[1]
        # to do: hidden_state融合位置信息

        emb_all = torch.cat([generated_embeddings, ungenerated_embeddings], dim=1)

        emb_all_pos = self.pos_encoding(emb_all)

        ungenerated_embeddings_pos = emb_all_pos[:, idx:, :]

        ungenerated_embeddings_pos , _ = self.cross_attention(ungenerated_embeddings_pos, hidden_states)

        #attn_weights =  attn_weights.cpu()

        emb_all_pos = torch.cat([emb_all_pos[:, :idx, :], ungenerated_embeddings_pos], dim=1)

        emb_all = emb_all_pos

        # Step2: Graph Construction
        ## sigmma
        if self.args.rn in [30,300]:
            self.sigma   = self.relation(emb_all, self.args.rn)
            
            ## W
            emb_all = emb_all / (self.sigma.unsqueeze(-1)+eps) # N*d
            W = torch.cdist(emb_all, emb_all, p=2)
            #W = (W ** 2) / emb_all.size(2)
            W = (W ** 2) / emb_all.size(2)
            W       = torch.exp(-W/2)

        ## keep top-k values


        ## yzh发现问题，需要排查！！！
        if self.args.k>0:
            topk, indices = torch.topk(W, self.args.k, dim=-1)
            mask = torch.zeros_like(W)
            mask = mask.scatter(2, indices, 1)
            mask = ((mask+mask.transpose(1, 2))>0).type(torch.float32)      # union, kNN graph
            #mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W    = W*mask

        ## normalize
        D       = W.sum(-1)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        S = D_sqrt_inv.unsqueeze(-1) * W * D_sqrt_inv.unsqueeze(1)

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        labels = F.one_hot(labels.to(torch.long), self.args.vocab_size + 1)
        labels = labels[:, :, :self.args.vocab_size].float()



        # propagator = torch.inverse(torch.eye(N).to(self.device)-self.alpha*S+eps)

        # results  = torch.matmul(propagator, labels)
        I = torch.eye(N, device=self.device).unsqueeze(0).expand(S.size(0), -1, -1)
        propagator = torch.inverse(I - self.alpha * S + eps * I)
        results = torch.bmm(propagator, labels)

        return results



