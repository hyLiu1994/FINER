import numpy as np

import torch
import torch.nn as nn


def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)


class SAKTMODEL(nn.Module):
    def __init__(self, params):
        super(SAKTMODEL, self).__init__()
        n_skill=params.n_question
        max_seq=params.seqlen
        embed_dim=params.qa_embed_dim

        self.n_skill = n_skill
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=5, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim) 

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

        #self._reset_parameters()
    
    # def _reset_parameters(self):
    #     r"""Initiate parameters in the model."""
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    # def init_params(self):
    #     nn.init.kaiming_normal_(self.pred.weight)
    #     # nn.init.kaiming_normal_(self.read_embed_linear.weight)
    #     # nn.init.constant_(self.read_embed_linear.bias, 0)
    #     nn.init.constant_(self.pred, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.embedding.weight)
        nn.init.kaiming_normal_(self.e_embedding.weight)
        nn.init.kaiming_normal_(self.pos_embedding.weight)

    def forward(self, question_ids, x, label,t=-1):
        batch_size = question_ids.shape[0]
        seqlen = question_ids.shape[1]
        # print(question_ids)
        # question_ids:[bs,seq_len] x:[bs,seq_len]
        # print(label.shape)
        device = x.device        
        # src_pad_mask = (x == 0)
        # tgt_pad_mask = (question_ids == 0)
        # mask = src_pad_mask & tgt_pad_mask

        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)

        pos_x = self.pos_embedding(pos_id)
        x = x + pos_x
        # x [bs,s_len,embed]
        e = self.e_embedding(question_ids)
        # print(e.shape)
        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + e)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        att_output = att_output[:, :-1]
        zeros_padding = torch.zeros(att_output[:, 0].shape).cuda()
        att_output = torch.cat([zeros_padding.unsqueeze(1), att_output], 1)
        # print(att_output.shape)
        x = self.ffn(att_output)
        # x = self.dropout(x)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)
        output = self.sigmoid(x.squeeze(-1))
        # target_mask = (question_ids != 0)
        if t == -1:
            mask = label.ge(0)
        else:
            mask_first = torch.zeros(size=[1, seqlen]).cuda()
            mask_first[0][t] = 1
            mask_first = mask_first.repeat(batch_size, 1)
            # print(mask_first.shape)
            # print(label.ge(0).shape)
            mask = (label.ge(0) * mask_first).bool()  # [batch_size * seq_len, 1]

        output = torch.masked_select(output, mask)
        label = torch.masked_select(label, mask)
        criterion = nn.BCELoss()
        loss = criterion(output, label)

        return loss, output,label



