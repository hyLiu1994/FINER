import numpy as np

import torch
import torch.nn as nn



class SAKT(nn.Module):
    def __init__(self, params):
        super(SAKT, self).__init__()
        self.n_skill = params.n_question
        self.embed_dim = params.qa_embed_dim

        self.embedding = nn.Embedding(2 * params.n_question + 1, params.qa_embed_dim).cuda()
        self.pos_embedding = nn.Embedding(params.seq_len, params.qa_embed_dim).cuda()
        self.e_embedding = nn.Embedding(params.n_question + 1, params.qa_embed_dim).cuda()

        self.multi_att = nn.MultiheadAttention(embed_dim=params.qa_embed_dim, num_heads=8, dropout=0.2).cuda()

        # self._reset_parameters()

    def future_mask(self,seq_length):
        future_mask = torch.tril(torch.ones((seq_length, seq_length)))
        return future_mask

    def forward(self, q_data, qa_data):

        x = self.embedding(qa_data)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).cuda()
        pos_x = self.pos_embedding(pos_id)
        x = x + pos_x

        e = self.e_embedding(q_data)

        att_mask = self.future_mask(x.size(0)).cuda()
        # print(att_mask)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = att_output[:, :-1]
        zeros_padding = torch.zeros(att_output[:, 0].shape).cuda()
        att_output = torch.cat([zeros_padding.unsqueeze(1), att_output], 1)
        # print(att_output)

        return att_output

