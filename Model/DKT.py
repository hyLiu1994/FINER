import torch
import torch.nn as nn
# from Model.memory import DKVMN
# import numpy as np
# import utils as utils
# from Model import Attention

class DKT(nn.Module):

    def __init__(self, params, student_num=None):
        super(DKT, self).__init__()
        self.n_question = params.n_question
        self.batch_size = params.batch_size
        self.q_embed_dim = params.q_embed_dim
        self.qa_embed_dim = params.qa_embed_dim
        self.memory_value_state_dim = params.qa_embed_dim
        self.final_fc_dim = params.final_fc_dim
        self.student_num = student_num

        self.mseloss = torch.nn.MSELoss()
        # Need Think
        # attention_head_size = int(self.learning_trend_embed_dim/self.num_attention_head)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)

        self.lstm_kt = nn.LSTM(self.qa_embed_dim, self.memory_value_state_dim, 1, batch_first=True).cuda()

        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.constant_(self.predict_linear.bias, 0)

    def init_embeddings(self):

        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

    # similar_user_future_sequence [batch_size*N, sequence_len, similar_user_num, future_sequence_len]
    def forward(self, q_data, qa_data):

        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        # q_embed_data [batch_size, seqlen, q_embed_dim]
        q_embed_data = self.q_embed(q_data)

        # qa_embed_data [batch_size, seqlen, qa_embed_dim]
        qa_embed_data = self.qa_embed(qa_data)

        all_read_value_content, (h_n, c_n)  = self.lstm_kt.forward(qa_embed_data)
        all_read_value_content = all_read_value_content[:, :-1]
        zeros_padding = torch.zeros(all_read_value_content[:, 0].shape).cuda()
        all_read_value_content = torch.cat([zeros_padding.unsqueeze(1), all_read_value_content], 1)

        return all_read_value_content