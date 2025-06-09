import torch
import torch.nn as nn
import numpy as np
import Model.utils as utils
from Model import Attention

class DKTMODEL(nn.Module):

    def __init__(self, params, student_num=None):
        super(DKTMODEL, self).__init__()
        self.params = params
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
        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.q_embed_dim, self.final_fc_dim,
                                           bias=True).cuda()
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)

        self.lstm_kt = nn.LSTM(self.qa_embed_dim, self.memory_value_state_dim, 1, batch_first=True).cuda()

        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.predict_linear.bias, 0)
        nn.init.constant_(self.read_embed_linear.bias, 0)

    def init_embeddings(self):

        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

    # similar_user_future_sequence [batch_size*N, sequence_len, similar_user_num, future_sequence_len]
    def forward(self, q_data, qa_data, target, t=-1, isTrain=True, student_id=None, annotated_qa=None):
        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        # q_embed_data [batch_size, seqlen, q_embed_dim]
        q_embed_data = self.q_embed(q_data)

        # qa_embed_data [batch_size, seqlen, qa_embed_dim]
        qa_embed_data = self.qa_embed(qa_data)

        all_read_value_content, (h_n, c_n) = self.lstm_kt.forward(qa_embed_data)
        all_read_value_content = all_read_value_content[:, :-1]
        zeros_padding = torch.zeros(all_read_value_content[:, 0].shape).cuda()
        all_read_value_content = torch.cat([zeros_padding.unsqueeze(1), all_read_value_content], 1)

        # predict_input [batch_size, seqLen, memory_dim+num_attention_heads*similar_user_num*attention_head_size + q_embed_size]
        predict_input = torch.cat([all_read_value_content, q_embed_data], 2)

        # read_content_embed [batch_size * seqLen, final_fc_dim]
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(batch_size * seqlen, -1)))

        # pred [batch_size * seqLen, 1]
        pred = self.predict_linear(read_content_embed)

        # target target_1d: [batch_size * seq_len, 1] [batch_size * seq_len, 1]
        target_1d = target  # [batch_size * seq_len, 1]
        if t == -1:
            mask = target_1d.ge(0)
        else:
            mask_first = torch.zeros(size=[seqlen, 1]).cuda()
            mask_first[t] = 1
            mask_first = mask_first.repeat(batch_size, 1)
            mask = (target_1d.ge(0) * mask_first).bool()  # [batch_size * seq_len, 1]
        # print(mask)
        # pred_1d = predicts.view(-1, 1)           # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)  # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)
        if annotated_qa is not None:
            annotated_output = []
            type_num = max(annotated_qa.unique())
            for i in range(1, type_num + 1):
                annotated_output.append([torch.sigmoid(torch.masked_select(pred_1d, (annotated_qa == i).view(-1, 1) & mask)).detach().cpu(),
                                         torch.masked_select(target_1d, (annotated_qa == i).view(-1, 1) & mask).detach().cpu()])

            return loss, torch.sigmoid(filtered_pred), filtered_target, torch.sigmoid(pred).reshape(self.params.batch_size, self.params.seqlen), annotated_output
        else:
            return loss, torch.sigmoid(filtered_pred), filtered_target, torch.sigmoid(pred).reshape(self.params.batch_size, self.params.seqlen)
