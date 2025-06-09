import torch
import torch.nn as nn
from Model.memory import DKVMN
import numpy as np
import Model.utils as utils

class DKVMNMODEL(nn.Module):

    def __init__(self, params, student_num=None):
        super(DKVMNMODEL, self).__init__()
        self.n_question = params.n_question
        self.batch_size = params.batch_size
        self.q_embed_dim = params.q_embed_dim
        self.qa_embed_dim = params.qa_embed_dim
        self.memory_size = params.memory_size
        self.memory_key_state_dim = params.memory_key_state_dim
        self.memory_value_state_dim = params.memory_value_state_dim
        self.final_fc_dim = params.final_fc_dim
        self.student_num = student_num

        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.q_embed_dim, self.final_fc_dim, bias=True)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)
        self.init_memory_key = nn.Parameter(torch.randn(self.memory_size, self.memory_key_state_dim))
        self.init_memory_value = nn.Parameter(torch.randn(self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal_(self.init_memory_key)
        nn.init.kaiming_normal_(self.init_memory_value)

        self.mem = DKVMN(memory_size=self.memory_size,
                   memory_key_state_dim=self.memory_key_state_dim,
                   memory_value_state_dim=self.memory_value_state_dim, init_memory_key=self.init_memory_key)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(params.batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

    def forward(self, q_data, qa_data, target,t=-1, student_id=None, annotated_qa=None):
        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        slice_q_data = torch.chunk(q_data, seqlen, 1)
        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)

        value_read_content_l = []
        input_embed_l = []
        for i in range(seqlen):
            ## Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)
            if_memory_write = slice_q_data[i].squeeze(1).ge(1)
            if_memory_write = utils.varible(torch.FloatTensor(if_memory_write.data.tolist()), 1)

            ## Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)
            ## Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            new_memory_value = self.mem.write(correlation_weight, qa, if_memory_write)


        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)

        predict_input = torch.cat([all_read_value_content, input_embed_content], 2)
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(batch_size*seqlen, -1)))

        pred = self.predict_linear(read_content_embed)
        target_1d = target                   # [batch_size * seq_len, 1]
        if t == -1:
            mask = target_1d.ge(0)
        else:
            mask_first = torch.zeros(size=[seqlen, 1]).cuda()
            mask_first[t] = 1
            mask_first = mask_first.repeat(batch_size, 1)
            mask = (target_1d.ge(0) * mask_first).bool()  # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)           # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        if annotated_qa is not None:
            annotated_output = []
            type_num = max(annotated_qa.unique())
            for i in range(1, type_num + 1):
                annotated_output.append([torch.sigmoid(torch.masked_select(pred_1d, (annotated_qa == i).view(-1, 1) & mask)).detach().cpu(),
                                       torch.masked_select(target_1d, (annotated_qa == i).view(-1, 1) & mask).detach().cpu()])

            return loss, torch.sigmoid(filtered_pred), filtered_target, torch.sigmoid(pred).reshape(batch_size, seqlen), annotated_output
        else:
            return loss, torch.sigmoid(filtered_pred), filtered_target, torch.sigmoid(pred).reshape(batch_size, seqlen)
