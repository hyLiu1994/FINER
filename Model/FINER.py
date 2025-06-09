import torch
import torch.nn as nn
from Model.DKVMN import MDKVMN
from Model.DKT import DKT
from Model.SAKT import SAKT
from Model.AKT import AKT
from Model.RKT import RKT
from Model.LSTMA import LSTMA
import numpy as np
import Model.utils as utils

class FINERMODEL(nn.Module):

    def __init__(self, params, student_num=None):
        super(FINERMODEL, self).__init__()
        self.params = params

        self.kt_output_dim = self.params.qa_embed_dim
        if 'DKT' in params.model_type:
            self.kt_model = DKT(params,student_num=student_num)
        elif 'DKVMN' in params.model_type:
            self.kt_model = MDKVMN(params,student_num=student_num)
        elif 'SAKT' in params.model_type:
            self.kt_model = SAKT(params)
        elif 'AKT' in params.model_type:
            self.kt_model = AKT(n_question=params.n_question,final_fc_dim=params.final_fc_dim,n_pid=-1,d_model=200,n_blocks=1,kq_same=1,dropout=0.05,model_type='akt'
                             ,n_heads=8,d_ff=2048,l2=1e-5,separate_qa=False)
        elif 'RKT' in params.model_type:
            self.kt_model = RKT(num_items=params.n_question,embed_size=200,num_attn_layers=1,num_heads=5,encode_pos=False,max_pos=10,drop_prob=0.2)
        elif 'LSTMA' in params.model_type:
            self.kt_model = LSTMA(params, student_num=student_num)

        self.length_embed_dim = self.idx_embed_dim = self.frequency_embed_dim = params.qa_embed_dim
        self.hatT_embed_dim = params.qa_embed_dim

        # Representation of FPTs
        # idx_num = 5697460
        self.length_embed_layer = nn.Embedding(30, self.length_embed_dim, padding_idx=0).cuda()
        # self.idx_embed_layer = nn.Embedding(idx_num, self.idx_embed_dim, padding_idx=0).cuda()
        self.frequency_embed_layer = nn.Embedding(300, self.frequency_embed_dim, padding_idx=0).cuda()
        # self.fc_length_idx = nn.Linear(self.length_embed_dim + self.idx_embed_dim, self.hatT_embed_dim, bias=True).cuda()
        self.fc_length_idx = nn.Linear(self.length_embed_dim, self.hatT_embed_dim, bias=True).cuda()
        self.FPTs_weight_linear = nn.Linear(self.frequency_embed_dim, 1, bias=True).cuda()

        # question embedding
        self.q_embed_layer = nn.Embedding(params.n_question + 1, params.q_embed_dim, padding_idx=0).cuda()

        self.knowledge_embedding_dim = self.memory_value_state_dim = params.qa_embed_dim
        # fusion moudle
        self.linear_fusion = nn.Linear((self.kt_output_dim+1)*(self.hatT_embed_dim + 1), self.knowledge_embedding_dim, bias=True).cuda()
        self.lstm_fusion = nn.LSTM(self.knowledge_embedding_dim, self.memory_value_state_dim, 1, batch_first=True).cuda()

        # final prediction Module
        self.read_embed_linear = nn.Linear(params.memory_value_state_dim * params.num_followup_attempts + params.q_embed_dim, params.final_fc_dim,
                                          bias=True).cuda()
        self.predict_linear = nn.Linear(params.final_fc_dim, 1, bias=True).cuda()


    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.kaiming_normal_(self.FPTs_weight_linear.weight)
        nn.init.kaiming_normal_(self.query_linear.weight)
        nn.init.kaiming_normal_(self.key_linear.weight)
        nn.init.kaiming_normal_(self.value_linear.weight)
        nn.init.constant_(self.predict_linear.bias, 0)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.FPTs_weight_linear.bias, 0)
        nn.init.constant_(self.query_linear.bias, 0)
        nn.init.constant_(self.key_linear.bias, 0)
        nn.init.constant_(self.value_linear.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.q_embed_layer.weight)
        # nn.init.kaiming_normal_(self.qa_embed.weight)
        nn.init.kaiming_normal_(self.frequency_embed_layer.weight)
        nn.init.kaiming_normal_(self.length_embed_layer.weight)
        # nn.init.kaiming_normal_(self.idx_embed_layer.weight)


    def calculate_weight_of_FPTs(self, FPTs_frequency): 
        # FPTs_frequency [batch_size, seqlen, num_learningpattern, num_followup_attempts, 2]
        FPTs_frequency = torch.sum(FPTs_frequency, dim=-1)
        # [batch_size, seqlen, num_learningpattern, num_followup_attempts]
        FPTs_embedding_frequency = torch.floor(torch.log2(FPTs_frequency) * 10).long()
        # [batch_size, seqlen, num_learningpattern, num_followup_attempts, frequency_embed_dim]
        FPTs_embedding_frequency = self.frequency_embed_layer(FPTs_embedding_frequency)

        # [batch_size * seqlen, num_learningpattern, num_followup_attempts, frequency_embed_dim]
        FPTs_embedding_frequency = FPTs_embedding_frequency.reshape(self.params.batch_size * self.params.seqlen, self.params.num_learningpattern, self.params.num_followup_attempts, self.frequency_embed_dim)

        # [batch_size * seqlen, num_learningpattern, num_followup_attempts]
        weight_of_FPTs = torch.sigmoid(self.FPTs_weight_linear(FPTs_embedding_frequency)).squeeze(-1)

        return weight_of_FPTs

    def get_mixed_FPTs_representation(self, FPTs):
        [FPTs_frequency, 
         FPTs_p, 
         FPTs_length,
         FPTs_idx] = FPTs

        # [batch_size, seqlen, num_learningpattern, num_followup_attempts]
        FPTs_length = FPTs_length[:,:,:,:,0]
        # [batch_size, seqlen, num_learningpattern]
        FPTs_length = FPTs_length[:,:,:,0]
        # [batch_size, seqlen, num_learningpattern, dep_embed_dim]
        FPTs_embedding_length = self.length_embed_layer(FPTs_length)
        # [batch_size * seqlen, num_learningpattern, dep_embed_dim]
        FPTs_embedding_length = FPTs_embedding_length.reshape(self.params.batch_size * self.params.seqlen, self.params.num_learningpattern, self.length_embed_dim)

        # [batch_size, seqlen, num_learningpattern, idx_embed_dim]
        # FPTs_embedding_idx = self.idx_embed_layer(FPTs_idx)
        # [batch_size * seqlen, num_learningpattern, idx_embed_dim]
        # FPTs_embedding_idx = FPTs_embedding_idx.reshape(self.params.batch_size * self.params.seqlen, self.params.num_learningpattern, self.idx_embed_dim)

        # [batch_size * seqlen, num_learningpattern, hatT_embed_dim]
        # hatT = torch.tanh(self.fc_length_idx(torch.cat([FPTs_embedding_length, FPTs_embedding_idx], 2)))
        hatT = torch.tanh(self.fc_length_idx(FPTs_embedding_length))
        # [batch_size * seqlen * num_learningpattern, 1, hatT_embed_dim]
        hatT = hatT.reshape(self.params.batch_size * self.params.seqlen * self.params.num_learningpattern, 1, self.hatT_embed_dim)

        # [batch_size * seqlen, num_learningpattern, num_followup_attempts, 2]
        FPTs_p = FPTs_p.reshape(self.params.batch_size * self.params.seqlen, self.params.num_learningpattern, self.params.num_followup_attempts, 2)
        # [batch_size * seqlen, num_learningpattern, num_followup_attempts, 1]
        FPTs_p = FPTs_p[:,:,:,1]
        # [batch_size * seqlen * num_learningpattern, num_followup_attempts, 1]
        FPTs_p = FPTs_p.reshape(self.params.batch_size * self.params.seqlen * self.params.num_learningpattern, self.params.num_followup_attempts, 1)

        # [batch_size * seqlen, num_learningpattern, num_followup_attempts, hatT_embed_dim]
        hatT = torch.matmul(FPTs_p, hatT)
        hatT = hatT.reshape(self.params.batch_size * self.params.seqlen, self.params.num_learningpattern, self.params.num_followup_attempts, self.hatT_embed_dim)
        
        # [batch_size * seqlen, num_learningpattern, num_followup_attempts]
        weight_of_FPTs = self.calculate_weight_of_FPTs(FPTs_frequency)
        weight_of_FPTs = weight_of_FPTs.unsqueeze(-1)

        # [batch_size * seqlen, num_followup_attempts, hatT_embed_dim]
        T = torch.mean(weight_of_FPTs * hatT, dim = 1)

        return T, weight_of_FPTs.reshape(self.params.batch_size, self.params.seqlen, self.params.num_learningpattern, self.params.num_followup_attempts)

    def model_learning_sequence(self, q_data, qa_data):
        if 'AKT' in self.params.model_type:
            Hs = self.kt_model.forward(q_data, qa_data, pid_data=None)
        elif 'RKT' in self.params.model_type:
            timespan = torch.arange(self.params.seqlen).cuda()
            rel = np.zeros((self.params.seqlen, self.params.seqlen))
            rel = torch.Tensor(rel).cuda()
            Hs = self.kt_model.forward(item_inputs=q_data,
                                    label_inputs=((qa_data - q_data) // self.n_question),
                                    item_ids=q_data,rel=rel,timestamp=timespan)
        else:
            Hs = self.kt_model.forward(q_data, qa_data)
        return Hs

    def fusion_followup_past(self, H, Trend):
        # H:     [self.params.batch_size, self.params.seqlen, self.kt_output_dim]
        # Trend: [self.params.batch_size, self.params.seqlen, num_followup_attempts, self.hatT_embed_dim]
        H_rep = H.unsqueeze(2).repeat([1,1,self.params.num_followup_attempts,1])
        H_rep = H_rep.reshape(self.params.batch_size * self.params.seqlen * self.params.num_followup_attempts, self.kt_output_dim)
        Trend = Trend.reshape(self.params.batch_size * self.params.seqlen * self.params.num_followup_attempts, self.hatT_embed_dim)
        H_rep = torch.cat([torch.ones(self.params.batch_size * self.params.seqlen * self.params.num_followup_attempts, 1).cuda(), 
                           H_rep], dim = -1).unsqueeze(-1)
        Trend = torch.cat([torch.ones(self.params.batch_size * self.params.seqlen * self.params.num_followup_attempts, 1).cuda(), 
                           Trend], dim = -1).unsqueeze(-2)
        fusion_H_Trend = torch.matmul(H_rep, Trend)
        fusion_H_Trend = fusion_H_Trend.reshape(self.params.batch_size * self.params.seqlen * self.params.num_followup_attempts,
                                                (self.kt_output_dim+1)*(self.hatT_embed_dim + 1))
        fusion_H_Trend = torch.tanh(self.linear_fusion(fusion_H_Trend))
        fusion_H_Trend = fusion_H_Trend.reshape(self.params.batch_size * self.params.seqlen, self.params.num_followup_attempts, self.knowledge_embedding_dim)
        fusion_H_Trend, (h_n, c_n) = self.lstm_fusion.forward(fusion_H_Trend)
        fusion_H_Trend = fusion_H_Trend.reshape(self.params.batch_size, self.params.seqlen, self.params.num_followup_attempts * self.memory_value_state_dim)

        return fusion_H_Trend

    # similar_user_future_sequence [batch_size*N, sequence_len, num_learningpattern, num_followup_attempts]
    def forward(self, q_data, qa_data, FPTs, target, t=-1, annotated_qa=None):

        Trend, weight_of_Trends = self.get_mixed_FPTs_representation(FPTs)
        # [self.params.batch_size, self.params.seqlen, num_followup_attempts, self.hatT_embed_dim]
        Trend = Trend.reshape(self.params.batch_size, self.params.seqlen, self.params.num_followup_attempts, self.hatT_embed_dim)
        Hs = self.model_learning_sequence(q_data, qa_data)
        Ek = self.fusion_followup_past(Hs, Trend)

        # q_embed_data [batch_size, seqlen, q_embed_dim]
        q_embed_data = self.q_embed_layer(q_data)
        # predict_input [batch_size, seqLen, memory_value_state_dim + hatT_embed_dim + q_embed_size]
        predict_input = torch.cat([Ek, q_embed_data], 2)
        # read_content_embed [batch_size * seqLen, final_fc_dim]
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(self.params.batch_size * self.params.seqlen, -1)))
        # pred [batch_size * seqLen, 1]
        pred = self.predict_linear(read_content_embed)

        # target target_1d: [batch_size * seq_len, 1] [batch_size * seq_len, 1]
        target_1d = target  # [batch_size * seq_len, 1]
        
        if t == -1:
            mask = target_1d.ge(0)
        else:
            mask_first = torch.zeros(size=[self.params.seqlen,1]).cuda()
            mask_first[t] = 1
            mask_first = mask_first.repeat(self.params.batch_size,1)
            mask = (target_1d.ge(0) * mask_first).bool()  # [batch_size * seq_len, 1]
        # mask = target_1d.ge(0)  # [batch_size * seq_len, 1]
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

            return loss, torch.sigmoid(filtered_pred), filtered_target, [weight_of_Trends, torch.sigmoid(pred).reshape(self.params.batch_size, self.params.seqlen)], annotated_output
        else:
            return loss, torch.sigmoid(filtered_pred), filtered_target, [weight_of_Trends, torch.sigmoid(pred).reshape(self.params.batch_size, self.params.seqlen)]