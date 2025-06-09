import torch
import torch.nn as nn
import numpy as np



def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)



class LSTMA(nn.Module):

    def __init__(self, params, student_num=None):
        super(LSTMA, self).__init__()
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

        # LSTM
        self.lstm_kt = nn.LSTM(self.qa_embed_dim, self.memory_value_state_dim, 1, batch_first=True).cuda()

        # Attention
        self.attention = nn.MultiheadAttention(embed_dim=params.qa_embed_dim, num_heads=1, dropout=0.2).cuda()

        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.predict_linear.bias, 0)
        nn.init.constant_(self.read_embed_linear.bias, 0)

    #[batch_size,seqlen.n_question]  <- [batch_size,sqelen]
    def onehot(self, questions, answers):
        result = np.zeros(shape=[self.max_seq, 2 * self.n_skill])
        for i in range(self.max_seq):
            if answers[i] > 0:
                result[i][questions[i]] = 1
            elif answers[i] == 0:
                result[i][questions[i] + self.n_skill] = 1
        return result

    def init_embeddings(self):

        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

    # similar_user_future_sequence [batch_size*N, sequence_len, similar_user_num, future_sequence_len]
    def forward(self, q_data, qa_data):

        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]

        #print(qa_data[0])

        #print(q_data[0])

        # q_embed_data [batch_size, seqlen, q_embed_dim]
        q_embed_data = self.q_embed(q_data)

        # qa_embed_data [batch_size, seqlen, qa_embed_dim]
        qa_embed_data = self.qa_embed(qa_data)

        # all_read_value_content[batch_size,seqlen,qa_embed_dim]
        all_read_value_content, (h_n, c_n)  = self.lstm_kt.forward(qa_embed_data)

        # q_embed_data = q_embed_data.permute(1, 0, 2)
        # qa_embed_data = qa_embed_data.permute(1, 0, 2)
        # all_read_value_content = all_read_value_content.permute(1, 0, 2)
        # 100 32 200
        # print(all_read_value_content.shape)

        device = qa_data.device

        att_mask = future_mask(qa_embed_data.size(0)).to(device)

        # print(qa_embed_data.shape)
        #
        # print(att_mask.shape)
        #print(target.shape)
        # Q  qa_embed_data   [32, 100, 200]  [L N E]
        # K
        # V  all_read_value_content  [32, 100, 200]  [S N E]

        att_out, att_weight = self.attention(qa_embed_data, qa_embed_data, all_read_value_content,attn_mask=att_mask)
        #100 32 200
        #print(att_out.shape)

        att_out = att_out[:, :-1]
        zeros_padding = torch.zeros(att_out[:, 0].shape).cuda()
        att_out = torch.cat([zeros_padding.unsqueeze(1), att_out], 1)


        return att_out

