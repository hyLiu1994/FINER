import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F

class selfAttention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        # one head dimension
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        # new_size : [batch_size, similar_user_num,num_attention_heads,attention_head_size]
        # hidden_size=num_attention_heads*attention_head_size
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        # x.permute : [batch_size, num_attention_heads, similar_user_num,  attention_head_size]
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        # x : [batch_size,similar_user_num,future_sequence_len*read_dim]
        # key : [batch_size,similar_user_num,hidden_size]
        # query : [batch_size,similar_user_num,hidden_size]
        # value : [batch_size,similar_user_num,hidden_size]
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)
        # key_heads : [batch_size, num_attention_heads, similar_user_num, attention_head_size]
        # query_heads : [batch_size, num_attention_heads, similar_user_num, attention_head_size]
        # value_heads : [batch_size, num_attention_heads, similar_user_num, attention_head_size]
        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)
        #query_heads: [batch_size, num_attention_heads, similar_user_num, attention_head_size]
        # key_heads.permute(0, 1, 3, 2) : [batch_size, num_attention_heads,  attention_head_size,similar_user_num]
        # attention_scores : [batch_size, num_attention_heads, similar_user_num,similar_user_num]
        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_probs : [batch_size, num_attention_heads, similar_user_num,similar_user_num]
        attention_probs = F.softmax(attention_scores, dim = -1)
        # context: [batch_size, num_attention_heads, similar_user_num, attention_head_size]
        context = torch.matmul(attention_probs, value_heads)
        #context = context.permute(0, 2, 1, 3).contiguous()
        #new_size = context.size()[ : -2] + (self.all_head_size , )
        #context = context.view(*new_size)
        return context