import torch
import math
import torch.nn as nn


class IngredientParser(nn.Module):
    def __init__(self, word_embedding, config):
        super(IngredientParser, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.head_size = config["head_size"]
        self.dropout = config["dropout"]
        self.layers = config["layers"]
        self.output_size = config["output_size"]
        self.freeze_embeddings = config["freeze_embeddings"]
        
        self.self_attn = \
            nn.ModuleList([SelfattLayer(self.hidden_size, self.head_size, self.dropout) for _ in range(self.layers)])
        self.classifer = nn.Linear(self.hidden_size, self.output_size)
        self.tag_embedding = nn.Embedding(39, 300)
        self.word_embedding = word_embedding
        if self.freeze_embeddings:
            self.word_embedding.weight.requires_grad = False
        
    def forward(self, input_ids, tag_ids):
        word_repr = self.word_embedding(input_ids) 
        tag_repr = self.word_embedding(tag_ids)
        word_tag_repr = (word_repr + tag_repr) / 2
        
        for layer_id in range(self.layers):
            word_tag_repr = self.self_attn[layer_id](word_tag_repr) 
        
        logits = self.classifer(word_tag_repr) 
        return logits


class SelfattLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1):
        super(SelfattLayer, self).__init__()
        self.self_attn = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.ffn = FeedForwardLayer(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None):
        self_output = self.self_attn(input_tensor, input_tensor, attention_mask)
        attention_output = self.ffn(self_output, input_tensor)
        return attention_output


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob=0.2, ctx_dim=None):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.hidden_size = hidden_size 
        self.num_attention_heads = num_attention_heads 
        self.attention_head_size = int(hidden_size / num_attention_heads) 
        self.all_head_size = self.num_attention_heads * self.attention_head_size 

        if ctx_dim is None:
            ctx_dim = hidden_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size) 
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        """
        hidden_states: [batch_size, N, h1], N=1089 or max_length
        context: [batch_size, M, h2]
        """
        mixed_query_layer = self.query(hidden_states) 
        mixed_key_layer = self.key(context) 
        mixed_value_layer = self.value(context) 

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) 
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

  
class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(FeedForwardLayer, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        hidden_states: output of attention module
        input_tensor: input of attention module
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
