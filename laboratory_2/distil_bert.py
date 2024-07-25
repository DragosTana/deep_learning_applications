import torch
import torch.nn as nn
from typing import Optional
from activations import get_activation

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Class implementing the positional encoding layer.
    This is a simple implementation of the sinusoidal positional encoding from the "Attention is All You Need" paper.
    """
    def __init__(self, max_position_embeddings, dim):
        super().__init__()
        
        self.encoding = torch.zeros(max_position_embeddings, dim)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0, max_position_embeddings).unsqueeze(1).float()
        _2i = torch.arange(0, dim, 2).float()
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / dim)))
        self.encoding = self.encoding.unsqueeze(0)  # [1, max_position_embeddings, dim]

    def forward(self, seq_length):
        return self.encoding[:, :seq_length, :]


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        
        if config.sinusoidal_pos_embds:
            self.position_embeddings = SinusoidalPositionalEmbedding(config.max_position_embeddings, config.dim)
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        ### Args:
            - input_ids (torch.Tensor): torch.tensor(bs, max_seq_length) The token ids to embed.
        ### Outputs:
            - torch.tensor(bs, max_seq_length, dim) The embedded tokens plus positional embeddings.
        """

        input_embeds = self.word_embeddings(input_ids)  # [bs, max_seq_length, dim]
        seq_length = input_embeds.size(1)

        if isinstance(self.position_embeddings, SinusoidalPositionalEmbedding):
            position_embeddings = self.position_embeddings(seq_length)  # [1, max_seq_length, dim]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            position_embeddings = self.position_embeddings(position_ids)  # [1, max_seq_length, dim]

        embeddings = input_embeds + position_embeddings  # [bs, max_seq_length, dim]
        embeddings = self.LayerNorm(embeddings)  # [bs, max_seq_length, dim]
        embeddings = self.dropout(embeddings)  # [bs, max_seq_length, dim]
        return embeddings

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward layer. Straightforward from the "Attention is All You Need" paper
    with the exception of the dropout layer.
    """
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        self.activation = get_activation(config.activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward layer.
        
        ### Args:
            x: a float tensor with shape [batch_size, sequence_length, dim]
        ### Outputs:
            a float tensor with shape [batch_size, sequence_length, dim]
        """
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Vanilla multi-head attention layer. Straightforward from the "Attention is All You Need" paper.
    """
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.dim / config.num_attention_heads)
        self.dim = self.num_attention_heads * self.attention_head_size

        if config.dim % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.dim, config.num_attention_heads)
            )
        
        self.q_lin = nn.Linear(config.dim, self.dim)
        self.k_lin = nn.Linear(config.dim, self.dim)
        self.v_lin = nn.Linear(config.dim, self.dim)

        self.dropout = nn.Dropout(config.attention_dropout)
        
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, mask=None):
        query = self.q_lin(hidden_states) #[batch_size, sequence_length, dim]
        key = self.k_lin(hidden_states)
        value = self.v_lin(hidden_states)
        
        query = self.transpose_for_scores(query) #[batch_size, num_attention_heads, sequence_length, attention_head_size]
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.attention_head_size ** 0.5) #[batch_size, num_attention_heads, sequence_length, sequence_length]
        attention_scores = attention_scores + mask if mask is not None else attention_scores
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.out_lin(context_layer)
        return context_layer
    
class TransformerBlock(nn.Module):
    """
    Transformer block similar to the EncoderLayer but compatible with the DistilBERT 
    weights. 
    """
    
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.ffn = PositionWiseFeedForward(config)
        self.output_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None)-> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask) #! HERE
        sa_layer_norm = self.sa_layer_norm(attention_output + hidden_states)
        ffn_output = self.ffn(sa_layer_norm)
        output = self.output_layer_norm(ffn_output + sa_layer_norm)
        return output
    
class Transformer(nn.Module):
    """
    Transformer model implementation. Similar to the EncoderLayer but compatible with the DistilBERT weights.
    """
    def __init__(self, config):
       super().__init__()
       self.n_layers = config.n_layers
       self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(self.n_layers)])
       
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None)-> torch.Tensor:
  
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
            
        return hidden_states
    
class DistilBERTModel(nn.Module):
    """
    DistilBERT model. 
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.transformer = Transformer(config)
        self.config = config
        
    def generate_attention_mask(self, attention_mask):
        dtype = next(self.parameters()).dtype
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
        return attention_mask
    
    def forward(self, input_ids, attention_mask=None):
    
        if attention_mask is None:
            extended_attention_mask = None
        else:
            if len(attention_mask.size()) == 2:
                extended_attention_mask = self.generate_attention_mask(attention_mask)
            else:
                extended_attention_mask = attention_mask
        
        embeddings = self.embeddings(input_ids)
        hidden_states = self.transformer(embeddings, extended_attention_mask)
        return hidden_states

class DistiBertClassification(nn.Module):
    """
    DistilBERT model for text classification.
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.activation = get_activation(config.activation)
        self.distilbert = DistilBERTModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.dropout)

    def generate_attention_mask(self, attention_mask):
        dtype = next(self.parameters()).dtype
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
        return attention_mask
    
    def forward(self, input_ids, attention_mask=None):
            
        if attention_mask is None:
            extended_attention_mask = None
        else:
            extended_attention_mask = self.generate_attention_mask(attention_mask)
            
        hidden_states = self.distilbert(input_ids, extended_attention_mask) #[batch_size, seq_len, dim]
        hidden_states = hidden_states[:, 0] # [batch_size, dim]
        hidden_states = self.pre_classifier(hidden_states)  # [batch_size, dim]
        hidden_states = self.activation(hidden_states)  # [batch_size, dim]
        hidden_states = self.dropout(hidden_states) # [batch_size, dim]
        logits = self.classifier(hidden_states) # [batch_size, num_labels]
        return logits
    
