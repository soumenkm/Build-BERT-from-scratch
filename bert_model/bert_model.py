import torch, torchinfo
from typing import Tuple, List
from transformers import BertTokenizer

class BertEmbedding(torch.nn.Module):
    
    def __init__(self, vocab_size: int, embedding_size: int, max_context_legth: int, dropout_prob: float):
        
        super(BertEmbedding, self).__init__()
        self.V = vocab_size
        self.d = embedding_size
        self.Tmax = max_context_legth
        self.p = dropout_prob
        
        self.word_embedding = torch.nn.Embedding(num_embeddings=self.V, embedding_dim=self.d)
        self.pos_embedding = torch.nn.Embedding(num_embeddings=self.Tmax, embedding_dim=self.d)
        self.seg_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=self.d)
        
        self.layernorm = torch.nn.LayerNorm(normalized_shape=self.d)
        self.dropout = torch.nn.Dropout(p=self.p)
        
        pos_inputs = torch.arange(start=0, end=self.Tmax, step=1) # (Tmax,)
        pos_inputs = pos_inputs.unsqueeze(dim=0) # (1, Tmax)
        self.register_buffer(name="pos_inputs", tensor=pos_inputs)
        
    def forward(self, inputs: torch.tensor, segment_ids: torch.tensor) -> torch.tensor:
        
        assert inputs.shape[-1] <= self.Tmax and inputs.dim() == 2, f"inputs.shape = {inputs.shape} can be at most (b, {self.Tmax})"
        assert segment_ids.shape[-1] <= self.Tmax and segment_ids.dim() == 2, f"segment_ids.shape = {segment_ids.shape} can be at most (b, {self.Tmax})"
        assert segment_ids.unique().sum() in [0, 1], f"segment ids = {segment_ids.unique()} should be of only 0 or 1"

        x1 = self.word_embedding(inputs) # (b, T, d)
        x2 = self.pos_embedding(self.pos_inputs[:, :inputs.shape[-1]].expand(*inputs.shape)) # (1, T) -> (b, T) -> (b, T, d) 
        x3 = self.seg_embedding(segment_ids) # (b, T, d)
        x = x1 + x2 + x3 # (b, T, d)
        x = self.layernorm(x) # (b, T, d)
        x = self.dropout(x) # (b, T, d)
        
        return x

class BertAttention(torch.nn.Module):
    
    def __init__(self, embedding_dim: int, num_heads: int, max_context_legth:int, dropout_prob: float):
        super(BertAttention, self).__init__()
        self.d = embedding_dim
        self.h = num_heads
        self.p = dropout_prob
        self.Tmax = max_context_legth
        
        self.multihead_self_attention = torch.nn.MultiheadAttention(embed_dim=self.d, num_heads=self.h, dropout=0.0, bias=True, batch_first=True)
        # self.attention_out_projection = torch.nn.Linear(in_features=self.d, out_features=self.d, bias=True) # MHA automatically adds an out_proj layer
        self.layernorm = torch.nn.LayerNorm(normalized_shape=self.d)
        self.dropout = torch.nn.Dropout(p=self.p)
    
    def forward(self, inputs: torch.tensor, padding_mask: torch.tensor) -> torch.tensor:
        """padding mask has 1 means padded and 0 means real"""
        assert inputs.shape[-2] <= self.Tmax and inputs.shape[-1] == self.d and inputs.dim() == 3, f"inputs.shape = {inputs.shape} can be at most (b, {self.Tmax}, {self.d})"
        assert padding_mask.shape[-1] <= self.Tmax and padding_mask.dim() == 2, f"padding_mask.shape = {padding_mask.shape} can be at most (b, {self.Tmax})"
        assert padding_mask.unique().sum() in [0, 1], f"padding_mask = {padding_mask.unique()} should be of only 0 or 1"
        
        padding_mask = (padding_mask == 1) # pytorch expects True means mask and False means real   
        x = self.multihead_self_attention(query=inputs, key=inputs, value=inputs, 
                                          key_padding_mask=padding_mask, need_weights=False, 
                                          is_causal=False)[0] # (b, T, d)
        # x = self.attention_out_projection(x) # (b, T, d)
        x = x + inputs # (b, T, d)
        x = self.layernorm(x) # (b, T, d)
        x = self.dropout(x) # (b, T, d)
        
        return x
        
class BertFeedForward(torch.nn.Module):
    
    def __init__(self, embedding_dim: int, max_context_legth: int, dropout_prob: float):
        super(BertFeedForward, self).__init__()
        self.d = embedding_dim
        self.p = dropout_prob
        self.Tmax = max_context_legth
        
        self.linear1 = torch.nn.Linear(in_features=self.d, out_features=self.d * 4, bias=True)
        self.gelu = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(in_features=self.d * 4, out_features=self.d, bias=True)
        self.layernorm = torch.nn.LayerNorm(normalized_shape=self.d)
        self.dropout = torch.nn.Dropout(p=self.p)
    
    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert inputs.shape[-2] <= self.Tmax and inputs.shape[-1] == self.d and inputs.dim() == 3, f"inputs.shape = {inputs.shape} can be at most (b, {self.Tmax}, {self.d})"

        x = self.linear1(inputs) # (b, T, d)
        x = self.gelu(x) # (b, T, d)
        x = self.linear2(x) # (b, T, d)
        x = x + inputs # (b, T, d)
        x = self.layernorm(x) # (b, T, d)
        x = self.dropout(x) # (b, T, d)
        
        return x

class BertTransformer(torch.nn.Module):
    
    def __init__(self, embedding_dim: int, num_heads: int, max_context_legth:int, dropout_prob: float):
        super(BertTransformer, self).__init__()
        self.d = embedding_dim
        self.h = num_heads
        self.p = dropout_prob
        self.Tmax = max_context_legth
        
        self.attention = BertAttention(embedding_dim=self.d, num_heads=self.h, max_context_legth=self.Tmax, dropout_prob=self.p)
        self.feed_forward = BertFeedForward(embedding_dim=self.d, max_context_legth=self.Tmax, dropout_prob=self.p)
        
    def forward(self, inputs: torch.tensor, padding_mask: torch.tensor) -> torch.tensor:
        """inputs.shape = (b, T, d), padding_mask.shape = (b, T)"""
             
        x = self.attention(inputs, padding_mask) # (b, T, d)
        x = self.feed_forward(x) # (b, T, d)
        
        return x

class BertEncoder(torch.nn.Module):
    
    def __init__(self, embedding_dim: int, num_heads: int, max_context_legth:int, num_layers: int, dropout_prob: float):
        super(BertEncoder, self).__init__()
        self.d = embedding_dim
        self.h = num_heads
        self.p = dropout_prob
        self.Tmax = max_context_legth
        self.L = num_layers
        
        self.transformer_blocks = torch.nn.ModuleList([BertTransformer(embedding_dim=self.d, 
                                                                       num_heads=self.h, 
                                                                       max_context_legth=self.Tmax, 
                                                                       dropout_prob=self.p) for i in range(self.L)])
    
    def forward(self, inputs: torch.tensor, padding_mask: torch.tensor) -> torch.tensor:
        """inputs.shape = (b, T, d), padding_mask.shape = (b, T)"""
        
        x = inputs # (b, T, d)
        for transformer in self.transformer_blocks: 
            x = transformer(x, padding_mask) # (b, T, d)
        
        return x

class BertPooler(torch.nn.Module):
    
    def __init__(self, embedding_dim: int, max_context_legth: int):
        super(BertPooler, self).__init__()
        self.d = embedding_dim
        self.Tmax = max_context_legth
        
        self.linear = torch.nn.Linear(in_features=self.d, out_features=self.d, bias=True)
        self.tanh = torch.nn.Tanh()
    
    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert inputs.shape[-2] <= self.Tmax and inputs.shape[-1] == self.d and inputs.dim() == 3, f"inputs.shape = {inputs.shape} can be at most (b, {self.Tmax}, {self.d})"

        x = self.linear(inputs[:, 0, :]) # (b, T, d) -> (b, d) only for [CLS] token
        x = self.tanh(x) # (b, d)
        
        return x

class Bert(torch.nn.Module):
    
    def __init__(self, config: dict):
        super(Bert, self).__init__()
        self.V = config["vocab_size"]
        self.d = config["embedding_dim"]
        self.h = config["num_heads"]
        self.p = config["dropout_prob"]
        self.Tmax = config["max_context_length"]
        self.L = config["num_layers"]
        
        self.embedding = BertEmbedding(vocab_size=self.V, embedding_size=self.d, max_context_legth=self.Tmax, dropout_prob=self.p)
        self.encoder = BertEncoder(embedding_dim=self.d, num_heads=self.h, max_context_legth=self.Tmax, num_layers=self.L, dropout_prob=self.p)
        self.pooler = BertPooler(embedding_dim=self.d, max_context_legth=self.Tmax)
        
    def forward(self, inputs: torch.tensor, segment_ids: torch.tensor=None, padding_mask: torch.tensor=None) -> torch.tensor:
        """inputs.shape = (b, T), segment_ids.shape = (b, T), padding_mask.shape = (b, T)
        padding_mask = 1 means it is padded token and 0 means it is real token
        """
        
        if segment_ids is None and padding_mask is None:
            segment_ids = torch.zeros(size=inputs.shape, dtype=torch.int64) # all sent_A
            padding_mask = torch.zeros(size=inputs.shape, dtype=torch.int64) # all real

        elif segment_ids is None and padding_mask is not None:
            raise ValueError("When padding_mask is provided, segment_ids cannot be None")
            
        elif segment_ids is not None and padding_mask is None:
            padding_mask = torch.zeros(size=inputs.shape) # all real
            
        x = self.embedding(inputs, segment_ids) # (b, T, d)
        x = self.encoder(x, padding_mask) # (b, T, d)
        z = self.pooler(x) # (b, d) only for [CLS] token
        
        return x, z # hidden and pooled output

def main():
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    batch_size = 32
    
    config = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": 768,
        "num_heads": 12,
        "dropout_prob": 0.2,
        "max_context_length": 512,
        "num_layers": 12
    }
    
    inputs = torch.randint(low=0, high=config["vocab_size"], size=(batch_size, config["max_context_length"]))
    padding_mask = torch.randint(low=0, high=2, size=(batch_size, config["max_context_length"]))
    segment_ids = torch.randint(low=0, high=2, size=(batch_size, config["max_context_length"]))
    
    model = Bert(config=config)
    print(model)
    
if __name__ == "__main__":
    
    main()