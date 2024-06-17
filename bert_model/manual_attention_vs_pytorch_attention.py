import torch

class CustomMultiheadAttention(torch.nn.Module):
    
    def __init__(self, embedding_dim, num_heads, bias=False):
        super(CustomMultiheadAttention, self).__init__()
        self.d = embedding_dim
        self.h = num_heads
        self.dh = self.d // self.h
        
        self.query_layer = torch.nn.Linear(in_features=self.d, out_features=self.d, bias=bias)
        self.key_layer = torch.nn.Linear(in_features=self.d, out_features=self.d, bias=bias)
        self.value_layer = torch.nn.Linear(in_features=self.d, out_features=self.d, bias=bias)
        self.out_proj_layer = torch.nn.Linear(in_features=self.d, out_features=self.d, bias=bias)
    
    def forward(self, query, key, value, key_padding_mask):
        """query.shape = key.shape = value.shape = (b, T, d)
        key_padding_mask.shape = (b, T)
        """
        
        b, T = query.shape[0], query.shape[1]
        
        Q = self.query_layer(query) # (b, T, d)
        K = self.key_layer(key) # (b, T, d)
        V = self.value_layer(value) # (b, T, d)
        
        Q = Q.reshape(shape=(b, T, self.h, self.dh)) # (b, T, h, dh)
        K = K.reshape(shape=(b, T, self.h, self.dh)) # (b, T, h, dh)
        V = V.reshape(shape=(b, T, self.h, self.dh)) # (b, T, h, dh)
        
        Q = Q.transpose(1, 2) # (b, h, T, dh)
        K = K.transpose(1, 2) # (b, h, T, dh)
        V = V.transpose(1, 2) # (b, h, T, dh)
        
        W = torch.matmul(input=Q, other=K.transpose(2, 3)) # (b, h, T, T)
        SW = W / torch.sqrt(input=torch.tensor(self.dh).to(torch.float32)) # (b, h, T, T)
        
        actual_mask = padding_mask.unsqueeze(1).unsqueeze(2).expand(*W.shape).to(torch.int64) # (b, T) -> (b, 1, 1, T) -> (b, h, T, T) 
        actual_mask = torch.where(condition=(actual_mask==1), input=torch.tensor(-torch.inf), other=torch.tensor(0.0)) # (b, T, T)
        SW = SW + actual_mask # (b, h, T, T) (auto broadcasted)
        
        A = SW.softmax(dim=-1) # (b, h, T, T)
        
        Z = torch.matmul(input=A, other=V) # (b, h, T, dh)
        Z = Z.transpose(1, 2).contiguous() # (b, T, h, dh)
        Z = Z.reshape(shape=(b, T, self.d)).contiguous() # (b, T, d)
        Z = self.out_proj_layer(Z) # (b, T, d)
        
        return Z, A

attention = torch.nn.MultiheadAttention(embed_dim=4, num_heads=2, bias=False, add_bias_kv=False, batch_first=True)    
custom_attention = CustomMultiheadAttention(embedding_dim=4, num_heads=2, bias=False)

WQ = custom_attention.state_dict()["query_layer.weight"]
WK = custom_attention.state_dict()["key_layer.weight"]
WV = custom_attention.state_dict()["value_layer.weight"]
WO = custom_attention.state_dict()["out_proj_layer.weight"]

new_dict = {}
for name in attention.state_dict().keys():
    if name == "in_proj_weight":
        val1 = torch.cat(tensors=[WQ, WK, WV], dim=0)
        new_dict[name] = val1
    elif name == "out_proj.weight":
        new_dict[name] = WO

attention.load_state_dict(new_dict, strict=False)

inputs = torch.rand(size=(2, 3, 4))
padding_mask = torch.tensor([[1, 1, 0], [1, 1, 0]]) == 0

out, A = attention(inputs, inputs, inputs, key_padding_mask=padding_mask, need_weights=True, average_attn_weights=False)
custom_out, custom_A = custom_attention(inputs, inputs, inputs, key_padding_mask=padding_mask)

print(out)
print(custom_out)