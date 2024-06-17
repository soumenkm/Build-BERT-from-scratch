import torch, torchinfo
from pathlib import Path

def assign_parameters(src: torch.tensor, tgt: torch.tensor) -> torch.tensor:
    if src.shape == tgt.shape:
        return torch.nn.Parameter(data=src)
    else:
        raise ValueError(f"source {src.shape} is not same as target {tgt.shape}")

def load_pretrain_model(source_model: "BertModel", target_model: "Bert") -> None:

    src_pnum = sum([i.numel() for i in source_model.parameters()])
    tgt_pnum = sum([i.numel() for i in target_model.parameters()])
    assert  src_pnum == tgt_pnum, f"Number of parameters (source={src_pnum}, target={tgt_pnum}) do not match"

    src_enc_list = []
    tgt_enc_list = []
    
    for i in range(target_model.L):
        src_enc_list.extend([
            f"encoder.layer.{i}.attention.output.dense.weight",
            f"encoder.layer.{i}.attention.output.dense.bias",
            f"encoder.layer.{i}.attention.output.LayerNorm.weight",
            f"encoder.layer.{i}.attention.output.LayerNorm.bias",
            f"encoder.layer.{i}.intermediate.dense.weight",
            f"encoder.layer.{i}.intermediate.dense.bias",
            f"encoder.layer.{i}.output.dense.weight",
            f"encoder.layer.{i}.output.dense.bias",
            f"encoder.layer.{i}.output.LayerNorm.weight",
            f"encoder.layer.{i}.output.LayerNorm.bias"
        ])
        tgt_enc_list.extend([
            f"encoder.transformer_blocks.{i}.attention.multihead_self_attention.out_proj.weight",
            f"encoder.transformer_blocks.{i}.attention.multihead_self_attention.out_proj.bias",
            f"encoder.transformer_blocks.{i}.attention.layernorm.weight",
            f"encoder.transformer_blocks.{i}.attention.layernorm.bias",
            f"encoder.transformer_blocks.{i}.feed_forward.linear1.weight",
            f"encoder.transformer_blocks.{i}.feed_forward.linear1.bias",
            f"encoder.transformer_blocks.{i}.feed_forward.linear2.weight",
            f"encoder.transformer_blocks.{i}.feed_forward.linear2.bias",
            f"encoder.transformer_blocks.{i}.feed_forward.layernorm.weight",
            f"encoder.transformer_blocks.{i}.feed_forward.layernorm.bias",
        ])

    source_params = [
        "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight",
        "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias",
    ] + src_enc_list + [
        "pooler.dense.weight",
        "pooler.dense.bias"
    ]

    target_params = [
        "embedding.word_embedding.weight",
        "embedding.pos_embedding.weight",
        "embedding.seg_embedding.weight",
        "embedding.layernorm.weight",
        "embedding.layernorm.bias"
    ] + tgt_enc_list + [
        "pooler.linear.weight",
        "pooler.linear.bias"
    ]

    src_tgt_param_map = dict(zip(source_params, target_params))
    qkv_dict = {}
    for i in range(target_model.L):    
        WQ = source_model.encoder.layer[i].attention.self.query.weight
        BQ = source_model.encoder.layer[i].attention.self.query.bias
        WK = source_model.encoder.layer[i].attention.self.key.weight
        BK = source_model.encoder.layer[i].attention.self.key.bias
        WV = source_model.encoder.layer[i].attention.self.value.weight
        BV = source_model.encoder.layer[i].attention.self.value.bias

        W = torch.cat([WQ, WK, WV], dim=0)
        B = torch.cat([BQ, BK, BV], dim=0)

        qkv_dict[f"encoder.transformer_blocks.{i}.attention.multihead_self_attention.in_proj_weight"] = torch.nn.Parameter(W)
        qkv_dict[f"encoder.transformer_blocks.{i}.attention.multihead_self_attention.in_proj_bias"] = torch.nn.Parameter(B)


    new_state_dict_target = qkv_dict
    state_dict_target = target_model.state_dict()

    for name, val in source_model.named_parameters():
        if "key" not in name and "query" not in name and "value" not in name: 
            new_state_dict_target[src_tgt_param_map[name]] = assign_parameters(src=val, tgt=state_dict_target[src_tgt_param_map[name]])

    keys_diff = list((set(state_dict_target.keys()) - set(new_state_dict_target.keys())))
    target_model.load_state_dict(new_state_dict_target, strict=False)
    print(f"State dict of target model is updated. The following names are not loaded: {keys_diff}")
    
    src_psum = sum([i.sum().detach().numpy() for i in source_model.parameters()])
    tgt_psum = sum([i.sum().detach().numpy() for i in target_model.parameters()])
    assert abs(src_psum - tgt_psum) < 1, f"Sum of parameters (source={src_psum}, target={tgt_psum}) do not match"
    
def compare_model_output(source_model: "BertModel", target_model: "Bert") -> None:
    
    source_model.eval()
    target_model.eval()
    
    with torch.no_grad():
        inputs = torch.randint(low=0, high=target_model.V, size=(2, 10))
        src_out = source_model(inputs)[-1] # pooled output
        tgt_out = target_model(inputs)[-1] # pooled output
        print("Output at [CLS] from source model: \n", src_out)
        print("Output at [CLS] from target model: \n", tgt_out)

def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    config = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": 768,
        "num_heads": 12,
        "dropout_prob": 0.2,
        "max_context_length": 512,
        "num_layers": 12
    }

    source_model = BertModel.from_pretrained("bert-base-cased")
    target_model = Bert(config)
    compare_model_output(source_model=source_model, target_model=target_model)
    load_pretrain_model(source_model=source_model, target_model=target_model)
    compare_model_output(source_model=source_model, target_model=target_model)
   
if __name__ == "__main__":
    from transformers import BertModel, BertTokenizer
    from bert_model import Bert
    
    main()