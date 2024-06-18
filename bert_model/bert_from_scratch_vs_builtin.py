from transformers import BertModel, BertTokenizer
from bert_model import Bert
import torch
from load_pretrain_model import load_pretrain_model

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

config = {
    "vocab_size": tokenizer.vocab_size,
    "embedding_dim": 768,
    "num_heads": 12,
    "dropout_prob": 0.2,
    "max_context_length": 512,
    "num_layers": 12
}
custom_model = Bert(config=config)
load_pretrain_model(source_model=model, target_model=custom_model)

input_texts = ["Hello there, how are you doing today?", "I am doing good, and you?"]
inputs_dict = tokenizer(input_texts[0], input_texts[1], return_tensors="pt")

inputs = inputs_dict["input_ids"] # (b, T)
segment_ids = inputs_dict["token_type_ids"] # (b, T)
padding_mask = inputs_dict["attention_mask"] # (b, T)

model.eval()
custom_model.eval()

with torch.no_grad():
    out = model(input_ids=inputs, token_type_ids=segment_ids, encoder_attention_mask=padding_mask, output_hidden_states=True)
    custom_out = custom_model(inputs=inputs, segment_ids=segment_ids, padding_mask=padding_mask)
    
    print(out.pooler_output.sum())
    print(custom_out["pooled_output"].sum())