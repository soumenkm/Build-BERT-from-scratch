import torch, torchinfo
from transformers import BertForMaskedLM, BertTokenizer
# from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

print(model.bert)
torchinfo.summary(model.bert.encoder.layer[0])

# tk_in = tokenizer("My name [MASK] Soumen Mondal", "I am studying AI [MASK] DS", return_tensors="pt")
# print(tk_in)
# output = model(input_ids=tk_in["input_ids"], 
#                attention_mask=tk_in["attention_mask"], 
#                token_type_ids=tk_in["token_type_ids"],
#                output_hidden_states=True)

# print(output)
# print(output.logits.shape)
# print([i.shape for i in output.hidden_states])
