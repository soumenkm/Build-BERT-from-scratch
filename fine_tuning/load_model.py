import torch
from pathlib import Path
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-base-cased")
print(model)