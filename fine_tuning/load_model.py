import torch, torchinfo
from pathlib import Path
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-base-cased")
torchinfo.summary(model)