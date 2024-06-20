import torch, tqdm, sys, torchinfo, os, json
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from typing import List, Tuple, Union
from transformers import BertModel, BertTokenizer
from bert_model.bert_model import BertPretrainLM
from bert_model.load_pretrain_model import load_pretrain_model
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

