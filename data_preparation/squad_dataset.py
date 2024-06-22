import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class SQUADataset(Dataset):
    
    def __init__(self, tokenizer: BertTokenizer, is_train: bool, frac: float):
        
        super(SQUADataset, self).__init__()
        self.tokenizer = tokenizer
        self.hf_ds = load_dataset("rajpurkar/squad")
        self.is_train = is_train
        self.frac = frac
        
        if self.is_train:
            self.dataset = self.hf_ds["train"].select(range(int(len(self.hf_ds["train"]) * self.frac)))
        else:
            self.dataset = self.hf_ds["validation"].select(range(int(len(self.hf_ds["validation"]) * self.frac)))

        self.dataset = self._map_dataset()
        self.dataset = self.dataset.filter(self._filter_example)
    
    def _filter_example(self, example: dict) -> dict:
        
        return example["end_index"] < self.tokenizer.model_max_length
            
    def _map_example(self, example: dict) -> dict:
        
        output = self.tokenizer(example["question"], 
                                example["context"], 
                                example["context"][: int(example["answers"]["answer_start"][0])],
                                example["answers"]["text"][0],
                                truncation=True, max_length=self.tokenizer.model_max_length)
        
        base_index = output["input_ids"].index(self.tokenizer.sep_token_id) # SEP token position marks start of context
        start = output["labels"].index(self.tokenizer.sep_token_id)
        end = len(output["labels"]) - 1
        start_index = base_index + start # Inclusive
        end_index = base_index + end - 2 # Inclusive
        assert start_index <= end_index
        
        output = {**output, **{"[PAD]": self.tokenizer.pad_token_id, "length": len(output["input_ids"]), "start_index": start_index, "end_index": end_index}}
        return output

    def _map_dataset(self) -> Dataset:
        
        mapped_dataset = self.dataset.map(self._map_example)
        return mapped_dataset
           
    def __len__(self) -> int:
        
        return len(self.dataset)
    
    def __getitem__(self, index) -> dict:
        
        return self.dataset[index]
          
def main():
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = SQUADataset(tokenizer=tokenizer, is_train=True, frac=0.01)
    val_ds = SQUADataset(tokenizer=tokenizer, is_train=False, frac=0.1)
    
    train_dl = DatasetUtils.prepare_dataloader(dataset=train_ds, is_ddp=False, batch_size=32, is_train=True, collate_fn=DatasetUtils.qa_collate_fn)
    val_dl = DatasetUtils.prepare_dataloader(dataset=val_ds, is_ddp=False, batch_size=32, is_train=False, collate_fn=DatasetUtils.qa_collate_fn)
    
    print(len(train_dl), len(val_dl))
    print(train_dl.__iter__().__next__())
    print(val_dl.__iter__().__next__())
    
if __name__ == "__main__":
    from dataset_utils import DatasetUtils
    main()