from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from transformers import BertTokenizer

class MNLIDataset(Dataset):
    
    def __init__(self, hf_dataset: Dataset, tokenizer: BertTokenizer, is_train: bool, frac: float):
        
        super(MNLIDataset, self).__init__()
        self.hf_ds = hf_dataset
        self.tokenizer = tokenizer
        self.is_train = is_train
        if self.is_train:
            self.dataset = Subset(self.hf_ds["train"], indices=range(int(frac * len(self.hf_ds["train"]))))
        else:
            concat_ds = ConcatDataset([self.hf_ds["validation_matched"],
                                       self.hf_ds["validation_mismatched"]])
            self.dataset = Subset(concat_ds, indices=range(int(frac * len(concat_ds)))) 
               
    def __len__(self) -> int:
        
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> dict:
        
        raw_example = self.dataset[index]
        
        sent_A = raw_example["premise"]
        sent_B = raw_example["hypothesis"]
        label = raw_example["label"]
        
        sent_A_token_ids = self.tokenizer.encode(sent_A, add_special_tokens=False, truncation=True, max_length=self.tokenizer.model_max_length//2-2)
        sent_B_token_ids = self.tokenizer.encode(sent_B, add_special_tokens=False, truncation=True, max_length=self.tokenizer.model_max_length//2-1)
        
        sent_A_token_ids = [self.tokenizer.cls_token_id] + sent_A_token_ids + [self.tokenizer.sep_token_id]
        sent_B_token_ids = sent_B_token_ids + [self.tokenizer.sep_token_id]
        sequence = sent_A_token_ids + sent_B_token_ids
        segment_ids = [0] * len(sent_A_token_ids) + [1] * len(sent_B_token_ids)
        
        example = {
            "input_seq": sequence,
            "label": int(label),
            "length": len(sequence),
            "pad_token_id": self.tokenizer.pad_token_id,
            "segment_seq": segment_ids
        }
        
        return example
        
def main():
    from dataset_utils import DatasetUtils
    
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    hf_ds = load_dataset("nyu-mll/multi_nli")
    train_ds = MNLIDataset(hf_dataset=hf_ds, tokenizer=tokenizer, is_train=True)
    val_ds = MNLIDataset(hf_dataset=hf_ds, tokenizer=tokenizer, is_train=False)
    
    train_dl = DatasetUtils.prepare_dataloader(dataset=train_ds, is_ddp=False, is_train=True, batch_size=32, collate_fn=DatasetUtils.cls_collate_fn)
    val_dl = DatasetUtils.prepare_dataloader(dataset=val_ds, is_ddp=False, is_train=False, batch_size=32, collate_fn=DatasetUtils.cls_collate_fn)
    
    print(train_dl.__iter__().__next__())

if __name__ == "__main__":
    
    main()