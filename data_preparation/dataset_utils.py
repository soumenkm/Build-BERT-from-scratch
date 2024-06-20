import torch, random, math
from pathlib import Path
from typing import Tuple, List

class DatasetUtils:
    
    def __init__(self):
        pass
    
    @staticmethod
    def collate_fn(example_list: List[dict]) -> dict:
        """examples_list[i] = dataset.__getitem__(i)"""

        input_sequence_batch = []
        label_sequence_batch = []
        is_next_batch = []
        pad_attention_mask_batch = []
        segment_id_batch = []
        
        for elem in example_list:
            input_sequence_batch.append(elem["input_seq"])
            label_sequence_batch.append(elem["label_seq"])
            is_next_batch.append(elem["is_next"])
            pad_attention_mask_batch.append(elem["pad_attn_mask"])
            segment_id_batch.append(elem["segment_seq"])
            
        input_sequence_batch = torch.stack(tensors=input_sequence_batch, dim=0)
        label_sequence_batch = torch.stack(tensors=label_sequence_batch, dim=0)
        is_next_batch = torch.stack(tensors=is_next_batch, dim=0)
        pad_attention_mask_batch = torch.stack(tensors=pad_attention_mask_batch, dim=0)
        segment_id_batch = torch.stack(tensors=segment_id_batch, dim=0)
        
        data_dict = {
            "input_seq_batch": input_sequence_batch, # (b, T)
            "label_seq_batch": label_sequence_batch, # (b, T)
            "is_next_batch": is_next_batch, # (b,)
            "pad_attn_mask_batch": pad_attention_mask_batch, # (b, T) 
            "segment_seq_batch": segment_id_batch # (b, T)
        }
        
        return data_dict
    
    @staticmethod
    def prepare_dataloader(dataset: torch.utils.data.Dataset, is_ddp: int, batch_size: int, is_train: bool, val_frac: float=0.2, num_workers: int=1) -> torch.utils.data.DataLoader:
        
        if is_train:
            if is_ddp:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset, shuffle=True, drop_last=True)
                train_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=DatasetUtils.collate_fn, sampler=sampler)
            else:
                train_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=DatasetUtils.collate_fn, drop_last=True)
        else:
            len_val_ds = len(dataset)
            indices = torch.randperm(n=len_val_ds)[:int(len_val_ds * val_frac)]
            val_ds = torch.utils.data.Subset(dataset=dataset, indices=indices)
            val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=DatasetUtils.collate_fn, drop_last=True)

        return train_dl if is_train else val_dl     
           
    @staticmethod
    def train_val_split(data_dir: Path, split_ratio: float) -> Tuple[List[Path], List[Path]]:
        
        files_list = []
        for i in Path.iterdir(data_dir):
            files_list.append(i)
        
        random.shuffle(files_list)
        split = int(split_ratio * len(files_list))
        
        return files_list[0:split], files_list[split:]
 