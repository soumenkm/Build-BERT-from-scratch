import torch, random, math
from pathlib import Path
from typing import Tuple, List
import random

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
    def cls_collate_fn(example_list: List[dict]) -> dict:
        """examples_list[i] = dataset.__getitem__(i)"""

        input_sequence_batch = []
        label_batch = []
        segment_id_batch = []
        padding_mask_batch = []
        length_batch = []
        
        for elem in example_list:
            length_batch.append(elem["length"])

        max_length = max(length_batch)
        
        for elem in example_list:
            input_sequence_batch.append(torch.tensor(elem["input_seq"] + [elem["pad_token_id"]] * (max_length - elem["length"])))
            label_batch.append(torch.tensor(elem["label"]))
            segment_id_batch.append(torch.tensor(elem["segment_seq"] + [0] * (max_length - elem["length"])))
            padding_mask_batch.append(torch.tensor([1] * elem["length"] + [elem["pad_token_id"]] * (max_length - elem["length"])))
            
        input_sequence_batch = torch.stack(tensors=input_sequence_batch, dim=0)
        label_batch = torch.stack(tensors=label_batch, dim=0)
        segment_id_batch = torch.stack(tensors=segment_id_batch, dim=0)
        padding_mask_batch = torch.stack(tensors=padding_mask_batch, dim=0)
        
        assert input_sequence_batch.shape == segment_id_batch.shape == padding_mask_batch.shape
        
        data_dict = {
            "input_seq_batch": input_sequence_batch, # (b, T)
            "label_batch": label_batch, # (b,)
            "length_batch": max_length, # (T)
            "segment_seq_batch": segment_id_batch, # (b, T)
            "pad_attn_mask_batch": padding_mask_batch # (b, T)
        }
        
        return data_dict
    
    @staticmethod
    def qa_collate_fn(example_list: List[dict]) -> dict:
        """examples_list[i] = dataset.__getitem__(i)"""

        input_sequence_batch = []
        start_index_batch = []
        end_index_batch = []
        segment_id_batch = []
        padding_mask_batch = []
        length_batch = []
        
        for elem in example_list:
            length_batch.append(elem["length"])

        max_length = max(length_batch)
        
        for elem in example_list:
            input_sequence_batch.append(torch.tensor(elem["input_ids"] + [elem["[PAD]"]] * (max_length - elem["length"])))
            start_index_batch.append(torch.tensor(elem["start_index"]))
            end_index_batch.append(torch.tensor(elem["end_index"]))
            segment_id_batch.append(torch.tensor(elem["token_type_ids"] + [0] * (max_length - elem["length"])))
            padding_mask_batch.append(torch.tensor(elem["attention_mask"] + [elem["[PAD]"]] * (max_length - elem["length"])))
            
        input_sequence_batch = torch.stack(tensors=input_sequence_batch, dim=0)
        start_index_batch = torch.stack(tensors=start_index_batch, dim=0)
        end_index_batch = torch.stack(tensors=end_index_batch, dim=0)
        segment_id_batch = torch.stack(tensors=segment_id_batch, dim=0)
        padding_mask_batch = torch.stack(tensors=padding_mask_batch, dim=0)
        
        assert input_sequence_batch.shape == segment_id_batch.shape == padding_mask_batch.shape
        
        data_dict = {
            "input_ids_batch": input_sequence_batch, # (b, T)
            "start_index_batch": start_index_batch, # (b,)
            "end_index_batch": end_index_batch, # (b,)
            "max_length_batch": max_length, # (T)
            "token_type_ids_batch": segment_id_batch, # (b, T)
            "attention_mask_batch": padding_mask_batch # (b, T)
        }
        
        return data_dict
    
    @staticmethod
    def prepare_dataloader(dataset: torch.utils.data.Dataset, is_ddp: int, batch_size: int, is_train: bool, collate_fn: "function", val_frac: float=0.2, num_workers: int=1) -> torch.utils.data.DataLoader:
        
        if is_train:
            if is_ddp:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset, shuffle=True, drop_last=True)
                train_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn, sampler=sampler)
            else:
                train_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn, drop_last=True)
        else:
            len_val_ds = len(dataset)
            indices = random.sample(range(len_val_ds), k=int(len_val_ds * val_frac))
            dataset = torch.utils.data.Subset(dataset=dataset, indices=indices)
            val_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn, drop_last=True)

        return train_dl if is_train else val_dl     
           
    @staticmethod
    def train_val_split(data_dir: Path, split_ratio: float) -> Tuple[List[Path], List[Path]]:
        
        files_list = []
        for i in Path.iterdir(data_dir):
            files_list.append(i)
        
        random.shuffle(files_list)
        split = int(split_ratio * len(files_list))
        
        return files_list[0:split], files_list[split:]
 