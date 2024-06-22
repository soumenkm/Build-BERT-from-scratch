import torch, tqdm, sys, torchinfo, os, json, math
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from typing import List, Tuple, Union
from transformers import BertModel, BertTokenizer
from data_preparation.squad_dataset import SQUADataset
from data_preparation.dataset_utils import DatasetUtils
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class BertModelForQA(torch.nn.Module):
    
    def __init__(self):
        
        super(BertModelForQA, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.d = self.bert.config.hidden_size
        self.start_linear = torch.nn.Linear(in_features=self.d, out_features=1, bias=False)
        self.end_linear = torch.nn.Linear(in_features=self.d, out_features=1, bias=False)
        
    def forward(self, input_ids: torch.tensor, token_type_ids: torch.tensor, attention_mask: torch.tensor) -> dict:
        """input_ids, token_type_ids, attention_mask shape are (b, T)
        attention_mask = 0 means padding and 1 means real
        """
        hidden_out = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state # (b, T, d)
        start_out = self.start_linear(hidden_out).squeeze(-1) # (b, T)
        end_out = self.end_linear(hidden_out).squeeze(-1) # (b, T)
        
        return {"start_out": start_out, "end_out": end_out} # returning logits

class BertFineTunerForQA:
    
    def __init__(self, device: torch.device, model: BertModelForQA, optimizer: "torch.optim.Optimizer", tokenizer: BertTokenizer, train_ds: Dataset, val_ds: Dataset, num_epochs: int, batch_size: int):
        
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.train_dl = DatasetUtils.prepare_dataloader(dataset=self.train_ds, is_ddp=False, batch_size=self.batch_size, is_train=True, collate_fn=DatasetUtils.qa_collate_fn)
        self.val_dl = DatasetUtils.prepare_dataloader(dataset=self.val_ds, is_ddp=False, batch_size=self.batch_size, is_train=False, collate_fn=DatasetUtils.qa_collate_fn)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1, eta_min=1e-8)
        self.clip_norm_value = 5.0
        
    def _unfreeze_layers(self, layers_list: List[str]) -> None:
        """['bert.encoder.layer.11']"""
        
        for val in self.model.parameters():
            val.requires_grad_(False)
        
        for name, val in self.model.named_parameters():
            for layer in layers_list:
                if layer in name:
                    val.requires_grad_(True)
                    break
        
        print(f"Gradients are computed for: ", [name for name, val in self.model.named_parameters() if val.requires_grad])   
    
    def _find_norm(self, is_grad: bool):
    
        norm = 0
        for val in self.model.parameters():
            if val.requires_grad:
                norm += ((val.grad if is_grad else val) ** 2).sum().item()
        norm = norm ** 0.5
        
        return norm
    
    def _forward_batch(self, batch: dict, is_train: bool) -> dict:
        
        inputs = batch["input_ids_batch"].to(self.device) # (b, T)
        padding_mask = batch["attention_mask_batch"].to(self.device) # (b, T)
        segment_ids = batch["token_type_ids_batch"].to(self.device) # (b, T)
        
        if is_train:
            self.model.train()
            out = self.model(input_ids=inputs, token_type_ids=segment_ids, attention_mask=padding_mask) # Dict[start_out = (b, T), end_out = (b, T)]
            out["start_out"].requires_grad_(True)
            out["end_out"].requires_grad_(True)
            assert out["start_out"].requires_grad == True and out["end_out"].requires_grad == True
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(input_ids=inputs, token_type_ids=segment_ids, attention_mask=padding_mask) # Dict[start_out = (b, T), end_out = (b, T)]
  
        return out 
    
    def _calc_loss_batch(self, pred_outputs: dict, true_outputs: dict) -> torch.tensor:
        
        pred_outputs = {name: val.to(self.device) for name, val in pred_outputs.items()} # (b, T)
        true_outputs = {name: val.to(self.device) for name, val in true_outputs.items()} # (b,)

        loss_start = torch.nn.functional.cross_entropy(input=pred_outputs["start_out"], target=true_outputs["start_out"])
        loss_end = torch.nn.functional.cross_entropy(input=pred_outputs["end_out"], target=true_outputs["end_out"])
        loss = (loss_start + loss_end)/2
        
        return loss # returns the computational graph also along with it
        
    def _calc_acc_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        
        pred_outputs = {name: val.to(self.device) for name, val in pred_outputs.items()} # (b, T)
        true_outputs = {name: val.to(self.device) for name, val in true_outputs.items()} # (b,)

        acc_start = (pred_outputs["start_out"].argmax(dim=-1) == true_outputs["start_out"])
        acc_end = (pred_outputs["end_out"].argmax(dim=-1) == true_outputs["end_out"])
        acc = (acc_start & acc_end).to(torch.float32).mean()
        
        return acc # returns the tensor as a scalar number
    
    def _optimize_batch(self, batch: dict) -> Tuple[float, float]:  
        
        pred_out = self._forward_batch(batch=batch, is_train=True) # Dict[start_out = (b, T), end_out = (b, T)]

        true_start_out = batch["start_index_batch"] # (b,)
        true_end_out = batch["end_index_batch"] # (b,)
        true_out = {"start_out": true_start_out, "end_out": true_end_out}
                
        loss = self._calc_loss_batch(pred_outputs=pred_out, true_outputs=true_out) # scalar
        acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out) # scalar
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.clip_norm_value, norm_type=2.0)
        self.optimizer.step()
        
        return loss.item(), acc.item()
    
    def _optimize_dataloader(self, ep: int) -> dict:
        
        num_steps = len(self.train_dl)
        loss_list = []
        acc_list = []
        
        with tqdm.tqdm(iterable=self.train_dl, 
            desc=f"{self.device}, ep: {ep}/{self.num_epochs-1}", 
            total=num_steps,
            unit=" step") as pbar:
        
            for i, batch in enumerate(pbar):
                                
                loss, acc = self._optimize_batch(batch=batch)
                self.scheduler.step(ep + i/num_steps)

                loss_list.append(loss)
                acc_list.append(acc)
                
                pbar.set_postfix({"L": f"{loss:.3f}",
                                  "A": f"{acc:.3f}",
                                  "lr": f"{self.optimizer.param_groups[0]["lr"]:.3e}",
                                  "gn": f"{self._find_norm(True):.2f}",
                                  "pn": f"{self._find_norm(False):.4f}"})

        loss_dl = sum(loss_list)/len(loss_list)
        acc_dl = sum(acc_list)/len(acc_list)
            
        return {"loss_dl": f"{loss_dl:.3f}", "acc_dl": f"{acc_dl:.3f}"}
    
    def _validate_dataloader(self, ep: int) -> dict:
        
        loss_list = []
        acc_list = []
    
        with tqdm.tqdm(iterable=self.val_dl, 
                  desc=f"{self.device}, ep: {ep}/{self.num_epochs-1}", 
                  total=len(self.val_dl),
                  unit=" batch") as pbar:
        
            for batch in pbar:  
                pred_out = self._forward_batch(batch=batch, is_train=False)
                
                true_start_out = batch["start_index_batch"] # (b,)
                true_end_out = batch["end_index_batch"] # (b,)
                true_out = {"start_out": true_start_out, "end_out": true_end_out}
                
                loss = self._calc_loss_batch(pred_outputs=pred_out, true_outputs=true_out)
                acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                
                loss_list.append(loss.item())
                acc_list.append(acc.item())
                
                pbar.set_postfix({"loss_val": f"{loss:.3f}",
                                  "acc_val": f"{acc:.3f}"})
                
        loss_dl = sum(loss_list)/len(loss_list)
        acc_dl = sum(acc_list)/len(acc_list)
            
        return {"loss_dl": f"{loss_dl:.3f}", "acc_dl": f"{acc_dl:.3f}"}
    
    def finetune(self, layers_list: List[str]) -> None:
        
        self._unfreeze_layers(layers_list=layers_list)
        for ep in range(self.num_epochs):
            train_dict = self._optimize_dataloader(ep=ep)
            val_dict = self._validate_dataloader(ep=ep)
            print(f"{self.device}, ep: {ep}/{self.num_epochs-1}, Train info: {train_dict}, Val info: {val_dict}")
          
def main(num_epochs=10, batch_size=32):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModelForQA()
    torchinfo.summary(model)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5, weight_decay=0.1)
    
    train_ds = SQUADataset(tokenizer=tokenizer, is_train=True, frac=0.5)
    val_ds = SQUADataset(tokenizer=tokenizer, is_train=False, frac=0.5)
    
    finetuner = BertFineTunerForQA(device=device, model=model, optimizer=optimizer, tokenizer=tokenizer, train_ds=train_ds, val_ds=val_ds, num_epochs=num_epochs, batch_size=batch_size)
    finetuner.finetune(layers_list=["start_linear", "end_linear", "encoder.layer.11"])
    
if __name__ == "__main__":
    
    main()