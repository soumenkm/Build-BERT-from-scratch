import torch, tqdm, sys, torchinfo, os, json, math
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from typing import List, Tuple, Union
from transformers import BertModel, BertTokenizer
from bert_model.bert_model import BertFinetuneCLSLM
from data_preparation.mnli_dataset import MNLIDataset
from bert_model.load_pretrain_model import load_pretrain_model
from data_preparation.dataset_utils import DatasetUtils
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from torch.utils.data import Subset
import torch.multiprocessing as mp

class BertCLSFineTuner:
    
    def __init__(self, 
                 is_ddp: bool,
                 device: Union[int, torch.device],
                 train_dataset: "torch.utils.data.Dataset", 
                 val_dataset: "torch.utils.data.Dataset", 
                 model: BertFinetuneCLSLM, 
                 optimizer: "torch.optim.Optimizer",
                 num_epochs: int,
                 batch_size: int,
                 val_frac: float,
                 checkpoint_dir: Path):
        
        super(BertCLSFineTuner, self).__init__()
        self.is_ddp = is_ddp
        self.device = device
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.model = DDP(model.to(self.device), device_ids=[self.device]) if self.is_ddp else model.to(self.device)
        self.optimizer = optimizer
        self.tokenizer = self.train_ds.tokenizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.checkpoint_dir = checkpoint_dir
        
        self.train_dl = DatasetUtils.prepare_dataloader(self.train_ds, self.is_ddp, self.batch_size, is_train=True, collate_fn=DatasetUtils.cls_collate_fn)
        self.val_dl = DatasetUtils.prepare_dataloader(self.val_ds, self.is_ddp, self.batch_size, is_train=True, collate_fn=DatasetUtils.cls_collate_fn)
        self.device_info = f"[GPU_{self.device} (DDP)]" if self.is_ddp else f"[{str(self.device).upper()} (SEQ)]"
        
        self.start_epoch = 0 # to keep track of resuming from checkpoint
        
        # Lists to store metrics
        self.train_list = []
        self.val_list = []
        
        # Cosine anealing
        self.initial_lr = 0.00001
        self.peak_lr = 0.01
        self.min_lr = 0.1 * self.initial_lr
        self.warmup_steps = 100
        self.total_steps = len(self.train_dl) * self.num_epochs
        self.current_step = 0
       
    def _forward_batch(self, batch: dict, is_train: bool) -> torch.tensor:
        
        inputs = batch["input_seq_batch"].to(self.device) # (b, T)
        padding_mask = batch["pad_attn_mask_batch"].to(self.device) # (b, T)
        segment_ids = batch["segment_seq_batch"].to(self.device) # (b, T)
        
        if is_train:
            self.model.train()
            out = self.model(inputs=inputs, segment_ids=segment_ids, padding_mask=padding_mask)
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(inputs=inputs, segment_ids=segment_ids, padding_mask=padding_mask)
  
        return out # (b, c)
    
    def _calc_loss_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        
        pred_outputs = pred_outputs.to(self.device)
        true_outputs = true_outputs.to(self.device)

        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
    
        loss = torch.nn.functional.cross_entropy(input=pred_outputs, target=true_outputs)
        
        return loss # returns the computational graph also along with it
        
    def _calc_acc_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        
        pred_outputs = pred_outputs.to(self.device)
        true_outputs = true_outputs.to(self.device)

        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
    
        acc = (pred_outputs.argmax(dim=-1) == true_outputs).to(torch.float32).mean()
        
        return torch.tensor(acc.item()) # returns the tensor as a scalar number
    
    def _optimize_batch(self, batch: dict) -> Tuple[float, float]:
        
        if self.current_step < self.warmup_steps:
            lr = self.initial_lr + self.current_step * (self.peak_lr - self.initial_lr) / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]["lr"] = lr        
        
        pred_out = self._forward_batch(batch=batch, is_train=True) # (b, c)
        true_out = batch["label_batch"] # (b,)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        loss = self._calc_loss_batch(pred_outputs=pred_out, true_outputs=true_out)
        acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
        loss.backward()
                
        if self.current_step > self.warmup_steps:
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0, norm_type=2.0)
                    
        self.optimizer.step()
        self.current_step += 1
        
        return loss.item(), acc.item()
    
    def _optimize_dataloader(self, ep: int) -> dict:
        
        num_steps = len(self.train_dl)
        loss_list = []
        acc_list = []
        
        with tqdm.tqdm(iterable=self.train_dl, 
            desc=f"{self.device_info}, ep: {ep}/{self.num_epochs-1}", 
            total=num_steps,
            unit=" step") as pbar:
        
            for batch in pbar:
                                
                loss, acc = self._optimize_batch(batch=batch)
                loss_list.append(loss)
                acc_list.append(acc)
                
                pbar.set_postfix({"loss": f"{loss:.3f}",
                                  "acc": f"{acc:.3f}"})
        
        loss_dl = sum(loss_list)/len(loss_list)
        acc_dl = sum(acc_list)/len(acc_list)
            
        return {"loss_dl": f"{loss_dl:.3f}", "acc_dl": f"{acc_dl:.3f}"}

    def _validate_dataloader(self, ep: int) -> dict:
        
        loss_list = []
        acc_list = []
        
        with tqdm.tqdm(iterable=self.val_dl, 
                  desc=f"{self.device_info}, ep: {ep}/{self.num_epochs-1}", 
                  total=len(self.val_dl),
                  unit=" batch") as pbar:
        
            for batch in pbar:  
                pred_out = self._forward_batch(batch=batch, is_train=False)
                true_out = batch["label_batch"]
                
                loss = self._calc_loss_batch(pred_outputs=pred_out, true_outputs=true_out)
                acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                
                loss_list.append(loss.item())
                acc_list.append(acc.item())
                
                pbar.set_postfix({"loss_val": f"{loss:.3f}",
                                  "acc_val": f"{acc:.3f}"})
                
        loss_dl = sum(loss_list)/len(loss_list)
        acc_dl = sum(acc_list)/len(acc_list)
            
        return {"loss_dl": f"{loss_dl:.3f}", "acc_dl": f"{acc_dl:.3f}"}

    def _save_checkpoint(self, ep: int) -> None:
        
        checkpoint = {
            "epoch": ep,
            "model_state": self.model.state_dict(),
            "opt_state": self.optimizer.state_dict(),
            "train_list": self.train_list,
            "val_list": self.val_list,
        }
        
        if not Path.exists(self.checkpoint_dir):
            Path.mkdir(self.checkpoint_dir, parents=True, exist_ok=True)
            
        checkpoint_path = Path(self.checkpoint_dir, f"checkpoint_epoch_{ep}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"{self.device_info}, ep: {ep}/{self.num_epochs-1}, Checkpoint saved: {checkpoint_path}")
    
    def _postprocess_load_checkpoint(self, checkpoint_path: Path) -> None:
        
        print(f"{self.device_info}, Checkpoint loaded from {checkpoint_path}")
          
    def _load_checkpoint(self, checkpoint_dir: Path) -> None:
        
        if not Path.exists(checkpoint_dir):
            print(f"{self.device_info}, No such directory: {checkpoint_dir}, starting fresh training")
            self.start_epoch = 0
            return None
        
        checkpoint_files = []
        for i in Path.iterdir(checkpoint_dir):
            if i.suffix == ".pth":
                checkpoint_files.append((i, i.stat().st_mtime))
        
        # If no checkpoints found, return 0 to start from scratch
        if not checkpoint_files:
            print(f"{self.device_info}, No checkpoints found on directory: {checkpoint_dir}, starting fresh training")
            self.start_epoch = 0
            return None
        
        # Find the latest checkpoint file
        latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])[0]
        map_loc = torch.device(f"cuda:{self.device}") if self.is_ddp else self.device
        checkpoint = torch.load(latest_checkpoint, map_location=map_loc)
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["opt_state"])
        
        # Move optimizer states to the correct device
        for state in self.optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(self.device)
            elif isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)
        
        self.train_list = checkpoint["train_list"]
        self.val_list = checkpoint["val_list"]
        
        self.start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch

        if self.is_ddp:
            if (self.device == 0):
                self._postprocess_load_checkpoint(latest_checkpoint)
        else:
            self._postprocess_load_checkpoint(latest_checkpoint)
        
        # Synchronize all processes to ensure checkpoint loading is complete
        if self.is_ddp:
            torch.distributed.barrier()
        
        return None
    
    def _save_metric(self, ep: int, save_dir: Path) -> None:
        
        history = {"train": self.train_list,
                   "val": self.val_list}

        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, 'training_validation_metrics.json')
        json.dump(history, open(save_path, "w"), indent=4)
        
        print(f"{self.device_info}, ep: {ep}/{self.num_epochs-1}, Train and validation metrics saved at: {save_path}")

    def _postprocess_train(self, ep: int, train_dict: dict) -> None:
        
        val_dict = self._validate_dataloader(ep=ep)
        self.val_list.append(val_dict)
        
        msg = f"ep: {ep}/{self.num_epochs-1}, Train info: {train_dict}, Val info: {val_dict}"
        print(f"{self.device_info}, {msg}")
        self._save_checkpoint(ep=ep)
        self._save_metric(ep=ep, save_dir=self.checkpoint_dir)
        
        # # Synchronize all processes to ensure checkpoint saving is complete
        # if self.is_ddp:
        #     torch.distributed.barrier()

    def train(self, is_load_checkpoint: bool) -> None:
        
        if is_load_checkpoint:
            self._load_checkpoint(checkpoint_dir=self.checkpoint_dir)
            
        for ep in range(self.start_epoch, self.num_epochs):
            if self.is_ddp:
                self.train_dl.sampler.set_epoch(ep)
            
            train_dict = self._optimize_dataloader(ep=ep)
            self.train_list.append(train_dict)
            
            if self.is_ddp: 
                if (self.device == 0):
                    self._postprocess_train(ep=ep, train_dict=train_dict)
            else:
                self._postprocess_train(ep=ep, train_dict=train_dict)

def ddp_setup(world_size: int, rank: int) -> None:
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)
    
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(device=rank)
    
def ddp_cleanup() -> None:
    
    torch.distributed.destroy_process_group()

def main(rank: int, is_ddp: bool, world_size: int, num_epochs: int, batch_size: int, is_load_checkpoint: bool) -> None:
    
    if is_ddp:
        if torch.cuda.is_available():
            ddp_setup(world_size=world_size, rank=rank)
            device = rank
        else:
            raise ValueError("Cuda is not available, set is_ddp=False and run again")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    config = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": 768,
        "num_heads": 12,
        "dropout_prob": 0.2,
        "max_context_length": 512,
        "num_layers": 12
    }

    model = BertFinetuneCLSLM(config=config, tokenizer=tokenizer, num_classes=3)
    torchinfo.summary(model)
    
    source_model = BertModel.from_pretrained("bert-base-cased")
    load_pretrain_model(source_model=source_model, target_model=model.bert)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    
    hf_ds = load_dataset("nyu-mll/multi_nli")
    train_ds = MNLIDataset(hf_dataset=hf_ds, tokenizer=tokenizer, is_train=True, frac=1.0)
    val_ds = MNLIDataset(hf_dataset=hf_ds, tokenizer=tokenizer, is_train=False, frac=1.0)
        
    trainer = BertCLSFineTuner(is_ddp=is_ddp,
                               device=device,
                               train_dataset=train_ds,
                               val_dataset=val_ds,
                               model=model,
                               optimizer=optimizer,
                               num_epochs=num_epochs,
                               batch_size=batch_size,
                               val_frac=0.2,
                               checkpoint_dir=Path(Path.cwd(), "ckpt"))
    
    trainer.train(is_load_checkpoint=is_load_checkpoint)
    
    if is_ddp:
        ddp_cleanup()
        
if __name__ == "__main__":
    cuda_ids = [1,2,6]
    cvd = ""
    for i in cuda_ids:
        cvd += str(i) + ","
        
    os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    num_epochs = 5
    batch_size = 32
    is_load_checkpoint = True
    is_ddp = True if len(cuda_ids) > 1 else False
    
    if is_ddp:
        world_size = len(cuda_ids)
        mp.spawn(fn=main, args=(is_ddp, world_size, num_epochs, batch_size, is_load_checkpoint), nprocs=world_size)
    else:
        main(rank=None, is_ddp=False, world_size=None, num_epochs=num_epochs, batch_size=batch_size, is_load_checkpoint=is_load_checkpoint)