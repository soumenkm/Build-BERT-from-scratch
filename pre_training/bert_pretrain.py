import torch, tqdm, sys, torchinfo, os, json
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from typing import List, Tuple, Union
from transformers import BertModel, BertTokenizer
from bert_model.bert_model import BertPretrainLM
from data_preparation.dataset import BertPreTrainDataset
from bert_model.load_pretrain_model import load_pretrain_model
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

class BertPreTrainer:
    
    def __init__(self, 
                 is_ddp: bool,
                 device: Union[int, torch.device],
                 train_dataset: BertPreTrainDataset, 
                 val_dataset: BertPreTrainDataset, 
                 model: BertPretrainLM, 
                 optimizer: "torch.optim.Optimizer",
                 num_epochs: int,
                 batch_size: int,
                 val_frac: float,
                 checkpoint_dir: Path):
        
        super(BertPreTrainer, self).__init__()
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
        
        self.train_dl = self._create_dataloader(is_train=True)
        self.val_dl = self._create_dataloader(is_train=False)
        self.device_info = f"[GPU_{self.device} (DDP)]" if self.is_ddp else f"[{str(self.device).upper()} (SEQ)]"
        
        self.start_epoch = 0 # to keep track of resuming from checkpoint
        
        # Lists to store metrics
        self.train_list = []
        self.val_list = []
    
    def _create_dataloader(self, is_train: bool) -> "torch.utils.data.DataLoader":
        
        if is_train:
            if self.is_ddp:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.train_ds, shuffle=True, drop_last=True)
                train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=4, shuffle=False, collate_fn=self.train_ds.collate_fn, sampler=sampler)
            else:
                train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=4, shuffle=True, collate_fn=self.train_ds.collate_fn, drop_last=True)
        else:
            len_val_ds = len(self.val_ds)
            indices = torch.randperm(n=len_val_ds)[:int(len_val_ds * self.val_frac)]
            val_ds = torch.utils.data.Subset(dataset=self.val_ds, indices=indices)
            val_dl = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, num_workers=4, shuffle=False, collate_fn=self.val_ds.collate_fn, drop_last=True)

        return train_dl if is_train else val_dl
       
    def _forward_batch(self, batch: dict, is_train: bool) -> dict:
        
        inputs = batch["input_seq_batch"].to(self.device) # (b, T)
        padding_mask = batch["pad_attn_mask_batch"].to(self.device) # (b, T)
        segment_ids = batch["segment_seq_batch"].to(self.device) # (b, T)
        mlm_label_mask = batch["label_seq_batch"].to(self.device) # (b, T)
        
        if is_train:
            self.model.train()
            out = self.model(inputs=inputs, segment_ids=segment_ids, padding_mask=padding_mask, mlm_label_mask=mlm_label_mask)
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(inputs=inputs, segment_ids=segment_ids, padding_mask=padding_mask, mlm_label_mask=mlm_label_mask)
  
        return out
    
    def _calc_loss_batch(self, pred_outputs: dict, true_outputs: dict) -> Tuple[torch.tensor, dict]:
        
        pred_outputs_nsp = pred_outputs["nsp"].to(self.device)
        pred_outputs_mlm = pred_outputs["mlm"].to(self.device)
        true_outputs_nsp = true_outputs["nsp"].to(self.device)
        true_outputs_mlm = true_outputs["mlm"].to(self.device)
        
        assert pred_outputs_nsp.dim() == 2, f"pred_outputs['nsp'].shape = {pred_outputs_nsp.shape} must be (b, 2)"
        assert true_outputs_nsp.dim() == 1, f"true_outputs['nsp'].shape = {true_outputs_nsp.shape} must be (b,)"
        assert pred_outputs_mlm.dim() == 2, f"pred_outputs['mlm'].shape = {pred_outputs_mlm.shape} must be (bm, V)"
        assert true_outputs_mlm.dim() == 1, f"true_outputs['mlm'].shape = {true_outputs_mlm.shape} must be (bm,)"
    
        loss_nsp = torch.nn.functional.cross_entropy(input=pred_outputs_nsp, target=true_outputs_nsp)
        loss_mlm = torch.nn.functional.cross_entropy(input=pred_outputs_mlm, target=true_outputs_mlm)
        loss = loss_nsp + loss_mlm
        
        return loss, {"loss_nsp": loss_nsp.item(), "loss_mlm": loss_mlm.item()} # returns the computational graph also along with it
        
    def _calc_acc_batch(self, pred_outputs: dict, true_outputs: dict) -> dict:
        
        pred_outputs_nsp = pred_outputs["nsp"].to(self.device)
        pred_outputs_mlm = pred_outputs["mlm"].to(self.device)
        true_outputs_nsp = true_outputs["nsp"].to(self.device)
        true_outputs_mlm = true_outputs["mlm"].to(self.device)
        
        assert pred_outputs_nsp.dim() == 2, f"pred_outputs['nsp'].shape = {pred_outputs_nsp.shape} must be (b, 2)"
        assert true_outputs_nsp.dim() == 1, f"true_outputs['nsp'].shape = {true_outputs_nsp.shape} must be (b,)"
        assert pred_outputs_mlm.dim() == 2, f"pred_outputs['mlm'].shape = {pred_outputs_mlm.shape} must be (bm, V)"
        assert true_outputs_mlm.dim() == 1, f"true_outputs['mlm'].shape = {true_outputs_mlm.shape} must be (bm,)"
    
        acc_nsp = (pred_outputs_nsp.argmax(dim=-1) == true_outputs_nsp).to(torch.float32).mean()
        acc_mlm = (pred_outputs_mlm.argmax(dim=-1) == true_outputs_mlm).to(torch.float32).mean()
        acc = {"acc_nsp": acc_nsp.item(), "acc_mlm": acc_mlm.item()}
        
        return acc # returns the scalar number
    
    def _optimize_batch(self, batch: dict) -> Tuple[dict, dict]:
        
        pred_out = self._forward_batch(batch=batch, is_train=True) # dict.keys() = ["mlm", "nsp"]
        mlm_true_out = batch["label_seq_batch"]
        mlm_true_out = mlm_true_out[mlm_true_out != 0]
        true_out = {"nsp": batch["is_next_batch"], "mlm": mlm_true_out}
        
        self.optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = self._calc_loss_batch(pred_outputs=pred_out, true_outputs=true_out)
        acc_dict = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
        loss.backward()
        self.optimizer.step()
        
        return loss_dict, acc_dict
    
    def _optimize_dataloader(self, ep: int) -> dict:
        
        num_steps = len(self.train_dl)
        loss_m_list = []
        loss_n_list = []
        acc_m_list = []
        acc_n_list = []
        
        with tqdm.tqdm(iterable=self.train_dl, 
                  desc=f"{self.device_info}, ep: {ep}/{self.num_epochs-1}", 
                  total=num_steps,
                  unit=" step") as pbar:
        
            for batch in pbar:
                loss_dict, acc_dict = self._optimize_batch(batch=batch)
                loss_m_list.append(loss_dict["loss_mlm"])
                loss_n_list.append(loss_dict["loss_nsp"])
                acc_m_list.append(acc_dict["acc_mlm"])
                acc_n_list.append(acc_dict["acc_nsp"])
                
                pbar.set_postfix({"Ln": f"{loss_dict['loss_nsp']:.3f}",
                                  "Lm": f"{loss_dict['loss_mlm']:.3f}",
                                  "An": f"{acc_dict['acc_nsp']:.3f}",
                                  "Am": f"{acc_dict['acc_mlm']:.3f}"})
        
        loss_m_dl = sum(loss_m_list)/len(loss_m_list)
        loss_n_dl = sum(loss_n_list)/len(loss_n_list)
        acc_m_dl = sum(acc_m_list)/len(acc_m_list)
        acc_n_dl = sum(acc_n_list)/len(acc_n_list)
            
        return {"loss_mlm_dl": loss_m_dl, "loss_nsp_dl": loss_n_dl, "acc_mlm_dl": acc_m_dl, "acc_nsp_dl": acc_n_dl}

    def _validate_dataloader(self, ep: int) -> dict:
        
        loss_m_list = []
        loss_n_list = []
        acc_m_list = []
        acc_n_list = []
        
        with tqdm.tqdm(iterable=self.val_dl, 
                  desc=f"{self.device_info}, ep: {ep}/{self.num_epochs-1}", 
                  total=len(self.val_dl),
                  unit=" batch") as pbar:
        
            for batch in pbar:  
                pred_out = self._forward_batch(batch=batch, is_train=False)
                mlm_true_out = batch["label_seq_batch"]
                mlm_true_out = mlm_true_out[mlm_true_out != 0]
                true_out = {"nsp": batch["is_next_batch"], "mlm": mlm_true_out}
                
                loss_dict = self._calc_loss_batch(pred_outputs=pred_out, true_outputs=true_out)[-1]
                acc_dict = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                
                loss_m_list.append(loss_dict["loss_mlm"])
                loss_n_list.append(loss_dict["loss_nsp"])
                acc_m_list.append(acc_dict["acc_mlm"])
                acc_n_list.append(acc_dict["acc_nsp"])
                
                pbar.set_postfix({"Ln_v": f"{loss_dict['loss_nsp']:.3f}",
                                  "Lm_v": f"{loss_dict['loss_mlm']:.3f}",
                                  "An_v": f"{acc_dict['acc_nsp']:.3f}",
                                  "Am_v": f"{acc_dict['acc_mlm']:.3f}"})
                
        loss_m_dl = sum(loss_m_list)/len(loss_m_list)
        loss_n_dl = sum(loss_n_list)/len(loss_n_list)
        acc_m_dl = sum(acc_m_list)/len(acc_m_list)
        acc_n_dl = sum(acc_n_list)/len(acc_n_list)
            
        return {"loss_mlm_dl": loss_m_dl, "loss_nsp_dl": loss_n_dl, "acc_mlm_dl": acc_m_dl, "acc_nsp_dl": acc_n_dl}

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
        
        return None
    
    def _save_metric(self, ep: int, save_dir: Path) -> None:
        
        history = {"train": self.train_list,
                   "val": self.val_list}

        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, 'training_validation_metrics.json')
        json.dump(history, open(save_path, "w"))
        
        print(f"{self.device_info}, ep: {ep}/{self.num_epochs-1}, Train and validation metrics saved at: {save_path}")

    def _postprocess_train(self, ep: int, train_dict: dict) -> None:
        
        val_dict = self._validate_dataloader(ep=ep)
        self.val_list.append(val_dict)
        
        msg = f"ep: {ep}/{self.num_epochs-1}, Train info: {train_dict}, Val info: {val_dict}"
        print(f"{self.device_info}, {msg}")
        self._save_checkpoint(ep=ep)
        self._save_metric(ep=ep, save_dir=self.checkpoint_dir)

    def train(self, is_load_checkpoint: bool) -> None:
        
        if is_load_checkpoint:
            self._load_checkpoint(checkpoint_dir=self.checkpoint_dir)
            
        # Synchronize all processes to ensure checkpoint loading is complete
        if self.is_ddp:
            torch.distributed.barrier()
        
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
            
            # Synchronize all processes to ensure checkpoint saving is complete
            if self.is_ddp:
                torch.distributed.barrier()

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

    model = BertPretrainLM(config=config, tokenizer=tokenizer)
    torchinfo.summary(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    data_dir = Path(Path.cwd(), "data/pre_training/gutenberg/clean")
    train_files_list, val_files_list = BertPreTrainDataset.train_val_split(data_dir=data_dir, split_ratio=0.8)

    train_ds = BertPreTrainDataset(text_files_list=train_files_list, tokenizer=tokenizer, max_context_legth=512, is_train=True)
    val_ds = BertPreTrainDataset(text_files_list=val_files_list, tokenizer=tokenizer, max_context_legth=512, is_train=False)
    
    trainer = BertPreTrainer(is_ddp=is_ddp,
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
    cuda_ids = [0,1,2,3,4,5,6,7]
    cvd = ""
    for i in cuda_ids:
        cvd += str(i) + ","
        
    os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    num_epochs = 2
    batch_size = 32
    is_load_checkpoint = True
    is_ddp = True if len(cuda_ids) > 1 else False
    
    if is_ddp:
        world_size = len(cuda_ids)
        mp.spawn(fn=main, args=(is_ddp, world_size, num_epochs, batch_size, is_load_checkpoint), nprocs=world_size)
    else:
        main(rank=None, is_ddp=False, world_size=None, num_epochs=num_epochs, batch_size=batch_size, is_load_checkpoint=is_load_checkpoint)