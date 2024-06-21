import torch, tqdm, sys, torchinfo, os, json, math
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from typing import List, Tuple, Union
from transformers import BertModel, BertTokenizer, DataCollatorWithPadding
from datasets import load_dataset

class BertModelFinetuneCLSLM(torch.nn.Module):
    
    def __init__(self, num_classes: int):
        super(BertModelFinetuneCLSLM, self).__init__()
        self.c = num_classes
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        for val in self.bert.parameters():
            val.requires_grad_(False)
        for val in self.bert.pooler.parameters():
            val.requires_grad_(True)
        for val in self.bert.encoder.layer[-1].parameters():
            val.requires_grad_(True)
            
        self.d = self.bert.config.hidden_size
        self.cls_head = torch.nn.Linear(in_features=self.d, out_features=self.c, bias=True)
        
    def forward(self, inputs: torch.tensor, segment_ids: torch.tensor, padding_mask: torch.tensor) -> torch.tensor:
        """inputs.shape = segment_ids.shape = padding_mask.shape = (b, T)
        segment_ids = 0 means sentence A and padding position and 1 means sentence B
        padding_mask = 0 means it is padded token and 1 means it is real token
        """
        
        x = self.bert(input_ids=inputs, token_type_ids=segment_ids, attention_mask=padding_mask)
        pooled_out = x.pooler_output # (b, d)
        z1 = self.cls_head(pooled_out) # (b, c)
        
        return z1

def _find_norm(model: BertModelFinetuneCLSLM, is_grad: bool):
    
    norm = 0
    for val in model.cls_head.parameters():
        norm += ((val.grad if is_grad else val) ** 2).sum().item()
    norm = norm ** 0.5
    
    return norm

def finetune(num_epochs: int, train_dl: "Dataloader", model: BertModelFinetuneCLSLM, optimizer: "Optimizer", device: torch.device) -> None:
    
    print(f"The gradients are computed for: ", [name for name, val in model.named_parameters() if val.requires_grad]) 
    model = model.to(device)
    model.train()
    
    for ep in range(num_epochs):
        train_loss = 0
        train_acc = 0
        with tqdm.tqdm(iterable=train_dl, 
            desc=f"[{device}], ep: {ep}/{num_epochs-1}", 
            total=len(train_dl),
            unit=" step") as pbar:
        
            for batch in pbar:
                
                inputs = batch["input_ids"].to(device) # (b, T)
                padding_mask = batch["attention_mask"].to(device) # (b, T)
                segment_ids = batch["token_type_ids"].to(device) # (b, T)
                true_out = batch["labels"].to(device) # (b,)

                pred_out = model(inputs=inputs, segment_ids=segment_ids, padding_mask=padding_mask)
                pred_out.requires_grad_(True)
                
                loss = torch.nn.functional.cross_entropy(pred_out, true_out)
                acc = (pred_out.argmax(dim=-1) == true_out).to(torch.float32).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()    
                torch.nn.utils.clip_grad_norm_(model.cls_head.parameters(), max_norm=2.0)             
                optimizer.step()
                                 
                pbar.set_postfix({"L": f"{loss.item():.3f}",
                                  "A": f"{acc.item():.3f}",
                                  "gn": f"{_find_norm(model, True):.2f}",
                                  "pn": f"{_find_norm(model, False):.4f}"})
                train_loss += loss.item()
                train_acc += acc.item()
        print(f"Train loss: {train_loss/len(train_dl):.3f}, Train acc: {train_acc/len(train_dl):.3f}")

def main() -> None:
    
    num_epochs = 15
    batch_size = 16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": 768,
        "num_heads": 12,
        "dropout_prob": 0.1,
        "max_context_length": 512,
        "num_layers": 12
    }

    model = BertModelFinetuneCLSLM(num_classes=3)
    torchinfo.summary(model) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)
    
    # Load dataset and prepare data loader
    hf_ds = load_dataset("multi_nli")
    
    # Select only 10,000 examples from the training dataset before tokenization
    small_train_ds = hf_ds["train"].select(range(16000))
    
    def tokenize_function(examples):
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)
    
    tokenized_train_ds = small_train_ds.map(tokenize_function, batched=True)
    tokenized_train_ds.set_format("torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dl = torch.utils.data.DataLoader(tokenized_train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    finetune(num_epochs=num_epochs, train_dl=train_dl, model=model, optimizer=optimizer, device=device)

    print("DONE")

 
if __name__ == "__main__":
    cuda_ids = [0]
    cvd = ""
    for i in cuda_ids:
        cvd += str(i) + ","
        
    os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    
    main()