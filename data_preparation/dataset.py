from transformers import BertTokenizer
import nltk, torch, tqdm, random, json, math
from pathlib import Path
from typing import Tuple, List
# nltk.download("punkt")

class BertPreTrainDataset(torch.utils.data.Dataset):
    
    def __init__(self, text_files_list: List[str], tokenizer: BertTokenizer, max_context_legth: int):
        
        super(BertPreTrainDataset, self).__init__()
        self.text_files_list = text_files_list
        self.num_text_files = len(text_files_list)
        self.groups_length = 10
        self.groups_list = self._prepare_group()
        self.max_context_legth = max_context_legth
        self.tokenizer = tokenizer
        self.index = self._index_corpus()
        print(self.index)
    
    def _prepare_group(self) -> List[List[Path]]:
        
        groups_list = []
        for i in range(int(math.ceil(self.num_text_files/self.groups_length))):
            groups = self.text_files_list[self.groups_length * i: self.groups_length * (i+1)]
            groups_list.append(groups)
            
        return groups_list
    
    def _index_corpus(self) -> dict:
        
        master_index = {}
        cum_length = 0
        Path.mkdir(Path(self.groups_list[0][0].parent.parent, "group"), parents=False, exist_ok=True)
        
        for i, group in enumerate(self.groups_list):
            group_sent_pair_list = self._prepare_sentence_pair_group(group_id=i, group_files_list=group)
            num_examples_group = len(group_sent_pair_list)
            cum_length += num_examples_group
            
            json_path = Path(group[0].parent.parent, f"group/group_{i}.json")
            data = {"group_id": i, "group_json_path": str(json_path), "len_group": num_examples_group, "group_data": group_sent_pair_list}
            master_index[cum_length] = {k: v for k, v in data.items() if k != "group_data"}
            
            with open(json_path, "w") as f:
                json.dump(data, f)
            
        return dict(sorted(master_index.items()))
                    
    def _prepare_sentence_pair_group(self, group_id: int, group_files_list: List[Path]) -> Tuple[int, List[int], List[int]]:
        
        group_text = ""
        for file in group_files_list:
            with open(file=file, mode="r") as fp:
                group_text += fp.read()
            
        group_sents = nltk.tokenize.sent_tokenize(text=group_text, language="english")
        
        group_sents_token_ids = []
        num_discarded_sents = 0
        for sent in group_sents:
            encoded = self.tokenizer.encode(text=sent, add_special_tokens=False)
            if len(encoded) > self.max_context_legth//2: 
                num_discarded_sents += 1
            group_sents_token_ids.append(encoded)
                
        running_len_sent = 0
        grow_sents_token_ids = []
        cum_sents_token_ids = []
        
        for sents_token_ids in group_sents_token_ids:
            running_len_sent += len(sents_token_ids)
            if running_len_sent + 2 < self.max_context_legth//2: # after adding 2 special tokens [CLS] and [SEP]
                grow_sents_token_ids.extend(sents_token_ids)
            else:
                if grow_sents_token_ids:
                    cum_sents_token_ids.append(grow_sents_token_ids) # if the length of single sentence is more than 256 then we don't add it
                grow_sents_token_ids = []
                running_len_sent = 0
        
        # Prepare sentence A and sentence B
        shuffled_cum_sents_token_ids = random.sample(cum_sents_token_ids, k=len(cum_sents_token_ids))
        
        sent_tuple_list = []
        num_sents = len(cum_sents_token_ids)
        is_next_list = random.choices(population=[0,1], weights=[50,50], k=num_sents)
        
        for i in range(num_sents):
            is_next = is_next_list[i]
            if i < num_sents - 1:
                if is_next:
                    sent_a = cum_sents_token_ids[i]
                    sent_b = cum_sents_token_ids[i+1]
                else:
                    sent_a = cum_sents_token_ids[i]
                    sent_b = shuffled_cum_sents_token_ids[i+1]
            else:
                sent_a = cum_sents_token_ids[i]
                sent_b = cum_sents_token_ids[0]
            
            assert len(sent_a) <= self.max_context_legth // 2 - 2, f"{i}, {len(sent_a)}" # [SEP] and [CLS] will be added
            assert len(sent_b) <= self.max_context_legth // 2 - 2, f"{i}, {len(sent_b)}" # [SEP] will be added (still -2 for safety)
            
            sent_tuple_list.append((is_next, sent_a, sent_b))
        
        total_num_sents = len(sent_tuple_list) * 2
        print(f"Group: {group_id}, Overall {num_discarded_sents} sentences are of length more than {self.max_context_legth//2} which are discarded"+
              f" and this constitutes {num_discarded_sents/(num_discarded_sents+total_num_sents)*100:.2f}% of total sentences of {total_num_sents}")

        return sent_tuple_list
           
    @staticmethod
    def train_val_split(data_dir: Path, split_ratio: float) -> Tuple[List[Path], List[Path]]:
        
        files_list = []
        for i in Path.iterdir(data_dir):
            files_list.append(i)
        
        random.shuffle(files_list)
        split = int(split_ratio * len(files_list))
        
        return files_list[0:split], files_list[split:]
        
def main():
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    data_dir = Path(Path.cwd(), "data/pre_training/gutenberg/clean")
    train_files_list, val_files_list = BertPreTrainDataset.train_val_split(data_dir=data_dir, split_ratio=0.8)

    train_ds = BertPreTrainDataset(text_files_list=train_files_list, tokenizer=tokenizer, max_context_legth=512)
    
if __name__ == "__main__":
    
    main()