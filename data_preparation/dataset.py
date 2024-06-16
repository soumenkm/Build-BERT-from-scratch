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
        self.tokenizer.model_max_length = self.max_context_legth # sets the maximum context length of tokenizer
        self.global_index = self._index_corpus()
        self.len_ds = self.__len__()
        self.current_key = None
        self.current_group_data = None
        
        print(self.global_index)
        print(f"length of dataset: {self.len_ds}")
    
    def __len__(self) -> int:
        
        return sorted(list(self.global_index.keys()))[-1]
    
    def __getitem__(self, index: int) -> dict:
        
        if index > self.len_ds or index < 0:
            raise IndexError(f"{index} should be in between 0 to {self.len_ds}")
        
        keys_list = sorted(list(self.global_index.keys()))
        for key in keys_list:
            if index <= key:
                local_index = index - (key - self.global_index[key]["len_group"])
                break
        
        if self.current_key != key:
            self.current_key = key
            # Load the group in memory
            group_path = Path(self.global_index[self.current_key]["group_json_path"])
            with open(group_path, "r") as f:
                self.current_group_data = json.load(f)
        
        raw_example = self.current_group_data["group_data"][local_index] # (isNext, SentA, SentB)
        example = self._prepare_example(raw_example=raw_example)
        
        return example
        
    def _prepare_example(self, raw_example: Tuple[int, List[int], List[int]]) -> dict:
        
        # Pad the sequence
        sequence = raw_example[1] + raw_example[2] # [CLS], t1, ..., tA, [SEP], u1, ..., uB, [SEP]
        segment_A = [0] * len(raw_example[1]) 
        segment_B = [1] * len(raw_example[2])
        assert len(sequence) <= self.max_context_legth, f"{len(sequence)} must be smaller than {self.max_context_legth}"
        
        extra_len = self.max_context_legth - len(sequence)
        padded_sequence = sequence + [self.tokenizer.pad_token_id] * extra_len
        segment_pad = [0] * extra_len
        assert len(padded_sequence) == self.max_context_legth, f"{len(padded_sequence)} must be equal to {self.max_context_legth}"

        pad_attention_mask_tensor = torch.tensor([0] * len(sequence) + [-torch.inf] * extra_len)
        segment_id_tensor = torch.tensor(segment_A + segment_B + segment_pad)
        is_next_tensor = torch.tensor(raw_example[0])
        
        # Mask the sequence
        normal_tokens_pos = [i for i in range(len(padded_sequence)) if padded_sequence[i] not in self.tokenizer.all_special_ids]
        
        num_selected_masks = int(0.15 * len(normal_tokens_pos))
        assert num_selected_masks >= 3, f"num_selected_masks={num_selected_masks} should be at least 3"
        
        num_actual_masks = max(int(0.8 * num_selected_masks), 1)
        num_replace = max(int(0.1 * num_selected_masks), 1)
        num_remain = max(int(0.1 * num_selected_masks), 1)
        
        selected_masks_pos = random.sample(normal_tokens_pos, k=num_selected_masks)
        random.shuffle(selected_masks_pos)
        
        actual_masks_pos = selected_masks_pos[:num_actual_masks]
        replace_pos = selected_masks_pos[num_actual_masks:num_actual_masks+num_replace]
        remain_pos = selected_masks_pos[num_actual_masks+num_replace:]
        
        x_sequence = [None] * len(padded_sequence)
        y_sequence = [None] * len(padded_sequence)
        
        for i in range(len(padded_sequence)):
            if i in actual_masks_pos:
                x_sequence[i] = self.tokenizer.mask_token_id
                y_sequence[i] = padded_sequence[i]
            elif i in replace_pos:
                j = random.choice(list(set(normal_tokens_pos) - set(selected_masks_pos)))
                x_sequence[i] = padded_sequence[j]
                y_sequence[i] = padded_sequence[i]
            elif i in remain_pos:
                x_sequence[i] = padded_sequence[i]
                y_sequence[i] = padded_sequence[i]
            else:
                x_sequence[i] = padded_sequence[i]
                y_sequence[i] = 0
                
        input_sequence_tensor = torch.tensor(x_sequence)
        label_sequence_tensor = torch.tensor(y_sequence)
        
        example = {
            "input_seq": input_sequence_tensor,
            "label_seq": label_sequence_tensor,
            "is_next": is_next_tensor,
            "pad_attn_mask": pad_attention_mask_tensor,
            "segment_seq": segment_id_tensor
        }
        
        return example   
             
    def _prepare_group(self) -> List[List[Path]]:
        
        groups_list = []
        for i in range(int(math.ceil(self.num_text_files/self.groups_length))):
            groups = self.text_files_list[self.groups_length * i: self.groups_length * (i+1)]
            groups_list.append(groups)
            
        return groups_list
    
    def _index_corpus(self) -> dict:
        
        group_dir = Path(self.groups_list[0][0].parent.parent, "group")
        if Path.exists(Path(group_dir, "index.json")):
            with open(Path(group_dir, "index.json"), "r") as f:
                index_dict = json.load(f)
            print(f"Index is not created but loaded from {Path(group_dir, 'index.json')}."+
                  " If you want to index it fresh then delete the index file first and run again")
            index_dict = {int(k): v for k, v in index_dict.items()}
            
            return index_dict
        
        master_index = {}
        cum_length = 0
        Path.mkdir(group_dir, parents=False, exist_ok=True)
        
        for i, group in enumerate(self.groups_list):
            group_sent_pair_list = self._prepare_sentence_pair_group(group_id=i, group_files_list=group)
            num_examples_group = len(group_sent_pair_list)
            cum_length += num_examples_group
            
            json_path = Path(group_dir, f"group_{i}.json")
            data = {"group_id": i, "group_json_path": str(json_path), "len_group": num_examples_group, "group_data": group_sent_pair_list}
            master_index[cum_length] = {k: v for k, v in data.items() if k != "group_data"}
            
            with open(json_path, "w") as f:
                json.dump(data, f)
        
        index_dict = dict(sorted(master_index.items()))
        with open(Path(group_dir, "index.json"), "w") as f:
            json.dump(index_dict, f, indent=4)
        
        return index_dict
                    
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
            if len(encoded) + 2 > self.max_context_legth//2: # [CLS] and [SEP] will be added later 
                num_discarded_sents += 1
            else:
                group_sents_token_ids.append(encoded) # Add only those whose length is less than context_length//2
                
        running_len_sent = 0
        grow_sents_token_ids = []
        cum_sents_token_ids = []
        
        for sents_token_ids in group_sents_token_ids:
            running_len_sent += len(sents_token_ids)
            if running_len_sent + 2 < self.max_context_legth//2: # [CLS] and [SEP] will be counted
                grow_sents_token_ids.extend(sents_token_ids)
            else:  
                cum_sents_token_ids.append(grow_sents_token_ids)
                grow_sents_token_ids = []
                running_len_sent = 0
        
        # Prepare sentence A and sentence B        
        sent_tuple_list = []
        num_sents = len(cum_sents_token_ids)
        is_next_list = random.choices(population=[0,1], weights=[50,50], k=num_sents)
        
        for i in range(num_sents):
            is_next = is_next_list[i]
            sent_a = cum_sents_token_ids[i]
            
            if is_next and i < num_sents - 1:
                sent_b = cum_sents_token_ids[i+1]
            else:
                shuffled_index = random.choice(list(set(range(num_sents)) - set([i, i+1])))
                sent_b = cum_sents_token_ids[shuffled_index]
            
            sent_a = [self.tokenizer.cls_token_id] + sent_a + [self.tokenizer.sep_token_id]
            sent_b = sent_b + [self.tokenizer.sep_token_id]
            
            assert len(sent_a) <= self.max_context_legth // 2, f"{i}, {len(sent_a)}" 
            assert len(sent_b) <= self.max_context_legth // 2, f"{i}, {len(sent_b)}" 
            
            sent_tuple_list.append((is_next, sent_a, sent_b)) 
        
        # for j,i in enumerate(sent_tuple_list):
        #     print(i[0], len(i[1]), len(i[2])+1)
        #     if j == 10: break
        
        # number of sentences are same since it is paired with repeated sentences so num_seq is same with number of sentences
        print(f"Group: {group_id}, Overall {num_discarded_sents} sentences are of length more than {self.max_context_legth//2} which are discarded"+
              f" and this constitutes {num_discarded_sents/(num_discarded_sents+num_sents)*100:.2f}% of total sentences of {num_sents}")

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
    
    print(train_ds[0])
    
if __name__ == "__main__":
    
    main()