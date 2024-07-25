from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset   
import torch
import re

class IMDB(Dataset):
    def __init__(self,
             tokenizer_name: str = "distilbert/distilroberta-base",
             max_seq_len: int =512,
             num_workers: int = 16,
             cache_dir: str = "./data", 
             shuffle: bool = False,
             longformer: bool = False,
             ): 
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.longformer = longformer
        
        if type(self.tokenizer_name) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, return_tensors="pt")
        else:
            self.tokenizer = self.tokenizer_name
        self.data = self._raw_text_to_tokens()
        print("IMDB dataset loaded and tokenized!")
        
    @staticmethod
    def clean_text(x):
        x = re.sub('<.*?>', ' ', x)
        x = re.sub('http\S+', ' ', x)
        x = re.sub('\s+', ' ', x)
        return x.lower().strip()

    def _preprocess_data(self, examples):
        examples["text"] = [self.clean_text(text) for text in examples["text"]]
        tokenized_data = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_seq_len)
        
        if self.longformer:
            attention_mask = torch.tensor(tokenized_data["attention_mask"])
            attention_mask[:, 0] = 2
            tokenized_data["attention_mask"] = attention_mask.tolist()
            
        return tokenized_data
    
    def _raw_text_to_tokens(self):
        print("Loading IMDB dataset...")
        raw_data = load_dataset("imdb", cache_dir=self.cache_dir, trust_remote_code=True)
        
        tokenized_imdb = raw_data.map(self._preprocess_data, batched=True, num_proc=self.num_workers, remove_columns=["text"])
        
        return tokenized_imdb
    
    def split(self):
        return self.data["train"], self.data["test"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

class Hyperpartisan(Dataset):
    def __init__(self,
                 tokenizer_name: str = "distilbert-base-uncased",
                 max_seq_len: int = 512,
                 num_workers: int = 16,
                 cache_dir: str = "./data", 
                 shuffle: bool = False,
                 longformer: bool = False,
                ): 
        """
        Initializes the Hyperpartisan dataset.

        Args:
            tokenizer_name (str): Name of the tokenizer.
            max_seq_len (int): Maximum sequence length for tokenization.
            num_workers (int): Number of workers for data processing.
            cache_dir (str): Directory to cache the dataset.
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.longformer = longformer
        
        if type(self.tokenizer_name) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, return_tensors="pt")
        else:
            self.tokenizer = self.tokenizer_name
        self.data = self._raw_text_to_tokens()
        print("Hyperpartisan dataset loaded and tokenized!")
    
    @staticmethod
    def clean_text(text):
        cleaned_text = re.sub(r"[a-zA-Z]+\/[a-zA-Z]+", " ", text)
        cleaned_text = re.sub(r"\n", " ", cleaned_text)
        cleaned_text = re.sub(r"&#160;", "", cleaned_text)
        cleaned_text = re.sub(r'<.*?>', '', cleaned_text)
        
        #remove urls
        cleaned_text = re.sub(r'http\S+', '', cleaned_text)
        cleaned_text = re.sub(r'www\S+', '', cleaned_text)
        cleaned_text = re.sub(r'href\S+', '', cleaned_text)
        
        #remove multiple spaces
        cleaned_text = re.sub(r"[ \s\t\n]+", " ", cleaned_text)
        
        #remove repetitions
        cleaned_text = re.sub(r"([!?.]){2,}", r"\1", cleaned_text)
        cleaned_text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", cleaned_text)
        
        return cleaned_text
        
    def _preprocess_data(self, examples):
        examples["text"] = [self.clean_text(text) for text in examples["text"]]
        if self.max_seq_len is None:
            tokenized_examples = self.tokenizer(examples["text"])
        else:
            tokenized_examples = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_seq_len)
        tokenized_examples["labels"] = examples["hyperpartisan"]
        tokenized_examples["labels"] = [int(label) for label in tokenized_examples["labels"]]
        if self.longformer:
            attention_mask = torch.tensor(tokenized_examples["attention_mask"])
            attention_mask[:, 0] = 2
            tokenized_examples["attention_mask"] = attention_mask.tolist()
            
        return tokenized_examples
    
    def _raw_text_to_tokens(self):
        print("Loading Hyperpartisan News Detection dataset...")
        raw_data = load_dataset("SemEvalWorkshop/hyperpartisan_news_detection", "byarticle", cache_dir=self.cache_dir, trust_remote_code=True)
        raw_data = raw_data.remove_columns(['title', 'url', 'published_at'])
        tokenized_data = raw_data.map(self._preprocess_data, batched=True, num_proc=self.num_workers, remove_columns=["text", "hyperpartisan"])
        return tokenized_data["train"]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def split(self, split_ratio: float = 0.8):
        train_size = int(split_ratio * len(self))
        test_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, test_size])
    
class CommonsenseQA(Dataset):
    def __init__(self,
                 tokenizer_name: str = "bert-base-uncased",
                 max_seq_len: int = 512,
                 cache_dir: str = "./data", ):
        
        self.max_seq_len = max_seq_len
        self.cache_dir = cache_dir
        
        if type(tokenizer_name) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        else:
            self.tokenizer = tokenizer_name
        self.data = self._raw_text_to_tokens()
        print("CommonsenseQA dataset loaded and tokenized!")
        
    def _raw_text_to_tokens(self):
        dataset = load_dataset("tau/commonsense_qa", cache_dir=self.cache_dir, trust_remote_code=True)
        dataset = dataset.remove_columns(["question_concept", "id"])
        tokenized_dataset = dataset.map(self._process_function, batched=False, num_proc=16, remove_columns=["question", "choices", "answerKey"])
        return tokenized_dataset

    def _process_function(self, data):
        question = data["question"]
        candidates = data["choices"]["text"]  
        labels = data["choices"]["label"]
        answer = data["answerKey"]
        #print("answer: ", answer) 
        #print("labels: ", labels)
        #print("question: ", question)
        question_tokens = ['[question]'] + self.tokenizer.tokenize(question) + ['[/question]']
        candidates_tokens = [['[ent]'] + self.tokenizer.tokenize(candidate)  + ['[/ent]'] for candidate in candidates]
        all_tokens = [self.tokenizer.cls_token] + question_tokens + [item for sublist in candidates_tokens for item in sublist]
        predictied_indicies = [k for k, token in enumerate(all_tokens) if token == '[ent]']
        if answer not in labels:
            answer_index = -1
        else:
            answer_index = labels.index(answer)
        all_tokens = self.tokenizer.convert_tokens_to_ids(all_tokens)
        return {"input_ids": torch.tensor(all_tokens), "label": torch.tensor(answer_index), "prediction_indices": torch.tensor(predictied_indicies)}        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def split(self):
        return self.data["train"], self.data["validation"], self.data["test"]
    
if __name__ == "__main__":
    data = CommonsenseQA()
    train, validation, test = data.split()
    print(len(train), len(validation), len(test))
    print(train[0])
    print(data.tokenizer.decode(train[0]["input_ids"]))
    