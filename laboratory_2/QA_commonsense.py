from transformers import AutoModel
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from data.data import CommonsenseQA
from torch.optim.lr_scheduler import PolynomialLR
from torch.optim import AdamW
from trainer import Trainer
import random
import numpy as np
import argparse

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_name='distilbert-base-uncased'):
    """Small model surgery to add 4 new tokens to the vocabulary of a distilbert model."""
    model = AutoModel.from_pretrained(model_name)
    current_embed = model.embeddings.word_embeddings.weight
    current_vocab_size, embed_size = current_embed.size()
    new_embed = model.embeddings.word_embeddings.weight.new_empty(current_vocab_size + 4, embed_size)
    new_embed.normal_(mean=torch.mean(current_embed).item(), std=torch.std(current_embed).item())
    new_embed[:current_vocab_size] = current_embed
    model.embeddings.word_embeddings.num_embeddings = current_vocab_size + 4
    del model.embeddings.word_embeddings.weight
    model.embeddings.word_embeddings.weight = torch.nn.Parameter(new_embed)
    print("Loaded model")
    print(model)
    return model

def get_distilbert_tokenizer(tokenizer_name='distilbert-base-uncased'):
    """add [question], [/question], [ent], [/ent] special tokens to the tokenizer"""
    from transformers import DistilBertTokenizer
    additional_tokens = ['[question]', '[/question]', '[ent]', '[/ent]'] 
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_tokens(additional_tokens)
    return tokenizer


class QuestionAnsweringModel(torch.nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super(QuestionAnsweringModel, self).__init__()
        
        self.model = load_model(model_name)
        self.answer_score = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.loss = torch.nn.CrossEntropyLoss()
        
    def forward(self, data):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]
        prediction_indices = data["prediction_indices"]
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Select the hidden states of the predicted indices
        batch_size = hidden_states.size(0)
        predicted_states = hidden_states[torch.arange(batch_size).unsqueeze(1), prediction_indices]
        logits = self.answer_score(predicted_states).squeeze(-1)
        
        if labels is not None:
            loss = self.loss(logits, labels)
            return loss, logits
        return logits

    def predict(self, data):
        logits = self(data)
        predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def train_step(self, data):
        loss, logits = self(data)
        return loss, logits

    def test_step(self, data):
        loss, logits = self(data)
        return loss, logits
        
def main(args):

    tokenizer = get_distilbert_tokenizer()
    data = CommonsenseQA(tokenizer_name=tokenizer, max_seq_len=512)
    train, validation, test = data.split() 

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train, batch_size=16, shuffle=True, collate_fn=data_collator)
    test_loader = DataLoader(test, batch_size=16, shuffle=True, collate_fn=data_collator) # NOTE: it has not the label, can't use for evaluation
    validation_loader = DataLoader(validation, batch_size=16, shuffle=False, collate_fn=data_collator)

    num_epochs = args.epochs
    total_steps = len(train_loader) * num_epochs 

    model = QuestionAnsweringModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = PolynomialLR(optimizer, total_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def compute_metrics(data, output):
        loss, logits = output
        preds = torch.argmax(logits, dim=1)
        return {"accuracy": (preds == data["labels"]).float().mean().item()}

    trainer = Trainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        default_root_dir="./data",
        optimizer=optimizer,
        scheduler=scheduler,
        compute_metrics=compute_metrics,
        logger="wandb",
        log=args.log,
        max_epochs=num_epochs,
        use_mixed_precision=False,
        gradient_accumulation_steps=1, 
        warmup_steps=0.1*total_steps,
        project_name="QA_commonsense",
    )

    trainer.train(model, train_loader, validation_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--default_root_dir", type=str, default="./model/")
    parser.add_argument("--val_check_interval", type=int, default=10)
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--project_name", type=str, default="WikihopQA")


    args = parser.parse_args()
    main(args)
