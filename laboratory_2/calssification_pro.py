"""Just for fun but I get CUDA out of memory error"""

from laboratory_2.distil_bert import DistilBERTModel
from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from torch import optim
from data import Hyperpartisan
import torch
from trainer import Trainer
from torch.optim.lr_scheduler import PolynomialLR


model_for_weights = AutoModel.from_pretrained("distilbert-base-uncased")
config = model_for_weights.config # prendiamo il config del modello per inizializzare il nostro modello
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def pad_to_max_length(input_ids, attention_mask, max_length, pad_token_id):
    """
    Pad input_ids and attention_mask to max_length
    """
    padding_length = max_length - input_ids.size(1)
    assert padding_length == max_length - attention_mask.size(1)
    if padding_length > 0:
        input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), value=pad_token_id)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length), value=0)
    return input_ids, attention_mask


def get_model_activation(model, input_ids, max_seq_len, truncate_seq_len):
    
    seq_len = input_ids.size(1)
    if seq_len < max_seq_len:
        attention_mask = torch.ones(input_ids.size(), dtype=torch.long)
        input_ids, attention_mask = pad_to_max_length(input_ids, attention_mask, max_seq_len, tokenizer.pad_token_id)
        attention_mask = attention_mask.to(input_ids.device)
        return [model(input_ids, attention_mask)]
    else:
        all_activations = []
        for i in range(0, seq_len, truncate_seq_len):
            if i + max_seq_len > seq_len:
                input_ids_slice = input_ids[:, i:]
            else:
                input_ids_slice = input_ids[:, i:i+max_seq_len]
            attention_mask = torch.ones(input_ids_slice.size(), dtype=torch.long)
            input_ids_slice, attention_mask = pad_to_max_length(input_ids_slice, attention_mask, max_seq_len, tokenizer.pad_token_id)
            attention_mask = attention_mask.to(input_ids.device)
            all_activations.append(model(input_ids_slice, attention_mask))
        return all_activations               
        
        
class LongNLPClassifier(nn.Module):
    def __init__(self, config, num_classes, max_seq_len, stride):
        super(LongNLPClassifier, self).__init__()
        
        self.transformer = DistilBERTModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        self.loss = nn.CrossEntropyLoss()
        self.config = config
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.stride = stride
        assert self.max_seq_len <= self.config.max_position_embeddings, "max_seq_len must be less than or equal to config.max_position_embeddings"
        
    def forward(self, data):
        input_ids = data["input_ids"]
        labels = data["labels"]
        
        # Implement sliding window
        chunk_hidden_states = []
        for i in range(0, input_ids.size(1), self.stride):
            chunk_input_ids = input_ids[:, i:i+self.max_seq_len]
            attention_mask = torch.ones_like(chunk_input_ids)
            
            # Pad if necessary
            if chunk_input_ids.size(1) < self.max_seq_len:
                padding_length = self.max_seq_len - chunk_input_ids.size(1)
                chunk_input_ids = torch.nn.functional.pad(chunk_input_ids, (0, padding_length), value=self.transformer.config.pad_token_id)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length), value=0)
            
            chunk_output = self.transformer(chunk_input_ids, attention_mask=attention_mask)
            chunk_hidden_states.append(chunk_output[:, 0, :])  # Get the [CLS] token representation

        # Average hidden states
        averaged_hidden_state = torch.mean(torch.stack(chunk_hidden_states), dim=0)
        
        # Pass through classifier
        logits = self.classifier(averaged_hidden_state)
        loss = self.loss(logits, labels)
        
        return loss, logits
        
    def train_step(self, data):
        loss, logits = self(data)
        return loss, logits
    
    def test_step(self, data):
        loss, logits = self(data)
        return loss, logits
        
# Initialize the dataset
dataset = Hyperpartisan(tokenizer_name="distilbert-base-uncased", max_seq_len=None, num_workers=16, cache_dir="./data", shuffle=True)
#dataset = Hyperpartisan(tokenizer_name="distilbert-base-uncased", max_seq_len=512, num_workers=16, cache_dir="./data", shuffle=True)

train_dataset, test_dataset = dataset.split()

# Define dataloaders
datacollator = DataCollatorWithPadding(tokenizer=dataset.tokenizer)

# Nota che questa roba funziona solo con batch_size=1
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=datacollator)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=datacollator)
# Initialize the model, loss function, and optimizer

num_epochs = 15
total_steps = len(train_loader) * num_epochs // 8   

model = LongNLPClassifier(config=config, num_classes=2, max_seq_len=512, stride=128)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
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
    log=False,
    max_epochs=num_epochs,
    use_mixed_precision=False,
    gradient_accumulation_steps=8, 
    warmup_steps=0.1*total_steps,
    project_name="Classification",
)

trainer.train(model, train_loader, test_loader)