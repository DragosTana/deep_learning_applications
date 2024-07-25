import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding
from trainer import Trainer
from torch.optim.lr_scheduler import PolynomialLR   
from data import IMDB, Hyperpartisan
from laboratory_2.distil_bert import DistilBERTModel


class NLPClassifier(nn.Module):
    def __init__(self, transformer_model_name, num_classes):
        super(NLPClassifier, self).__init__()
        
        model_for_weights = AutoModel.from_pretrained("distilbert-base-uncased")
        config = model_for_weights.config 
        self.transformer = DistilBERTModel(config)
        self.transformer.load_state_dict(model_for_weights.state_dict())
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask=None):
        hidden_state = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits
    
    def train_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, self.num_classes), labels.view(-1))
        return loss, outputs
    
    def test_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, self.num_classes), labels.view(-1))
        return loss, outputs

# Initialize the dataset
#dataset = IMDB(tokenizer_name="distilbert-base-uncased", max_seq_len=512, num_workers=16, cache_dir="./data", shuffle=True)
dataset = Hyperpartisan(tokenizer_name="distilbert-base-uncased", max_seq_len=512, num_workers=16, cache_dir="./data", shuffle=True)

train_dataset, test_dataset = dataset.split()

# Define dataloaders
datacollator = DataCollatorWithPadding(tokenizer=dataset.tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=datacollator)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=datacollator)

# Initialize the model, loss function, and optimizer
num_epochs = 15
total_steps = num_epochs * len(train_loader) 

model = NLPClassifier(transformer_model_name="distilbert-base-uncased", num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
scheduler = PolynomialLR(optimizer, total_steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
 
# Se va troppo piano implementate mixed precision
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
    log=True,
    max_epochs=num_epochs,
    use_mixed_precision=True,
    gradient_accumulation_steps=1, 
    warmup_steps=0.1*total_steps,
    project_name="Classification",
)

trainer.train(model, train_loader, test_loader)