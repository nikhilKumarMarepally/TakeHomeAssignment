from transformers import BertModel, BertTokenizer
import torch
from datasets import Dataset
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from transformers import BertModel
from datasets import Dataset

#Dataset we use for training
data = {
    "sentence": [
        "Machine learning is fascinating.",
        "Deep learning is a subset of machine learning.",
        "Transformers are powerful models for NLP.",
        "Social media is fake.",
        "AI is revolutionizing healthcare.",
        "Self-driving cars are the future.",
        "AI is impacting almost every industry.",
        "Machine learning algorithms are improving.",
        "Artificial intelligence helps in decision making.",
        "Social media addiction is harmful.",
        "The internet is full of misinformation.",
        "The future of technology is AI.",
        "Natural language processing is transforming communication.",
        "AI-driven tools are changing the way we work.",
        "Automation through AI can help increase productivity.",
        "Machine learning can help in analyzing big data.",
        "AI can predict stock market trends.",
        "Neural networks can be used in voice recognition.",
        "Robots powered by AI are becoming common.",
        "The AI market is growing rapidly.",
        "Deep learning is a key part of modern AI.",
        "Machine learning can solve complex problems.",
        "AI is a driving force for innovation.",
        "Social media platforms are increasing their influence.",
        "Machine learning models can be used for fraud detection.",
        "AI is used in facial recognition technologies.",
        "The development of AI is accelerating.",
        "AI can help in personalized healthcare.",
        "Social media platforms are sources of fake news."
    ],
    "task_a_label": [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    "task_b_labels": [1, 0, 2, 2, 0, 0, 1, 0, 2, 0, 1, 2, 1, 0, 2, 1, 0, 1, 2, 1, 0, 0, 1, 0, 2, 2, 0, 2, 1]
}

'''
task_a_label, is sentence classification (1 if the sentence is related to ML, else 0)
tasl_b_label, is sentiment_classification (0:bad, 1:neutral, 2:good)
Model: I will be using the same Bert model, for 2 different tasks (sentiment_analysis and sentence classifier)
'''

class MultiTaskLearningTransformer(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_sentence_classes=3, num_sentiment_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.sentence_classifier = torch.nn.Linear(768, num_sentence_classes)
        self.sentiment_analysis = torch.nn.Linear(768, num_sentiment_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :]
        return self.sentence_classifier(sentence_embeddings), self.sentiment_analysis(outputs.last_hidden_state)

'''
The DatasetWrapper class is a custom dataset wrapper that converts a given dataset to a format compatible with PyTorch's Dataset class. 
'''

class DatasetWrapper(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.columns = dataset.column_names

    def __getitem__(self, idx):
        item = {key: torch.tensor(self.dataset[key][idx]) if not isinstance(self.dataset[key][idx], str) else self.dataset[key][idx] for key in self.columns}
        return item

    def __len__(self):
        return len(self.dataset)

'''
Function for tokenization,Special tokens and subword splits are assigned -100 to ignore them during loss computation.
'''
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, padding=True, is_split_into_words=False)
    labels_task_b = []
    for i, sentence in enumerate(examples["sentence"]):
        task_b_label = examples["task_b_labels"][i]
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_idx is None or word_idx == previous_word_idx else task_b_label for previous_word_idx, word_idx in zip([None] + word_ids[:-1], word_ids)]
        labels_task_b.append(label_ids)
    tokenized_inputs["task_a_label"] = examples["task_a_label"]
    tokenized_inputs["task_b_labels"] = labels_task_b
    return tokenized_inputs

# Load tokenizer and dataset
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
dataset = Dataset.from_dict(data)

'''
Train test split and generating train_loader, test_loader
'''

train_data, test_data = train_test_split(dataset, test_size=0.2)
train_dataset, test_dataset = Dataset.from_dict(train_data), Dataset.from_dict(test_data)
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

train_dataloader = DataLoader(DatasetWrapper(train_dataset), batch_size=2, shuffle=True)
test_dataloader = DataLoader(DatasetWrapper(test_dataset), batch_size=2, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskLearningTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
classification_loss_fn = torch.nn.CrossEntropyLoss()
ner_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
num_epochs = 5


'''
    Training loop
'''
def train_model(model, train_dataloader, optimizer, classification_loss_fn, ner_loss_fn, num_epochs=3, device='cuda'):
    for epoch in range(num_epochs):
        model.train()
        total_classification_loss, total_ner_loss = 0, 0
        correct_classifications, total_classifications = 0, 0
        all_ner_preds, all_ner_labels = [], []
        
        for batch in train_dataloader:
            input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            task_a_labels, task_b_labels = batch["task_a_label"].to(device), batch["task_b_labels"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_logits, ner_logits = model(input_ids, attention_mask)
            
            # Computing losses
            class_loss = classification_loss_fn(class_logits, task_a_labels)
            ner_loss = ner_loss_fn(ner_logits.view(-1, 3), task_b_labels.view(-1))
            total_loss = class_loss + ner_loss
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            
            total_classification_loss += class_loss.item()
            total_ner_loss += ner_loss.item()
            correct_classifications += (torch.argmax(class_logits, dim=-1) == task_a_labels).sum().item()
            total_classifications += task_a_labels.size(0)
            
            valid_indices = task_b_labels.view(-1) != -100
            all_ner_preds.extend(torch.argmax(ner_logits, dim=-1).view(-1)[valid_indices].cpu().numpy())
            all_ner_labels.extend(task_b_labels.view(-1)[valid_indices].cpu().numpy())
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Classification Loss: {total_classification_loss / len(train_dataloader):.4f}, NER Loss: {total_ner_loss / len(train_dataloader):.4f}")
        print(f"Classification Accuracy: {correct_classifications / total_classifications:.4f}")
        print(f"NER F1-Score: {f1_score(all_ner_labels, all_ner_preds, average='macro'):.4f}")
    
    return model

def evaluate_model(model, eval_dataloader, classification_loss_fn, ner_loss_fn, device='cuda'):
    model.eval()
    total_classification_loss, total_ner_loss = 0, 0
    correct_classifications, total_classifications = 0, 0
    all_ner_preds, all_ner_labels = [], []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            task_a_labels, task_b_labels = batch["task_a_label"].to(device), batch["task_b_labels"].to(device)
            
            # Forward pass
            class_logits, ner_logits = model(input_ids, attention_mask)
            
            # Compute losses
            class_loss = classification_loss_fn(class_logits, task_a_labels)
            ner_loss = ner_loss_fn(ner_logits.view(-1, 3), task_b_labels.view(-1))
            
            total_classification_loss += class_loss.item()
            total_ner_loss += ner_loss.item()
            correct_classifications += (torch.argmax(class_logits, dim=-1) == task_a_labels).sum().item()
            total_classifications += task_a_labels.size(0)
            
            valid_indices = task_b_labels.view(-1) != -100
            all_ner_preds.extend(torch.argmax(ner_logits, dim=-1).view(-1)[valid_indices].cpu().numpy())
            all_ner_labels.extend(task_b_labels.view(-1)[valid_indices].cpu().numpy())
    
    print(f"Evaluation Classification Loss: {total_classification_loss / len(eval_dataloader):.4f}")
    print(f"Evaluation NER Loss: {total_ner_loss / len(eval_dataloader):.4f}")
    print(f"Evaluation Classification Accuracy: {correct_classifications / total_classifications:.4f}")
    print(f"Evaluation NER F1-Score: {f1_score(all_ner_labels, all_ner_preds, average='macro'):.4f}")

print("-----------------Training & Evaluation----------------")
model = train_model(model, train_dataloader, optimizer, classification_loss_fn, ner_loss_fn, num_epochs=num_epochs, device=device)
evaluate_model(model, test_dataloader, classification_loss_fn, ner_loss_fn, device=device)



