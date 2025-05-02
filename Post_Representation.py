# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:24:00 2024

@author: gnkhata
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from collections import Counter
import numpy as np
from ReadInputStancDistr import read_data
from Text_preprocessing import process_texts2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
MAX_SEQ_LENGTH = 512
STANCE_LABELS = ['support', 'deny', 'query', 'comment']
NUM_CLASSES = 3  # rumor, not-rumor, unverified
HIDDEN_SIZE = 256

# Load BERT model and tokenizer
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name).to(device)

def compute_class_weights(labels, num_classes):
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())
    return torch.tensor(
        [total_count / label_counts.get(i, 1) for i in range(num_classes)],
        dtype=torch.float
    ).to(device)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=MAX_SEQ_LENGTH).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

class RumorDataset(Dataset):
    def __init__(self, posts_with_replies, stances_list, stance_distributions, structures, labels):
        self.posts_with_replies = posts_with_replies
        self.stances_list = stances_list
        self.stance_distributions = stance_distributions
        self.structures = structures
        self.labels = labels

    def __len__(self):
        return len(self.posts_with_replies)

    def __getitem__(self, idx):
        source_post = self.posts_with_replies[idx][0]
        replies = self.posts_with_replies[idx][1:]
        stances = self.stances_list[idx][1:]  # Skip source post stance
        #stance_distribution = self.stance_distributions[idx]
        #structure_info = torch.tensor(self.structures[idx], dtype=torch.float).to(device)
    
        # Compute source post embedding
        source_post_embedding = get_bert_embedding(source_post)
    
        # Aggregate reply embeddings by stance
        stance_embeddings = {label: [] for label in STANCE_LABELS}
        for reply, stance in zip(replies, stances):
            reply_embedding = get_bert_embedding(reply)
            stance_embeddings[stance].append(reply_embedding)
    
        aggregated_stance_embeddings = []
        for label in STANCE_LABELS:
            if stance_embeddings[label]:
                aggregated_stance_embeddings.append(torch.mean(torch.stack(stance_embeddings[label]), dim=0))
            else:
                aggregated_stance_embeddings.append(torch.zeros_like(source_post_embedding))
    
        # Normalize stance distribution and ensure 2D
        '''
        stance_distribution_tensor = torch.tensor(
            [stance_distribution.get(label, 0) for label in STANCE_LABELS], 
            dtype=torch.float
        ).unsqueeze(0).to(device)
    
        # Ensure structure info is 2D
        structure_info = structure_info.unsqueeze(0)
        '''
        # Combine all features
        combined_features = torch.cat([source_post_embedding] + aggregated_stance_embeddings, dim=1)
    
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return combined_features, label


class RumorClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(RumorClassifier, self).__init__()


        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Check the shape of x
        #print(f"Shape of x before attention: {x.shape}")
        
        # Final fully connected layer and softmax
        return self.softmax(self.fc(x))


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')

def train_model(model, train_loader, val_loader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_class_weights = compute_class_weights(train_labels, 3)
    criterion = nn.CrossEntropyLoss(weight=train_class_weights)
    #criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        all_preds, all_labels = [], []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate training metrics
        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples
        train_macro_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Macro F1: {train_macro_f1:.4f}")

        # Validation phase
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_samples
        val_macro_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        val_classification_report = classification_report(all_val_labels, all_val_preds, target_names=['Rumor', 'Not Rumor', 'Unverified'])
        val_confusion_matrix = confusion_matrix(all_val_labels, all_val_preds)

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Macro F1: {val_macro_f1:.4f}")
        print("Validation Classification Report:\n", val_classification_report)
        print("Validation Confusion Matrix:\n", val_confusion_matrix)

        # Save the best model based on validation macro F1
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_model_state = model.state_dict()
            print("New best model found and saved!")

    # Save the best model to disk
    if best_model_state is not None:
        save_path = "./best_model.pth"
        torch.save(best_model_state, save_path)
        print(f"Best model saved with Macro F1: {best_f1:.4f} at {save_path}")

    print("Training complete.")
if __name__ == "__main__":
    # Example data loading and preprocessing
    
    # Load the data (replace this with your actual data loading logic)
    train_posts, val_posts, train_posts_stances, val_posts_stances, train_stance_distr, val_stance_distr, train_structures, val_structures, train_labels, val_labels = read_data(
        flag="train", 
        data_mode="src_twt_all_replies", 
        model="agg_stanc_struct"
    )
    #print(len(train_structures[3]))
    # Process the text (if needed)
    train_posts = [process_texts2(thread) for thread in train_posts]
    val_posts = [process_texts2(thread) for thread in val_posts]

    # Dataset and DataLoader initialization
    train_dataset = RumorDataset(
        posts_with_replies=train_posts,
        stances_list=train_posts_stances,
        #stance_distributions=train_stance_distr,
        #structures=train_structures,
        labels=train_labels
    )
    val_dataset = RumorDataset(
        posts_with_replies=val_posts,
        stances_list=val_posts_stances,
        #stance_distributions=val_stance_distr,
        #structures=val_structures,
        labels=val_labels
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model initialization
    input_dim = 768 * 5 #+ 4 + len(train_structures[0])  # BERT embeddings + 4 stance distributions + structure info
    model = RumorClassifier(
        input_dim=input_dim,
        hidden_dim=256,
        num_classes=NUM_CLASSES
    ).to(device)

    # Training the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        learning_rate=3e-4
    )

