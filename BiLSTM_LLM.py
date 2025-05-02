# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:40:27 2024

@author: gnkhata
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, RobertaTokenizer, RobertaModel
import numpy as np
from collections import Counter
from ReadInputStancDistr import read_data, remove_key_from_dicts
from Text_preprocessing import process_texts2
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
EMBEDDING_DIM = 32
HIDDEN_SIZE = 64
LLM_HIDDEN_SIZE = 768  # Hidden size for GPT embeddings
NUM_CLASSES = 3
STANCE_VOCAB = {'support': 0, 'deny': 1, 'query': 2, 'comment': 3}
BATCH_SIZE = 16
EPOCHS = 25
MAX_SEQ_LENGTH = 17

# Initialize LLM
bert_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model)
lang_model = BertModel.from_pretrained(bert_model).to(device)

'''
# Load RoBERTa model and tokenizer
roberta_model = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
lang_model = RobertaModel.from_pretrained(roberta_model).to(device)
'''
'''
# Load GPT model and tokenizer
gpt2_model = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
gpt_model = GPT2Model.from_pretrained(gpt2_model).to(device)
tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding token
gpt_model.config.pad_token_id = gpt_model.config.eos_token_id
lang_model = gpt_model.to(device)
'''
def compute_class_weights(labels, num_classes):
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())
    class_weights = [total_count / label_counts[i] if i in label_counts else 0 for i in range(num_classes)]
    return torch.tensor(class_weights, dtype=torch.float).to(device)

class BlendedDataset(Dataset):
    def __init__(self, source_posts, stance_sequences, labels):
        valid_data = [(src_pst, stnc_seq, lbl) for src_pst, stnc_seq, lbl in zip(source_posts, stance_sequences, labels) if len(stnc_seq) > 0]
        self.source_posts, self.stance_sequences, self.labels = zip(*valid_data) if valid_data else ([], [], [])
        
    def __len__(self):
        return len(self.source_posts)

    def __getitem__(self, idx):
        source_post = self.source_posts[idx]
        stance_seq = self.stance_sequences[idx]
        label = self.labels[idx]

        # Process the source post using BERT tokenizer
        inputs = tokenizer(source_post, padding='max_length', truncation=True, 
                           max_length=MAX_SEQ_LENGTH, return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = lang_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            post_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # Mean pooling

        return post_embedding, torch.tensor(stance_seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    post_embeddings, stance_sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in stance_sequences])
    padded_sequences = nn.utils.rnn.pad_sequence(stance_sequences, batch_first=True, padding_value=0)

    return (
        torch.stack(post_embeddings).to(device),
        padded_sequences.to(device),
        torch.stack(labels).to(device),
        lengths.to(device),
    )

class BlendedRumorModel(nn.Module):
    def __init__(self, embedding_dim, hidden_size, llm_hidden_size, num_classes):
        super(BlendedRumorModel, self).__init__()
        self.embedding = nn.Embedding(len(STANCE_VOCAB), embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.fc_stance = nn.Linear(hidden_size * 2, 64)
        self.fc_post = nn.Linear(llm_hidden_size, 64)

        self.combined_fc = nn.Linear(64 + 64, 128)
        self.output_fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, post_embedding, stance_sequence, lengths):
        # Process stance sequence with LSTM
        embedded_stance = self.embedding(stance_sequence)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded_stance, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        stance_embedding = output[range(len(output)), lengths - 1]
        stance_features = self.fc_stance(stance_embedding)

        # Process post embedding
        post_features = self.fc_post(post_embedding)

        # Combine stance and post features
        combined = torch.cat((stance_features, post_features), dim=1)
        combined = self.dropout(self.relu(self.combined_fc(combined)))

        # Output layer
        logits = self.output_fc(combined)
        return logits

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for post_embedding, stance_sequence, labels, lengths in data_loader:
            outputs = model(post_embedding, stance_sequence, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=['True', 'False', 'Unverified'])

    return avg_loss, accuracy, macro_f1, report

def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_macro_f1 = 0.0
    best_model = None
    best_epoch = 0
    save_model_path = './Best_Model/BiLSTM/bert/batch'+str(BATCH_SIZE)+'_embDim'+str(EMBEDDING_DIM)+'_best_model_dfs3.pth'
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for post_embedding, stance_sequence, labels, lengths in train_loader:
            optimizer.zero_grad()
            outputs = model(post_embedding, stance_sequence, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_macro_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Train Macro F1: {train_macro_f1:.4f}")

        val_loss, val_accuracy, val_macro_f1, val_report = evaluate(model, val_loader, criterion)
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, Macro F1: {val_macro_f1:.4f}")
        print(f'Val Classification report:\n {val_report}')

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            print("New best model found and saved.")

    #torch.save(best_model.state_dict(), save_model_path)
    torch.save(copy.deepcopy(model).state_dict(), save_model_path)
    print(f"Training complete. Best model saved with Macro F1: {best_macro_f1:.4f}")
    print(f"Best model saved to: {save_model_path}, Best epoch: {best_epoch}")
    #return best_model

if __name__ == "__main__":
    # Load data
    train_posts, val_posts, train_stances, val_stances, train_labels, val_labels = read_data(
        flag="train", data_mode="src_twt_all_replies", model="combined"
    )
    
    #Replace @user menations with $mention$ and urls with $URL$ and normalize #hashtags 
    train_posts = process_texts2(train_posts)
    val_posts = process_texts2(val_posts)
    
    train_stances = [[STANCE_VOCAB[stance] for stance in seq] for seq in train_stances]
    val_stances = [[STANCE_VOCAB[stance] for stance in seq] for seq in val_stances]

    train_dataset = BlendedDataset(train_posts, train_stances, train_labels)
    val_dataset = BlendedDataset(val_posts, val_stances, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = BlendedRumorModel(EMBEDDING_DIM, HIDDEN_SIZE, LLM_HIDDEN_SIZE, NUM_CLASSES).to(device)
    #train_class_weights = compute_class_weights(train_labels, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss(weight=train_class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_model = train(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
