# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:40:27 2024

@author: gnkhata
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os 
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, RobertaTokenizer, RobertaModel
import numpy as np
from collections import Counter
from ReadInputPheme import read_data
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
EPOCHS = 100
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
    def __init__(self, embedding_dim, hidden_size, gpt_hidden_size, num_classes):
        super(BlendedRumorModel, self).__init__()
        self.embedding = nn.Embedding(len(STANCE_VOCAB), embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.fc_stance = nn.Linear(hidden_size * 2, 64)
        self.fc_post = nn.Linear(gpt_hidden_size, 64)

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
    report = classification_report(all_labels, all_preds, target_names=['True', 'False', 'Unverified'], labels=np.unique(all_labels))
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=np.unique(all_labels))
    return avg_loss, accuracy, macro_f1, report, conf_matrix

def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_val_macro_f1 = 0
    best_val_acc = 0
    best_epoch = 0
    
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

        val_loss, val_accuracy, val_macro_f1, val_report, val_conf_matrix = evaluate(model, val_loader, criterion)
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, Macro F1: {val_macro_f1:.4f}")
        print(f'Val Classification report:\n {val_report}')
        print("Confusion matrix:\n", val_conf_matrix)
        print()
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch

    print(f'Best val macro f1:\n {best_val_macro_f1}, Best val acc: {best_val_acc} Best Epoch: {best_epoch}')
    return best_val_macro_f1, best_val_acc 
    #return best_model



if __name__ == "__main__":
    start_time = datetime.now()
    # Define hyperparameters and settings
    
    fold = 0
    
    macro_f1s = list()
    accs = list()
    
        
    # Initialize an empty dictionary
    data_dict = {}
        
    main_dir = "./PHEME_veracity/all-rnr-annotated-threads/"
    dirs = [
        "charliehebdo-all-rnr-threads", "ebola-essien-all-rnr-threads", 
        "ferguson-all-rnr-threads", "germanwings-crash-all-rnr-threads", 
        "gurlitt-all-rnr-threads", "ottawashooting-all-rnr-threads",
        "prince-toronto-all-rnr-threads", "putinmissing-all-rnr-threads", 
        "sydneysiege-all-rnr-threads"
    ]

    for topic in dirs:
        dir_path = os.path.join(main_dir, topic)
        topic_data = read_data(dir_path, model_type="BiLSTM_LLM")
        data_dict[topic] = topic_data
        
    # Print the dictionary to check
    for test_topic in dirs:
        fold += 1
        train_posts, train_stances, train_labels = list(), list(), list()
        val_posts, val_stances, val_labels = list(), list(), list()
        
        for topic in dirs:
            src_posts, rep_stances, labels = data_dict[topic]
            if test_topic == topic:
                val_posts.extend(src_posts)
                val_stances.extend(rep_stances)
                val_labels.extend(labels)
            else:
                train_posts.extend(src_posts)
                train_stances.extend(rep_stances)
                train_labels.extend(labels)

        '''
        print(f" Data statistics for {test_topic}:\n Train data: {len(train_texts)}, {len(train_labels)}\n  Test data: {len(val_texts)}, {len(val_labels)}")
        print()
        '''
        
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
               
        # Train the model
        print(f" Training fold {fold}, Test topic: {test_topic}\n")
        best_val_macro_f1, best_val_acc = train(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
        macro_f1s.append(best_val_macro_f1)
        accs.append(best_val_acc)
        print(f"Fold {fold} Test Macro f1s: {macro_f1s}\n Test Accuracies: {accs}")
        print(f"Fold {fold} Average Test Macro f1 {sum(macro_f1s)/len(macro_f1s)}\n Average test accuracy: {sum(accs)/len(accs)}")
        print()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Running time: {elapsed_time}")
    