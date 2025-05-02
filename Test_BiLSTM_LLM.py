# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:36:06 2024

@author: gnkhata
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from ReadInputStancDistr import read_data
from BiLSTM_LLM import BlendedDataset, collate_fn, BlendedRumorModel
from Text_preprocessing import process_texts2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
EMBEDDING_DIM = 32
HIDDEN_SIZE = 64
LLM_HIDDEN_SIZE = 768  
NUM_CLASSES = 3
STANCE_VOCAB = {'support': 0, 'deny': 1, 'query': 2, 'comment': 3}
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 17



def load_model(model_path):
    model = BlendedRumorModel(EMBEDDING_DIM, HIDDEN_SIZE, LLM_HIDDEN_SIZE, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
 
def evaluate(model, data_loader, criterion):
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
    report = classification_report(all_labels, all_preds, target_names=['rumor', 'not-rumor', 'unverified'])
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, macro_f1, report, conf_matrix

if __name__ == "__main__":
    # Load the test data
    test_posts, test_labels, test_stances = read_data(flag="test", data_mode="src_twt_all_replies", model="combined")
    test_posts = process_texts2(test_posts)
    test_stances = [[STANCE_VOCAB[stance] for stance in seq] for seq in test_stances]
    test_dataset = BlendedDataset(test_posts, test_stances, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    # Load the saved model
    model_path = './Best_Model/BiLSTM/bert/batch16_embDim32_best_model_bfs4.pth'
    model = load_model(model_path)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model on the test set
    test_loss, test_accuracy, test_macro_f1, test_report, test_conf_matrix = evaluate(model, test_loader, criterion)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print("Test Classification Report:\n", test_report)
    print("Confusion Matrix:\n", test_conf_matrix)
