import torch
import torch.nn as nn
import numpy as np
import copy
from datetime import datetime
from collections import Counter
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel,GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from ReadInputStancDistr import read_stanc_data
from Utilities import compute_class_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_embedding_oversampling(train_loader, model):
    embeddings = []
    labels = []

    # Extract embeddings and labels
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].cpu().numpy()  # Move labels to CPU
            
            # Get embeddings from the model
            outputs = model.l_model(input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output.cpu().numpy()  # Move embeddings to CPU
            
            embeddings.append(pooled_output)
            labels.extend(label)

    # Concatenate embeddings and labels
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)

    # Apply SMOTE for oversampling
    smote = SMOTE(random_state=42)
    augmented_embeddings, augmented_labels = smote.fit_resample(embeddings, labels)

    # Optionally add Gaussian noise to augment data
    noise = np.random.normal(0, 0.01, augmented_embeddings.shape)
    augmented_embeddings += noise

    # Convert back to tensors
    augmented_embeddings = torch.tensor(augmented_embeddings, dtype=torch.float32).to(device)
    augmented_labels = torch.tensor(augmented_labels, dtype=torch.long).to(device)

    return augmented_embeddings, augmented_labels

def custom_collate_fn(batch):
    """
    Custom collate function to process batches from the RumorDataset.
    """
    # Separate batch into individual components
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Handle token_type_ids only if present
    if 'token_type_ids' in batch[0]:
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    else:
        token_type_ids = None

    labels = torch.stack([item['label'] for item in batch])
    ids = [item['id'] for item in batch]  # Keep IDs as a list of strings

    # Return the batch as a dictionary
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "label": labels,
        "id": ids
    }


# Define the BERT-based model for multiclass classification
class RumorVerificationModel(nn.Module):
    def __init__(self, lang_model, num_classes):
        super(RumorVerificationModel, self).__init__()
        self.l_model = lang_model
        self.fc = nn.Linear(self.l_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.l_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.fc(pooled_output)
        x = self.dropout(x)
        logits = self.softmax(x)
        return logits

# Define custom dataset for multiclass
class RumorDataset(Dataset):
    def __init__(self, ids,  texts, labels, tokenizer, max_length, l_model, seq_mode, lang_model):
        self.ids = ids
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.l_model = l_model
        self.seq_mode = seq_mode
        self.lang_model = lang_model
        if self.seq_mode == "concat":
            print("Initialising Input seq in concat mode /n")
            print()
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        id_ = self.ids[idx] 
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.seq_mode == "one":
            if self.l_model == "gpt2":
                encoding = self.tokenizer.encode_plus(
                    text,
                    None,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    pad_to_max_length=True,
                    return_token_type_ids=True,
                    truncation=True
                )
    
                input_ids = encoding["input_ids"]
                attention_mask = encoding["attention_mask"]
                token_type_ids = encoding["token_type_ids"]
    
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                    "label": torch.tensor(label, dtype=torch.long),
                    'id':id_
                }
            else:
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    return_token_type_ids=False,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'label': torch.tensor(label, dtype=torch.long),
                    'id':id_
                }
            

# Function to train the model for multiclass
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, LR_RATE, BATCH_SIZE, L_MODEL, DATA_MODE, SEQ_MODE):
    target_names = ['deny 0', 'query 1', 'comment 2', 'support 3']
    best_model_path = './Best_Model/' + DATA_MODE + '/' + L_MODEL + '/1best_model1.pth'
    best_val_macro_f1 = float('-inf')
    best_val_acc = float('-inf')
    best_model_state = None
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0
        train_predictions = []
        train_targets = []

        # Apply embedding-level oversampling
        augmented_embeddings, augmented_labels = apply_embedding_oversampling(train_loader, model)

        # Create a new DataLoader for the oversampled data
        oversampled_dataset = torch.utils.data.TensorDataset(augmented_embeddings, augmented_labels)
        oversampled_loader = DataLoader(oversampled_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Train on the oversampled data
        for inputs, labels in oversampled_loader:
            optimizer.zero_grad()

            outputs = model.fc(inputs)  # Pass embeddings directly to the classifier
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted_labels = torch.max(outputs, dim=1)
            total_train_correct += (predicted_labels == labels).sum().item()
            total_train_samples += labels.size(0)
            train_predictions.extend(predicted_labels.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_accuracy = total_train_correct / total_train_samples
        train_loss = train_loss / len(oversampled_loader)
        train_f1 = f1_score(train_targets, train_predictions, average='macro')

        # Validation
        model.eval()
        val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted_labels = torch.max(outputs, dim=1)
                total_val_correct += (predicted_labels == labels).sum().item()
                total_val_samples += labels.size(0)
                val_predictions.extend(predicted_labels.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_accuracy = total_val_correct / total_val_samples
        val_loss = val_loss / len(val_loader)
        val_macro_f1 = f1_score(val_targets, val_predictions, average='macro')

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_acc =  val_accuracy
            best_model_state = copy.deepcopy(model)
            best_epoch = epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Train F1: {train_f1:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Macro F1: {val_macro_f1:.4f}')
        print(classification_report(val_targets, val_predictions, target_names=target_names, labels=np.unique(val_targets)))
        print(confusion_matrix(val_targets, val_predictions, labels=np.unique(val_targets)))
        print()

    torch.save(best_model_state.state_dict(), best_model_path)
    print(f"Training ended, best model stored in: {best_model_path}, Best epoch: {best_epoch}, Best Val Macro f1: {best_val_macro_f1}, Best Val Acc: {best_val_acc} ")
 
if __name__ == "__main__": 
    start_time = datetime.now()       
    # Define hyperparameters and settings
    L_MODEL = 'bert'
    BATCH_SIZE = 16
    LR_RATE = 0.001
    NUM_EPOCHS = 500
    DATA_MODE = 'Stanc_Det'
    SEQ_MODE  = 'one'
    MAX_LENGTH = 128
    NUM_CLASSES = 4  # Change this based on the number of classes in your dataset
    

    # Initialize the model, criterion, and optimizer
    if L_MODEL == "bert":
        print("Training BERT model:")
        # Load  tokenizer and bert model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        lang_model = BertModel.from_pretrained('bert-base-uncased')
        model = RumorVerificationModel(lang_model, NUM_CLASSES).to(device)
    
    elif L_MODEL == "roberta":
        # Load pre-trained RoBERTa model and tokenizer
        print("Training RoBERTa model:")
        lang_model = RobertaModel.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RumorVerificationModel(lang_model, NUM_CLASSES).to(device)
    elif L_MODEL == "gpt2":
        # Load pre-trained gpt2 model and tokenizer
        print("Training gpt2 model:")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        lang_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
        tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding token
        lang_model.config.pad_token_id = lang_model.config.eos_token_id
        model = lang_model.to(device)
        
    
    # Assuming you have loaded your dataset into texts and labels
    train_ids, train_posts, train_labels, val_ids, val_posts, val_labels  = read_stanc_data()
    counter = Counter(train_labels)
    print(counter)
    counter = Counter(val_labels)
    print(counter)
    train_dataset = RumorDataset(train_ids, train_posts, train_labels, tokenizer, MAX_LENGTH,L_MODEL,SEQ_MODE,lang_model)
    val_dataset = RumorDataset(val_ids, val_posts, val_labels, tokenizer, MAX_LENGTH,L_MODEL,SEQ_MODE,lang_model)
    
    train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=custom_collate_fn  # Pass the custom collate function
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn  # Pass the custom collate function
    )
    
    #train_class_weights = compute_class_weights(train_labels, NUM_CLASSES)
    #criterion = nn.CrossEntropyLoss(weight=train_class_weights)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

    
    # Train the model

    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS,LR_RATE, BATCH_SIZE,L_MODEL,DATA_MODE, SEQ_MODE)
    end_time = datetime.now()
    elapsed_time = end_time-start_time
    print(f"Execution time: {elapsed_time}")
    
    '''
    # Get predicted class indices
    predicted_labels = torch.argmax(predicted_probs, dim=1)
    
    # Compute F1 score for each class
    f1_scores = f1_score(true_labels, predicted_labels, average=None)
    
    # Compute macro-average F1 score
    macro_f1 = torch.tensor(f1_scores).mean().item()
    '''
