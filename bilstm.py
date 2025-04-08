import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import io
from PIL import Image

torch.manual_seed(82)
np.random.seed(62)

HINDI_TRAIN = 'hindi_train.txt'
ENGLISH_TRAIN = 'english_train.txt'
HINDI_TEST = 'hindi_test.txt'
ENGLISH_TEST = 'english_test.txt'

def read_pos_data(file_path):
    sentences = []
    current_sentence = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:  # Ensure we have both word and tag
                    word = parts[0]
                    tag = parts[-1]  # Last element is the tag
                    current_sentence.append((word, tag))
            else:
                if current_sentence:  # End of a sentence
                    sentences.append(current_sentence)
                    current_sentence = []
        
        if current_sentence:
            sentences.append(current_sentence)
            
        return sentences
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Returning empty list.")
        return []

# Load datasets
print("Loading datasets...")
hindi_train = read_pos_data(HINDI_TRAIN)
english_train = read_pos_data(ENGLISH_TRAIN)
hindi_test = read_pos_data(HINDI_TEST)
english_test = read_pos_data(ENGLISH_TEST)

# Extract vocab and tag sets
def build_vocab_and_tags(data):
    word_counter = Counter()
    tag_counter = Counter()
    
    for sentence in data:
        for word, tag in sentence:
            word_counter[word] += 1
            tag_counter[tag] += 1
    
    return word_counter, tag_counter

# Create mappings for words and tags
def create_mappings(counter, min_freq=2, add_special=True):
    # Sort by frequency, then alphabetically for consistent ordering
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    item_to_idx = {}
    
    if add_special:
        item_to_idx['<PAD>'] = 0
        item_to_idx['<UNK>'] = 1
        start_idx = 2
    else:
        start_idx = 0
    
    for item, freq in sorted_items:
        if freq >= min_freq:
            item_to_idx[item] = len(item_to_idx)
    
    return item_to_idx

# Create separate vocabularies and tag sets for Hindi and English
hindi_word_counter, hindi_tag_counter = build_vocab_and_tags(hindi_train)
english_word_counter, english_tag_counter = build_vocab_and_tags(english_train)

hindi_word_to_idx = create_mappings(hindi_word_counter, min_freq=1)
hindi_tag_to_idx = create_mappings(hindi_tag_counter, min_freq=1, add_special=False)
hindi_idx_to_tag = {idx: tag for tag, idx in hindi_tag_to_idx.items()}

english_word_to_idx = create_mappings(english_word_counter, min_freq=1)
english_tag_to_idx = create_mappings(english_tag_counter, min_freq=1, add_special=False)
english_idx_to_tag = {idx: tag for tag, idx in english_tag_to_idx.items()}


# Create PyTorch Dataset
class POSDataset(Dataset):
    def __init__(self, data, word_to_idx, tag_to_idx):
        self.data = data
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.data[idx]
        words = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word, _ in sentence]
        tags = [self.tag_to_idx[tag] for _, tag in sentence]
        
        return torch.tensor(words), torch.tensor(tags), len(words)

# Collate function for padding sequences
def collate_fn(batch):
    words, tags, lengths = zip(*batch)
    words_padded = pad_sequence(words, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
    return words_padded, tags_padded, torch.tensor(lengths)

# Create separate DataLoaders for Hindi and English
hindi_train_dataset = POSDataset(hindi_train, hindi_word_to_idx, hindi_tag_to_idx)
hindi_test_dataset = POSDataset(hindi_test, hindi_word_to_idx, hindi_tag_to_idx)
english_train_dataset = POSDataset(english_train, english_word_to_idx, english_tag_to_idx)
english_test_dataset = POSDataset(english_test, english_word_to_idx, english_tag_to_idx)

hindi_train_loader = DataLoader(hindi_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
hindi_test_loader = DataLoader(hindi_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
english_train_loader = DataLoader(english_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
english_test_loader = DataLoader(english_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Define LSTM model for POS tagging
class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, num_layers=1, dropout=0.5):
        super(LSTMTagger, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)  # *2 for bidirectional
    
    def forward(self, sentence, lengths):
        embeds = self.embedding(sentence)
        
        # Pack padded batch of sequences for RNN
        packed = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(packed)
        
        # Unpack padding
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Get tag space scores
        tag_space = self.hidden2tag(lstm_out)
        
        return tag_space

# Model hyperparameters
EMBEDDING_DIM = 32 #originally 64
HIDDEN_DIM = 32 # reduced device constraints
NUM_LAYERS = 1 # reduced due to device constraints ; 2 for better results
DROPOUT = 0.3
NUM_EPOCHS = 5

# Initialize models for Hindi and English
hindi_model = LSTMTagger(len(hindi_word_to_idx), EMBEDDING_DIM, HIDDEN_DIM, len(hindi_tag_to_idx), NUM_LAYERS, DROPOUT)
english_model = LSTMTagger(len(english_word_to_idx), EMBEDDING_DIM, HIDDEN_DIM, len(english_tag_to_idx), NUM_LAYERS, DROPOUT)

hindi_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
english_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index

hindi_optimizer = optim.Adam(hindi_model.parameters(), lr=0.001)
english_optimizer = optim.Adam(english_model.parameters(), lr=0.001)

# Function to calculate accuracy
def calc_accuracy(outputs, tags, lengths):
    _, predicted = torch.max(outputs, dim=2)
    total_correct = 0
    total_tokens = 0
    
    for i, length in enumerate(lengths):
        total_correct += (predicted[i, :length] == tags[i, :length]).sum().item()
        total_tokens += length.item()
    
    return total_correct / total_tokens

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_acc = 0
        batches = 0
        
        for words, tags, lengths in train_loader:
            # Forward pass
            outputs = model(words, lengths)
            
            # Flatten outputs and tags for loss calculation
            batch_size, seq_len, tag_dim = outputs.size()
            outputs_flat = outputs.view(-1, tag_dim)
            tags_flat = tags.view(-1)
            
            # Calculate loss
            loss = criterion(outputs_flat, tags_flat)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            accuracy = calc_accuracy(outputs, tags, lengths)
            total_acc += accuracy
            batches += 1
        
        # Print epoch stats
        avg_loss = total_loss / batches
        avg_acc = total_acc / batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Training Accuracy: {avg_acc:.4f}')
    
    return model

# Evaluate model function
def evaluate_model(model, test_loader, idx_to_tag):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for words, tags, lengths in test_loader:
            outputs = model(words, lengths)
            _, predicted = torch.max(outputs, dim=2)
            
            # Collect predictions and targets (ignoring padding)
            for i, length in enumerate(lengths):
                all_predictions.extend(predicted[i, :length].tolist())
                all_targets.extend(tags[i, :length].tolist())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted')
    
    cm = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'micro': {
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        },
        'macro': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        },
        'weighted': {
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1': f1_weighted
        },
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets
    }

# Plot confusion matrix
def plot_confusion_matrix(cm, tags, title):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tags, yticklabels=tags)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {title} POS Tagging')
    plt.tight_layout()
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Return the figure
    return Image.open(buf)

hindi_model = train_model(hindi_model, hindi_train_loader, hindi_criterion, hindi_optimizer, num_epochs=NUM_EPOCHS)

hindi_eval_results = evaluate_model(hindi_model, hindi_test_loader, hindi_idx_to_tag)

english_model = train_model(english_model, english_train_loader, english_criterion, english_optimizer, num_epochs=NUM_EPOCHS)

english_eval_results = evaluate_model(english_model, english_test_loader, english_idx_to_tag)

# Print metrics for Hindi
print("\n===== Hindi Model Results =====")
print(f"Accuracy: {hindi_eval_results['accuracy']:.4f}")

print("\nMicro Average:")
print(f"Precision: {hindi_eval_results['micro']['precision']:.4f}")
print(f"Recall: {hindi_eval_results['micro']['recall']:.4f}")
print(f"F1 Score: {hindi_eval_results['micro']['f1']:.4f}")

print("\nMacro Average:")
print(f"Precision: {hindi_eval_results['macro']['precision']:.4f}")
print(f"Recall: {hindi_eval_results['macro']['recall']:.4f}")
print(f"F1 Score: {hindi_eval_results['macro']['f1']:.4f}")

print("\nWeighted Average:")
print(f"Precision: {hindi_eval_results['weighted']['precision']:.4f}")
print(f"Recall: {hindi_eval_results['weighted']['recall']:.4f}")
print(f"F1 Score: {hindi_eval_results['weighted']['f1']:.4f}")

# Print metrics for English
print("\n English Model Results")
print(f"Accuracy: {english_eval_results['accuracy']:.4f}")

print("\nMicro Average:")
print(f"Precision: {english_eval_results['micro']['precision']:.4f}")
print(f"Recall: {english_eval_results['micro']['recall']:.4f}")
print(f"F1 Score: {english_eval_results['micro']['f1']:.4f}")

print("\nMacro Average:")
print(f"Precision: {english_eval_results['macro']['precision']:.4f}")
print(f"Recall: {english_eval_results['macro']['recall']:.4f}")
print(f"F1 Score: {english_eval_results['macro']['f1']:.4f}")

print("\nWeighted Average:")
print(f"Precision: {english_eval_results['weighted']['precision']:.4f}")
print(f"Recall: {english_eval_results['weighted']['recall']:.4f}")
print(f"F1 Score: {english_eval_results['weighted']['f1']:.4f}")

# Generate and save confusion matrices
print("\nGenerating confusion matrices...")
hindi_tag_names = list(hindi_idx_to_tag.values())
english_tag_names = list(english_idx_to_tag.values())

hindi_cm_image = plot_confusion_matrix(hindi_eval_results['confusion_matrix'], hindi_tag_names, "Hindi")
hindi_cm_image.save('hindi_bilstm_confusion_matrix.png')

english_cm_image = plot_confusion_matrix(english_eval_results['confusion_matrix'], english_tag_names, "English")
english_cm_image.save('english_bilstm_confusion_matrix.png')

print("\nTraining and evaluation complete!")


