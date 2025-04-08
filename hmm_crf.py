import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

class HMMTagger:
    """
    Hidden Markov Model implementation for POS tagging from scratch
    """
    def __init__(self):
        self.states = []  # All possible POS tags
        self.observations = []  # All possible words
        self.start_p = {}  # Initial probabilities
        self.trans_p = {}  # Transition probabilities
        self.emit_p = {}  # Emission probabilities
        self.smoothing = 1  # Smoothing factor for unseen combinations
    
    def train(self, sequences):
        """
        Train the HMM model on sequences of (word, tag) pairs
        
        Args:
            sequences: List of lists, where each inner list contains (word, tag) tuples
        """
        # Collect unique states (tags) and observations (words)
        all_tags = set()
        all_words = set()
        for sequence in sequences:
            for word, tag in sequence:
                all_tags.add(tag)
                all_words.add(word)
        
        self.states = list(all_tags)
        self.observations = list(all_words)
        
        # Initialize counters
        tag_counts = Counter()
        start_counts = Counter()
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)
        
        # Collect counts from training data
        for sequence in sequences:
            if sequence:  # Check if sequence is not empty
                # Count initial tags
                _, first_tag = sequence[0]
                start_counts[first_tag] += 1
                
                # Count tags and emissions
                for i, (word, tag) in enumerate(sequence):
                    tag_counts[tag] += 1
                    emission_counts[tag][word] += 1
                    
                    # Count transitions (tag -> next tag)
                    if i < len(sequence) - 1:
                        _, next_tag = sequence[i + 1]
                        transition_counts[tag][next_tag] += 1
        
        # Calculate probabilities with smoothing
        
        # Initial probabilities
        total_sequences = len(sequences)
        self.start_p = {tag: (count + self.smoothing) / (total_sequences + self.smoothing * len(self.states)) 
                        for tag, count in start_counts.items()}
        
        # Add smoothing for states not seen at start
        for tag in self.states:
            if tag not in self.start_p:
                self.start_p[tag] = self.smoothing / (total_sequences + self.smoothing * len(self.states))
        
        # Transition probabilities
        self.trans_p = {}
        for tag1 in self.states:
            self.trans_p[tag1] = {}
            total = sum(transition_counts[tag1].values()) + self.smoothing * len(self.states)
            for tag2 in self.states:
                self.trans_p[tag1][tag2] = (transition_counts[tag1][tag2] + self.smoothing) / total
        
        # Emission probabilities
        self.emit_p = {}
        for tag in self.states:
            self.emit_p[tag] = {}
            total = sum(emission_counts[tag].values()) + self.smoothing * len(self.observations)
            for word in self.observations:
                self.emit_p[tag][word] = (emission_counts[tag][word] + self.smoothing) / total
    
    def viterbi(self, observation_sequence):
        """
        Viterbi algorithm to find the most likely sequence of tags
        
        Args:
            observation_sequence: List of words
            
        Returns:
            List of predicted tags
        """
        V = [{}]  # Viterbi matrix
        path = {}  # Best path
        
        # Initialize base cases (first observation)
        for state in self.states:
            # Check if the word is in the vocabulary
            word = observation_sequence[0]
            emit_p = self.emit_p[state].get(word, self.smoothing)
            
            V[0][state] = self.start_p[state] * emit_p
            path[state] = [state]
        
        # Run Viterbi algorithm for t > 0
        for t in range(1, len(observation_sequence)):
            V.append({})
            new_path = {}
            
            word = observation_sequence[t]
            
            for curr_state in self.states:
                # Handle unknown words
                if word not in self.observations:
                    emit_p = self.smoothing
                else:
                    emit_p = self.emit_p[curr_state].get(word, self.smoothing)
                
                max_prob = -1
                max_state = None
                
                for prev_state in self.states:
                    prob = V[t-1][prev_state] * self.trans_p[prev_state][curr_state] * emit_p
                    
                    if prob > max_prob:
                        max_prob = prob
                        max_state = prev_state
                
                V[t][curr_state] = max_prob
                new_path[curr_state] = path[max_state] + [curr_state]
            
            # Don't need to remember the old paths
            path = new_path
        
        # Find the final maximum probability
        max_prob = -1
        max_state = None
        
        for state in self.states:
            if V[len(observation_sequence) - 1][state] > max_prob:
                max_prob = V[len(observation_sequence) - 1][state]
                max_state = state
        
        return path[max_state]
    
    def predict(self, sequences):
        """
        Predict tags for a list of word sequences
        
        Args:
            sequences: List of lists, where each inner list contains words
            
        Returns:
            List of lists of predicted tags
        """
        predictions = []
        
        for sequence in sequences:
            predicted_tags = self.viterbi(sequence)
            predictions.append(predicted_tags)
        
        return predictions
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

def word_to_features(sent, i):
    word = sent[i]
    
    # Basic features
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:] if len(word) >= 3 else word,  # Suffix
        'word[-2:]': word[-2:] if len(word) >= 2 else word,  # Suffix
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isalpha()': word.isalpha(),
    }
    
    # Features for first word
    if i == 0:
        features['BOS'] = True
    else:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit(),
        })
    
    # Features for last word
    if i == len(sent)-1:
        features['EOS'] = True
    else:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
        })
    
    return features

def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

def evaluate_model(true_tags, pred_tags):
    """
    Evaluate model performance with precision, recall, F1-score and accuracy using different averaging methods
    """
    true_flat = [tag for sublist in true_tags for tag in sublist]
    pred_flat = [tag for sublist in pred_tags for tag in sublist]
    
    # Calculate accuracy
    accuracy = accuracy_score(true_flat, pred_flat)
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_flat, pred_flat, average='micro')
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_flat, pred_flat, average='macro')
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_flat, pred_flat, average='weighted')
    
    # Generate confusion matrix
    labels = sorted(list(set(true_flat)))
    cm = confusion_matrix(true_flat, pred_flat, labels=labels)
    
    # Create results dictionary
    results = {
        # Micro average (same as accuracy for multi-class classification)
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        
        # Macro average (treats all classes equally)
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        
        # Weighted average (accounts for class imbalance)
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'confusion_matrix_labels': labels
    }
    
    return results

def process_data(filename):
    """
    Process data from the input file into sequences of (word, tag) pairs
    """
    sequences = []
    current_sequence = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    word, tag = parts[0], parts[1]
                    current_sequence.append((word, tag))
    
    # Add the last sequence if it exists
    if current_sequence:
        sequences.append(current_sequence)
    
    return sequences

def plot_confusion_matrix(cm, labels, title, filename):
    """
    Plot and save confusion matrix as an image
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(12, 10))
    
    # Determine the number of labels to show
    num_labels = len(labels)
    
    # If there are too many labels, select a subset to display
    max_labels_to_show = 30
    if num_labels > max_labels_to_show:
        # Select most frequent classes
        label_counts = cm.sum(axis=1)
        top_indices = label_counts.argsort()[-max_labels_to_show:][::-1]
        shown_labels = [labels[i] for i in top_indices]
        shown_cm = cm[top_indices][:, top_indices]
    else:
        shown_labels = labels
        shown_cm = cm
    
    # Plot the confusion matrix
    sns.heatmap(shown_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=shown_labels, yticklabels=shown_labels)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Process both Hindi and English data
    languages = ['hindi', 'english']
    
    for language in languages:
        # Load training and test data
        train_file = f"{language}_train.txt"
        test_file = f"{language}_test.txt"
        
        train_sequences = process_data(train_file)
        test_sequences = process_data(test_file)
        
        # Prepare data for evaluation
        test_words = [[word for word, _ in sequence] for sequence in test_sequences]
        test_tags = [[tag for _, tag in sequence] for sequence in test_sequences]
        
        # Create results directory
        results_dir = f"{language}_results"
        models_dir = f"{language}_models"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Train and evaluate HMM
        hmm_model_path = os.path.join(models_dir, f"{language}_hmm_model.pkl")
        
        # Check if model exists, if not, train and save it
        if not os.path.exists(hmm_model_path):
            hmm = HMMTagger()
            hmm.train(train_sequences)
            # Save the trained model
            hmm.save(hmm_model_path)
        else:
            # Load the existing model
            hmm = HMMTagger.load(hmm_model_path)
        
        hmm_predictions = hmm.predict(test_words)
        hmm_results = evaluate_model(test_tags, hmm_predictions)
        
        # Plot and save HMM confusion matrix
        plot_confusion_matrix(
            hmm_results['confusion_matrix'],
            hmm_results['confusion_matrix_labels'],
            f"{language.capitalize()} HMM Confusion Matrix",
            os.path.join(results_dir, f"{language}_hmm_confusion_matrix.png")
        )
        
        # Train and evaluate CRF
        crf_model_path = os.path.join(models_dir, f"{language}_crf_model.pkl")
        
        # Prepare data for CRF
        train_words = [[word for word, _ in sequence] for sequence in train_sequences]
        train_tags = [[tag for _, tag in sequence] for sequence in train_sequences]
        
        X_train = [sent_to_features(sent) for sent in train_words]
        y_train = train_tags
        
        X_test = [sent_to_features(sent) for sent in test_words]
        
        # Check if model exists, if not, train and save it
        if not os.path.exists(crf_model_path):
            # Define the CRF model
            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True
            )
            
            # Train the model
            crf.fit(X_train, y_train)
            
            # Save the trained model
            with open(crf_model_path, 'wb') as f:
                pickle.dump(crf, f)
        else:
            # Load the existing model
            with open(crf_model_path, 'rb') as f:
                crf = pickle.load(f)
        
        # Make predictions
        crf_predictions = crf.predict(X_test)
        crf_results = evaluate_model(test_tags, crf_predictions)
        
        # Plot and save CRF confusion matrix
        plot_confusion_matrix(
            crf_results['confusion_matrix'],
            crf_results['confusion_matrix_labels'],
            f"{language.capitalize()} CRF Confusion Matrix",
            os.path.join(results_dir, f"{language}_crf_confusion_matrix.png")
        )
        
        # Save detailed results to files
        
        # Save HMM metrics
        with open(os.path.join(results_dir, f"{language}_hmm_results.txt"), 'w', encoding='utf-8') as f:
            f.write(f"{language.capitalize()} HMM Results\n")
            f.write("-" * (len(language) + 12) + "\n\n")
            f.write("Micro-average metrics:\n")
            f.write(f"  Precision: {hmm_results['precision_micro']:.4f}\n")
            f.write(f"  Recall: {hmm_results['recall_micro']:.4f}\n")
            f.write(f"  F1: {hmm_results['f1_micro']:.4f}\n\n")
            f.write("Macro-average metrics:\n")
            f.write(f"  Precision: {hmm_results['precision_macro']:.4f}\n")
            f.write(f"  Recall: {hmm_results['recall_macro']:.4f}\n")
            f.write(f"  F1: {hmm_results['f1_macro']:.4f}\n\n")
            f.write("Weighted-average metrics:\n")
            f.write(f"  Precision: {hmm_results['precision_weighted']:.4f}\n")
            f.write(f"  Recall: {hmm_results['recall_weighted']:.4f}\n")
            f.write(f"  F1: {hmm_results['f1_weighted']:.4f}\n\n")
            f.write(f"Accuracy: {hmm_results['accuracy']:.4f}\n")
        
        # Save CRF metrics
        with open(os.path.join(results_dir, f"{language}_crf_results.txt"), 'w', encoding='utf-8') as f:
            f.write(f"{language.capitalize()} CRF Results\n")
            f.write("-" * (len(language) + 12) + "\n\n")
            f.write("Micro-average metrics:\n")
            f.write(f"  Precision: {crf_results['precision_micro']:.4f}\n")
            f.write(f"  Recall: {crf_results['recall_micro']:.4f}\n")
            f.write(f"  F1: {crf_results['f1_micro']:.4f}\n\n")
            f.write("Macro-average metrics:\n")
            f.write(f"  Precision: {crf_results['precision_macro']:.4f}\n")
            f.write(f"  Recall: {crf_results['recall_macro']:.4f}\n")
            f.write(f"  F1: {crf_results['f1_macro']:.4f}\n\n")
            f.write("Weighted-average metrics:\n")
            f.write(f"  Precision: {crf_results['precision_weighted']:.4f}\n")
            f.write(f"  Recall: {crf_results['recall_weighted']:.4f}\n")
            f.write(f"  F1: {crf_results['f1_weighted']:.4f}\n\n")
            f.write(f"Accuracy: {crf_results['accuracy']:.4f}\n")
        

if __name__ == "__main__":
    main()
