import re
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Set, Optional

# Corrected FST implementation
class MorphFST:
    def __init__(self):
        self.transitions = {}  # (state, input) -> (next_state, output)
        self.final_states = set()
        self.start_state = 0
        self.state_counter = 1
        self.final_outputs = {}  # final_state -> output to append at the end
    
    def add_rule(self, pattern, replacement, features):
        """Add a morphological rule to the FST."""
        current_state = self.start_state
        
        # Find the shared prefix between pattern and replacement
        prefix_len = 0
        for i in range(min(len(pattern), len(replacement))):
            if pattern[i] == replacement[i]:
                prefix_len += 1
            else:
                break
        
        # Process the pattern
        for i, char in enumerate(pattern):
            if i < prefix_len:
                # Part of the shared prefix - output the same character
                output = char
            else:
                # After the shared prefix - output empty string
                output = ""
            
            # Create the next state
            if (current_state, char) in self.transitions:
                next_state = self.transitions[(current_state, char)][0]
            else:
                next_state = self.state_counter
                self.state_counter += 1
            
            self.transitions[(current_state, char)] = (next_state, output)
            current_state = next_state
        
        # Mark the final state and store what remains of the replacement
        self.final_states.add(current_state)
        self.final_outputs[current_state] = replacement[prefix_len:]
        
        # Store features with the final state
        self.transitions[(current_state, "FEATURES")] = features
    
    def process(self, input_string):
        """Process an input string through the FST."""
        current_state = self.start_state
        output = []
        
        for char in input_string:
            if (current_state, char) in self.transitions:
                next_state, out_char = self.transitions[(current_state, char)]
                if out_char:  # Only append non-empty outputs
                    output.append(out_char)
                current_state = next_state
            else:
                return None, None  # No valid transition
        
        if current_state in self.final_states:
            # Append any final output for this state
            if current_state in self.final_outputs:
                output.append(self.final_outputs[current_state])
            
            features = self.transitions.get((current_state, "FEATURES"), {})
            return ''.join(output), features
        
        return None, None  # Didn't end in a final state


# Simplified CRF implementation for morphological analysis
class SimpleCRF:
    def __init__(self):
        self.features = {}  # Feature functions
        self.weights = {}   # Weights for features
        self.labels = set()  # Possible morphological features
        self.trained = False
    
    def add_feature_function(self, name, function):
        """Add a feature function to the CRF."""
        self.features[name] = function
    
    def _extract_features(self, word, position):
        """Extract features for a word at a specific position."""
        result = {}
        for name, func in self.features.items():
            result[name] = func(word, position)
        return result
    
    def train(self, words, labels):
        """Train the CRF with labeled examples."""
        # Very simplified training
        if not self.trained:
            # Initialize weights
            for feature in self.features:
                self.weights[feature] = np.random.randn()
            
            # Add labels to the set
            for label_set in labels:
                for label in label_set:
                    self.labels.add(label)
            
            # For a real CRF, you would optimize weights here
            # This is just a placeholder
            self.trained = True
            return "CRF trained (simplified implementation)"
        return "CRF already trained"
    
    def predict(self, word):
        """Predict morphological features for a word."""
        if not self.trained:
            return "CRF not trained yet"
        
        # Extract features for each position in word
        word_features = []
        for i in range(len(word)):
            word_features.append(self._extract_features(word, i))
        
        # Very simplified prediction logic
        # In a real CRF, you would compute the most likely sequence
        potential_labels = {}
        for label in self.labels:
            score = 0
            for pos, features in enumerate(word_features):
                for feat_name, feat_value in features.items():
                    if feat_value:
                        score += self.weights[feat_name]
            potential_labels[label] = score
        
        # Return labels with highest scores
        threshold = max(potential_labels.values()) * 0.8 if potential_labels else 0
        return {k: v for k, v in potential_labels.items() if v >= threshold}


class EnhancedMorphAnalyzer:
    def __init__(self, use_fst=True, use_crf=False):
        self.word_to_root = {}
        self.word_to_pos = {}
        self.word_to_features = {}
        self.root_to_forms = defaultdict(list)
        self.pos_patterns = {}
        
        # Models
        self.use_fst = use_fst
        self.use_crf = use_crf
        self.fst_models = defaultdict(MorphFST)  # POS -> FST model
        self.crf_model = SimpleCRF()
        
        # Set up basic CRF features if using CRF
        if use_crf:
            self._setup_crf_features()
    
    def _setup_crf_features(self):
        """Set up basic feature functions for the CRF."""
        # Character at position
        self.crf_model.add_feature_function(
            "char", lambda word, pos: word[pos] if 0 <= pos < len(word) else ""
        )
        
        # Previous character
        self.crf_model.add_feature_function(
            "prev_char", lambda word, pos: word[pos-1] if 0 < pos < len(word) else ""
        )
        
        # Next character
        self.crf_model.add_feature_function(
            "next_char", lambda word, pos: word[pos+1] if 0 <= pos < len(word)-1 else ""
        )
        
        # Is vowel
        self.crf_model.add_feature_function(
            "is_vowel", lambda word, pos: word[pos].lower() in "aeiou" if 0 <= pos < len(word) else False
        )
        
        # Is position start of word
        self.crf_model.add_feature_function(
            "is_start", lambda word, pos: pos == 0
        )
        
        # Is position end of word
        self.crf_model.add_feature_function(
            "is_end", lambda word, pos: pos == len(word) - 1
        )
    
    def load_file(self, filename):
        """Load training data from a file."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                data_lines = file.readlines()
                print(f"Successfully loaded {len(data_lines)} lines from {filename}")
                entries_loaded = self.load_data(data_lines)
                return f"Successfully trained on {entries_loaded} entries from {filename}"
        except FileNotFoundError:
            return f"Error: File {filename} not found."
        except Exception as e:
            return f"Error loading file: {str(e)}"

    def load_data(self, data_lines):
        """Load data from various formats."""
        words_for_crf = []
        labels_for_crf = []
        
        loaded_entries = 0
        skipped_lines = 0
        
        # Process lines in batches for multi-line entries
        i = 0
        while i < len(data_lines):
            line = data_lines[i].strip()
            i += 1
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Try to parse the line
            entry_parsed = False
            
            # Try tab-separated format first (most common)
            parts = line.split('\t')
            if len(parts) >= 4:
                # Format: word lemma POS features
                word, root, pos, features = parts[:4]
                self._add_entry(word, root, pos, features)
                loaded_entries += 1
                entry_parsed = True
                
                # Collect data for CRF training
                if self.use_crf:
                    words_for_crf.append(word)
                    labels_for_crf.append(self._parse_features(features))
            
            # If tab parsing didn't work, try space-separated
            if not entry_parsed:
                parts = line.split()
                if len(parts) >= 4:
                    word, root, pos = parts[:3]
                    features = ' '.join(parts[3:])
                    self._add_entry(word, root, pos, features)
                    loaded_entries += 1
                    entry_parsed = True
                    
                    # Collect data for CRF training
                    if self.use_crf:
                        words_for_crf.append(word)
                        labels_for_crf.append(self._parse_features(features))
            
            # If still not parsed, count as skipped
            if not entry_parsed:
                skipped_lines += 1
                print(f"Skipped line: {line}")
        
        # Train CRF model if enabled and we have data
        if self.use_crf and words_for_crf:
            crf_result = self.crf_model.train(words_for_crf, labels_for_crf)
            print(f"CRF training result: {crf_result}")
        
        print(f"Loaded {loaded_entries} entries, skipped {skipped_lines} invalid lines")
        return loaded_entries
    
    def _add_entry(self, word, root, pos, features):
        """Add a word entry to the analyzer."""
        word = word.strip()
        root = root.strip()
        pos = pos.strip()
        
        self.word_to_root[word] = root
        self.word_to_pos[word] = pos
        self.word_to_features[word] = self._parse_features(features)
        self.root_to_forms[root].append((word, pos, self.word_to_features[word]))
        
        # Add rule to appropriate FST if using FST
        if self.use_fst and root != word:
            # Avoid KeyError for new POS
            if pos not in self.fst_models:
                self.fst_models[pos] = MorphFST()
            
            self.fst_models[pos].add_rule(word, root, self.word_to_features[word])
        
        # Learn morphological patterns (for backup)
        if root != word:
            self._learn_pattern(word, root, pos, self.word_to_features[word])
    
    def _parse_features(self, feature_str):
        """Parse the feature string into a dictionary."""
        features = {}
        if not feature_str or feature_str == '_':
            return features
            
        for feature_pair in feature_str.split('|'):
            if '=' in feature_pair:
                key, value = feature_pair.split('=', 1)
                features[key] = value
        return features
    
    def _learn_pattern(self, word, root, pos, features):
        """Learn morphological patterns based on the differences between word and root."""
        # Simple suffix-based pattern
        if word.startswith(root):
            suffix = word[len(root):]
            if suffix:
                key = (pos, tuple(sorted(features.items())))
                if key not in self.pos_patterns:
                    self.pos_patterns[key] = []
                if (root, suffix) not in self.pos_patterns[key]:
                    self.pos_patterns[key].append((root, suffix))
        # Simple prefix-based pattern
        elif word.endswith(root):
            prefix = word[:-len(root)]
            if prefix:
                key = (pos, tuple(sorted(features.items())))
                if key not in self.pos_patterns:
                    self.pos_patterns[key] = []
                if (prefix, root) not in self.pos_patterns[key]:
                    self.pos_patterns[key].append((prefix, root))
        # Handle more complex transformations
        else:
            # For simplicity, we'll just store the full transformation
            key = (pos, tuple(sorted(features.items())))
            if key not in self.pos_patterns:
                self.pos_patterns[key] = []
            if (word, root) not in self.pos_patterns[key]:
                self.pos_patterns[key].append((word, root))
    
    def analyze(self, word, pos=None):
        """Analyze a word and return its possible analyses."""
        analyses = []
        
        # Direct lookup first
        if word in self.word_to_root:
            analyses.append({
                'word': word,
                'root': self.word_to_root[word],
                'pos': self.word_to_pos[word],
                'features': self.word_to_features[word],
                'method': 'lookup'
            })
            # Return early if position of speech matches
            if pos and self.word_to_pos[word] == pos:
                return analyses
        
        # Try FST if enabled
        if self.use_fst:
            pos_models = [self.fst_models[p] for p in self.fst_models if not pos or p == pos]
            for fst in pos_models:
                root, features = fst.process(word)
                if root:
                    pos_tag = next((p for p, m in self.fst_models.items() if m == fst), "UNKNOWN")
                    analyses.append({
                        'word': word,
                        'root': root,
                        'pos': pos_tag,
                        'features': features,
                        'method': 'fst'
                    })
        
        # Try CRF if enabled
        if self.use_crf and not analyses:
            predicted_features = self.crf_model.predict(word)
            if isinstance(predicted_features, dict) and predicted_features:
                # For simplicity, just use the word as root (a real implementation would do better)
                analyses.append({
                    'word': word,
                    'root': word,  # Placeholder, CRF doesn't predict the root
                    'pos': pos or "UNKNOWN",
                    'features': predicted_features,
                    'method': 'crf',
                    'confidence': 'medium'
                })
        
        # Try pattern matching as fallback
        if not analyses:
            for (pos_tag, feature_items), patterns in self.pos_patterns.items():
                if pos and pos != pos_tag:
                    continue
                    
                for pattern in patterns:
                    if len(pattern) == 2:
                        # Try suffix patterns
                        if isinstance(pattern[0], str) and isinstance(pattern[1], str):
                            if word.startswith(pattern[0]):
                                possible_root = word[len(pattern[0]):]
                                analyses.append({
                                    'word': word,
                                    'root': possible_root,
                                    'pos': pos_tag,
                                    'features': dict(feature_items),
                                    'method': 'pattern',
                                    'confidence': 'low'
                                })
                            # Try prefix patterns
                            elif word.endswith(pattern[1]):
                                possible_root = word[:-len(pattern[1])]
                                analyses.append({
                                    'word': word,
                                    'root': possible_root,
                                    'pos': pos_tag,
                                    'features': dict(feature_items),
                                    'method': 'pattern',
                                    'confidence': 'low'
                                })
        
        return analyses if analyses else [{'word': word, 'root': word, 'pos': pos or "UNKNOWN", 'features': {}, 'method': 'default', 'confidence': 'unknown'}]
    
    def generate_form(self, root, pos, features):
        """Generate an inflected form given a root, POS, and features."""
        # First check if we already know this form
        for word, word_pos, word_features in self.root_to_forms.get(root, []):
            if word_pos == pos and all(word_features.get(k) == v for k, v in features.items()):
                return {'word': word, 'method': 'lookup'}
        
        # Try using FST in reverse
        if self.use_fst and pos in self.fst_models:
            # This is simplified - a real FST would support bidirectional operation
            # Here we're just checking patterns we've learned
            for (pattern_pos, feature_items), patterns in self.pos_patterns.items():
                if pattern_pos != pos:
                    continue
                
                # Check if features match
                feature_dict = dict(feature_items)
                if all(feature_dict.get(k) == v for k, v in features.items()):
                    for pattern in patterns:
                        if len(pattern) == 2 and isinstance(pattern[0], str) and isinstance(pattern[1], str):
                            # If root is at start, add suffix
                            if pattern[0] == root:
                                return {'word': root + pattern[1], 'method': 'pattern'}
                            # If root is at end, add prefix
                                return {'word': pattern[0] + root, 'method': 'pattern'}
        
        # Fallback: Use the most common affix for this POS and feature combination
        feature_key = tuple(sorted(features.items()))
        affix_counts = defaultdict(int)
        
        for word, word_pos, word_features in self.root_to_forms.get(root, []):
            if word_pos == pos:
                # Check if features are similar
                if all(word_features.get(k) == v for k, v in features.items() if k in word_features):
                    # Calculate affix
                    if word.startswith(root):
                        affix = ('suffix', word[len(root):])
                    elif word.endswith(root):
                        affix = ('prefix', word[:-len(root)])
                    else:
                        continue
                    
                    affix_counts[affix] += 1
        
        # Use the most common affix
        if affix_counts:
            best_affix = max(affix_counts.items(), key=lambda x: x[1])[0]
            if best_affix[0] == 'suffix':
                return {'word': root + best_affix[1], 'method': 'common_affix'}
            else:
                return {'word': best_affix[1] + root, 'method': 'common_affix'}
        
        # Last resort: try to apply a general rule for this POS
        for word, word_pos, word_features in self.root_to_forms[root]:
            if word_pos == pos:
                # Just return the first form with matching POS
                return {'word': word, 'method': 'pos_match'}
        
        # If all else fails, return the root
        return {'word': root, 'method': 'default'}
    
    def get_stats(self):
        """Return statistics about the trained analyzer."""
        pos_counts = defaultdict(int)
        feature_counts = defaultdict(int)
        root_counts = defaultdict(int)
        
        for word, pos in self.word_to_pos.items():
            pos_counts[pos] += 1
            
        for word, features in self.word_to_features.items():
            for feature, value in features.items():
                feature_counts[f"{feature}={value}"] += 1
                
        for root, forms in self.root_to_forms.items():
            root_counts[root] = len(forms)
        
        return {
            "total_words": len(self.word_to_root),
            "unique_roots": len(self.root_to_forms),
            "pos_distribution": dict(pos_counts),
            "top_features": dict(sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "roots_with_most_forms": dict(sorted(root_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }

# Example usage
def main():
    # Create the analyzer
    analyzer = EnhancedMorphAnalyzer(use_fst=True, use_crf=True)
    
    # Load data from file
    training_file = "train.txt"
    result = analyzer.load_file(training_file)
    print(result)
    
    # Print some stats
    print("\n====== ANALYZER STATISTICS ======")
    stats = analyzer.get_stats()
    print(f"Total words in training data: {stats['total_words']}")
    print(f"Unique root words: {stats['unique_roots']}")
    print("\nPOS tag distribution:")
    for pos, count in sorted(stats['pos_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {pos}: {count}")
    
    # Test analysis
    print("\n====== ANALYSIS EXAMPLES ======")
    test_words = ["comes", "walked", "books", "stories", "unknown", "coming", "booking"]
    for word in test_words:
        print(f"Analysis for '{word}':")
        analyses = analyzer.analyze(word)
        for analysis in analyses:
            print(f"  Root: {analysis['root']}")
            print(f"  POS: {analysis['pos']}")
            print(f"  Features: {analysis['features']}")
            print(f"  Method: {analysis['method']}")
            if 'confidence' in analysis:
                print(f"  Confidence: {analysis['confidence']}")
            print()
    
    # Test form generation
    print("\n====== GENERATION EXAMPLES ======")
    test_generation = [
        ("walk", "VERB", {"Tense": "Pres", "VerbForm": "Part"}),
        ("walk", "VERB", {"Tense": "Past", "VerbForm": "Fin"}),
        ("book", "NOUN", {"Number": "Plur"}),
        ("story", "NOUN", {"Number": "Plur"})
    ]
    
    for root, pos, features in test_generation:
        print(f"Generate form for '{root}' + {pos} + {features}:")
        generated = analyzer.generate_form(root, pos, features)
        print(f"  Generated: {generated['word']}")
        print(f"  Method: {generated['method']}")
        print()

if __name__ == "__main__":
    main()
