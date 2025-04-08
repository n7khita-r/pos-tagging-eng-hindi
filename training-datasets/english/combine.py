import os
import re

# Files to process
files = [
    'en_ewt-ud-dev.conllu',
    'en_ewt-ud-test.conllu',
    'en_ewt-ud-train.conllu',
    'en_gum-ud-dev.conllu',
    'en_gum-ud-test.conllu',
    'en_gum-ud-train.conllu'
]

output_file = 'train.txt'

def extract_number_mood(features):
    """Extract Number and Mood information from features column"""
    if features == '*' or features == '_':
        return '_'
    
    number_match = re.search(r'Number=(\w+)', features)
    mood_match = re.search(r'Mood=(\w+)', features)
    
    result = []
    if number_match:
        result.append(f"Number={number_match.group(1)}")
    if mood_match:
        result.append(f"Mood={mood_match.group(1)}")
    
    return '|'.join(result) if result else '_'

# Open output file
with open(output_file, 'w', encoding='utf-8') as outf:
    # Process each file
    for filename in files:
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found, skipping.")
            continue
            
        print(f"Processing {filename}...")
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip comment lines and empty lines
                if line.startswith('#') or line.strip() == '':
                    continue
                
                # Split the line into columns
                columns = line.strip().split('\t')
                
                # Check if this is a valid data line (should have at least 10 columns)
                if len(columns) >= 10:
                    try:
                        # Extract the required columns
                        word = columns[1]  # Word form
                        lemma = columns[2]  # Lemma (word root)
                        pos = columns[3]    # POS tag
                        
                        # Extract Number and Mood from features
                        features = columns[5]
                        number_mood = extract_number_mood(features)
                        
                        # Write to output file
                        outf.write(f"{word}\t{lemma}\t{pos}\t{number_mood}\n")
                    except IndexError:
                        # Skip malformed lines
                        continue

print(f"Extraction complete. Results saved to {output_file}")
