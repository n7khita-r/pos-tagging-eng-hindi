import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Function to count POS tag frequencies from a given file
def count_pos_tags(filename):
    pos_counter = Counter()
    
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 2:  # Ensure line has both word and tag
                _, tag = parts
                pos_counter[tag] += 1
    return pos_counter

# File paths
eng_file1 = "english_train.txt"
hindi_file1 = "hindi_train.txt"
eng_file = "english_test.txt"
hindi_file = "hindi_test.txt"
eng_file2="eng.txt"
eng_file3="hin.txt"

# Count tag frequencies
eng_pos_countstext = count_pos_tags(eng_file)
hindi_pos_countstext = count_pos_tags(hindi_file)
eng_pos_countstrain = count_pos_tags(eng_file1)
hindi_pos_countstrain = count_pos_tags(hindi_file1)
eng = count_pos_tags(eng_file2)
hin = count_pos_tags(eng_file3)

# Plot bar graph with Zipf's law overlay
def plot_pos_bar_with_zipf(pos_counts, language, filename):
    # Sort tags by frequency (descending)
    sorted_counts = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
    tags, counts = zip(*sorted_counts)
    
    # Create ranks for each tag
    ranks = np.arange(1, len(tags) + 1)
    
    # Calculate Zipf's law theoretical distribution
    total_occurrences = sum(counts)
    zipf_counts = [total_occurrences * (1/r) for r in ranks]
    
    # Normalize to match the total of actual frequencies
    zipf_sum = sum(zipf_counts)
    zipf_counts = [f * (total_occurrences / zipf_sum) for f in zipf_counts]
    
    # Create bar graph with Zipf overlay
    plt.figure(figsize=(12, 6))
    
    # Plot actual distribution as bars
    plt.bar(ranks, counts, color='red', alpha=0.6, label=f'Actual {language} POS Counts')
    
    # Plot Zipf's law as a line
    plt.plot(ranks, zipf_counts, 'b-', linewidth=2, label="Zipf's Law")
    
    # Labels and title
    plt.xlabel("POS Tag Rank")
    plt.ylabel("Frequency")
    plt.title(f"{language} POS Tag Distribution vs Zipf's Law")
    plt.legend()
    
    # Add x-tick labels (tags) for top N tags to avoid overcrowding
    top_n = min(20, len(tags))  # Show top 20 tags or all if less than 20
    plt.xticks(ranks[:top_n], tags[:top_n], rotation=45)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Create one bar graph for each dataset
plot_pos_bar_with_zipf(eng_pos_countstext, "English Text", "english_test.png")
plot_pos_bar_with_zipf(hindi_pos_countstext, "Hindi Text", "hindi_test.png") 
plot_pos_bar_with_zipf(eng_pos_countstrain, "English Training", "english_training.png")
plot_pos_bar_with_zipf(hindi_pos_countstrain, "Hindi Training", "hindi_training.png")
plot_pos_bar_with_zipf(eng, "English BIS-POS tags", "eng.png")
plot_pos_bar_with_zipf(hin, "Hindi BIS-POS tags", "hin.png")
