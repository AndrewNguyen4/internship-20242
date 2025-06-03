import pandas as pd
import matplotlib.pyplot as plt


file_path = r'C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_train_clean_10k.txt'
output_file = r'C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_dev_no_metadata.txt'
data = pd.read_csv(file_path, sep="\t", encoding="utf-8", low_memory=False)

# Extract the last column (assuming it contains text messages)
text_messages = data.iloc[:, -1].astype(str)  

# text_messages.to_csv(output_file, index=False, header=False, encoding="utf-8")

# print(f"Saved {len(text_messages)} messages to {output_file}")

# Compute word count for each message
word_counts = text_messages.apply(lambda x: len(x.split()))

# Count occurrences of each word length
word_length_distribution = word_counts.value_counts().sort_index()

# Print detailed distribution
print("Word Length Distribution:")
for length, count in word_length_distribution.items():
    print(f"{length} words: {count} messages")


# Plot the distribution of word counts
plt.figure(figsize=(10, 6))
plt.hist(data['word_count'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.title('Distribution of Text Message Lengths (Word Count)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Print summary statistics
print(data['word_count'].describe())