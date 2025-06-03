from collections import Counter

# Step 1: Count words
with open(r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_train_clean_10k.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    words = [word for line in lines for word in line.strip().split()]
    word_counts = Counter(words)
    top_words = set(word for word, _ in word_counts.most_common(8000))

print("a")

# Step 2: Filter lines
filtered_lines = [line for line in lines if all(word in top_words for word in line.strip().split())]

# Step 3: Write result
with open(r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_train_clean_8k.txt", "w", encoding="utf-8") as f:
    f.writelines(filtered_lines)
