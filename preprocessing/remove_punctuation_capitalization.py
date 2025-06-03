import re

def clean_text(text):
    """Convert to lowercase and remove punctuation except apostrophes in contractions."""
    return re.sub(r"[^\w\s']", "", text).lower()

def process_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            cleaned_line = clean_text(line.strip())
            outfile.write(cleaned_line + "\n")

# Replace with your actual file paths
input_file = r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_dev.txt"
output_file = r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_dev_clean.txt"
process_file(input_file, output_file)
print(f"Processed text saved to {output_file}")
