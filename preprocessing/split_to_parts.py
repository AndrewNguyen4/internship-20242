import os
import math

# Split a text data file to multiple parts to prevent running out of RAM

def split_file(input_file, output_dir, parts):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read all lines from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    part_size = math.ceil(total_lines / parts)

    for i in range(parts):
        start = i * part_size
        end = start + part_size
        part_lines = lines[start:end]

        part_file = os.path.join(output_dir, f"{i+1}.txt")
        with open(part_file, 'w', encoding='utf-8') as pf:
            pf.writelines(part_lines)

    print(f"Split {input_file} into {parts} parts in '{output_dir}'")

# Example usage:
split_file(r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_train_clean2_10k.txt", r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\train_parts", 20)
