def remove_duplicate_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    original_line_count = len(lines)
    unique_lines = list(dict.fromkeys(lines))  # Preserves order and removes duplicates
    new_line_count = len(unique_lines)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)

    print(f"Original lines: {original_line_count}")
    print(f"Unique lines: {new_line_count}")

# Example usage
remove_duplicate_lines(r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_train_clean_10k.txt", r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_train_clean2_10k.txt")
