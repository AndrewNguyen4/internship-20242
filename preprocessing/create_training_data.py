import random

def generate_test_samples(input_file, output_file, is_word_completion, min_prefix_words=2, min_completion_chars=1):
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    test_samples = []

    for sentence in sentences:
        words = sentence.split()
        if len(words) < min_prefix_words + 1:  # Ensure there's enough context
            continue  

        if is_word_completion:
            # Choose a word for completion with at least min_prefix_words before it
            completion_index = random.randint(min_prefix_words, len(words) - 1)  
            prefix_words = words[:completion_index]  # Keep words before the chosen word
            word_to_complete = words[completion_index]
            num_visible_chars = random.randint(min_completion_chars, max(1, len(word_to_complete) - 1))
            visible_part = word_to_complete[:num_visible_chars]

            input_text = " ".join(prefix_words + [visible_part])
            expected_output = word_to_complete[num_visible_chars:]
        else:
            # Next-word prediction: take a complete prefix and predict the next full word
            prefix_length = random.randint(min_prefix_words, len(words) - 1)
            input_text = " ".join(words[:prefix_length])
            expected_output = words[prefix_length]

        if not f"{input_text} | {expected_output}".rstrip().endswith("|"):
            test_samples.append(f"{input_text} | {expected_output}")

    # Save the test set
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(test_samples))

    print(f"Generated {'word completion' if is_word_completion else 'next-word prediction'} test set saved to: {output_file}")

# Generate separate test sets
generate_test_samples(r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_test_clean.txt", r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_test_completion.txt", is_word_completion=True)
generate_test_samples(r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_test_clean.txt", r"C:\Users\AndrewNguyen\Desktop\CodingProject\20242\Internship\mobiletext\sent_test_prediction.txt", is_word_completion=False)
