'''
Cleans messages in testing and development sets.
Removes messages not seen in the 64k vocab.
'''

def clean_messages(vocab_file, messages_file, output_file):
    # Load vocabulary into a set for fast lookup
    with open(vocab_file, 'r', encoding='utf-8') as vf:
        vocab = set(word.strip() for word in vf if word.strip())

    with open(messages_file, 'r', encoding='utf-8') as mf, \
         open(output_file, 'w', encoding='utf-8') as of:

        for line in mf:
            words = line.strip().split()
            if all(word in vocab for word in words):
                of.write(line)

# Example usage
print('start')
clean_messages(r"mobiletext/vocab_posts_20k.txt", r"mobiletext/sent_train_clean.txt", r"mobiletext/sent_train_clean_20k.txt")
print('completed')