ham_words = 0
ham_count = 0
spam_words = 0
spam_count = 0
spam_exclamations = 0

with open("SMSSpamCollection.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        parts = line.split('\t', 1)
        label, text = parts
        words = text.split()

        if label == "ham":
            ham_words += len(words)
            ham_count += 1
        elif label == "spam":
            spam_words += len(words)
            spam_count += 1
            if text.endswith("!"):
                spam_exclamations += 1

if ham_count > 0:
    print("Avg in ham:", ham_words / ham_count)
if spam_count > 0:
    print("Avg in spam:", spam_words / spam_count)
print("Spam with !: ", spam_exclamations)