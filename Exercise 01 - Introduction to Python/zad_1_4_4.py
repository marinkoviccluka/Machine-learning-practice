counts = {}

with open("song.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip()
        words = line.split()
        for word in words:
            word = word.lower()
            counts[word] = counts.get(word, 0) + 1

once_words = []
for w, c in counts.items():
    if c == 1:
        once_words.append(w)

print("Num words showing once", len(once_words))
print("Words showing once")
for w in once_words:
    print(w)
