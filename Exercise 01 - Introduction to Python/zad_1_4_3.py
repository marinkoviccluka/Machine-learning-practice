numbers = []

while True:
    s = input("Num or Done: ")
    if s == "Done":
        break
    try:
        num = float(s)
        numbers.append(num)
    except ValueError:
        print("Wrong entry")

if len(numbers) == 0:
    print("List empty")
else:
    print("Num count:", len(numbers))
    print("Avg", sum(numbers) / len(numbers))
    print("Min:", min(numbers))
    print("Max:", max(numbers))
    numbers.sort()
    print("Sort:", numbers)
