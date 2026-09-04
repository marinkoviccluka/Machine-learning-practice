while True:
    grade_str = input("Grade (0.0-1.0): ")

    try:
        grade = float(grade_str)
    except ValueError:
        print("Nan")
        continue
    if grade < 0.0 or grade > 1.0:
        print("try 0.0-1.0.")
        continue
    
    break

if grade >= 0.9:
    print("A")
elif grade >= 0.8:
    print("B")
elif grade >= 0.7:
    print("C")
elif grade >= 0.6:
    print("D")
else:
    print("F")
