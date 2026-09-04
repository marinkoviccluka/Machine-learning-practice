def total_euro(hours, rate):
    return hours * rate

hours = float(input("hours: "))
rate = float(input("rate: "))
total = total_euro(hours, rate)
print("Total:", total)