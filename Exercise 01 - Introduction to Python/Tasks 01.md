# Tasks 

## Task 1.4.1
Write a program that requests the user to enter working hours and how much they are paid per working hour. Use the built-in Python method `input()`. After that, calculate how much the user earned and print it to the screen. Finally, rewrite the solution so that the total amount is calculated in a separate function called `total_euro`.

**Example:**
Working hours: 35 h, euros/h: 8.5, Total: 297.5 euros

## Task 1.4.2
Write a program that requests the user to type in one number representing some kind of grade, located between 0.0 and 1.0. Print which category the grade belongs to based on the following conditions:

- `>= 0.9` → A
- `>= 0.8` → B
- `>= 0.7` → C
- `>= 0.6` → D
- `< 0.6` → F

If the user did not type in a number, print an error message to the screen (use `try` and `except` statements). Also, if the number is outside the interval [0.0, 1.0], an appropriate message must be printed.

## Task 1.4.3
Write a program that requests the user to enter numbers in an infinite loop until the user types "Done" (without quotes). While doing so, store the numbers in a list. After that, print how many numbers the user entered, their mean, minimum, and maximum value. Sort the list and print it to the screen.

Additionally: safeguard the program against incorrect input (e.g. a letter instead of a digit) in such a way that the program ignores that input and prints an appropriate message.

## Task 1.4.4
Write a Python script that will load a text file named `song.txt`. It is necessary to build a dictionary that uses as keys all the distinct words that appear in the file, while the values equal the number of times each word (key) appears in the file. How many words appear only once in the file? Print them.

## Task 1.4.5
Write a Python script that will load a text file named `SMSSpamCollection.txt` [1]. This file contains 5574 SMS messages, where some are labeled as spam and some as ham.

**Example excerpt from the file:**
```
ham Yup nextstop.
ham Ok lar... Joking wif u oni...
spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken's stuff!
```
