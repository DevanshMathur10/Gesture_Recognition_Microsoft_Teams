import string

# Create a list of all digits
digits = [str(i) for i in range(10)]

# Create a list of all lowercase letters
lowercase_letters = list(string.ascii_lowercase)

# Create a list of all uppercase letters
uppercase_letters = list(string.ascii_uppercase)

# Combine all digits and letters into a single list
all_digits_and_letters = digits + uppercase_letters

print(all_digits_and_letters)
print(len(all_digits_and_letters))