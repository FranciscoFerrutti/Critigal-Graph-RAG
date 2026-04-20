"""
This script counts the number of commas in a string before a specified character.
Example usage:
```python count_comas_before_char.py file.txt , a```
In this example, the script will read the contents of `file.txt`, count the number of commas before the character 'a' for each line, and print all distinct counts.
So, if file.txt contains:
```
hello, world, this is a test
a, b, c
hello, world, this is a test
```
The output will be:
```
2
0
```
"""
import sys

def count_commas_before_char(file_path, char):
    counts = set()
    with open(file_path, 'r') as file:
        for line in file:
            count = 0
            for c in line:
                if c == char:
                    break
                if c == ',':
                    count += 1
            counts.add(count)
    return counts

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python count_comas_before_char.py <file_path> <char>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    char = sys.argv[2]

    print(f"Counting commas before character '{char}' in file '{file_path}'...")
    
    counts = count_commas_before_char(file_path, char)
    for count in counts:
        print(count)
