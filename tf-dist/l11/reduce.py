from functools import reduce

def add(a, b):
    return a + b

print(reduce(add, [1, 2, 3]))
