#!/usr/bin/env py
# -*- coding: utf-8 -*-
# This script is a simple addition program with a user interface and a test function.
# get a function which add two numbers
def add(a, b):
    return a + b
# call add function with user interface
def main():
    print("Welcome to the addition program!")
    try:
        num1 = float(input("Enter the first number: "))
        num2 = float(input("Enter the second number: "))
        result = add(num1, num2)
        print(f"The sum of {num1} and {num2} is {result}.")
    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    main()
# test the add function
def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(-1, -1) == -2
    assert add(1.0, 2.5) == 4.0
    print("All tests passed!")
    
# run the test function
if __name__ == "__main__":
    test_add()
    
