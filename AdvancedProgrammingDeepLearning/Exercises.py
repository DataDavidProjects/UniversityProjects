import numpy

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self, item):
        el = self.items[-1]
        del self.items[-1]
        return el

    def peak(self):
        return self.items[-1]

    def size(self):
        return len(self.items)

    def isempty(self):
        if len(self.items) >0 : return False
        else: return True

class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        el = self.items[0]
        del self.items[0]
        return el

    def isEmpty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

class Node:
    def __init__(self,initdata):
        self.data = initdata
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self,newdata):
        self.data = newdata

    def setNext(self,newnext):
        self.next = newnext

def checkParenthesis(parenthesis):
  s = Stack()
  for char in parenthesis:
    if char == '(':
      s.push(char)
    elif char ==')':
      if s.isEmpty():
        return False
      else:
        s.pop()
  return s.isEmpty()

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

def sum_up(n):
    if n == 1:
        return 1
    return n + sum_up(n - 1)

print(factorial(3))

def create_matrix(n,m):
    # n rows , m columns
    # A matrix is a collection of vectors/lists
    # Define the create_vector function of n elements with 0 as placeholders
    create_vector_placeholder = lambda n: [0]*n
    # create the collection of vectors with placeholders 0
    matrix = [ create_vector_placeholder(n)] * m
    return matrix


def swap(a,b):
    a,b = b,a
    return a,b

# From total seconds to unit time

def time_parser(input_sec):
    h = 3600
    m = 60
    s = 1
    for t,d in zip([h, m, s],['Hour','Minutes',"Seconds"]):
        unit_time = input_sec // t
        input_sec = input_sec % t
        print(d,unit_time)



def is_perfect(number):
    """
    In number theory, a perfect number is a
    positive integer that is equal to the sum of its positive divisors,
    excluding the number itself.
    For instance, 6 has divisors 1, 2 and 3, and 1 + 2 + 3 = 6, so 6 is a perfect number.
    """
    assert type(number) == int and number > 0, "Definition of perfect number required positive integer"
    # Algo
    # Find divisors, except the number itself
    # Sum divisors
    # Check definition and return True or False

    #______________Codig Solution__________________________________________
    # Find divisors: at steps -1 we iterate in a top down manner and by definition we exclude the number
    divisors = [ i for i in range(number-1,0,-1) if (number%i) == 0 ]
    # Sum divisors
    sum_divisors = sum(divisors)
    # Return based on definition
    if sum_divisors == number: return bool(1)
    else: return  bool(0)
    #______________________________________________________________________

print(is_perfect(6))

# init a dictionary:
bag_dict = dict()
def bag(bag_dict,element):
    # if the element e is in the bag, increase its count in the collection
    # otherwise this means e occurs 0 times in the bag, so set its count to 1
    if element in bag_dict:
        bag_dict[element] += 1
    else:
        bag_dict[element] = 1

def countAstarB(word):
    word = word.lower() # we disregard the letter cases
    c = 0
    if len(word)>=3:
        # the last character's position is len(word)-1, but the pattern
        # we search for is 3 character long, so we shall include position
        # len(word)-3
        for pos in range(0, len(word)-2):
            if word[pos] == "a" and word[pos+2] == "b":
                c = c+1
    return c


def staircase(n):
    for i in range(1, n+1):
         print(("#" * i).rjust(n))


def diagonalDifference(arr):
    # Write your code here
    principal = 0
    secondary = 0
    n = len(arr)
    for i in range(0, n):
        principal += arr[i][i]
        secondary += arr[i][n - i - 1]

    return abs(principal - secondary)


# Permutation Matrix (n!,n) with (first element repetition)= n!/n
def print_permute_list(x,i=0):
    # x must be a list object
    assert type(x) == list
    # Recursive Backtracking
    if i >= len(x):
        print(x)
        return
    for j in range(i,len(x)):
        # swap element in position i and j
        x[i], x[j] = x[j], x[i]
        print_permute_list(x,i+1)
        x[i], x[j] = x[j], x[i]

M = print_permute_list([1,2,3])

def permutation_matrix(x):
    assert type(x) == list
    n = len(x)
    if x == [] or n == 1:
        return [x]
    else:
        sub_x = []
        matrix  = []
        for i in range(n):
            # Swap operation
            x[0] , x[i] = x[i] ,x[0]
            # Recursion , WE REPEAT UNTIL HERE each recursion
            sub_x = permutation_matrix( x[1:n]) # the list is starting by the element 2 of the collection
            k = len(sub_x)
            # Create the matrix concatenating matrix( as a list of lists )
            # the matrix is a concatenation of itself with a list ( that contains the fixed and the swapped elements)
            matrix = matrix +[  [x[0]]+sub_x[j]   for j in range(k) ]
        return  matrix

def sum_elements(l):
    assert type(l)== list
    if len(l)==0:
        return 0
    else:
        return l[0] + sum_elements(l[1:])

sum_elements([1,2,3,4])

# Fibonacci like series
def nth_of_sequence(n,a=1,b=2,c=3):
    if n < 3:
         return 1
    for i in range(3,n+1):
        a, b, c = b, c, (3*a + 2*b + c)
    return c
nth_of_sequence(10,0,6,7)

# flat list of list
# [item for sublist in list_of_list for item in sublist]

a = [
    {"a":1,"b":3.1},
    {"a":2,"c":4 },
    {"c":2,"d":-5}
]
import pandas as pd
def merge_dicts(list_of_dict,constraints = [int, float]):
    df = pd.DataFrame(list_of_dict)
    assert df.apply(lambda x: x.dtype).isin(constraints).all()
    assert sum([ 0 if type(item) == str else 1 for sublist in list(df.columns) for item in sublist ]) ==0
    return dict(df.max())
merge_dicts(a)


# flatten list of list
def lista(): return list([1,2,3,4])
matrix = [lista() for i in range(10)]
elementi  = [ element for riga in matrix for element in riga]