
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
