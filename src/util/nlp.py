import numpy as np
import re

def stringToArray(string):
    array = string.split()
    for e in range(len(array)):
        array[e] = eval(removeEverythingFromString(array[e]))
    return array

def keepNumbers(string):
    s = stripPunctuation(string)
    s = s.lower()
    s = "".join(s.split())
    numbers = re.compile('\d+(?:\.\d+)?')
    s = numbers.findall(s)
    return s
def removeEverythingFromString(string):
    string = stripPunctuation(string)
    string = string.lower()
    string = "".join(string.split())
    return string
def lowercaseSplit(string):
    string = string.lower()
    string = "".join(string.split())
    return string

def lower_nopunct(string):
    s = stripPunctuation(string)
    s = s.lower()
    return s


def stripPunctuation(text):
    punctutation_cats = set(['Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'])
    return ''.join(x for x in text
                   if unicodedata.category(x) not in punctutation_cats)

