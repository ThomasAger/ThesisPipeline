from collections import defaultdict
from functools import wraps
saved_data = defaultdict(list)

def save(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        saved_data[func.__name__].append({"args": args, "kwargs": kwargs, "result": result})
        return result

    return wrapper


@save
def add_nums(num1, num2):
   return num1 + num2


result = add_nums(1, 2)
print(saved_data, result)
result = add_nums(3, 4)
print(saved_data, result)
result = add_nums(num2=2, num1=1)
print(saved_data, result)


def decorator_function(input_function):
    @wraps(input_function)
    def wrapper_function():
        print('a {} {}'.format(input_function.__name__, "fart"))
        return input_function()
    return wrapper_function


def msg():
    print("hello")

decorated_display = decorator_function(msg)
print(decorated_display.__name__)


class TransformerClass():
    new_dict = {}
    def __init__(self, dct):
        self.new_dict = dct
        self.new_dict["hi"] = "hello"
        self.new_dict["bye"] = "goodbye"

    def getDct(self):
        return self.new_dict

class MainClass():
    dct = {}
    def __init__(self):


        self.dct = {"hi":"asdjolfasdjof",
                    "bye":"sdojkfgsjfffd"}
        transform = TransformerClass(self.dct)
        new_dict = transform.getDct()
        self.manuallySetClassVariables(new_dict)

    # Need to automate this in a for loop
    def manuallySetClassVariables(self, new_dict):
        self.hi_message = new_dict["hi"]
        self.bye_message = new_dict["bye"]

    def setClassVariables(self, new_dict):
        matching_dct = {"hi": self.hi_message,
                        "bye": self.bye_message}
        for key, value in new_dict:
            matching_dct[key] = new_dict[key]

test = MainClass()

