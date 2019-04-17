#comprehension
#dictionary
data = {i : np.random.randn() for i in range(7)}#this will create a dictionary
#other ways to create dictionary
some_dict = {}
words = ['apple', 'bat', 'bar', 'atom', 'book']
for word in words:
    some_dict.setdefault(word[0],[]).append(word)
#or
from collections import defaultdict
some_dict = defaultdict(list)
print(some_dict)
for word in words:
    some_dict[word[0]].append(word)

#print statement we can use seperator
for i in squares(15):
    print(i, end=' ')#for printting in a single line, in place of space we can use anything we can use \r to print clear print 
	
#sting replace
a = 'i have small car'
b = a.replace('small', 'big')

#decoding
decode = b.encode('UTF-8')

#None
a=None
a==None, a is None#both operation will return True

#datetime
from datetime import datetime, date, time
dt = datetime(2018, 7, 30, 14, 58, 10)#defining a date
dt.day, dt.month, dt.year, dt.hour, dt.date(), dt.time()#extracting info from date

#strftime, strptime
dt.strftime('%m/%d/%Y %H:%M')#this will format time in the give format
dt.strftime('%m')#returns month
#strptime is like reading date in a particular format and strftime is to change format
datetime.strptime(date_string,date_string_format).strftime(convert_to_date_string_format)
datetime.datetime.strptime("01/27/2012","%m/%d/%Y").strftime('%m-%d-%Y')#example

#itertools
import itertools
first_letter = lambda x : x[0]#a function which returns first letter of a word
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
for letter, names in itertools.groupby(names, first_letter):#this code will group names based on return value of function first_letter
    print(letter, list(names))
#permutation
for combis in itertools.permutations(['a','b','c'], 3):#all three permutation on a, b and c, here 3 means all 3 letter permutations
    print(combis)

#counter	
from collections import Counter
a = ['a', 'b', 'a']
print(Counter(a).most_common(1))#output is [('a', 2)]

#sorted
sorted(arr)#it gives sorted array also you can write some rule on which the sorting should be done using keys
arr = ['abhi', 'zzzzzzz','yy','mmll']
sorted(arr)#output ['abhi', 'mmll', 'yy', 'zzzzzzz']
sorted(arr, key=lambda x : len(x))#sort the data based on length of the words, output ['yy', 'abhi', 'mmll', 'zzzzzzz']

#underscore in loops
for _ in range(10):#it means do something 10 times, we need not to specify a variable there
	do something
	
#to take user input
abc = input('write something ')
abc = int(input('write something'))#only takes integer input

#evaluate
abc = eval(input('write something'))#will evaluate and return the value, here we can input a String, that will be evaluated as Python code

1 == 1.0#this will return true in python

'a' > 'b'#will return false

#code to find the cube root of a number
cube = 0.5
epsilon = 0.01
num_guess = 0
low = min(0, cube)
high = max(0, cube)

if cube < 1:
    low = cube
    high = 1
guess = (high + low)/2.0

while abs(guess ** 3 - cube) >= epsilon:
    if guess ** 3 < cube:
        low = guess
    else:
        high = guess
    guess = (high + low)/2.0
    num_guess += 1

print('number of guesses',num_guess)
print('number closest to cube root is',guess)
	
#global variable
def some_global(num):
    global a
    a = num
print(some_global(3), a)#prints out None, 3
	
#append vs extend
x = [1,2,3]
x.append([2,3])
print(x)#[1, 2, 3, [2, 3]]

x = [1,2,3]
x.extend([2,3])
print(x)#[1, 2, 3, 2, 3]

#decorator
from time import time
def time_it(func):
    t1 = time()
    func()
    t2 = time()
    print('the time taken was ',t2-t1)
    return func
@time_it    
def abc():
    sum=0
    for i in range(1000000):
        sum+=i
    #print(sum)   
#abc = time_it(abc)#either you can write this or you can write @time_it above function
abc()

#try except
try:
    print(1/0)
except:
    raise ValueError('some error')#raise explicitly value error
	
#assert
def someFunc():
    a = 1
    b = 0
    assert not b == 0, 'denominator is zero'#will give assertion error if b==0
    return a/b
print(someFunc())
	
#appending zeros, adding leading Zeros
str(3).zfill(3)

###### numpy and pandas #######
	
#null filling time series data using interpolate pandas
Data.interpolate(inplace=True, axis=0)

#** also we can use read_csv and read_table to read files from the internet

#repeat
np.repeat(1,20)#this will create a numpy array of 20 elements all 1

#groupby
#whenever you group by usually the grouped label come as index to not do that we use
df.groupby(['colToGroup'], as_index=False)#as_index = False means dont use this as index but create seperate index

#to get all the numeric columns
numCols = data.describe().columns#as describe will only output numeric columns , to get the object columns we ca use data.describe(include=['O'])

#to group data based on month name using date field
data.groupby(data['Date field'].dt.strftime('%B')).size()

#dummy variables
data = pd.get_dummies(data, columns=columns_to_get_dummies_of)
data.columns#here you will see all the columns + columns which are created using get_dummies

#use columns
usecols=np.arange(1,21)#we can give something like this also on pd.read_csv



