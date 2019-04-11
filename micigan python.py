
	
---------------------------------------------------taking inputs----------------------------------------------------------------------------

#it will always consider input as string
a = input("get a number\n")
print(a*3)

#only integer inputs it will take else error
a = int(input("get a number\n"))
print(a*3)

# list input
a = list(input("get a number\n"))
print(a*3)


---------------------------------------------------assigning function to a variable---------------------------------------------------------
def add_num(x,y):
    print(x+y)
sum = add_num
sum(2,4)

---------------------------------------------------type---------------------

type(some_name)
#return type , eg, int ,string , NoneType , function

tuples - immutable eg , a = (1,'2',4,'c')

list - mutable

** as string is a list of chars .. so operation u can do on list u can do on string

--------------------------------------------------------unpacking---------------------------------

d = ('abhi','sing','jiya','muru')

mf, *ml, hl = d

print(mf+' '+str(ml)+' '+hl)

------------------------------------------- .format -------------------------------------------------

sales_record = {
    'price':3.24,
    'num_item':4,
    'person':'abhishek'
}

statement = '{} bought {} items for {} rs each'
print(statement.format(sales_record['person'],sales_record['num_item'],sales_record['price']))

---------------------------------------------------something (but its not working)---------------------
from decimal import *
getcontext().prec = 2

print(Decimal(2.4343543534))

---------------------------------------------------------basic CSV operation----------------------------

import csv
import urllib.request
import operator

with open('cars.csv') as csvFile:
    mpg = list(csv.DictReader(csvFile))

#print(mpg[:3])
#print(len(mpg))
print(mpg[0].keys())

#average HWYMG for all the data
print(round(sum(float(d['HWYMG']) for d in mpg)/len(mpg),2))
print(round(sum(float(d['COMB']) for d in mpg)/len(mpg),2))
#group by Model - gives different type of groups (creats a list of distinct model)
Model = set(d['Model'] for d in mpg)
print(Model)

HMPGbyMODL=[]
for m in Model:
    sumpg = 0
    count = 0
    for d in mpg:
        if d['Model']==m:
            sumpg+=float(d['HWYMG'])
        count+=1
    HMPGbyMODL.append((m,round(sumpg/count,2)))
HMPGbyMODL.sort(key=lambda x:x[1],reverse=True)
print(HMPGbyMODL)
# another way of sorting
#for s in sorted(HMPGbyMODL,key=operator.itemgetter(1),reverse=True):
#    print(s)

-------------------------------------------------------- date  and time -----------------------------------------------------

 epoch = january 1st 1970
 
import datetime as dt
import time as tm

print(tm.time())

print(dt.datetime.fromtimestamp(tm.time()).year)
print(dt.datetime.fromtimestamp(tm.time()).month)
print(dt.datetime.fromtimestamp(tm.time()).hour)

#create delta for subtraction or addition etc
delta = dt.timedelta(days = 100)
print(delta)

#example
today = dt.date.today()
print(today)
print(today-delta)

----------------------------------------------------------------list comprehension-------------------------------------------

a=[]
for n in range(50):
    if n%2==0:
        a.append(n)

print(a)

#list comprehension
b=[n for n in range(50) if n%2==0]
print(b)

--------------------------------------------------- basic numpy ---------------------------------------------------------------

import numpy as np
mylist = [1,3,6,2,7,4,8]

x = np.array(mylist)
print(x)

y = np.array([[1,2,3],[2,3,4]])
print(y)

#(row, column)
print(y.shape)

#like range
n = np.arange(0,11,2)
print(n)

#reshape the above array
m = n.reshape(2,3)
print(m)

# divide 1 to 10 into 20 parts
o = np.linspace(1,10,20)
print(o)

# resizes
p = np.resize(o,(5,4))
print(p)

#array of ones
print(np.ones((3,2)))

#arrays of zeros
print(np.zeros((3,2)))

#eye
print(np.eye((3)))

#diagonal
print(np.diag([2,3,4]))

#we can use * to multiply the elemests
print(np.array([1,2,3]*2))
print(np.array(([1,2,3],[2,3,4])*2))

#repeat each element 3rice
print(np.repeat([1,2,3],3))

#defining that the array will have int
p = np.ones([2,3],int)
print(p)

#stack 2 arrays
print(np.vstack([p,2*p]))
# stack horzontal	
print(np.hstack([p,2*p]))

#basic operation
x = np.array([1,2,3])
y = np.array([3,4,5])
print(x+y)
print(x*y)
print(x-y)
print(x.dot(y))#3*1 + 4*2 + 5*3

y = np.array([x,x**2])
print(y)
#transpose
print(y.T)
#type
print(y.dtype)

#type casting
y = y.astype('f')
print(y.dtype)

#basic array operation

a = np.array([-2,-1,5,-5,6,8,4,2])
print(a.sum())
print(a.max())
print(a.mean())
#standard deviation
print(a.std())
#print index of max or min
print(a.argmax())
#indexing and slicing
a = np.arange(13)**2
print(a)
#for 1d array its same as list
print(a[0],a[1],a[:3])

#for 2d array
#an 2d array
b = np.resize(a,(3,4))
print(b)
#second row second column - index starts at 0
print(b[2,2])
print(b[2,:])
print(b[:2,:-1])
#print every second element from last row
print(b[-1,1::2])
print(b[-1,::2])
#conditonal
print(b[b>20])
#also we can assin
# b[b>30] = 40
#if u print b after last statement all the values > 30  will become 30
b2 = b[:2,:2]
print(b2)
b2[:]=0
#changing in b2 changed b also
print(b)
#so for this we use copy
b = np.resize(a,(3,4))
print(b)
b2 = b.copy()
print(b2)
b2[:] = 0
print(b)
print(b2)

#iterate over arrays
test = np.random.randint(0,10,(4,3))
print(test)
print('-------------------------')
for row in test:
    print(row)
print('-------------------------')
for i in range(len(test)):
    print(test[i])
print('------------------------')
 #enumerate returns i , L[i]
for i,row in enumerate(test):
    print('row',i,'is',row)

test2 = test**2
print('-------------------------')
#iterating through both array
for i , j in zip(test,test2):
    print(i,'+',j,'=',i+j)
	
#len of an array  = no of rows
# rank  = no of dimensions

--->>>>>

import numpy as np
a = np.array([[1,2,3],[2,3,4]])
print(a)
#no of rows
print(len(a))
#no of columns
print(a.shape[1])
#no of dimension
print(a.ndim)

a = np.arange(15).reshape(3, 5)
print(a)
print(a.dtype.name)

#format ,you can write dtype= or simply complex
b = np.array([(1,2,3),(3,4,5)],dtype=complex)
print(b)

#function empty creates an array whose initial content is random and depends on the state of the memory.

c = np.empty((2,3),float)
print(c)
#pi = pi !
print(np.pi)

#example of pi

x = np.arange(0,2*np.pi,np.pi/2)
print(np.sin(np.pi))

#if array is to large it prints ...,
print(np.arange(10000))

# to disable the above write  -> np.set_printoptions(threshold='nan')

#dot product and simple product

a = np.array([(1,2),
              (2,3)])
b = np.array([(2,3),
              (1,2)])
print(a*b)
#dot product
print('------------')
print(a.dot(b))
print('------------')
print(np.dot(a,b))

#also opertaions line a+=b or a*=3 applies to matrix operation

print('------------')
#exp = e power the elememt in array eg e^1
d = np.exp(a)
print(d)

e = np.arange(12).reshape(3,4)
print(e)
#sum of each column
print(e.sum(axis=0))
#sum of each row
print(e.sum(axis=1))
#cumulative sum of each row
print(e.cumsum(axis=1))

# also we can use np.sqrt(array) or np.add(arr1,arr2)


#need to specify range of random integers
a = np.random.randint(1,5,(2,3))
print(a)

#no need to specify range of random integers
b = np.random.random((2,3))
print(b)

#print reverse array
c = np.arange(10)
print(c)
print(c[::-1])

d = np.array([-125, 1, -125, 27, -125, 125, 216, 343, 512, 729],dtype=int)
print(d)
#cube root of a no is possiby multivalued so might return NaN
for i in d:
    print(i,i**(1/3))

#the below from function generate an 5,4 array of indexes eg , 00,01,02,03:10,11,12,13 so on
#so for eg 11 is sent to f(x,y) it will return 10*1+1=11
def f(x,y):
    return 10*x+y

f = np.fromfunction(f,(5,4),dtype=int)
print(f)

#blank = : eg f[-1] = f[-1,:] *columns can be left blank not rows
print(f[:,-1])
print(f[...,-1])
print()
#print(f[:,1])
#all below three are same
print(f[1,:])
print(f[1])
print(f[1,...]) #three dots
print()
#for multidimension array eg a[1,...] means a[1,:,:,...]
#eg
g = np.array( [[[  0,  1,  2],               # a 3D array (two stacked 2D arrays)
                 [ 10, 12, 13]],
                [[100,101,102],
                 [110,112,113]]])
print()
print(g)
print()
print(g[1,:,:])#here as we have stacked array 1 = second array and rest parameter we can use as 2d array
print()
print(g[...,2])#from both arrays take 2nd column
print()
#we can also iterate over array using for loop
for row in f:#here row is just a variable
    print(row)
#if u want to trear each element of an array individually
print()
for i in f.flat:
    print(i)

print()
#print(f.size)
#make a single vector
print(np.resize(f,f.size))
print()
a = np.floor(10*np.random.random((3,4)))
print(a)
print(a.shape)
print(a.ravel())  # returns the array, flattened
print(a.reshape(6,2))
print(a.T)
print(a.shape)
print(a.T.shape)
#** the ndarray.resize method modifies the array itself # not confirmed
print(a)
b = np.resize(a,(4,3))
print(b)
print(a)
#when -1 the other dimension is automatically calculated
print(a.reshape(2,-1))
print(a.reshape(-1,2))
# column_stack stacks 1D arrays as columns into a 2D array. It is equivalent to vstack only for 1D arrays:
a = np.floor(10*np.random.random((2,2)))
b = np.floor(10*np.random.random((2,2)))
print()
print(a,b)
print()
print(a)
print()
print(b)
c = np.column_stack((a,b))
print()
print(c)
d = np.array([1,2,3])
print(d)
#newaxis is used to increase the dimension
print(d[:,np.newaxis])  # This allows to have a 2D columns vector
print()
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.hstack((a,b)))
print(np.vstack((a,b)))
print(np.column_stack((a,b)))
print()
print(a[:,np.newaxis]) # as a is 2D array it will become 3D array
print()
print(b[:,np.newaxis]) # as a is 2D array it will become 3D array
print()
print(np.column_stack((a[:,np.newaxis],b[:,np.newaxis])))
print()
print(np.vstack((a[:,np.newaxis],b[:,np.newaxis])))
print(np.hstack((a[:,np.newaxis],b[:,np.newaxis])))
#in np.concatenate we also give one more parameter axis which will define along which axis concate should happen

n = np.r_[1:4,0,4] # allows 1:4
print(n)

print()
m = np.array([[0, 1, 2], [3, 4, 5]])
print(np.r_['-1',m,m])

#np.r_('a,b,c',arr1,arr2)
#a = axis to concatenate along
#b = the minimum number of dimensions to force the entries to
#c = which axis should contain the start of the arrays
print()
print(np.r_['1,2', [1,2,3], [4,5,6]])
print(np.r_['1,2,0', [1,2,3], [4,5,6]])

#splitting array into smaller ones
m = np.floor(10*np.random.random((2,12)))
print(m)

print(np.hsplit(m,3))#horizontal split into 3
print()
print(np.hsplit(m,(3,4)))   # Split m after the third and the fourth column
#0 to 2 one matrix , 3 one matrix , 4 to last one matrix
#vsplit splits along the vertical axis, and array_split allows one to specify along which axis to split.

print(id(m))   # id is a unique identifier of an object
#the view method creates a new array object that looks at the same data.
print(a)
print(a.shape)
z = a.view()
print(z)
print(z is a)#false as z and a are diferent
print(z.base is a)#true as z is view of data owned by a
a.shape = 3,1
z.shape = 1,3
print(a.shape)#a's shape remain same
print(z.shape)
z[0,1] = 90
print(z)
print(a)#a's data changes
y = a.copy()#makes seperate copy no relation to original object

a = np.arange(12)**2                       # the first 12 square numbers
print(a)
print()
i = np.array( [ 1,1,3,8,5 ] )              # an array of indices
print(a[i])                                # the elements of a at the positions i
print()
j = np.array( [ [ 3, 4], [ 9, 7 ] ] )      # a bidimensional array of indices
print(a[j])                                # the same shape as j

print()


palette = np.array( [ [0,0,0],                # black
                       [255,0,0],              # red
                       [0,255,0],              # green
                       [0,0,255],              # blue
                       [255,255,255] ] )       # white

image = np.array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette
                     [ 0, 3, 4, 0 ]  ] )
print(image)
print()
print(palette[image])
print()
a = np.arange(12).reshape(3,4)
print()
print(a)
i = np.array( [ [0,1],                        # indices for the first dim of a
                 [1,2] ] )
print(i)
print()
j = np.array( [ [2,1],                        # indices for the second dim
                 [3,3] ] )
print()
print(j)
print()
l = [i,j]
print(l)
print()
print(a[:,j])#it will make 3d array where each line is 2d array based on indexes from j below example
print()
b=np.array([1, 2, 3, 6])
print(b[j])
print()
s = np.array([i,j])
print(s)
print()
print(tuple(s))
print()
#a time series inexing example
time = np.linspace(20,145,5)
print(time)
data = np.sin(np.arange(20).reshape(5,4))
print(data)
ind = np.argmax(data,axis=0)
print()
print(ind)
time_max = time[ind]
print(time_max)
print()
print(data.shape[1]) # no of rows for 2d matrix
print()
data_max = data[ind, range(4)] #data[ind, range(data.shape[1])]
print(data_max)
print(data.max(axis=0))#direct
g = np.arange(5)
print(g)
print(g[[2,3]]+1)
 
 #masking
 
a = np.arange(12).reshape(3,4)
print(a)
print()
b = [False,True,True]
print(a[b])
print()
print(a[[True,False],[False,True,True]])

a = np.arange(12).reshape(3,4)
b1 = np.array([False,True,True])             # first dim selection
b2 = np.array([True,False,True,True])
print()
print(a[b1,b2])


-------------------------------------------------------Pandas ----------------------------------------------------------------
#The result of an operation between unaligned Series will have the union of the indexes involved.
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
print(s)

print()

print(s[1:])
print(s[:-1])
print(s[1:]+s[:-1])

arr1 = np.array([1,2,3,4,5])
#arr2 = {1,2,3,4,5}
arr2 = {'india':'cricket','japan':'karate','china':'momo','france':'kiss','germany':'jews'}

#the below is applicable on for key,value paired iterables
a = pd.Series(arr2,index={'india','china','japan','usa'})
print(a)
print()

# Data Frames
ser1 = pd.Series({'name':'abhishek',
                  'item':'peanut butter',
                  'price':300})
ser2 = pd.Series({'name':'jiya',
                  'item':'chocolate',
                  'price':60})
ser3 = pd.Series({'name':'badadum',
                  'item':'bums',
                  'price':100})

df1 = pd.DataFrame([ser1,ser2,ser3],index=['store1','store1','store2'])
print(df1)
#here u have to use the loc attribute
print(df1.loc['store1'])
print(type(df1))
print()
#.loc can take row and column index
print(df1.loc[:,['item','price']])
print()
#transpose
print(df1.T)
print()
#drop row
#df1.drop['store1']
#it wont delete the original dataframe on displays a copy of it
# if u want to drop take a copy and drop
copy_df = df1.copy()
#assigning to original df after dropping
copy_df = copy_df.drop('store1')
print(copy_df)
#.drop has parameter inplace if its tru the df will be updated instead of just displaying a copy
#second is axis which is set to 0 meaning row we can set it ti 1 for column
#or simply use del
copy_df1 = df1.copy()
print(df1)
#takes column to be deleted
del copy_df1['name']
print(copy_df1)
print(copy_df1.item)
print()
print(df1)
df1['location'] = None
print()
print(df1)

####-------------------------

df = pd.read_csv(r'C:\Users\avishek\Downloads\oly.csv')
df = pd.read_csv(r'C:\Users\avishek\Downloads\oly.csv',index_col=0,skiprows=1)

#its what it says
#urllib.request.urlretrieve('https://github.com/irJERAD/Intro-to-Data-Science-in-Python/blob/master/MyNotebooks/olympics.csv','oly.csv')
print(df.head())
print()
#print(df.keys())
print()
#print(df.columns)# same as above
#below removing unwanted things from index
for i in df.index:
    df.rename(index={i:i.split(' ')[0]},inplace=True)
print()
print(df.index)
print()
for i in df.columns:
    if i[:1]=='?':
        df.rename(columns={i:i.split(' ')[1]}, inplace=True)
    elif i[:2]=='01':
        df.rename(columns={i:'Gold'+i[4:]}, inplace=True)
    elif i[:2]=='02':
        df.rename(columns={i:'Silver'+i[4:]}, inplace=True)
    elif i[:2]=='03':
        df.rename(columns={i:'Bronze'+i[4:]}, inplace=True)
print()
# above we are using i[4:] for that run second  pd.read_csv and u will know
print(df.columns)
print()
#bollean masking only countries with gold in summer olympics
print(df['Gold']>0)
print('--------------------')
only_gold = df.where(df['Gold']>0)
print(only_gold.head())
print(only_gold['Gold'].count())#coount ignores NaN
only_gold = only_gold.dropna()#drop rows with nodata
print()
print(only_gold.head())
# we can write the above thing as
only_gold = df[df['Gold']>0] # here the NaN rows are automatically removed
print()
print(only_gold.head())
# also we can use 'and' and 'or' while boolean masking
only_gold_silver = df[(df['Gold']>5) & (df['Silver']>5)]
print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
print(only_gold_silver)

###---------------

df = pd.read_csv(r'C:\Users\avishek\Downloads\oly.csv')
df = pd.read_csv(r'C:\Users\avishek\Downloads\oly.csv',index_col=0,skiprows=1)

#its what it says
#urllib.request.urlretrieve('https://github.com/irJERAD/Intro-to-Data-Science-in-Python/blob/master/MyNotebooks/olympics.csv','oly.csv')
print(df.head())
print()
#print(df.keys())
print()
#print(df.columns)# same as above
#below removing unwanted things from index
for i in df.index:
    df.rename(index={i:i.split(' ')[0]},inplace=True)
print()
print(df.index)
print()
for i in df.columns:
    if i[:1]=='?':
        df.rename(columns={i:i.split(' ')[1]}, inplace=True)
    elif i[:2]=='01':
        df.rename(columns={i:'Gold'+i[4:]}, inplace=True)
    elif i[:2]=='02':
        df.rename(columns={i:'Silver'+i[4:]}, inplace=True)
    elif i[:2]=='03':
        df.rename(columns={i:'Bronze'+i[4:]}, inplace=True)
print()
# above we are using i[4:] for that run second  pd.read_csv and u will know
print(df.columns)
print()
#setting summer gold wins as index
df['country'] = df.index # assigning the index data to new column 'country' so not to get lost
df = df.set_index('Gold')
print(df.head())
# as u can see the after the above print a blank row after 'Gold' header
#to get rid of that the below code is
df = df.reset_index()
print()
print(df.head())
#multilevel indexing (similar to composite keys in RDBMS)

#-----------------

import numpy as np
import pandas as pd
import urllib.request

df = pd.read_csv(r'C:\Users\avishek\Downloads\census.csv')
print(df.head())
#distinct
print('------------------')
print(df['SUMLEV'].unique())
df = df[df['SUMLEV'] == 50]
print()
print(df.head())
#print(df.columns)
print()
columns_to_keep = [
       'STNAME',
       'CTYNAME',
       'POPESTIMATE2010',
       'POPESTIMATE2011',
       'POPESTIMATE2012',
       'POPESTIMATE2013',
       'POPESTIMATE2014',
       'POPESTIMATE2015',
       'BIRTHS2010',
       'BIRTHS2011',
       'BIRTHS2012',
       'BIRTHS2013',
       'BIRTHS2014',
       'BIRTHS2015'
]

df = df[columns_to_keep]
print(df.head())
df = df.set_index(['STNAME', 'CTYNAME'])
print()
#df = df.reset_index()#cause of the blank line we get when we set index ** still to understand
print(df.head())
print('-----------------')
print(df.loc['Michigan', 'Washtenaw County'])
#for multiple
print('---------')
print(df.loc[[('Michigan', 'Washtenaw County'),('Michigan', 'Wayne County')]])


#df.fillna for NaN filling
df.sort_index() # for sorting index



# ser1 = pd.Series({'name':'abhishek',
#                   'item':'peanut butter',
#                   'price':30.50})
# ser2 = pd.Series({'name':'jiya',
#                   'item':'chocolate',
#                   'price':60.20})
# ser3 = pd.Series({'name':'badadum',
#                   'item':'bums',
#                   'price':100.00})

df = pd.DataFrame([{'name':'abhishek',
                  'item':'peanut butter',
                  'price':30.50},{'name':'jiya',
                  'item':'chocolate',
                  'price':60.20},{'name':'badadum',
                  'item':'bums',
                  'price':100.00}],index=['store1','store1','store2'])

print(df)
print()
df['Date'] = ['december 1', 'January 1', 'march 1']
print(df)
print()
df['Delievered'] = True
print(df)
print()
df['feedback'] = ['Positive', None, 'Negative']
#df['feedback'] = pd.Series({'Store1' : 'Positive' , 'Store2' : 'Negative'})
print(df)
print()

adf = df.reset_index()
print(adf)
adf['Date'] = pd.Series({0 : 'December', 2 : 'January'})
print()
print(adf)

#------------ joins ---

staff_df = pd.DataFrame([{'name':'abhishek', 'Role':'Director'},
                   {'name':'jiya', 'Role':'Course teacher'},
                   {'name':'badadum', 'Role':'Grader'}])

staff_df = staff_df.set_index('name')
student_df = pd.DataFrame([{'name':'abhishek', 'School':'Business'},
                   {'name':'jiya', 'School':'Law'},
                   {'name':'janam', 'School':'Engineering'}])
student_df = student_df.set_index('name')

#outer join or union
print(pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True))
print()
#inner join or intersection
print(pd.merge(staff_df, student_df, how='inner' , left_index=True, right_index=True))

# same we can do how='left' or how='right'
# also if u want to join on columns instead of righ and left_index
# u write right_on and left_on and colimn name like right_on = 'name'
# also if suppose both dataframes have a common column names location
# and we join on names then in the output we will get location_x and location_y ,
# x for the left df and y for the right df


#----------------------

print(df.head())
#method chaining
print(df.where(df['SUMLEV'] == 50).\
    dropna().\
    set_index(['STNAME', 'CTYNAME']).\
    rename(columns={'ESTIMATESBASE2010' : 'Estimates Base 2010'}).head())

# also check applymap

#---------------- apply -------------

df = pd.read_csv(r'C:\Users\avishek\Downloads\census.csv')
#print(df.columns)
#print(df.index)
columns_to_keep = [
       'STNAME',
       'CTYNAME',
       'POPESTIMATE2010',
       'POPESTIMATE2011',
       'POPESTIMATE2012',
       'POPESTIMATE2013',
       'POPESTIMATE2014',
       'POPESTIMATE2015',
       'BIRTHS2010',
       'BIRTHS2011',
       'BIRTHS2012',
       'BIRTHS2013',
       'BIRTHS2014',
       'BIRTHS2015'
]

df = df[columns_to_keep]
#print(df.head())
df = df.set_index(['STNAME', 'CTYNAME'])
def min_max(row):
    data = row[[
        'POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015'
    ]]
    row['max'] = np.max(data)
    row['min'] = np.min(data)
    return row
#    return pd.Series({'min':np.min(data), 'max':np.max(data)})

#print(min_max(df).head())
print()
print(df.apply(min_max, axis=1).head())
print()
# similar kinda code with lambda

row = [ 'POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015' ]
print(df.apply(lambda x : np.max(x[row]), axis=1).head())

#-------------------- group by --------------------------

#print(df.columns)
#print(df.index)
df = df[df['SUMLEV'] == 50]
#print(df.head())

for state in df['STNAME'].unique():
    average = np.average(df.where(df['STNAME'] == state).dropna()['CENSUS2010POP'])
    #or
    #average = np.average(df[df['STNAME'] == state]['CENSUS2010POP'])
    print('countries in state',state,'have an average of',str(average))
print('---------------------------------')
#same code with groupby
for group, frame in df.groupby('STNAME'):
    avg = np.average(frame['CENSUS2010POP'])
    print('countries in state',group,'have an average of',str(average))

#-----

df = df.set_index('STNAME')

def fun(item):
    if item[0] < 'M':
        return 0
    if item[0] < 'Q':
        return 1
    return 2

for group, frame in df.groupby(fun):
    print('there are ',str(len(frame)),'records in the group',str(group))

#-----


df = pd.read_csv(r'C:\Users\avishek\Downloads\census.csv')

df = df[df['SUMLEV'] == 50]
print(df.groupby('STNAME').agg({'CENSUS2010POP' : np.average}))
#groupby(level = 0) means index in a single index d

#using apply instead of agg
#print(df.groupby('Category').apply(lambda x : np.sum(x['Weight (oz.)'])))

#print(df.groupby('Category').apply(lambda df,a,b: sum(df[a] * df[b]), 'Weight (oz.)', 'Quantity'))
#here weight and quantity are addition string parameters passed
#same code as above without lambda
# Or alternatively without using a lambda:
# def totalweight(df, w, q):
#        return sum(df[w] * df[q])
#        
# print(df.groupby('Category').apply(totalweight, 'Weight (oz.)', 'Quantity'))


df = pd.read_csv(r'C:\Users\avishek\Downloads\census.csv')

df = df[df['SUMLEV'] == 50]
print(df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({'sum':np.sum, 'avg':np.average}))

#-------------scaled

df = pd.DataFrame(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D','D-'],
                  index=['excellent','excellent','excellent',
                         'good','good','good',
                         'ok','ok','ok',
                         'poor','poor','poor'])

df.rename(columns={0:'Grades'},inplace=True)
#print(df)
print()

# render above data as categorical data
print(df['Grades'].astype('category'))
print()
#below we are giving orders like d < c < b <a
grades = df['Grades'].astype('category',
                             categories = ['D-','D','D+','C-','C','C+','B-','B','B+','A-','A','A+'],
                             ordered = True)
print(grades > 'C')

# check pd.DataFrame.cut()
# pd.cut(s, 3, labels=['Small', 'Medium', 'Large'])

#------ pivot


df = pd.read_csv(r'cars1.csv')
print(df.head())
print()
#pivot table
print(df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean))
print('---------')
print(df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean,np.min], margins=True))

#print(pd.pivot_table(Bikes, index=['Manufacturer','Bike Type']))
#print(Bikes.pivot_table(index=['Manufacturer','Bike Type']))
#both the above statements lead to


# pandas timestamps


#timestamp
print(pd.Timestamp('9/1/16 10:05AM'))
print()
#period
print(pd.Period('1/2016'))
print()
print(pd.Period('3/5/2016'))

#datetimeindex
print()
t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
print(t1)
print()
print(type(t1.index))

#period index
print()
t2 = pd.Series(list('def'), [pd.Period('2016-10'), pd.Period('2016-11'), pd.Period('2016-12')])
print(t2)
print()
print(type(t2.index))

#convert to datetime

print()
d1 = ['2 june 2016', 'Aug 29, 2014', '2015-06-26', '7/12/16']
t3 = pd.DataFrame(np.random.randint(10,100, (4,2)), index=d1, columns=list('ab'))
print(t3)

t3.index = pd.to_datetime(t3.index)
print()
print(t3)
print()
print(pd.to_datetime('2012-02-10', dayfirst=True))
print()

#timedeltas
print(pd.Timestamp('2016-09-02')-pd.Timestamp('2016-09-04'))
print(pd.Timestamp('2016-09-02')+pd.Timedelta('12D 3H'))
print()


#dates in Dataframe
dates = pd.date_range('10-01-2016', periods=9, freq='2W-SUN')
print(dates)
print()
df = pd.DataFrame({'Count1' : 100 + np.random.randint(-5,10,9).cumsum(),
                   'Count2': 100 + np.random.randint(-5, 10, 9)}, index=dates)
print(df)
print()
print(df.diff())
print()
#finding mean count for each month in above df ,,
print(df.resample('M').mean())
#Convenience method for frequency conversion and resampling of time series.
# Object must have a datetime-like index (DatetimeIndex, PeriodIndex,
# or TimedeltaIndex), or pass datetime-like values to the on or level keyword.


#changing frequency using asfreq
print(df.asfreq('W', method='ffill')) #forward fill
print()

#plotting 
df.plot()
plt.show()

#---------- distribution

#binomial
print(np.random.binomial(1, 0.5))
# 1 means the no. of times and 0.5 means 50% chance of getting a one or zero
print()

#now running the simulation 1000 times and dividing the result by 1000
print(np.random.binomial(1000, 0.5)/1000)
#as we can see the result is close to 0.5

#something to look at
x = np.random.binomial(20, .5, 10000)
#print(x>=15)
print((x>=15).mean(),' ',np.count_nonzero(x>=15))

#chances of back to back tornado
print()
chance_of_tornado = 0.01
tornado_event = np.random.binomial(1, chance_of_tornado, 1000000)
two_days_in_a_row = 0

for i in range(1, len(tornado_event)-1):
    if tornado_event[i] == 1 and tornado_event[i-1] == 1:
        two_days_in_a_row += 1

print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))

#---

import scipy.stats as stats

#normal

d = np.random.normal(0.75, size=1000)# 0.75 expexted value and sd = 1
print(np.sqrt(np.sum((np.mean(d)-d)**2)/len(d)))
#above is the formula for SD also below one is the same
print(np.std(d))

print(stats.kurtosis(d))
print(stats.skew(d))


#--------------- matplotlib -------------------------------------------

#---histogram--------

np.random.seed(100)
age = list(np.random.randint(20,100) for i in range(100))
ids = [i for i in range(100)]
plt.figure(1)
plt.subplot(121)
bins = [i for i in range(10,101,5)]
plt.hist(age, bins=bins, histtype='bar', rwidth=0.9)
#print(age)
plt.subplot(122)
plt.hist(age, bins=20, histtype='bar', rwidth=0.9)
plt.xlabel('x')
plt.ylabel('y')
plt.title('some chart')
plt.legend()
plt.show()

#------------ acatter -------
np.random.seed(100)
age = list(np.random.randint(20,100) for i in range(100))
toys = list(np.random.randint(0,10) for i in range(100))
plt.scatter(age, toys, label='scatter', color='k', marker='*', s=100)
plt.xlabel('age')
plt.ylabel('toys')
plt.title('some chart')
plt.legend()
plt.show()

#----------- fake legends ----------------------

plt.plot([],[], color='c', label='A')
plt.plot([],[], color='r', label='B')
plt.plot([],[], color='b', label='C')
plt.plot([],[], color='k', label='D')
plt.legend()
plt.show()



