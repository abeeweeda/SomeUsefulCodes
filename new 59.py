
#3D array (two stacked 2D arrays)
g = np.array( [[[   0,  1,  2],     
                 [ 10, 12, 13]],
                [[100,101,102],
                 [110,112,113]]])
print()
print(g)
print()
print(g[1,:,:])#we can think of this as 2 2d vertically stacked array so g[1] will give the 2nd 2d array
print()
print(g[...,2])#from both arrays take 3rd column
print()

#mean vs average
#np.mean will return simple mean, but in np.average you can give additional parameters for weights

#iterating over 3d array
for row in g:#two loops over two 2d arrays
    print(row)
	
#flatten the arrays
for i in g.flat:#flats the array, the g.flat creates an iterator object
    print(i)
	
for i in g.flatten():#.flatten() creates a new array object
    print(i)

g.ravel()#similar to flatten but if you modify the ravel object it will modify the original array
g.reshape((-1,)) or g.reshape((-1)) #or we can simply reshape

#size
g.size #gives you number of element in an array its simply row * columns * 3rd dim (if any) * so on
np.resize(g, g.size)#we can use this also to flatten an array

#transpose
g.T
np.transpose(g)#same as above but with added parametes to decide axes

#stacking
a=[1,2,3]
b=[4,5,6]
print(np.column_stack([a,b]))
print('----------')
print(np.vstack([a,b]))
print('----------')
print(np.hstack([a,b]))
print('----------')
print(np.stack([a,b], axis=-1))
print('----------')
print(np.concatenate((a, b), axis=0))
#also try the above with 2d arrays, also in the above functions we ca write like this np.vstack((a,b)), () instead of []

#to add dimension
a = np.array([1,2,3,4])
a[:,np.newaxis]#this will convert the above 1d array to 2d array

#np.r_('a,b,c',arr1,arr2) (it is row wise merging, axis 0)
#a = axis to concatenate along
#b = the minimum number of dimensions to force the entries to
#c = which axis should contain the start of the arrays
m = np.array([[0, 1, 2], [3, 4, 5]])
print(np.r_['-1',m,m]) #this is same as np.hstack((m,m))
#similar we have np.c_ (column wise merging, axis 1)

#flooring
np.floor(somedecimalarray)
np.ceil(somedecimalarray)

#split arrays
print(np.hsplit(m,3))#horizontal split into 3
print()
print(np.hsplit(m,(3,4)))   # Split m after the third and the fourth column
#0 to 2 one matrix , 3 one matrix , 4 to last one matrix
#vsplit splits along the vertical axis, and array_split allows one to specify along which axis to split.

#unique identifier
id(m) #will give you one unique number.
n=m #now id(m) will be equal to id(n)

#view
z = a.view()
print(z)
print(z is a)#false as z and a are diferent
print(z.base is a)#true as z is view of data owned by a
#if you change z's data it will also change a's data

#use other array as indices
a = np.arange(12)**2                       # the first 12 square numbers
print(a)
print()
i = np.array( [ 1,1,3,8,5 ] )              # an array of indices
print(a[i])                                # the elements of a at the positions i
print()
j = np.array( [ [ 3, 4], [ 9, 7 ] ] )      # a bidimensional array of indices
print(a[j])                                # the same shape as j

#masking
a = np.arange(12).reshape(3,4)
print(a)
print()
b = [False,True,True]
print(a[b])
print()
print(a[[True,True,False]])

#indexing multi dim arrays
y = np.arange(35).reshape(5,7)
y[[0,2,4],[0,1,2]]#this means give me numbers which are at [0,0],[2,1] and [4,2]
# for more visit https://docs.scipy.org/doc/numpy/user/basics.indexing.html

#full
np.full((2,2),3)#creates a 2d array size 2,2 with all elements as 3

#type conversion
a.astype(float32)

#sorting
a.sort()#sorts a and if you now print a it will be sorted

#I/O operation, saving loading
#from numpy binary files
np.load(file[, mmap_mode, allow_pickle, …])	Load arrays or pickled objects from .npy, .npz or pickled files.
np.save(file, arr[, allow_pickle, fix_imports])	Save an array to a binary file in NumPy .npy format.
np.savez(file, *args, **kwds)	Save several arrays into a single file in uncompressed .npz format.
np.savez_compressed(file, *args, **kwds)	Save several arrays into a single file in compressed .npz format.

#from text files
loadtxt(fname[, dtype, comments, delimiter, …])	Load data from a text file.
savetxt(fname, X[, fmt, delimiter, newline, …])	Save an array to a text file.
genfromtxt(fname[, dtype, comments, …])	Load data from a text file, with missing values handled as specified.
fromregex(file, regexp, dtype[, encoding])	Construct an array from a text file, using regular expression parsing.
fromstring(string[, dtype, count, sep])	A new 1-D array initialized from text data in a string.
ndarray.tofile(fid[, sep, format])	Write array to a file as text or binary (default).
ndarray.tolist()	Return the array as a (possibly nested) list.

#** .copy() operation is called as deep copying

#The term broadcasting describes how numpy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes


######################## pandas ####################################

#series
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
print(s)
#we can index series in a same way as numpy array, a[1:3] will give 2nd and 3rd row, a[::-1] will print in reverse

#key value pair
arr2 = {'india':'cricket','japan':'karate','china':'momo','france':'kiss','germany':'jews'}
print(a)#this will print india, japan etc as index and cricket, karate etc as values
#the below is applicable on for key,value paired iterables
a = pd.Series(arr2,index={'india','china','japan','usa'})
print(a)#this will print only the indexed india, china, japan, for usa as we dont have any value so null will get printed

#dataframe
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
print(df1)#it will have name, item, price as columns
#or you can define as below
df1 = pd.DataFrame({'name':['abhishek','jiya','badadum'],'item':['peanut butter','chocolate','bums'],'price':[300,60,100]},index=['store1','store1','store2'])
#or you can do like this
df = pd.DataFrame([{'name':'abhishek','item':'peanut butter','price':300},
                   {'name':'jiya','item':'chocolate','price':60},
                   {'name':'badadum','item':'bums','price':100}],
                  index=['store1','store1','store2'])#list of dictionaries
#you can add new columns and data as simple
df['Date'] = ['december 1', 'January 1', 'march 1']#add column called date with given values
df['feedback'] = ['Positive', None, 'Negative']#other column showing you can add None values also
df['Delievered'] = True#this will create a column named Delievered and all the values of that column will be True
df['newcol'] = pd.Series({'store2' : 'someData'})#this will create a column newcol and value for index store2 will be someData and rest values will be NaN's

#indexing
df1.loc['store1']#print 1st and 2nd row
df1.iloc[0]#will print the first row but as a series
df1.iloc[0:1]#prints first row in dataframe format similarly we can do df1.iloc[:]
df1.loc[:,['item','price']]#all rows and only item and price column
df1.loc['store1':'store2']#it will output all the rows whose index is store1 till the row whose index is store2 including both in the same order as dataframe


#transposing
df1.T#the indexes will become columns and vice verse

#delete remove or drop a column or row
df1.drop('item', axis=1)#drops column original DF remains unchanged, you can use parameter inplace=True if you want to change the original DF or assign this to new variable
#** remember try to use .copy() for storing the new DF unless you don't mind the original DF getting changed
df1.drop('store1')#drops rows original DF remains unchanged

df2 = df1.copy()#made a copy to made sure the df1 is not affected
del df2['price'] #del is a common python keyword to remove any object, its not specific to pandas, in pandas we have drop as seen above


#read csv excel and common things
df = pd.read_csv(r'C:\Users\avishek\Downloads\oly.csv',index_col=0,skiprows=1)
#here we are reading a csv file, where first column 0th index we are using as index and we are asking to skip 1 row as it might be not required
df.head()#get first 5 rows of DF simiarly we have df.tail()
df.keys()#get the column value as indexes
df.columns#same as above
df.columns.values#get the column values but in array format
df.index#will give you all the index values
#renaming indexes by removing the things after space e.g. if index is 'abc def' then we will get abc
for i in df.index:
    df.rename(index={i:i.split(' ')[0]}, inplace=True)
#similar we can do this for column names, if they are not proper

#masking
df.SomeColumn>0 #it will create a series of boolean values which we can use to filter dataframe
df[df.SomeColumn>0]#this will give only those rows where value of SomeColumn is > 0
df.where(df.SomeColumn>0)#this will make all the columns of all the rows NaN's where SomeColumn is <=0
#the above NaN's data you can drop using dropna() function
df.where(df.SomeColumn>0).dropna()#in drop na we have many parameters to set depending on what you want to drop
df.where(df.SomeColumn>0, 10)#this will make all the columns of all the rows as 10 where SomeColumn is <=0
df.where(df.SomeColumn>0)['SomeColumn'].count()#counts the not null values from SomeColumn
df[(df['col1']>5) & (df['col2']>5)]#also you can use multiple conditions with & or |, just remember to put paranthesis between conditions

#appending data / rows to dataframe
df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
df = df.append(df2)

#remove duplicate values
df.drop_duplicates(inplace=True)

#get null count from each column
df.apply(lambda x: sum(x.isnull()))

#create a 2d numpy array from dataframe
df.values

#value counts
df['SomeColumn'].value_counts() #this will give the value count of each of the distinct value in SomeColumn
df.groupby('SomeColumn').size() #this will give same result as above with sorted index
df.Gold.value_counts().sort_index() #this will give same result as above

#sorting values
#sorting index we have seen above
df.sort_values(by=['col1'])#sort by col1
df.sort_values(by=['col1','col2'])#sort by col1 then within col1 sort by col2

#setting value of other columns after filtering on some other column
df.loc[df.AAA >= 5, 'BBB'] = -1#here it will set the value of column BBB to -1 where column AAA>=5
df.loc[df.AAA >= 5, ['BBB', 'CCC']] = 555#here similarly we will set values of two columns based on other column

#the above code can be also use for filtering
df.loc[df.AAA >= 5, 'BBB']#give the values of column BBB where we have column AAA>=5

#masking using other dataframe
df_mask = pd.DataFrame({'AAA': [True] * 4,
                        'BBB': [False] * 4,
                        'CCC': [True, False] * 2})
df.where(df_mask, -1000)
#using the above mask we are setting the values of df dataframe to -1000 when False(as np.where sets the value to nan where false and replaces with the parameter values )

#changing index
df.reset_index()#create one more column called index which will have index values and the index will become 0,1,2,3.. if you want to keep this change you can use inplace=True
#you can do df = df.reset_index()
df.set_index('col1')#sets col1 as index , and you can assign or do inplace=True as above to keep the changes
df['indexcol'] = df.index#just in case you are making some index changes and you want to go back to old index, you can create a column where you can keep the indexes
#also you can have multilevel indexing

#get unique values
df.col1.unique()#return the unique values of col1
df.col1.nunique()#return the number of unique values of col1, similar to len(df.col1.unique())

#join or merge
#outer join or union
print(pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True))
#inner join or intersection
print(pd.merge(staff_df, student_df, how='inner' , left_index=True, right_index=True))
#usually if wanna join on index as above we make the primary key column as or the column we wanna join on index or we can do as below
# same we can do how='left' or how='right'
# also if u want to join on columns instead of right and left_index
# u write right_on and left_on and column name like right_on = 'name'
# always we should have one from left_index or left_on and similarly right_index or right_on
# also if suppose both dataframes have a common column names location
# and we join on names then in the output we will get location_x and location_y ,
# x for the left df and y for the right df

#method chaining
df.where(df['col4'] == 50).dropna().set_index(['col0', 'col1']).rename(columns={'col2' : 'col3'}).head()
 
#mapping
mapp = {True:'TT', False:'FF'}#create a dict
df['NewTestCol'] = df['NewTestCol'].map(mapp)#it replace False with FF and True with TT for column NewTestCol

#apply
def someFunc(row):
    data = row['col2']+row['col3']#col2 and col3 are some string columns so we are concatenating them
    return data
df.apply(someFunc, axis=1).head()#applying that someFunc to the dataframe
#another use is to count the nulls
df.apply(lambda x: sum(x.isnull()))#by default the axis = 0, which means it will count the null for each column and return


#group by
df.groupby('YEAR')['someNumericColumn'].agg({'sum':np.sum, 'avg':np.average})#creates two column one sum other avg grouped by YEAR for column someNumericColumn
df.groupby('YEAR')['someNumericColumn'].sum()#simple group by then sum
#also you can write your own functions or use lambda function using apply like below
df1.groupby('index')['item'].apply(lambda x : '-'.join(x))#here we are concatenating valus of column item after grouping on column index
#also you can simply group by on a column and then it creates a grouped object which you can iterate through like below
for group, frame in df.groupby(fun):
    print('there are ',str(len(frame)),'records in the group',str(group))

#datatype conversion
#similar to np array here also we can use astype to convert to different datatype
#we can also use astype to convert to ordered categorical datatype

#bin or cut
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3)#cuts the range 1 to 7 into three different parts and assign the values to those bins
pd.qcut(np.array([1, 7, 5, 4, 6, 3]), 3)#cuts into 3 different regions and each cut will have equal number of elements 6/3 = 2 elements each
#also you can define your own bins
bins = [0, 1, 5, 10, 25, 50, 100]#here the bins will be (0, 1] < (1, 5] < (5, 10] < (10, 25] < (25, 50] < (50, 100]
pd.cut(np.array([46.50,44.20,100.00,42.12]), bins)
#also you can label your bins
bins = [0, 1, 5, 10, 25, 50, 100]
labels = ['not ok','ok','fine','good','very good','excellent']#as we have 7 elements which means 6 bins
pd.cut(np.array([46.50,44.20,100.00,42.12]), bins=bins, labels=labels)

#diff
df.diff()#subtracts elements rowwise, for columnwise put axis=1, also you can give periods

#pivot table
df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)#self explanatory
#similary we can have multiple aggregate functions like aggfunc=[np.mean,np.min]

#crosstab, here the data can be any array or series it does not have to be dataframe
pd.crosstab(rows, columns, rownames=['someName'], colnames=['someName'])
#example, also pass numpy arrays or series, not lists
a = np.array([1,2,3])
b = np.array([3,4,5])
pd.crosstab(a,b)

#pandas date time
#convert to date time
d1 = ['2 june 2016', 'Aug 29, 2014', '2015-06-26', '7/12/16']
t3 = pd.DataFrame(np.random.randint(10,100, (4,2)), index=d1, columns=list('ab'))
t3.index = pd.to_datetime(t3.index)#using pd.to_datetime to convert to datetime

#adding subtracting times
pd.Timestamp('2016-09-02')-pd.Timestamp('2016-09-04')
pd.Timestamp('2016-09-02')+pd.Timedelta('12D 3H')
pd.Timestamp('2016-09-02')+pd.DateOffset(minutes=60)

#creating date range
pd.date_range('12-APR-2019', periods=9, freq='2W-SUN')
#it will create a date range of 9 dates starting from first sunday after or on 12-Apr-19 will 2 weeks difference 
#e.g. 14th is the first sunday after 12th so first date will be 14th the second date will be 2 weeks later i.e. 28th and so on

#resample
dates = pd.date_range('12-APR-2019', periods=9, freq='2W-SUN')
df = pd.DataFrame({'Count1' : 100 + np.random.randint(-5,10,9).cumsum(),
                   'Count2': 100 + np.random.randint(-5, 10, 9)}, index=dates)#creating a dataframe with datetime index
df.resample('M').mean()#this code will kinda group data on month and then take mean of it

#changing frequency
df.asfreq('W', method='ffill')#this code will update the index of the previos dataframe from byweekly to weekly and to popualte the values for
#the new indices we use forward fill ffill which will just copy the previous value to the next newly created index

#for getting year month etc from date we use .dt operator ,** .dt operator works on series
df['year'] = df['dateTimeColumn'].dt.year #this creates a year column and .dt.year extracts year from dateTimeColumn

#null not nulls
df['columns'].isnull()
df['columns'].notnull()
#you can use this like this 
df[df['col1'].isnull()]#it will return a dataframe where all the values for col1 is null

#some commmon functions
df.shape#gives the shape rows * columns
df.info()#it will column names and how many non null values and data type
df.describe()#gives numerical columns and their counts max min quantiles mean std
#for getting the detail of non numeric column
df.describe(include=['O'])#means include object type
df.describe(include=np.object)#same as above








