
#general imports
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('nbAgg')#for using it as inline in jupyter notebook
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
import seaborn as sns
import geopy.distance as geodist#for calulating geo distance based on lat lons
from scipy import stats, integrate
import os

#if we want to get file path
print(plt.__file__)#here we want to get the path of plt function

#data preprocessing , imputation, scaling
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder

Scaler = StandardScaler()
X_train_Scaled = Scaler.fit_transform(X_train)#use fit transform on training data
X_test_Scaled = Scaler.transform(X_test)#use only transform on test data
#similarly we can do for MinMaxScaler and Normalizer

#different tree based classifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


#other imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix

#confusion matrix
labels=[0,1]
cnf_mat = confusion_matrix(predctions, y_originals, labels)#List of labels to index the matrix

#feature importance which we can use for feature selection
#for tree based classifier we can get the feature importance like which of the features are important
#example
model = ExtraTreesClassifier(n_estimators=c-1,max_features=0.5,n_jobs=-1,random_state=seed)#here we defined ExtraTreesClassifier
model.fit(X_train, y_train)#we fitted data
importances = forest.feature_importances_#get all the values from feature_importances_
indices = np.argsort(importances)#we get the sorted indices, means if at zero index we have 2 which means feature 2 is most important
model.feature_importances_.shape#here we can see how many features are there in feature importance it should be equal to the fitted number of columns
f_imp = pd.Series(importances[indices], list(columns[indices]))#we can create a series of feature importance features
f_imp.plot(kind='bar', label='extratreenormal')#we can plot the feature importance to see which is more important feature
plt.legend()
plt.show()

#feature selection using RFE, recursive feature elimination
from sklearn.feature_selection import RFE
estimator = ExtraTreesClassifier(n_estimators=c-1,max_features=0.5,n_jobs=-1,random_state=seed)#here we define an estimator
selector = RFE(estimator, 0.75*(numOfCols))#here we define a selector which will use 75% of all feature which are best feature, If None half of the features are selected
selector = selector.fit(X_train, y_train)#then we use selector to fit the training data

#loading huge data using iterator object, let us say we have some million records we cant load all at once if our system is not capable
#also we want to load only certain data, so loading full data and then filtering is a overhead a better solution
data_temp=pd.read_csv(dataPath, iterator=True, chunksize=1000, usecols['col1','col2','col3'])
data_main=pd.concat([chunk[chunk.col1=2012] for chunk in data_temp])
#what we are doing above is first loading the data using iterator with 1000 rows and few columns and then concatenating the data
#1000 rows at a time also we are only loading those rows where col1=2012

#pickling
import pickle
#with is use as context manager
with open('model.pickle', 'wb') as pick:#create a model.pickle file and open it in write binary mode
    pickle.dump(classifier, pick)#dump the classifier into that file
#to load pickle
pkl_file = open('model.pickle', 'rb')
pickle.load(pkl_file)

#get unique count from each column
df.apply(lambda x : x.nunique())

#groupby on multiple cols
c = cast.groupby([cast.year // 10 * 10,'type']).size()#it will create a multilevel index
c.unstack()#we can unstack or pivot the second index a cols using unstack

#convert using astype
df.date = df.date.astype(np.datetime64)

#filling NaN's
df.fillna(' ')#here we are filling NaN's with spaces
#or we can use some method like ffill or bfill
df.col1.fillna(method='bfill')#backward fill, fills the current NaN using the next values
#usually we might miss the few values in the beginnig or last when we use only bfill or ffill, so we generally use one of bfill or ffill first then use the other

#to ignore warnings in jupyter notebook or generally
import warnings
warnings.filterwarnings('ignore')

#display all the columns in jupyter notebook while displaying data
pd.set_option('display.max_columns', None)

#list all the directories in a path
filenames = os.listdir(filepath)

#filter out the csv's from all the files
filenames = [f for f in filenames if f.endswith(".csv")]

#create correlation matrix
df.corr()

#to find skewness
df.skew()

############################ matplotlib ###########################
#** some (very few) of the below codes might not work, or you might have to make some changes based on the installation and version you are using

#to get matplotlib version
matplotlib.__version__

#making a sine plot, line plot
def sinplot(flip=1):
    x = np.linspace(0,14,100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
sinplot()#calling sinplot

**capture1

#box plot
np.random.seed(sum(map(ord, "abhishek")))#here we are just seeding as to get same plot everytime we run the code
sns.set_style('whitegrid')
data = np.random.normal(size=(20, 6)) + np.arange(6)/2#here data is having 20 rows 6 columns
sns.boxplot(data=data)

**capture2

#distplot
sns.distplot(x)

**capture3

#count plot
sns.countplot(x='Survived', data=trainData, hue='Sex', palette='Blues_r')

**capture4

#simple histogram
trainData.Age.plot(kind='hist')#histogram of Age column from trainData

**capture5

#sns Facetgrid
g = sns.FacetGrid(trainData, col='Survived')
g.map(sns.distplot, 'Age', bins=20)#here we are creating distplot with 20 bins for column Age based on column Survived, column survived is having 2 values 0 and 100

**capture6

#sns pairplot or scatterplot
sns.pairplot(dataset, size=6, hue='column_To_Color',  x_vars='X_col',y_vars='y_col')
plt.show()

**capture7

#basic bar chart
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]
plt.bar(x, y, label = 'x')
plt.title('bar chart')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

**capture8

#basic scatter plot
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]
plt.scatter([],[], marker = '*', s = 50, label = 'x', color ='r')#to display small legend as for size=150(next line) the legend will be very big
plt.scatter(x, y, marker = '*', s = 150, color = 'r')
plt.title('bar chart')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

**capture9

#basic histogram
import numpy as np
x = np.random.randint(10, 100, 100)
bins = [n for n in np.arange(10, 110, 10)]
print(bins)
plt.hist(x, bins=bins, histtype='bar', rwidth=0.8)

**capture10

#basic stackplot
days = [1, 2, 3, 4, 5]
sleeping = [6, 5, 4, 8, 12]
eating = [3, 6, 3, 4, 2]
playing = [5, 3, 2, 4, 3]
working = [10, 10, 15, 8, 7]

plt.stackplot(days, sleeping, eating, playing, working, 
              colors = ['m', 'c', 'r', 'b'])
#for fake legends
plt.plot([],[], label = 'sleep', color='m', linewidth=5)
plt.plot([],[], label = 'eat', color='c', linewidth=5)
plt.plot([],[], label = 'play', color='r', linewidth=5)
plt.plot([],[], label = 'work', color='b', linewidth=5)
plt.legend()

**capture11

#pie chart with popped out slice
#pie chart with popped out slice
slice = np.array([7, 2, 2, 13])
activities = ['sleep', 'eat', 'play', 'work']

def my_autopct(pct, slice):#this function is used to display values
    total = slice.sum()
    val = pct*total/100
    return "{:.1f}% ({:d})".format(pct, val)

plt.pie(slice, labels=activities, 
        colors = ['m', 'c', 'r', 'b'], 
        startangle=90,
        shadow=True, 
        explode=(0, 0.1, 0, 0),#for which slice to pop out
        autopct=lambda pct: func(pct, slice),
        textprops=dict(color="w"))
        #autopct = '%1.1f%%'
plt.show()

**capture12

#plot date
plt.plot_date(date, closep, '-')
plt.show()

**capture13

#defining subplot and figure and axis rotation
fig = plt.figure(1)#here we define a figure with number 1
ax1 = plt.subplot2grid((1,1),(0,0))#here we define a grid of size 1*1 and ax1 plot will be at position 0,0, as we have only 1*1 space we only have one location 0,0
ax1.plot_date(date, closep, '-')
plt.xlabel('date')
plt.ylabel('price')
plt.title('stocks')
plt.subplots_adjust(left=0.09, bottom=.2)
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(45)#for rotating x ticklabels
ax1.grid(True, color = 'g', linestyle='-')#for grid
plt.show()


**capture14

#formatting plot using fill between and other formattings
fig = plt.figure(1)#here we define a figure with number 1
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.plot_date(date, closep, '-', label='price')
ax1.fill_between(date, closep[0], closep, where=closep>closep[0], color='g', alpha=0.3)
ax1.fill_between(date, closep[0], closep, where=closep<closep[0], color='r', alpha=0.3)
ax1.plot([], [], linewidth=5, label='loss', color='r', alpha=0.5)
ax1.plot([], [], linewidth=5, label='gain', color='g', alpha=0.5)
plt.xlabel('date')
plt.ylabel('price')
ax1.xaxis.label.set_color('c')#label coloring
ax1.yaxis.label.set_color('r')
ax1.set_yticks([0, 125, 250, 375, 600])#y points

plt.title('stocks')
plt.subplots_adjust(left=0.10, bottom=.2)
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(45)
ax1.grid(True, color='g', linestyle='--')
ax1.axhline(closep[0], color='k', linewidth=3)#for horizontal line at close[0]
ax1.spines['left'].set_color('c')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_linewidth(3)
plt.legend()
plt.show()

**capture15

#for finance plots we have candlestic plot
from matplotlib.finance import candlestick_ohlc#this is deprecated so you can use mpl_finance instead

**capture16

#different plot styles
from matplotlib import style
style.use('ggplot')
style.use('fivethirtyeight')
style.use('dark_background')
#we can use many more
print(plt.style.available)#you can get all the available styles from here

#live plot, animated plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open(r'C:\Users\avishek\Downloads\DataFiles\live_data.txt','r').read()#this data in the source file we can keep changing and the plot will keep updating
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:#to check if line has some data and is not null
            x, y = line.split(',')
            xs.append(x)
            ys.append(y)
    ax1.clear()#clear the previously draw plot
    ax1.plot(xs, ys)

ani = animation.FuncAnimation(fig, animate, interval=1000)#its like call this function after every second
plt.show()

#annotation in plots
fig, ax = plt.subplots()#define subplots it returns figure and axis

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = ax.plot(t, s, lw=2)

ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
ax.set_ylim(-2, 2)#setting y axis limit, the values shown on y axis will be from -2 to 2 same we can do set_xlim
plt.show()

**capture17

#text in plots
font = {'family': 'serif','color': 'darkred','weight': 'normal','size': 16,}
df1.plot(kind='scatter',x='CITY (kWh/100 km)', y='CITY (Le/100 km)')
plt.text(16,2.6,'scatter plot', fontdict=font)
plt.show()

**capture18

#more subplot2grid
fig = plt.figure()
#here we define a plot which spans 6rows and one column
ax1 = plt.subplot2grid((6,1),(0,0), rowspan=2, colspan=1)#ax1 is defined as row 0 to row 1, this spans 2 rows
ax2 = plt.subplot2grid((6,1),(2,0), rowspan=3, colspan=1)#ax2 is defined as row 2 to row 5, this spans 3 rows
ax3 = plt.subplot2grid((6,1),(5,0), rowspan=1, colspan=1)#ax1 is defined as row 6 to row 6, this spans 1 row
ax1.plot(x, y)
ax2.plot(x, y1)
ax3.plot(x, y2)
plt.show()

**capture19

#3dplots
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x = [1,2,3,4,5,6,7,8,9,10]
y = [5,6,7,8,2,5,6,3,7,2]
z = [1,2,6,3,2,7,3,3,7,2]

ax1.plot_wireframe(x,y,z, color='k')

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()

**capture20

#figsize
fig = plt.figure(figsize=(8,8))#define a figure of size 8, 8
ax = fig.gca()#getting the current axis 
data[['engine-size','fuel-type']].boxplot(by = 'fuel-type', ax=ax)

#violin plot
sns.violinplot(x='fuel-type', y='engine-size', data=aData)

**capture21

#linear model plot lm plot
sns.lmplot(x='city-mpg', y='price', data=aData, hue='fuel-type', palette='Set2')

**capture22

#joint plot
sns.jointplot('engine-size', 'price', data=aData, alpha=0.3)

**capture23

#pairplot
#when not mentioned any x or y it will create pairs of all the numeric columns and on diagonal plot the histograms
import seaborn as sns; sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
g = sns.pairplot(iris)

**capture24

#subplot
x = np.random.rand(10)
y = np.random.rand(10)
z = np.sqrt(x**2 + y**2)

plt.subplot(321)#3 by 2 grid and in that 1st graph
plt.scatter(x, y, s=80, c=z, marker=">")

plt.subplot(322)#3 by 2 grid and in that 2nd graph
plt.scatter(x, y, s=80, c=z, marker=(5, 0))

verts = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1]])
plt.subplot(323)#3 by 2 grid and in that 3rd graph
plt.scatter(x, y, s=80, c=z, marker=verts)

plt.subplot(324)#3 by 2 grid and in that 4th graph
plt.scatter(x, y, s=80, c=z, marker=(5, 1))

plt.subplot(325#3 by 2 grid and in that 5th graph
plt.scatter(x, y, s=80, c=z, marker='+')

plt.subplot(326)#3 by 2 grid and in that 6th graph
plt.scatter(x, y, s=80, c=z, marker=(5, 2))
plt.show()

**capture25

#subplots
data = {'apples': 10, 'oranges': 15, 'lemons': 5, 'limes': 20}
names = list(data.keys())
values = list(data.values())

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)#also here we are saying to share the y axis so all the three plots will have a common y axis
axs[0].bar(names, values)
axs[1].scatter(names, values)
axs[2].plot(names, values)
fig.suptitle('Categorical Plotting')

**capture26

#get months from date and plot based on that
data.groupby(data['Date field'].dt.strftime('%B')).size().plot(kind = 'bar')

**capture27

#imageshow
plt.imshow(image.reshape((28,28)), cmap=plt.cm.gray_r)

**capture28

#figsize
plt.figure(figsize=(6, 4))#here 6 is width and 4 is heigth

#figaspect
fig = plt.figure(figsize=plt.figaspect(0.5))#aspect 0.5 means width is 2wice the height
ax = fig.add_subplot(111)
ax.set(xlim=[0.5, 4.5], ylim=[-2, 8], title='example', ylabel='y-axis', xlabel='x-axis')
plt.show()

**capture29

#basic subplots
fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].set(title='upper left')
ax[0,1].set(title='upper right')
ax[1,0].set(title='lower left')
ax[1,1].set(title='lower right')
#to itertate over the axes
for x in ax.flat:#ax is an array of axis
    x.set(xticks=[], yticks=[])#set xtics and ytics to blank
plt.show()

**capture30

#another subplots example
plt.style.use('classic')
fig, ax = plt.subplots(nrows=3)
fig.suptitle('maintitle')#super title
x=np.linspace(0, 10, 100)
y1, y2, y3=np.cos(x), np.cos(x+1), np.cos(x+2)
names = ['signal 1', 'signal 2', 'signal 3']
for a, y, name in zip(ax, [y1, y2, y3], names):
	a.plot(x, y)
	a.set(xticks=[], yticks=[], title=name)
plt.show()

**capture31

#formatting bar chart bar
plt.style.use('classic')
np.random.seed(1)
fig, ax = plt.subplots(ncols=2, figsize=plt.figaspect(0.5))
x = np.arange(5)
y = np.random.randn(5)

#if you want to modify the properties we can captute like below
vert_bars = ax[0].bar(x, y, color='lightblue', align='center')
horiz_bars = ax[1].barh(x, y, color='lightblue', align='center')
ax[0].axhline(0, color='gray', linewidth=2)
ax[1].axvline(0, color='gray', linewidth=2)

#try hard you will understand, just think what is in vert_bars
for bar, height in zip(vert_bars, y):#we can iterate over individual bars
	if height < 0:
		bar.set(edgecolor='darkred', color='salmon', linewidth=3)

print(vert_bars)
plt.show()

**capture32

#using fill between
fig, ax = plt.subplots()
x = np.linspace(0,10,200)
y1 = 2 * x + 1
y2 = 3 * x + 1
y_mean = 0.5 * x * np.cos(2 * x) + 2.5* x + 1.1
ax.fill_between(x, y1, y2, color='yellow')
ax.plot(x, y_mean, color='black')
plt.show()

**capture33

#you can create the same plot as above using data_obj
fig, ax = plt.subplots()

x = np.linspace(0,10,200)

data_obj = {	 'x': x, #here we define all the data to be use in  chart in form of dictionary
		'y1': 2 * x + 1,
		'y2': 3 * x + 1,
		'mean': 0.5 * x * np.cos(2 * x) + 2.5* x + 1.1
	    }

ax.fill_between('x', 'y1', 'y2', color='yellow', data=data_obj)
ax.plot('x', 'mean', color='black', data=data_obj)
plt.show()

#getting sample data and creating map and use cmap
from matplotlib.cbook import get_sample_data
data = np.load(get_sample_data(r'axes_grid/bivariate_normal.npy'))
fig, ax = plt.subplots()
im = ax.imshow(data, cmap='gist_earth')
fig.colorbar(im)#we are adding a colorbar to a figure not the axis
plt.show()

**capture34

#adding colorbar axis
np.random.seed(1)
fig, ax = plt.subplots(ncols=3, figsize=plt.figaspect(0.5))
plt.style.use('classic')

data1 = np.random.random((10,10))
data2 = 2*np.random.random((10,10))
data3 = 3*np.random.random((10,10))
fig.tight_layout() #makes the subplots fill up the figure
cax = fig.add_axes([0.25, 0.1, 0.55, 0.03])

for a,data in zip(ax, [data1, data2, data3]):
    im = a.imshow(data, vmin=0, vmax=3, interpolation='nearest')

fig.colorbar(im, cax=cax, orientation='horizontal')#here cax is colorbar axis and is defined above
plt.show()

**capture35

#markers in matplotlib
xs, ys = np.mgrid[:4, 9:0:-1]
markers = [".", "+", ",", "x", "o", "D", "d", "", "8", "s", "p", "*", "|", "_", "h", "H", 0, 4, "<", "3",
           1, 5, ">", "4", 2, 6, "^", "2", 3, 7, "v", "1", "None", None, " ", ""]
descripts = ["point", "plus", "pixel", "cross", "circle", "diamond", "thin diamond", "",
             "octagon", "square", "pentagon", "star", "vertical bar", "horizontal bar", "hexagon 1", "hexagon 2",
             "tick left", "caret left", "triangle left", "tri left", "tick right", "caret right", "triangle right", "tri right",
             "tick up", "caret up", "triangle up", "tri up", "tick down", "caret down", "triangle down", "tri down",
             "Nothing", "Nothing", "Nothing", "Nothing"]
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for x, y, m, d in zip(xs.T.flat, ys.T.flat, markers, descripts):
    ax.scatter(x, y, marker=m, s=100, color='red')
    ax.text(x + 0.1, y - 0.1, d, size=14)
ax.set_axis_off()
plt.show()

**capture36

#barchart edge color
fig, ax = plt.subplots()
ax.bar([1,2,3,4],[10,25,15,20], ls='dashed', ec='red', linewidth=4)#ec can be written as edgecolor also, ecolor stands for error color
plt.show()

**capture37

#marker coloring
fig, ax = plt.subplots()
t = np.arange(0.0, 5.0, 0.1)
a = np.exp(-t) * np.cos(2*np.pi*t)
#plt.plot(t, a, '--r', t, a, 'dy', mec='g')
plt.plot(t, a, 'r:D', mfc='y', mec='g')#D big diamond, d small diamond, mec=marker edge color, mfc=marker face color
plt.show()

**capture38

#cmaps present
cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

# Turn off *all* ticks & spines
for ax in axes:
    ax.set_axis_off()
	
#mathtext, latent text
plt.scatter([1, 2, 3, 4], [4, 3, 2, 1])
plt.title(r'$\sigma_i=15$', fontsize=20)#see the title
plt.show()

**capture39

#mpl.rc, below instead of using the predefined color or line style it will cycle through the one defined in rc without having to explicitly mention
import matplotlib as mpl
from matplotlib.rcsetup import cycler
mpl.rc('axes', prop_cycle=cycler('color', 'rgc') +
                          cycler('lw', [1, 2, 4]) +
                          cycler('linestyle', ['-', '-.', ':']))#by default only color cycles ,this is for other property cycling
t = np.arange(0.0, 5.0, 0.2)
plt.plot(t, t)
plt.plot(t, t**2)
plt.plot(t, t**3)
plt.show()#rc might stand for run and configure

**capture40

#hide the boundry line
fig, ax = plt.subplots()
ax.plot([-2, 2, 3, 4], [-10, 20, 25, 5])
ax.spines['top'].set_visible(False)#this is used to hide the top line of the box
ax.xaxis.set_ticks_position('bottom')  # no ticklines at the top
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')  # no ticklines on the right
plt.show()

**capture41

