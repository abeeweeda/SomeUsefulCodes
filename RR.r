Sys.setlocale("LC_ALL","C")

# declaring a vector
vec = c(1,2,3,4,5)

#creating dataframe
df = data.frame(vector11, vector2)

#access a column
df$vector1

#adding new column
df$vector3 = somevector

#combine two dataframes by stacking
df_new = rbind(df1, df2)

#load csv data
csv_data = read.csv("file path with / (forward slash).csv")

getwd() #get current working directory

str(data) #gets structure of data

round(number, 2)#round to 2 decimal points

summary(data) #gets summary of data like description in python

#subsetting a data
new_data = subset(df, some criteria like Year==2000) #for multiple criteria put | or & signs

#save to csv
write.csv("path/filename.csv")

#list variable
ls()

#remove variable
rm(some variable / data frame or any other object etc)

#in r studio view data in table
View(data)

#wrap a string to multiple lines if too long
strwrap("some very long string")

# mean and standard deviation
mean(), sd()#also using these function use na.rm=TRUE as if there are NAs the values wont get calculated

#getting index of min or max values of a columns
which.max(df$column)
which.min(df$column)
#this index we can use for other purpose like
df$column2[which.max(df$column)] 

#scatter plot
plot(x, y)

#number of rows
nrows(date frame or column)

#column names
names(data frame or table)

#indexes in r start from 1 not zero, if you type vec[1] the first value will be printed, if you type vec[-1]
#all values except 1st value if vec[-2] all value except 2nd value vec[1:2] values from 1 to 2 both inclusive will be printed
#to reverse an array use rev e.g. rev(1:3) will pring 3 2 1

#histogram
hist(columnname)

#boxplot
boxpot(column1 ~ column2)#here column1 is of which distribution is shown and column two are different columns in graph
#e.g. boxplot(population ~ country) will show boxplots of population of different countries
#outliers are plotted as dots , first calclulte IQR or height of the box so any number outside of the top line of box
#or 3rd quartile + IQR or bottom line of box - IQR is considered outlier 

#labeling plots
boxplot(a ~ b, xlab="some", ylab="some", main="title")

#table function
table(columnA)#counts of all the distinct values of columnA
table(columnA, columnB)#counts of all the distinct disctinct combination of values of columnA and columnB
#like a pivot columnA as rows and columnB as columns and values as countries

#groupby
tapply(columntogroup, columntogroupby, function like mean or sd or sum or someother)
#if there are any NA's then we need to remove those
tapply(columntogroup, columntogroupby, function like mean or sd or sum or someother, na.rm=TRUE)

#get index of a string
match("india", RegionColumn)

#plot color
plot(x, y, ... col="red")
#limit the x or y values
plot(x, y, xlim = c(0,100), ylim = c(0,200))
#bins in histogram
hist(x, breaks=50)# will create 50 bins, but if you do xlim then the breaks will be on the original range 
#not the reduced range

#change data to numeric
as.numeric(data)

#converting na's to 0
data[is.na(data)]=0

#date conversion
as.Date(strptime(mvt$Date, "%m/%d/%y %H:%M")) the format is the format in which we recieve the date data
#to get months or weekdays
months(dateData), weekdays(dateData)
#strptime takes character input and makes datetime object and strftime vice verse
#also strptime is used when there is time element if there is no time we can simply use as.Date i guess not sure

#more subsetting, for egample we want to compare multiple or conditions
#instead of doinf a | b | c we can do like
locs = c("india", "japan", "china")
subsetted = subset(data, locations %in% locs)

#line plots
plot(x, y, type="l")#here l(its small l not or symbol) is line
#you can add more lines to above plot using line functions
lines(x1, y1)
lines(x2, y2)
#also you can draw a verticle line at any point in x axis
abline(v=xaxispoint, lwd=2)#lwd is line width and axispoint is the x axis coordinate where vertical line needs to be drawn
#here v is for vertical line for horizontal line put h instead

#we can also use function in table function e.g.
table(region , is.na(married)) # this will give how many null value are there for married column region wise

#mean of logical values give the ratio
#e.g. mean(c(TRUE, FALSE, TRUE, TRUE) = 3/4
#you can use the above trick in tapply functions to find the proportions

#join or merge
merge(A, B, by.x=columnfrom A, by.y=columnfrom B, all.x=TRUE)#here all.x=TRUE means all values from x -> left join

#sorting
sort(table(column), decreasing=FALSE)# ascending

#jitter function adds or substracts a small value from data points 
#you can use it when there is too much ovelap of points white plotting the data

#linear regression
#baseline = average value of y
modl = lm(y ~ x1 + x2, data=data)
#or
mod1 = lm(y ~ ., data=data)#here . means remaining all variables
#then we can use summary function on the model for details, more the stars more significant the variable
mod1$residuals #gives us the residuals
mod1$fitted.values #gives the fitted values of the model

#sum of squared errors
SSE = sum(mod1$residuals^2)

#summary of linear model
#estimate -> coefficient
#std error -> how much the coefficient is likely to vary from estimate
#t-value -> estimate/std error, larger the abs(t value) more significant the variable likely
#pr(>|t|) -> p-value, p<=0.05, if abs(t value) is large p-value is small and vice verse
#stars -> more stars more significant, max of 3 stars, a period(.) means almost significant

#** also if there is multicollinearity then you might not see the significance of those variable 
#even if they are significant

#correlation
cor(var1, var2)
cor(dataframe)
round(cor(dataframe),2)#round to two decimal values in the matrix
#correlation between independent and dependent variable is a good thing for obvious reason

#making predictions
predict(modl, newdata="test Data")
#when calculating SST on test data use mean of the training data
#** also test set R2 can be negative

#stepwise regression
mod2 = step(mod1)
#it will remove insignificant variable and create a new model considering R2 and model complexity

#AIC  Akaike information criterion (AIC) - it can be informally thought of as the 
#quality of the model with a penalty for the number of variables in the model.

#remove NA/NULL rows
data = na.omit(data)

#unordered factor variable with n different values will be treates a n-1 column with binary values and 
#the most repeted values or reference will be the one with all n-1 as 0's

#to manually define the reference level of a factor variable
column = relevel(column, value to be made reference from column)

#for single variable model R2 = correlation^2

#install package
install.packages("zoo")#has some helpful methods for time series
library(zoo) #like import in python

#create a lagged variable 
lagvar = lag(zoo(variable),-2, na.pad=TRUE)

#if while reading a column name is not as per r standard like its numeric r will add an X in front of it

#**logistic regression can be thought as extension of linear regression 
#logistic regression
modl = glm(y ~ X1 + X2 + .., data=data, family=binomial)#binomial for logistic regression
#here also we can use the summary function same as linear regression
#here instead of R2 we have AIC lower the AIC better the model
#logistic regression model tends to overfit if we have large numeber of varaibles,
#also the accuracy might differ for the same exact model made twice or by differnt computer when we have large number of variables

#in logistic baseline model accuracy is count(max of count of either 1 or 0)/total count of number of rows
#it bascically means if we have 10 rows 6 are 1's and 4 are 0's and if we just simply assign 1 to all
#our accuracy will be 6/10 as we will get 6 correct anyways

#train test split
install.packages("caTools")
library(caTools)
#now we can use
set.seed(somenumber)#for setting seed
split_var = sample.split(data$column (the y columns), SplitRatio=0.75)#75 % in train means TRUE rest in test
#also this splitting takes care of skewed classes
data_train = subset(data, split_var==TRUE)
data_test = subset(data, split_var==FALSE)

#predict the probabilities
predicts = predict(mod1, type="response")#this will predict probs of the training set
#for test data
predicts = predict(mod1, type="response", newdata=testData)#this will predict probs of the test set

#for making the confusion matrix
table(original_y_values, predicts>=0.5)#here 0.5 is the threshold
#we can choose accordingly as per our need like if you want more sensitivity or more specficity


#ROC curve (receiver operator characterstic) ->to decide which threshold is best
#on y axis we have TPR = sensitivity and on x axis we have FPR = 1 - specitivity
#it starts at (0,0) for threshold value of 1 means all points are categorized as 0, means TPR=0 which means FPR = 1 - 1 = 0
#and ends at (1,1) for threshold 0 means all points are categorized as 1, means TPR=1 which means FPR = 1 - 0 = 1

#ROC curves
install.packages("ROCR")
library(ROCR)
ROCRPred = prediction(predictedProbabilities, originalvalues)
ROCRPerf = performance(ROCRPred, "tpr", "fpr")
plot(ROCRPerf)#plotting ROC plot
plot(ROCRPerf , colorize=TRUE)#if you want colorful plot
plot(ROCRPerf, colorize=TRUE, print.cutoffs.at=seq(0,1,0.1), text.adj=c(-0.2,1.7))
#plotting the thresholds also seq is like range in python and text.adj is for adjusting the points as not to be on line

#AUC Area under the curve 
#perfect is 100%, 50% is pure guess a 45 degree line in ROC curve
auc = as.numeric(performance(ROCRPred, "auc")@y.values)#same as performance instead of tpr/fpr we have here auc

#handling missing data
#filling missing values based on non missing values in data - multiple imputation
install.packages("mice") #multiple imputation by chained equation
library(mice)
#we do the multiple imputation by not including the output or y variable
#so for this we create a new dataframe without the output variable
newFrame = originalDataframe_witout_output_or_any_other_variable_which_we_dont_need_for_imputation
random.seed(somenumber)#for obvious reason
imputedDF = complete(mice(newFrame))
#now you can replace the original columns in original DF with the columns from imputedDF

#removing unwanted variables from data
unvar = c("A", "B", "C", "D")#these are the vars we dont need
dataF = dataF[,!(names(dataF) %in% unvar)]

#all but one var in our model
mod1 = lm(y ~.-X1, data=Data)

#also if you want to remove some variable you can also simply do
rmvar = setdiff(names(df), c(variable_you_want_to_remove))
dataF = dataF[rmvar]

#CART - Classification and Regression Trees
install.packages("rpart")
library("rpart")
install.packages("rpart.plot")#plotting of rpart
library("rpart.plot")
treemodl = rpart(y~.-X1-X2, data=Data, method="class", minbucket=25)#here we are removing X1 and X2 from independent variables in the model
#here method = class is for classification and minbucket is for minimum number of data points needed for a split
#if we put minbucket=1 does not necessarily means we will have 1 observation in the leaf node and training accuracy will be 100% or SSE=0
#as there are other parameters of rpart which may try to make the model not overfit
#here if we skip the method="class" for the above model even for classification, if we put method=class the split will happen only if the probability 
#of happeing of one group is >0.5 and other group<0.5 but if we skip the method parameter the split may take place even if both have less than 0.5 prob.
prp(treemodl)#to see the tree structure.
#you can add digits=someNumber arguments to see the output to some significant digits or use roundint = FALSE sometimes it rounds to integer
#cart predictions
predicts = predict(treemod1, newdata = new_data, type="class")

#like logistic regression we if we want probabilities we can skip the type option in the above line
#and we can use our own threshold to define the cutoffs, if we use type="class" its like threshold of 0.5

#for creating ROC curve here
ROCPred = predict(treemod1, newdata=new_data)#here we wont give the type argument
#ROCRPred will give probabilities of both 0 and 1, we need probilities of 1 for that we can do
pred = predictions(ROCPred[,2], originalValues)#here ROCpred[,2] means take 2nd column which has the probabilities of output being 1
perf = performance(pred, "tpr","fpr")
plot(perf)#plotting ROC

#Random Forest
install.packages("randonForest")
library("randomForest")
mod1 = randomForest(X ~ y, data=data_train, nodesize=25, ntree=200)#here node size is the minimum number of observation in an subset(terminal node)
#the less the nodesize the larger the tree, ntree is the number of trees 
#after you run the above model it might ask you if you want to do regression , this happens if your variable has very few distinct values
#so for classification we need to convert our output variable into factors using as.factor
predicts = predict(mod1, newdata=new_data)
#here we will get the result not the probabilities
#if you want the probabilities
predicts = predict(mod1, newdata=new_data, type="prob")

#to view the number of times, aggregated over all of the trees in the random forest model, that a certain variable is selected for a split.
vu = varUsed(MODEL, count=TRUE)
vusorted = sort(vu, decreasing = FALSE, index.return = TRUE)
dotchart(vusorted$x, names(mod1$forest$xlevels[vusorted$ix]))
#the above code will create a dot chart with variable on y axis and number of times that variable is split on x

#to get variable importance we can plot
varImpPlot(mod1)#the higher the meandecreasegini more important the variable is in reducing the impurity over all the time it is used for split

#cross validation
install.packages("caret")
install.packages("e1071")
library("caret")
library("e1071")
numFolds = trainControl(method="cv", number=10)#10 folds cross validation
cpGrid = expand.grid(.cp=seq(0.01, 0.5, 0.01))#cp is complexity parameter like AIC or R2, adjusting for accuracy vs complexity of model
tr = train(X ~ y, data=TrainData, method="rpart", trControl=numFolds, tuneGrid = cpGrid)#this code will give you the best cp value for optimal model
train(y = Data$y, x = subset(Data, select=-c(y)), method = "rpart", ...)#in case of many factor variable do this instead of the above
#we can use the cp we get from above method to train our model simply print tr you will get the cp
#there are two ways you can get the best model one put cp value in the model like below
modl = rpart(X ~ y, data=TrainData, method="class", cp=the_cp_we_got_from_above_method)
#second
mod1 = tr$finalModel
#then we can do the predict and check for accuracy

#create a matrix in r
someMatrix = matrix(c(1,2,3,4,1,2,3,4,1,2,3,4), byrow=TRUE, nrow=3)#it will create a matrix with 3 rows each with element 1,2,3,4
#we can use * operator to do elmentwise multiplication of two matrices of same size
#also check rowsum rowSums and other matrix operation/functions

#loss function in radnom forest
modl = rpart(X ~ y, data=trainData, method="class", cp=some_cp_value_from_Cross_Validation, parms=list(loss=loss_matrix_see_doc))

#interaction variables
modl1 = glm(X ~ y1 + y2 + y1:y2, data=traindata, family=binomial)#here y1:y2 is the interaction variable (synergy effect) between y1 and y2

#randomly select n data points from your data
newData = trainData[sample(nrow(trainData), 2000), ]#here we are selecting 2000 random rows from trainData

#Text Analytics
tweet = read.csv('C:/Users/212698957/Downloads/tweets.csv', stringsAsFactors = FALSE)
#here stringAsFactor = FALSE means dont treat the string columns as a categorical variable because many fucntions might not work with factor data
#install text mining packages
install.packages("tm")
install.packages("SnowballC")#this helps in using the tm package
#create a corpus
corpus = VCorpus(VectorSource(tweet$Tweet))#here Tweet is a text column from tweet dataset
#also there is simple corpus e.g. corpus(VectorSource(tweet$Tweet)), here VCorpus is Volatile corpus
#to see the first line of the corpus
corpus[[1]][[1]]
#or
corpus[[1]]$content
corpus = tm_map(corpus, tolower)#making everything lower case
#or
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removePunctuation)# remove punctuation
stopwords("english")#gives stopwords from english for getting first 10 you can do [1:10]
corpus = tm_map(corpus, removeWords, c(stopwords("english"), "any_other_words_you_want_to_remove"))#remove stopwords
corpus = tm_map(corpus, stemDocument)#perform stemming
#after you made the transformation you can access the corpus elements simply by corpus[[1]]
frequencies = DocumentTermMatrix(corpus)#create a frequency matrix of the words
findFreqTerms(frequencies, lowfreq = 20)#to get all the words which appeared atleast 20 times in our tweets
sparse = removeSparseTerms(frequencies, 0.995) #keep only those terms which appear 0.5% or more tweets
tweetSparse = as.data.frame(as.matrix(sparse))#make dataframe of the sparse matrix
colnames(tweetSparse) = make.names(colnames(tweetSparse))#to make sure all the names of the column are ok 
#as there might be many columnnames starting with numbers and r struggles with that

#paste command like string concat
paste("A", "abhishek")
#or
paste0("A", "abhishek")
#output is "Aabhishek" #nospace

#grepl function returns TRUE if a string is found in another string
grepl("cat","dogs and cats",fixed=TRUE) #returns TRUE
grepl("cat","dogs and rats",fixed=TRUE) #returns FALSE

#number of character in a text
nchar("abhishek")#answer is 8

#length of a list
length(c("abhi", "shek"))#answer is 20

#Sparse Lightweight Arrays and Matrices, if the function rowSums doesnt run because of memory error run the below function
library(slam)
wordCount = rollup(dtm, 2, FUN=sum)$v #here dtm is the matrix, 2 is the dimension to be rolled, FUN is the function

#some good packages for text data  "RTextTools", "tau", "RWeka", and "textcat" 
#also see n-gram

#read data from text file
data = read.table('data.txt', header = FALSE, sep = '|', quote = "\"")#if header are there header=TRUE, sep is seperator
#last argument is something i dont know

#delete a column in dataframe
data$columnname = NULL

#remove duplicate
data = unique(data)

#heirarical clustering (dendrograms)
distances = dist(dataFrameWithTheColumnsOnWhichWeNeedClustering, method = "euclidean")#dist calculates the distances between all the points
#also we can use the same function if we want to compute the distace of elments within a vector, instead of dataframe put vector
clusterModl = hclust(distances, method = "ward.D")
#ward.D is a minimum variance method which tries to find compact and spherical clusters, 
#tries to minimize the variance within cluster and maximize distance among clusters
plot(clusterModl)#plots dendrogram
rect.hclust(clusterIntensity, k=3, border="red")#visualize the cluster here we the dendogram will show the 3 clusters with rectangular border in red
clusterGroups = cutree(clusterModl, k=10)#dividing data into 10 clusters
#clusterGroups will give you clustergroup number for each row, like clusterGroups[19] will tell you which cluster 19th row belongs to
#for example we clustered the movies data based on genre and we want to see Action is mostly in which group
tapply(movies$Action, clusterGroups, mean)#as Action is a binary variable here we will get the percentage of movies in each group which are action
#splitting different clusters
spl = split(data, clusterGroups)#if you have a variable named split you might need to remove it using rm(split)
spl[[1]]#will give you first cluster data, its same as subset(data, clusterGroups==1)
lapply(spl, colMeans)#this will give colMeans for each split

#visulalise an image, let us say we have an n by n matrix or n^2 vector
image(imageMatrix)
#if we have vector
dim(imageVector)=c(n,n)
#now we can use image function
image(imageVector, axes=FALSE, col = grey(seq(0,1,length=256)))#axis = FALSE mean we dont wanna see the axis, here col = .. is for greyscale image

#** when the vector is too huge the heirarical clustering wont work on low memory computer as there will be too many pairwise distance calculations
#e.g.lets say we have n=365636 elements, we need n(n-1)/2 pairs of distances which is about 67 billion pairs

#Kmeans
KMC = kmeans(data, centers=k, iter.max=1000)#here k is number of cluster, iter.max is maximum iteration
Clusters = KMC$cluster #getting the cluster
KMC$centers #to get the centers
#you can use the str(KMC) to see what info we can extract using $ sign
#now again displaying image
dim(Clusters) = c(m,n)
image(Clusters, axes=FALSE, col=rainbow(k))
#here just to make image look nice we use rainbow(k) as we have k cluster so we show different color for each cluster

#now to use one clustered image as training to test other image as testing we need to install flexclust
install.packages("flexclust")
library("flexclust")
#here what we are doing we have one image of healthy body part and one image of not so healthy body part and the healthy image we have
#already clustered above using k means, now we want to use this clustered image as to identify the problem in the other image
#in order to do that we need to convert the data into kcca object
#KCCA - k Centriods Cluster Analysis
KMC.kcca = as.kcca(KMC, data)#here KMC is the original k means model and data is the original data fed to kmeans
predictedCluster = predict(KMC.kcca, newdata=newDataToBeClustered)
#now you can change the dimension of the predicterdCluster using dim and see the image using image
#also read about modified fuzzy k-means

#the below function gives 6 variables with most average in a after sorting,
#here the columns were word names and data was number of times the word occured in a data point and
#we wanted on an average top 6 words across all the data points/rows
tail(sort(colMeans(data)))

#Normalizing Data
preproc = preProcess(data)#this is like fit, for example you will run this on train data and run the below on both train and test
NormData = predict(preproc, data)#here NormData is normalized Data

#Visualization
#ggplot
library("ggplot2")
#scatter plot
scatterPlot = ggplot(data, aes(x=X, y=y))#here scatterPlot is a variable, aes is aesthetics
scatterPlot + geom_point()#it will plot points 
scatterPlot + geom_line()#it will plot lines
scatterPlot + geom_point(color="blue", size=3, shape=17)#we can add color shape size etc to point or other object
scatterPlot + geom_point(color="blue", size=3, shape=17) + ggtitle("x vs y")#to add title
#to save the plot to a file
plotVar = scatterPlot + geom_point(color="blue", size=3, shape=17) + ggtitle("x vs y")#create a variable, different number different shapes
pdf("plot.pdf")#create a pdf file
print(plotVar)#save the plot variable into the above created pdf file
dev.off()#close the file

ggplot(data, aes(x=X, y=y, color=z)) + geom_point()#if you want color your scatter plot based on other variable z
ggplot(data, aes(x=X, y=y, color=z)) + geom_point() + scale_color_brewer(palette="Dark2")#adding color palette
ggplot(data, aes(x=X, y=y)) + geom_point() + stat_smooth(method="lm")#plotting regression line it will have a default confidence inteval of 95%
ggplot(data, aes(x=X, y=y)) + geom_point() + stat_smooth(method="lm", level=0.99)#99% confidence interval
ggplot(data, aes(x=X, y=y)) + geom_point() + stat_smooth(method="lm", se=FALSE)#no confidence interval
ggplot(data, aes(x=X, y=y)) + geom_point() + stat_smooth(method="lm", se=FALSE, col="somecolr")#coloring regression line

#line plot
ggplot(data, aes(x=X, y=y)) + geom_line(aes(group=1))#here group=1 means we need one group or one line not sure what it does
#suppose you have y variable as days like sunday monday and all, and your line plot will sort alphabetically
#to sort them like sunday monday, we need to conver that column to ordered factor varible
Data$WeekdayVar = factor(Data$WeekdayVar, ordered = TRUE, levels = c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday")
ggplot(data, aes(x=X, y=y)) + geom_line(aes(group=1)) + xlab("x label") + ylab("y label")#labelling
ggplot(data, aes(x=X, y=y)) + geom_line(aes(group=1), linetype=2, alpha=0.3)#alpha is same as in python and linetype will change linestyle like dashed
ggplot(data, aes(x=X, y=y)) + geom_line(aes(group=z))#if you want to group by a third varible z , here we will have different line for each z
#remember z might needs to be a fator variable
ggplot(data, aes(x=X, y=y)) + geom_line(aes(group=z, color=z), size=1)#here we are coloring differnt lines for each z and then size is for line width

#convert factor to numeric
newVar = as.numeric(as.character(FactorVarWeWantToConvert))
 
#heat map
ggplot(data, aes(x=X, y=y)) + geom_tile(aes(fill=z))
ggplot(data, aes(x=X, y=y)) + geom_tile(aes(fill=z)) + scale_fill_gradient(name="legend label", low="white",high="red") + theme(axis.title.y = element_blank())
#scale fill gradient name changes the legend name, low and high defines the colors theme axis.title.y = element_blank means hide the y axis label

#maps
install.packages("maps")
install.packages("ggmap")
#load both using library
#** you will need google maps API's for using the maps package
locationNamelikeChicago = get_map(location="chicago", zoom=11)#it wont work without google map api
ggmap(locationNamelikeChicago)
#if you have api key you might need to run the below code for the above codes to work
if(!requireNamespace("devtools")) install.packages("devtools") devtools::install_github("dkahle/ggmap", ref = "tidyup", force = TRUE)
library(ggmap) register_google(key = "yourAPI")
#once its working
ggmap(locationNamelikeChicago) + geom_point(data=data, aes=(x=lon, y=lat))

#map data, US map data is already in R
statesMap = map_data("state")#load united states map data
str(statesMap)# it will have lat, long, group, order, region, subregion
#we can use the lat lons feom above data to plot the US map using polygon
#to plot the map
ggplot(statesMap, aes(x=long, y=lat, group=group)) + geom_polygon(fill="white", color="black")#white is the background and black is the border
ggplot(statesMap, aes(x=long, y=lat, group=group, fill=z)) + geom_polygon(fill="white", color="black")#here we fill the areas based on some other var
ggplot(statesMap, aes(x=long, y=lat, group=group, fill=z)) + geom_polygon(color="black") + scale_fill_gradient(low="black", high="red", guide="legend")
#above the guide="legend" is to make sure we get a legend
#now for example we have all the values of z from 0, 10 and there are few outliers like 100, 150 out plot may not look good, so to limit the values we can do
ggplot(statesMap, aes(x=long, y=lat, group=group, fill=z)) + geom_polygon(color="black") + scale_fill_gradient(low="black", high="red", guide="legend", limit=c(0,10))

#barchart
ggplot(data, aes(x=X, y=y)) + geom_bar(stat = "identity") + geom_text(aes(label=y))#here stat="identity" says use the value of y as it is no sum or count etc
#here the bar might be sorted based on alpabetical order to sort them based on y values we need to update our dataframe
dataFrame = transform(dataFrame, x=reorder(x, -Y))#here we are making x an ordered variable and -Y means in decreasing order of Y
#after ordering we can now again plot the bar chart
ggplot(data, aes(x=X, y=y)) + geom_bar(stat = "identity", fill="darkblue") + geom_text(aes(label=y, vjust=-0.4)) 
+ theme(axis.ticks.x = element_blank(), axis.text.x = element_text(angle=45, hjust=1))#here fill will color the bars, vjust will adjust the text
#axis.ticks.x = element_blank will remove the x axis ticks( ticks are small lines), element_text will put the x axis text at an angle od 45 degree
#and hjust will adjust the x axis text 

#world map 
world_map = map_data("world")#same as above like we did for state
#you can plot lat lon on ploygon and you can get a world map
ggplot(world_map, aes(x=long, y=lat, group=group)) + geom_polygon(fill="white", color="black") + coord_map("mercator")
#here mercator is kind of a projection there are other kinds also
#now for plotting te data you need to merge the map_data with the dataframe having the data
#we can merge in the region name, after merging the order of the data might change and if you plot the data it wont plot correctly
#to order the data properly
world_map_merged = world_map_merged[order(world_map_merged$group, world_map_merged$order),]
#because witin each group lat lons are ordered based order field
#also sometimes the region names might not match in your data and map data so you might need to check that also
ggplot(world_map, aes(x=long, y=lat, group=group)) + geom_polygon(aes(fill=z), color="black") + coord_map("ortho", orientation=c(30,40,0))
#here we are filling the maps based on column z and we are usign a different type of projection, 
#its a globe type projection centered at (30, 40 ,0), you can give you own center coordinates

#converting columns to values, for example we have one column like year and other and many columns for each country and then we have data
#we want one column for year one column for country and one column for data, similar to crosstable in qlik
install.packages("reshape2")
library("reshape2")
data = melt(data,id="Year")#here melt is a function in reshape2 package, 
#id means which column to make first column other columns will be made as rows

#some other parameters of geom_polygon
#we can simply write fill inside the ggplot instead of geom_polygon
ggplot(world_map, aes(x=long, y=lat, group=group, fill=z)) + geom_polygon(color="black")
#here alpha is for transparency
ggplot(world_map, aes(x=long, y=lat, group=group, fill=z)) + geom_polygon(color="black", alpha=0.3)
#here linetype=3 is for dashed border and size is for border width
ggplot(world_map, aes(x=long, y=lat, group=group, fill=z)) + geom_polygon(color="black", linetype=3, size=2)
#here inside scale_fill_gradient low and high are the color gradients, breaks=c(0,1) in case we are plotting binary outcome, 
#labels are given as we have binary outcome so 0 for lab1 and lab2 is for 1, name is for legend title and guide is to make sure we get legend
ggplot(world_map, aes(x=long, y=lat, group=group, fill=z)) + geom_polygon(color="black")
scale_fill_gradient(low = "blue", high = "red", guide = "legend", breaks= c(0,1), labels = c("lab1", "lab2"), name = "plot")

#plotting networks
install.packages("igraph")
library("igraph")
g = graph.data.frame(edges, FALSE, users)#here edges is dataframe having egde points like which vertices are connected
#, and users is the data frame having vetices
#e.g. edges might have one row with two columns and values A, B, while vertices will have many column with one column as 
#the value of the edges like id, and id will have two rows A and B , and there might be other columns which define other
#characterstic for A and B like here we have which school A and B go to , where they live etc
plot(g, vertex.size=5, vertex.label=NA)#to plot the graph, here all the vertices are of same size means same diameter, 
#and vertex.label=NA means not to show labels as the you might not be able to see the graph itself
V(g)$size = degree(g)/2+2#here V is vertex
plot(g, vertex.label=NA)
#the above two lines means instead of defining same vertex size for all we can change it bases on degrees, 
#different diameter based in degree, means if a degree of a vertex is more it will be bigger 
#here degrees(g) will give you number or other node(vertex) connected to a particular vertex
V(g)$color[V(g)$gender =="M"] = "red"
V(g)$color[V(g)$gender =="F"] = "grey"
plot(g, vertex.label=NA)
#the above three lines means, let us say we have a column in our user df as gender,
#and we need to color the vertices based on gender, this is how we can do it

#text data using word clouds
install.packages("wordcloud")
library("wordcloud")
#we are using the same tweets dataset here, the first argumnets is the words which is in our data the columns,
#and second is the frequencies so here we are using colSums, scale is used here as we have many words which might not fit so to scale them
wordcloud(colnames(allTweets), colSums(allTweets), scale=c(2, 0.25))
#for parameters writer ?wordcloud

#for color pallette we can install
install.packages("RColorBrewer")
library("RColorBrewer")
#and to use one pallette we can do, here we did [5:9] as colors 1:4 in this pallette are very light
wordcloud(colnames(allTweets), colSums(allTweets), scale=c(2, 0.25),colors=brewer.pal(9, "Blues")[5:9])

#ggplot histogram
ggplot(data = parole, aes(x = X)) + geom_histogram(binwidth = 5, boundary = 0, color = 'black', fill = 'cornflowerblue')
#the above is a basic histogram
ggplot(data = parole, aes(x = X)) + geom_histogram(binwidth = 5, boundary = 0) + facet_grid(y ~ .)
#the above code will create multiple plots one above another for different values of variable y
ggplot(data = parole, aes(x = X)) + geom_histogram(binwidth = 5, boundary = 0) + facet_grid(.~y)
#the above code will create side by side histogram for variable y
#below we are manually defining color pallette
colorPalette = c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
ggplot(data = parole, aes(x = X, fill = y)) + geom_histogram(binwidth = 5, boundary = 0) + scale_fill_manual(values=colorPalette)
#this will create stacked histogram for y values and using the colors from the pallette defined above
ggplot(data = parole, aes(x = X, fill = y)) + geom_histogram(binwidth = 5, boundary = 0, position="identity") + scale_fill_manual(values=colorPalette)
#here position="identity" means dont stack histogram put it one above other
ggplot(data = parole, aes(x = X, fill = y)) + geom_histogram(binwidth = 5, boundary = 0, position="identity", alpha=0.5) 
+ scale_fill_manual(values=colorPalette)
#here alpha=0.5 is used because once we put one histogram over other you might not be able to see one or other





