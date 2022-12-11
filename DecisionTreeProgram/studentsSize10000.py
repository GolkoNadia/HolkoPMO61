from id3 import buildTree as id3Tree
from id3 import predictDecisionID3
from c45 import buildTree as c45Tree
from c45 import predictDecisionC45
from chaid import buildTree as chaidTree
from chaid import predictDecisionCHAID
from myFunctions import *
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier



#read files
dataSet = pd.read_csv('../DecisionTreeProgram/new10000.csv') #you could add index_col=0 if there's an index

# dataSet = dataSet.sample(frac = 1, random_state=21)
# dataSet.to_csv('new10000.csv',index=False)

targetColumnName = list(dataSet.keys())[-1]
targetUniqueList = dataSet[targetColumnName].unique()
feature_col_tree = dataSet.columns.to_list()
feature_col_tree.remove(targetColumnName)
# print(dataSet)

def categorical_chart(cols):
   plt.figure(figsize=(10,4))
   sns.countplot(data=dataSet, x=cols, hue='Result')
   plt.title('college admission based on '+cols)
   # plt.show()

for cols in feature_col_tree:
   categorical_chart(cols)


#MEASURE TIME SIZE 100
start = time.time()
id3Alghoritm = id3Tree(dataSet)
end = time.time()
print('Building decision tree with ID3 (size: 100). Time: ', end - start)
pprint.pprint(id3Alghoritm)

start = time.time()
c45Alghoritm = c45Tree(dataSet)
end = time.time()
print('Building decision tree with C4.5 (size: 100). Time: ', end - start)
# pprint.pprint(c45Alghoritm)

start = time.time()
chaidAlghoritm = chaidTree(dataSet, targetUniqueList)
end = time.time()
print('Building decision tree with CHAID (size: 100). Time: ', end - start)
# pprint.pprint(chaidAlghoritm)

print('\nSplit train/test (size: 100 (80/20))')
X = dataSet.drop(targetColumnName, axis = 1)
y = dataSet[targetColumnName]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=27)
print_class_proportions(y)
print_class_proportions(y_train)
print_class_proportions(y_test)

trainDataSet = x_train
trainDataSet[targetColumnName] = y_train

start = time.time()
id3Alghoritm = id3Tree(trainDataSet)
end = time.time()
print('Building decision tree with ID3. Time: ',end - start)
# pprint.pprint(id3Alghoritm)

start = time.time()
c45Alghoritm = c45Tree(trainDataSet)
end = time.time()
print('Building decision tree with C4.5. Time: ',end - start)
# pprint.pprint(c45Alghoritm)

start = time.time()
chaidAlghoritm = chaidTree(trainDataSet, targetUniqueList)
end = time.time()
print('Building decision tree with CHAID. Time: ',end - start)
# pprint.pprint(chaidAlghoritm)

#PREDICTION DECISION ON TEST DATA
start = time.time()
predictListID3 = predictDecisionID3(x_test,id3Alghoritm,targetUniqueList)
end = time.time()
print('\nDoing prediction for test data with ID3. Time: ', end - start)

start = time.time()
predictListC45 = predictDecisionC45(x_test,c45Alghoritm,targetUniqueList)
end = time.time()
print('Doing prediction for test data with C4.5. Time: ', end - start)

start = time.time()
predictListCHAID = predictDecisionCHAID(x_test,chaidAlghoritm,targetUniqueList)
end = time.time()
print('Doing prediction for test data with CHAID. Time: ', end - start)

listWithoutNone = removeNoneValue(predictListID3, list(y_test))
predictListID3 = listWithoutNone[list(listWithoutNone.keys())[0]]
actualListID3 = listWithoutNone[list(listWithoutNone.keys())[1]]

listWithoutNone = removeNoneValue(predictListC45, list(y_test))
predictListC45 = listWithoutNone[list(listWithoutNone.keys())[0]]
actualListC45 = listWithoutNone[list(listWithoutNone.keys())[1]]

listWithoutNone = removeNoneValue(predictListCHAID, list(y_test))
predictListCHAID = listWithoutNone[list(listWithoutNone.keys())[0]]
actualListCHAID = listWithoutNone[list(listWithoutNone.keys())[1]]

# confusionMatrix(actualListID3, predictListID3,targetUniqueList)

#METRICS
print('\nMetrics for ID3')
score_report(actualListID3,predictListID3,targetUniqueList)

print('Metrics for C4.5')
score_report(actualListC45,predictListC45,targetUniqueList)

print('Metrics for CHAID')
score_report(actualListCHAID,predictListCHAID,targetUniqueList)




dataSet = pd.read_csv('../DecisionTreeProgram/new10000.csv') #you could add index_col=0 if there's an index
dataSet = dataSet.sample(frac = 1, random_state=21)

print('Split by cross validation (test size = 20)')
# dataSet = dataSet.sample(frac = 1, random_state=2)

fold =dataSet.values.tolist()
# print(dataSet)

k_fold = 2000
cross_val = crossValidation(fold,k_fold)
countFold = len(fold)//k_fold

start = time.time()
for i in range(0,countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   trainID3decisionTree = id3Tree(trainDF)
end = time.time()
print('Building decision tree with ID3. Time: ', str(end - start))

accuracyScoreList = []
precisionScoreList = []
recallScoreList = []
f1ScoreList = []
timeForPredictionWithID3 = 0

for i in range(0,countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   testDF = pd.DataFrame(cross_val['test'][i])
   testDF.columns = dataSet.columns
   # print('trainDF',trainDF)
   # print('testDF',testDF)
   trainID3decisionTree = id3Tree(trainDF)

   start = time.process_time()
   predictListForID3 = predictDecisionID3(testDF,trainID3decisionTree,targetUniqueList)
   timeForPredictionWithID3 += (time.process_time() - start)
   actualListForID3 = testDF[testDF.columns[-1]].values

   listWithoutNone = removeNoneValue(predictListForID3,actualListForID3)
   predictListForID3 = listWithoutNone[list(listWithoutNone.keys())[0]]
   actualListForID3 = listWithoutNone[list(listWithoutNone.keys())[1]]

   accuracyScoreList.append(accuracy_score(actualListForID3, predictListForID3))
   precisionScoreList.append(precision_score(actualListForID3, predictListForID3, average='weighted'))
   recallScoreList.append(recall_score(actualListForID3, predictListForID3, average='weighted'))
   f1ScoreList.append(f1_score(actualListForID3, predictListForID3, average='weighted'))

print('Time for predict (ID3): ', timeForPredictionWithID3)
print('Average metrics for ID3 alghoritm')
print(accuracyScoreList)
printAverageMetrics(accuracyScoreList,precisionScoreList,recallScoreList,f1ScoreList, countFold)
accuracyScoreList.clear()
precisionScoreList.clear()
recallScoreList.clear()
f1ScoreList.clear()


start = time.time()
for i in range(0,countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   trainC45decisionTree = c45Tree(trainDF)
end = time.time()
print('Building decision tree with C4.5. Time: ',end - start)

timeForPredictionWithC45 = 0
for i in range(0,countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   testDF = pd.DataFrame(cross_val['test'][i])
   testDF.columns = dataSet.columns

   trainC45decisionTree = c45Tree(trainDF)
   start = time.process_time()
   predictListForC45 = predictDecisionC45(testDF, trainC45decisionTree, targetUniqueList)
   timeForPredictionWithC45 += (time.process_time() - start)
   actualListForC45 = testDF[testDF.columns[-1]].values

   listWithoutNone = removeNoneValue(predictListForC45,actualListForC45)
   predictListForC45 = listWithoutNone[list(listWithoutNone.keys())[0]]
   actualListForC45 = listWithoutNone[list(listWithoutNone.keys())[1]]

   accuracyScoreList.append(accuracy_score(actualListForC45, predictListForC45))
   precisionScoreList.append(precision_score(actualListForC45, predictListForC45, average='weighted'))
   recallScoreList.append(recall_score(actualListForC45, predictListForC45, average='weighted'))
   f1ScoreList.append(f1_score(actualListForC45, predictListForC45, average='weighted'))

print('Time for predict (C4.5): ', timeForPredictionWithC45)
print('Average metrics for C4.5 alghoritm')
printAverageMetrics(accuracyScoreList,precisionScoreList,recallScoreList,f1ScoreList, countFold)
accuracyScoreList.clear()
precisionScoreList.clear()
recallScoreList.clear()
f1ScoreList.clear()


start = time.time()
for i in range(0,countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   trainCHAIDdecisionTree = chaidTree(trainDF, targetUniqueList)
end = time.time()
print('Building decision tree with CHAID. Time: ',end - start)

timeForPredictionWithCHAID = 0
for i in range(0,countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   testDF = pd.DataFrame(cross_val['test'][i])
   testDF.columns = dataSet.columns

   trainCHAIDdecisionTree = chaidTree(trainDF, targetUniqueList)

   start = time.process_time()
   predictListForCHAID = predictDecisionCHAID(testDF, trainCHAIDdecisionTree, targetUniqueList)
   timeForPredictionWithCHAID += (time.process_time() - start)
   actualListForCHAID = testDF[testDF.columns[-1]].values

   listWithoutNone = removeNoneValue(predictListForCHAID,actualListForCHAID)
   predictListForCHAID = listWithoutNone[list(listWithoutNone.keys())[0]]
   actualListForCHAID = listWithoutNone[list(listWithoutNone.keys())[1]]

   accuracyScoreList.append(accuracy_score(actualListForCHAID, predictListForCHAID))
   precisionScoreList.append(precision_score(actualListForCHAID, predictListForCHAID, average='weighted'))
   recallScoreList.append(recall_score(actualListForCHAID, predictListForCHAID, average='weighted'))
   f1ScoreList.append(f1_score(actualListForCHAID, predictListForCHAID, average='weighted'))

print('Time for predict (CHAID): ', timeForPredictionWithC45)
print('Average metrics for CHAID alghoritm')
printAverageMetrics(accuracyScoreList,precisionScoreList,recallScoreList,f1ScoreList, countFold)
accuracyScoreList.clear()
precisionScoreList.clear()
recallScoreList.clear()
f1ScoreList.clear()



print('Split by cross validation (test size = 10)')
fold = dataSet.values.tolist()
k_fold = 1000
cross_val = crossValidation(fold, k_fold)
countFold = len(fold) // k_fold

start = time.time()
for i in range(0, countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   trainID3decisionTree = id3Tree(trainDF)
end = time.time()
print('Building decision tree with ID3. Time: ', end - start)

accuracyScoreList = []
precisionScoreList = []
recallScoreList = []
f1ScoreList = []
timeForPredictionWithID3 = 0

for i in range(0, countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   testDF = pd.DataFrame(cross_val['test'][i])
   testDF.columns = dataSet.columns

   trainID3decisionTree = id3Tree(trainDF)
   start = time.process_time()
   predictListForID3 = predictDecisionID3(testDF, trainID3decisionTree, targetUniqueList)
   timeForPredictionWithID3 += (time.process_time() - start)
   actualListForID3 = testDF[testDF.columns[-1]].values

   listWithoutNone = removeNoneValue(predictListForID3, actualListForID3)
   predictListForID3 = listWithoutNone[list(listWithoutNone.keys())[0]]
   actualListForID3 = listWithoutNone[list(listWithoutNone.keys())[1]]

   accuracyScoreList.append(accuracy_score(actualListForID3, predictListForID3))
   precisionScoreList.append(precision_score(actualListForID3, predictListForID3, average='weighted'))
   recallScoreList.append(recall_score(actualListForID3, predictListForID3, average='weighted'))
   f1ScoreList.append(f1_score(actualListForID3, predictListForID3, average='weighted'))

print('Time for predict (ID3): ', timeForPredictionWithID3)
print('Average metrics for ID3 alghoritm')
printAverageMetrics(accuracyScoreList, precisionScoreList, recallScoreList, f1ScoreList, countFold)
accuracyScoreList.clear()
precisionScoreList.clear()
recallScoreList.clear()
f1ScoreList.clear()

start = time.time()
for i in range(0, countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   trainC45decisionTree = c45Tree(trainDF)
end = time.time()
print('Building decision tree with C4.5. Time: ', end - start)

timeForPredictionWithC45 = 0
for i in range(0, countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   testDF = pd.DataFrame(cross_val['test'][i])
   testDF.columns = dataSet.columns

   trainC45decisionTree = c45Tree(trainDF)
   start = time.process_time()
   predictListForC45 = predictDecisionC45(testDF, trainC45decisionTree, targetUniqueList)
   timeForPredictionWithC45 += (time.process_time() - start)
   actualListForC45 = testDF[testDF.columns[-1]].values

   listWithoutNone = removeNoneValue(predictListForC45, actualListForC45)
   predictListForC45 = listWithoutNone[list(listWithoutNone.keys())[0]]
   actualListForC45 = listWithoutNone[list(listWithoutNone.keys())[1]]

   accuracyScoreList.append(accuracy_score(actualListForC45, predictListForC45))
   precisionScoreList.append(precision_score(actualListForC45, predictListForC45, average='weighted'))
   recallScoreList.append(recall_score(actualListForC45, predictListForC45, average='weighted'))
   f1ScoreList.append(f1_score(actualListForC45, predictListForC45, average='weighted'))

print('Time for predict (C4.5): ', timeForPredictionWithC45)
print('Average metrics for C4.5 alghoritm')
printAverageMetrics(accuracyScoreList, precisionScoreList, recallScoreList, f1ScoreList, countFold)
accuracyScoreList.clear()
precisionScoreList.clear()
recallScoreList.clear()
f1ScoreList.clear()

start = time.time()
for i in range(0, countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   trainCHAIDdecisionTree = chaidTree(trainDF, targetUniqueList)
end = time.time()
print('Building decision tree with CHAID. Time: ', end - start)

timeForPredictionWithCHAID = 0
for i in range(0, countFold):
   trainDF = pd.DataFrame(cross_val['train'][i])
   trainDF.columns = dataSet.columns
   testDF = pd.DataFrame(cross_val['test'][i])
   testDF.columns = dataSet.columns

   trainCHAIDdecisionTree = chaidTree(trainDF, targetUniqueList)
   start = time.process_time()
   predictListForCHAID = predictDecisionCHAID(testDF, trainCHAIDdecisionTree, targetUniqueList)
   timeForPredictionWithCHAID += (time.process_time() - start)
   actualListForCHAID = testDF[testDF.columns[-1]].values

   listWithoutNone = removeNoneValue(predictListForCHAID, actualListForCHAID)
   predictListForCHAID = listWithoutNone[list(listWithoutNone.keys())[0]]
   actualListForCHAID = listWithoutNone[list(listWithoutNone.keys())[1]]

   accuracyScoreList.append(accuracy_score(actualListForCHAID, predictListForCHAID))
   precisionScoreList.append(precision_score(actualListForCHAID, predictListForCHAID, average='weighted'))
   recallScoreList.append(recall_score(actualListForCHAID, predictListForCHAID, average='weighted'))
   f1ScoreList.append(f1_score(actualListForCHAID, predictListForCHAID, average='weighted'))

print('Time for predict (CHAID): ', timeForPredictionWithC45)
print('Average metrics for CHAID alghoritm')
printAverageMetrics(accuracyScoreList, precisionScoreList, recallScoreList, f1ScoreList, countFold)
accuracyScoreList.clear()
precisionScoreList.clear()
recallScoreList.clear()
f1ScoreList.clear()


dataSet = pd.read_csv('../DecisionTreeProgram/new10000.csv') #you could add index_col=0 if there's an index

dataSet['school'] = dataSet['school'].map({'city': 0, 'village': 1, 'home':2}).astype(int)
dataSet['sex'] = dataSet['sex'].map({'Male': 0, 'Female': 1}).astype(int)
dataSet['Education'] = dataSet['Education'].map({'both': 0, 'at least one': 1, 'no one':2}).astype(int)
dataSet['Salary'] = dataSet['Salary'].map({'big': 0, 'middle': 1, 'low':2}).astype(int)
dataSet['Motivation'] = dataSet['Motivation'].map({'high': 0, 'not know': 1, 'low':2}).astype(int)
dataSet['Result'] = dataSet['Result'].map({'NO': 0, 'YES': 1}).astype(int)

start = time.time()

clf=RandomForestClassifier(n_estimators=100)

X = dataSet.drop(targetColumnName, axis = 1)
y = dataSet[targetColumnName]


x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=2)

clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
end = time.time()
print('Build Random Forest. Time: ', end - start)

print('Metrics for RandomForest')
score_report(y_test,y_pred,targetUniqueList)



