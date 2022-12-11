# from id3 import buildTree as id3Tree
# from id3 import search as searchPredict
# from c45 import buildTree as c45Tree
# from chaid import buildTree as chaidTree
# from myFunctions import *
import pandas as pd
import numpy as np
import pprint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

#пропорції міток класу
def class_proportions(array: np.ndarray):
    # Мітки класу і їх кількість
    labels, counts = np.unique(array, return_counts=True)
    counts_perc = counts/array.size
    res = dict()
    for label, count2 in zip(labels, zip(counts, counts_perc)):
        res[label] = count2
    return res

#друк таблиці: пропорції міток класу
def print_class_proportions(array: np.ndarray):
    proportions = class_proportions(array)
    print("{:<10} {:<15} {:<10}".format('Мітка','Кількість','Відсоткoове співвідношення'))
    for i in proportions:
        val, val_perc = proportions[i]
        val_perc_100 = round(val_perc * 100, 2)
        print("{:<10} {:<15} {:<10}".format(i, val,val_perc_100))

def confusionMatrix(actualList,predictList, targetUniqueList):
    conf_matrix = confusion_matrix(actualList, predictList)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    tick_marks = np.arange(len(targetUniqueList))
    plt.xticks(tick_marks, targetUniqueList)
    plt.yticks(tick_marks, targetUniqueList)

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)

    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    plt.show()

def score_report(y_test, predictList, targetUniqueList):
    print('Accuracy: %.3f' % accuracy_score(y_test, predictList))
    print('Precision: %.3f' % precision_score(y_test, predictList, average='weighted', zero_division=0))
    print('Recall: %.3f' % recall_score(y_test, predictList, average='weighted', zero_division=0))
    print('F1 Score: %.3f' % f1_score(y_test, predictList, average='weighted', zero_division=0))
    # print()
    # print(classification_report(y_test, predictList, target_names=targetUniqueList))

def random_data(datas, count):
    data_df = pd.DataFrame(columns=datas.keys())
    np.random.seed(12)
    for key in datas:
        for i in range(count):
            data_df.loc[i,key] = str(np.random.choice(datas[key], 1)[0])
    return  data_df
    # print(data_df)
            # data_df.loc[i, 'school'] = str(np.random.choice(datas['school'], 1)[0])
            # data_df.loc[i, 'sex'] = str(np.random.choice(datas['sex'], 1)[0])
            # data_df.loc[i, 'parentHigherEducation'] = str(np.random.choice(datas['parentHigherEducation'], 1)[0])
            # data_df.loc[i, 'Salary'] = str(np.random.choice(datas['Salary'], 1)[0])
            # data_df.loc[i, 'Motivation'] = str(np.random.choice(datas['Motivation'], 1)[0])
            # data_df.loc[i, 'Result'] = str(np.random.choice(datas['Result'], 1)[0])


def createRandomDataFiles(datas):
    data_df_100 = random_data(datas,100)
    data_df_100.to_csv('randomData100.csv',index=False)
    data_df_1000 = random_data(datas,1000)
    data_df_1000.to_csv('randomData1000.csv',index=False)
    data_df_10000 = random_data(datas,10000)
    data_df_10000.to_csv('randomData10000.csv',index=False)


def removeNoneValue(predictList, actualList):
    lstIndex = []
    for i in range(0, len(predictList)):
        if predictList[i] == None:
            lstIndex.append(i)
    predictListWithoutNone = []
    actualListWithoutNone = []
    result = {'prediction': predictListWithoutNone, 'actual': actualListWithoutNone}
    for i in range(0, len(predictList)):
        if i not in lstIndex:
            predictListWithoutNone.append(predictList[i])
            actualListWithoutNone.append(actualList[i])
    return result

def printAverageMetrics(accuracyScoreList,precisionScoreList,recallScoreList,f1ScoreList, countFold):
    # print('Accuracy: ',accuracyScoreList)
    print('Average Accuracy: ', sum(accuracyScoreList)/countFold)
    # print('Precision: ', precisionScoreList)
    print('Average Precision: ', sum(precisionScoreList)/countFold)
    # print('Recall: ', recallScoreList)
    print('Average Recall: ', sum(recallScoreList)/countFold)
    # print('F1: ', f1ScoreList)
    print('Average F1: ', sum(f1ScoreList)/countFold)
    print()


def crossValidation(fold,k_fold):
    train = []
    test = []
    cross_val = {'train': train, 'test': test}
    rivDil = k_fold
    for i in range(0,len(fold),rivDil):
        test.append(fold[i:i+rivDil])
        train.append(fold[:i]+fold[i+rivDil:])

    return cross_val

