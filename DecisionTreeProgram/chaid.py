import numpy as np
import pandas as pd
import pprint
import math


#фукнція для розрахунку ентропії поточного атрибуту
def find_chi_square(df, attribute,umique_mark):
    Class = df.keys()[-1]  # останній стовпець таблиці (рішення)
    variables = df[attribute].unique()  #масив унікальних значень поточного атрибуту
    res = 0
    for variable in variables:
        for target_variable in umique_mark:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            sum = den/2
            chi_square=math.sqrt(math.pow(num-sum,2)/sum)
            res += chi_square
    return res

#функція повертає атрибут з найбільшим інф. прибутком і який ще не використовувався
def find_best_attribute(df,umique_mark):
    Chi_squre = [] #рахує інф. прибуток
    for attr in df.keys()[:-1]:
        Chi_squre.append(find_chi_square(df,attr,umique_mark))

    if (len(set(Chi_squre)) == 1):
        return 0
    return df.keys()[:-1][np.argmax(Chi_squre)]

#функція виводить таблицю обєктів, де значення атрибуту == поточному
def table(df, node, value):
    return df[df[node] == value]

def majorityVoice(df):
    resultCol = df.keys()[-1]
    res_dataFrame = df[resultCol].value_counts()
    maxValue = res_dataFrame.max()

    dict = res_dataFrame.to_dict()
    key_list = list(dict.keys())
    val_list = list(dict.values())
    position = val_list.index(maxValue)

    percent = (maxValue / len(df[resultCol])) * 100
    return str(key_list[position]) + ' [' + str(percent) + '%]'


#функція для побудови дерева
def buildTree(df,umique_mark):
    node = find_best_attribute(df,umique_mark)
    if node==0:
        lift = majorityVoice(df)
        return '?'+ lift

    attValue = np.unique(df[node])

    tree = {}
    tree[node] = {}

    for value in attValue:
        my_table = table(df, node, value)
        cl = df.keys().values

        clValue,counts=np.unique(my_table[cl[-1]], return_counts=True)

        if len(counts) == 1:
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(my_table,umique_mark)
    return tree


def search(tree, newData, columns):
    currentkeyOfTree = [*tree]
    valueOfNewData = newData[currentkeyOfTree[0]].values
    nextKeyOfTree = tree.get(currentkeyOfTree[0]).get(valueOfNewData[0])

    if(str(nextKeyOfTree)[0]=='?'):
        for i in columns:
            if (str(nextKeyOfTree).find(i) != -1):
                nextKeyOfTree = i

    if (nextKeyOfTree == None):
        return

    if (nextKeyOfTree in columns):
        return nextKeyOfTree
    else:
        return search(nextKeyOfTree, newData, columns)


def predictDecisionCHAID(array: np.ndarray,tree,columns):
    predictList =[]
    for i in range(array.shape[0]):
        df2 = pd.DataFrame(array.values[i],array.columns)
        predictList.append(search(tree,df2.transpose(),columns))

    return predictList

