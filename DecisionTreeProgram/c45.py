import numpy as np
import pandas as pd
import pprint
import math

#функція для розрахунку загальної енропії
def calculate_entropy(df):
    Class = df.keys()[-1]  # останній стовпець таблиці (рішення)
    entropy = 0
    values = df[Class].unique() #унікальні значення останього стовпця для поточного значення атрибута
    for value in values:
        ratio = df[Class].value_counts()[value] / len(df[Class]) #відносна к-сть елементів у множині D, які належать до одного класу
        ratio=round(ratio,10)
        entropy += -ratio * math.log2(ratio)
    return entropy

#фукнція для розрахунку ентропії поточного атрибуту
def find_entropy_attribute(df, attribute):
    Class = df.keys()[-1]  # останній стовпець таблиці (рішення)
    target_variables = df[Class].unique()  #масив унікальних значень останього стовпця для поточного значення атрибута
    variables = df[attribute].unique()  #масив унікальних значень поточного атрибуту
    sum_of_entropy = 0
    for variable in variables:
        entropy = 0 #енторопія значення атрибуту
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            ratio = num / den #відношення кількості значень поточного атрибуту, де a==v до к-сті поточного значення атрибуту

            if(ratio==0.0):
                entropy+=0.0
            else:
                entropy += -ratio * math.log2(ratio)
        ratio2 = den / len(df) #відношення кількості поточного значення атрибуту до кількості усіх значень атрибуту
        sum_of_entropy += -ratio2 * entropy #сума усіх ентропій значень атрибуту

    return abs(sum_of_entropy)



def find_split_info_attribute(df, attribute):
    variables = df[attribute].unique()  #масив унікальних значень поточного атрибуту
    split_info = 0
    for variable in variables:
        den = len(df[attribute][df[attribute] == variable])
        ratio2 = den / len(df) #відношення кількості поточного значення атрибуту до кількості усіх значень атрибуту

        split_info += -ratio2*math.log2(ratio2)
    return split_info


#функція повертає атрибут з найбільшим інф. прибутком і який ще не використовувався
def find_best_attribute(df):
    GainRatio=[]
    for attr in df.keys()[:-1]:
        split_inf=find_split_info_attribute(df, attr)
        if(round(calculate_entropy(df),7)==round(find_entropy_attribute(df,attr),7)):
            GainRatio.append(0.0)
        else:
            gain=calculate_entropy(df) - find_entropy_attribute(df, attr)
            if(split_inf==0):
                GainRatio.append(0.0)
            else:
                GainRatio.append(gain/split_inf)

    if (len(set(GainRatio)) == 1):
        return 0
    return df.keys()[:-1][np.argmax(GainRatio)]

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
def buildTree(df):
    node = find_best_attribute(df)
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
            tree[node][value] = buildTree(my_table)
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
        return  # 'I DONT KNOW'

    if (nextKeyOfTree in columns):
        # decision.append(nextKeyOfTree)
        return nextKeyOfTree
    else:
        return search(nextKeyOfTree, newData, columns)

def predictDecisionC45(array: np.ndarray,tree,columns):
    predictList =[]
    for i in range(array.shape[0]):
        df2 = pd.DataFrame(array.values[i],array.columns)
        predictList.append(search(tree,df2.transpose(),columns))

    return predictList



# df = pd.ExcelFile('C:/Users/Nadia.Golko/Desktop/Magisterska/credit/trainBigWithSame.xlsx').parse('Лист1')  # you could add index_col=0 if there's an index
# print(df)
# t = buildTree(df)
# pprint.pprint(t)


