import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from myFunctions import *

dataSet = pd.ExcelFile('../DecisionTreeProgram/titanic.xlsx').parse('Лист1')  # you could add index_col=0 if there's an index
# print(dataSet)

targetColumnName = list(dataSet.keys())[-1]
targetUniqueList = dataSet[targetColumnName].unique()

feature_col_tree = dataSet.columns.to_list()
feature_col_tree.remove(targetColumnName)

# def categorical_chart(cols):
#     plt.figure(figsize=(10,4))
#     sns.countplot(data=dataSet, x=cols, hue='survived')
#     plt.title('Survived admission based on '+cols)
#     # plt.show()
#
# for cols in feature_col_tree:
#     categorical_chart(cols)



dataSet['survived'] = dataSet['survived'].map({'no': 0, 'yes': 1}).astype(int)
dataSet['status'] = dataSet['status'].map({'first': 0, 'second': 1,'third': 2,'crew': 3}).astype(int)
dataSet['age'] = dataSet['age'].map({'adult': 0, 'child': 1}).astype(int)
dataSet['sex'] = dataSet['sex'].map({'male': 0, 'female': 1}).astype(int)


X = dataSet.drop(targetColumnName, axis = 1)
y = dataSet[targetColumnName]


x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=2)
# print('x_train',x_train)
# print('x_test',x_test)
# print_class_proportions(y)
print_class_proportions(y_train)
print_class_proportions(y_test)

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)


col = ['status','age','sex']
feature_imp = pd.Series(clf.feature_importances_,index=col).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
# plt.legend()
# plt.show()

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,4), dpi=200)
tree.plot_tree(clf,
               feature_names = feature_col_tree,
               class_names=targetColumnName,
               filled = True)
# plt.show()


# confusionMatrix(y_test, predictions,targetUniqueList)
score_report(y_test,predictions,targetUniqueList)



dataSet = pd.read_csv('../DecisionTreeProgram/randomTitanic.csv')
# print(dataSet)

dataSet['survived'] = dataSet['survived'].map({'no': 0, 'yes': 1}).astype(int)
dataSet['status'] = dataSet['status'].map({'first': 0, 'second': 1,'third': 2,'crew': 3}).astype(int)
dataSet['age'] = dataSet['age'].map({'adult': 0, 'child': 1}).astype(int)
dataSet['sex'] = dataSet['sex'].map({'male': 0, 'female': 1}).astype(int)


cv = KFold(n_splits=5)
accuracies = list()
max_attributes = len(list(dataSet))
depth_range = range(1, max_attributes + 1)

for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth=depth)
    for i, (train_index, test_index) in enumerate(cv.split(dataSet)):
        X_train = dataSet.loc[train_index, feature_col_tree]
        X_test = dataSet.loc[test_index, feature_col_tree]
        y_train = dataSet.loc[train_index, targetColumnName]
        y_test = dataSet.loc[test_index, targetColumnName]

        clf = DecisionTreeClassifier(criterion="entropy")
        model = tree_model.fit(X_train, y_train)
        valid = model.score(X_test,y_test)

        fold_accuracy.append(valid)

    avg = sum(fold_accuracy) / len(fold_accuracy)
    accuracies.append(avg)

df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))


kf = KFold(n_splits=5)
accuracyScoreList = []
precisionScoreList = []
recallScoreList = []
f1ScoreList = []

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train = dataSet.loc[train_index,feature_col_tree]
    X_test = dataSet.loc[test_index,feature_col_tree]
    y_train = dataSet.loc[train_index,targetColumnName]
    y_test = dataSet.loc[test_index,targetColumnName]

    clf = DecisionTreeClassifier(criterion="entropy",max_depth=4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # confusionMatrix(y_test,y_pred,targetUniqueList)
    accuracyScoreList.append(accuracy_score(y_test, y_pred))
    precisionScoreList.append(precision_score(y_test, y_pred, average='weighted'))
    recallScoreList.append(recall_score(y_test, y_pred, average='weighted'))
    f1ScoreList.append(f1_score(y_test, y_pred, average='weighted'))

printAverageMetrics(accuracyScoreList, precisionScoreList, recallScoreList, f1ScoreList, 5)



dataSet = pd.ExcelFile('../DecisionTreeProgram/titanic.xlsx').parse('Лист1')  # you could add index_col=0 if there's an index

dataSet['survived'] = dataSet['survived'].map({'no': 0, 'yes': 1}).astype(int)
dataSet['status'] = dataSet['status'].map({'first': 0, 'second': 1,'third': 2,'crew': 3}).astype(int)
dataSet['age'] = dataSet['age'].map({'adult': 0, 'child': 1}).astype(int)
dataSet['sex'] = dataSet['sex'].map({'male': 0, 'female': 1}).astype(int)

targetColumnName = list(dataSet.keys())[-1]
targetUniqueList = dataSet[targetColumnName].unique()


clf=RandomForestClassifier(n_estimators=100)

X = dataSet.drop(targetColumnName, axis = 1)
y = dataSet[targetColumnName]


x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=42)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

score_report(y_test,y_pred,targetUniqueList)

# for i in range(0,5):
#     #draw first random forest
#     fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=200)
#     tree.plot_tree(clf.estimators_[i],
#                    feature_names = feature_col_tree,
#                    class_names=targetColumnName,
#                    filled = True)
    # fig.savefig('rf_individualtree.png')
    # plt.show()


