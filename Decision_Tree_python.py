#Python code to predict diabetes based on several factors.

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.tree import plot_tree

# Function to plot the decision tree
def plot_decision_tree(clf, feature_names, class_names):
    plt.figure(figsize=(15, 8))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        fontsize=12,
        max_depth=2
    )
    plt.title("Decision Tree Visualization", fontsize=16)
    plt.show()

#read data
data = pd.read_csv("diabetes.csv")

#setting up the data
y=data.iloc[:,[8]]
x=data.iloc[:,[1,2,3,4,5,6,7]] #excluded Pregnancies

#preprocessing the data
le = preprocessing.LabelEncoder()
y=le.fit_transform(y)

#splitting the data into train and test
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.05,random_state=0)

#training the classifier
clf=tree.DecisionTreeClassifier(criterion='gini',min_samples_split=30,splitter="best")
clf=clf.fit(X_train,Y_train)

#predicting
y_pred=clf.predict(X_test)

#testing the accuracy
accuracy=accuracy_score(Y_test,y_pred)
print(str(accuracy*100)+"% accuracy")

#testing the precision
average_precision = average_precision_score(Y_test,y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
      
#visualising the training set results
precision, recall, _ = precision_recall_curve(Y_test, y_pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.show()

# Visualizing the decision tree
plot_decision_tree(clf, feature_names=x.columns, class_names=['No Diabetes', 'Diabetes'])

height=pd.Series(y).value_counts(normalize=True)
plt.bar(range(2),height.tolist()[::-1],1/1.5,color='blue',label="classes",alpha=0.8)
plt.title('Decision Tree Classification')
plt.xlabel('1 0')
plt.ylabel('occurences')
plt.legend()
plt.show()
