import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection,linear_model

# read data
iris_data=pd.read_csv('iris.csv',header=None,names=['sepal length','sepal width','petal length','petal width','classes']) #.data文件可以直接用pandas中的read_csv读取,返回DataFrame

# different class attribute value
setosa_sepal_length=iris_data.ix[iris_data.classes=='Iris-setosa','sepal length']
setosa_sepal_width=iris_data.ix[iris_data.classes=='Iris-setosa','sepal width']
setosa_petal_length=iris_data.ix[iris_data.classes=='Iris-setosa','petal length']
setosa_petal_width=iris_data.ix[iris_data.classes=='Iris-setosa','petal width']
versicolour_sepal_length=iris_data.ix[iris_data.classes=='Iris-versicolor','sepal length']
versicolour_sepal_width=iris_data.ix[iris_data.classes=='Iris-versicolor','sepal width']
versicolour_petal_length=iris_data.ix[iris_data.classes=='Iris-versicolor','petal length']
versicolour_petal_width=iris_data.ix[iris_data.classes=='Iris-versicolor','petal width']
virginica_sepal_length=iris_data.ix[iris_data.classes=='Iris-virginica','sepal length']
virginica_sepal_width=iris_data.ix[iris_data.classes=='Iris-virginica','sepal width']
virginica_petal_length=iris_data.ix[iris_data.classes=='Iris-virginica','petal length']
virginica_petal_width=iris_data.ix[iris_data.classes=='Iris-virginica','petal width']
print(iris_data.classes)
# plot scatter
plt.subplot(2,3,1)
plt.scatter(setosa_sepal_length,setosa_sepal_width,c='b')
plt.scatter(versicolour_sepal_length,versicolour_sepal_width,c='g')
plt.scatter(virginica_sepal_length,virginica_sepal_width,c='r')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('sepal length vs sepal width')
plt.subplot(2,3,2)
plt.scatter(setosa_sepal_length,setosa_petal_length,c='b')
plt.scatter(versicolour_sepal_length,versicolour_petal_length,c='g')
plt.scatter(virginica_sepal_length,virginica_petal_length,c='r')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.title('sepal length vs petal length')
plt.subplot(2,3,3)
plt.scatter(setosa_sepal_length,virginica_petal_width,c='b')
plt.scatter(versicolour_sepal_length,versicolour_petal_width,c='g')
plt.scatter(virginica_sepal_length,virginica_sepal_width,c='r')
plt.xlabel('sepal length')
plt.ylabel('petal width')
plt.title('sepal length vs petal width')
plt.subplot(2,3,4)
plt.scatter(setosa_sepal_width,setosa_petal_length,c='b')
plt.scatter(versicolour_sepal_width,versicolour_petal_length,c='g')
plt.scatter(virginica_sepal_width,virginica_petal_length,c='r')
plt.xlabel('sepal width')
plt.ylabel('petal length')
plt.title('sepal width vs petal length')
plt.subplot(2,3,5)
plt.scatter(setosa_sepal_width,setosa_petal_width,c='b')
plt.scatter(versicolour_sepal_width,versicolour_petal_width,c='g')
plt.scatter(virginica_sepal_width,virginica_sepal_width,c='r')
plt.xlabel('sepal width')
plt.ylabel('petal width')
plt.title('sepal width vs petal width')
plt.subplot(2,3,6)
plt.scatter(setosa_petal_length,setosa_petal_width,c='b')
plt.scatter(versicolour_petal_length,versicolour_petal_width,c='g')
plt.scatter(virginica_petal_length,virginica_petal_width,c='r')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('petal length vs petal width')
plt.show()
# data segment
attr_data=iris_data.iloc[:,0:4]#依然保留columns和index
class_data=iris_data.iloc[:,4]
# split data into train and test
X_train,X_test,y_train,y_test=model_selection.train_test_split(attr_data.values,class_data.values,test_size=0.25,random_state=0,stratify=class_data.values)
regr=linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
regr.fit(X_train,y_train)
print('Coefficients:%s, intercept %s'%(regr.coef_,regr.intercept_))
print("Score:%.2f"%regr.score(X_test,y_test))