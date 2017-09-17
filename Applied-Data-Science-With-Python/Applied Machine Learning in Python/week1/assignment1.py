#! python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
"""
classification:分类
clustering: 聚类

159
down vote
In general, in classification you have a set of predefined classes and want to know which class a new object belongs to.

Clustering tries to group a set of objects and find whether there is some relationship between the objects.

In the context of machine learning, classification is supervised learning and clustering is unsupervised learning.
"""
cancer = load_breast_cancer()
# print(cancer.keys())


def answer_zero():
    """Example"""
    return len(cancer['feature_names'])


def answer_one():
    columns = np.append(cancer['feature_names'], 'target')
    values = np.vstack((cancer['data'].T, cancer['target'])).T
    df = pd.DataFrame(columns=columns, data=values)
    return df


def answer_two():
    """
    malignant: 恶性的 0
    benign: 良性的 1
    """
    cancerdf = answer_one()
    index = np.append('malignant', 'benign')
    malignant_value = len(cancerdf['target']) - np.count_nonzero(cancerdf['target'])
    benign_value = np.count_nonzero(cancerdf['target'])
    data = np.append(malignant_value, benign_value)
    target = pd.Series(index=index, data=data)
    return target


def answer_three():
    cancerdf = answer_one()
    X = cancerdf.iloc[:, 0:30]  # 前面数值代表行，然后后面的数字代表列, loc一般是用列的名字来进行索引
    y = cancerdf.iloc[:, 30]
    return X, y


def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors=1)
    classifier = knn.fit(X_train, y_train)
    return classifier


def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1,-1)
    # numpy reshape: -1 parameter means that the size of the dimension, for which you passed -1, is being inferred(推断)
    # print(cancerdf.head())
    # print('--------------------------')
    # print(cancerdf.mean())
    # print('--------------------------')
    # print(cancerdf.mean()[:-1])
    # print('--------------------------')
    # print(cancerdf.mean()[:-1].values)
    # print('--------------------------')
    # print(cancerdf.mean()[:-1].values.reshape(1,-1))
    # print('--------------------------')
    classifier = answer_five()
    means_prediction = classifier.predict(means)[0]
    array = np.empty(shape=(0, 0))
    np.append(array, means_prediction)
    return array


def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    res = knn.predict(X_test)
    return res


def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    score_number = knn.score(X_test, y_test)
    return score_number


def accuracy_plot():
    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y),
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2),
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8)
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    plt.show()




if __name__ == '__main__':
    # answer_one()
    # answer_two()
    # answer_three()
    # answer_four()
    # answer_five()
    # answer_six()
    # answer_seven()
    # answer_eight()
    accuracy_plot()

