#! python3
# -*- coding: utf-8 -*-
"""
in statistical terms the input variables
are called independent variables and the
outcome variables are termed dependent variables
in machine learning we use the term features to
refer to the input, or independent variables
and target value or target label to refer to the
output, dependent variables

so widely used method for estimating w and b for
linear aggression problems is called least-squares linear regression
also known as ordinary least-squares

the learning algorithm finds the parameters that optimize an objective
function, typically to minimize some kind of loss function of the predicted

Regularization for SVMs: the C parameter
The strength of regularization is determined by C
Larger values of C: less regularization
Fit the training data as well as possible
Each individual data point is important to classify correctly

Smaller values of C: more regularization
More tolerant of errors on individual data points

非线性监督学习模型：由linear support vector machines -> kernelized support vector machines(也就是通常说的SVM支持向量机)
can provide more complex models that can go beyond linear decision boundaries, SVM can be used both classification
and regression

in essence, one way to think about what kernelized SVMs do, is they take the original input data space
and transform it to a new higher dimensional feature space, where it becomes much easier to classify
the transform to data using a linear classifier.

support vector machine with RBF kernel: using both C and gamma parameter
this example from the notebook shows the effect of varing C and gamma together.
if gamma is large, then C will have little to no effect.well, if gamma is small,
the model is much more constrained and the effective C will be similar to how it
would affect to a linear classifier.Typically, gamma and C are tuned together,
with the optimal combination typically in an intermediate range of values.

application of SVMs to a real dataset: unnormalized data
we can see the results with training set accuracy of 1.00 and
the test set accuracy of 0.63 that show that the support vector machine is over fitting

kernelized support vector machines(SVC):
important parameters

model complexity:
kernel:Type of kernel function to be used
default='rbf' for radial basis function
other types include 'polynomial'

kernel parameters
gamma(y):RBF kernel width
C:regularization parameter
Typically C and gamma are tuned at the same time

cross-validation
So far we've seen a number of supervised learning methods,
and when applying you to these methods we followed a consistent series of steps.
First, partitioning the data set into training and test sets using the Train/Test split function.
Then, calling the Fit Method on the training set to estimate the model.
And finally, applying the model by using the Predict Method to estimate a target value for the new data instances,
or by using the Score Method to evaluate the trained model's performance on the test set.

Cross-validation is a method that goes beyond evaluating a single model using a single Train/Test split of the data by
using multiple Train/Test splits, each of which is used to train and evaluate a separate model.

Cross-validation basically gives more stable and reliable estimates of how the classifiers likely to perform on average
by running multiple different training test splits and then averaging the results, instead of relying entirely on a
single particular training set.

The most common type of cross-validation is k-fold cross-validation most commonly with K set to 5 or 10.

So when you ask scikit-learn to do cross-validation for a classification task, it actually does instead what's called
"Stratified K-fold Cross-validation". The Stratified Cross-validation means that when splitting the data,
the proportions of classes in each fold are made as close as possible to the actual proportions of
the classes in the overall data set as shown here.

At one extreme we can do something called "Leave-one-out cross-validation",
which is just k-fold cross-validation, with K sets to the number of data samples in the data set.
In other words, each fold consists of a single sample as the test set and the rest of the data as the training set.
"""
import numpy as np
# import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 15
x = np.linspace(0, 10, n) + np.random.randn(n) / 5
y = np.sin(x) + x / 6 + np.random.randn(n) / 10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)
    plt.show()

# NOTE: Uncomment the function below to visualize the data, but be sure
# to **re-comment it before submitting this assignment to the autograder**.
"""
Question 1
Write a function that fits a polynomial LinearRegression model on
the training data X_train for degrees 1, 3, 6, and 9.
(Use PolynomialFeatures in sklearn.preprocessing to create
the polynomial features and then fit a linear regression model)
For each model, find 100 predicted values over the interval x = 0 to 10 (e.g. np.linspace(0,10,100))
and store this in a numpy array. The first row of this array should correspond to the output from the
model trained on degree 1, the second row degree 3, the third row degree 6, and the fourth row degree 9.
"""


def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    """
    这题不会。。。
    """
    # poly = PolynomialFeatures(2)
    # X_poly = poly.fit_transform(X_train.reshape(11, 1))
    # print(X_poly.shape)
    # print(np.linspace(0,10,100).shape)
    # print(poly.fit_transform(np.linspace(0,10,100).reshape(100,1)).shape)
    # print(np.zeros((4,100)))

    result = np.zeros((4,100))
    for i, degree in enumerate([1, 3, 6, 9]):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train.reshape(11, 1))
        linreg = LinearRegression().fit(X_poly, y_train)
        y = linreg.predict(poly.fit_transform(np.linspace(0,10,100).reshape(100,1)))
        result[i, :] = y
    print(result)
    return result


"""
Write a function that fits a polynomial LinearRegression model
on the training data X_train for degrees 0 through 9.
For each model compute the  R^2  (coefficient of determination)
regression score on the training data
as well as the the test data, and return both of these arrays in a tuple.
This function should return one tuple of numpy arrays (r2_train, r2_test).
Both arrays should have shape (10,)
"""


def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    # poly = PolynomialFeatures(degree=2)
    # print(X_test.shape)
    # r2_train = np.zeros(10)
    # print(r2_train.shape)

    r2_train = np.zeros(10)
    r2_test = np.zeros(10)
    for degree in range(10):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train.reshape(11, 1))
        linreg = LinearRegression().fit(X_poly, y_train)
        r2_train[degree] = linreg.score(X_poly, y_train)
        X_test_poly = poly.fit_transform(X_test.reshape(4, 1))
        r2_test[degree] = linreg.score(X_test_poly, y_test)
    return r2_train, r2_test


def answer_three():
    import numpy as np
    import matplotlib.pyplot as plt

    r2_train, r2_test = answer_two()
    degrees = np.arange(0,10)
    plt.figure()
    plt.plot(degrees, r2_train, degrees, r2_test)
    plt.show()
    """
    underfitting overfitting good_generalization
    蓝色: train_data
    黄色: test_data
    """
    return 0, 9, 6


def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    poly = PolynomialFeatures(degree=12)
    X_poly = poly.fit_transform(X_train.reshape(11, 1))
    X_test_poly = poly.fit_transform(X_test.reshape(4, 1))
    linreg = LinearRegression().fit(X_poly, y_train)
    LinearRegression_R2_test_score = linreg.score(X_test_poly, y_test)
    linlasso = Lasso(alpha=0.01,max_iter=10000).fit(X_poly, y_train)
    Lasso_R2_test_score = linlasso.score(X_test_poly, y_test)
    print(LinearRegression_R2_test_score)
    print(Lasso_R2_test_score)
    return LinearRegression_R2_test_score, Lasso_R2_test_score


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
#
#
# mush_df = pd.read_csv('mushrooms.csv')
# mush_df2 = pd.get_dummies(mush_df)
#
# X_mush = mush_df2.iloc[:,2:]
# y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
# X_subset = X_test2
# y_subset = y_test2


def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier().fit(X_train2, y_train2)
    features = []
    for feature, importance in zip(X_train2.columns, clf.feature_importances_):
        features.append((importance, feature))
    features.sort(reverse=True)
    return [feature[1] for feature in features[:5]]


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    svc = SVC(random_state=0)
    gamma = np.logspace(-4, 1, 6)
    train_scores, test_scores = validation_curve(svc, X_subset, y_subset,
                                                 param_name='gamma', param_range=gamma, scoring='accuracy')
    train_scores = train_scores.mean(axis=1)
    test_scores = test_scores.mean(axis=1)

    return train_scores, test_scores
"""
For this question, we're going to use the validation_curve function in sklearn.model_selection to determine training
and test scores for a Support Vector Classifier (SVC) with varying parameter values. Recall that the validation_curve
function, in addition to taking an initialized unfitted classifier object, takes a dataset as input and does its own
internal train-test splits to compute results.

Because creating a validation curve requires fitting multiple models, for performance reasons this question will use
just a subset of the original mushroom dataset: please use the variables X_subset and y_subset as input to the
validation curve function (instead of X_mush and y_mush) to reduce computation time.

The initialized unfitted classifier object we'll be using is a Support Vector Classifier with radial basis kernel. So
your first step is to create an SVC object with default parameters (i.e. kernel='rbf', C=1) and random_state=0.
Recall that the kernel width of the RBF kernel is controlled using the gamma parameter.

With this classifier, and the dataset in X_subset, y_subset, explore the effect of gamma on classifier accuracy by using
the validation_curve function to find the training and test scores for 6 values of gamma from 0.0001 to 10
(i.e. np.logspace(-4,1,6)). Recall that you can specify what scoring metric you want validation_curve to use by setting
the "scoring" parameter. In this case, we want to use "accuracy" as the scoring metric.
For each level of gamma, validation_curve will fit 3 models on different subsets of the data,
returning two 6x3 (6 levels of gamma x 3 fits per level) arrays of the scores for the training and test sets.
Find the mean score across the three models for each level of gamma for both arrays, creating two arrays of length 6,
and return a tuple with the two arrays.
e.g.
if one of your array of scores is
array([[ 0.5,  0.4,  0.6],
       [ 0.7,  0.8,  0.7],
       [ 0.9,  0.8,  0.8],
       [ 0.8,  0.7,  0.8],
       [ 0.7,  0.6,  0.6],
       [ 0.4,  0.6,  0.5]])
it should then become
array([ 0.5,  0.73333333,  0.83333333,  0.76666667,  0.63333333, 0.5])
This function should return one tuple of numpy arrays (training_scores, test_scores)
where each array in the tuple has shape (6,).
"""


def answer_seven():
    import numpy as np
    import matplotlib.pyplot as plt

    train_scores, test_scores = answer_six()
    gamma = np.logspace(-4, 1, 6)
    plt.figure()
    plt.plot(gamma, train_scores, 'b--', gamma, test_scores, 'g-')
    plt.show()
    return 0.001, 10, 0.1


if __name__ == '__main__':
    # part1_scatter()
    # answer_one()
    # answer_two()
    # answer_three()
    # answer_four()
    # test = map(lambda x: x%2==0, range(10))
    # print(test.__iter__())
    # print(type(test))
    # print(test)