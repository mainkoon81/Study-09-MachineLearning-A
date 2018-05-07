# Study-09-MachineLearning-A

----------------------------------------------------------------------------------------------------------------------------------------
## Warming-Up
> If our **df** has columns(x1, x2, y), we need to split the input and output into Numpy Arrays, in order to apply the classifiers in **scikit learn**...saying that..we convert a **Series** into a **NumpyArray**.
```
X = np.array(df[['x1','x2']])
y = np.array(df['y'])
```
> **Train** models in **sklearn**: `classifier.fit()`
```
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X,y)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X,y)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X,y)

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X,y)
```
<img src="https://user-images.githubusercontent.com/31917400/39083317-9f985d68-455a-11e8-9e54-426359e1c486.jpg" />

Logistic Regression didn't do so well, as it's a linear algorithm. Decision Trees and SVM managed to bound the data well, but..what about this?

[Note] Let's play with some of these parameters of SVM and tune them in such a way that they bound the desired area! The kernel that works best here is 'rbf', with large values of 'gamma'.
 - **kernel**(string): 'linear', 'poly', 'rbf'.
   - **C**(float): for 'linear'. wiggle, wiggle
   - **degree**(integer): for 'polynomial' kernel only.
   - **gamma**(float): for 'rbf' kernel only. Right in ur face.
```
classifier = SVC(kernel = 'rbf', gamma = 200)
classifier.fit(X,y)
```
# Here, the problem is your `X` and `y`. Before training, note "how to separate into training set & testing set"!
## `train_test_split(X, y, test_size)`
 - X_train: The training input
 - X_test: The testing input
 - y_train: The training labels
 - y_test: The testing labels
```
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
```
# Another problem is 









-------------------------------------------------------------------------------------------------------------------------------------
> Performance: How well is my model doing ? (Model-Validation with test results)
 - Regression: it predicts or returns a **value**
 - Classification: it determines or returns a **state**(+/-, Y/N, Cat/Dog/Bird...where the data-pt belongs to..)
<img src="https://user-images.githubusercontent.com/31917400/39257585-cac6de22-48a9-11e8-8f45-1bad945142f6.jpg" />

# 1) Validation I. (Classifier Model)
 - with `**Confusion-Matrix**`
<img src="https://user-images.githubusercontent.com/31917400/39336893-b9a1263a-49b1-11e8-88c1-d59895c7dbe4.jpg" />

## Metric_01: Accuracy...`accuracy_score(y_true, y_pred)` 
```
from sklearn.metrics import accuracy_score

accuracy_score(y_true, y_pred)
```
---------------------------------------------------------------------------------------------------------------------------------------
### But..when Accuracy won't work ?
<img src="https://user-images.githubusercontent.com/31917400/39308440-594fbf20-495d-11e8-97f5-cd51d15696d7.jpg" />

## Metric_02: Precision, Recall and F_Beta Score
Plus, between these two errors, sometimes, depending on situation, the one is more critical over others (FP vs FN)
 - Precision (to capture **FP** and avoid): Out of all data-pt our model diagnosed with **Positive** world, how many did our model classify correctly ? 
 - Recall (to capture **FN** and avoid): Out of all data-pt that are actually **positive**, how many did our model classify correctly ? 
<img src="https://user-images.githubusercontent.com/31917400/39315393-7321c1a8-496e-11e8-8875-20948e25ceab.jpg" />

#### a) F1-Score (Precision ~ Recall)
 - We can combine these two metrics into one metric, using **'Harmonic Mean'** which is called **'F1-Score'**(2xy/x+y)
 - F1-Score is closer to the **smallest** b/w Precison and Recall. If one of them is particularly low, the F1-Score kind of raises a flag ! 
<img src="https://user-images.githubusercontent.com/31917400/39317672-15d0cf0c-4974-11e8-90e7-a950a87be5e2.jpg" />

#### b) F_Beta-Score (Precision with 'FP' vs Recall with 'FN')
 - When our model care a bit more about the one than the other, we want sth more skewed towards the one.
 - Select Beta !!! 
<img src="https://user-images.githubusercontent.com/31917400/39332689-208dfa1a-49a0-11e8-9d33-d9d7f53a1626.jpg" />

 - Finding a good Beta requires a lot of intuition of our data. For example,
   - Detecting malfunctioning parts in a spaceship ?--(FN Warning: a high recall model): Beta=2
   - Sending phone notifications about videos a user may like ?--(not much costly: a decent precision and a decent recall): Beta=1
   - Sending promotional material in the mail to potential clients ?--(costly, so FP Warning: a high precision model): Beta=0.5 

## Metric_03: Roc-Curve (Receiver Operating Characteristic)
 - In the chart of "TPr vs FPr", **the area** under the curve is our metric value. 
 - Consider the data which is now one dimensional, so all the red, blue pt lie in 1 line and we want to find the correct **split**.
<img src="https://user-images.githubusercontent.com/31917400/39336446-7db2501a-49af-11e8-8248-87bbb0757c1d.jpg" />

 - How to construct those values? Examine every inch of the model based on "TPr&FPr"...For example, 
<img src="https://user-images.githubusercontent.com/31917400/39336710-ec5aa66a-49b0-11e8-97c3-8a86ec1b1800.jpg" />

# 2) Validation II. (Regression Model) 
 - with `**R-Squared**`

## Metric_01: MSE & R-Squared
```
from sklearn.metrics import r2_score

r2_score(y_true, y_pred)
```
 - Mean **Absolute** Error: The absolute value function is not differentiable. This is not good if we want to use the **gradient descent** methods. 
 - Mean **Squared** Error: To solve this, we use Mean Squared Error.
<img src="https://user-images.githubusercontent.com/31917400/39385600-092c2fa8-4a69-11e8-9df6-46255b58fa3c.jpg" />
 
 - R-Squared: It is based on comparing **our model** to the **simplest model**. 
   - The simplest model: taking avg of all the values (the horizontal line).
   - We wish `"**SSE** of our model" < "**SSE** of the simplest model(SSTo)"`. 
   - The error of our model(Full Model) becomes 'numerator', the error of the simplest model(Reduced Model) becomes 'denominator'. 
     - SSE: talking about the error of the model(Full Model)
     - SSR: talking about the improvement of the model(Reduced - Full)
     - SSTo: talking about the AVG of the data, so constant. It doesn't rely on the model(Reduced Model)
     - DF for SSE (for **residual**): the sample size - the number of parameters being estimated - intercept
     - DF for SSR: just the number of predictor variables(Reduced - Full)
<img src="https://user-images.githubusercontent.com/31917400/39673142-9d22219c-512e-11e8-8a6b-a10c0009b375.jpg" />

# 3) Model Selection
 - Two Errors:
   - Under-fitting (over-simplication): Error due to **bias**  
   - Over-fitting (over-complication): Error due to **variance** 
<img src="https://user-images.githubusercontent.com/31917400/39399635-489d9a4a-4b19-11e8-8b08-e8125166e173.jpg" />

 - But we cannot use our **test data** to train our model like above(model selection). What should we do ? 
 - **Cross Validation** to select model.
   - the best model-complexity would be the point where these two graphs just start to distance from each other. 
<img src="https://user-images.githubusercontent.com/31917400/39399711-b425a720-4b1a-11e8-8cdf-1736fc1631c8.jpg" />

 - EX) **K-fold** Cross Validation
   - First, dividing our data into a training set / a real-testing set
   - Only in the training set, 
     - 1. Breaking our data into **K-buckets** (K=4)
     - 2. Training our model K times 
       - each time using a different bucket as our **testing set** and the remaining all data-pt(K-1) as our **training set**. 
     - 3. Average the results to get our final model. 
   
<img src="https://user-images.githubusercontent.com/31917400/39400592-443556ca-4b2b-11e8-9aae-85aa4861433c.jpg" />

 - EX) **Learning Curve** (NO.of training_pt VS Error_size)
   - It helps detect overfitting/underfitting
     - Of course, when we train our model with small size of data, testing with CrossValidation will throw large size of error !
   - See where the errors converge to..which will tell under/over-fitting..
<img src="https://user-images.githubusercontent.com/31917400/39400828-b385dde8-4b2f-11e8-92a5-18574c54be5b.jpg" />

## `learning_curve(estimator, X, y)`
```
train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, num_trainings))
```
 - `estimator`: is the actual classifier we're using for the data
   - LogisticRegression(), GradientBoostingClassifier(), SVC(), etc
 - `X` and `y` is our data, split into **features** and **labels**.
 - **train_sizes**: are the sizes of the chunks of data used to draw each point in the curve.
 - train_scores: are the training scores for the algorithm trained on each chunk of data.
 - test_scores: are the testing scores for the algorithm trained on each chunk of data.
   - train_scores and test_scores will come in as a list of 'K'number of values, and this is because the function uses 'K' Fold CrossValidation.
 - we defined our curves with 'Training and Testing Error', but this function defines them with 'Training and Testing Score'. These are opposite, so the higher the error, the lower the score. Thus, when you see the curve, you need to flip it upside down.
    - The Logistic Regression model has a low training and testing score ==> higher error
    - The Decision Tree model has a high training and testing score ==> lower error
    - The Support Vector Machine model has a high training score, and a low testing score ==> lower train error & higher testing error

<img src="https://user-images.githubusercontent.com/31917400/39401453-6f46cd1e-4b3d-11e8-872c-9305d7f40f83.jpg" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve

estimator = LogisticRegression()
estimator = GradientBoostingClassifier()
estimator = SVC(kernel='rbf', gamma=1000)
```

It is good to randomize the data before drawing Learning Curves
```
def randomize(X, Y):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation,:]
    Y2 = Y[permutation]
    return X2, Y2

X2, y2 = randomize(X, y)

def draw_learning_curves(X, y, estimator, num_trainings):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X2, y2, cv=None, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, num_trainings))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.plot(train_scores_mean, 'o-', color="g",
             label="Training score")
    plt.plot(test_scores_mean, 'o-', color="y",
             label="Cross-validation score")
    plt.legend(loc="best")

    plt.show()
```
Therefore, 
 - in Logistic Regression, 
   - we pick the model with the highest F1-Score. 
   - parameters: coefficients of the polynomial 
   - Hyper(meta)-parameter: the degree of the polynomial 
 - in Decision Tree, 
   - we pick the model with the highest F1-Score.
   - parameters: thresholds in the leaves & nodes
   - Hyper(meta)-parameter: the depth of the trees
 - in SVM,
   - we pick the model with the highest F1-Score. 
   - parameters: Kernels(linear/poly/rbf/sigmoid)
   - Hyper(meta)-parameter**s**: C_value, degree, gamma, etc) 
<img src="https://user-images.githubusercontent.com/31917400/39488793-431e5956-4d7b-11e8-94a8-80a5c05852b5.jpg" />

> By the way, in SVM, tuning the parameters can be a lot of work, and GridSearchCV, a sklearn tool can offer an optimal parameter tune almost automatically. The optimization:

**a) Select Parameters**
Let's say we'd like to decide between the following parameters:
 - kernel: `poly` or `rbf`.
 - C: `0.1`, `1`, or `10`.

We pick what are the parameters we want to choose from, and form a `dictionary`. In this dictionary, the keys will be the names of the parameters, and the values will be the lists of possible values for each parameter.
```
parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}
```

**b) Decide Metric**
What metric we'll use to score each of the candidate models ? F1 ? 
```
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

scorer = make_scorer(f1_score)
```

**c) Creast GridSearch Object**
Use GridSearch object to fit the data.
```
from sklearn.model_selection import GridSearchCV

grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
grid_fit = grid_obj.fit(X, y)
```

**d) Get the best model**
```
best_clf = grid_fit.best_estimator_
```
# 4) final example
Improving a model with Grid Search
 - This initial model will overfit heavily. We use Grid Search to find better parameters for this model, to reduce the overfitting.

1. Reading and plotting the data
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
<img src="https://user-images.githubusercontent.com/31917400/39492986-32e28bb2-4d89-11e8-9541-9a59000f7214.jpg" />

2. Splitting our data into training and testing sets
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer

import random
random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
3. Fitting a DecisionTree model. Define the model with default hyperparameters(random_state=42)
```
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)
```
4. plot the model, and find the testing f1_score
```
def plot_model(X, y, clf):
    plt.scatter(X[np.argwhere(y==0).flatten(),0],X[np.argwhere(y==0).flatten(),1],s = 50, color = 'blue', edgecolor = 'k')
    plt.scatter(X[np.argwhere(y==1).flatten(),0],X[np.argwhere(y==1).flatten(),1],s = 50, color = 'red', edgecolor = 'k')

    plt.xlim(-2.05,2.05)
    plt.ylim(-2.05,2.05)
    plt.grid(False)
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off')

    r = np.linspace(-2.1,2.1,300)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((s,t),1)

    z = clf.predict(h)

    s = s.reshape((np.size(r),np.size(r)))
    t = t.reshape((np.size(r),np.size(r)))
    z = z.reshape((np.size(r),np.size(r)))

    plt.contourf(s,t,z,colors = ['blue','red'],alpha = 0.2,levels = range(-1,2))
    if len(np.unique(z)) > 1:
        plt.contour(s,t,z,colors = 'k', linewidths = 2)
    plt.show()

plot_model(X, y, clf)
print('The Training F1 Score is', f1_score(train_predictions, y_train))
print('The Testing F1 Score is', f1_score(test_predictions, y_test))
```
<img src="https://user-images.githubusercontent.com/31917400/39493683-ab52e770-4d8b-11e8-8984-9463eba7e1f5.jpg" />

Woah! Some heavy overfitting there. Not just from looking at the graph, but also from looking at the difference between the high training score (1.0) and the low testing score (0.7).Let's see if we can find better hyperparameters for this model to do better. We'll use grid search for this.

5. Use grid search to improve this model.
 - First define some parameters to perform grid search on. We suggest to play with max_depth, min_samples_leaf, and min_samples_split.
 - Make a scorer for the model using f1_score.
 - Perform grid search on the classifier, using the parameters and the scorer.
 - Fit the data to the new classifier.
 - Plot the model and find the f1_score.
 - If the model is not much better, try changing the ranges for the parameters and fit it again.
<img src="https://user-images.githubusercontent.com/31917400/39605982-f4d2a016-4f2a-11e8-9b71-0cd5e5a13a3f.jpg" />

`DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')`
GridSearch improved the `F1 Score` from 0.7 to 0.8 (and we lost some training score, but this is ok). Also, if you look at the plot, the second model has a much simpler boundary, which implies that it's less likely to overfit.






































































































