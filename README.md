# Study-09-MachineLearning-A

## CrossValidation ?
 - Cross means AVG.
 - Repeat CV procedure for each ML-modelling method(SVM,DTree,etc)during training, then select the model with minimum CV error. 

## split?
 - Training set: to construct classifier 
   - NB Count frequencies
   - DT: pick attributes to split on
 - Validation set: to pick algorithm and adjustment settings
   - Pick best-performing algorithm (NB vs DT vs …)
   - Fine tune the adjustments (tree depth, k in kNN, c in SVM (tbd)) 
 - Testing set: to estimate future error rate
   - Never report best of any runs 
   - Run only once, or report results of every run
----------------------------------------------------------------------------------------------------------------------------------------
## Training our model
If our **df** has columns(x1, x2, y), we need to split the input and output into Numpy Arrays, in order to apply the classifiers in **scikit learn**...saying that..we convert a **Series** into a **NumpyArray**.
```
X = np.array(df[['x1','x2']])
y = np.array(df['y'])
```
**Train** models in **sklearn**: `classifier.fit()`
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

Logistic Regression didn't do so well, as it's a linear algorithm. Decision Trees and SVM managed to bound the data well, but..what about this crazy dataset on the right ? Only SVM stands alone ? Let's play with some of these parameters of SVM and tune them in such a way that they bound the desired area! The kernel that works best here is 'rbf', with large values of 'gamma'.
   - **kernel**(string): 'linear', 'poly', 'rbf'
   - **C**(float): for 'linear'. wiggle, wiggle
   - **degree**(integer): for 'polynomial' kernel only
   - **gamma**(float): for 'rbf' kernel only. Right in ur face !
```
classifier = SVC(kernel = 'rbf', gamma = 200)
classifier.fit(X,y)
```
Next step would be using our model: `classifier.fit(X,y)` then `classifier.predict(new_X, new_y)`

---------------------------------------------------------------------------------------------------------------------------------------
# How to build and choose our model ? 
Thus, we say:
 - Step_00 Split-I: **split** on data - training set & `real testing set`. Set a side our `real testing set`.
 - Step_01 Split-II: using training set, prepare **Cross-Validation** to increase **fairness**. 
   - Split the training set again(into training and testing). We don't know the best 'K' value yet.
   - Eventually, we decide the optimal size of training or testing set for better modelling. And this hints the best 'K' value. 
 - Step_02 Build and **Optimize** our model before training  
   - Solve the problem of `over/underfitting`, using **'LearningCurve'**. 
     - (the regularization of parameters is available of course). 
   - Solve the problem of `parameter tuning`, using **'GridSearch'**.
 - Step_03 **Fit and Select** the best model/algorithm, using **Cross-Validation** results(metrics) 
   - In each fitting on 'train & test', use our **traditional validation metrics**, then **find the AVG ???**
     - For **Classification** Model: ROC curve is available.
     - For **Regression** Model: R-sqr is available.
 - Step_04: **test** the best model with our `real testing set`: Fit and validate, using our **traditional validation metrics** again.
---------------------------------------------------------------------------------------------------------------------------------------
## > Step_00 Split-I.
### "how to separate our `X` and `y` into training set & testing set"?
## `train_test_split(X, y, test_size, random_state)`
 - X_train: The training input
 - X_test: The testing input
 - y_train: The training **labels**
 - y_test: The testing **labels**
```
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
```
## > Step_01 Split-II.
### Save our original 'testing set' for later, but now focus on 'training set' solely. We use `Cross Validation` on the training set to simulate the validation set(test + train). In the end, `Cross Validation` allows us to `compare different ML algorithms`. What should I use? SVM? KNN? LogisticReg?
 - Let's say we have a dataset. Then we have 2 things to do. 
   - 1. Estimate **parameters** for the ML model. This means: **`training`**
   - 2. Evaluate **performance** for the ML model. How well the choosen ML model works. This means: **`testing`**: then this testing gives us the **MSE**, or accuracy called `cross_val_score(model, X_train, Y_train, cv=kfold)`.  
   - So we split the dataset and train & test with each ML model candidate.    
   <img src="https://user-images.githubusercontent.com/31917400/54567422-a7910a80-49cb-11e9-92bf-d31f92ed0a79.jpg" />
   
### To be more fair !!!
 - ex) **K-fold** Cross Validation
   - First, dividing our data into a training set / a real-testing set
   - Only in the training set, 
     - 1. Breaking our data into **K-buckets** (K=4)
     - 2. Training our model K-times 
       - each time using a **different bucket as our testing set** and the remaining as our **training set**. 
     - 3. After finishing K-times training with K different models, Average the results(such as **accuracy**) to select the best model(parameters). 
     <img src="https://user-images.githubusercontent.com/31917400/54525092-5a7d4c00-496b-11e9-9548-53d3b7528a75.jpg" />
   
 - ex) **Leave one out** Cross Validation
   - Each individual data point becomes a block. so it's a **`N`-fold** Cross Validation. 
 - (+) It matters less how the data gets divided. Every data point gets to be in a test set exactly once! The **variance** of the resulting estimate is reduced `as K is increased`! 
 - (-) The training has to be rerun from scratch K times, which means it takes K times as much computation to make an evaluation. 

## What if our data is **imbalanced**? 
An imbalanced dataset is one where a positive/negative signal occurs in only a small proportion of the total dataset. Often, the minority class in such a dataset will carry an extreme risk if it is not properly detected. 
 - __Metric Problem:__ Normal optimization metrics, such as `accuracy`, may not be indicative of true performance, especially when there is increased risk associated with false-negative or false-positive predictions.
 - __classification Problem:__ The classification output is biased as the classifiers are more sensitive to detecting the majority class and less sensitive to the minority class.
 
One thing we can do to improve our performance is to balance the dataset. We have two options to do this:
 - **`Undersampling`** the **majority** class — Reducing the number of samples from the majority class by randomly selecting a subset of data points from that class to use for training. **But the useful data or information might be thrown away**.
 - **`Oversampling`** the **minority** class — Increasing the number of the samples from the minority class in the training dataset. The common method is to **add copies** of data points from the minority class, which amplifies the decision region resulting in the improvement of evaluation metrics. **But it might result in overfitting**. Another method to oversample, which reduces this issue, is **SMOTE** (Synthetic Minority Oversampling Technique). **SMOTE** is an enhanced sampling method that creates synthetic samples based on the nearest neighbors of feature values in the minority class.
 
> Cross-Validation on Oversampled Data (using SMOTE)
 - In each iteration exclude some data for validation. The excluded data should not be used for feature selection, oversampling and model building.
 - Oversample the minority class only in the training set without the data already excluded for validation.
 - Repeat K times, where K is number of folds.

-------------------------------------------------------------------------------------------------------------------------------------
## > Step_02 Build a model (Fitting_Optimization)
[Note] What's our model ? 
 - Classification model: it determines or returns a **state**(+/-, Y/N, Cat/Dog/Bird...where the data-pt belongs to..)
 - Regression model: it predicts or returns a **value**
<img src="https://user-images.githubusercontent.com/31917400/39257585-cac6de22-48a9-11e8-8f45-1bad945142f6.jpg" />

[Note] Two Errors: Underfitting or Overfitting ?
 - Under-fitting (over-simplication): Error due to **bias**  
 - Over-fitting (over-complication): Error due to **variance** 
<img src="https://user-images.githubusercontent.com/31917400/39722493-d97eebd6-523a-11e8-96a5-5ff3e226e06b.jpg" />

## > Before-validation I: `learning_curve(estimator, X, y)`
Learning curve says how better does the model get at predicting the target as you the increase number of instances used to train it. Learning curve conventionally depicts improvement in performance on the vertical axis when there are changes in another parameter (on the horizontal axis), such as training set size (in machine learning) or iteration/time (in both machine and biological learning). 
 - **Fixing under/overfitting**
 - Collecting additional data in real life scenario might be really time consuming and expensive and doesn't ensure improvement in your model, thus it might be useful to plot these **learning curves** to determine `whether additional data collection is required`. However, some algorithms (mostly those in deep learning) can make use of more and more data to improve their performance.
 - So it hints us the necessary training `sample_size` and necessary `model_complexity`. 
 - How it works?
   - It compares`training set_size` with `Error_size`
   - See where the errors converge to..which will tell under/over-fitting.
   - It spits out the lists of `size of the training set` and the corresponding lists of `scores` of training/testing.
   - As we increase the size of training set, in general, **'training error'** would increase (because of entropy?) while **'testing error'**(fitting on new data) would decrease because our models become better.   
  <img src="https://user-images.githubusercontent.com/31917400/39400828-b385dde8-4b2f-11e8-92a5-18574c54be5b.jpg" />

```
train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, num_trainings))
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
from keras.wrappers.scikit_learn import KerasRegressor

model = LogisticRegression(penalty=’l2’, solver='lbfgs', multi_class='multinomial', n_jobs=-1).fit(X_train, y_train, blah...) 
model = GradientBoostingClassifier(blah..).fit(X_train, y_train, blah...)
model = SVC(kernel='rbf', gamma=1000, blah..).fit(X_train, y_train, blah...)
model = blah..blah...
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
    train_sizes, train_scores, test_scores = learning_curve(estimator, X2, y2, cv=None, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, num_trainings))

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
Finally,
```
model = KerasRegressor(build_fn=base_model_5, verbose=0)
X_LC, Y_LC = randomize(X_train, Y_train) 

draw_learning_curves(X_LC, Y_LC, estimator=model, num_trainings=500)
```

## > Before-validation II: `GridSearchCV(clf, parameters, scoring)`
 - **optimizing parameters**
> Basically, these are what constitute our models, and we would pick the model producing the highest F1-Score.  
 - in Logistic Regression, 
   - **parameters:** coefficients of the polynomial 
   - **Hyper(meta)-parameter:** the degree of the polynomial 
 - in Decision Tree, 
   - **parameters:** thresholds in the leaf & split
   - **Hyper(meta)-parameter:** the depth of the trees
 - in SVM,
   - **parameters:** Kernels(linear/poly/rbf/sigmoid)
   - **Hyper(meta)-parameters:** C_value, degree, gamma, etc) 
<img src="https://user-images.githubusercontent.com/31917400/39488793-431e5956-4d7b-11e8-94a8-80a5c05852b5.jpg" />

## Particularly, in **SVM**, tuning the parameters can be CRAZY, and `GridSearchCV` in a sklearn tool can offer an optimal parameter tune automatically. For example, 
### The optimization in SVM:
 - **a) Select Parameters**
Let's say we'd like to decide between the following parameters:
   - kernel: `poly` or `rbf`.
   - C: `0.1`, `1`, or `10`.
We pick what are the parameters we want to choose from, and form a `dictionary`. In this dictionary, the keys will be the names of the parameters, and the values will be the lists of possible values for each parameter.
```
parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}
```
 - **b) Decide Metric**
What metric we'll use to score each of the candidate models ? F1 ? 
```
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

scorer = make_scorer(f1_score)
```
 - **c) Creast GridSearch Object**
Use GridSearch object to fit the data.
```
from sklearn.model_selection import GridSearchCV

grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
grid_fit = grid_obj.fit(X, y)
```
 - **d) Get the best model**
```
best_clf = grid_fit.best_estimator_
```
[Note]: While `GridSearchCV` works with the explicitly declared values, `RandomizedSearchCV` doesn't search every value. Instead, it samples random subsets and then uses them to find the optimum parameter value, so it's faster. Another great thing about this method is that it samples from value distribution rather than the pre-determined values. RandomizedSearchCV is great when we have a handful no of hyperparameters. 

# NEXT...
---------------------------------------------------------------------------------------------------------------------------------------
## > Step_03 Validation Metrics 
# `Validation For Classifier`
 - Starting point: `**Confusion-Matrix**`
<img src="https://user-images.githubusercontent.com/31917400/39336893-b9a1263a-49b1-11e8-88c1-d59895c7dbe4.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/54570955-8a166d80-49d8-11e9-886a-a1d74c5bba08.jpg" />

 - A confusion matrix tells you What your model did right / wrong. 
 - __FP:__ Positive means REJECTION. It says Rejecting Null Hypothesis which is actually true. 
 - __FN:__ Negative means ACCEPTION. It says Accepting Null Hypothesis which is actually false.

### Metric_01: Accuracy...`accuracy_score(y_true, y_pred)` 
```
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
```
But..when Accuracy won't work ?
<img src="https://user-images.githubusercontent.com/31917400/39308440-594fbf20-495d-11e8-97f5-cd51d15696d7.jpg" />

### Metric_02: Precision, Recall and F_Beta Score...`f1_score(y_true, y_pred)`
Precision means `TP out of all Positive classification` while Recall means `TP out of real Positives`. 
```
from sklearn.metrics import f1_score
f1_score(y_true, y_pred)
```
> Sensitivity and Specificity
 - **Sensitivity(TPR)**: `TP / (TP+FN)` Ho: No disease: What % of **Rejecting Ho** were correctly identified? If it had `FN`, it'd get fucked up! So type-I-err is fine while type-II-err is critical.
   - Comapring it to **emotion**, someone who cry for all sad movies and more, has a high **sensitivity**. If I use him as a detector, I am sure I will detect all sad movies even if risking `FP`: type-I-err...`Detect More Positive than I need`.
   - but **avoiding `FN`: type-II-err at any expense**. 
   
 - Specificity(TNR):TN / (TN+FP) Ho: No disease: What % of **Accepting Ho** were correctly identified?
   - Conversely, some who is difficult to be sad, has a high **specificity**, should be sad by very specific sad movie...Detect Less than I need, so Not guaranteed.
   
> **Precision** and **Recall(Sensitivity)** in the perspective of `TP`(: "Reject HO")
 - Between these two errors(`FP` vs `FN`), sometimes, depending on situation, the one is more critical over others 
 - Precision (to capture **FP** and avoid): Out of all data-pt that **`our model diagnosed with **Positive**`**, how many did our model classify correctly ? 
 - Recall or Sensitivity (to capture **FN** and avoid): Out of all data-pt that are **`actually **positive**`**, how many did our model classify correctly ? 
<img src="https://user-images.githubusercontent.com/31917400/39315393-7321c1a8-496e-11e8-8875-20948e25ceab.jpg" />

a) `F1-Score` (Precision ~ Recall)
 - We can combine these two metrics into one metric, using **'Harmonic Mean'** which is called **'F1-Score'**(2xy/x+y)
 - F1-Score is closer to the **smallest** b/w Precison and Recall. If one of them is particularly low, the F1-Score kind of raises a flag ! 
 - If you are a police inspector and you want to catch criminals, you want to be sure that the person you catch is a criminal (Precision) and you also want to capture as many criminals (Recall) as possible. The F1 score manages this tradeoff.
<img src="https://user-images.githubusercontent.com/31917400/39317672-15d0cf0c-4974-11e8-90e7-a950a87be5e2.jpg" />

b) `F_Beta-Score` (Precision with 'FP' vs Recall with 'FN')
 - When our model care a bit more about the one than the other, we want sth more skewed towards the one.
 - Select Beta !!! 
<img src="https://user-images.githubusercontent.com/31917400/39332689-208dfa1a-49a0-11e8-9d33-d9d7f53a1626.jpg" />

 - Finding a good Beta requires a lot of intuition of our data. For example,
   - Detecting malfunctioning parts in a spaceship ?--(FN Warning: a high recall model): Beta=2
   - Sending phone notifications about videos a user may like ?--(not much costly: a decent precision and a decent recall): Beta=1
   - Sending promotional material in the mail to potential clients ?--(costly, so FP Warning: a high precision model): Beta=0.5 

### Metric_03: Roc-Curve (Receiver Operating Characteristic)
```
from sklearn.metrics import roc_curve
roc_curve(y_true, y_score)
```
**[Threshold]**
<img src="https://user-images.githubusercontent.com/31917400/54679465-41a59f80-4aff-11e9-97c1-1fec0ac95e87.jpg" />

Let's try **different thresholds** for deciding if the sample is Positive / Negative ?
 - For example, to examining ebola infection, let's say if it was super important to correctly classify every (+) sample, we could **set the probability-threshold to 0.1**. `Lowering the threshold !!`
   - `this will increase T**P**R and F**P**R at the same time`.
   - this will also decrease T**N**R and F**N**R at the same time... 
 - The `threshold` can be set to anything b/w 0 and 1, but how do we determine the best threshold ? 
   - For starter, we don't need to test every single option. 
   <img src="https://user-images.githubusercontent.com/31917400/54679868-4a4aa580-4b00-11e9-97e1-123c3f8fb4b1.jpg" />
   
   - Each threshold produces a **confusion matrix**. We cannot produce all confusion matrix as per all thresholod. Instead, ROC curve gives a simple way to summarize all of confusion matrix information: T`P`R(Recall or Sensitivity) vs F`P`R(1-Specificity)
     - **TPR: TP / (TP+FN)** Out of all real Positive, how many Positives are classified correctly? 
     - FPR: FP / (FP+TN) Out of all real Negatives, how many Negatives are classified as Positive?
     - Each `(FPR, TPR)` comes from a threshold. We start by a **`threshold from the bottom`** that classifies all samples as **Positive**. It represents a (1, 1) in the ROC curve. They are all damn Positive ! This means the model correctly classified all Positive samples but incorrectly classified all Negative samples. 
     - As this threshold increases, the point moves toward (0.2, 0.7), then (0.05, 0.6), then (0, 0). This means the model starts to give a bias to the TPR classification in exchange for detecting actual Negative samples(Negative should be Negative, not Positive..so Reduce FPR !). Of course if you can keep TPR as 1, it would be perfect(No need to add bias)!     
   <img src="https://user-images.githubusercontent.com/31917400/54683743-5b98af80-4b0a-11e9-91c8-092dd423882a.jpg" />

In the ROC curve we look at all sort of POSITIVES - TPR, FPR: 
 - In the chart of "TPr vs FPr", **the area** under the curve is our metric value: **AUC**
 - FPr and TPr (ROC metrics) measure the ability to distinguish between the classes.
 - Consider the data which is now one dimensional, so all the red, blue pt lie in 1 line and we want to find the correct **split**.
<img src="https://user-images.githubusercontent.com/31917400/39336446-7db2501a-49af-11e8-8248-87bbb0757c1d.jpg" />

 - How to construct those values? Examine every inch of the model based on "TPr&FPr"...For example, 
<img src="https://user-images.githubusercontent.com/31917400/39336710-ec5aa66a-49b0-11e8-97c3-8a86ec1b1800.jpg" />

<img src="https://user-images.githubusercontent.com/31917400/54684722-fc886a00-4b0c-11e9-8cbe-e3f93b67fc07.jpg" />

> What if the Target Class is imbalanced? 
 - With a large number of (-) samples: **Precision vs Recall(TPR)** is better rather than FPR vs TPR because **Precision is not affected by a large number of negative samples**. This often happens in the study of a rare disease where there are so many people w/o disease (negative samples). 
 - With a large number of (+) samples: the traditional **FPR vs TPR** is better because the precision and recall would reflect mostly the ability of prediction of the positive class and not the negative class which will naturally be harder to detect due to the smaller number of samples.
 - With a balanced: **RocCurve**
 
https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html

https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba 

# `Validation For Regressor`
 - Starting point: `**R2**`

### Metric_01: MSE & R-Squared
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

--------------------------------------------------------------------------------------------------------------------------------------
# Case Study
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
3. Fitting a DecisionTree model. Define the model with default hyperparameters(random_state=42). Note: We didn't use `y_test`. 
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

# What's the difference b/w DecisionTreeClassifier & DecisionTreeRegressor ? 
 - a Decision Tree does Classification, and a DecisionTreeRegressor is a Decision Tree that does Regression. This seems pretty in line with the other class names in the library, e.g. AdaBoostClassifier/Regressor, RandomForestClassifier/Regressor.
 - Scoring func are different..


# study-ML-intro

<img src="https://user-images.githubusercontent.com/31917400/34910969-bf8b8bee-f8b7-11e7-9d23-8d54d974c128.jpg" />

<img src="https://user-images.githubusercontent.com/31917400/35225294-d059a978-ff7f-11e7-9779-b3b51cca7549.jpg" />

---------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------
# project-Python-mini

__Package:__ nltk(natural language toolkit), scikit-learn, numpy, scipy, os, urllib(dataset), tarfile(dataset) 

__Story:__ A couple of years ago, J.K. Rowling (of Harry Potter fame) tried something interesting. She wrote a book, “The Cuckoo’s Calling,” under the name Robert Galbraith. The book received some good reviews, but no one paid much attention to it--until an anonymous tipster on Twitter said it was J.K. Rowling. The London Sunday Times enlisted two experts to compare the linguistic patterns of “Cuckoo” to Rowling’s “The Casual Vacancy,” as well as to books by several other authors. After the results of their analysis pointed strongly toward Rowling as the author, the Times directly asked the publisher if they were the same person, and the publisher confirmed. The book exploded in popularity overnight.

We have a set of emails, half of which were written by one person and the other half by another person at the same company . Our objective is to classify the emails as written by one person or the other based **only on the text of the email.** 

We will start by giving a list of strings. Each string is the text of an email, which has undergone some basic preprocessing; we will then provide the code to split the dataset into training and testing sets.

One particular feature of Naive Bayes is that it’s a good algorithm for working with **text classification.** When dealing with text, it’s very common to treat each unique word as a feature, and since the typical person’s vocabulary is many thousands of words, this makes for a large number of features. The relative simplicity of the algorithm and the independent features assumption of Naive Bayes make it a strong performer for classifying texts. 
 - authors and labels:
   - Sara has label 0
   - Chris has label 1
 - When training you may see the following error: `UserWarning: Duplicate scores. Result may depend on feature ordering.There are probably duplicate features, or you used a classification score for a regression task. warn("Duplicate scores. Result may depend on feature ordering.")` This is a warning that two or more words happen to have the same usage patterns in the emails--as far as the algorithm is concerned, this means that two features are the same. Some algorithms will actually break (mathematically won’t work) or give multiple different answers (depending on feature ordering) when there are duplicate features and sklearn is giving us a warning. Good information, but not something we have to worry about.  
 - If you find that the code is causing memory errors, you can also try setting test_size = 0.5 in the email_preprocess.py file.
```
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
```
 - 'features_train' and 'features_test' are the features for the training and testing datasets, respectively 'labels_train' and 'labels_test' are the corresponding item labels.
```
features_train, features_test, labels_train, labels_test = preprocess()
```


------------------------------------------------------
## 0. DecisionTree
> Titanic: Make predictions about passenger survival
 - Survived: Outcome of survival (0 = No; 1 = Yes)
 - Pclass: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
 - Name: Name of passenger
 - Sex: Sex of the passenger
 - Age: Age of the passenger (Some entries contain NaN)
 - SibSp: Number of siblings and spouses of the passenger aboard
 - Parch: Number of parents and children of the passenger aboard
 - Ticket: Ticket number of the passenger
 - Fare: Fare paid by the passenger
 - Cabin Cabin number of the passenger (Some entries contain NaN)
 - Embarked: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)

Since we're interested in the outcome of survival for each passenger or crew member, we can remove the Survived feature from this dataset and store it as its own separate variable outcomes. We will use these outcomes as our prediction targets.



## 1. NaiveBayes
Q. Create and train a Naive Bayes classifier. Use it to make predictions for the test set. What is the accuracy?

Q. Compare the time to train the classifier and to make predictions with it. What is faster, training or prediction?
 - accuracy: 0.83
 - training time: 1.729 s
 - predicting time: 0.28 s
```
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

t0 = time() ########################################
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" ###

t0 = time() ##########################################
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s" ###

accuracy = clf.score(features_test, labels_test); accuracy
```

## 2. Support Vector Machine
Q. Tackle the exact same email author ID problem as the above project, but now with an SVM. What we find will help clarify some of the practical differences between the two algorithms. It also gives us a chance to play around with parameters a lot more than Naive Bayes did.
 - accuracy: 0.98407281001137659
 - training time: 397.373 s
 - predicting time: 44.041 s
```
from sklearn.svm import SVC
clf = SVC(kernel='linear')

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" 

t0 = time()  
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test); accuracy 
```
### if too slow  
```
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]
```
These lines effectively slice the training dataset down to 1% of its original size, tossing out 99% of the training data. For example, 
### Voice recognition(Siri) and transaction blocking(ATM) need to happen in real time, with almost no delay, thus in this case, time matters.

Q. What about RBF with different 'C' say, C =10., =100., =1000., =10000. ? (the boundary is getting more complex)
 - accuracy: 0.6160, 0.6160, 0.8213, 0.8925 then if we use 'full data', it reaches **99%**
 - training time: 0.201 s
 - predicting time: 2.134 s
```
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10.)

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" 

t0 = time()  
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test); accuracy 
```

## 00. main project
https://www.cs.cmu.edu/~enron/
Mine the email inboxes and financial data of Enron to identify **persons of interest** in one of the greatest corporate fraud cases in American history. We want to see if there are **any patterns within the emails of people** who were persons of interest in the fraud case and see if we can identify those people....Navigate through the data and extract some information !
> What is a **person of interest** ? 
 - indicted
 - settled w/o admitting guilt
 - testified in exchange for immunity

The Enron fraud is a big, messy and totally fascinating story about corporate malfeasance of nearly every imaginable type. The Enron email and financial datasets are also big, messy treasure troves of information, which become much more useful once you know your way around them a bit. We’ve combined the email and finance data into a single dataset. The aggregated Enron email + financial dataset is stored in a dictionary, where each key in the dictionary is a person’s name and the value is a dictionary containing all the features of that person. The email + finance (E+F) data dictionary is stored as a pickle file, which is a handy way to store and load python objects directly. 







