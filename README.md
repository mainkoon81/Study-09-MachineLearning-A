# Study-09-MachineLearning-A

----------------------------------------------------------------------------------------------------------------------------------------
## Warming-Up
> 1. If our **df** has columns(x1, x2, y), we need to split the input and output into Numpy Arrays, in order to apply the classifiers in **scikit learn**...saying that..we convert a **Series** into a **NumpyArray**.
```
X = np.array(df[['x1','x2']])
y = np.array(df['y'])
```
> 2. **Train** models in **sklearn**: `classifier.fit()`
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
Let's play with some of these parameters of SVM and tune them in such a way that they bound the desired area! The kernel that works best here is 'rbf', with large values of 'gamma'.
 - **kernel**(string): 'linear', 'poly', 'rbf'.
   - **C**(float): for 'linear' only.
   - **degree**(integer): for 'polynomial' kernel only.
   - **gamma**(float): for 'rbf' kernel only.
```
classifier = SVC(kernel = 'rbf', gamma = 200)
classifier.fit(X,y)
```
> 3. Validation-Prep (data splitting)
 - Regression: it predicts or returns a **value**
 - Classification: it determines or returns a **state**(+/-, Y/N, Cat/Dog/Bird...where the data-pt belongs to..)
<img src="https://user-images.githubusercontent.com/31917400/39257585-cac6de22-48a9-11e8-8f45-1bad945142f6.jpg" />

## Split...`train_test_split(X, y, test_size)`
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
-------------------------------------------------------------------------------------------------------------------------------------
> 4. How well is my model doing ? (Validation with test results)

# 1) Validation (Classifier Model)
 - with **Confusion-Matrix**
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

# 2) Validation (Regression Model) 
 - with **R-Squared**

## Metric_01: Mean Absoulte Error vs. Mean Squared Error vs. R-Squared
 - Mean **Absolute** Error: The absolute value function is not differentiable. This is not good if we want to use the **gradient descent** methods. 
 - Mean **Squared** Error: To solve this, we use Mean Squared Error.
<img src="https://user-images.githubusercontent.com/31917400/39385600-092c2fa8-4a69-11e8-9df6-46255b58fa3c.jpg" />
 
 - R-Squared: It is based on comparing **our model** to the **simplest model**. 
   - The simplest model: taking avg of all the values (the horizontal line).
   - We wish `"**MSE** of our model" < "**MSE** of the simplest model"`. Our model becomes 'numerator', the simplest model becomes 'denominator'.   
<img src="https://user-images.githubusercontent.com/31917400/39385733-96e8ed22-4a69-11e8-89f5-e6162a16d303.jpg" />

# 3) Model Selection
 - Two Errors:
   - Under-fitting (over-simplication): Error due to **bias**  
   - Over-fitting (over-complication): Error due to **variance** 
<img src="https://user-images.githubusercontent.com/31917400/39399635-489d9a4a-4b19-11e8-8b08-e8125166e173.jpg" />

 - But we cannot use our **test data** to train our model like above(model selection). What should we do ? 
 - **Cross Validation** to select model.
   - the best model-complexity would be the point where these two graphs just start to distance from each other. 
<img src="https://user-images.githubusercontent.com/31917400/39399711-b425a720-4b1a-11e8-8cdf-1736fc1631c8.jpg" />

 - For example, **K-fold** Cross Validation
   - Separating our data into a training set / a real-testing set
   - In the training set, 
     - 1. Breaking our data into **K-buckets** (K=4)
     - 2. Training our model K times
       - each time using a different bucket as our **testing set** and the remaining all data-pt as our training set. 
     - 3. Average the results to get our final model. 
   
<img src="https://user-images.githubusercontent.com/31917400/39400592-443556ca-4b2b-11e8-9aae-85aa4861433c.jpg" />

 - Learning Curve (NO.of training_pt VS Error_size)
   - **Learning Curve** helps detect overfitting/underfitting
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






































































































