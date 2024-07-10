# Introduction

This project was born out of the need to understand machine learning 
concepts and how to apply them to a real world dataset using modern 
machine learning libraries. 

Jupyter notebook with all the code can be found 
[here](./thyroid_disease/thyroid_disease_classification.ipynb)

## Problem Definition
In our case, the problem we will be exploring is **binary classification** (a sample can only be one of two things). 

This is because we're going to be using a number of differnet **features** (pieces of information) about a person to predict whether they have thyroid cancer or not.

In a statement,

> Given clinical parameters about a patient, can we predict whether or not they have thyroid cancer?

## Data

The original data came from the [Differentiated Thyroid Cancer Recurrence](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence) dataset from the UCI Machine Learning Repository.



Howevever, we've downloaded it in a formatted way from [Kaggle](https://www.kaggle.com/datasets/jainaru/thyroid-disease-data).

The dataset contains 16 attributes (features). **Attributes** (also called **features**) are the variables what we'll use to predict our **target variable**.

Attributes and features are also referred to as **independent variables** and a target variable can be referred to as a **dependent variable**.

> We use the independent variables to predict our dependent variable.

Or in our case, the independent variables are a patients different medical attributes and the dependent variable is whether or not they have thyroid cancer.

## Evaluation

For this project we will be using accuracy as our main metric 
for measuring the performance of our mdoels. Accuracy measures 
how often our model is correct. 
$$\text{Accuracy} = \frac{\text{Correct predictions}}{\text{All predictions}}$$
This metric was chosen beacuse we want to make sure that our model makes 
a little false negative and false postive predictions as possible.

## Tools I Used

- **Python:** Python is the lingua franca of almost anything related to 
machine learning as most ML libraries are python libraries. 
- **Matplotlib:** The most widely used visualization and plotting 
library. 
- **Scikit-Learn:** Scikit-Learn is a very popular machine learning 
library that implements many common ML algorithms and also makes 
implementing ML pipelines easy. 
- **Numpy:** Nupmy is a python library for scientific computing. 
It is commonly used for its array based operations and for its 
computaional speed when performing said scientific computations.
- **Pandas:** Pandas is a python library for data analysis 
and data manipulation tasks. 
 - **Seaborn:** Seaborn is another plotting and visualization 
 python library but it is built upon matplotlib. It is used mainly 
 for creating beautiful statistical graphs and plots.

 # Results

 ## Exploratory Data Analysis (EDA)

 A light EDA was conducted as this is medical data and I am not a 
 subject matter expert in this area. In a real world scenario, 
 one would ideally have access to a subject matter expert 
 that can help in understanding the data and it features.

I will direct you to look at the notebook for this work instead 
of reproducing it here. 

## Machine Learning Modeling

### Models
There were a total of five initial models used for this project:
- Logistic Regression
- k Nearest Neighbor Classifier
- Random Forest Classifier
- SGD Classifier
- Support Vector Machine Classifier

### Preprocessing

There was some light preprocessing done of the data before it was 
used for training. All of the features, except for one, were of 
the categorical type. I standardized the numerical feature 
and used a one hot encoder for the rest of the categorical 
features. A label encoder was also used for the target variable.

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

num_attribs = ["Age"]

cat_attribs = X.columns.to_list()[1:]

num_pipeline = Pipeline([
    ("standarize", StandardScaler())
])

cat_pipeline = Pipeline([
    ("one_hot", OneHotEncoder())
])

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

X_prepared = preprocessing.fit_transform(X)
y_encoded = LabelEncoder().fit_transform(y)
```

### Accuracy With Default Hyperparameters

With the models trained using their default hyperparameters 
we get the results below for accuracy:

| LogisticRegression | RandomForest | KNN | SGDClassifier |
|--------------------|--------------|-----|---------------|
| 94.81%             | 97.40%       | 92.21% | 93.51%     |


### Accuracy After Hyperparameter Optimization

After using RandomizedSearchCV for our hyperparameter 
optimization, we get the results below for the best 
hyperparameters on accuracy:

| LogisticRegression | RandomForest | KNN | SGDClassifier |
|--------------------|--------------|-----|---------------|
| 94.80%             | 94.47%       | 91.85% | 93.16%     |


We can see that there was not much of an improvement and 
even some models performed worse. 

The RandomForestClassifier ended up being the best model overall.

### ROC and AUC Metrics

Below we have the ROC plot along with the AUC metric 
plotted on the lower right corner. This was created using 
a RandomForestClassifer with the best hyperparameters found.

![ROC and AUC Metrics](/thyroid_disease/assets/ROC_AUC.png)

### Confusion Matrix

Below we have the confusion matrix from our best model. 
We can see that there were only 4 misclassifications on 
our test set. There were 4 false negatives predicted.

![Confusion Matrix](/thyroid_disease/assets/Confusion_Matrix.png)

### Voting Classifier 

We will use two new approaches in our classification task. We will use a Voting Classifier and then a Stacking Classifier.

A voting classifier first takes in a list of estimators and then trains them on the training set. When the training of all the classifiers is done 
you can then use the voting classifier to make predictions on new data. It works by either using hard voting or soft voting. 

- Hard voting (a.k.a majority voting), means that the classifier wil predict the class that has been predicted the most by all the classifiers.
- Soft voting works by predicting the class with the highest class probability averaged over all the individual classifiers. 

The image below give an illustartion of the proccess (credit to the book "Hands on Machine Learning"). We will just use hard 
voting for this task to see if our performance metrics improve. We will also use the best hyperparameters that were found 
during hyperparameter optimization for our classifiers.

The hard voting classifier only achieved an accuracy of **94.81%**.

![Voting](/thyroid_disease/assets/voting.png)

### Stacking Classifier

With this method we unfortunately don't achieve any improvement, so we will just move onto the next ensemle method. 
The next ensemble method will be a stacking classifier. A stacking classifier works by having a set of cassifiers make a prediction, 
which then get fed into a blender. The blender then uses those predictions to make the final prediction. This idea is shown in the image below. 
(Credit to the book "Hands on Machine Learning").

The stacking classifier as only achieved an accuracy of **94.81%**.

![Stacking](/thyroid_disease/assets/stacking.png)

# Conclusion

## What I Learned

- **Programming:** This project helped me become more proficient 
in my python programming skills. I learned how to write more reusable 
code by writing better functions, how to debug code quickly, and 
to write more pythonic code.
- **Machine Learning:** I learned about different classification 
models and how to implement them. I also gained experience in 
writing machine learning pipelines for transforming the data 
and making predictions. 
- **Plotting:** I was able to practice and sharpen my plotting 
skills for visualizing the models performance. 

## Optimal Model

In the end it surprisingly turned out that the default hyperparameters 
for the random forest model out performed on accuracy all the other 
models including those that were optimzied using radomized search 
cross-validation.