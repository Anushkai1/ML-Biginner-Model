# Favorite Film Category Prediction based on Age and Gender

## Overview
This project explores the use of machine learning to predict a person's favorite film category based on their age and gender. It utilizes a simple dataset to demonstrate the process of building a classification model.

## Project Steps

### 1. Import libs
```py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree
```

### 2. Load CSV

```py
df = pd.read_csv("film_data.csv")
```

### 3. Split features and target

```py
X = df.drop(columns=['Favorite_Film_Category'])
Y = df['Favorite_Film_Category']
```

### 4. Split X,Y for train and test

```py
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values , test_size = 0.2) # Test size 20% of all data
```

### 5. Initiate and train model

```py
model2 = DecisionTreeClassifier()
model2.fit(X_train,Y_train) #model.fit(Features,Target) 
```

### 6. Test model

```py

predictions = model.predict(X_test)

score = accuracy_score(Y_test, predictions)
print(score)
```

### 7. Persit Model

```py
# Persiting allow Model save can Predict later without training
joblib.dump(model2, "film_recommonder.joblib")
```

### 8. Load Model

```py
model3 = joblib.load("film_recommonder.joblib")
predictions = model3.predict( [ [21, 0] ] )
print(predictions)
```

### 9. Export Dicesion Tree

```py
tree.export_graphviz(model2,
                     out_file="film_recommonder_tree.dot",
                     feature_names = ["Age", "Gender"],
                     class_names = sorted(Y.unique()),
                     label='all',
                     filled=True,
                     rounded=True
                    )
```
<br>
<h2><a href="https://youtu.be/7eh4d6sabA0">Watch tutorial in Youtube</a></h2>

