# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import packages and display the data.

2.Print placement and salary data.

3.Check for null and duplicate values.

4.Use Logistic Regression to predict accuracy and confusion matrix.

5.Display the results


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AMAN ALAM
RegisterNumber:  212224240011

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Akash M
RegisterNumber:  212224230013
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from google.colab import files
uploaded = files.upload()
import pandas as pd
import io

data = pd.read_csv(io.BytesIO(uploaded['PlacementData.csv']))
data.head()
# Copy data and drop unnecessary columns
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

# Check for missing values and duplicates
data1.isnull().sum()
data1.duplicated().sum()

# Encode categorical variables using LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

data1.head()
# Separate features and target
x = data1.iloc[:, :-1]  # Features
y = data1["status"]     # Target (Placement Status)
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# Initialize and train Logistic Regression model
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
# Predict on test data
y_pred = lr.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report1 = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report1)
# Predict placement status for a new student
new_data = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
placement_status = lr.predict(new_data)
print("Predicted Placement Status:", placement_status)

*/
```

## Output:
![image](https://github.com/user-attachments/assets/c7589b3e-3ea9-4ab1-90da-725b8646d2b6)

![image](https://github.com/user-attachments/assets/30d53afc-00dd-4168-ab9e-aacbedee0344)

![image](https://github.com/user-attachments/assets/2f1a2bbe-9108-4b52-ab57-c2892079d158)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
