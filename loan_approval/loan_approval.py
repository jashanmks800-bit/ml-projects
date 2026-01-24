import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay

df = pd.read_csv(r'C:\Users\jasha\OneDrive\Desktop\loan_approval_dataset.csv')
df.columns = df.columns.str.strip()
le = LabelEncoder()
y = le.fit_transform(df['loan_status'])

x = df.drop('loan_status', axis=1)

numerical_columns = df.drop(columns=['loan_id','education','self_employed','loan_status']).columns
categorical_columns = ['education','self_employed']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
clf.fit(x_train, y_train)
print("random forest trained successfully")

y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
