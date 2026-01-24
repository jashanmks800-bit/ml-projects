import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)

df = pd.read_csv(r'C:\Users\jasha\OneDrive\Desktop\Telco-Customer-Churn.csv')
print(df.head())
print(df.info())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df['HighValueCustomer'] = (df['MonthlyCharges']>80).astype(int)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

x = df.drop(columns=["customerID", "Churn"])
y = df['Churn']

numerical_cols = x.select_dtypes(include=['int64','float64']).columns
catagorical_cols = x.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

catagorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', catagorical_transformer, catagorical_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight=({0:1, 1:3}),
    ))
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)

model.fit(x_train, y_train)
print("model trained")
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))
