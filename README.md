# NAME:MAGESHKUMAR U
# REGNO:212224240085
# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
<img width="397" height="245" alt="image" src="https://github.com/user-attachments/assets/9e302750-f4f4-4a05-ba67-38068458cf41" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="129" height="252" alt="image" src="https://github.com/user-attachments/assets/05226e1d-3181-4cc1-acf6-ae56b7858965" />

```
df.dropna()
```
<img width="410" height="518" alt="image" src="https://github.com/user-attachments/assets/e152dad0-52fe-44f2-ae2b-7b8cf4c7838e" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
<img width="138" height="170" alt="image" src="https://github.com/user-attachments/assets/09e0e44c-c48f-4bf0-b352-a8e89a8d39c8" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
<img width="391" height="260" alt="image" src="https://github.com/user-attachments/assets/e915d03b-1f3f-47e4-afb6-fc5cda112cb2" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
<img width="448" height="449" alt="image" src="https://github.com/user-attachments/assets/625c976d-715d-47e0-ad7a-b4ed04ce555c" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="424" height="446" alt="image" src="https://github.com/user-attachments/assets/a2418a3b-b4bd-4802-9c9a-dfdcca1a1e75" />

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="442" height="510" alt="image" src="https://github.com/user-attachments/assets/a1387e64-d637-4465-bb57-6ee1a93deb7e" />

```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
<img width="427" height="250" alt="image" src="https://github.com/user-attachments/assets/131d23d5-f82f-40a7-936f-91bfec61fae1" />

```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="448" height="443" alt="image" src="https://github.com/user-attachments/assets/526a4449-af3a-48df-912d-0a537d850a03" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="192" height="600" alt="image" src="https://github.com/user-attachments/assets/71f367dd-dd8d-4384-8b27-7b049c346963" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="1093" height="512" alt="image" src="https://github.com/user-attachments/assets/f2c857a6-ad00-4401-b7aa-db2dac79da84" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="961" height="506" alt="image" src="https://github.com/user-attachments/assets/361e3530-a1ba-4f8f-81f8-7696d0293859" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="410" height="90" alt="image" src="https://github.com/user-attachments/assets/d9edccce-d2de-4762-b3ef-a3163d174119" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="449" height="446" alt="image" src="https://github.com/user-attachments/assets/18b641bf-1387-41e3-8e7c-55201b87adaf" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="1109" height="519" alt="image" src="https://github.com/user-attachments/assets/0762a4c8-64a1-4558-b719-0a34cfcb3ef0" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="967" height="512" alt="image" src="https://github.com/user-attachments/assets/f6471ff4-1c58-46fc-85eb-abe5dad87540" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
<img width="755" height="98" alt="image" src="https://github.com/user-attachments/assets/16bdde4e-3952-452c-916a-202d40247b65" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="405" height="93" alt="image" src="https://github.com/user-attachments/assets/add0113f-40e1-4db1-b852-33f2a90c07c7" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
<img width="582" height="42" alt="image" src="https://github.com/user-attachments/assets/06dfce46-205f-4921-9fdb-bac86fc8820f" />

```
!pip install skfeature-chappers
```
<img width="1348" height="364" alt="image" src="https://github.com/user-attachments/assets/ca94134b-d033-4f59-b6f6-6abc0f76bebf" />

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="971" height="514" alt="image" src="https://github.com/user-attachments/assets/455ae8ee-bbe3-4419-ae58-c76985f85492" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
<img width="869" height="74" alt="image" src="https://github.com/user-attachments/assets/1a708da4-b2b0-40f5-8db8-0d5f49650889" />

```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="970" height="508" alt="image" src="https://github.com/user-attachments/assets/b2e3c8b8-e3f2-428b-8483-9c7ea3b0a6a8" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
<img width="1319" height="831" alt="image" src="https://github.com/user-attachments/assets/b395ea7b-82af-4ffb-93fc-0d6de6667aac" />

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
