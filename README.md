## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
 import pandas as pd
 df=pd.read_csv("/content/Encoding Data.csv")
 df
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/8280bc5d-5ce7-4130-b866-ee527ec23b60)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/7b91ac9b-22e7-411f-a710-2dd86be36f4b)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/a1ff83d9-72bb-4b1b-904c-93142e679ac6)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/b0bcb8c1-d6ac-4ca7-9222-29ad2ab16df4)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/2bf89dbe-04f2-4ace-96da-23f188657579)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/5970afc4-fb93-47d0-b226-9d61828f5a9e)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/402347bd-6159-4d96-815d-7f25fb85ea1f)
```
pip install --upgrade category_encoders
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/ecf7e327-e684-44a5-813f-241e2503e707)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
fb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/64af0072-185d-43ca-855f-b6708b44197b)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/c467077d-c5c3-4291-b033-69c3562db2b6)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/a28b854f-32cc-4b16-9f99-f2e48c9e0f39)

```
df.skew()
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/7027ced3-64bb-43fa-8332-2a732e950573)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/ae9dfdf4-2d38-4ba5-89b0-f0679c6bb3d8)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/59e3fdc0-4278-4cf7-86bf-b06ffee2f5ec)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/2a1ad552-d4e5-44df-b3fd-5a58fbab836f)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/8cd74655-507b-4a4b-88c9-23cadbcca0a6)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"]) df
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/a1b3971f-a889-4944-85b9-f76811232b0d)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/a5c30d89-5fc2-4791-a4b9-81a7cd543cbd)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/452920da-415c-41be-a2ab-1888ee114b01)
```
import matplotlib.pyplot as plt import seaborn as sns import statsmodels.api as sm import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/5e9a5f91-94cd-4bb4-8886-fcf981ca4399)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/4375be76-06d0-4d00-ad5e-200184e67191)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/praveenv23013808/EXNO-3-DS/assets/145824728/bf861332-31c8-4a21-a7fa-fe0276134c67)

# RESULT:
  Hence performing Feature Encoding and Transformation process is Successful.

       
