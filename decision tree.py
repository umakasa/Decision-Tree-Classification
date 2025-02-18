import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data=pd.read_csv('/content/IRIS.csv')

x=data.drop('species',axis=1)
y=data['species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)

print(f"accuracy:{accuracy:.4f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))

from sklearn.tree import plot_tree
plot_tree(model,feature_names=x.columns,class_names=np.unique(y),filled=True,rounded=True)

