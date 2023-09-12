# LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url ="https://bit.ly/w-data"
s = pd.read_csv(url)
print("Import Data")
s.head(10)

s.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hour Studied')
plt.ylabel('Percentage Score')
plt.show()

x = s.iloc[:,:-1].values
y=s.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model  import LinearRegression
reg =LinearRegression()
reg.fit(x_train,y_train)
print("Training Complete")

line = reg.coef_*x+reg.intercept_
plt.scatter(x,y)
plt.plot(x, line);
plt.show()

print(x_test)
y_pred = reg.predict(x_test)

df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


xyz = 9.25
xyz_reshaped = np.array(xyz).reshape(1, -1)
own = reg.predict(xyz_reshaped)
print("No of Hours = {}".format(xyz))
print("Predicted Score = {}".format(own[0]))

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
