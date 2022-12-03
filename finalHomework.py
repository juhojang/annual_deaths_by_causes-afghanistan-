import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


a=pd.read_csv('C:/Data_Handling/annual_deaths_by_causes.csv')
a=a.loc[:,['country','protein_energy_malnutrition','alcohol_use_disorders','drug_use_disorders','interpersonal_violence']]
a=a.groupby('country')
for key,group in a:
    a=group
    description=group.describe()
    print(description)
    break
fig,ax = plt.subplots(2,1)
ax[0].boxplot([a['protein_energy_malnutrition'],a['alcohol_use_disorders'],a['drug_use_disorders'],a['interpersonal_violence']])
ax[0].set_xticks([1,2,3,4],['protein_energy_malnutrition','alcohol_use_disorders','drug_use_disorders','interpersonal_violence'])
ax[0].set_xlabel('cause of death')
ax[0].set_ylabel('number of death person')
ax[1].scatter(a['alcohol_use_disorders'],a['interpersonal_violence'])
ax[1].set_xlabel('number of death[alcohol_use]')
ax[1].set_ylabel('number of death[interpersonal_violence]')
plt.show()

X_train,X_test,y_train,y_test=train_test_split(a['alcohol_use_disorders'],a['interpersonal_violence'],test_size=0.3,random_state=1)

lr=LinearRegression()
X_train=X_train.values.reshape(21,1)
y_train=y_train.values.reshape(21,1)
X_test=X_test.values.reshape(9,1)
y_test=y_test.values.reshape(9,1)
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
