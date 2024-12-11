#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats
import catboost
from catboost import CatBoostRegressor as cbr
from sklearn.utils import resample
from sklearn.utils import shuffle
import xgboost


df1 = pd.read_csv(r'ch2024_misc_cell_types_specimen_2021_combined_long.csv')
#df1b = df1[[" Age"," Infancy Vaccine"," Ethnicity"," Race"," Biological Sex",'IgG_PT_day0',"Fold Change (day 0 vs day 14)"]]
df1b = df1[[" Age"," Infancy Vaccine"," Biological Sex",'MonocytesT','Classical Monocytes','Fold change - Monocytes']]


lenth = len(df1b.columns)
print(lenth)

#X_train = df1b.iloc[:, :1]
X_train = df1b.iloc[:, :lenth-1]
y_train = df1b.iloc[:, -1]

df2 = pd.read_csv(r'monocytes_day0_2023.csv')
#df2b = df2[["Age"," Infancy Vaccine"," Ethnicity","Race"," Biological Sex",'IgG_PT_day0',"Fold Change (day 0 vs day 14)"]]#"Fold_change"]]
df2b = df2[[" Age"," Infancy Vaccine"," Biological Sex",'MonocytesT','Classical Monocytes']]
#X_test = df2b.iloc[:,:1]
X_test = df2b.iloc[:,:]
#y_test = df2b.iloc[:, -1]
subject = df2["Subject ID"]


cat_predictions = {}

cat_model = cbr(iterations=50, depth=3, learning_rate=0.05, loss_function='RMSE')

cat_feat_indices = np.where(X_train.dtypes != float)[0]

#Train model on training dataset
cat_model.fit(X_train, y_train,cat_features=cat_feat_indices)

y_pred = cat_model.predict(X_test)
print(subject)
print(X_test)
print(y_pred)

for i in y_pred:
	print(i)

#spearman = stats.spearmanr(y_test, y_pred)
#print(spearman)

#spearman1 = stats.spearmanr(y_test, y_pred_1)
#print(spearman1)