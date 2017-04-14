import pandas as pd 
import numpy as np 


df_labels=pd.read_csv('drug-label/drug_labels.csv')
df_recalls=pd.read_csv('drug-enforcement/drug_recalls.csv')


print df_labels.head()
print df_recalls.head()