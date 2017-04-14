import pandas as pd 
import json




filename='drug-enforcement-0001-of-0001.json'
with open(filename) as json_data:
	d=json.load(json_data)


df=pd.DataFrame(d['results'])
print df.head()

df.to_csv('drug_enforement.csv')