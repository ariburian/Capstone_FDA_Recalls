import pandas as pd
import numpy as np 
import time
import datetime

df1=pd.read_csv('drug_enforcement_class_1_2012.csv')
df2=pd.read_csv('drug_enforcement_class_1_2013.csv')
df3=pd.read_csv('drug_enforcement_class_1_2014.csv')
df4=pd.read_csv('drug_enforcement_class_1_2015.csv')
df5=pd.read_csv('drug_enforcement_class_1_2016.csv')
df6=pd.read_csv('drug_enforcement_class_2_2012.csv')
df7=pd.read_csv('drug_enforcement_class_2_2013a.csv')
df8=pd.read_csv('drug_enforcement_class_2_2013b.csv')
df9=pd.read_csv('drug_enforcement_class_2_2014a.csv')
df10=pd.read_csv('drug_enforcement_class_2_2014b.csv')
df11=pd.read_csv('drug_enforcement_class_2_2015a.csv')
df12=pd.read_csv('drug_enforcement_class_2_2015b.csv')
df13=pd.read_csv('drug_enforcement_class_2_2015c.csv')
df14=pd.read_csv('drug_enforcement_class_2_2016.csv')

frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14]

df = pd.concat(frames)

def get_ndc(value):
    value_split=value.split(' ')
    for i,item in enumerate(value_split):
        if (item=='NDC'):
            if ('#s' in value_split[i+1]):
                ndc_num=value_split[i+2]
            else:
                ndc_num=value_split[i+1]
                ndc_num=ndc_num.replace('#','')
            ndc_num=ndc_num.replace(',','')
            ndc_num=ndc_num.replace('.','')
                
            return ndc_num



df['unq_ndc']=df['Product Description'].apply(get_ndc)


def time_to_unix(date):
    unix_time=time.mktime(datetime.datetime.strptime(date, "%m/%d/%Y").timetuple())
    return unix_time

df['unix']=df['Recall Initiation Date'].apply(time_to_unix)

df_target=df[['unix','unq_ndc']]

print df_target.head(20)
print df_target.shape
print df['unq_ndc'].isnull().sum()

df_target.to_csv('drug_recalls.csv')