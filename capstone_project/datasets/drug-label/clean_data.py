import numpy as np 
import pandas as pd 
from string import ascii_lowercase





df1=pd.read_csv('drug_label_csv/part1.csv',low_memory=False)
df2=pd.read_csv('drug_label_csv/part2.csv',low_memory=False)
df3=pd.read_csv('drug_label_csv/part3.csv',low_memory=False)
df4=pd.read_csv('drug_label_csv/part4.csv',low_memory=False)
df5=pd.read_csv('drug_label_csv/part5.csv',low_memory=False)
df6=pd.read_csv('drug_label_csv/part6.csv',low_memory=False)	


frames = [df1, df2, df3, df4, df5, df6]

df = pd.concat(frames)

id_cols=['id','openfda','package_label_principal_display_panel','references','set_id']
spl_cols=[]
num_cols=['version']
list_of_bool_cols=[]
list_of_nlp_cols=[]

## This removes all of the reduntant 'table' columns from the dataframe
for col in df.columns:
    if 'table'in col:
        del df[col]
    if 'spl' in col:
    	spl_cols.append(col)




for col in df.columns:
	if df[col].isnull().sum()>=len(df)*0.7:
		### if 70% of the column is empty, I'm turning that column into a boolean with a 1 for having a value, else 0
		df[col]=df[col].apply(lambda x: 0 if pd.isnull(x) else 1)
		list_of_bool_cols.append(col)
	elif (col=='effective_time'):
		pass
	else:
		list_of_nlp_cols.append(col)

print "I have %s bool cols" %len(list_of_bool_cols)
print "I have %s nlp cols" %len(list_of_nlp_cols)



df['effective_time']=df['effective_time'].apply(lambda x: str(x))
df['date']=df['effective_time'].apply(lambda x: int(x[6:8]))
df['month']=df['effective_time'].apply(lambda x: int(x[4:6]))
df['year']=df['effective_time'].apply(lambda x: int(x[0:4]))
df=df.drop('effective_time',axis=1)



def get_ndc(value):
    list_of_ndcs=[]
    if type(value)==float:
        return '0'
    value_split=value.split(' ')
    for i,item in enumerate(value_split):
        item=item.lower()
        if (item=='ndc'):
            ndc_num=value_split[i+1]
            ndc_num=ndc_num.lower()
            for c in ascii_lowercase:
                ndc_num=ndc_num.replace(c,'') 
            ndc_num=ndc_num.replace('-','')
            ndc_num=ndc_num.replace("'",'')                   
            ndc_num=ndc_num.replace('#','')
            ndc_num=ndc_num.replace(']','')
            ndc_num=ndc_num.replace(':','')
            ndc_num=ndc_num.replace(',','')
            ndc_num=ndc_num.replace('\\','')
            ndc_num=ndc_num.replace('.','')
            ndc_num=ndc_num.replace('>','')
            ndc_num=ndc_num.replace('*','')
            ndc_num=ndc_num.replace('/','')
            ndc_num=ndc_num.replace('(','')
            ndc_num=ndc_num.replace(')','')
            ndc_num=ndc_num.replace('_','')


            if ndc_num=='':
                ndc_num='0'
            list_of_ndcs.append(int(ndc_num))
    return list_of_ndcs

def get_unq_ndc(value):
    value_split=value.split(' ')
    for i,item in enumerate(value_split):
        if (item=='u\'product_ndc\':'):
            ndc_num=value_split[i+1]
            ndc_num=ndc_num.replace('[','')
            ndc_num=ndc_num.replace(']','')
            ndc_num=ndc_num.replace('u','')
            ndc_num=ndc_num.replace("'",'')
            ndc_num=ndc_num.replace(',','')

            return ndc_num

df['NDCs']=df['package_label_principal_display_panel'].apply(get_ndc)

df['unq_ndc']=df['openfda'].apply(get_unq_ndc)
df['version']=df['version'].apply(lambda x: int(x))

for col in id_cols:
	del df[col]

print df.head()

#for col in list_of_nlp_cols:
'''
##Tfidf
tsvd/PCA
add in date and bools
then return csv to model on in a differnent script
'''



df.to_csv('drug_labels.csv')