
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


# In[ ]:




# In[ ]:




# In[2]:

df=pd.read_csv('training_data_cleansed.csv')


# In[3]:

nlp_features=[u'adverse_reactions', u'clinical_pharmacology',
       u'contraindications', u'description', u'dosage_and_administration',
       u'how_supplied', u'indications_and_usage', u'overdosage',
       u'spl_product_data_elements']


# In[4]:

df[nlp_features]


# In[5]:

# def turn_list_into_string(list_of_items):
#     new_string=''
#     for item in list_of_items:
#         new_string += item
#         #print new_string,'\n'
#     return new_string

# def stringify_me(list_of_items):
#     return ' '.join(list_of_items)
        


# In[6]:

df['contraindications'][0]


# In[7]:

#df['contraindications'][0]==turn_list_into_string(df['contraindications'][0])


# In[8]:

df['contraindications'][0]


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[9]:

test1=df['adverse_reactions'].fillna('none')


# In[10]:

test1[3]


# In[11]:

for col in nlp_features:
    df[col]=df[col].fillna('none')


# In[12]:

# def clean_nulls(text):
#     if text == 'NaN':
#         text='no value'
#     return text


# In[13]:

#df['test1']=df['adverse_reactions'].apply(clean_nulls)


# In[14]:

df['adverse_reactions'].isnull().sum()


# In[15]:

tfidf_vectorizer=TfidfVectorizer()#ngram_range=(1,3))


# In[16]:

test1=tfidf_vectorizer.fit_transform(df['adverse_reactions'])


# In[17]:

test1.todense().shape


# In[18]:

test1_dense=test1.todense()


# In[19]:

test1_dense[2].sum()


# In[ ]:




# In[20]:

tfidf_vectorizer=TfidfVectorizer()
new_tfidf=pd.DataFrame(tfidf_vectorizer.fit_transform(df['adverse_reactions']).todense())


# In[21]:

new_tfidf.head()


# In[22]:

tfidf_matrix=[]
for col in nlp_features:
    tfidf_vectorizer=TfidfVectorizer()
    new_tfidf=pd.DataFrame(tfidf_vectorizer.fit_transform(df[col]).todense())
    print len(new_tfidf)
    tfidf_matrix.append(new_tfidf)

    
    


# In[23]:

for matrix in tfidf_matrix:
    print matrix.shape


# In[24]:

# df_matrices=tfidf_matrices[0].join(tfidf_matrices[1],rsuffix='_1')
# df_matrices=df_matrices.join(tfidf_matrices[2],rsuffix='_2')
# df_matrices=df_matrices.join(tfidf_matrices[3],rsuffix='_3')
# df_matrices=df_matrices.join(tfidf_matrices[4],rsuffix='_4')
# df_matrices=df_matrices.join(tfidf_matrices[5],rsuffix='_5')
# df_matrices=df_matrices.join(tfidf_matrices[6],rsuffix='_6')
# df_matrices=df_matrices.join(tfidf_matrices[7],rsuffix='_7')
# df_matrices=df_matrices.join(tfidf_matrices[8],rsuffix='_8')


# ''' option 1: do PCA separately on each NLP col before joining, and then only join the required cols, based on checking the explained_variance_ratio_'''

# In[25]:

len(tfidf_matrix)


# In[26]:

# X_std=[]
# for i in range(len(tfidf_matrix)):
#     X_std.append(StandardScaler().fit_transform(tfidf_matrix[i]))


# In[27]:

# pca=PCA()
# pca.fit_transform(tfidf_matrix[0])
# exp_var=pca.explained_variance_ratio_
# print np.cumsum(exp_var)


# In[28]:

tsvd=TruncatedSVD(n_components=100)




# In[29]:

feature_1=tsvd.fit_transform(tfidf_matrix[0])
print 'feature 1 complete'


# In[30]:

feature_2=tsvd.fit_transform(tfidf_matrix[1])
print 'feature 2 complete'


# In[31]:

feature_3=tsvd.fit_transform(tfidf_matrix[2])
print 'feature 3 complete'


# In[32]:

feature_4=tsvd.fit_transform(tfidf_matrix[3])
print 'feature 4 complete'


# In[33]:

feature_5=tsvd.fit_transform(tfidf_matrix[4])
print 'feature 5 complete'


# In[34]:

feature_6=tsvd.fit_transform(tfidf_matrix[5])
print 'feature 6 complete'


# In[35]:

feature_7=tsvd.fit_transform(tfidf_matrix[6])
print 'feature 7 complete'


# In[ ]:

feature_8=tsvd.fit_transform(tfidf_matrix[7])
print 'feature 8 complete'


# In[ ]:

feature_9=tsvd.fit_transform(tfidf_matrix[8])
print 'feature 9 complete'


df_final=pd.DataFrame(feature_1.todense())
df_final=df_final.join(pd.DataFrame(feature_2.todense()))
df_final=df_final.join(pd.DataFrame(feature_3.todense()))
df_final=df_final.join(pd.DataFrame(feature_4.todense()))
df_final=df_final.join(pd.DataFrame(feature_5.todense()))
df_final=df_final.join(pd.DataFrame(feature_6.todense()))
df_final=df_final.join(pd.DataFrame(feature_7.todense()))
df_final=df_final.join(pd.DataFrame(feature_8.todense()))
df_final=df_final.join(pd.DataFrame(feature_9.todense()))

df_final.to_csv('pca_v_1.csv')




# In[ ]:

exp_var=tsvd.explained_variance_ratio_
print np.cumsum(exp_var)


# In[ ]:

'''
for item in X_std:




'''


# In[ ]:

# pca=PCA(n_components=2)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

# tfidf_nlp_cols=tfidf_vectorizer.fit_transform(df[nlp_features])


# # In[ ]:

# tfidf_nlp_cols.shape


# In[ ]:

# type(tfidf_nlp_cols)


# # In[ ]:

# tfidf_nlp_cols.todense().shape


# In[ ]:

# tsvd=TruncatedSVD()


# # In[ ]:

# tsvd_transformed=tsvd.fit_transform(tfidf_nlp_cols)


# # In[ ]:




# # In[ ]:




# # In[ ]:




# # In[ ]:

# tsne=TSNE(verbose=10)


# # In[ ]:

# tsne_2_comp=tsne.fit_transform(tsvd_transformed)


# # In[ ]:

# plt.scatter(tsne_2_comp[:,0],tsne_2_comp[:,1])


# # In[ ]:




# # In[ ]:




# # In[ ]:




# # In[ ]:

# num_features=[ u'abuse', u'accessories', u'alarms',
#        u'animal_pharmacology_and_or_toxicology', u'ask_doctor',
#        u'ask_doctor_or_pharmacist', u'assembly_or_installation_instructions',
#        u'boxed_warning', u'calibration_instructions',
#        u'carcinogenesis_and_mutagenesis_and_impairment_of_fertility',
#        u'cleaning', u'clinical_studies', u'components',
#        u'controlled_substance', u'dependence', u'diagram_of_device',
#        u'disposal_and_waste_handling', u'do_not_use',
#        u'dosage_forms_and_strengths', u'drug_abuse_and_dependence',
#        u'drug_and_or_laboratory_test_interactions', u'drug_interactions',
#        u'environmental_warning', u'food_safety_warning',
#        u'general_precautions', u'geriatric_use',
#        u'guaranteed_analysis_of_feed', u'health_care_provider_letter',
#        u'health_claim', u'information_for_owners_or_caregivers',
#        u'information_for_patients', u'instructions_for_use',
#        u'intended_use_of_the_device', u'labor_and_delivery',
#        u'laboratory_tests', u'mechanism_of_action', u'microbiology',
#        u'nonclinical_toxicology', u'nonteratogenic_effects',
#        u'nursing_mothers', u'other_safety_information',
#        u'patient_medication_information', u'pediatric_use',
#        u'pharmacodynamics', u'pharmacogenomics', u'pharmacokinetics',
#        u'precautions', u'pregnancy', u'pregnancy_or_breast_feeding',
#        u'questions', u'recent_major_changes', u'residue_warning', u'risks',
#        u'route', u'safe_handling_warning', u'spl_indexing_data_elements',
#        u'spl_medguide', u'spl_patient_package_insert',
#        u'statement_of_identity', u'summary_of_safety_and_effectiveness',
#        u'teratogenic_effects', u'troubleshooting',
#        u'use_in_specific_populations', u'user_safety_warnings', u'version',
#        u'veterinary_indications', u'warnings_and_cautions', u'when_using',
#        u'date', u'month', u'year']


# # In[ ]:

# features=[u'adverse_reactions', u'clinical_pharmacology',
#        u'contraindications', u'description', u'dosage_and_administration',
#        u'how_supplied', u'indications_and_usage', u'overdosage',
#        u'spl_product_data_elements', u'abuse', u'accessories', u'alarms',
#        u'animal_pharmacology_and_or_toxicology', u'ask_doctor',
#        u'ask_doctor_or_pharmacist', u'assembly_or_installation_instructions',
#        u'boxed_warning', u'calibration_instructions',
#        u'carcinogenesis_and_mutagenesis_and_impairment_of_fertility',
#        u'cleaning', u'clinical_studies', u'components',
#        u'controlled_substance', u'dependence', u'diagram_of_device',
#        u'disposal_and_waste_handling', u'do_not_use',
#        u'dosage_forms_and_strengths', u'drug_abuse_and_dependence',
#        u'drug_and_or_laboratory_test_interactions', u'drug_interactions',
#        u'environmental_warning', u'food_safety_warning',
#        u'general_precautions', u'geriatric_use',
#        u'guaranteed_analysis_of_feed', u'health_care_provider_letter',
#        u'health_claim', u'information_for_owners_or_caregivers',
#        u'information_for_patients', u'instructions_for_use',
#        u'intended_use_of_the_device', u'labor_and_delivery',
#        u'laboratory_tests', u'mechanism_of_action', u'microbiology',
#        u'nonclinical_toxicology', u'nonteratogenic_effects',
#        u'nursing_mothers', u'other_safety_information',
#        u'patient_medication_information', u'pediatric_use',
#        u'pharmacodynamics', u'pharmacogenomics', u'pharmacokinetics',
#        u'precautions', u'pregnancy', u'pregnancy_or_breast_feeding',
#        u'questions', u'recent_major_changes', u'residue_warning', u'risks',
#        u'route', u'safe_handling_warning', u'spl_indexing_data_elements',
#        u'spl_medguide', u'spl_patient_package_insert',
#        u'statement_of_identity', u'summary_of_safety_and_effectiveness',
#        u'teratogenic_effects', u'troubleshooting',
#        u'use_in_specific_populations', u'user_safety_warnings', u'version',
#        u'veterinary_indications', u'warnings_and_cautions', u'when_using',
#        u'date', u'month', u'year']


# # In[ ]:

# len(features)


# # In[ ]:

# X_train,X_test,y_train,y_test=train_test_split(df[features],df['target'],test_size=0.3, random_state=100)


# # In[ ]:

# lr=LogisticRegression(n_jobs=-1)


# # In[ ]:

# lr.fit(X_train,y_train)


# # In[ ]:

# lr.score(X_train,y_train)


# # In[ ]:

# predictions=lr.predict(X_train)


# # In[ ]:

# predictions.sum()


# # In[ ]:



