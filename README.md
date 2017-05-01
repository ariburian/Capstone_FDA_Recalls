### Capstone:  FDA Recalls

This is the site of scripts I wrote to build my capstone project at General Assembly in Spring 2017.

The project is not yet complete, but a significant portion of work has been completed.

The project goal is to build a model that can use Natural Language Processing on open-source data obtained from the FDA on all approved drugs, to predict which of these drugs will be recalled (Class I or Class II recall) within 2 years of their release.  Less than 1% of drugs experience a recall, so my model is scored based on its precision and recall on drugs it predicts will be recalled.

I gathered the data in JSON format and merged and cleaned the dataset in these jupyter notebooks:

[clean_data.py](https://github.com/ariburian/Capstone_FDA_Recalls/blob/master/capstone_project/datasets/drug-label/clean_data.py)

[labels_w_recalls.ipynb](https://github.com/ariburian/Capstone_FDA_Recalls/blob/master/capstone_project/datasets/labels_w_recalls.ipynb)

[initial_modeling.ipynb](https://github.com/ariburian/Capstone_FDA_Recalls/blob/master/capstone_project/modeling/initial_modeling.ipynb)

I then identified features on which to base my model, and performed Tfidf on the desired features, and used dimensional reduction methods to make the data manageable and efficient, and then fed the data into a Random Forest Classifier, followed by a grid search cross-validation.  This can be seen in these jupyter notebooks:

[PCA version 2.ipynb](https://github.com/ariburian/Capstone_FDA_Recalls/blob/master/capstone_project/modeling/PCA%20version%202.ipynb)

[initial_modeling.ipynb](https://github.com/ariburian/Capstone_FDA_Recalls/blob/master/capstone_project/modeling/initial_modeling.ipynb) 

[Random Forest model.ipynb](https://github.com/ariburian/Capstone_FDA_Recalls/blob/master/capstone_project/modeling/Random%20Forest%20model.ipynb)

My project isn’t yet completed, but at present it is able to predict drugs that will be recalled with 0.8% precision and 9% recall, which might not sound like a lot, but for a patient, doctor or pharmaceutical investor to see a short list of 1,500 out of 40,000 new drugs that are in the ‘heighted alert for potential recall’ category, they might give a little more thought before getting involved with these drugs.
