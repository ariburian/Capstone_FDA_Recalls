# Capstone: FDA Recalls

This is the site of scripts I wrote to build my capstone project at General Assembly in Spring 2017.

The project is not yet complete, but a significant portion of work has been completed.

The project goal is to build a model that can use Natural Language Processing on open-source data obtained from the FDA on all approved to drugs, to predict which of these drugs will be recalled (Class I or Class II recall) within 2 years of their release.  Less than 1% of drugs experience a recall, so my model is scored based on its precision and recall on drugs it predicts will be recalled.

I gathered the data in JSON format and merged and cleaned the dataset in these jupyter notebooks:  clean_data.py   and   labels_w_recalls.ipynb

I then identified features on which to base my model, and performed Tfidf on the desired features, and used dimensional reduction methods to make the data manageable and efficient, and then fed the data in a Random Forest Classifier.  This can be seen in this jupyter notebook:

My project isn’t yet completed, but at present it is able to predict drugs that will be recalled with 4% precision and 9% recall, which might not sound like a lot, but for a patient, doctor or pharmaceutical investor to see a short list of 3,000 out of 40,000 new drugs that are in the ‘heighted alert for potential recall’ category, they might give a little more thought before getting involved with these drugs.

