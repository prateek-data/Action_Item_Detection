# Action_Item_Detection
Code to classify text/email sentences as action items or not 

## How to execute the project
1. `pip install -r requirements.txt` to download package dependencies.
2. Run `download_script.py` to download additional dependencies.
3. Run `server.py` to launch Flask api.
4. Run `run_pipeline.py` to execute entire code pipeline.

### Flask API 
The request object should be of the type application/json. <br/>
Example request object : {"Text": "Please create an assignment and forward it by EOD"}  <br/>
Example response object : {"Text": "Please create an assignment and forward it by EOD", "Is_Actionable": true} <br/>
<br/>

API Endpoints:
- **/nlp_matcher** : Runs NLP rule match algorithm.
- **/ml_classifier** : Runs Machine Learning classifier.
- **/dl_model** : Runs Deep Learning model.

## Project Structure

| Files |	Description |
| ------------- | ------------- |
| download_script.py	| Downloads and installs additional files/dependencies. |
| data_preprocessing.py |	Code to process Enron email dataset. |
| nlp_rule_match.py |	Code for Rule based logic. |
| ml_classifier.py	| Code to train and test Machine Learning models. |
| dl_models.py |	Code to train and test Deep Learning models. |
| run_pipeline.py |	Code to execute entire pipeline. |
| server.py |	Code for Flask api. |

## Project PipeLine
<br/>
1. Load Enron emails and take a sample of 5,000 emails to work on due to memory constraints.
2. Process the emails by extracting message from payload, cleaning the text and tokenizing then into sentences.
3. Use Spacy + Regex logic to create a rule based matcher to classify sentences into Actionable or not.
4. Select 1250 Non Actionable sentences from the Rule based model and add them to Actionable sentences provided by actions.csv file to create dataset.
5. Use dataset to create TF-IDF word embeddings to train and test Machine Learning models.
6. Load Glove Word Embeddings to map dataset text to vectors and use it to train and test Deep Learning models.

## Feature Extraction Process 

**NLP Rule Based Matcher** <br/>
1. Convert each word in sentence to its Part-of-speech tag using Spacy.
2. Use regex to fetch chunks of patterns indicating actionable sentences. <br/> 

**Machine Learning Classifiers** <br/>
1. Convert each word in sentence to its Part-of-speech tag using Spacy.
2. Create n-gram TF-IDF word vectors for all words and pos tags in corpus.
3. Merge the two TF-IDF matrices to create features to train Logistic Regression and Random Forest classifiers. <br/>

**Deep Learning Models** <br/>
1. Load pretrained Glove embeddings and map every word in corpus to its vector form.
2. Use Glove vector mappings to train LSTM and CNN models. <br/>


















