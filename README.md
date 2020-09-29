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
