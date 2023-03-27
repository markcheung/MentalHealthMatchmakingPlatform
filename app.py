# Load the libraries
from fastapi import FastAPI, HTTPException
from model import modelNN, Dataset
from torch.utils.data import DataLoader
import torch
import transformers
import torch
import numpy as np
from transformers import BertConfig
from transformers import AutoModelForTokenClassification

issues = ['Anxiety', 'Borderline Personality Disorder', 'Depression',  'Eating Disorder', 'OCD', 'Substance Abuse']
# Load the model
print('loading model')
number_of_issues = 6
          

configuration = BertConfig()
# model = modelNN(configuration, 768, number_of_issues,'bert')
# sd = torch.load('models/model_state_dict.pth', map_location='cpu')
import os
# model  = modelNN.from_pretrained("C:/Users/markc/Documents/Projects/Matchmaking platform/Fastapi-tutorial/trained_model/") #,configuration,768,number_of_issues,'bert')
model  = modelNN.from_pretrained("C:/Users/markc/Documents/Projects/Matchmaking platform/Demo/trained_model/",768,number_of_issues) #,configuration,768,number_of_issues,'bert')

model.eval()    
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to the matchmaking algo platform"}


# Define the route to the sentiment predictor
@app.post("/classify_description")
def classify_description(description):

    if(not(description)):
        raise HTTPException(status_code=400, 
                            detail = "Please provide a valid description")

    data = Dataset([(description, [0,0,0,0,0,0])],96)
    train_dataloader = DataLoader(data)
    feats, _ = next(iter(train_dataloader))
    scores = model(feats)
    scores = torch.sigmoid(scores)
    most_likely_issue = scores.argmax()
    scores = scores.detach().numpy().tolist()
    scores = scores[0]
    # for score in scores:
        # print(score)
    issues_scores = [ issue+': %.3f' % score for issue,score in zip(issues,scores)]
    issues_scores = [ (issue, score) for issue,score in zip(issues,scores)]

    # scores = [ '%.2f' % elem for elem in scores ]
    # print(description, scores)

    return {
            "description": description, 
            "most likely issue": issues[most_likely_issue],
            # "list of issues": issues,
            # "scores": scores,
            "issues scores": issues_scores
            }
