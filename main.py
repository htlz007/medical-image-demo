from fastapi import FastAPI, UploadFile
import torch

app = FastAPI()          # create the app

@app.get("/")            # GET endpoint at root
def health():
    return {"status": "ok"}   # health check

@app.post("/predict")    # POST endpoint that accepts a file
async def predict(file: UploadFile):
    # load your model, run inference, return the result
    return {"class": 0, "prob": 0.87}