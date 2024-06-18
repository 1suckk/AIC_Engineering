import uvicorn
from fastapi import FastAPI

from pydantic import BaseModel

import pickle
import pandas as pd
import numpy as np

from fastapi.middleware.cors import CORSMiddleware
origins = ["*"]

app = FastAPI(title="ML API")

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

with open("girder.pkl", "rb") as fr:
    modelDump = pickle.load(fr)

loadedModel = modelDump["model"]
features = modelDump["features"]
label = modelDump["label"]
preProcessing = modelDump["preprocessing"]

class inDataset(BaseModel):
    inTerm : float
    inThickness : float

xScaler = preProcessing[0]

@app.post("/predict", status_code=200)
async def predict_tf(x: inDataset):
  inTerm = x.inTerm
  inThickness = x.inThickness

  #inTerm = 2700
  #inThickness = 250

  inDf = pd.DataFrame( [[ inTerm, inThickness ]] )
  inDfScaled = xScaler.transform(inDf)

  result = loadedModel.predict(inDfScaled)[0]
  return {"prediction":result}

@app.get("/")
async def root():
    return {"message":"onine"}

#### 실 서버 용
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9999, log_level="debug",
                proxy_headers=True, reload=True)

