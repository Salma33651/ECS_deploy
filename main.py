from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# تحميل النموذج
model = joblib.load("./RF_model_compressed.joblib")

label_names = {
    1: 'Walking',
    3: 'Shuffling',
    4: 'Stairs (Ascending)',
    5: 'Stairs (Descending)',
    6: 'Standing',
    7: 'Sitting',
    8: 'Lying'
}

class InputData(BaseModel):
    back_x: float
    back_y: float
    back_z: float
    thigh_x: float
    thigh_y: float
    thigh_z: float

@app.post("/predict")
def predict(data: InputData):
    # تحويل البيانات إلى DataFrame
    input_df = pd.DataFrame([data.dict()])
    # عمل التنبؤ
    prediction = model.predict(input_df)
    
    # action = " "
    # if prediction[0] != 8:
    #     action = "Not_Fall"
    # else:
    #     action = "Fall"
    return {"prediction": label_names[prediction[0]]}


