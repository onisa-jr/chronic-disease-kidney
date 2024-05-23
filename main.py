# onisajr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch import nn
import numpy as np
import joblib
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CDK_ClASSFIER(nn.Module):
    def __init__(self,feature, num_classes=2) -> None:
        super().__init__()
        self.classfier = nn.Sequential(
            nn.Linear(feature, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.classfier(x))


class CDK_stages(nn.Module):
    def __init__(self,feature, num_classes=5) -> None:
        super().__init__()
        self.classfier = nn.Sequential(
            nn.Linear(feature, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classfier(x)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

stage_cdk_model_path = "model/cdk_stages.pth"
class_cdk_model_path = "model/cdk_classfier.pth"
model_stage_cdk = load_model(CDK_stages(13, 5), stage_cdk_model_path)
model_class_cdk = load_model(CDK_ClASSFIER(13), class_cdk_model_path)


# API FOR CHRONIC DISEASES IN KIDNEY
app = FastAPI()

class DETECT_CDK(BaseModel): # stage of cdk
    # Bp,Sg,Al,Su,Rbc ,Bu,Sc,Sod,Pot,Hemo,Wbcc,Rbcc,Htn
    blood_pressure: float
    specific_gravity: float
    albium: float
    sugar: float
    red_blood_cell:float
    blood_urea:float
    serum_creatinine:float
    sodium :float
    potassium:float
    hemoglobin:float
    white_blood_cell_count:float
    red_blood_cell_count:float
    hypertension:float

class CDK_STAGE(BaseModel): # class of cdk
    # bp limit,sg,al,rbc,bu,sc,sod,pot,hemo,wbcc,rbcc,htn,class,stage
    bp_limit: float
    specific_gravity: float
    albium: float
    red_blood_cell:float
    blood_urea:float
    serum_creatinine:float
    sodium :float
    potassium:float
    hemoglobin:float
    white_blood_cell_count:float
    red_blood_cell_count:float
    hypertension:float
    cdk_class: int
    

def normalize_input(data, scaler, model_type:str):
    if model_type == "detect":
        feature_order = [
        "blood_pressure", "specific_gravity", "albium", "sugar", "red_blood_cell", "blood_urea",
        "serum_creatinine", "sodium", "potassium", "hemoglobin", 
        "white_blood_cell_count", "red_blood_cell_count", "hypertension"
        ]

        # Define which features should be normalized and which should not
        features_to_normalize_list = [
            "specific_gravity", "albium", "blood_urea","sugar",
            "serum_creatinine", "sodium", "potassium", "hemoglobin", 
            "white_blood_cell_count", "red_blood_cell_count"
        ]
        exclude_features = ["blood_pressure","red_blood_cell", "hypertension"]

    if  model_type == "stage":
        feature_order = [
        "bp_limit", "specific_gravity", "albium", "red_blood_cell", "blood_urea",
        "serum_creatinine", "sodium", "potassium", "hemoglobin", 
        "white_blood_cell_count", "red_blood_cell_count", "hypertension", "cdk_class"
        ]

        # Define which features should be normalized and which should not
        features_to_normalize_list = [
            "specific_gravity", "albium", "blood_urea",
            "serum_creatinine", "sodium", "potassium", "hemoglobin", 
            "white_blood_cell_count", "red_blood_cell_count"
        ]
        exclude_features = ["bp_limit","red_blood_cell", "hypertension", "cdk_class"]



    data = data.model_dump()
    # Separate features to be normalized and those to be excluded
    features_to_normalize = [data[key] for key in features_to_normalize_list]
    features_to_exclude = {key: data[key] for key in exclude_features}
    
    # Normalize only the selected features
    features_array = np.array(features_to_normalize).reshape(1, -1)
    normalized_array = scaler.transform(features_array)
    
    # Combine normalized features with excluded features in the correct order
    normalized_data = {key: value for key, value in zip(features_to_normalize_list, normalized_array.flatten())}
    normalized_data.update(features_to_exclude)
    
    # Ensure the final list of features is in the correct order
    ordered_data = [normalized_data[feature] for feature in feature_order if feature in normalized_data]
    
    return torch.tensor(ordered_data, dtype=torch.float32).unsqueeze(dim=0)

def predict_cdk_class(args):
    scaler = joblib.load("model/MinMaxScaler_classfier.pkl")
    features = normalize_input(args, scaler, "detect")
    class_cdk = int(torch.argmax(model_class_cdk(features)))
    return class_cdk

def predict_cdk_stages(args):
    scaler = joblib.load("model/MinMaxScaler_stage.pkl")
    features = normalize_input(args, scaler, "stage")
    stage_cdk = int(torch.argmax(F.softmax(model_stage_cdk(features), dim=1)))
    print("stage: ", stage_cdk)
    return stage_cdk

@app.post("/detect_cdk")
def detect_cdk(arg: DETECT_CDK):
    try:
        cdk_class = predict_cdk_class(arg)
        return {"class": cdk_class, "status": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while predicting CDK class: {str(e)}")

@app.post("/cdk_stage")
def get_stage(arg: CDK_STAGE):
    try:
        stage = predict_cdk_stages(arg)
        return {"stage": stage, "status": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while predicting CDK stage: {str(e)}")


if __name__ == "__main__":
    pass