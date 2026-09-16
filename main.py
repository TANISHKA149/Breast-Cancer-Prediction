from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
# Load trained model
model = joblib.load("cancer_prediction_app/final_cancer_model.pkl")
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for breast cancer prediction using a trained SVM model",
    version="1.0")
# Exact feature order used during model training
FEATURE_ORDER = ["radius_mean","texture_mean","perimeter_mean","area_mean",
                 "smoothness_mean","compactness_mean", "concavity_mean","concave points_mean",
                 "symmetry_mean","fractal_dimension_mean","radius_se",
                "texture_se","perimeter_se", "area_se", "smoothness_se","compactness_se","concavity_se",
                 "concave points_se","symmetry_se", "fractal_dimension_se", "radius_worst",
                "texture_worst", "perimeter_worst","area_worst","smoothness_worst","compactness_worst",
                "concavity_worst", "concave points_worst", "symmetry_worst","fractal_dimension_worst",
                "radius_perimeter_ratio"]
class CancerInput(BaseModel):
    radius_mean: float = Field(..., gt=0)
    texture_mean: float = Field(..., gt=0)
    perimeter_mean: float = Field(..., gt=0)
    area_mean: float = Field(..., gt=0)
    smoothness_mean: float = Field(..., gt=0)
    compactness_mean: float = Field(..., gt=0)
    concavity_mean: float = Field(..., ge=0)
    concave_points_mean: float = Field(..., ge=0)
    symmetry_mean: float = Field(..., gt=0)
    fractal_dimension_mean: float = Field(..., gt=0)

    radius_se: float = Field(..., gt=0)
    texture_se: float = Field(..., gt=0)
    perimeter_se: float = Field(..., gt=0)
    area_se: float = Field(..., gt=0)
    smoothness_se: float = Field(..., gt=0)
    compactness_se: float = Field(..., gt=0)
    concavity_se: float = Field(..., ge=0)
    concave_points_se: float = Field(..., ge=0)
    symmetry_se: float = Field(..., gt=0)
    fractal_dimension_se: float = Field(..., gt=0)

    radius_worst: float = Field(..., gt=0)
    texture_worst: float = Field(..., gt=0)
    perimeter_worst: float = Field(..., gt=0)
    area_worst: float = Field(..., gt=0)
    smoothness_worst: float = Field(..., gt=0)
    compactness_worst: float = Field(..., gt=0)
    concavity_worst: float = Field(..., ge=0)
    concave_points_worst: float = Field(..., ge=0)
    symmetry_worst: float = Field(..., gt=0)
    fractal_dimension_worst: float = Field(..., gt=0)

    radius_perimeter_ratio: float = Field(..., gt=0)


@app.get("/")
def home():
    return {
        "message": "Breast Cancer Prediction API is running"
    }


@app.post("/predict")
def predict(data: CancerInput):
    input_dict = data.model_dump()

    #Convert API names back to dataset names
    input_dict["concave points_mean"] = input_dict.pop("concave_points_mean")
    input_dict["concave points_se"] = input_dict.pop("concave_points_se")
    input_dict["concave points_worst"] = input_dict.pop("concave_points_worst")
    #IMPORTANT: arrange features in exact training order
    input_data = pd.DataFrame([[input_dict[feature] for feature in FEATURE_ORDER]],
                             columns=FEATURE_ORDER)
    prediction = model.predict(input_data)[0]
    result = "Malignant" if prediction == 1 else "Benign"
    return {"prediction": int(prediction),"result": result}
