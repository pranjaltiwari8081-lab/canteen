from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# -----------------------------------
# Initialize App
# -----------------------------------
app = FastAPI(
    title="Canteen Food Recommendation API",
    version="4.0",
    description="Predicts the Top 10 most suitable food items based on user preferences"
)

# -----------------------------------
# Load Model and Scaler
# -----------------------------------
model = joblib.load("item_model.pkl")
scaler = joblib.load("item_scaler.pkl")

# -----------------------------------
# Encodings (as per your mapping)
# -----------------------------------
item_mapping = {
    0: "Biryani",
    1: "Brownie",
    2: "Burger",
    3: "Cold Coffee",
    4: "Idli",
    5: "Pasta",
    6: "Pizza",
    7: "Samosa",
    8: "Sandwich",
    9: "Tea"
}

category_mapping = {
    "Dessert": 0,
    "Drink": 1,
    "Fast Food": 2,
    "Main Course": 3,
    "Snack": 4,
    "Tiffin": 5
}

# -----------------------------------
# Request Schema
# -----------------------------------
class RecommendRequest(BaseModel):
    category: str
    group_size: int
    avg_spend: float
    rating: float
    delivery_time: int

# -----------------------------------
# Response Schema
# -----------------------------------
class RecommendResponse(BaseModel):
    top_items: list[dict]

# -----------------------------------
# Prediction Endpoint
# -----------------------------------
@app.post("/recommend", response_model=RecommendResponse)
def recommend_food(request: RecommendRequest):
    try:
        # Validate category
        if request.category not in category_mapping:
            return {"top_items": [{"error": f"Invalid category: {request.category}"}]}

        # Prepare input
        cate_num = category_mapping[request.category]
        inp = np.array([[cate_num, request.group_size, request.avg_spend, request.rating, request.delivery_time]])
        pre = scaler.transform(inp)

        # Predict top 10 items based on probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(pre)[0]
            top_indices = np.argsort(probs)[::-1][:10]
            top_items = [
                {"item": item_mapping.get(i, "Unknown"), "confidence": round(float(probs[i]), 3)}
                for i in top_indices
            ]
        else:
            # Fallback if model doesn’t support probability
            pred = model.predict(pre)[0]
            top_items = [{"item": item_mapping.get(pred, "Unknown"), "confidence": None}]

        return {"top_items": top_items}

    except Exception as e:
        return {"top_items": [{"error": str(e)}]}
