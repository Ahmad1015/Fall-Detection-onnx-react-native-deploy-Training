"""
Fall Detection Inference Server with Ensemble Voting
=====================================================
Runs ONNX models and applies weighted voting for fall detection.

Usage:
    python inference_server.py

Models used (with voting weights based on recall performance):
- SVM (RBF):            99.35% recall → weight 1.0
- XGBoost:              98.69% recall → weight 0.95
- Random Forest:        98.69% recall → weight 0.95
- Logistic Regression:  97.39% recall → weight 0.90
"""

import os
import json
import time
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque

# Fast API for server
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Installing FastAPI dependencies...")
    os.system("pip install fastapi uvicorn pydantic")
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

# ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    print("Installing ONNX Runtime...")
    os.system("pip install onnxruntime")
    import onnxruntime as ort

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_DIR = "training_result"  # Folder containing ONNX models

# Voting weights based on recall performance
MODEL_WEIGHTS = {
    "SVM": 1.00,              # 99.35% recall - best
    "XGBoost": 0.95,          # 98.69% recall
    "Random Forest": 0.95,    # 98.69% recall
    "Logistic Regression": 0.90  # 97.39% recall
}

# Feature extraction window
WINDOW_SIZE = 200  # samples (~2 seconds at 100Hz)
FALL_THRESHOLD = 0.5  # Probability threshold for fall detection

# ============================================================================
# DATA MODELS
# ============================================================================
class SensorData(BaseModel):
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

class PredictionResponse(BaseModel):
    timestamp: str
    fall_detected: bool
    final_probability: float
    model_predictions: Dict[str, float]
    voting_details: Dict[str, Dict]
    features_used: Dict[str, float]

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def compute_features(acc_data: np.ndarray, gyro_data: np.ndarray) -> np.ndarray:
    """
    Extract features from accelerometer and gyroscope data window.
    
    Features (9 total):
    - acc_max: Maximum acceleration magnitude
    - gyro_max: Maximum gyroscope magnitude  
    - acc_kurtosis: Kurtosis of acceleration magnitude
    - gyro_kurtosis: Kurtosis of gyroscope magnitude
    - lin_max: Maximum linear acceleration (acc - gravity)
    - acc_skewness: Skewness of acceleration magnitude
    - gyro_skewness: Skewness of gyroscope magnitude
    - post_gyro_max: Max gyro in post-impact window
    - post_lin_max: Max linear acc in post-impact window
    """
    from scipy import stats
    
    # Compute magnitudes
    acc_mag = np.sqrt(np.sum(acc_data**2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro_data**2, axis=1))
    
    # Estimate gravity (mean of acc during window)
    gravity = np.mean(acc_data, axis=0)
    lin_acc = acc_data - gravity
    lin_mag = np.sqrt(np.sum(lin_acc**2, axis=1))
    
    # Split for post-impact features (last 1/3 of window)
    split_idx = len(acc_mag) * 2 // 3
    post_gyro = gyro_mag[split_idx:]
    post_lin = lin_mag[split_idx:]
    
    # Compute features
    features = {
        'acc_max': np.max(acc_mag),
        'gyro_max': np.max(gyro_mag),
        'acc_kurtosis': stats.kurtosis(acc_mag) if len(acc_mag) > 4 else 0,
        'gyro_kurtosis': stats.kurtosis(gyro_mag) if len(gyro_mag) > 4 else 0,
        'lin_max': np.max(lin_mag),
        'acc_skewness': stats.skew(acc_mag) if len(acc_mag) > 2 else 0,
        'gyro_skewness': stats.skew(gyro_mag) if len(gyro_mag) > 2 else 0,
        'post_gyro_max': np.max(post_gyro) if len(post_gyro) > 0 else 0,
        'post_lin_max': np.max(post_lin) if len(post_lin) > 0 else 0,
    }
    
    # Convert to array in correct order
    feature_order = ['acc_max', 'gyro_max', 'acc_kurtosis', 'gyro_kurtosis', 
                     'lin_max', 'acc_skewness', 'gyro_skewness', 
                     'post_gyro_max', 'post_lin_max']
    feature_array = np.array([features[f] for f in feature_order], dtype=np.float32)
    
    return feature_array, features

# ============================================================================
# ENSEMBLE INFERENCE ENGINE
# ============================================================================
class FallDetectionEnsemble:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.data_buffer = {
            'acc': deque(maxlen=WINDOW_SIZE),
            'gyro': deque(maxlen=WINDOW_SIZE)
        }
        self._load_models()
    
    def _load_models(self):
        """Load all ONNX models and the scaler."""
        print("\n" + "=" * 60)
        print("🔄 LOADING MODELS")
        print("=" * 60)
        
        model_files = {
            "XGBoost": "xgboost_fall_detection.onnx",
            "Random Forest": "random_forest_fall_detection.onnx",
            "Logistic Regression": "logistic_regression_fall_detection.onnx",
            "SVM": "svm_fall_detection.onnx"
        }
        
        for name, filename in model_files.items():
            path = os.path.join(self.model_dir, filename)
            if os.path.exists(path):
                self.models[name] = ort.InferenceSession(path)
                size_kb = os.path.getsize(path) / 1024
                print(f"✓ Loaded {name} ({size_kb:.1f} KB)")
            else:
                print(f"✗ Missing {name}: {path}")
        
        # Load scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Loaded scaler")
        else:
            print(f"⚠️ No scaler found - SVM and LR may not work correctly!")
        
        print(f"\n📊 Loaded {len(self.models)} models")
        print("=" * 60)
    
    def add_sensor_data(self, ax: float, ay: float, az: float, 
                        gx: float, gy: float, gz: float):
        """Add new sensor reading to the buffer."""
        self.data_buffer['acc'].append([ax, ay, az])
        self.data_buffer['gyro'].append([gx, gy, gz])
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.data_buffer['acc'])
    
    def predict(self) -> Tuple[bool, float, Dict]:
        """
        Run ensemble prediction with weighted voting.
        
        Returns:
            fall_detected: Boolean indicating if fall was detected
            final_probability: Weighted average probability
            details: Dictionary with per-model predictions
        """
        if len(self.data_buffer['acc']) < WINDOW_SIZE // 2:
            raise ValueError(f"Need at least {WINDOW_SIZE // 2} samples, have {len(self.data_buffer['acc'])}")
        
        # Extract features
        acc_data = np.array(list(self.data_buffer['acc']))
        gyro_data = np.array(list(self.data_buffer['gyro']))
        features, feature_dict = compute_features(acc_data, gyro_data)
        
        # Run each model
        predictions = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, session in self.models.items():
            input_name = session.get_inputs()[0].name
            weight = MODEL_WEIGHTS.get(name, 1.0)
            
            # Scale features for SVM and Logistic Regression
            if name in ["SVM", "Logistic Regression"] and self.scaler:
                scaled_features = self.scaler.transform(features.reshape(1, -1)).astype(np.float32)
            else:
                scaled_features = features.reshape(1, -1)
            
            # Run inference
            outputs = session.run(None, {input_name: scaled_features})
            
            # Extract probability (output format varies)
            if len(outputs) > 1:
                # Classification output (label, probabilities)
                proba = outputs[1][0][1] if isinstance(outputs[1][0], dict) else outputs[1][0][1]
            else:
                proba = outputs[0][0][1] if len(outputs[0].shape) > 1 else outputs[0][0]
            
            prob_float = float(proba)
            predictions[name] = {
                "probability": prob_float,
                "weight": weight,
                "weighted_contribution": prob_float * weight,
                "vote": "FALL" if prob_float > 0.5 else "NO FALL"
            }
            
            weighted_sum += prob_float * weight
            total_weight += weight
        
        # Compute weighted average
        final_probability = weighted_sum / total_weight if total_weight > 0 else 0.0
        fall_detected = final_probability > FALL_THRESHOLD
        
        return fall_detected, final_probability, predictions, feature_dict

# ============================================================================
# FASTAPI SERVER
# ============================================================================
app = FastAPI(
    title="Fall Detection Ensemble API",
    description="ONNX-based fall detection with weighted voting ensemble",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ensemble instance
ensemble = None

@app.on_event("startup")
async def startup():
    global ensemble
    ensemble = FallDetectionEnsemble(MODEL_DIR)

@app.get("/")
async def root():
    return {
        "status": "running",
        "models_loaded": list(ensemble.models.keys()) if ensemble else [],
        "buffer_size": ensemble.get_buffer_size() if ensemble else 0,
        "window_size": WINDOW_SIZE,
        "threshold": FALL_THRESHOLD
    }

@app.post("/data", response_model=None)
async def receive_sensor_data(data: SensorData):
    """Receive sensor data and add to buffer."""
    if not ensemble:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    ensemble.add_sensor_data(data.ax, data.ay, data.az, data.gx, data.gy, data.gz)
    
    return {
        "status": "received",
        "buffer_size": ensemble.get_buffer_size(),
        "ready_for_prediction": ensemble.get_buffer_size() >= WINDOW_SIZE // 2
    }

@app.get("/predict")
async def predict_fall():
    """Run fall prediction on current buffer."""
    if not ensemble:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        fall_detected, probability, model_preds, features = ensemble.predict()
        
        # Log predictions
        print(f"\n{'🚨 FALL DETECTED!' if fall_detected else '✓ No fall'}")
        print(f"Final Probability: {probability:.4f}")
        print("Model Predictions:")
        for name, pred in model_preds.items():
            emoji = "🔴" if pred["vote"] == "FALL" else "🟢"
            print(f"  {emoji} {name}: {pred['probability']:.4f} (weight: {pred['weight']})")
        
        return PredictionResponse(
            timestamp=datetime.now().isoformat(),
            fall_detected=fall_detected,
            final_probability=probability,
            model_predictions={k: v["probability"] for k, v in model_preds.items()},
            voting_details=model_preds,
            features_used=features
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_direct")
async def predict_direct(data: SensorData):
    """Receive data AND predict in one call (for real-time use)."""
    if not ensemble:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Add data
    ensemble.add_sensor_data(data.ax, data.ay, data.az, data.gx, data.gy, data.gz)
    
    # Check if enough data
    if ensemble.get_buffer_size() < WINDOW_SIZE // 2:
        return {
            "status": "collecting",
            "buffer_size": ensemble.get_buffer_size(),
            "need": WINDOW_SIZE // 2
        }
    
    # Predict
    fall_detected, probability, model_preds, features = ensemble.predict()
    
    # Log
    status_emoji = "🚨" if fall_detected else "✓"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {status_emoji} P={probability:.3f} | " +
          " | ".join([f"{k[:3]}:{v['probability']:.2f}" for k, v in model_preds.items()]))
    
    return {
        "timestamp": datetime.now().isoformat(),
        "fall_detected": fall_detected,
        "final_probability": probability,
        "model_predictions": {k: v["probability"] for k, v in model_preds.items()},
        "buffer_size": ensemble.get_buffer_size()
    }

@app.post("/clear")
async def clear_buffer():
    """Clear the sensor data buffer."""
    if ensemble:
        ensemble.data_buffer['acc'].clear()
        ensemble.data_buffer['gyro'].clear()
    return {"status": "cleared"}

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🏃 FALL DETECTION ENSEMBLE INFERENCE SERVER")
    print("=" * 60)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Window size: {WINDOW_SIZE} samples")
    print(f"Fall threshold: {FALL_THRESHOLD}")
    print("\nVoting Weights:")
    for model, weight in MODEL_WEIGHTS.items():
        print(f"  - {model}: {weight}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
