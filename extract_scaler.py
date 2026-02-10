"""
Extract scaler parameters from pickle file for use in React Native.
Run this after training to get mean and scale values.
"""

import pickle
import json
import os

# Load the scaler
scaler_path = "training_result/scaler.pkl"
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Extract parameters
mean = scaler.mean_.tolist()
scale = scaler.scale_.tolist()

# Feature order
feature_order = [
    'acc_max', 'gyro_max', 'acc_kurtosis', 'gyro_kurtosis',
    'lin_max', 'acc_skewness', 'gyro_skewness',
    'post_gyro_max', 'post_lin_max'
]

print("=" * 60)
print("SCALER PARAMETERS FOR REACT NATIVE")
print("=" * 60)

print("\n// Copy these to FallDetectionService.ts:\n")
print(f"const SCALER_MEAN = {mean};")
print(f"const SCALER_SCALE = {scale};")

print("\n\nDetailed values:")
print("-" * 60)
for i, feature in enumerate(feature_order):
    print(f"{feature:20} mean={mean[i]:10.4f}  scale={scale[i]:10.4f}")

# Save as JSON for reference
scaler_json = {
    "mean": mean,
    "scale": scale,
    "features": feature_order
}

output_path = "training_result/scaler_params.json"
with open(output_path, 'w') as f:
    json.dump(scaler_json, f, indent=2)

print(f"\n[OK] Saved to {output_path}")
