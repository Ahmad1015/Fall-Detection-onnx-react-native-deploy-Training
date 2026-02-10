import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from scipy.stats import kurtosis, skew
import glob
import os

# 1. Train the Model (using user's provided code)
print("Training Model...")
try:
    train = pd.read_csv("Train.csv")
    test = pd.read_csv("Test.csv")
    
    TARGET = "fall"
    DROP = ["Unnamed: 0", "label", TARGET]
    
    X_train = train.drop(columns=DROP, errors="ignore")
    y_train = train[TARGET]
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_lambda=1.5,
        min_child_weight=2,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )
    
    xgb.fit(X_train, y_train)
    print("Model Trained Successfully.")
except Exception as e:
    print(f"Error training model: {e}")
    exit()

# 2. Load and Process New Data
CSV_FILE = "received_data.csv"
if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found. Please collect data first.")
    exit()

print(f"Processing {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

# Ensure columns exist
required_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
if not all(col in df.columns for col in required_cols):
    print(f"Error: CSV missing required columns: {required_cols}")
    print(f"Found: {df.columns.tolist()}")
    exit()

# Calculate Magnitudes
# Assuming accelerometer is in g (1g ~ 9.8m/s^2). 
# If model expects m/s^2, we might need to multiply by 9.8. 
# Let's check Train.csv values if possible, but for now assume consistent units if collected similarly.
# Actually, let's look at the feature names: 'acc_max' ~ 26. That suggests m/s^2 (2.6g is reasonable for fall, 26 m/s^2).
# Expo returns g. So we multiply by 9.81.
G = 9.81
df['smv_a'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2) * G
df['smv_g'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2) # rad/s or deg/s? Expo is rad/s usually.
# Check Train.csv 'gyro_max' ~ 7. If rad/s, 7 is ~400 deg/s. Reasonable.

# Sliding Window
WINDOW_SIZE = 20 # 2 seconds at 10Hz
STEP_SIZE = 10   # 1 second overlap

predictions = []
last_fall_time = None  # Track last detected fall to avoid duplicates
FALL_COOLDOWN = 3.0  # Don't report another fall within 3 seconds

print(f"Running predictions on {len(df)} samples...")

for i in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
    window = df.iloc[i:i+WINDOW_SIZE]
    
    # Feature Extraction
    smv_a = window['smv_a'].values
    smv_g = window['smv_g'].values
    
    acc_max = np.max(smv_a)
    gyro_max = np.max(smv_g)
    
    k_a = kurtosis(smv_a)
    k_g = kurtosis(smv_g)
    
    s_a = skew(smv_a)
    s_g = skew(smv_g)
    
    # Linear Acceleration (approximate by removing gravity)
    # Simple high-pass or just subtract G? 
    # Let's use: lin_acc = abs(smv_a - 9.81)
    lin_acc = np.abs(smv_a - 9.81)
    lin_max = np.max(lin_acc)
    
    # Post-impact features
    # Find index of max acceleration
    idx_max = np.argmax(smv_a)
    
    # If peak is at the end, post window is empty. Handle gracefully.
    if idx_max < len(window) - 1:
        post_g = smv_g[idx_max:]
        post_l = lin_acc[idx_max:]
        post_gyro_max = np.max(post_g)
        post_lin_max = np.max(post_l)
    else:
        post_gyro_max = 0
        post_lin_max = 0
    
    # FALL PATTERN DETECTION
    # Real falls have a characteristic temporal pattern:
    # 1. Freefall phase (acc < 1g) before peak
    # 2. Impact (high acc)
    # 3. Relative stillness after impact
    
    # Check for freefall before peak
    has_freefall = False
    if idx_max >= 3:  # Need at least 3 samples before peak
        pre_impact = smv_a[:idx_max]
        # Freefall: acceleration magnitude < 0.7g (6.87 m/s²) for at least 2 consecutive samples
        freefall_threshold = 0.7 * 9.81
        freefall_samples = np.sum(pre_impact < freefall_threshold)
        if freefall_samples >= 2:
            has_freefall = True
    
    # Check for post-impact stillness (or at least reduced movement)
    has_stillness = False
    if idx_max < len(window) - 3:  # Need at least 3 samples after peak
        post_impact = smv_a[idx_max+1:]
        # Stillness: standard deviation of post-impact acceleration is low
        if len(post_impact) >= 3:
            post_std = np.std(post_impact)
            if post_std < 3.0:  # Low variation after impact
                has_stillness = True
        
    # Construct feature vector matching training data order
    features = pd.DataFrame([{
        'acc_max': acc_max,
        'gyro_max': gyro_max,
        'acc_kurtosis': k_a,
        'gyro_kurtosis': k_g,
        'lin_max': lin_max,
        'acc_skewness': s_a,
        'gyro_skewness': s_g,
        'post_gyro_max': post_gyro_max,
        'post_lin_max': post_lin_max
    }])
    
    # Reorder columns to match X_train if needed
    features = features[X_train.columns]
    
    pred = xgb.predict(features)[0]
    prob = xgb.predict_proba(features)[0][1]
    
    timestamp = window.iloc[-1]['timestamp']
    
    # Enhanced Fall Detection Logic
    # Real falls MUST have a freefall phase - this is the most distinctive feature
    # We now require: Model prediction + Freefall + High impact
    
    if pred == 1 and prob > 0.9:
        # Model thinks it's a fall with high confidence
        if acc_max > 30.0:  # Very high acceleration (3g+)
            # Strict requirement: MUST have freefall before impact
            if has_freefall:
                # Check if we recently reported a fall (deduplication)
                current_time = pd.to_datetime(timestamp)
                should_report = True
                
                if last_fall_time is not None:
                    time_since_last = (current_time - last_fall_time).total_seconds()
                    if time_since_last < FALL_COOLDOWN:
                        should_report = False  # Too soon after last fall, likely same event
                
                if should_report:
                    # Confirmed freefall pattern before impact - likely a real fall
                    print(f"[{timestamp}] ** FALL DETECTED! (Prob: {prob:.4f})")
                    print(f"  Features: acc_max={acc_max:.2f}, gyro_max={gyro_max:.2f}, lin_max={lin_max:.2f}")
                    print(f"  Pattern: Freefall=True, Rotation={gyro_max:.2f} rad/s")
                    last_fall_time = current_time
            else:
                # High impact but NO freefall = phone was dropped/thrown, not a person falling
                print(f"[{timestamp}] Sharp Impact (Not a Fall). Prob: {prob:.4f}, AccMax: {acc_max:.2f}")
                print(f"  Reason: No freefall detected (phone dropped/shaken, not person falling)")
        else:
            print(f"[{timestamp}] Ignored (Low Intensity). Prob: {prob:.4f}, AccMax: {acc_max:.2f}")
    elif pred == 1 and prob > 0.7:
        # Model thinks it might be a fall, but lower confidence
        # Let's print these for debugging
        print(f"[{timestamp}] DEBUG: Potential event (Low Confidence)")
        print(f"  Prob: {prob:.4f}, AccMax: {acc_max:.2f}, GyroMax: {gyro_max:.2f}")
        print(f"  Freefall: {has_freefall}, Kurtosis: acc={k_a:.2f}, gyro={k_g:.2f}")
    else:
        pass

# Print summary statistics
print(f"\nSummary Statistics:")
print(f"Total windows analyzed: {len(range(0, len(df) - WINDOW_SIZE, STEP_SIZE))}")
print(f"Max acceleration in dataset: {df['smv_a'].max():.2f} m/s²")
print(f"Max gyroscope in dataset: {df['smv_g'].max():.2f} rad/s")

print("\nDone.")
