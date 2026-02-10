"""
Fall Detection Ensemble Model Training
=======================================
Trains 4 models (XGBoost, Random Forest, Logistic Regression, SVM) with extensive 
hyperparameter tuning via GridSearchCV, evaluates with focus on recall, generates 
visualizations, and exports to ONNX for mobile deployment.

Supports: GPU acceleration (XGBoost), Multi-core parallelization (all models)
"""

# ============================================================================
# INSTALL DEPENDENCIES (uncomment for Google Colab or fresh environment)
# ============================================================================
# !pip install xgboost scikit-learn skl2onnx onnx onnxmltools onnxruntime matplotlib seaborn pandas numpy

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    recall_score, precision_score, f1_score, roc_curve, auc
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

# ONNX conversion
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
CV_FOLDS = 5   # 5-fold CV - better sample representation per fold (~123 falls per fold)
N_JOBS = -1    # Use all available CPU cores for parallel processing
SCORING = 'recall'  # Optimize for recall - don't miss falls!

# GPU Configuration for XGBoost (will auto-detect)
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    USE_GPU = result.returncode == 0
except:
    USE_GPU = False

print("=" * 70)
print("FALL DETECTION ENSEMBLE MODEL TRAINING")
print("=" * 70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"GPU Available: {USE_GPU}")
print(f"Cross-Validation Folds: {CV_FOLDS}")
print(f"Parallel Jobs: {N_JOBS} (all cores)")
print(f"Optimization Metric: {SCORING}")
print("=" * 70)

# ============================================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================================
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)
print("✓ Created output directories: plots/, models/")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

# The first column with no name (shows as empty before acc_max) becomes 'Unnamed: 0' in pandas
# This is just the row index from when the CSV was saved, we drop it
TARGET = "fall"
DROP_COLS = ["label", TARGET]

# Check for unnamed columns (index columns saved to CSV)
unnamed_cols = [col for col in train.columns if 'Unnamed' in col]
if unnamed_cols:
    print(f"Dropping index columns: {unnamed_cols}")
    DROP_COLS.extend(unnamed_cols)

X_train = train.drop(columns=DROP_COLS, errors="ignore")
y_train = train[TARGET]

X_test = test.drop(columns=DROP_COLS, errors="ignore")
y_test = test[TARGET]

print(f"Features ({len(X_train.columns)}): {list(X_train.columns)}")
print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Class distribution (train): Falls={y_train.sum()}, Non-falls={len(y_train)-y_train.sum()}")
print(f"Class imbalance ratio: {(len(y_train) - y_train.sum()) / y_train.sum():.2f}:1")

# ============================================================================
# SCALE FEATURES (important for SVM and Logistic Regression)
# ============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keep unscaled for tree-based models
X_train_np = X_train.values
X_test_np = X_test.values

# ============================================================================
# DEFINE EXTENSIVE HYPERPARAMETER GRIDS
# ============================================================================
print("\n" + "=" * 70)
print("HYPERPARAMETER GRIDS (Extensive Search)")
print("=" * 70)

# Class weight for imbalanced data
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# XGBoost Grid - Extensive
xgb_param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 2, 3, 5],
    'reg_lambda': [0.5, 1.0, 1.5, 2.0],
}
print(f"XGBoost: {np.prod([len(v) for v in xgb_param_grid.values()]):,} combinations")

# Random Forest Grid - Extensive  
rf_param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 3, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5, 0.7],
    'bootstrap': [True, False],
}
print(f"Random Forest: {np.prod([len(v) for v in rf_param_grid.values()]):,} combinations")

# Logistic Regression Grid - Extensive
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'penalty': ['l1', 'l2'],
    'max_iter': [500, 1000, 2000],
}
print(f"Logistic Regression: ~162 valid combinations")

# SVM Grid - Extensive (RBF kernel)
svm_param_grid = {
    'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 0.5, 1],
    'kernel': ['rbf'],
    'class_weight': ['balanced', None],
}
print(f"SVM (RBF): {np.prod([len(v) for v in svm_param_grid.values()]):,} combinations")

print("\n⚠️  Note: Using RandomizedSearchCV for large grids to speed up training")

# ============================================================================
# TRAINING HELPER FUNCTION
# ============================================================================
def train_model_with_grid_search(name, model, param_grid, X_train, y_train, 
                                  use_randomized=True, n_iter=100):
    """Train a model with GridSearchCV/RandomizedSearchCV and return results."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    actual_fits = min(n_iter, total_combinations) if use_randomized and total_combinations > 200 else total_combinations
    total_fits = actual_fits * CV_FOLDS
    
    print(f"\n📊 TRAINING INFO:")
    print(f"   Total parameter combinations: {total_combinations:,}")
    print(f"   Iterations to try: {actual_fits}")
    print(f"   CV Folds: {CV_FOLDS}")
    print(f"   Total fits: {total_fits:,}")
    print(f"   Scoring: {SCORING}")
    print(f"   Parallel jobs: {N_JOBS} (all cores)")
    
    # Time estimate based on model type
    time_estimates = {
        'XGBoost': 2.5,  # seconds per fit with GPU
        'Random Forest': 0.8,
        'Logistic Regression': 0.05,
        'SVM (RBF)': 0.3
    }
    est_time_per_fit = time_estimates.get(name, 1.0)
    est_total_time = total_fits * est_time_per_fit / 2  # /2 because of parallelization
    print(f"\n⏱️  ESTIMATED TIME: {est_total_time/60:.1f} minutes (rough estimate)")
    
    if use_randomized and total_combinations > 200:
        from sklearn.model_selection import RandomizedSearchCV
        print(f"\n🔄 Using RandomizedSearchCV with {n_iter} iterations")
        search = RandomizedSearchCV(
            model, param_grid, 
            n_iter=n_iter,
            cv=CV_FOLDS, 
            scoring=SCORING,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbose=2  # More verbose output
        )
    else:
        print(f"\n🔄 Using GridSearchCV (exhaustive search)")
        search = GridSearchCV(
            model, param_grid, 
            cv=CV_FOLDS, 
            scoring=SCORING,
            n_jobs=N_JOBS,
            verbose=2  # More verbose output
        )
    
    print(f"\n🚀 Starting training at {datetime.now().strftime('%H:%M:%S')}...")
    print(f"   Watch progress below (each [CV] = one fold completed)")
    print("-" * 70)
    
    search.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    print("-" * 70)
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"   Time taken: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"   Best CV Score (Recall): {search.best_score_:.4f}")
    print(f"   Best Parameters:")
    for k, v in search.best_params_.items():
        print(f"      - {k}: {v}")
    
    return search.best_estimator_, search.cv_results_, search.best_params_, search.best_score_

# ============================================================================
# TRAIN ALL MODELS
# ============================================================================
results = {}

# 1. XGBoost
print("\n" + "=" * 70)
print("MODEL 1/4: XGBoost (Tree-based, Boosting)")
print("=" * 70)

# XGBoost 3.x uses device='cuda' for GPU, not tree_method='gpu_hist'
xgb_base = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    tree_method='hist',
    device='cuda' if USE_GPU else 'cpu',  # XGBoost 3.x GPU config
    n_jobs=N_JOBS if not USE_GPU else 1
)

xgb_model, xgb_cv_results, xgb_best_params, xgb_best_score = train_model_with_grid_search(
    "XGBoost", xgb_base, xgb_param_grid, X_train_np, y_train, 
    use_randomized=True, n_iter=150
)
results['XGBoost'] = {
    'model': xgb_model, 'cv_results': xgb_cv_results, 
    'best_params': xgb_best_params, 'best_score': xgb_best_score
}

# 2. Random Forest
print("\n" + "=" * 70)
print("MODEL 2/4: Random Forest (Tree-based, Bagging)")
print("=" * 70)

rf_base = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)

rf_model, rf_cv_results, rf_best_params, rf_best_score = train_model_with_grid_search(
    "Random Forest", rf_base, rf_param_grid, X_train_np, y_train,
    use_randomized=True, n_iter=150
)
results['Random Forest'] = {
    'model': rf_model, 'cv_results': rf_cv_results,
    'best_params': rf_best_params, 'best_score': rf_best_score
}

# 3. Logistic Regression
print("\n" + "=" * 70)
print("MODEL 3/4: Logistic Regression (Linear Baseline)")
print("=" * 70)

# Filter valid penalty/solver combinations
lr_param_grid_valid = {
    'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
    'max_iter': [1000, 2000],
}

lr_base = LogisticRegression(
    class_weight='balanced',
    random_state=RANDOM_STATE,
    solver='lbfgs',
    n_jobs=N_JOBS
)

lr_model, lr_cv_results, lr_best_params, lr_best_score = train_model_with_grid_search(
    "Logistic Regression", lr_base, lr_param_grid_valid, X_train_scaled, y_train,
    use_randomized=False
)
results['Logistic Regression'] = {
    'model': lr_model, 'cv_results': lr_cv_results,
    'best_params': lr_best_params, 'best_score': lr_best_score
}

# 4. SVM (RBF kernel)
print("\n" + "=" * 70)
print("MODEL 4/4: SVM with RBF Kernel (Non-linear)")
print("=" * 70)

svm_base = SVC(
    kernel='rbf',
    probability=True,  # Required for soft voting / ONNX probability outputs
    random_state=RANDOM_STATE,
    cache_size=1000  # Increase cache for faster training
)

svm_param_grid_reduced = {
    'C': [0.1, 0.5, 1, 5, 10, 50, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 0.5, 1],
    'class_weight': ['balanced', None],
}

svm_model, svm_cv_results, svm_best_params, svm_best_score = train_model_with_grid_search(
    "SVM (RBF)", svm_base, svm_param_grid_reduced, X_train_scaled, y_train,
    use_randomized=True, n_iter=80
)
results['SVM (RBF)'] = {
    'model': svm_model, 'cv_results': svm_cv_results,
    'best_params': svm_best_params, 'best_score': svm_best_score
}

# ============================================================================
# EVALUATE ALL MODELS ON TEST SET
# ============================================================================
print("\n" + "=" * 70)
print("TEST SET EVALUATION (Focus on Recall)")
print("=" * 70)

test_results = {}

for name, data in results.items():
    model = data['model']
    
    # Use scaled features for SVM and LR
    if name in ['SVM (RBF)', 'Logistic Regression']:
        X_eval = X_test_scaled
    else:
        X_eval = X_test_np
    
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    test_results[name] = {
        'accuracy': acc, 'recall': rec, 'precision': prec, 
        'f1': f1, 'auc': roc_auc, 'fpr': fpr, 'tpr': tpr,
        'y_pred': y_pred, 'y_proba': y_proba
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Recall:    {rec:.4f}  ← KEY METRIC (missed falls)")
    print(f"  Precision: {prec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {roc_auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fall', 'Fall']))

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

# 1. Model Comparison Bar Chart
print("Creating model_comparison.png...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Metrics comparison
metrics = ['accuracy', 'recall', 'precision', 'f1']
model_names = list(test_results.keys())
x = np.arange(len(model_names))
width = 0.2

for i, metric in enumerate(metrics):
    values = [test_results[m][metric] for m in model_names]
    bars = axes[0].bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
    # Add value labels
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(model_names, rotation=15, ha='right')
axes[0].legend(loc='lower right')
axes[0].set_ylim(0, 1.15)

# Recall Focus Chart
recall_values = [test_results[m]['recall'] for m in model_names]
bars = axes[1].bar(model_names, recall_values, color=colors, alpha=0.8, edgecolor='black')
axes[1].axhline(y=0.9, color='red', linestyle='--', label='Target Recall (0.9)')
for bar, val in zip(bars, recall_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('Recall', fontsize=12)
axes[1].set_title('Recall Comparison (Don\'t Miss Falls!)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].set_ylim(0, 1.15)

plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/model_comparison.png")

# 2. ROC Curves
print("Creating roc_curves.png...")
fig, ax = plt.subplots(figsize=(10, 8))

for i, (name, data) in enumerate(test_results.items()):
    ax.plot(data['fpr'], data['tpr'], color=colors[i], lw=2,
            label=f'{name} (AUC = {data["auc"]:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/roc_curves.png")

# 3. Confusion Matrices
print("Creating confusion_matrices.png...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (name, data) in enumerate(test_results.items()):
    ax = axes[idx // 2, idx % 2]
    cm = confusion_matrix(y_test, data['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Fall', 'Fall'],
                yticklabels=['Non-Fall', 'Fall'],
                annot_kws={'size': 14})
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f'{name}\nRecall: {data["recall"]:.3f}', fontsize=12, fontweight='bold')

plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/confusion_matrices.png")

# 4. Parameter Tuning Visualization
print("Creating parameter_tuning.png...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, data) in enumerate(results.items()):
    ax = axes[idx // 2, idx % 2]
    
    cv_results = data['cv_results']
    mean_scores = cv_results['mean_test_score']
    
    # Sort by score
    sorted_indices = np.argsort(mean_scores)[-20:]  # Top 20
    top_scores = mean_scores[sorted_indices]
    
    ax.barh(range(len(top_scores)), top_scores, color=colors[idx], alpha=0.7)
    ax.set_xlabel('Mean CV Recall Score', fontsize=11)
    ax.set_ylabel('Hyperparameter Configuration (Top 20)', fontsize=10)
    ax.set_title(f'{name}\nBest: {data["best_score"]:.4f}', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.axvline(x=data['best_score'], color='red', linestyle='--', label='Best')
    ax.legend()

plt.suptitle('GridSearchCV Results (Recall Optimization)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/parameter_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/parameter_tuning.png")

# ============================================================================
# EXPORT TO ONNX FORMAT
# ============================================================================
print("\n" + "=" * 70)
print("EXPORTING MODELS TO ONNX FORMAT")
print("=" * 70)

# Initial type for ONNX conversion
n_features = X_train.shape[1]
initial_type = [('float_input', FloatTensorType([None, n_features]))]

def export_to_onnx(model, name, filename, initial_type):
    """Export sklearn/XGBoost model to ONNX format.
    
    IMPORTANT: For sklearn models, we set options={type(model): {'zipmap': False}}
    so that probability outputs are plain float tensors instead of 
    ONNX_TYPE_SEQUENCE (Sequence of Maps), which is NOT supported by 
    onnxruntime-react-native on mobile devices.
    """
    try:
        if 'XGB' in str(type(model)):
            # Use onnxmltools for XGBoost (already outputs tensors)
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType as FTT
            onnx_model = convert_xgboost(model, initial_types=[('float_input', FTT([None, n_features]))])
        else:
            # Use skl2onnx for sklearn models
            # zipmap=False → output probabilities as a 2D float tensor
            # instead of Sequence<Map> which mobile ONNX runtime can't handle
            onnx_model = convert_sklearn(
                model, 
                initial_types=initial_type,
                options={type(model): {'zipmap': False}}
            )
        
        onnx.save_model(onnx_model, filename)
        file_size = os.path.getsize(filename) / 1024  # KB
        print(f"✓ {name}: {filename} ({file_size:.1f} KB)")
        return True
    except Exception as e:
        print(f"✗ {name}: Failed - {e}")
        return False

# Export each model
print("\nExporting models...")
export_to_onnx(results['XGBoost']['model'], 'XGBoost', 
               'models/xgboost_fall_detection.onnx', initial_type)
export_to_onnx(results['Random Forest']['model'], 'Random Forest',
               'models/random_forest_fall_detection.onnx', initial_type)
export_to_onnx(results['Logistic Regression']['model'], 'Logistic Regression',
               'models/logistic_regression_fall_detection.onnx', initial_type)
export_to_onnx(results['SVM (RBF)']['model'], 'SVM (RBF)',
               'models/svm_fall_detection.onnx', initial_type)

# Also save the scaler for preprocessing on mobile
import pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler: models/scaler.pkl")

# ============================================================================
# VALIDATE ONNX MODELS
# ============================================================================
print("\n" + "=" * 70)
print("VALIDATING ONNX MODELS")
print("=" * 70)

def validate_onnx_model(onnx_path, original_model, X_test_sample, name, use_scaled=False):
    """Validate ONNX model outputs match original model."""
    try:
        # Load ONNX model
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        
        # Get original predictions
        original_proba = original_model.predict_proba(X_test_sample)[:, 1]
        
        # Get ONNX predictions
        X_input = X_test_sample.astype(np.float32)
        onnx_output = sess.run(None, {input_name: X_input})
        
        # ONNX probability output (format varies by model type)
        if len(onnx_output) > 1:
            onnx_proba = np.array([p[1] for p in onnx_output[1]])
        else:
            onnx_proba = onnx_output[0][:, 1] if len(onnx_output[0].shape) > 1 else onnx_output[0]
        
        # Check if outputs match (within tolerance)
        max_diff = np.max(np.abs(original_proba - onnx_proba))
        matches = max_diff < 0.01
        
        status = "✓ PASS" if matches else "⚠ MISMATCH"
        print(f"{status} {name}: Max diff = {max_diff:.6f}")
        return matches
    except Exception as e:
        print(f"✗ {name}: Validation failed - {e}")
        return False

# Test with 5 samples
test_samples = 5
validate_onnx_model('models/xgboost_fall_detection.onnx', 
                    results['XGBoost']['model'], X_test_np[:test_samples], 'XGBoost')
validate_onnx_model('models/random_forest_fall_detection.onnx',
                    results['Random Forest']['model'], X_test_np[:test_samples], 'Random Forest')
validate_onnx_model('models/logistic_regression_fall_detection.onnx',
                    results['Logistic Regression']['model'], X_test_scaled[:test_samples], 
                    'Logistic Regression', use_scaled=True)
validate_onnx_model('models/svm_fall_detection.onnx',
                    results['SVM (RBF)']['model'], X_test_scaled[:test_samples],
                    'SVM (RBF)', use_scaled=True)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("=" * 70)

print("\n📊 MODEL PERFORMANCE (Test Set):")
print("-" * 50)
print(f"{'Model':<25} {'Recall':>10} {'Precision':>10} {'F1':>10}")
print("-" * 50)

best_recall = 0
best_model = ""
for name, data in test_results.items():
    print(f"{name:<25} {data['recall']:>10.4f} {data['precision']:>10.4f} {data['f1']:>10.4f}")
    if data['recall'] > best_recall:
        best_recall = data['recall']
        best_model = name

print("-" * 50)
print(f"\n🏆 Best Model for Fall Detection: {best_model} (Recall: {best_recall:.4f})")

print("\n📁 OUTPUT FILES:")
print("-" * 50)
print("Plots:")
for f in os.listdir('plots'):
    print(f"  - plots/{f}")
print("\nONNX Models:")
for f in os.listdir('models'):
    size = os.path.getsize(f'models/{f}') / 1024
    print(f"  - models/{f} ({size:.1f} KB)")

print(f"\n⏱️  Total runtime: Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "=" * 70)
print("Ready for mobile deployment with soft voting ensemble!")
print("Use probability outputs from all models for soft voting on mobile.")
print("=" * 70)
