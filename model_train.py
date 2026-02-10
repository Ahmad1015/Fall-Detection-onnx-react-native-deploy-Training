import pandas as pd

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

print(train.shape, test.shape)
train.head()
TARGET = "fall"
DROP = ["Unnamed: 0", "label", TARGET]

X_train = train.drop(columns=DROP, errors="ignore")
y_train = train[TARGET]

X_test = test.drop(columns=DROP, errors="ignore")
y_test = test[TARGET]
from xgboost import XGBClassifier

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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

pred = xgb.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

probs = xgb.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.show()

from sklearn.model_selection import cross_val_score

scores = cross_val_score(xgb, X_train, y_train, cv=5)
print("CV Mean:", scores.mean())