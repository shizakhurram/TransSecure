# TransSecure
AI-powered system for detecting and preventing fraudulent credit card transactions.
# TransSecure: Credit Card Fraud Detection using Machine Learning

## Overview
The **TransSecure** project focuses on detecting fraudulent credit card transactions using machine learning techniques. This model analyzes past transactions to predict whether a new transaction is fraudulent or legitimate.

## Dataset
The model is trained on a credit card transaction dataset containing numerical input features extracted from transaction details. The dataset includes both normal and fraudulent transactions.

## Steps Involved
1. **Importing Libraries**
2. **Loading Dataset**
3. **Exploratory Data Analysis**
4. **Outlier Detection using:**
   - Isolation Forest
   - Local Outlier Factor
   - Support Vector Machine (One-Class SVM)
5. **Feature Scaling**
6. **Training and Evaluating Models**
7. **Performance Metrics**

## Libraries Used
```python
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
```

## Results
### 🔹 Running Isolation Forest...
**Detected Fraudulent Transactions:** 702
✅ **Accuracy Score:** 0.9975
📊 **Classification Report:**
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    284315
           1       0.29      0.29      0.29       492

    accuracy                           1.00    284807
   macro avg       0.64      0.64      0.64    284807
weighted avg       1.00      1.00      1.00    284807
```

### 🔹 Running Local Outlier Factor...
**Detected Fraudulent Transactions:** 984
✅ **Accuracy Score:** 0.9965
📊 **Classification Report:**
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    284315
           1       0.00      0.00      0.00       492

    accuracy                           1.00    284807
   macro avg       0.50      0.50      0.50    284807
weighted avg       1.00      1.00      1.00    284807
```

### 🔹 Running Support Vector Machine (One-Class SVM)...
**Detected Fraudulent Transactions:** 561
✅ **Accuracy Score:** 0.8878
📊 **Classification Report:**
```
              precision    recall  f1-score   support

           0       1.00      0.89      0.94      4994
           1       0.01      0.83      0.02         6

    accuracy                           0.89      5000
   macro avg       0.50      0.86      0.48      5000
weighted avg       1.00      0.89      0.94      5000
```

## Conclusion
The **TransSecure** project demonstrates the effectiveness of machine learning in detecting fraudulent transactions. Among the models tested, Isolation Forest provided the best balance of precision and recall. Further improvements can be made by fine-tuning the models, handling class imbalances, and incorporating additional features.

## Future Improvements
- Use **Deep Learning** techniques such as Autoencoders.
- Implement **Oversampling/Undersampling** techniques for handling class imbalance.
- Optimize hyperparameters using **GridSearchCV** or **RandomizedSearchCV**.
- Test additional models such as **Random Forest, XGBoost, and Neural Networks**.

---
### 📌 **Author:** Shiza Khurram

