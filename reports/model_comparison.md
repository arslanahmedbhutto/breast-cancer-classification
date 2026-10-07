# Model comparison

Ranked by mean 5-fold CV F1 on the training split (80%). Test columns are on the held-out 20% split (114 samples). Positive class = malignant.

| Model | CV F1 | CV ROC-AUC | Test Acc | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| **MLP** | 0.967 | 0.995 | 0.974 | 1.000 | 0.929 | 0.963 | 0.995 |
| SVM (RBF) | 0.967 | 0.995 | 0.974 | 0.976 | 0.952 | 0.964 | 0.995 |
| Logistic Regression | 0.964 | 0.996 | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| K-Nearest Neighbors | 0.957 | 0.986 | 0.956 | 0.974 | 0.905 | 0.938 | 0.982 |
| Gradient Boosting | 0.955 | 0.991 | 0.965 | 1.000 | 0.905 | 0.950 | 0.995 |
| Random Forest | 0.946 | 0.988 | 0.974 | 1.000 | 0.929 | 0.963 | 0.994 |
| Naive Bayes | 0.917 | 0.987 | 0.921 | 0.923 | 0.857 | 0.889 | 0.989 |
| Decision Tree | 0.872 | 0.887 | 0.921 | 0.946 | 0.833 | 0.886 | 0.945 |
