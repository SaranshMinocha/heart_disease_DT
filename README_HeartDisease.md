# Heart Disease Prediction — Decision Tree Classifier

## What this project does
Predicts whether a patient has heart disease based on 13 clinical features using a
Decision Tree Classifier. Demonstrates how interpretable ML models can assist medical
diagnosis while providing full transparency into every decision made.

## Dataset
- **Heart Disease Dataset** — 303 patients, 14 columns
- 13 input features (age, cholesterol, chest pain type, blood pressure etc.)
- Target: 0 = No Disease, 1 = Disease
- No missing values

## Features Used
| Feature | Description |
|---|---|
| age | Patient age |
| sex | Gender (1=male, 0=female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure |
| chol | Cholesterol level |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Blood disorder type |

## Model Architecture
```
Raw Patient Data (13 features)
→ Train/Test Split (80/20)
→ Decision Tree Classifier (max_depth=5)
→ Predictions + Probability Scores
```

## Results
- **Accuracy: 81.97%**
- Training time: Under 1 second

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| No Disease | 0.82 | 0.79 | 0.81 |
| Disease | 0.82 | 0.84 | 0.83 |

## Confusion Matrix
| | Predicted No Disease | Predicted Disease |
|---|---|---|
| Actual No Disease | 23 ✅ | 6 ⚠️ |
| Actual Disease | 5 ❌ | 27 ✅ |

- 5 missed disease patients (False Negatives) — most critical error in medical context
- 6 healthy patients falsely alarmed (False Positives) — less critical, leads to further testing

## Key Concepts Applied

### Decision Tree
A flowchart-like model that makes sequential decisions based on feature values.
Each internal node asks one question, each branch represents an answer,
each leaf node gives a final prediction. Fully interpretable — you can trace
exactly why any prediction was made.

### Entropy and Gini Impurity
The tree selects which feature to split on by measuring which split creates
the purest groups. Gini = 0 means a node is completely pure (all one class).
Gini = 0.5 means maximum uncertainty (50/50 split). The tree always picks
the split that reduces impurity the most (highest Information Gain).

### Max Depth (Overfitting Prevention)
Without limits a decision tree memorizes every training example (100% train accuracy,
poor test accuracy). max_depth=5 forces the model to find general patterns
rather than memorizing specific patients.

### Precision vs Recall in Medical Context
In heart disease detection, recall for Disease class is the critical metric.
Missing a sick patient (False Negative) is far more dangerous than
falsely alarming a healthy patient (False Positive).
Current model achieves 0.84 recall for Disease — decent but real medical
deployment would require higher recall, potentially sacrificing precision.

### Decision Tree Visualization
Unlike neural networks, decision trees are fully interpretable.
The visualization shows every decision path from root to leaf,
revealing that chest pain type (cp) is the most predictive feature
for heart disease — consistent with real clinical practice.

## What I Learned
- How decision trees use entropy and information gain to choose splits
- Why max_depth is critical for preventing overfitting in decision trees
- How to interpret confusion matrices in high-stakes medical domains
- Why recall matters more than precision for disease detection
- The interpretability advantage of decision trees over neural networks
- How to visualize decision tree structure using plot_tree

## Comparison: Decision Tree vs Neural Network for Medical Data
| | Decision Tree | Neural Network |
|---|---|---|
| Interpretability | Full — see every decision | Black box |
| Training Time | Under 1 second | Minutes |
| Accuracy (this dataset) | 81.97% | Similar |
| Doctor Trust | High — explainable | Low — unexplainable |
| Best For | Small tabular medical data | Large complex datasets |

## Tech Stack
- Python
- Pandas
- Scikit-learn (DecisionTreeClassifier, plot_tree)
- Matplotlib
- Seaborn
- NumPy
