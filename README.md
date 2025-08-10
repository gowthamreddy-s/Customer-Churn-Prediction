# Customer Churn Prediction

## Project Overview  
This project focuses on predicting customer churn for a telecom company using machine learning techniques. Customer churn—when customers leave a service—is a significant challenge in the telecom industry. Early detection of potential churners enables companies to implement proactive retention strategies, minimizing revenue loss.

## Dataset  
- **Source:** Telco Customer Churn dataset from Kaggle  
- **Samples:** 7,043 customers  
- **Target Variable:** `Churn` (1 = Yes, 0 = No)  
- **Features:** Demographics (e.g., gender, senior citizen), service     details (InternetService, Contract type), billing information (MonthlyCharges, TotalCharges), etc.

## Key Challenges  
- **Class Imbalance:** Majority class (Not Churn) dominates (~74%), making it difficult for models to detect actual churners.  
- **Data Cleaning:** Handling missing values and converting categorical variables were essential preprocessing steps.

## Methodology

### Data Preprocessing  
- Removed irrelevant columns such as customerID  
- Handled missing values in `TotalCharges`  
- Converted categorical variables using Label Encoding  

### Handling Imbalanced Classes  
- Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset by synthetically oversampling the minority churn class  
- Balanced classes improved model recall and F1-score, critical for detecting churners effectively

### Model Training & Selection  
Tested multiple algorithms including Logistic Regression, Random Forest, K-Nearest Neighbors, SVM, and XGBoost.  
- **Best Performer:** Tuned Random Forest model after SMOTE and hyperparameter optimization using GridSearchCV.

### Hyperparameter Tuning  
- Optimized parameters like `n_estimators`, `max_depth`, and `min_samples_split`  
- Achieved a recall of 67% for the churn class, significantly better than baseline

## Results  
| Metric     | Class 1 (Churn) | Notes                          |
|------------|-----------------|--------------------------------|
| Precision  | 0.54            | Percentage of predicted churners who were correct |
| Recall     | 0.67            | Percentage of actual churners correctly detected  |
| F1-Score   | 0.59            | Balance between precision and recall                |
| Accuracy   | ~76%            | Overall accuracy (less reliable due to imbalance)  |

**Note:** Recall is prioritized as catching churners is more valuable than overall accuracy.

## Business Impact  
The model allows the telecom company to identify customers at high risk of churning, enabling targeted retention efforts and reducing customer loss.

## What I Learned  
- The importance of data preprocessing and handling class imbalance in real-world datasets  
- How SMOTE can help improve minority class detection  
- The role of hyperparameter tuning in enhancing model performance  
- Balancing business value with statistical metrics (e.g., recall vs. accuracy)

## How to Run

1. Clone this repository  
2. Install dependencies:  
```bash
pip install -r requirements.txt
```
## Author
Gowtham Reddy S