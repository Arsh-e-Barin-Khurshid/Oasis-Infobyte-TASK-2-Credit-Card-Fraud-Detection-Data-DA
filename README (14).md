
# **Credit Card Fraud Detection Project**
## 1. Project Overview
This project focuses on detecting fraudulent transactions from a dataset of credit card transactions. Using machine learning classification techniques, we aim to accurately identify fraudulent transactions. The project involves data cleaning, exploratory data analysis (EDA), model development, hyperparameter tuning, and model evaluation.









## 2. Dataset Description
The dataset contains credit card transactions performed by European cardholders in September 2013. The dataset is highly imbalanced, with the majority of transactions being non-fraudulent. It contains 31 features: 28 anonymized features (V1, V2, ..., V28), along with Amount, Time, and the Class label, which indicates whether a transaction is fraudulent (1) or not (0).

Key columns include:

Time: Seconds elapsed between this transaction and the first in the dataset.
Amount: Transaction amount.
Class: Target variable where 1 indicates fraud and 0 indicates a legitimate transaction.




## 3. Problem Statement
Credit card fraud detection is crucial for financial security, but the imbalanced nature of fraud datasets makes it challenging to accurately classify fraudulent transactions. The goal is to build a machine learning model that can predict whether a transaction is fraudulent or not while minimizing false positives and negatives.


## 4. Data Preprocessing
The dataset required significant preprocessing to prepare it for model training:

Scaling: Features like Amount and Time were scaled using StandardScaler to normalize the data.
Imbalance Handling: The dataset was highly imbalanced, so techniques like resampling (over-sampling or under-sampling) or Synthetic Minority Oversampling Technique (SMOTE) were employed to balance the target class.
Feature Engineering: New features were created by scaling existing ones, and irrelevant features were dropped.


## 5. Exploratory Data Analysis (EDA)
To better understand the data, various plots were created:

Correlation Heatmap: Showed the relationships between features and highlighted key correlations with the target variable.
Class Distribution: Displayed the skewness in the dataset, where only a small fraction of transactions were fraudulent.
Box Plot: Provided insights into the distribution of transaction amounts for fraudulent vs. non-fraudulent transactions, revealing possible outliers.
Pair Plot: Displayed feature pair relationships to identify any separability between fraud and non-fraud transactions. EDA helped in identifying patterns and trends in the data, which is crucial for selecting the right algorithms.


## 6. Model Development
Multiple machine learning models were trained to classify fraud transactions:

Logistic Regression
Random Forest Classifier
XGBoost Classifier
Gradient Boosting
Decision Tree Each model was evaluated based on precision, recall, and F1-score, as well as accuracy. Given the imbalanced nature of the dataset, precision and recall were emphasized over accuracy.
## 7. Hyperparameter Tuning
Hyperparameter tuning was performed using Grid Search and Random Search to improve model performance. The hyperparameters for the classifiers, such as the number of estimators for Random Forest or the learning rate for XGBoost, were optimized. This process was critical for improving the model's ability to detect fraud while minimizing false alarms.

A line plot was created to show how model performance (accuracy, precision, recall) varied with changes in hyperparameters such as n_estimators and max_depth.
## 8. Model Evaluation
The performance of each model was evaluated using various metrics:

Confusion Matrix: Provided insight into the true positives, true negatives, false positives, and false negatives.
Precision and Recall: Key metrics for dealing with imbalanced datasets. Precision measures how many of the predicted fraud transactions were actually fraud, and recall measures how many actual fraud transactions were detected.
ROC Curve and AUC: Used to evaluate the performance of classifiers. The area under the curve (AUC) was used as a performance indicator.
The ROC curve visualization demonstrated the trade-off between sensitivity and specificity.

## 9. Data Visualization
Throughout the project, several types of visualizations were created to provide insights into the dataset and model performance:

Correlation Heatmap: Displayed feature correlations.
Box Plot: Visualized the distribution of transaction amounts.
Count Plot: Showed the class distribution of fraud and non-fraud transactions.
ROC Curve: Displayed the model's classification capabilities. These plots were essential in understanding data distribution and evaluating model behavior.


## 10. Conclusion and Future Work
The project demonstrated that machine learning can be effectively applied to detect credit card fraud, with models like Random Forest and XGBoost showing high performance when combined with hyperparameter tuning. Despite the success, the model still faces challenges such as false positives, which could be reduced through more advanced techniques like ensemble methods or deep learning. Future work could involve integrating more real-time detection systems and experimenting with deep learning models for better accuracy.
