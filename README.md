Email Spam Detection with Logistic Regression
In this project, I built two email spam classification models using Logistic Regression:

Without SMOTE: Trained on the original imbalanced dataset.
With SMOTE: Trained on SMOTE-resampled data to handle class imbalance.

What is SMOTE?
SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples for the minority class (in this case, spam) to balance the dataset and improve model performance on imbalanced data.

Results:

Without SMOTE:
Accuracy: 95.35%
Precision, Recall, F1-score for spam (0.0): 0.99, 0.63, 0.77
Classification Report shows issues with recall for spam detection.

With SMOTE:
Accuracy: 97.58%
Precision, Recall, F1-score for spam (0.0): 0.93, 0.88, 0.90
SMOTE improves recall and overall performance.

Files:
spam_classifier_model.pkl: Model without SMOTE.
spam_classifier_model_SMOTE.pkl: Model with SMOTE.

SMOTE significantly improved the model's ability to detect spam emails by balancing the data.

