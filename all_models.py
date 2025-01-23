#1.Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc, classification_report
import lightgbm as lgb
import joblib
import plotly.express as px
import math
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from openai import AsyncOpenAI
import nest_asyncio
import asyncio
#commenting below command for now
# %pip install nest_asyncio
from sklearn.manifold import TSNE



#2.Data Loading
file_path = "/content/Credit_Card_Eligibility_Data.xlsx"
data = pd.read_excel(file_path)

#3.Data Pre-processing
target = 'ok'
X = data.drop(columns=[target])
y = data[target]

# Count of features before data cleaning
print(f"Number of features before cleaning: {X.shape[1]}")


# # Scaling the target variable
# scaler = MinMaxScaler()
# y = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

# Remove columns with too many missing values (e.g., more than 50%)
missing_threshold = 0.5  # You can adjust this threshold as needed
columns_with_missing_data = X.columns[X.isnull().mean() >= missing_threshold]
X = X.loc[:, X.isnull().mean() < missing_threshold]

# Remove columns with only one unique value
columns_with_one_value = X.columns[X.nunique() <= 1]
X = X.loc[:, X.nunique() > 1]

# Features dropped
dropped_features = list(columns_with_missing_data) + list(columns_with_one_value)

# Count of features after data cleaning
print(f"Number of features after cleaning: {X.shape[1]}")

# Print the names of the dropped features
if dropped_features:
    print("\nDropped features:")
    print(dropped_features)
else:
    print("\nNo features were dropped.")

# Identify column types
binary_columns = [col for col in X.columns if set(X[col].dropna().unique()).issubset({0, 1, 't', 'f'})]
numeric_columns = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
categorical_columns = [col for col in X.columns if 1 <= X[col].nunique() <= 50 and col not in binary_columns]
numeric_columns = [col for col in numeric_columns if col not in categorical_columns]


# Handle binary columns
for col in binary_columns:
    X[col] = X[col].replace([float('inf'), float('-inf')], 0).fillna(0)
    # Replace NaN with 0
    X[col] = X[col].fillna(0)
    X[col] = X[col].map({'t': 1, 'f': 0, 0: 0}).fillna(0).astype(int)

# Handle numeric columns
for col in numeric_columns:
    X[col] = (X[col]
              .replace([float('inf'), float('-inf')], 0)
              .fillna(0)
              .astype(str)
              .str.replace(';', '.', regex=False)
              .str.extract(r'(\d+\.?\d*)')[0]
              .fillna(0)
              .astype(float))

# Encoding categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    unique_values = X[col].astype(str).unique()
    le.fit(unique_values)
    X[col] = le.transform(X[col].astype(str))
    label_encoders[col] = le


#Check data imbalance
# Count the occurrences of each class in the target column
target_counts = y.value_counts()

# Calculate the percentage of each class
target_percentage = y.value_counts(normalize=True) * 100

# Print the counts and percentages
print(f"Target counts:\n{target_counts}")
print(f"\nTarget percentages:\n{target_percentage}")

# Plot a pie chart
plt.figure(figsize=(6, 6))
plt.pie(target_counts, labels=[f"Class {i} - {pct:.2f}%" for i, pct in zip(target_counts.index, target_percentage)],
        autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=140, wedgeprops={'edgecolor': 'black'})
plt.title("Class Distribution in Target Column")
plt.show()

# 4. Anomaly Detection using Isolation Forest

# Use only numeric columns for anomaly detection
X_numeric = X[numeric_columns]

# Anomaly detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
X['Anomaly_Flag'] = iso_forest.fit_predict(X_numeric)
X['Anomaly_Flag'] = X['Anomaly_Flag'].apply(lambda x: 1 if x == -1 else 0)

# Display anomalies in the data
anomalies = X[X['Anomaly_Flag'] == 1]

# Print the anomalies table
print("Anomalies Detected:")
print(anomalies)

# Optionally, you can also plot the distribution of anomalies if needed
sns.countplot(x='Anomaly_Flag', data=X)
plt.title('Anomaly Flag Distribution')
plt.xlabel('Flag (-1: Anomaly, 1: Normal)')
plt.ylabel('Count')
# Add counts under the plot
flag_counts = X['Anomaly_Flag'].value_counts()
for i, count in enumerate(flag_counts):
    plt.text(i, count + 5, f'{count}', ha='center', va='bottom', fontsize=12, color='black')

plt.show()



# # Filter rows flagged as anomalies
# X_cleaned = X[X['Anomaly_Flag'] == 0].drop(columns=['Anomaly_Flag'])

# # Filter corresponding rows in y using the same indices
# y_cleaned = y[X_cleaned.index]

# # Check the shapes of X_cleaned and y_cleaned
# print(f"Original X shape: {X.shape}")
# print(f"Original y shape: {y.shape}")
# print(f"Cleaned X shape: {X_cleaned.shape}")
# print(f"Cleaned y shape: {y_cleaned.shape}")


# X = X_cleaned
# y = y_cleaned

#Clustering for data interpretation


# Apply KMeans clustering to the combined data
kmeans = KMeans(n_clusters=3, random_state=42)
X['Cluster_Label'] = kmeans.fit_predict(X)

# Plot distribution of clusters
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster_Label',hue='Cluster_Label', data=X, palette='viridis', legend=False)
plt.title('Cluster Distribution')
plt.xlabel('Cluster Label')
plt.ylabel('Count')
plt.show()

#Scatter plot for clusters
# Assuming the features are in X_combined and the cluster label is in 'Cluster_Label'
tsne = TSNE(n_components=2, random_state=42)
tsne_components = tsne.fit_transform(X.drop('Cluster_Label', axis=1))  # Drop 'Cluster_Label' for t-SNE

# Create a new DataFrame with t-SNE components and cluster labels
tsne_df = pd.DataFrame(tsne_components, columns=['TSNE1', 'TSNE2'])
tsne_df['Cluster_Label'] = X['Cluster_Label']

# Plotting the 2D scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster_Label', data=tsne_df, palette='viridis')
plt.title('t-SNE of Features with Cluster Labels')
plt.show()


# Plot distribution of anomalies within clusters
plt.figure(figsize=(10, 6))
sns.countplot(x='Anomaly_Flag', hue='Cluster_Label', data=X, palette='viridis')
plt.title('Anomaly Flag Distribution within Clusters')
plt.xlabel('Anomaly Flag (0: Normal, 1: Anomaly)')
plt.ylabel('Count')
plt.show()



# Cluster summary and averages
cluster_summary = X.groupby('Cluster_Label').describe()
cluster_averages = X.groupby('Cluster_Label').mean()

# Print cluster summary and averages for interpretation
print("Cluster Summary (Descriptive Statistics):")
print(cluster_summary)

print("\nCluster Feature Averages:")
print(cluster_averages)

# Optional: Save cluster summary and feature averages to Excel
cluster_summary.to_excel("Cluster_Summary_Detailed.xlsx", sheet_name="Summary_Detailed")
cluster_averages.to_excel("Cluster_Feature_Averages.xlsx", sheet_name="Feature_Averages")


#Data Interpretation of Clusters

# Allow nested async loops
nest_asyncio.apply()


# Asynchronous function to interpret clusters
async def interpret_clusters(cluster_averages, openai_api_key):
    # Convert cluster_averages to a JSON-like string for the prompt
    cluster_data = cluster_averages.reset_index().to_string(index=False)

    # Construct the OpenAI prompt
    prompt = f"""
    You are a data scientist providing business insights. The following table summarizes the average feature values for clusters created using K-Means clustering. Interpret the clusters and describe their potential business implications:

    {cluster_data}

    Please describe each cluster, its characteristics, and its business implications in detail.
    """

    # OpenAI API client
    client = AsyncOpenAI(api_key=openai_api_key)

    # Make the asynchronous API call
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    print("response", response)
    print("choices", response.choices)  # Access the 'choices' attribute directly
    print("message", response.choices[0].message)  # Access the message attribute in the first choice
    print(dir(response.choices[0].message))  # Print the attributes of the message object

    # Extract and return the interpretation
    interpretation = response.choices[0].message.content

    return interpretation


# Run the async code in a notebook
async def main():
    openai_api_key = "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"  # Replace with your OpenAI API key
    interpretation = await interpret_clusters(cluster_averages, openai_api_key)
    print("\nCluster Interpretations:\n")
    print(interpretation)

# Execute the main function in the current notebook environment
#added a fix here await comes under async
asyncio.run(main())


# Train Random Forest model for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X.drop(columns=['Cluster_Label']), X['Cluster_Label'])

# Get feature importances from the trained model
feature_importances = rf.feature_importances_

# Create a DataFrame for the feature importances
importance_df = pd.DataFrame({
    'Feature': X.drop(columns=['Cluster_Label']).columns,
    'Importance': feature_importances
})

# Sort the features by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Convert 'Importance' to percentages
importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100

#plotting graph
fig = px.pie(
    importance_df,
    values='Percentage',
    names='Feature',
    title='Feature Importance',
    hover_data={'Percentage': ':.4f'},  # Display percentage with 4 decimal places for hover
    color_discrete_sequence=px.colors.qualitative.Pastel  # Pastel color palette
)

# Customize the trace for label representation
fig.update_traces(
    hovertemplate='<b>Feature:</b> %{label}<br><b>Percentage:</b> %{value:.4f}%<extra></extra>',
    textinfo='percent+label',  # Show percentage and label on slices
    textfont=dict(size=12),  # Adjust text font size
    textposition='inside'  # Ensure text is displayed inside the pie slices
)

# Customize the layout for a neat presentation
fig.update_layout(
    title_x=0.44,  # Center align the title
    font=dict(size=14),  # General font size
    margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins
    legend=dict(  # Place legend outside the chart
        x=1,
        y=0.5,
        traceorder='normal',
        font=dict(size=12)
    )
)

# Show the pie chart
fig.show()

# 5. Exploratory Data Analysis (EDA)
def plot_bar_graphs(numeric_columns, X):
    num_cols = len(numeric_columns)
    rows = math.ceil(num_cols / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
    axes = axes.flatten()
    for idx, column in enumerate(numeric_columns):
      sns.histplot(X[column], kde=True, bins=20, color='skyblue', ax=axes[idx])
      axes[idx].set_title(f"Distribution of {column}")
      axes[idx].set_xlabel("Value")
      axes[idx].set_ylabel("Count")  #
    for idx in range(num_cols, len(axes)):
      fig.delaxes(axes[idx])
plt.tight_layout()
plt.show()

def plot_pie_charts(categorical_columns, X):
   num_cols = len(categorical_columns)
   rows = math.ceil(num_cols / 4)
   fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
   axes = axes.flatten()
   for idx, column in enumerate(categorical_columns):
       data = X[column].value_counts()
       axes[idx].pie(data, startangle=90, colors=sns.color_palette('viridis', len(data)))
       axes[idx].set_title(f"Distribution of {column}")
       axes[idx].set_ylabel("")
   for idx in range(num_cols, len(axes)):
       fig.delaxes(axes[idx])
plt.tight_layout()
plt.show()


# EDA Plots
plot_bar_graphs(numeric_columns, X)
plot_pie_charts(categorical_columns, X)



# Correlation Heatmap
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
sns.heatmap(X[numeric_columns].corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap for Numeric Features")

plt.subplot(1, 2, 2)
sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap for All Features")

plt.tight_layout()
plt.show()



#Model Training using Random Forest


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=np.digitize(y, bins=[0.5]))

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

cv = StratifiedKFold(n_splits=5)
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
model = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)

grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring="roc_auc", verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'random_forest_model.pkl')



# Print the Best Parameters and ROC AUC Score
print("Best Parameters:", grid_search.best_params_)
print("Best ROC AUC Score (Cross-Validation):", grid_search.best_score_)

# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
# import joblib

# # 1. Stratified Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y
# )

# # 2. Define Class Weights for Handling Imbalance
# class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
# class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# # 3. Hyperparameter Grid
# param_grid = {
#     "n_estimators": [50, 100, 150],
#     "max_depth": [None, 10, 20],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
# }

# # 4. Initialize Random Forest with Class Weights
# model = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)

# # 5. Set a Target ROC-AUC Threshold
# target_roc_auc = 0.9  # Desired ROC-AUC threshold
# max_iterations = 10   # Maximum number of iterations to try
# iteration = 0
# best_score = 0
# best_model = None
# best_params = None

# while iteration < max_iterations:
#     print(f"\nIteration {iteration + 1}: Searching for best model...")

#     # Perform Grid Search with AUC ROC as Scoring Metric
#     cv = StratifiedKFold(n_splits=5)
#     grid_search = GridSearchCV(
#         model, param_grid=param_grid, cv=cv, scoring="roc_auc", verbose=1, n_jobs=-1
#     )

#     # Train the Model
#     grid_search.fit(X_train, y_train)
#     current_best_model = grid_search.best_estimator_
#     current_best_params = grid_search.best_params_
#     current_best_score = grid_search.best_score_

#     print(f"Best Parameters: {current_best_params}")
#     print(f"Best ROC AUC Score (Cross-Validation): {current_best_score:.4f}")

#     # Update the best model if current is better
#     if current_best_score > best_score:
#         best_score = current_best_score
#         best_model = current_best_model
#         best_params = current_best_params

#     # Save the best model
#     joblib.dump(best_model, f'random_forest_model_iteration_{iteration + 1}.pkl')

#     # Check if the target ROC-AUC score is met
#     if best_score >= target_roc_auc:
#         print(f"Target ROC-AUC score of {target_roc_auc} achieved! Stopping iterations.")
#         break

#     # Increment iteration counter
#     iteration += 1

# if iteration == max_iterations:
#     print(f"Maximum iterations reached. Best ROC-AUC score: {best_score:.4f}")

# # Final Best Model and Parameters
# print("\nFinal Best Parameters:", best_params)
# print(f"Final Best ROC AUC Score (Cross-Validation): {best_score:.4f}")


#Eavluation of RF model


# 1. Make predictions on the training set to calculate the threshold
train_predictions_proba = best_model.predict_proba(X_train)[:, 1]

# 2. Calculate the threshold based on the 13% quantile from training predictions
threshold = np.quantile(train_predictions_proba, 1 - 0.13)  # Corresponds to the 87% quantile
print(f"Threshold based on 13% quantile: {threshold:.4f}")

# 3. Make predictions on the test set
test_predictions_proba = best_model.predict_proba(X_test)[:, 1]
predictions = (test_predictions_proba >= threshold).astype(int)

# 4. Generate ROC-AUC and Gini Coefficient
fpr, tpr, _ = roc_curve(y_test, test_predictions_proba)
roc_auc = auc(fpr, tpr)
gini_coeff = 2 * roc_auc - 1

# 5. Calculate Classification Metrics
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
correct_predictions = (predictions == y_test).sum()
incorrect_predictions = (predictions != y_test).sum()

# 6. Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)

# 7. Store Metrics in a DataFrame
metrics_table = pd.DataFrame({
    "Metric": [
        "Threshold",
        "Gini Coefficient",
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "Correct Predictions",
        "Incorrect Predictions"
    ],
    "Value": [
        threshold,
        gini_coeff,
        roc_auc,
        precision,
        recall,
        f1,
        correct_predictions,
        incorrect_predictions
    ]
})

# Print the metrics table
print("\nClassification Metrics Table:")
print(metrics_table)

# 8. Confusion Matrix in Tabular Form
conf_matrix_table = pd.DataFrame(
    conf_matrix,
    index=["Actual Class 0", "Actual Class 1"],
    columns=["Predicted Class 0", "Predicted Class 1"]
)
print("\nConfusion Matrix:")
print(conf_matrix_table)


# import pandas as pd
# import numpy as np
# import joblib

# from sklearn.preprocessing import LabelEncoder
# import joblib

# # Example: Encode categorical features
# label_encoders = {}
# categorical_columns = ["Income_type", "Education_type", "Family_status", "Housing_type", "Occupation_type"]

# for col in categorical_columns:
#     le = LabelEncoder()
#     X_train[col] = le.fit_transform(X_train[col])
#     label_encoders[col] = le  # Store the encoder for later use

# # Save the encoders
# joblib.dump(label_encoders, "label_encoders.pkl")

# # Load the trained model
# best_model = joblib.load("random_forest_model.pkl")



# # Load the saved encoders
# label_encoders = joblib.load("label_encoders.pkl")

# # Single entry data (without the target column)
# single_entry = [
#     5008814, 1, 1, 1, 1, 0, 0, 0, 0, 2, 15, 427400,
#     34.86857362, 13.43557363, "Working", "Higher education",
#     "Civil marriage", "Rented apartment", "Other"
# ]

# # Define column names (must match training data)
# columns = [
#     "ID", "Gender", "Own_car", "Own_property", "Work_phone", "Phone",
#     "Email", "Unemployed", "Num_children", "Num_family", "Account_length",
#     "Total_income", "Age", "Years_employed", "Income_type",
#     "Education_type", "Family_status", "Housing_type", "Occupation_type"
# ]

# # Create a DataFrame for the single entry
# single_entry_df = pd.DataFrame([single_entry], columns=columns)

# # Preprocess the single entry
# categorical_columns = ["Income_type", "Education_type", "Family_status", "Housing_type", "Occupation_type"]

# for col in categorical_columns:
#     # Handle unseen categories by mapping them to 'unknown' or a default category
#     valid_categories = label_encoders[col].classes_
#     single_entry_df[col] = single_entry_df[col].apply(
#         lambda x: x if x in valid_categories else valid_categories[0]
#     )
#     # Transform using the encoder
#     single_entry_df[col] = label_encoders[col].transform(single_entry_df[col])

# # Handle missing values or scaling if applied during training
# single_entry_df = single_entry_df.fillna(0)

# # Predict the class and probabilities
# predicted_class = best_model.predict(single_entry_df)[0]
# predicted_proba = best_model.predict_proba(single_entry_df)[0]

# # Print results
# print(f"Predicted Class: {predicted_class}")
# print(f"Predicted Probabilities: {predicted_proba}")





#Model Training with xgb
# 1. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=np.digitize(y, bins=[0.5])
)

# 2. Compute Class Imbalance Ratio for scale_pos_weight
imbalance_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

# 3. Define the Model (with scale_pos_weight)
model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric="auc",
    scale_pos_weight=imbalance_ratio,
)

# 4. Define Cross-Validation Strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. Define Specific Hyperparameters (instead of a grid)
hyperparameters = {
    "n_estimators": 200,
    "max_depth": 10,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}
model.set_params(**hyperparameters)

# 6. Evaluate the Model Using Stratified K-Fold Cross-Validation
# If X_train and y_train are Pandas DataFrames or Series, reset index and convert to NumPy arrays at the start.
if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
          X_train = X_train.reset_index(drop=True).to_numpy()
          y_train = y_train.reset_index(drop=True).to_numpy()
scores = []
for train_idx, test_idx in cv.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[test_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[test_idx]

    model.fit(X_train_fold, y_train_fold)
    y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
    roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
    gini = 2 * roc_auc - 1
    scores.append(gini)

mean_gini = np.mean(scores)  # Mean Gini for this set of hyperparameters
print(f"Hyperparameters: {hyperparameters}, Mean Gini: {mean_gini:.4f}")

# Save the model if it's good
if mean_gini > 0.8:  # Example threshold
    joblib.dump(model, "best_xgboost_model.pkl")
    print("Model saved as 'best_xgboost_model.pkl'")


joblib.dump(model, "best_xgboost_model.pkl")  # Save the model
best_model_xgb = joblib.load("best_xgboost_model.pkl")  # Load the model for usage

#Model Eval with xgb
# Making predictions
predictions_proba = best_model_xgb.predict_proba(X_test)[:, 1]
threshold = 0.5
predictions = (predictions_proba >= threshold).astype(int)

# ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, predictions_proba)
roc_auc = auc(fpr, tpr)
gini_coeff = 2 * roc_auc - 1


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC-AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
print(f"Gini Coefficient: {gini_coeff:.4f}")
plt.legend(loc="lower right")
plt.show()

# KS Statistic Plot
def ks_statistic(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return max(tpr - fpr)

ks_stat = ks_statistic(y_test, predictions_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr - fpr, color="purple", lw=2, label=f"KS Statistic = {ks_stat:.2f}")
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("Cumulative Difference (TPR - FPR)")
plt.title("KS Curve")
plt.legend(loc="lower right")
plt.show()

# 9. Results and Custom Metric Analysis
results = pd.DataFrame({
    "Actual (Scaled)": y_test,
    # "Actual (Original)": scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(),
    "Predicted Score": predictions_proba
})
print(results)
print(classification_report(y_test, predictions))

# Custom Metric vs Threshold plot
def plot_threshold_vs_custom_metric(predictions_proba, y_test):
    thresholds = np.linspace(0, 1, 101)
    custom_metric_values = []

    for threshold in thresholds:
        predictions = (predictions_proba >= threshold).astype(int)

        correct_predictions = (predictions == y_test).sum()
        incorrect_predictions = (predictions != y_test).sum()

        custom_metric = correct_predictions * 100 - incorrect_predictions * 500
        custom_metric_values.append(custom_metric)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, custom_metric_values, label="Custom Metric", color='blue', lw=2)
    plt.xlabel("Threshold")
    plt.ylabel("Custom Metric")
    plt.title("Custom Metric vs Threshold")
    plt.grid(True)
    plt.show()
# Plotting custom metric vs threshold
plot_threshold_vs_custom_metric(predictions_proba, y_test)

correct_predictions = (predictions == y_test).sum()
incorrect_predictions = (predictions != y_test).sum()

print(f"Number of correct predictions: {correct_predictions}")
print(f"Number of incorrect predictions: {incorrect_predictions}")




# 1. Make predictions on the training set to calculate the threshold
train_predictions_proba = best_model_xgb.predict_proba(X_train)[:, 1]

# 2. Calculate the threshold based on the 13% quantile from training predictions
threshold = np.quantile(train_predictions_proba, 1 - 0.13)  # Corresponds to the 87% quantile
print(f"Threshold based on 13% quantile: {threshold:.4f}")

# 3. Make predictions on the test set
test_predictions_proba = best_model_xgb.predict_proba(X_test)[:, 1]
predictions = (test_predictions_proba >= threshold).astype(int)

# 4. Generate ROC-AUC and Gini Coefficient
fpr, tpr, _ = roc_curve(y_test, test_predictions_proba)
roc_auc = auc(fpr, tpr)
gini_coeff = 2 * roc_auc - 1

# 5. Calculate Classification Metrics
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
correct_predictions = (predictions == y_test).sum()
incorrect_predictions = (predictions != y_test).sum()

# 6. Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)

# 7. Store Metrics in a DataFrame
metrics_table = pd.DataFrame({
    "Metric": [
        "Threshold",
        "Gini Coefficient",
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "Correct Predictions",
        "Incorrect Predictions"
    ],
    "Value": [
        threshold,
        gini_coeff,
        roc_auc,
        precision,
        recall,
        f1,
        correct_predictions,
        incorrect_predictions
    ]
})

# Print the metrics table
print("\nClassification Metrics Table:")
print(metrics_table)

# 8. Confusion Matrix in Tabular Form
conf_matrix_table = pd.DataFrame(
    conf_matrix,
    index=["Actual Class 0", "Actual Class 1"],
    columns=["Predicted Class 0", "Predicted Class 1"]
)
print("\nConfusion Matrix:")
print(conf_matrix_table)




#Model Training with lighgbm
# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=np.digitize(y, bins=[0.5]))

# 7. Model Building (LightGBM with Grid Search)

# Define LightGBM parameter grid
param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [-1, 10, 20, 30],
    "num_leaves": [50, 100],
    "learning_rate": [0.1, 0.05, 0.01],
    "min_child_samples": [10, 20, 30],
}

model = lgb.LGBMClassifier(random_state=42, class_weight='balanced')

# Modify GridSearchCV to use F1-score as the evaluation metric
grid_search = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5),
    scoring="roc_auc",  # Use F1-score as the evaluation metric
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Get the best model
best_model1 = grid_search.best_estimator_

# Saving the best model
joblib.dump(best_model1, 'lightgbm_model_with_f1_score.pkl')

# Print the best parameters
print("Best Parameters found by Grid Search:", grid_search.best_params_)


# 8. Model Evaluation with lightgbm

# Making predictions
predictions_proba = best_model1.predict_proba(X_test)[:, 1]
threshold = 0.5
predictions = (predictions_proba >= threshold).astype(int)

# ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, predictions_proba)
roc_auc = auc(fpr, tpr)
gini_coeff = 2 * roc_auc - 1


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC-AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
print(f"Gini Coefficient: {gini_coeff:.4f}")
plt.legend(loc="lower right")
plt.show()

# KS Statistic Plot
def ks_statistic(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return max(tpr - fpr)

ks_stat = ks_statistic(y_test, predictions_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr - fpr, color="purple", lw=2, label=f"KS Statistic = {ks_stat:.2f}")
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("Cumulative Difference (TPR - FPR)")
plt.title("KS Curve")
plt.legend(loc="lower right")
plt.show()

# 9. Results and Custom Metric Analysis
results = pd.DataFrame({
    "Actual (Scaled)": y_test,

    "Predicted Score": predictions_proba
})
print(results)
print(classification_report(y_test, predictions))

# Custom Metric vs Threshold plot
def plot_threshold_vs_custom_metric(predictions_proba, y_test):
    thresholds = np.linspace(0, 1, 101)
    custom_metric_values = []

    for threshold in thresholds:
        predictions = (predictions_proba >= threshold).astype(int)

        correct_predictions = (predictions == y_test).sum()
        incorrect_predictions = (predictions != y_test).sum()

        custom_metric = correct_predictions * 100 - incorrect_predictions * 500
        custom_metric_values.append(custom_metric)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, custom_metric_values, label="Custom Metric", color='blue', lw=2)
    plt.xlabel("Threshold")
    plt.ylabel("Custom Metric")
    plt.title("Custom Metric vs Threshold")
    plt.grid(True)
    plt.show()
# Plotting custom metric vs threshold
plot_threshold_vs_custom_metric(predictions_proba, y_test)

correct_predictions = (predictions == y_test).sum()
incorrect_predictions = (predictions != y_test).sum()

print(f"Number of correct predictions: {correct_predictions}")
print(f"Number of incorrect predictions: {incorrect_predictions}")




# 1. Make predictions on the training set to calculate the threshold
train_predictions_proba = best_model1.predict_proba(X_train)[:, 1]

# 2. Calculate the threshold based on the 13% quantile from training predictions
threshold = np.quantile(train_predictions_proba, 1 - 0.13)  # Corresponds to the 87% quantile
print(f"Threshold based on 13% quantile: {threshold:.4f}")

# 3. Make predictions on the test set
test_predictions_proba = best_model1.predict_proba(X_test)[:, 1]
predictions = (test_predictions_proba >= threshold).astype(int)

# 4. Generate ROC-AUC and Gini Coefficient
fpr, tpr, _ = roc_curve(y_test, test_predictions_proba)
roc_auc = auc(fpr, tpr)
gini_coeff = 2 * roc_auc - 1

# 5. Calculate Classification Metrics
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
correct_predictions = (predictions == y_test).sum()
incorrect_predictions = (predictions != y_test).sum()

# 6. Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)

# 7. Store Metrics in a DataFrame
metrics_table = pd.DataFrame({
    "Metric": [
        "Threshold",
        "Gini Coefficient",
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "Correct Predictions",
        "Incorrect Predictions"
    ],
    "Value": [
        threshold,
        gini_coeff,
        roc_auc,
        precision,
        recall,
        f1,
        correct_predictions,
        incorrect_predictions
    ]
})

# Print the metrics table
print("\nClassification Metrics Table:")
print(metrics_table)

# 8. Confusion Matrix in Tabular Form
conf_matrix_table = pd.DataFrame(
    conf_matrix,
    index=["Actual Class 0", "Actual Class 1"],
    columns=["Predicted Class 0", "Predicted Class 1"]
)
print("\nConfusion Matrix:")
print(conf_matrix_table)



