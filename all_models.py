#1.Import libraries
import pandas as pd
import numpy as np
import shap
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
from sklearn.manifold import TSNE
from kneed import KneeLocator
import streamlit as st
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import logging


# Main Streamlit App
def main():
    st.title("Data Visualization and Model Building")

    mode = st.sidebar.selectbox("Select Mode", ["Train Model", "Test Model"])

    if mode == "Train Model":
        train_model()
    elif mode == "Test Model":
        test_model()

# Train Model Function
def train_model():
    st.subheader("Upload Training Dataset")
    uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=["csv", "xls", "xlsx"])

    if uploaded_file:
        # Determine file type based on the uploaded file extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)  # Read CSV file
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)  # Read Excel file

        st.write("### Dataset Overview")
        st.dataframe(df.head())

        # Preprocess data
        target_column = "ok"
        feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns]
        y = df[target_column]

        print("Unique values in the column:", y.unique())

        # Data Cleaning and Processing (Train phase)
        X, dropped_features, categorical_columns, numeric_columns,binary_columns = data_processing(X, is_training=True)
        print("categorical_cols", categorical_columns)
        print("binary_columns",binary_columns)

        # Visualizations
        if st.checkbox("Show Data Visualizations"):
            show_data_imbalance(y)
            # show_correlation_heatmap(X)
        
        # Exploratory Data Analysis (EDA)
        perform_eda(X)

        # Anomaly Detection
        detect_anomalies(X)

        # Drop the 'Anomaly_Flag' column after anomaly detection
        if 'Anomaly_Flag' in X.columns:
            X = X.drop(columns=['Anomaly_Flag'])

        # Clustering (Optional)
        st.write("Clustering Results")
        cluster_averages = show_clustering(X)

        # Interpret Clusters
        st.write("### Cluster Interpretations")
        openai_api_key = "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"  # Fetch API key from secrets or another source
        interpretation = asyncio.run(interpret_clusters(cluster_averages, openai_api_key))
        st.text(interpretation)

        # Drop 'Cluster_Label' column if it exists
        if 'Cluster_Label' in X.columns:
            X = X.drop(columns=['Cluster_Label'])
            # st.write("### 'Cluster_Label' Dropped After Clustering")

        # Call the feature importance plotting function
        st.write("### Feature Importance Visualization")
        feature_importance_fig = plot_feature_importance(X, y)  # Get the figure
        st.plotly_chart(feature_importance_fig)  # Display the plot in Streamlit

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=np.digitize(y, bins=[0.5]))
        
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        cv = StratifiedKFold(n_splits=5)
        # class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        # class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        model = RandomForestClassifier(random_state=42, class_weight="balanced")

        grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring="roc_auc", verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        
        # Save the trained model
        with open('trained_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        # Call the evaluation metrics and plots
        metrics_table = evaluate_model(best_model, X_train, X_test, y_train, y_test)
        # Extract the Gini Coefficient from the metrics table
        model_score = metrics_table.loc[metrics_table["Metric"] == "Gini Coefficient", "Value"].values[0]

        # Display the Model Score
        st.write(f"### Model Score: {model_score:.4f}")
        st.write("### Model Training Completed!")
        # Extract Feature Importances
        feature_importances = dict(zip(X_train.columns, best_model.feature_importances_))

        # Filter only categorical feature importances
        categorical_importances = {feat: imp for feat, imp in feature_importances.items() if feat in categorical_columns}

        print("categorical_importances", categorical_importances)
        # 🚀 Call Feature Analysis Functions
        numerical_feature_analysis(df, target_col=target_column, numeric_columns=numeric_columns)
        categorical_feature_analysis(df, target_col=target_column, feature_importances=categorical_importances, categorical_features=categorical_columns)
        binary_feature_analysis(df, target_col=target_column, binary_columns=binary_columns, feature_importances=feature_importances)
   
    else:
        st.write("Model or Data not available!")

# Test Model Function

def test_model():
     # Set the threshold slider before dataset upload
    st.sidebar.write("## Adjust Threshold")
    threshold = st.number_input(
            "Enter threshold value:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            format="%.2f"
        )
    st.subheader("Upload Test Dataset")
    uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=["csv", "xls", "xlsx"])
    
    if uploaded_file:
        # Determine file type based on the uploaded file extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)  # Read CSV file
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)  # Read Excel file

        st.write("### Dataset Overview")
        st.dataframe(df.head())

        # Check if the target column exists
        target_column = "ok"
        if target_column in df.columns:
            # If target column exists, separate features and target
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns]
            y = df[target_column]
        else:
            # If no target column, just use all columns as features
            feature_columns = df.columns.tolist()  # All columns are features
            X = df[feature_columns]
            y = None  # No target available

        # Data Cleaning and Processing
        X, dropped_features, categorical_columns, numeric_columns, binary_columns = data_processing(X, is_training=False)

        # Show dropped features
        if dropped_features:
            st.write("### Dropped Features")
            st.write(dropped_features)

        # Load the trained model
        try:
            with open('trained_model.pkl', 'rb') as f:
                rf = pickle.load(f)
        except FileNotFoundError:
            st.write("Model not found. Please train the model first.")
            return
        
        # Make predictions (probabilities)
        test_predictions_proba = rf.predict_proba(X)[:, 1]

        # Apply threshold to make predictions
        predictions = (test_predictions_proba >= threshold).astype(int)

        # Display predictions
        st.write("### Predictions:")
        st.write(pd.DataFrame({"Predicted": predictions,
                               "Probability": test_predictions_proba}))

                # --------------- SHAP EXPLANATION START ---------------
            # Initialize SHAP explainer for Random Forest
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)

        # Extract SHAP values for the selected record
        selected_index = st.number_input("Select a row to explain:", min_value=0, max_value=len(X)-1, value=0, step=1)

        # Extract SHAP values for the selected record
        # Use shap_values directly if binary output isn't handled well
        shap_contributions = shap_values[selected_index] if len(shap_values) == len(X) else shap_values[1][selected_index]
        # shap_contributions[:, 1] selects the second column (class 1 contributions)
        shap_contributions = shap_contributions[:, 1] 

        feature_names = X.columns.tolist()
        feature_values = X.iloc[selected_index].values

        # 🔍 Debugging Section (Add this before the assertion)
        print(f"Feature Names Length: {len(feature_names)}")
        print(f"SHAP Contributions Length: {len(shap_contributions)}")
        print(f"Feature Values Length: {len(feature_values)}")

        # Check for feature mismatches
        print("Model Feature Names:", rf.feature_names_in_)
        print("Test Data Feature Names:", feature_names)

        # Confirm all lengths match before creating the DataFrame
        assert len(feature_names) == len(shap_contributions) == len(feature_values), "Mismatch in array lengths!"
                # Debugging the shape
        print(f"SHAP Contributions Type: {type(shap_contributions)}")
        print(f"SHAP Contributions Shape: {getattr(shap_contributions, 'shape', 'N/A')}")
        print(f"SHAP Contributions Example: {shap_contributions}")
                # Flatten if needed
       
        if isinstance(shap_contributions, np.ndarray) and shap_contributions.ndim > 1:
            shap_contributions = shap_contributions.flatten()
        elif isinstance(shap_contributions, list) and isinstance(shap_contributions[0], (list, np.ndarray)):
            shap_contributions = np.array(shap_contributions).flatten()
        # Create the DataFrame
        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": shap_contributions,
            "Value": feature_values
        }).sort_values(by="Contribution", key=abs, ascending=False).head(5)  

        print("feature_importance_shaply", feature_importance)

        # Display prediction score and decision
        prediction_score = test_predictions_proba[selected_index]
        decision = "Approve" if prediction_score >= threshold else "Decline"

        st.markdown(f"## Scoring details for row #{selected_index}")
        st.markdown(f"### **{prediction_score:.5f}** — **{decision}**")

        # Show top 5 contributing features
        st.write("### Top 5 Feature Contributions:")
        st.table(feature_importance)
        # Sort top 5 features by absolute contribution
        feature_importance = feature_importance.sort_values(by="Contribution", key=abs, ascending=True)

                # --------------- VISUALIZATION SECTION ---------------
        # --------------- PLOT SECTION ---------------
        st.write("### Visual Explanation (Top 5 Features)")

        # Set up the figure with a wider width to accommodate more spacing
        fig, ax = plt.subplots(figsize=(10, 6))  # Increased width for balanced layout

        # Color mapping: Blue for positive, Red for negative contributions
        colors = ['#4A90E2' if val > 0 else '#D0021B' for val in feature_importance['Contribution']]

        # Horizontal bar plot
        bars = ax.barh(
            feature_importance['Feature'],         # Y-axis: Feature names
            feature_importance['Contribution'],    # X-axis: Contribution values
            color=colors,                          # Color based on contribution direction
            height=0.5                             # Adjust bar thickness
        )

        # --------------- FIX: REMOVE Y-TICK LABELS ---------------
        ax.set_yticks([])  # Remove automatic Y-axis labels to avoid duplication

        # --------------- ANNOTATIONS (TEXT LABELS) ---------------
        # Add SHAP contribution scores and feature values with proper spacing
        for bar, (feature, contrib, value) in zip(bars, 
            zip(feature_importance['Feature'], 
                feature_importance['Contribution'], 
                feature_importance['Value'])):
            
            bar_width = bar.get_width()
            bar_y = bar.get_y() + bar.get_height() / 2  # Center vertically
            
            # 1️⃣ Feature Name (aligned to the far left)
            ax.text(-0.75, bar_y, f'{feature}', va='center', ha='right', fontsize=10, color='black')
            
            # 2️⃣ SHAP Contribution Score (closer to the bar for better alignment)
            ax.text(-0.65, bar_y, f'{value}', va='center', ha='right', fontsize=10, color='black')
            
                        # 3️⃣ Feature Value with Dynamic Offset
            if contrib > 0:
                # Dynamic offset: increases with bar length
                offset = 0.02 + abs(bar_width) * 0.1
                ax.text(bar_width + offset, bar_y, f'{contrib:.4f}', va='center', ha='left', fontsize=10, color='black')
            else:
                # For negative contributions, keep the value close to the bar
                ax.text(bar_width - 0.03, bar_y, f'{contrib:.4f}', va='center', ha='right', fontsize=10, color='black')

                            # --------------- DYNAMIC X-AXIS LIMIT ---------------
        # Calculate dynamic range to extend positive side by 20%
        max_contrib = feature_importance['Contribution'].max()
        x_max = max_contrib + (0.2 * abs(max_contrib))  # Add 20% buffer
        x_min = -0.8  # Keep the negative side fixed

        # Apply the new x-axis limits
        ax.set_xlim(x_min, x_max)

        # --------------- PLOT STYLING ---------------
        # Remove spines for a clean look
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)

        # Remove ticks
        ax.tick_params(left=False, bottom=False)

        # Add light grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.3)

        # Adjust x-axis limits for better spacing
        ax.set_xlim(-0.8, 0.15)  # Allows enough room on the left for feature names and scores

        # Remove axis labels for a clean look
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Adjust layout for better spacing
        plt.tight_layout()

        # Display plot in Streamlit
        st.pyplot(fig)
        plt.close(fig)
        # --------------- SHAP EXPLANATION END ---------------

# Numerical Feature Analysis
def numerical_feature_analysis(df, target_col, numeric_columns):
    st.subheader("Numerical Feature Analysis")
    # numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

    
    for feature in numeric_columns:
        if feature != target_col:
            stats = df.groupby(target_col)[feature].agg(['mean', 'min', 'max', 'median']).reset_index()

            # Apply heatmap styling
            styled_stats = stats.style.background_gradient(cmap='YlGnBu', subset=['mean', 'min', 'max', 'median']).format("{:.2f}")

            st.write(f"**{feature}**")
            st.dataframe(styled_stats)

def binary_feature_analysis(df, target_col, binary_columns, feature_importances):
    st.subheader("Binary Feature Analysis")

    for feature in binary_columns:
        if feature != target_col:
            st.markdown(f"### **{feature}**")

            # ✅ NA Ratio Calculation
            na_ratio = df[feature].isna().mean() * 100

            # ✅ Overall Distribution (f and t percentages)
            overall_counts = df[feature].value_counts(normalize=True).sort_index() * 100
            overall_counts = overall_counts.rename(index={0: 'f', 1: 't'})

            # ✅ Distribution by Target Classes (good/bad)
            dist_by_class = df.groupby(target_col)[feature].value_counts(normalize=True).unstack().fillna(0) * 100
            dist_by_class = dist_by_class.rename(columns={0: 'f', 1: 't'}, index={0: 'bad', 1: 'good'})

            # ✅ Add "total" row for overall dataset
            dist_by_class.loc['total'] = overall_counts

            # ✅ Get Feature Importance (normalize to percentage)
            feature_importance = feature_importances.get(feature, 0) * 100  # Assuming importance is between 0 and 1
            importance_angle = (feature_importance / 100) * 360

            # ✅ Donut Chart for Feature Importance
            fig, ax = plt.subplots(figsize=(1, 1))
            wedges, _ = ax.pie(
                [feature_importance, 100 - feature_importance],
                colors=['#4A90E2', '#E0E0E0'],
                startangle=90,
                counterclock=False,
                wedgeprops={'width': 0.3, 'edgecolor': 'white'}
            )
            ax.set_aspect('equal')
            plt.axis('off')  # Hide axis

            # ✅ Display Layout
            col1, col2, col3 = st.columns([1, 2, 3])
            with col1:
                st.pyplot(fig)  # Display Feature Importance Donut Chart
            with col2:
                st.write(f"**NA Ratio:** {na_ratio:.2f}%")  # Display NA Ratio as text
            with col3:
                st.write("**Distribution:**")
                st.table(dist_by_class.style.format("{:.2f}%"))

def categorical_feature_analysis(df, target_col, feature_importances, categorical_features):
    # ✅ Safety Check for Missing Argument
    if categorical_features is None:
        categorical_features = []
    st.subheader("Categorical Feature Analysis")
    st.write(f"Detected Categorical Features: {categorical_features}")

    for feature in categorical_features:
        counts = df.groupby([feature, target_col]).size().unstack(fill_value=0)

        if not counts.empty:
            total_records = len(df)  # Total number of records in the dataset

            # ✅ Calculate Good % (proportion within 'good' class)
            if 1 in counts.columns:
                good_total = df[target_col].value_counts().get(1, 0)  # Total 'good' records
                counts['Good %'] = (counts[1] / good_total) * 100 if good_total else 0
            else:
                counts['Good %'] = 0

            # ✅ Calculate Bad % (proportion within 'bad' class)
            if 0 in counts.columns:
                bad_total = df[target_col].value_counts().get(0, 0)  # Total 'bad' records
                counts['Bad %'] = (counts[0] / bad_total) * 100 if bad_total else 0
            else:
                counts['Bad %'] = 0

            # ✅ Correct Total % (percentage of total dataset)
            counts['Total %'] = (counts[[0, 1]].sum(axis=1) / total_records) * 100

            # Distribute Importance Proportionally Based on Total %
            feature_importance = feature_importances.get(feature, 0) * 100

            # 🚀 NEW: Importance Based on Class Separation (Good % - Bad %)
            separation = abs(counts['Good %'] - counts['Bad %'])

            # Normalize the separation and distribute feature importance
            if separation.sum() > 0:
                counts['Importance'] = (separation / separation.sum()) * feature_importance
            else:
                counts['Importance'] = 0  # Handle edge cases with no separation

            total_importance = counts['Importance'].sum()

            if total_importance > 0:
                counts['Importance'] = (counts['Importance'] / total_importance) * 100
            else:
                counts['Importance'] = 0  # Handle edge cases with no separation


            # Sort by Importance
            counts = counts[['Good %', 'Bad %', 'Total %', 'Importance']].sort_values(by='Importance', ascending=False)


            # Apply heatmap styling
            styled_counts = counts.style \
                .background_gradient(cmap='coolwarm', subset=['Good %', 'Bad %', 'Total %']) \
                .background_gradient(cmap='YlOrRd', subset=['Importance']) \
                .format("{:.2f}%")

            st.write(f"**{feature}**")
            st.dataframe(styled_counts)
        else:
            st.warning(f"No data available for feature: {feature}")
            
# Function for anomaly detection
def detect_anomalies(X):
    st.subheader("Anomaly Detection")

    # Select numeric columns for anomaly detection
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    X_numeric = X[numeric_columns]

    # Check if numeric columns exist
    if X_numeric.empty:
        st.write("No numeric columns available for anomaly detection.")
        return

    # Fixed contamination level for anomaly detection
    contamination = 0.05  # Default 5% contamination level
    # st.write(f"Using a contamination level of: {contamination}")

    # Anomaly detection using Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    X['Anomaly_Flag'] = iso_forest.fit_predict(X_numeric)
    X['Anomaly_Flag'] = X['Anomaly_Flag'].apply(lambda x: 1 if x == -1 else 0)

    # Display anomalies
    anomalies = X[X['Anomaly_Flag'] == 1]

    # st.write("### Anomalies Detected")
    if not anomalies.empty:
        st.write(f"Number of anomalies detected: {len(anomalies)}")
        st.dataframe(anomalies)
    else:
        st.write("No anomalies detected.")

    # Plot distribution of anomaly flags
    st.write("### Anomaly Flag Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Anomaly_Flag', data=X, ax=ax)
    ax.set_title(f'Anomaly Flag Distribution (Contamination = {contamination})')
    ax.set_xlabel('Flag (1: Anomaly, 0: Normal)')
    ax.set_ylabel('Count')

    # Add counts on the bars
    flag_counts = X['Anomaly_Flag'].value_counts()
    for i, count in enumerate(flag_counts):
        ax.text(i, count + 5, f'{count}', ha='center', va='bottom', fontsize=12, color='black')

    st.pyplot(fig)

def data_processing(X, is_training=True):
    dropped_features = []  # Initialize dropped_features list to return at the end

    # Drop features based on certain conditions (train and test phases)
    if is_training:
        # Training mode: Clean the data and save dropped features
        st.write(f"Number of features before cleaning: {X.shape[1]}")

        # Remove columns with too many missing values (e.g., more than 50%)
        missing_threshold = 0.5
        columns_with_missing_data = X.columns[X.isnull().mean() >= missing_threshold]
        X = X.loc[:, X.isnull().mean() < missing_threshold]

        # Remove columns with only one unique value
        columns_with_one_value = X.columns[X.nunique() <= 1]
        X = X.loc[:, X.nunique() > 1]

                # 🚀 Remove alphanumeric ID-like columns
        def detect_alphanumeric_ids(df):
            alphanumeric_id_cols = []
            for col in df.select_dtypes(include=['object']).columns:
                # Check if more than 80% of the values are alphanumeric (e.g., IDs like 'BORR0001')
                alphanumeric_ratio = df[col].apply(lambda x: bool(re.match(r'^[A-Za-z0-9]+$', str(x)))).mean()
                if alphanumeric_ratio > 0.8 and df[col].nunique() / len(df) > 0.9:
                    alphanumeric_id_cols.append(col)
            return alphanumeric_id_cols

        # Detect and remove alphanumeric ID columns
        alphanumeric_ids = detect_alphanumeric_ids(X)
        X = X.drop(columns=alphanumeric_ids, errors='ignore')

        # ✅ Handling Comma-Separated Numeric Values
        def clean_comma_separated_numbers(df):
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].str.contains(',', na=False).any():  # Detect commas
                    try:
                        df[col] = df[col].str.replace(',', '', regex=False).astype(float)
                        st.write(f"Cleaned numeric column: {col}")
                    except ValueError:
                        pass  # Skip if conversion fails (non-numeric content)
            return df

        X = clean_comma_separated_numbers(X)

        # Features dropped
        dropped_features = list(columns_with_missing_data) + list(columns_with_one_value) + alphanumeric_ids

        # Count of features after data cleaning
        st.write(f"Number of features after cleaning: {X.shape[1]}")

        # Print the names of the dropped features
        if dropped_features:
            st.write("Dropped features:")
            st.write(dropped_features)

        # Save the dropped features for future use
        with open("dropped_features.pkl", "wb") as f:
            pickle.dump(dropped_features, f)
    else:
        # Prediction mode: Load dropped features and apply them
        with open("dropped_features.pkl", "rb") as f:
            dropped_features = pickle.load(f)

        # Drop the same features as in training
        X = X.drop(columns=dropped_features, errors="ignore")

    # Identify column types
    binary_columns = [col for col in X.columns if set(X[col].dropna().unique()).issubset({0, 1, 't', 'f'})]
    numeric_columns = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    # categorical_columns = [col for col in X.columns if 1 <= X[col].nunique() <= 50 and col not in binary_columns]
    # ✅ Only Consider Object-Type Columns as Categorical
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # numeric_columns = [col for col in numeric_columns if col not in categorical_columns]
    
    print("categorical_columns", categorical_columns)

    # ✅ Exclude binary columns from categorical features
    categorical_columns = [col for col in categorical_columns if col not in binary_columns]

    # Handle binary columns
    for col in binary_columns:
        X[col] = X[col].replace([float('inf'), float('-inf')], 0).fillna(0)
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

    # Encoding categorical features (same for both training and testing)
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        unique_values = X[col].astype(str).unique()
        le.fit(unique_values)
        X[col] = le.transform(X[col].astype(str))
        label_encoders[col] = le

    # Save label encoders only during training phase
    if is_training:
        with open("label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)
    else:
        # For testing, load the label encoders and apply them
        with open("label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)

        # Apply label encoding on testing data
        for col, le in label_encoders.items():
            if col in X.columns:
                # Handle unseen labels during testing (map to 'Unknown')
                unseen_labels = ~X[col].isin(le.classes_)
                X.loc[unseen_labels, col] = 'Unknown'  # Map unseen labels to 'Unknown'

                 #Replace 'Unknown' with the most frequent label or any valid class
                most_frequent_label = le.classes_[0]  # You can adjust this if you want a different label
                X[col] = X[col].replace('Unknown', most_frequent_label)

                # Transform the labels for testing
                X[col] = le.transform(X[col].astype(str))

    return X, dropped_features, categorical_columns, numeric_columns, binary_columns

def perform_eda(X):
    # Get numeric and categorical columns
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Plot bar graphs for numeric columns
    plot_bar_graphs(numeric_columns, X)

    # Plot pie charts for categorical columns
    plot_pie_charts(categorical_columns, X)

    # Plot correlation heatmaps
    plot_correlation_heatmaps(X, numeric_columns)

# Bar Graphs for Numeric Columns
def plot_bar_graphs(numeric_columns, X):
    num_cols = len(numeric_columns)
    rows = math.ceil(num_cols / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
    axes = axes.flatten()
    
    for idx, column in enumerate(numeric_columns):
        sns.histplot(X[column], kde=True, bins=20, color='skyblue', ax=axes[idx])
        axes[idx].set_title(f"Distribution of {column}")
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Count")
    
    for idx in range(num_cols, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    st.pyplot(fig)

# Pie Charts for Categorical Columns
def plot_pie_charts(categorical_columns, X):
    if categorical_columns.empty:  # Check if the Index is empty
        print("No categorical columns to plot.")
        return
    
    # Calculate the number of rows needed for the pie charts
    rows = (len(categorical_columns) + 3) // 4  # Ensure at least one row if there are columns
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
    
    # if rows == 1:
    #     axes = [axes]  # Ensure axes is iterable even if there's just one row
    
    # # Flatten the axes array to easily iterate over it
    # axes = axes.flatten()
    if isinstance(axes, np.ndarray):
     axes = axes.flatten()  # Works if axes is already an array
    else:
        axes = np.array([axes])  # Convert single Axes object to NumPy array

    for i, col in enumerate(categorical_columns):
        value_counts = X[col].value_counts()
        axes[i].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f'Distribution of {col}')

    plt.tight_layout()
    plt.show()

# Correlation Heatmaps
def plot_correlation_heatmaps(X, numeric_columns):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    # Heatmap for numeric features only
    sns.heatmap(X[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap for Numeric Features")
    
    # Heatmap for all features
    # sns.heatmap(X.corr(), annot=True, cmap='coolwarm', ax=axes[1])
    # axes[1].set_title("Correlation Heatmap for All Features")

    plt.tight_layout()
    st.pyplot(fig)

# Function to show data imbalance
def show_data_imbalance(y):
    st.subheader("Class Distribution in Target Column")
    target_counts = y.value_counts()
    target_percentage = y.value_counts(normalize=True) * 100

    st.write("### Target Counts")
    st.write(target_counts)
    st.write("### Target Percentages")
    st.write(target_percentage)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        target_counts,
        labels=[f"Class {i} - {pct:.2f}%" for i, pct in zip(target_counts.index, target_percentage)],
        autopct='%1.1f%%',
        colors=['skyblue', 'salmon'],
        startangle=140,
        wedgeprops={'edgecolor': 'black'},
        textprops={'fontsize': 8}  # Reduced label font size
    )
    ax.set_title("Class Distribution", fontsize=1)
    st.pyplot(fig)

# Function to calculate and plot feature importance
def plot_feature_importance(X, y):
    # Feature Importance - Random Forest
    rf_feature_importance = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_feature_importance.fit(X, y)
    feature_importances = rf_feature_importance.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("feature_importance:", importance_df)
    importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
    
    # Plot Feature Importance Pie Chart
    fig = px.pie(
        importance_df,
        values='Percentage',
        names='Feature',
        title='Feature Importance',
        hover_data={'Percentage': ':.4f'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_traces(
        hovertemplate='<b>Feature:</b> %{label}<br><b>Percentage:</b> %{value:.4f}%<extra></extra>',
        textinfo='percent+label',
        textfont=dict(size=12),
        textposition='inside'
    )

    fig.update_layout(
        title_x=0.44,
        font=dict(size=14),
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(
            x=1,
            y=0.5,
            traceorder='normal',
            font=dict(size=12)
        )
    )
    
    # Display the plot
    return fig



# Function to show correlation heatmap
def show_correlation_heatmap(X):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', ax=ax)
    plt.title("Correlation Heatmap for Features")
    st.pyplot(fig)

# Function for clustering (Optional)
def show_clustering(X):
    from sklearn.cluster import KMeans
    from kneed import KneeLocator
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE

    st.write("### Dynamic K-Means Clustering")

    # Step 1: Calculate WCSS for a range of cluster numbers
    wcss = []
    max_clusters = 10  # You can adjust this based on your dataset

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)  # Replace X with your dataset
        wcss.append(kmeans.inertia_)

    # Step 2: Use KneeLocator to find the optimal number of clusters
    knee_locator = KneeLocator(range(1, max_clusters + 1), wcss, curve="convex", direction="decreasing")
    optimal_k = knee_locator.knee
    st.write(f"Optimal number of clusters: {optimal_k}")

    # Step 3: Visualize the Elbow Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--', label='WCSS')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal Clusters: {optimal_k}')
    plt.title('Elbow Method using Knee Locator')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.legend()
    st.pyplot(plt)

    # Step 4: Apply KMeans clustering to the combined data with optimal_k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    X['Cluster_Label'] = kmeans.fit_predict(X)

    # Step 5: Plot distribution of clusters
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster_Label', hue='Cluster_Label', data=X, palette='viridis', legend=False)
    plt.title('Cluster Distribution')
    plt.xlabel('Cluster Label')
    plt.ylabel('Count')
    st.pyplot(plt)

    # Step 6: Scatter plot for clusters using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_components = tsne.fit_transform(X.drop('Cluster_Label', axis=1))  # Drop 'Cluster_Label' for t-SNE

    # Create a new DataFrame with t-SNE components and cluster labels
    tsne_df = pd.DataFrame(tsne_components, columns=['TSNE1', 'TSNE2'])
    tsne_df['Cluster_Label'] = X['Cluster_Label']

    # Step 7: Plotting the 2D scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster_Label', data=tsne_df, palette='viridis')
    plt.title('t-SNE of Features with Cluster Labels')
    st.pyplot(plt)

    # Step 8: Plot distribution of anomalies within clusters (optional, if 'Anomaly_Flag' exists)
    if 'Anomaly_Flag' in X.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Anomaly_Flag', hue='Cluster_Label', data=X, palette='viridis')
        plt.title('Anomaly Flag Distribution within Clusters')
        plt.xlabel('Anomaly Flag (0: Normal, 1: Anomaly)')
        plt.ylabel('Count')
        st.pyplot(plt)

    # Step 9: Cluster summary and averages
    cluster_summary = X.groupby('Cluster_Label').describe()
    cluster_averages = X.groupby('Cluster_Label').mean()

    # Print cluster summary and averages for interpretation
    st.write("### Cluster Summary (Descriptive Statistics):")
    st.write(cluster_summary)

    st.write("\n### Cluster Feature Averages:")
    st.write(cluster_averages)

    # Return cluster averages for further processing
    return cluster_averages

# Allow nested async loops
nest_asyncio.apply()

import asyncio
import openai
from openai import AsyncOpenAI
import time

# Define the async cluster interpretation function
async def interpret_clusters(cluster_averages, openai_api_key, max_retries=3):
    """Interprets K-Means clusters using OpenAI GPT model with retry logic."""
    
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

    # Retry mechanism
    for attempt in range(max_retries):
        try:
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

            # Extract and return the interpretation
            return response.choices[0].message.content
        
        except openai.APIConnectionError as e:
            print(f"OpenAI API connection error (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retrying
            else:
                print("Failed after multiple retries.")
                return None  # Return None if all retries fail
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None  # Return None if an unknown error occurs

# Proper async execution
def run_async_interpretation(cluster_averages, openai_api_key):
    """Runs the async function safely in a synchronous environment."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(interpret_clusters(cluster_averages, openai_api_key))


    # # Optional: Save cluster summary and feature averages to Excel
    # cluster_summary.to_excel("Cluster_Summary_Detailed.xlsx", sheet_name="Summary_Detailed")
    # cluster_averages.to_excel("Cluster_Feature_Averages.xlsx", sheet_name="Feature_Averages")


# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC-AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend(loc="lower right")
    st.pyplot(plt)


# ✅ Function to calculate and plot the KS Curve
def plot_ks_curve(y_test, test_predictions_proba):
    # Sort predicted probabilities and corresponding actual classes
    sorted_indices = np.argsort(test_predictions_proba)
    y_sorted = y_test.iloc[sorted_indices].values
    proba_sorted = test_predictions_proba[sorted_indices]

    # ✅ Calculate Cumulative TPR and FPR
    cumulative_tpr = np.cumsum(y_sorted) / np.sum(y_sorted)           # TPR for positives
    cumulative_fpr = np.cumsum(1 - y_sorted) / np.sum(1 - y_sorted)   # FPR for negatives

    # ✅ Calculate KS Statistic
    ks_diff = np.abs(cumulative_tpr - cumulative_fpr)
    ks_stat = np.max(ks_diff)                     # Maximum KS difference
    ks_index = np.argmax(ks_diff)                # Index where KS is maximum
    ks_threshold = proba_sorted[ks_index]        # Corresponding predicted probability

    # ✅ Plot the KS Curve
    plt.figure(figsize=(8, 6))
    plt.plot(proba_sorted, cumulative_tpr, color="blue", lw=2, label="Cumulative TPR")
    plt.plot(proba_sorted, cumulative_fpr, color="gray", lw=2, label="Cumulative FPR")

   # ✅ Draw Green Line ONLY BETWEEN TPR and FPR
    plt.plot(
        [ks_threshold, ks_threshold],                             # x-coordinates (vertical line)
        [cumulative_fpr[ks_index], cumulative_tpr[ks_index]],     # y-coordinates (between FPR and TPR)
        color="green", linestyle="--", lw=2,
        label=f"KS = {ks_stat:.3f} at {ks_threshold:.3f}"
    )
    # ✅ Add Markers for TPR and FPR at KS Point
    plt.scatter(ks_threshold, cumulative_tpr[ks_index], color="orange", label=f"TPR = {cumulative_tpr[ks_index]:.2f}")
    plt.scatter(ks_threshold, cumulative_fpr[ks_index], color="purple", label=f"FPR = {cumulative_fpr[ks_index]:.2f}")

    # Add Labels & Legends
    plt.xlabel("Predicted Probability")
    plt.ylabel("Cumulative Proportion")
    plt.title("K-S Curve")
    plt.legend(loc="lower right")

    # Display Plot
    st.pyplot(plt)

    return ks_stat

# Function to plot KDE distribution with "Good" and "Bad" labels
def plot_kde_distribution(y_test, test_predictions_proba):
    # Creating DataFrame
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Score': test_predictions_proba
    })

    # Mapping labels to "Good" and "Bad"
    results_df['Class'] = results_df['Actual'].map({1: 'Good', 0: 'Bad'})

    # Plotting KDE
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=results_df,
        x='Score',
        hue='Class',
        fill=False,
        common_norm=False,
        palette={'Good': 'blue', 'Bad': 'gray'},
        linewidth=2
    )

    # Customizing the plot
    plt.title("Density Distribution by Classes", fontsize=16)
    plt.xlabel("Score", fontsize=12)
    plt.xlim(0, None)  # Ensures x-axis starts from 0
    plt.ylabel("Density", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Classes', labels=['Good', 'Bad'], loc='upper right', fontsize=10)

    # Display the plot
    plt.tight_layout()
    st.pyplot(plt)




import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ✅ Profit Forecast Plot Function
def plot_profit_forecast(test_predictions_proba, y_test, benefit_correct=100, cost_mistake=500):
    st.subheader("📈 Profit Forecast with Adjustable Parameters")

    # ✅ Initialize Session State to Track Button Click
    if "update_profit" not in st.session_state:
        st.session_state.update_profit = False  # Default state is False

    # 🎯 Form for Benefit & Cost Inputs
    with st.form(key="profit_form"):
        benefit_input = st.number_input(
            "💰 Benefit for Correct Predictions (True Positives)",
            min_value=0, max_value=10000, value=benefit_correct, step=50
        )

        cost_input = st.number_input(
            "⚠️ Cost for Incorrect Predictions (False Positives)",
            min_value=0, max_value=10000, value=cost_mistake, step=50
        )

        # 🚀 Update Button
        submit_button = st.form_submit_button(label="🔄 Update Profit Graph")

        # ✅ Update Session State when Button is Clicked
        if submit_button:
            st.session_state.update_profit = True
            st.session_state.benefit_correct = benefit_input
            st.session_state.cost_mistake = cost_input

    # 🚀 Plot Graph if Button Clicked OR for Default View
    if st.session_state.update_profit or "benefit_correct" not in st.session_state:
        benefit_correct = st.session_state.get("benefit_correct", benefit_correct)
        cost_mistake = st.session_state.get("cost_mistake", cost_mistake)

        thresholds = np.linspace(0, 1, 101)
        profit_values, correctly_approved_values, incorrectly_approved_values = [], [], []

        for threshold in thresholds:
            predictions = (test_predictions_proba >= threshold).astype(int)
            TP = np.sum((predictions == 1) & (y_test == 1))  # True Positives
            FP = np.sum((predictions == 1) & (y_test == 0))  # False Positives

            correctly_approved = TP * benefit_correct
            incorrectly_approved = FP * cost_mistake
            profit = correctly_approved - incorrectly_approved

            profit_values.append(profit)
            correctly_approved_values.append(correctly_approved)
            incorrectly_approved_values.append(incorrectly_approved)

        max_profit_index = np.argmax(profit_values)
        best_threshold = thresholds[max_profit_index]
        max_profit = profit_values[max_profit_index]

        # 📊 Profit Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, profit_values, label="Profit", color='blue', lw=2)
        ax.axhline(y=0, color='yellow', linestyle='--', linewidth=1, label="Break-even Line")

        ax.axvline(
            x=best_threshold, color='purple', linestyle='--', linewidth=2,
            label=f"Max Profit @ {best_threshold:.2f} (${max_profit:,.0f})"
        )

        ax.plot(thresholds, correctly_approved_values, label="Correctly Approved", color='green', linestyle='--', lw=2)
        ax.plot(thresholds, incorrectly_approved_values, label="Incorrectly Approved", color='red', linestyle='--', lw=2)

        ax.set_xlim(0, 1)
        y_min, y_max = min(profit_values), max(profit_values)
        y_buffer = max(20000, 0.5 * abs(y_max))
        ax.set_ylim(y_min, y_max + y_buffer) if y_min < 0 else ax.set_ylim(0, y_max + y_buffer)

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Amount ($)")
        ax.set_title("Profit Forecast and Approved Amounts")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        # Key Metrics
        st.markdown(f"**Max Profit:** ${max_profit:,.0f} at Threshold = {best_threshold:.2f}")
        st.markdown(f"**Benefit per Correct Prediction:** ${benefit_correct}")
        st.markdown(f"**Cost per Incorrect Prediction:** ${cost_mistake}")

# Example Placeholder (Replace with your actual data)
# plot_profit_forecast(test_predictions_proba, y_test)




# Function to evaluate the model
def evaluate_model(best_model, X_train, X_test, y_train, y_test):
    # Make predictions on the training set to calculate the threshold
    train_predictions_proba = best_model.predict_proba(X_train)[:, 1]

    # Make predictions on the test set
    test_predictions_proba = best_model.predict_proba(X_test)[:, 1]
     # Streamlit input for threshold
    # st.sidebar.write("## Adjust Threshold")
    # threshold = st.sidebar.slider("Select threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    threshold = 0.5
    predictions = (test_predictions_proba >= threshold).astype(int)

    # Generate ROC-AUC and Gini Coefficient
    fpr, tpr, _ = roc_curve(y_test, test_predictions_proba)
    roc_auc = auc(fpr, tpr)
    gini_coeff = 2 * roc_auc - 1

    # Calculate Classification Metrics
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    correct_predictions = (predictions == y_test).sum()
    incorrect_predictions = (predictions != y_test).sum()

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)

    # Store Metrics in a DataFrame
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
            0.5,
            gini_coeff,
            roc_auc,
            precision,
            recall,
            f1,
            correct_predictions,
            incorrect_predictions
        ]
    })
# Display the metrics table
    st.write("### Classification Metrics Table")
    st.write(metrics_table)

    # Display the confusion matrix in a nice format
    conf_matrix_table = pd.DataFrame(
        conf_matrix,
        index=["Actual Class 0", "Actual Class 1"],
        columns=["Predicted Class 0", "Predicted Class 1"]
    )
    st.write("### Confusion Matrix")
    st.write(conf_matrix_table)

    
    # # Display ROC curve
    plot_roc_curve(fpr, tpr, roc_auc)

    #KDE Plot for Density Distrivution of classes

    plot_kde_distribution(y_test, test_predictions_proba)

    # KS Statistic Calculation and Plot
    # ks_stat = max(tpr - fpr)
    # plot_ks_curve(fpr, tpr, ks_stat)
    ks_stat = plot_ks_curve(y_test, test_predictions_proba)

    # Plot Custom Metric vs Threshold
    plot_profit_forecast(test_predictions_proba, y_test)

    # Display classification report
    # st.write("### Classification Report")
    # st.text(classification_report(y_test, predictions))
    

    return metrics_table

if __name__ == "__main__":
    main()
