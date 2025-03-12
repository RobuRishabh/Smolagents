import streamlit as st
import pandas as pd
from smolagents import CodeAgent, LiteLLMModel
from tools.preprocess import PreprocessDataTool
from tools.train_evaluate import TrainAndEvaluateTool
from tools.target_suitability import TargetSuitabilityTool
from dotenv import load_dotenv
import os
import traceback
import logging

# Load .env variables
load_dotenv()

# Streamlit UI Setup
st.set_page_config(page_title="SmolAgent ML Agent", layout="wide")
st.title("🤖 SmolAgent-Powered ML Assistant")

# Sidebar: Upload CSV
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.sidebar.success("✅ File uploaded!")

    # Show preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))

    # Select columns and task
    target_column = st.sidebar.selectbox("🎯 Select Target Variable", df.columns)
    feature_columns = st.sidebar.multiselect("🔍 Select Feature Columns", [col for col in df.columns if col != target_column])
    task_type = st.sidebar.radio("📈 Task Type", ["Regression", "Classification"])

    model_options = {
        "Regression": ["Linear Regression", "Random Forest Regressor"],
        "Classification": ["Logistic Regression", "Random Forest Classifier"]
    }
    model_type = st.sidebar.selectbox("🤖 Model Type", model_options[task_type])

    if st.sidebar.button("🚀 Run SmolAgent"):
        if not feature_columns:
            st.error("❌ Please select at least one feature column.")
        else:
            with st.spinner("Running agent..."):
                try:
                    # Dataset preview and schema
                    df_preview = df.head(10).to_string()
                    df_schema = str(df.dtypes)

                    user_prompt = f"""
                    [INST]
                    You are an AI assistant tasked with performing a complete machine learning workflow. Follow these steps precisely using the provided dataset `df` and tools. Return results as specified.

                    1. Target Suitability Analysis
                    - Use the TargetSuitabilityTool to analyze the target column '{target_column}' in the dataset `df`.
                    - Determine if '{target_column}' is appropriate for a {task_type} task:
                        * For Classification: Check if the target has discrete classes (e.g., categorical or binary values) and count unique values
                        * For Regression: Verify if the target is continuous (numeric) and check its distribution
                    - If unsuitable (e.g., too few unique values for classification or non-numeric for regression), raise an error with a clear explanation
                    - Output: Print a brief summary of the suitability check (e.g., "Target is suitable: 3 unique classes found" or "Target is suitable: continuous variable")

                    2. Data Preprocessing
                    - If the target is suitable, use PreprocessDataTool to prepare the dataset:
                        * Features: Use only the selected columns {feature_columns}
                        * Target: Use '{target_column}'
                        * Handle missing values: Fill numeric columns with median, categorical with mode
                        * Encode categorical features: Use one-hot encoding for nominal data, label encoding for ordinal
                        * Scale numeric features: Apply StandardScaler
                        * Split data: 80% train, 20% test with random_state=42
                    - Output: Return preprocessed X_train, X_test, y_train, y_test

                    3. Model Training
                    - Use TrainAndEvaluateTool to train a {model_type} model:
                        * For Regression:
                        - Linear Regression: Use sklearn.linear_model.LinearRegression
                        - Random Forest Regressor: Use sklearn.ensemble.RandomForestRegressor with n_estimators=100, random_state=42
                        * For Classification:
                        - Logistic Regression: Use sklearn.linear_model.LogisticRegression with max_iter=1000
                        - Random Forest Classifier: Use sklearn.ensemble.RandomForestClassifier with n_estimators=100, random_state=42
                    - Train the model using the preprocessed X_train and y_train

                    4. Model Evaluation and Visualization
                    - Evaluate the trained model using TrainAndEvaluateTool:
                        * For Regression:
                        - Compute Mean Squared Error (MSE) and R² Score on test set
                        - Create a Plotly residual plot: scatter plot of (y_test - predictions) vs. predictions
                        * For Classification:
                        - Compute accuracy score
                        - Generate classification report (precision, recall, f1-score)
                        - Create a Plotly confusion matrix
                    - Output: Return a tuple (metrics_dict, plotly_fig) where:
                        * metrics_dict contains:
                        - Regression: {{'mse': value, 'r2': value}}
                        - Classification: {{'accuracy': value, 'report': classification_report_string}}
                        * plotly_fig is the Plotly figure object (residual plot or confusion matrix)

                    Extra Context:
                    - Dataset preview (first 10 rows):
                    {df_preview}
                    - Data types of columns:
                    {df_schema}

                    Error Handling:
                    - If any step fails, raise an exception with a detailed message explaining what went wrong
                    - Validate inputs (e.g., ensure feature columns exist in df)

                    Return Format:
                    - Return (metrics_dict, plotly_fig) as a tuple
                    - Ensure metrics_dict is a dictionary and plotly_fig is a valid Plotly figure object

                    [/INST]
                    """

                    # LiteLLMModel in Completion Mode for llama2
                    model = LiteLLMModel(
                        model_id="ollama_chat/gemma3:4b",
                        api_base="http://127.0.0.1:11434",
                        num_ctx=8000,
                    )

                    tools = [
                        TargetSuitabilityTool(),
                        PreprocessDataTool(),
                        TrainAndEvaluateTool()
                    ]

                    agent = CodeAgent(
                        model=model,
                        tools=tools,
                        additional_authorized_imports=[
                            "pandas", "numpy", "sklearn.model_selection", "sklearn.linear_model",
                            "sklearn.ensemble", "sklearn.metrics", "plotly.graph_objects",
                            "plotly.express", "plotly.figure_factory", "sklearn.preprocessing"
                        ],
                        max_steps=6
                    )

                    # Run agent with instruct-style single prompt
                    response = agent.run(user_prompt, additional_args={"df": df})

                    st.success("✅ Agent completed execution!")

                    if isinstance(response, tuple) and len(response) == 2:
                        metrics, fig = response
                        st.subheader("📈 Model Performance")
                        if task_type == "Regression":
                            st.write(f"🔹 Mean Squared Error (MSE): **{metrics['mse']:.4f}**")
                            st.write(f"🔹 R² Score: **{metrics['r2']:.4f}**")
                        else:
                            st.write(f"🔹 Accuracy: **{metrics['accuracy']:.4f}**")
                            st.text("Classification Report:")
                            st.text(metrics['report'])
                        if fig:
                            st.plotly_chart(fig)
                    else:
                        st.subheader("📄 Raw Agent Output")
                        st.code(str(response))

                except Exception as e:
                    st.error(f"❌ Agent execution failed: {str(e)}")
                    st.error(f"Traceback:\n{traceback.format_exc()}")

else:
    st.info("📂 Upload a dataset to get started.")
