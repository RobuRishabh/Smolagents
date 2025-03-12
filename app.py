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

# Load environment variables
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

    # Model options
    model_options = {
        "Regression": ["Linear Regression", "Random Forest Regressor"],
        "Classification": ["Logistic Regression", "Random Forest Classifier"]
    }
    model_type = st.sidebar.selectbox("🤖 Model Type", model_options[task_type])

    # Run Agent Button
    if st.sidebar.button("🚀 Run SmolAgent"):
        if not feature_columns:
            st.error("❌ Please select at least one feature column.")
        else:
            with st.spinner("Running agent..."):
                try:
                    # Dataset preview and schema
                    df_preview = df.head(10).to_string()
                    df_schema = str(df.dtypes)

                    # Create task
                    task = f"""
                    1. Check if the target '{target_column}' in dataset `df` is suitable for a {task_type} task using TargetSuitabilityTool.
                    2. If suitable, preprocess the dataset using features {feature_columns} and target '{target_column}' with PreprocessDataTool.
                    3. Train a {model_type} model using TrainAndEvaluateTool.
                    4. Evaluate the model and return metrics and a Plotly figure (residual plot or confusion matrix).
                    """

                    # Setup model and agent
                    model = LiteLLMModel(
                        model_id="ollama/llama2:latest",
                        api_base="http://127.0.0.1:11434",
                        mode="completion",
                        # custom_role_conversions={
                        #     "system": "system",
                        #     "user": "user",
                        #     "assistant": "assistant"
                        # }
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
                            "plotly.express", "plotly.figure_factory"
                        ],
                        max_steps=10
                    )

                    # Log messages for debugging
                    messages = [
                        {"role": "system", "content": "You are an expert assistant who can solve tasks using code."},
                        {"role": "user", "content": task}
                    ]
                    logging.debug("Messages being passed to the model:")
                    for message in messages:
                        logging.debug(f"Role: {message['role']}, Content: {message['content']}")

                    # Run the agent
                    response = agent.run(task, additional_args={"df": df})

                    # Handle response
                    st.success("✅ Agent completed execution!")

                    if isinstance(response, tuple) and len(response) == 2:
                        metrics, fig = response
                        st.subheader("📈 Model Performance")
                        if task_type == "Regression":
                            st.write(f"🔹 Mean Squared Error (MSE): **{metrics['mse']:.4f}**")
                            st.write(f"🔹 R² Score: **{metrics['r2']:.4f}**")
                            if fig:
                                st.plotly_chart(fig)
                        else:
                            st.write(f"🔹 Accuracy: **{metrics['accuracy']:.4f}**")
                            st.text("Classification Report:")
                            st.text(metrics['report'])
                            if fig:
                                st.plotly_chart(fig)
                    else:
                        st.text("Raw agent response:")
                        st.code(str(response))

                except Exception as e:
                    st.error(f"❌ Agent execution failed: {str(e)}")
                    st.error(f"Traceback: {traceback.format_exc()}")  # Print full traceback
else:
    st.info("📂 Upload a dataset to get started.")