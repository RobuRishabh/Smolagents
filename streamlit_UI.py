# streamlit_UI.py
import streamlit as st
import pandas as pd

def run_ui():
    st.set_page_config(page_title="SmolAgent ML Agent", layout="wide")
    st.title("🤖 SmolAgent-Powered ML Assistant")

    st.sidebar.header("📂 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, low_memory=False)
        st.sidebar.success("✅ File uploaded!")

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(10))

        target_column = st.sidebar.selectbox("🎯 Select Target Variable", df.columns)
        feature_columns = st.sidebar.multiselect("🔍 Select Feature Columns", 
                                                [col for col in df.columns if col != target_column])
        task_type = st.sidebar.radio("📈 Task Type", ["Regression", "Classification"])

        model_options = {
            "Regression": ["Linear Regression", "Random Forest Regressor"],
            "Classification": ["Logistic Regression", "Random Forest Classifier"]
        }
        model_type = st.sidebar.selectbox("🤖 Model Type", model_options[task_type])

        run_button = st.sidebar.button("🚀 Run SmolAgent")

        if run_button:
            if not feature_columns:
                st.error("❌ Please select at least one feature column.")
                return None, None
            else:
                user_inputs = {
                    "df": df,
                    "target_column": target_column,
                    "feature_columns": feature_columns,
                    "task_type": task_type,
                    "model_type": model_type
                }
                return user_inputs, None

        return None, None

    else:
        st.info("📂 Upload a dataset to get started.")
        return None, None

def display_results(task_type, final_result):
    if final_result and isinstance(final_result, tuple) and len(final_result) == 3:
        metrics, fig, code_str = final_result

        st.subheader("📈 Model Performance")
        if task_type == "Regression":
            st.write(f"🔹 Mean Squared Error (MSE): **{metrics.get('mse', 'N/A'):.4f}**")
            st.write(f"🔹 R² Score: **{metrics.get('r2', 'N/A'):.4f}**")
        else:
            st.write(f"🔹 Accuracy: **{metrics.get('accuracy', 'N/A'):.4f}**")
            if "report" in metrics:
                st.text("Classification Report:")
                if isinstance(metrics["report"], str):
                    st.text(metrics["report"])
                elif isinstance(metrics["report"], dict):
                    report_df = pd.DataFrame(metrics["report"]).T.round(3)
                    st.dataframe(report_df)

        if fig:
            st.subheader("📉 Visualization")
            st.plotly_chart(fig)
        else:
            st.warning("⚠️ No visualization generated.")

        st.subheader("💻 Generated Python Code")
        st.code(code_str)
    else:
        st.error("❌ No valid result generated.")