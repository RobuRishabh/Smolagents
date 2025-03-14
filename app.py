# app.py
from smolagents import CodeAgent, LiteLLMModel
from tools.ml_pipeline import MLPipelineTool
import yaml
from dotenv import load_dotenv
import os
import streamlit as st
import traceback
import logging
from streamlit_UI import run_ui, display_results

# Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

load_dotenv()

with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)
ML_WORKFLOW_PROMPT = prompts["ml_workflow"]

def run_agent(user_inputs):
    if not user_inputs:
        return None

    df = user_inputs["df"]
    target_column = user_inputs["target_column"]
    feature_columns = user_inputs["feature_columns"]
    task_type = user_inputs["task_type"]
    model_type = user_inputs["model_type"].replace(' ', '')  

    with st.spinner("Running agent..."):
        try:
            # logger.info("Starting agent execution")
            # df_preview = df.head(10).to_string()
            # df_schema = str(df.dtypes)

            user_prompt = ML_WORKFLOW_PROMPT.format(
                df=df,
                feature_columns=feature_columns,
                target_column=target_column,
                task_type=task_type,
                model_type=model_type 
            )

            model = LiteLLMModel(
                model_id="ollama_chat/qwen2.5-coder:3b",
                api_base="http://127.0.0.1:11434",
                num_ctx=8000,
            )

            tools = [
                MLPipelineTool()
            ]
            agent = CodeAgent(
                model=model,
                tools=tools,
                additional_authorized_imports=[
                    "pandas", "numpy", "sklearn.model_selection", "sklearn.linear_model",
                    "sklearn.ensemble", "sklearn.metrics", "plotly.graph_objects",
                    "plotly.express", "plotly.figure_factory", "sklearn.preprocessing",
                    "os", "sklearn", "typing", "yaml", "dotenv", "streamlit", "traceback",
                    "logging", "matplotlib.pyplot"
                ],
                max_steps=10 # Change it according to your needs
            )

            additional_args = {
                "df": df,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "task_type": task_type,
                "model_type": model_type  # Use preprocessed model_type
            }

            # logger.info("Running agent in non-streaming mode")
            response = agent.run(user_prompt, additional_args=additional_args, stream=False)

            import ast
            final_result = None
            for line in str(response).splitlines():
                if "result =" in line:
                    result_code = line.split("result =")[1].strip()
                    final_result = ast.literal_eval(result_code)
                    # logger.info("Extracted final result from response")
                    break

            if not final_result:
                # logger.warning("No final result found in response")
                metrics, fig = agent.tools[0].forward(df, feature_columns, target_column, task_type, model_type)
                code_str = None
                final_result = (metrics, fig, code_str)
                # logger.info("Fallback: Executed tools directly")

            st.success("✅ Agent completed execution!")
            return final_result

        except Exception as e:
            # logger.error(f"Agent execution failed: {str(e)}")
            st.error(f"❌ Agent execution failed: {str(e)}")
            st.error(f"Traceback:\n{traceback.format_exc()}")
            return None

if __name__ == "__main__":
    user_inputs, _ = run_ui()
    if user_inputs:
        final_result = run_agent(user_inputs)
        if final_result:
            display_results(user_inputs["task_type"], final_result, None)