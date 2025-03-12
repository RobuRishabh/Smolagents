from smolagents.tools import Tool
import pandas as pd

class TargetSuitabilityTool(Tool):
    name = "target_suitability"
    description = "Check if the target column is suitable for Regression or Classification tasks."
    inputs = {
        "df": {"type": "object", "description": "The dataset as a pandas DataFrame."},
        "target_column": {"type": "string", "description": "The name of the target column."},
        "task_type": {"type": "string", "description": "The ML task type: 'Regression' or 'Classification'."}
    }
    output_type = "array"  # Returns: (is_suitable: bool, warning_message: str)

    def forward(self, df: pd.DataFrame, target_column: str, task_type: str):
        """
        Check if the target column is suitable for the given ML task.

        Returns:
            tuple: (bool, warning_message)
        """
        y = df[target_column]

        if task_type == "Regression":
            if y.dtype == "object" or y.nunique() < 10:
                return False, (
                    "⚠️ The selected target variable is categorical or has too few unique values. "
                    "Consider switching to a classification task."
                )
        elif task_type == "Classification":
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
                return False, (
                    "⚠️ The selected target variable appears continuous. "
                    "Consider switching to a regression task."
                )

        return True, ""
