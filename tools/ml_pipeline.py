# tools/ml_pipeline.py
from smolagents.tools import Tool
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from typing import List, Tuple, Dict, Any

# MLPipelineTool for executing the ML pipeline
class MLPipelineTool(Tool):
    name = "ml_pipeline"
    description = "Executes a complete ML pipeline and returns metrics and a plot."
    inputs = {
        "df": {"type": "object", "description": "Pandas DataFrame with the dataset."},
        "feature_columns": {"type": "array", "description": "List of feature column names."},
        "target_column": {"type": "string", "description": "Target column name."},
        "task_type": {"type": "string", "description": "Task type: 'Regression' or 'Classification'."},
        "model_type": {"type": "string", "description": "Model type: 'Linear Regression', 'Random Forest Regressor', 'Logistic Regression', or 'Random Forest Classifier'."}
    }
    output_type = "object"  # Returns (metrics_dict, plotly_fig)

    def forward(self, df: pd.DataFrame, feature_columns: List[str], target_column: str, 
                task_type: str, model_type: str) -> Tuple[Dict[str, Any], Any]:
        """Execute the ML pipeline."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input 'df' must be a non-empty pandas DataFrame.")
        if not all(col in df.columns for col in feature_columns + [target_column]):
            raise ValueError("Some columns not found in DataFrame.")

        # Target suitability
        target_series = df[target_column]
        if task_type == "Regression" and not pd.api.types.is_numeric_dtype(target_series):
            raise ValueError("Target must be numeric for regression.")
        elif task_type == "Classification" and target_series.nunique() >= 20:
            raise ValueError("Too many unique values for classification.")

        # Preprocess
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        le = LabelEncoder()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col].astype(str))
        if y.dtype == 'object':
            y = le.fit_transform(y.astype(str))

        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = pd.Series(y).apply(pd.to_numeric, errors='coerce').fillna(0)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train and evaluate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_dict = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100)
        }
        model = model_dict[model_type]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "Regression":
            metrics = {"mse": mean_squared_error(y_test, y_pred), "r2": r2_score(y_test, y_pred)}
            residuals = y_test - y_pred
            y_test = np.array(y_test).flatten()
            residuals = np.array(residuals).flatten()
            fig = px.scatter(x=y_test, y=residuals, title="Residual Analysis",
                             labels={'x': 'Actual Values', 'y': 'Residuals'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
        else:
            metrics = {"accuracy": accuracy_score(y_test, y_pred), "report": classification_report(y_test, y_pred)}
            cm = confusion_matrix(y_test, y_pred)
            fig = ff.create_annotated_heatmap(z=cm, x=list(range(cm.shape[1])), y=list(range(cm.shape[0])),
                                              annotation_text=cm.astype(str), colorscale='Blues')
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")

        return metrics, fig
