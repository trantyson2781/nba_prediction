"""Plotly figure builders for the notebooks (kept out of the pipeline code)."""
from __future__ import annotations

import pandas as pd


def confusion_matrix_fig(y_true, pred, title="Confusion Matrix"):
    import plotly.figure_factory as ff
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, pred)
    z = cm[::-1]
    fig = ff.create_annotated_heatmap(
        z, x=["Predicted Loss", "Predicted Win"],
        y=["Actual Win", "Actual Loss"], colorscale="Viridis")
    fig.update_layout(title=f"<b>{title}</b>", xaxis_title="Model Prediction",
                      yaxis_title="Actual Outcome", template="plotly_dark")
    return fig


def feature_importance_fig(model, feature_names, title="Feature Importance"):
    import plotly.express as px
    fi = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    fi = fi.sort_values("Importance", ascending=True)
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 title=f"<b>{title}</b>", template="plotly_dark")
    return fig


def calibration_fig(cal_table: pd.DataFrame, title="Reliability"):
    import plotly.graph_objects as go
    t = cal_table.dropna(subset=["avg_pred"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=t["avg_pred"], y=t["win_rate"], mode="lines+markers",
                             name="model"))
    fig.update_layout(title=f"<b>{title}</b>", xaxis_title="Predicted P(home win)",
                      yaxis_title="Observed win rate", template="plotly_dark")
    return fig
