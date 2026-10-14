"""
UMAP Embedding Dimensionality Reduction and Interactive Visualization Engine.
Generates 2D/3D manifold projections with cluster evaluation metrics.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score
from config import CATEGORY_COLORS, PATH_UMAP_EMBEDDINGS

class UMAPEmbeddingVisualizer:
    """Computes UMAP projections and renders interactive Plotly scatter figures."""

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.reducer_2d = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=2,
            metric=self.metric,
            random_state=self.random_state
        )
        self.reducer_3d = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=3,
            metric=self.metric,
            random_state=self.random_state
        )

    def fit_transform(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        texts: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Projects embeddings to 2D and 3D coordinates and computes clustering metrics."""
        coords_2d = self.reducer_2d.fit_transform(embeddings)
        coords_3d = self.reducer_3d.fit_transform(embeddings)

        df = pd.DataFrame({
            "x_2d": coords_2d[:, 0],
            "y_2d": coords_2d[:, 1],
            "x_3d": coords_3d[:, 0],
            "y_3d": coords_3d[:, 1],
            "z_3d": coords_3d[:, 2],
            "label": labels,
            "text_snippet": [t[:120] + "..." if len(t) > 120 else t for t in texts]
        })

        # Calculate cluster separation metrics
        # Map labels to integers for sklearn metrics
        unique_labels = sorted(list(set(labels)))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        int_labels = np.array([label_to_int[l] for l in labels])

        sil_score = float(silhouette_score(coords_2d, int_labels)) if len(unique_labels) > 1 else 0.0
        db_score = float(davies_bouldin_score(coords_2d, int_labels)) if len(unique_labels) > 1 else 0.0

        metrics = {
            "silhouette_score_2d": sil_score,
            "davies_bouldin_index_2d": db_score,
            "n_samples": len(embeddings),
            "original_dim": embeddings.shape[1]
        }

        # Cache coordinates
        df.to_parquet(PATH_UMAP_EMBEDDINGS)
        return df, metrics

    @staticmethod
    def create_2d_scatter(df: pd.DataFrame) -> go.Figure:
        """Creates interactive 2D UMAP scatter plot."""
        fig = px.scatter(
            df,
            x="x_2d",
            y="y_2d",
            color="label",
            color_discrete_map=CATEGORY_COLORS,
            hover_data=["text_snippet"],
            labels={"x_2d": "UMAP Component 1", "y_2d": "UMAP Component 2", "label": "Category"}
        )
        fig.update_layout(
            paper_bgcolor="#151D2E",
            plot_bgcolor="#0B1020",
            font=dict(color="#F1F5F9", family="Inter, sans-serif"),
            legend=dict(
                title=dict(text="Annotation Category", font=dict(color="#F1F5F9")),
                font=dict(color="#94A3B8"),
                bgcolor="rgba(21, 29, 46, 0.8)",
                bordercolor="#293548",
                borderwidth=1
            ),
            xaxis=dict(gridcolor="#293548", zerolinecolor="#293548"),
            yaxis=dict(gridcolor="#293548", zerolinecolor="#293548"),
            margin=dict(l=40, r=40, t=40, b=40),
            height=600
        )
        fig.update_traces(marker=dict(size=6, opacity=0.85, line=dict(width=0.5, color="#111827")))
        return fig

    @staticmethod
    def create_3d_scatter(df: pd.DataFrame) -> go.Figure:
        """Creates interactive 3D UMAP scatter plot."""
        fig = px.scatter_3d(
            df,
            x="x_3d",
            y="y_3d",
            z="z_3d",
            color="label",
            color_discrete_map=CATEGORY_COLORS,
            hover_data=["text_snippet"],
            labels={"x_3d": "UMAP 1", "y_3d": "UMAP 2", "z_3d": "UMAP 3", "label": "Category"}
        )
        fig.update_layout(
            paper_bgcolor="#151D2E",
            font=dict(color="#F1F5F9", family="Inter, sans-serif"),
            scene=dict(
                bgcolor="#0B1020",
                xaxis=dict(backgroundcolor="#0B1020", gridcolor="#293548", color="#94A3B8"),
                yaxis=dict(backgroundcolor="#0B1020", gridcolor="#293548", color="#94A3B8"),
                zaxis=dict(backgroundcolor="#0B1020", gridcolor="#293548", color="#94A3B8")
            ),
            legend=dict(
                title=dict(text="Annotation Category", font=dict(color="#F1F5F9")),
                font=dict(color="#94A3B8"),
                bgcolor="rgba(21, 29, 46, 0.8)",
                bordercolor="#293548",
                borderwidth=1
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            height=650
        )
        fig.update_traces(marker=dict(size=4, opacity=0.85))
        return fig

