"""
Transformer Multi-Head Self-Attention Visualization Engine.
Extracts attention tensors and generates interactive Plotly attention heatmaps.
"""

from typing import Dict, Any, List, Tuple
import torch
import numpy as np
import plotly.graph_objects as go

def extract_attention_weights(model, tokenizer, text: str, max_length: int = 64, device: str = "cpu") -> Tuple[List[str], np.ndarray]:
    """
    Extracts attention weights across all layers and heads for the input text.
    Returns (tokens, attention_matrix) of shape (num_layers, num_heads, seq_len, seq_len).
    """
    model.eval()
    model.to(device)
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    tokens = [t.replace("Ġ", " ").replace("##", "") for t in tokenizer.convert_ids_to_tokens(input_ids[0])]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        # attentions is a tuple of length num_layers, each (batch_size, num_heads, seq_len, seq_len)
        attentions = outputs.attentions
        # Stack to (num_layers, num_heads, seq_len, seq_len)
        attentions_stack = torch.stack(attentions).squeeze(1).cpu().numpy()

    return tokens, attentions_stack

def create_attention_heatmap_figure(
    tokens: List[str],
    attention_matrix: np.ndarray,
    layer_idx: int = 0,
    head_idx: int = 0
) -> go.Figure:
    """
    Renders an interactive Plotly heatmap for a specific transformer layer and attention head.
    """
    num_layers, num_heads, seq_len, _ = attention_matrix.shape
    layer_idx = min(max(0, layer_idx), num_layers - 1)
    head_idx = min(max(0, head_idx), num_heads - 1)

    attn = attention_matrix[layer_idx, head_idx]

    fig = go.Figure(data=go.Heatmap(
        z=attn,
        x=tokens,
        y=tokens,
        colorscale="Viridis",
        colorbar=dict(title=dict(text="Weight", font=dict(color="#F1F5F9")), tickfont=dict(color="#94A3B8")),
        hoverongaps=False
    ))

    fig.update_layout(
        title=dict(
            text=f"Self-Attention Weights • Layer {layer_idx + 1}/{num_layers} • Head {head_idx + 1}/{num_heads}",
            font=dict(color="#F1F5F9", size=14)
        ),
        paper_bgcolor="#151D2E",
        plot_bgcolor="#0B1020",
        font=dict(color="#F1F5F9", family="Inter, sans-serif"),
        xaxis=dict(tickangle=-45, tickfont=dict(color="#94A3B8", size=10), gridcolor="#293548"),
        yaxis=dict(tickfont=dict(color="#94A3B8", size=10), gridcolor="#293548", autorange="reversed"),
        margin=dict(l=60, r=40, t=50, b=80),
        height=520
    )
    return fig

