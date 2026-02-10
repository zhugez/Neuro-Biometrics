"""
Visualization Script for Comprehensive Evaluation Results
Reads output_v2.json and creates interactive Plotly charts
"""
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Load results
LOG_FILE = 'output_v2.json'
try:
    with open(LOG_FILE, 'r') as f:
        data = json.load(f)
    results = data['results']
except FileNotFoundError:
    print(f"Error: {LOG_FILE} not found. Please run main.py first.")
    exit()

# Colors for models
MODELS = list(set([r['model_name'] for r in results]))
COLORS = px.colors.qualitative.Plotly
COLOR_MAP = {model: COLORS[i % len(COLORS)] for i, model in enumerate(MODELS)}

NOISE_TYPES = list(set([r['noise_type'] for r in results]))

def parse_stats(stat_str):
    """Parse 'mean ± std' string"""
    if '±' in stat_str:
        mean, std = stat_str.split(' ± ')
        return float(mean), float(std)
    return float(stat_str), 0.0

# ==========================================
# 1. Summary Bar Charts with Error Bars
# ==========================================
def plot_summary_metric(metric_key, title, y_title, range_y=None):
    fig = go.Figure()
    
    for model in MODELS:
        means, stds, noises = [], [], []
        for noise in NOISE_TYPES:
            res = next((r for r in results if r['noise_type'] == noise and r['model_name'] == model), None)
            if res:
                m, s = parse_stats(res['stats'][metric_key])
                means.append(m)
                stds.append(s)
                noises.append(noise)
        
        fig.add_trace(go.Bar(
            name=model,
            x=noises,
            y=means,
            error_y=dict(type='data', array=stds),
            marker_color=COLOR_MAP[model],
            text=[f'{m:.2f}' for m in means],
            textposition='auto'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Noise Type',
        yaxis_title=y_title,
        barmode='group',
        template='plotly_white'
    )
    if range_y:
        fig.update_yaxes(range=range_y)
    
    return fig

fig_p1 = plot_summary_metric('p@1', 'Test P@1 (Mean ± Std)', 'P@1 Score', [0, 1.05])
fig_p1.write_html('summary_p1.html')

fig_sisnr = plot_summary_metric('si_snr', 'Test SI-SNR (Mean ± Std)', 'SI-SNR (dB)')
fig_sisnr.write_html('summary_sisnr.html')

fig_auroc = plot_summary_metric('auroc', 'Novelty Detection AUROC', 'AUROC', [0.5, 1.05])
fig_auroc.write_html('summary_novelty_auroc.html')

print("✓ Saved Summary Bar Charts")

# ==========================================
# 2. Robustness Curves (Metric vs SNR)
# ==========================================
fig_rob = make_subplots(rows=1, cols=3, subplot_titles=[n.upper() for n in NOISE_TYPES],
                        shared_yaxes=True)

snr_levels = [0, 5, 10, 20]

for i, noise in enumerate(NOISE_TYPES):
    for model in MODELS:
        res = next((r for r in results if r['noise_type'] == noise and r['model_name'] == model), None)
        if res and 'robustness' in res['best_run']['test']:
            rob_data = res['best_run']['test']['robustness']
            # Sort keys just in case
            y_vals = []
            valid_snrs = []
            for snr in snr_levels:
                s_key = str(snr)
                if s_key in rob_data:
                    y_vals.append(rob_data[s_key]['p@1'])
                    valid_snrs.append(snr)
            
            fig_rob.add_trace(go.Scatter(
                x=valid_snrs, y=y_vals,
                mode='lines+markers',
                name=model if i == 0 else None,
                line=dict(color=COLOR_MAP[model]),
                showlegend=(i == 0)
            ), row=1, col=i+1)

fig_rob.update_layout(title="Robustness Analysis: P@1 vs Input SNR", template="plotly_white")
fig_rob.update_xaxes(title_text="Input SNR (dB)")
fig_rob.update_yaxes(title_text="P@1 Score", range=[0, 1.05])
fig_rob.write_html("robustness_snr.html")
print("✓ Saved Robustness Curves")

# ==========================================
# 3. CMC Curves (Rank-1 to Rank-20)
# ==========================================
fig_cmc = go.Figure()
# Plotting average CMC across noise types for each model? 
# Or just pick one noise type? Let's do Mean over noise types for simplicity or specific noise.
# Let's plot for "gaussian" noise as representative.

target_noise = "gaussian"
if target_noise in NOISE_TYPES:
    for model in MODELS:
        res = next((r for r in results if r['noise_type'] == target_noise and r['model_name'] == model), None)
        if res:
            cmc_vals = res['best_run']['test']['cmc']
            ranks = list(range(1, len(cmc_vals) + 1))
            fig_cmc.add_trace(go.Scatter(
                x=ranks, y=cmc_vals,
                mode='lines',
                name=f"{model} ({res['noise_type']})",
                line=dict(color=COLOR_MAP[model])
            ))
            
fig_cmc.update_layout(title=f"CMC Curve ({target_noise})", 
                      xaxis_title="Rank", yaxis_title="Identification Rate",
                      template="plotly_white", xaxis=dict(range=[1, 20]))
fig_cmc.write_html("cmc_curve.html")
print("✓ Saved CMC Curve")

# ==========================================
# 4. ROC & DET Curves (Verification)
# ==========================================
fig_roc = go.Figure()
fig_det = go.Figure()

for model in MODELS:
    # Aggregating across noise types? No, specific noise is better.
    res = next((r for r in results if r['noise_type'] == 'gaussian' and r['model_name'] == model), None)
    if res:
        roc_data = res['best_run']['test']['roc']
        det_data = res['best_run']['test']['det']
        
        # ROC
        fig_roc.add_trace(go.Scatter(
            x=roc_data['fpr'], y=roc_data['tpr'],
            mode='lines', name=model,
            line=dict(color=COLOR_MAP[model])
        ))
        
        # DET (Log-Log)
        fig_det.add_trace(go.Scatter(
            x=det_data['fpr'], y=det_data['fnr'],
            mode='lines', name=model,
            line=dict(color=COLOR_MAP[model])
        ))

fig_roc.update_layout(title="ROC Curve (Gaussian)", xaxis_title="FAR", yaxis_title="TAR", template="plotly_white")
fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
fig_roc.write_html("roc_curve.html")

fig_det.update_layout(
    title="DET Curve (Gaussian)", 
    xaxis_title="FAR", yaxis_title="FRR", 
    template="plotly_white",
    xaxis_type="log", yaxis_type="log"
)
fig_det.write_html("det_curve.html")

print("✓ Saved ROC/DET Curves")
print("All visualization complete.")
