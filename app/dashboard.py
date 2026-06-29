import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'pipeline_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')


# --- REBUILD MODEL ARCHITECTURE ---
class AcousticCNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_features)
            output_dim = self.encoder(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.position_head = nn.Linear(128, 1)
        self.severity_head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.fc(x)
        return self.position_head(x).squeeze(), self.severity_head(x)


# --- SYSTEM INITIALIZATION ---
print(">>> [SYSTEM] Booting Magnora Acoustic Twin SCADA...")

df = pd.read_csv(DATA_PATH)
X_raw = df.drop(columns=['sensor_id', 'leak_position', 'leak_size']).values
Y_size = df['leak_size'].values - df['leak_size'].min()
num_classes = len(np.unique(Y_size))
max_dist = float(df['leak_position'].max() + 20)

in_features = X_raw.shape[1]
scaler = joblib.load(os.path.join(MODEL_DIR, 'acoustic_scaler.pkl'))
X_scaled = scaler.transform(X_raw)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AcousticCNN(in_features=in_features, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'acoustic_dnn.pth'), map_location=device, weights_only=True))
model.eval()

print(">>> [SUCCESS] Neural Weights and Scalers Loaded.")


# --- INITIALIZE STATIC 3D PIPELINE ---
def create_base_3d_figure():
    fig = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 8)
    for t in theta:
        fig.add_trace(go.Scatter3d(
            x=np.linspace(0, max_dist, 50), y=np.cos(t) * np.ones(50) * 2, z=np.sin(t) * np.ones(50) * 2,
            mode='lines', line=dict(color='rgba(0, 255, 204, 0.3)', width=3), hoverinfo='skip'
        ))
    # لوله مرکزی
    fig.add_trace(go.Scatter3d(
        x=np.linspace(0, max_dist, 50), y=np.zeros(50), z=np.zeros(50),
        mode='lines', line=dict(color='rgba(0, 255, 204, 0.1)', width=15), hoverinfo='skip'
    ))
    # نقطه نشت اولیه (پنهان)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(color='rgba(0,0,0,0)', size=1, symbol='circle', opacity=0.9, line=dict(width=0)),
        name="Anomaly Point"
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(title='Pipeline Distance (m)', showgrid=False, zeroline=False, showbackground=False,
                       color='#00ffcc'),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            camera=dict(eye=dict(x=0.5, y=-1.5, z=0.5))
        ),
        margin=dict(l=0, r=0, b=0, t=0), showlegend=False,
        uirevision='constant'  # <--- این خط زوم و چرخش کاربر را حفظ می‌کند
    )
    return fig


# --- INITIALIZE STATIC WAVEFORM ---
def create_base_wave_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.linspace(0, 2, 300), y=np.zeros(300), mode='lines',
        line=dict(color='#00ffcc', width=1.5), fill='tozeroy', fillcolor="rgba(0, 255, 204, 0.2)"
    ))
    fig.update_layout(
        title=dict(text="LIVE ACOUSTIC FREQUENCY SIGNATURE", font=dict(color='#888', size=12)),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, b=10, t=30),
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-4, 4]), showlegend=False,
        uirevision='constant'  # <--- این خط زوم موج را حفظ می‌کند
    )
    return fig


# --- DASHBOARD SPA (FRONTEND) ---
app = Dash(__name__, title="Magnora Pipeline Twin")

app.layout = html.Div(style={
    'backgroundColor': '#050505', 'color': '#00ffcc', 'fontFamily': 'Orbitron, monospace',
    'height': '100vh', 'padding': '20px', 'overflow': 'hidden', 'boxSizing': 'border-box'
}, children=[
    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '20px'},
             children=[
                 html.H1("MAGNORA // OMNI-CORE: PIPELINE DIGITAL TWIN",
                         style={'textShadow': '0 0 15px rgba(0, 255, 204, 0.8)', 'letterSpacing': '2px',
                                'margin': '0 20px 0 0'}),
                 html.Button("⏸ PAUSE", id="play-pause-btn", n_clicks=0, style={
                     'backgroundColor': 'rgba(0, 255, 204, 0.1)', 'color': '#00ffcc', 'border': '1px solid #00ffcc',
                     'padding': '10px 20px', 'fontSize': '16px', 'fontWeight': 'bold', 'cursor': 'pointer',
                     'borderRadius': '5px', 'boxShadow': '0 0 10px rgba(0, 255, 204, 0.3)',
                     'transition': 'all 0.3s ease'
                 })
             ]),

    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}, children=[
        html.Div(id='kpi-status', style={
            'flex': '1', 'margin': '0 10px', 'padding': '20px',
            'border': '1px solid #00ffcc', 'borderRadius': '8px',
            'textAlign': 'center', 'fontSize': '22px', 'fontWeight': 'bold',
            'boxShadow': 'inset 0 0 20px rgba(0,255,204,0.1)'
        }),
        html.Div(id='kpi-pos', style={
            'flex': '1', 'margin': '0 10px', 'padding': '20px',
            'border': '1px solid #ff0055', 'borderRadius': '8px',
            'textAlign': 'center', 'fontSize': '26px', 'fontWeight': 'bold',
            'color': '#ff0055', 'textShadow': '0 0 10px #ff0055',
            'boxShadow': 'inset 0 0 20px rgba(255,0,85,0.1)'
        }),
        html.Div(id='kpi-severity', style={
            'flex': '1', 'margin': '0 10px', 'padding': '20px',
            'border': '1px solid #ffea00', 'borderRadius': '8px',
            'textAlign': 'center', 'fontSize': '22px', 'fontWeight': 'bold',
            'color': '#ffea00',
            'boxShadow': 'inset 0 0 20px rgba(255,234,0,0.1)'
        })
    ]),

    html.Div(style={'display': 'flex', 'flexDirection': 'column', 'height': 'calc(100vh - 180px)'}, children=[
        dcc.Graph(id='live-3d-pipeline', figure=create_base_3d_figure(), style={'flex': '6', 'minHeight': '0'}),
        dcc.Graph(id='live-acoustic-wave', figure=create_base_wave_figure(), style={'flex': '4', 'minHeight': '0'})
    ]),

    dcc.Interval(id='stream-interval', interval=1500, n_intervals=0),
    # زمان آپدیت را به 1.5 ثانیه تغییر دادیم برای پایداری بیشتر
    dcc.Store(id='data-index', data=0)  # نگهدارنده ایندکس داده‌ها برای حالت Pause
])


@app.callback(
    Output('stream-interval', 'disabled'),
    Output('play-pause-btn', 'children'),
    Output('play-pause-btn', 'style'),
    Input('play-pause-btn', 'n_clicks'),
    State('play-pause-btn', 'style')
)
def toggle_play_pause(n_clicks, current_style):
    if n_clicks % 2 == 1:
        # حالت Pause
        new_style = current_style.copy()
        new_style.update({'color': '#ffea00', 'borderColor': '#ffea00', 'boxShadow': '0 0 10px rgba(255, 234, 0, 0.3)'})
        return True, "▶ RESUME", new_style
    else:
        # حالت Play
        new_style = current_style.copy()
        new_style.update({'color': '#00ffcc', 'borderColor': '#00ffcc', 'boxShadow': '0 0 10px rgba(0, 255, 204, 0.3)'})
        return False, "⏸ PAUSE", new_style


@app.callback(
    Output('data-index', 'data'),
    Input('stream-interval', 'n_intervals'),
    State('data-index', 'data')
)
def update_index(n_intervals, current_index):
    return (current_index + 1) % len(X_scaled)


@app.callback(
    Output('kpi-status', 'children'), Output('kpi-pos', 'children'), Output('kpi-severity', 'children'),
    Output('live-3d-pipeline', 'figure'), Output('live-acoustic-wave', 'figure'),
    Input('data-index', 'data'),
    State('live-3d-pipeline', 'figure'), State('live-acoustic-wave', 'figure')
)
def update_dashboard_data(idx, fig3d, fig_wave):
    x_tensor = torch.tensor([X_scaled[idx]], dtype=torch.float32).to(device)

    with torch.no_grad():
        pos_pred, size_logits = model(x_tensor)
        pos_val = pos_pred.item()
        size_val = torch.argmax(size_logits, dim=1).item()

    status_text = "🟢 TELEMETRY: ACTIVE"
    pos_text = f"📍 LEAK DETECTED: {pos_val:.1f} M"
    severity_text = f"⚠️ SEVERITY LEVEL: {size_val}"

    # --- UPDATE 3D PIPELINE (Only the marker) ---
    marker_color = '#ff0055' if size_val > 0 else '#00ffcc'
    marker_size = 15 + (size_val * 8)

    # Trace 9 is the anomaly marker based on create_base_3d_figure
    fig3d['data'][9]['x'] = [pos_val]
    fig3d['data'][9]['marker']['color'] = marker_color
    fig3d['data'][9]['marker']['size'] = marker_size
    fig3d['data'][9]['marker']['line']['width'] = 2 if size_val > 0 else 0

    # --- UPDATE ACOUSTIC WAVEFORM ---
    wave_x = np.linspace(0, 2, 300)
    base_wave = np.sin(2 * np.pi * 15 * wave_x)
    noise_level = 0.1 + (size_val * 0.6)
    noise = np.random.normal(0, noise_level, 300)
    wave_y = base_wave + noise

    fig_wave['data'][0]['y'] = wave_y
    fig_wave['data'][0]['line']['color'] = marker_color
    fill_color = f"rgba({255 if size_val > 0 else 0}, {0 if size_val > 0 else 255}, {85 if size_val > 0 else 204}, 0.2)"
    fig_wave['data'][0]['fillcolor'] = fill_color

    return status_text, pos_text, severity_text, fig3d, fig_wave


if __name__ == '__main__':
    print(">>> [READY] Starting SCADA Web Server on http://0.0.0.0:8050")
    app.run(host='0.0.0.0', port=8050, debug=False)