# =============================================================================
# ZERO RESTAN OPTIMIZER — Streamlit Dashboard
# PTPN IV PalmCo
# =============================================================================
# Cara menjalankan:
#   1. pip install streamlit pandas numpy scikit-learn xgboost openpyxl
#   2. Letakkan file model_artifacts.pkl dan data_kak_amel.xlsx di folder yang sama
#   3. streamlit run zero_restan_app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zero Restan Optimizer",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: #0F172A;
      color: #E2E8F0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background-color: #1E293B;
      border-right: 1px solid #334155;
  }
  section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
  section[data-testid="stSidebar"] .stNumberInput label,
  section[data-testid="stSidebar"] .stSelectbox label { color: #94A3B8 !important; font-size: 0.82rem !important; }

  /* Cards */
  .card {
      background: #1E293B;
      border: 1px solid #334155;
      border-radius: 14px;
      padding: 1.2rem 1.5rem;
      margin-bottom: 0.6rem;
  }
  .card-label {
      font-size: 0.72rem;
      font-weight: 600;
      color: #64748B;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 0.3rem;
  }
  .card-value {
      font-family: 'DM Mono', monospace;
      font-size: 2rem;
      font-weight: 500;
      color: #F1F5F9;
      line-height: 1.1;
  }
  .card-sub {
      font-size: 0.78rem;
      color: #64748B;
      margin-top: 0.25rem;
  }

  /* Warning badges */
  .badge-green  { background:#064E3B; color:#34D399; border:1px solid #065F46; border-radius:8px; padding:0.5rem 1rem; font-weight:700; font-size:1rem; }
  .badge-yellow { background:#78350F; color:#FCD34D; border:1px solid #92400E; border-radius:8px; padding:0.5rem 1rem; font-weight:700; font-size:1rem; }
  .badge-red    { background:#7F1D1D; color:#FCA5A5; border:1px solid #991B1B; border-radius:8px; padding:0.5rem 1rem; font-weight:700; font-size:1rem; }

  /* Page title */
  .page-title {
      font-size: 1.9rem;
      font-weight: 800;
      color: #F1F5F9;
      letter-spacing: -0.03em;
  }
  .page-sub {
      font-size: 0.85rem;
      color: #64748B;
      margin-top: -0.2rem;
      margin-bottom: 1.2rem;
  }

  /* Section header */
  .sec-header {
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #475569;
      margin-bottom: 0.5rem;
      margin-top: 1rem;
  }

  /* Hide Streamlit default decoration */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────
DATA_FILE      = "data_kak_amel.xlsx"
MODEL_FILE     = "model_artifacts.pkl"
HISTORY_FILE   = "restan_history.json"     # penyimpanan data input harian
RESTAN_WARN_YELLOW = 10_000                # kg
RESTAN_WARN_RED    = 25_000                # kg
FEATURES = [
    'Jumlah Pemanen', '% Langsir', 'Curah Hujan', 'Jumlah Trip/Hari',
    'Jam_Timbang_Fix', 'Hujan_Flag',
    'Produksi_lag1', 'Produksi_lag2', 'Produksi_lag3',
    'Restan_lag1', 'Restan_lag2', 'Restan_lag3',
    'Produksi_roll3', 'Produksi_roll7',
    'Restan_roll3', 'Restan_roll7',
    'Bulan', 'Minggu_ke', 'Hari_dalam_minggu', 'Log_Produksi',
]


# ─────────────────────────────────────────────────────
# HELPERS — Data & Model
# ─────────────────────────────────────────────────────
@st.cache_data
def load_base_data():
    df = pd.read_excel(DATA_FILE)
    df['Waktu'] = pd.to_datetime(df['Waktu'])
    df = df.sort_values('Waktu').reset_index(drop=True)
    # Hapus hari libur
    mask = (df['Produksi'] == 0) & (df['Jumlah Pemanen'] == 0)
    df = df[~mask].reset_index(drop=True)
    # Fix jam
    df['Jam_Timbang_Fix'] = df['Jam Timbang Pertama'].apply(
        lambda x: x + 24 if 0 < x < 3 else x)
    df['Hujan_Flag'] = (df['Curah Hujan'] > 0).astype(int)
    return df


def load_history():
    """Load data input harian dari file JSON lokal."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def history_to_df(history):
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history)
    df['Waktu'] = pd.to_datetime(df['Waktu'])
    df = df.sort_values('Waktu').reset_index(drop=True)
    return df


def build_full_df(df_base, history):
    """Gabung data base + history input harian."""
    if not history:
        return df_base.copy()
    df_hist = history_to_df(history)
    # Rename agar kolom match
    df_hist = df_hist.rename(columns={'jam_timbang': 'Jam_Timbang_Fix',
                                      'curah_hujan': 'Curah Hujan',
                                      'jumlah_pemanen': 'Jumlah Pemanen',
                                      'persen_langsir': '% Langsir',
                                      'jumlah_trip': 'Jumlah Trip/Hari',
                                      'produksi_aktual': 'Produksi',
                                      'restan_aktual': 'Restan'})
    df_hist['Hujan_Flag'] = (df_hist['Curah Hujan'] > 0).astype(int)

    # Kolom yang ada di base tapi tidak di hist → isi NaN
    for col in df_base.columns:
        if col not in df_hist.columns:
            df_hist[col] = np.nan

    df_combined = pd.concat([df_base, df_hist], ignore_index=True)
    df_combined = df_combined.sort_values('Waktu').drop_duplicates('Waktu').reset_index(drop=True)
    return df_combined


def add_lag_features(df):
    df = df.copy().sort_values('Waktu').reset_index(drop=True)
    for lag in [1, 2, 3]:
        df[f'Produksi_lag{lag}'] = df['Produksi'].shift(lag)
        df[f'Restan_lag{lag}']   = df['Restan'].shift(lag)
    for w in [3, 7]:
        df[f'Produksi_roll{w}'] = df['Produksi'].shift(1).rolling(w).mean()
        df[f'Restan_roll{w}']   = df['Restan'].shift(1).rolling(w).mean()
    df['Bulan']             = df['Waktu'].dt.month
    df['Minggu_ke']         = df['Waktu'].dt.isocalendar().week.astype(int)
    df['Hari_dalam_minggu'] = df['Waktu'].dt.dayofweek
    df['Log_Produksi']      = np.log1p(df['Produksi'])
    return df


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    return None


def retrain_model(df_full):
    """Retrain model dengan data terbaru (dipanggil saat ada data baru)."""
    df_feat = add_lag_features(df_full)
    df_model = df_feat[FEATURES + ['Produksi', 'Waktu']].dropna()
    if len(df_model) < 20:
        return None, "Data terlalu sedikit untuk training"

    X = df_model[FEATURES]
    y = df_model['Produksi']

    tscv = TimeSeriesSplit(n_splits=5)

    # Fit ketiga model
    ridge = Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=10))])
    rf    = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    xgb   = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                         random_state=42, verbosity=0)

    models = {'Ridge': ridge, 'Random Forest': rf, 'XGBoost': xgb}
    scores = {}
    for name, m in models.items():
        s = cross_val_score(m, X, y, cv=tscv, scoring='r2')
        scores[name] = s.mean()

    best_name = max(scores, key=scores.get)
    best_model = models[best_name]
    best_model.fit(X, y)

    artifacts = {
        'best_model': best_model,
        'best_model_name': best_name,
        'ridge': ridge.fit(X, y),
        'rf': rf.fit(X, y),
        'xgb': xgb.fit(X, y),
        'features': FEATURES,
        'scores': scores,
        'df_model': df_model,
        'last_trained': datetime.now().isoformat(),
    }
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(artifacts, f)
    # Bust cache
    load_model.clear()

    return artifacts, f"Model {best_name} (R²={scores[best_name]:.3f})"


def build_input_row(tomorrow, input_vals, df_full):
    """Buat satu baris fitur untuk prediksi besok."""
    df_feat = add_lag_features(df_full)
    last_row = df_feat.sort_values('Waktu').iloc[-1]

    row = {
        'Jumlah Pemanen'     : input_vals['jumlah_pemanen'],
        '% Langsir'          : input_vals['persen_langsir'],
        'Curah Hujan'        : input_vals['curah_hujan'],
        'Jumlah Trip/Hari'   : input_vals['jumlah_trip'],
        'Jam_Timbang_Fix'    : input_vals['jam_timbang'],
        'Hujan_Flag'         : int(input_vals['curah_hujan'] > 0),
        'Produksi_lag1'      : last_row['Produksi'],
        'Produksi_lag2'      : last_row.get('Produksi_lag1', np.nan),
        'Produksi_lag3'      : last_row.get('Produksi_lag2', np.nan),
        'Restan_lag1'        : last_row['Restan'],
        'Restan_lag2'        : last_row.get('Restan_lag1', np.nan),
        'Restan_lag3'        : last_row.get('Restan_lag2', np.nan),
        'Produksi_roll3'     : last_row.get('Produksi_roll3', np.nan),
        'Produksi_roll7'     : last_row.get('Produksi_roll7', np.nan),
        'Restan_roll3'       : last_row.get('Restan_roll3', np.nan),
        'Restan_roll7'       : last_row.get('Restan_roll7', np.nan),
        'Bulan'              : tomorrow.month,
        'Minggu_ke'          : tomorrow.isocalendar()[1],
        'Hari_dalam_minggu'  : tomorrow.weekday(),
        'Log_Produksi'       : last_row.get('Log_Produksi', np.nan),
    }
    return pd.DataFrame([row])[FEATURES]


def restan_status(restan_kg):
    if restan_kg <= 0:
        return 'green', '🟢 ZERO RESTAN', 'Semua buah berhasil diangkut!'
    elif restan_kg < RESTAN_WARN_YELLOW:
        return 'green', '🟢 AMAN', f'Restan {restan_kg:,.0f} kg — dalam batas normal'
    elif restan_kg < RESTAN_WARN_RED:
        return 'yellow', '🟡 PERHATIAN', f'Restan {restan_kg:,.0f} kg — pertimbangkan tambah armada'
    else:
        return 'red', '🔴 KRITIS', f'Restan {restan_kg:,.0f} kg — butuh tindakan segera!'


# ─────────────────────────────────────────────────────
# LOAD STATE
# ─────────────────────────────────────────────────────
df_base  = load_base_data()
history  = load_history()
df_full  = build_full_df(df_base, history)
artifacts = load_model()

tomorrow = datetime.today().date() + timedelta(days=1)
today    = datetime.today().date()

# ─────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌴 Zero Restan\n**Optimizer**")
    st.markdown("---")

    st.markdown('<p class="sec-header">📅 Input Harian Baru</p>', unsafe_allow_html=True)
    st.caption(f"Data untuk: **{tomorrow.strftime('%A, %d %b %Y')}**")

    with st.form("input_harian"):
        curah_hujan    = st.number_input("Curah Hujan (mm) — dari BMKG", min_value=0.0, step=0.1, value=0.0)
        jumlah_pemanen = st.number_input("Jumlah Pemanen (orang)", min_value=0, step=1, value=90)
        jumlah_trip    = st.number_input("Jumlah Trip/Hari (rencana)", min_value=0, step=1, value=20)
        persen_langsir = st.number_input("% Langsir (%)", min_value=0.0, max_value=100.0, step=0.1, value=37.0)
        jam_timbang    = st.number_input("Jam Timbang Pertama (0–24)", min_value=0.0, max_value=24.0, step=0.01, value=13.5)

        st.markdown("---")
        st.caption("*Isi setelah hari selesai (aktual):*")
        produksi_aktual = st.number_input("Produksi Aktual (kg)", min_value=0, step=100, value=0)
        restan_aktual   = st.number_input("Restan Aktual (kg)", min_value=0, step=100, value=0)

        submitted = st.form_submit_button("💾 Simpan & Prediksi", use_container_width=True)

    if submitted:
        new_entry = {
            'Waktu'           : str(tomorrow),
            'curah_hujan'     : curah_hujan,
            'jumlah_pemanen'  : jumlah_pemanen,
            'jumlah_trip'     : jumlah_trip,
            'persen_langsir'  : persen_langsir,
            'jam_timbang'     : jam_timbang,
            'produksi_aktual' : produksi_aktual,
            'restan_aktual'   : restan_aktual,
        }
        # Update atau tambah
        existing_dates = [h['Waktu'] for h in history]
        if str(tomorrow) in existing_dates:
            idx = existing_dates.index(str(tomorrow))
            history[idx] = new_entry
        else:
            history.append(new_entry)

        save_history(history)

        # Retrain jika ada data aktual baru
        if produksi_aktual > 0:
            df_full = build_full_df(df_base, history)
            with st.spinner("🔄 Melatih ulang model..."):
                artifacts, msg = retrain_model(df_full)
            st.success(f"✅ Model diperbarui: {msg}")
        else:
            st.success("✅ Data tersimpan")

        st.rerun()

    st.markdown("---")
    if artifacts:
        st.markdown('<p class="sec-header">🤖 Info Model</p>', unsafe_allow_html=True)
        st.caption(f"Model aktif: **{artifacts.get('best_model_name','—')}**")
        if 'scores' in artifacts:
            for k, v in artifacts['scores'].items():
                st.caption(f"• {k}: R²={v:.3f}")
        if 'last_trained' in artifacts:
            st.caption(f"Terakhir dilatih:\n{artifacts['last_trained'][:16]}")
    else:
        st.warning("⚠️ Model belum tersedia.\nUpload model_artifacts.pkl dari Colab.")

    st.markdown("---")
    if st.button("🔄 Retrain Manual", use_container_width=True):
        df_full = build_full_df(df_base, history)
        with st.spinner("Melatih model..."):
            artifacts, msg = retrain_model(df_full)
        st.success(f"✅ {msg}")
        st.rerun()


# ─────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────
st.markdown('<p class="page-title">🌴 Zero Restan Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">PTPN IV PalmCo — Sistem Prediksi & Monitoring Produksi Sawit</p>', unsafe_allow_html=True)

# ── Input values dari form (atau default) ──────────
input_vals = {
    'curah_hujan'    : curah_hujan    if submitted else 0.0,
    'jumlah_pemanen' : jumlah_pemanen if submitted else int(df_base['Jumlah Pemanen'].median()),
    'jumlah_trip'    : jumlah_trip    if submitted else int(df_base['Jumlah Trip/Hari'].median()),
    'persen_langsir' : persen_langsir if submitted else float(df_base['% Langsir'].median()),
    'jam_timbang'    : jam_timbang    if submitted else float(df_base['Jam_Timbang_Fix'].median()),
}


# ─────────────────────────────────────────────────────
# SECTION 1: PREDIKSI BESOK
# ─────────────────────────────────────────────────────
st.markdown('<p class="sec-header">🎯 Prediksi Produksi & Rekomendasi Armada</p>', unsafe_allow_html=True)

if artifacts and artifacts.get('best_model') is not None:
    try:
        df_full_feat = build_full_df(df_base, history)
        X_pred = build_input_row(tomorrow, input_vals, df_full_feat)
        model  = artifacts['best_model']
        pred_produksi = float(model.predict(X_pred)[0])
        pred_produksi = max(0, pred_produksi)

        # Rata-rata tonase/trip historis
        avg_tonase = df_base[df_base['Rata-rata Tonase/Trip'] > 0]['Rata-rata Tonase/Trip'].median()
        target_trip = int(np.ceil(pred_produksi / avg_tonase)) if avg_tonase > 0 else 0
        tersedia_trip = input_vals['jumlah_trip']
        selisih_trip  = tersedia_trip - target_trip

        # Estimasi restan
        est_restan = max(0, (target_trip - tersedia_trip) * avg_tonase)
        status_color, status_label, status_msg = restan_status(est_restan)
        badge_class = f"badge-{status_color}"

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""<div class="card">
                <div class="card-label">Prediksi Produksi</div>
                <div class="card-value">{pred_produksi/1000:.1f}<span style="font-size:1rem;color:#64748B"> ton</span></div>
                <div class="card-sub">{tomorrow.strftime('%d %b %Y')}</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""<div class="card">
                <div class="card-label">Target Trip Dibutuhkan</div>
                <div class="card-value">{target_trip}<span style="font-size:1rem;color:#64748B"> trip</span></div>
                <div class="card-sub">Asumsi {avg_tonase:,.0f} kg/trip</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            selisih_color = "#34D399" if selisih_trip >= 0 else "#FCA5A5"
            selisih_sign  = "+" if selisih_trip >= 0 else ""
            st.markdown(f"""<div class="card">
                <div class="card-label">Selisih Armada</div>
                <div class="card-value" style="color:{selisih_color}">{selisih_sign}{selisih_trip}<span style="font-size:1rem;color:#64748B"> trip</span></div>
                <div class="card-sub">Tersedia: {tersedia_trip} | Butuh: {target_trip}</div>
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown(f"""<div class="card">
                <div class="card-label">Estimasi Restan</div>
                <div class="card-value">{est_restan/1000:.1f}<span style="font-size:1rem;color:#64748B"> ton</span></div>
                <div class="card-sub">Jika armada tidak ditambah</div>
            </div>""", unsafe_allow_html=True)

        # Warning badge
        st.markdown(f'<div class="{badge_class}" style="margin:0.5rem 0 1rem 0;">{status_label} — {status_msg}</div>',
                    unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error prediksi: {e}")
else:
    st.info("⚠️ Model belum dimuat. Jalankan Colab script terlebih dahulu, lalu upload model_artifacts.pkl.")

    # Tampilkan placeholder cards
    col1, col2, col3, col4 = st.columns(4)
    for col, label in zip([col1,col2,col3,col4], ['Prediksi Produksi','Target Trip','Selisih Armada','Est. Restan']):
        with col:
            st.markdown(f"""<div class="card">
                <div class="card-label">{label}</div>
                <div class="card-value" style="color:#334155">—</div>
                <div class="card-sub">Model belum tersedia</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# SECTION 2: ZERO-RESTAN TRACKER (30 hari terakhir)
# ─────────────────────────────────────────────────────
st.markdown('<p class="sec-header">📊 Zero-Restan Tracker — 30 Hari Terakhir</p>', unsafe_allow_html=True)

# Gabung data base + history untuk chart
df_chart = df_full.copy()
df_chart = df_chart.sort_values('Waktu')
df_chart_30 = df_chart[df_chart['Waktu'] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]

if len(df_chart_30) == 0:
    df_chart_30 = df_chart.tail(30)

fig_tracker = go.Figure()

# Area produksi
fig_tracker.add_trace(go.Scatter(
    x=df_chart_30['Waktu'], y=df_chart_30['Produksi']/1000,
    name='Produksi (ton)', mode='lines+markers',
    line=dict(color='#34D399', width=2.5),
    marker=dict(size=5, color='#34D399'),
    fill='tozeroy', fillcolor='rgba(52,211,153,0.08)',
    hovertemplate='<b>%{x|%d %b}</b><br>Produksi: %{y:.1f} ton<extra></extra>'
))

# Garis Restan
restan_col = df_chart_30['Restan'] if 'Restan' in df_chart_30.columns else pd.Series(dtype=float)
if restan_col.notna().any():
    fig_tracker.add_trace(go.Scatter(
        x=df_chart_30['Waktu'], y=restan_col/1000,
        name='Restan (ton)', mode='lines+markers',
        line=dict(color='#F87171', width=2.5, dash='solid'),
        marker=dict(size=5, color='#F87171'),
        fill='tozeroy', fillcolor='rgba(248,113,113,0.08)',
        hovertemplate='<b>%{x|%d %b}</b><br>Restan: %{y:.1f} ton<extra></extra>'
    ))

# Garis target zero restan
fig_tracker.add_hline(y=0, line_dash='dot', line_color='#475569', line_width=1.5,
                      annotation_text='Target: Zero Restan', annotation_position='bottom right',
                      annotation_font_color='#64748B', annotation_font_size=10)

# Threshold kuning & merah
fig_tracker.add_hline(y=RESTAN_WARN_YELLOW/1000, line_dash='dash', line_color='#FCD34D',
                      line_width=1, annotation_text=f'Warn {RESTAN_WARN_YELLOW//1000}t',
                      annotation_position='top left', annotation_font_color='#FCD34D', annotation_font_size=9)
fig_tracker.add_hline(y=RESTAN_WARN_RED/1000, line_dash='dash', line_color='#FCA5A5',
                      line_width=1, annotation_text=f'Kritis {RESTAN_WARN_RED//1000}t',
                      annotation_position='top left', annotation_font_color='#FCA5A5', annotation_font_size=9)

fig_tracker.update_layout(
    paper_bgcolor='#0F172A', plot_bgcolor='#1E293B',
    font=dict(family='Plus Jakarta Sans', color='#94A3B8', size=11),
    height=380,
    margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    xaxis=dict(gridcolor='#1E3A5F', linecolor='#334155', tickformat='%d %b'),
    yaxis=dict(gridcolor='#1E3A5F', linecolor='#334155', title='Ton (×1000 kg)'),
    hovermode='x unified',
)
st.plotly_chart(fig_tracker, use_container_width=True)


# ─────────────────────────────────────────────────────
# SECTION 3: STATISTIK RINGKAS
# ─────────────────────────────────────────────────────
st.markdown('<p class="sec-header">📈 Statistik Periode Ini</p>', unsafe_allow_html=True)

col_a, col_b, col_c, col_d = st.columns(4)
df_stat = df_chart_30[df_chart_30['Produksi'] > 0]

with col_a:
    avg_prod = df_stat['Produksi'].mean() if len(df_stat) > 0 else 0
    st.markdown(f"""<div class="card">
        <div class="card-label">Rata-rata Produksi/Hari</div>
        <div class="card-value">{avg_prod/1000:.1f}<span style="font-size:1rem;color:#64748B"> ton</span></div>
        <div class="card-sub">30 hari terakhir</div>
    </div>""", unsafe_allow_html=True)

with col_b:
    if 'Restan' in df_stat.columns:
        zero_days = (df_stat['Restan'] == 0).sum()
        total_days = len(df_stat)
        pct_zero = zero_days / total_days * 100 if total_days > 0 else 0
    else:
        zero_days, pct_zero = 0, 0
    st.markdown(f"""<div class="card">
        <div class="card-label">Hari Zero-Restan</div>
        <div class="card-value" style="color:#34D399">{zero_days}<span style="font-size:1rem;color:#64748B"> hari</span></div>
        <div class="card-sub">{pct_zero:.0f}% dari total hari kerja</div>
    </div>""", unsafe_allow_html=True)

with col_c:
    if 'Restan' in df_stat.columns:
        avg_restan = df_stat['Restan'].mean()
    else:
        avg_restan = 0
    st.markdown(f"""<div class="card">
        <div class="card-label">Rata-rata Restan/Hari</div>
        <div class="card-value" style="color:#F87171">{avg_restan/1000:.1f}<span style="font-size:1rem;color:#64748B"> ton</span></div>
        <div class="card-sub">30 hari terakhir</div>
    </div>""", unsafe_allow_html=True)

with col_d:
    total_prod = df_stat['Produksi'].sum() if len(df_stat) > 0 else 0
    st.markdown(f"""<div class="card">
        <div class="card-label">Total Produksi</div>
        <div class="card-value">{total_prod/1_000_000:.2f}<span style="font-size:1rem;color:#64748B"> ribu ton</span></div>
        <div class="card-sub">30 hari terakhir</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# SECTION 4: TABEL HISTORY INPUT HARIAN
# ─────────────────────────────────────────────────────
if history:
    st.markdown('<p class="sec-header">📋 Riwayat Input Harian</p>', unsafe_allow_html=True)
    df_hist_display = pd.DataFrame(history)
    df_hist_display['Waktu'] = pd.to_datetime(df_hist_display['Waktu']).dt.strftime('%d %b %Y')
    df_hist_display.columns = [c.replace('_', ' ').title() for c in df_hist_display.columns]
    st.dataframe(df_hist_display.iloc[::-1], use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────
st.markdown("---")
st.caption("🌴 Zero Restan Optimizer · PTPN IV PalmCo · Built with Streamlit")
