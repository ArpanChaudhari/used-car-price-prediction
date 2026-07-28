import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AUTOVAL — Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

# ── Load model & dataset (cached) ────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model      = joblib.load("car_price_model.joblib")
    df         = pd.read_csv("used_car_dataset.csv")
    # Frequency encoding maps — must match training notebook exactly
    brand_freq = df["Brand"].value_counts(normalize=True)
    model_freq = df["model"].value_counts(normalize=True)
    brand_to_models = {
        brand: sorted(df[df["Brand"] == brand]["model"].unique().tolist())
        for brand in sorted(df["Brand"].unique())
    }
    return model, brand_freq, model_freq, brand_to_models

ML_MODEL, BRAND_FREQ, MODEL_FREQ, BRAND_TO_MODELS = load_assets()

# ── Encoding maps (match notebook Cell 30) ───────────────────────────────────
TRANSMISSION_MAP = {"Manual": 0, "Automatic": 1}
OWNER_MAP        = {"1st Owner": 0, "2nd Owner": 1}
FUEL_MAP         = {"Diesel": 0, "Petrol": 1, "Hybrid/CNG": 2}

FUEL_TYPES    = ["Petrol", "Diesel", "Hybrid/CNG"]
TRANSMISSIONS = ["Manual", "Automatic"]
OWNER_TYPES   = ["1st Owner", "2nd Owner"]
ALL_BRANDS    = sorted(BRAND_TO_MODELS.keys())

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@700;900&display=swap');

  /* ── Base ────────────────────────────────────────────── */
  html, body, [class*="css"] {
    background-color: #111111 !important;
    color: #d0d0d0 !important;
    font-family: 'Rajdhani', 'Segoe UI', sans-serif !important;
  }
  .stApp { background-color: #111111 !important; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 2.5rem 2rem 2.5rem !important; max-width: 100% !important; }

  /* ── Sidebar ─────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background-color: #161616 !important;
    border-right: 1px solid #222 !important;
  }
  [data-testid="stSidebar"] * { color: #d0d0d0 !important; }

  /* ── Selectbox ───────────────────────────────────────── */
  [data-testid="stSelectbox"] > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #e8e8e8 !important;
    border-radius: 4px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
  }
  [data-testid="stSelectbox"] svg { fill: #666 !important; }

  /* ── Slider ──────────────────────────────────────────── */
  [data-testid="stSlider"] > div > div > div > div { background-color: #f5c842 !important; }
  [data-testid="stSlider"] > div > div > div       { background-color: #2a2a2a !important; }
  .stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #f5c842 !important;
    border-color: #111 !important;
  }
  [data-testid="stSlider"] p {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    color: #555 !important;
  }

  /* ── Number input ────────────────────────────────────── */
  [data-testid="stNumberInput"] input {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #e8e8e8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 14px !important;
    border-radius: 4px !important;
  }

  /* ── Buttons ─────────────────────────────────────────── */
  [data-testid="stButton"] button {
    background-color: transparent !important;
    border: 1px solid #333 !important;
    color: #888 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    border-radius: 3px !important;
    transition: all 0.15s !important;
    padding: 6px 16px !important;
  }
  [data-testid="stButton"] button:hover { border-color: #f5c842 !important; color: #f5c842 !important; }

  /* ── Metric ──────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 12px 16px;
  }
  [data-testid="stMetricLabel"] {
    color: #888 !important;
    font-size: 12px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
  }
  [data-testid="stMetricValue"] { color: #f5c842 !important; font-size: 28px !important; }
  [data-testid="stMetricDelta"] { color: #4caf82 !important; }

  hr { border-color: #222 !important; }

  /* ── Section labels ──────────────────────────────────── */
  .section-label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 13px;
    font-weight: 700;
    color: #f5c842;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 16px;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 10px;
  }

  /* ── Field labels ────────────────────────────────────── */
  .field-label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 12px;
    font-weight: 700;
    color: #aaaaaa;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 6px;
    margin-top: 14px;
  }

  /* ── Gold CTA ────────────────────────────────────────── */
  .gold-btn > button {
    background-color: #f5c842 !important;
    color: #111 !important;
    border: none !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 4px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    border-radius: 4px !important;
    cursor: pointer !important;
    text-transform: uppercase !important;
  }
  .gold-btn > button:hover { background-color: #d4a800 !important; color: #111 !important; }

  /* ── Radio pills ─────────────────────────────────────── */
  [data-testid="stRadio"] > div { gap: 8px !important; flex-wrap: wrap !important; }
  [data-testid="stRadio"] label {
    border: 1px solid #333 !important;
    border-radius: 3px !important;
    padding: 6px 16px !important;
    color: #999 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    cursor: pointer !important;
    background: transparent !important;
    transition: all 0.15s !important;
    text-transform: uppercase !important;
  }
  [data-testid="stRadio"] label:has(input:checked) {
    background: #f5c842 !important;
    color: #111 !important;
    border-color: #f5c842 !important;
    font-weight: 700 !important;
  }
  [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display: none !important; }

  /* ── Output cards ────────────────────────────────────── */
  .output-card {
    background: #161616;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 24px;
    margin-bottom: 14px;
  }
  .output-label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 12px;
    font-weight: 700;
    color: #f5c842;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 20px;
    display: block;
  }
  .price-big {
    font-family: 'Orbitron', monospace !important;
    font-size: 40px;
    color: #f5c842;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1.1;
  }
  .price-sub {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px;
    color: #888;
    margin-top: 4px;
    letter-spacing: 1px;
  }
  .price-range {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px;
    color: #666;
    margin-top: 6px;
    margin-bottom: 20px;
  }

  .factor-row   { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 9px; align-items: center; }
  .factor-label {
    font-family: 'Rajdhani', sans-serif !important;
    color: #888;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 1px;
  }
  .factor-neg  { font-family: 'Share Tech Mono', monospace !important; color: #e05a5a; font-size: 12px; }
  .factor-pos  { font-family: 'Share Tech Mono', monospace !important; color: #4caf82; font-size: 12px; }
  .factor-none { font-family: 'Share Tech Mono', monospace !important; color: #444;    font-size: 12px; }

  .info-row {
    display: flex;
    justify-content: space-between;
    padding: 9px 0;
    border-bottom: 1px solid #1e1e1e;
    align-items: center;
  }
  .info-key {
    font-family: 'Rajdhani', sans-serif !important;
    color: #888;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1px;
  }
  .info-val {
    font-family: 'Share Tech Mono', monospace !important;
    color: #aaa;
    font-size: 12px;
  }
  .info-val-hi {
    font-family: 'Share Tech Mono', monospace !important;
    color: #f0f0f0;
    font-size: 13px;
    font-weight: 700;
  }

  .await-wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 40px 0; gap: 16px;
  }
  .await-ring {
    width: 56px; height: 56px; border: 1.5px solid #333; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
  }
  .await-ring-inner { width: 36px; height: 36px; border: 1.5px solid #3a3a3a; border-radius: 50%; }
  .await-text {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 13px;
    font-weight: 600;
    color: #555;
    letter-spacing: 3px;
    text-transform: uppercase;
  }

  .alert-box {
    background: rgba(224,90,90,0.08);
    border: 1px solid rgba(224,90,90,0.3);
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 13px;
    font-weight: 600;
    color: #e05a5a;
    margin-top: 12px;
    letter-spacing: 1px;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
def init_state():
    defaults = dict(
        brand=ALL_BRANDS[0],
        car_model=BRAND_TO_MODELS[ALL_BRANDS[0]][0],
        year=2018,
        km=45000,
        fuel="Petrol",
        transmission="Manual",
        owners="1st Owner",
        result=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Real ML Prediction ────────────────────────────────────────────────────────
def predict_price(brand, car_model, year, km, fuel, transmission, owner):
    age       = 2024 - year
    tx_enc    = TRANSMISSION_MAP[transmission]
    owner_enc = OWNER_MAP[owner]
    fuel_enc  = FUEL_MAP[fuel]
    brand_enc = float(BRAND_FREQ.get(brand, BRAND_FREQ.mean()))
    model_enc = float(MODEL_FREQ.get(car_model, MODEL_FREQ.mean()))

    X = pd.DataFrame({
        "Age":          [age],
        "kmDriven":     [km],
        "Transmission": [tx_enc],
        "Owner":        [owner_enc],
        "FuelType":     [fuel_enc],
        "Brand_enc":    [brand_enc],
        "Model_enc":    [model_enc],
    })

    log_price  = ML_MODEL.predict(X)[0]
    price_rs   = np.exp(log_price)
    price_l    = price_rs / 100_000

    # Confidence from variance across forest estimators (use .values to avoid warnings)
    X_arr      = X.values
    tree_preds = np.array([t.predict(X_arr)[0] for t in ML_MODEL.estimators_])
    std_log    = tree_preds.std()
    confidence = float(max(60.0, min(98.0, 100.0 - std_log * 55)))

    low  = np.exp(log_price - std_log) / 100_000
    high = np.exp(log_price + std_log) / 100_000

    return dict(
        price=round(price_l, 2),
        low=round(low, 2),
        high=round(high, 2),
        confidence=round(confidence, 1),
        price_rs=round(price_rs, 0),
    )

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:14px 0;border-bottom:1px solid #1e1e1e;background:#111;margin-bottom:0;">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="width:38px;height:38px;background:#f5c842;border-radius:50%;
                display:flex;align-items:center;justify-content:center;
                font-weight:900;font-size:13px;color:#111;font-family:'Orbitron',monospace;">
      AV
    </div>
    <div>
      <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:18px;letter-spacing:4px;color:#fff;line-height:1;">AUTOVAL</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#666;letter-spacing:1px;margin-top:2px;">Random Forest Valuation Engine v3.0</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="width:7px;height:7px;background:#4caf82;border-radius:50%;"></div>
    <span style="font-family:'Rajdhani',sans-serif;font-size:12px;font-weight:600;color:#666;letter-spacing:2px;">MODEL ONLINE</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#161616;border:1px solid #222;border-radius:6px;
            padding:36px 48px;margin-top:20px;">
  <div style="font-family:'Rajdhani',sans-serif;font-size:12px;font-weight:700;color:#f5c842;letter-spacing:4px;text-transform:uppercase;margin-bottom:14px;">// Price Estimator</div>
  <div style="font-family:'Orbitron',monospace;font-size:42px;font-weight:900;color:#fff;line-height:1.05;
              letter-spacing:-1px;margin-bottom:18px;">
    USED CAR<br><span style="color:#f5c842;">PRICE</span> PREDICTOR
  </div>
  <p style="font-family:'Rajdhani',sans-serif;font-size:16px;font-weight:500;color:#888;line-height:1.7;max-width:520px;margin:0;letter-spacing:0.5px;">
    Enter your vehicle specifications below. Our Random Forest model,
    trained on 9,582 real Indian market transactions, returns an accurate
    market valuation in under 2 seconds.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:20px;'>", unsafe_allow_html=True)
left, right = st.columns([2.2, 1], gap="large")

# ════════════════════════════════════════════════════════════
# LEFT COLUMN
# ════════════════════════════════════════════════════════════
with left:

    # ── Section 01 ────────────────────────────────────────
    st.markdown('<div class="section-label">01 — VEHICLE IDENTITY</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="field-label">BRAND</div>', unsafe_allow_html=True)
        brand = st.selectbox(
            "Brand", ALL_BRANDS,
            index=ALL_BRANDS.index(st.session_state.brand) if st.session_state.brand in ALL_BRANDS else 0,
            label_visibility="collapsed", key="brand_sel"
        )
        if brand != st.session_state.brand:
            st.session_state.brand = brand
            st.session_state.car_model = BRAND_TO_MODELS[brand][0]
            st.rerun()

    with c2:
        st.markdown('<div class="field-label">MODEL</div>', unsafe_allow_html=True)
        models_for_brand = BRAND_TO_MODELS[st.session_state.brand]
        model_idx = (models_for_brand.index(st.session_state.car_model)
                     if st.session_state.car_model in models_for_brand else 0)
        car_model = st.selectbox(
            "Model", models_for_brand, index=model_idx,
            label_visibility="collapsed", key="model_sel"
        )
        st.session_state.car_model = car_model

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="field-label">MANUFACTURE YEAR</div>', unsafe_allow_html=True)
        year = st.slider("Year", 1990, 2024, st.session_state.year,
                         label_visibility="collapsed", key="year_sl")
        st.session_state.year = year
        st.markdown(
            f'<div style="font-size:11px;color:#f5c842;margin-top:-8px;">'
            f'Selected: {year} &nbsp;·&nbsp; Age: {2024 - year} yrs</div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown('<div class="field-label">MILEAGE (KM DRIVEN)</div>', unsafe_allow_html=True)
        km = st.slider("KM", 1000, 500000, st.session_state.km, step=1000,
                       label_visibility="collapsed", key="km_sl")
        st.session_state.km = km
        st.markdown(
            f'<div style="font-size:11px;color:#f5c842;margin-top:-8px;">'
            f'Selected: {km:,} km</div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section 02 ────────────────────────────────────────
    st.markdown('<div class="section-label">02 — MECHANICAL PROFILE</div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="field-label">FUEL TYPE</div>', unsafe_allow_html=True)
        fuel = st.radio(
            "Fuel", FUEL_TYPES,
            index=FUEL_TYPES.index(st.session_state.fuel),
            horizontal=True, label_visibility="collapsed", key="fuel_r"
        )
        st.session_state.fuel = fuel

    with c6:
        st.markdown('<div class="field-label">TRANSMISSION</div>', unsafe_allow_html=True)
        tx = st.radio(
            "TX", TRANSMISSIONS,
            index=TRANSMISSIONS.index(st.session_state.transmission),
            horizontal=True, label_visibility="collapsed", key="tx_r"
        )
        st.session_state.transmission = tx

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section 03 ────────────────────────────────────────
    st.markdown('<div class="section-label">03 — OWNERSHIP HISTORY</div>', unsafe_allow_html=True)

    st.markdown('<div class="field-label">PREVIOUS OWNER COUNT</div>', unsafe_allow_html=True)
    owners = st.radio(
        "Owners", OWNER_TYPES,
        index=OWNER_TYPES.index(st.session_state.owners),
        horizontal=True, label_visibility="collapsed", key="owners_r"
    )
    st.session_state.owners = owners

    st.markdown("""
    <div style="font-size:10px;color:#444;letter-spacing:1px;margin-top:14px;
                padding:10px 14px;border:1px solid #1e1e1e;border-radius:4px;">
      MODEL SCOPE: Trained on Indian used car market (2 owner categories).
      Valuations reflect current INR market pricing.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CTA button ────────────────────────────────────────
    st.markdown('<div class="gold-btn">', unsafe_allow_html=True)
    if st.button("ESTIMATE PRICE ->", use_container_width=True, key="estimate_btn"):
        with st.spinner("Running Random Forest model..."):
            time.sleep(0.6)
        try:
            st.session_state.result = predict_price(
                brand=st.session_state.brand,
                car_model=st.session_state.car_model,
                year=st.session_state.year,
                km=st.session_state.km,
                fuel=st.session_state.fuel,
                transmission=st.session_state.transmission,
                owner=st.session_state.owners,
            )
        except Exception as e:
            st.session_state.result = {"error": str(e)}
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# RIGHT COLUMN
# ════════════════════════════════════════════════════════════
with right:

    result = st.session_state.result

    if result is None:
        st.markdown("""
        <div class="output-card">
          <span class="output-label">VALUATION OUTPUT</span>
          <div class="await-wrap">
            <div class="await-ring"><div class="await-ring-inner"></div></div>
            <div class="await-text">Awaiting input</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    elif "error" in result:
        st.markdown(f"""
        <div class="output-card">
          <span class="output-label">VALUATION OUTPUT</span>
          <div class="alert-box">Prediction error: {result['error']}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        age      = 2024 - st.session_state.year
        age_pct  = round(age * 6.5, 1)
        km_pct   = round((st.session_state.km / 10000) * 1.5, 1)
        fuel_tag = {"Diesel": "+high resale", "Hybrid/CNG": "+eco bonus"}.get(st.session_state.fuel, "neutral")
        tx_tag   = "+premium" if st.session_state.transmission == "Automatic" else "standard"
        fuel_cls = "factor-pos" if st.session_state.fuel != "Petrol" else "factor-none"
        tx_cls   = "factor-pos" if st.session_state.transmission == "Automatic" else "factor-none"
        owner_tag = "single owner" if st.session_state.owners == "1st Owner" else "-resale risk"
        owner_cls = "factor-pos"  if st.session_state.owners == "1st Owner" else "factor-neg"

        st.markdown(f"""
        <div class="output-card">
          <span class="output-label">VALUATION OUTPUT</span>
          <div class="price-big">Rs {result['price']:.2f}L</div>
          <div class="price-sub">approx Rs {result['price_rs']:,.0f}</div>
          <div class="price-range">Range: Rs {result['low']:.1f}L -- Rs {result['high']:.1f}L</div>
          <div style="font-size:10px;color:#555;letter-spacing:1px;margin-bottom:6px;">
            CONFIDENCE -- {result['confidence']}%
          </div>
          <div style="height:3px;background:#222;border-radius:2px;margin-bottom:20px;">
            <div style="height:100%;width:{result['confidence']}%;background:#4caf82;border-radius:2px;"></div>
          </div>
          <div style="border-top:1px solid #222;padding-top:16px;">
            <div style="font-size:10px;color:#555;letter-spacing:1px;margin-bottom:10px;">KEY FACTORS</div>
            <div class="factor-row">
              <span class="factor-label">Age ({age} yrs)</span>
              <span class="factor-neg">-{age_pct}% est.</span>
            </div>
            <div class="factor-row">
              <span class="factor-label">Mileage ({st.session_state.km:,} km)</span>
              <span class="factor-neg">-{km_pct}% est.</span>
            </div>
            <div class="factor-row">
              <span class="factor-label">Fuel type</span>
              <span class="{fuel_cls}">{fuel_tag}</span>
            </div>
            <div class="factor-row">
              <span class="factor-label">Transmission</span>
              <span class="{tx_cls}">{tx_tag}</span>
            </div>
            <div class="factor-row">
              <span class="factor-label">Ownership</span>
              <span class="{owner_cls}">{owner_tag}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Model Info ────────────────────────────────────────
    st.markdown("""
    <div class="output-card">
      <span class="output-label">MODEL INFO</span>
      <div class="info-row">
        <span class="info-key">Algorithm</span>
        <span class="info-val-hi">Random Forest</span>
      </div>
      <div class="info-row">
        <span class="info-key">Estimators</span>
        <span class="info-val">100 trees</span>
      </div>
      <div class="info-row">
        <span class="info-key">Training samples</span>
        <span class="info-val">9,582</span>
      </div>
      <div class="info-row">
        <span class="info-key">Target variable</span>
        <span class="info-val">log(Price INR)</span>
      </div>
      <div class="info-row">
        <span class="info-key">Brand coverage</span>
        <span class="info-val">39 brands</span>
      </div>
      <div class="info-row" style="border-bottom:none;">
        <span class="info-key">Market</span>
        <span class="info-val">India (INR)</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
