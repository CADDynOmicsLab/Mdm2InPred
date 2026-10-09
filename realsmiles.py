import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tempfile
import shutil
import os
import math
from padelpy import from_smiles
from rdkit import Chem
from rdkit.Chem import Draw, AllChem    
from io import BytesIO
import base64
from openai import OpenAI
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler

# ------------------------------
# CONFIG - Update paths
# ------------------------------
MODEL_PATHS = {
    "Model 1": "lightgbm.pkl",
    "Model 2": "rf.joblib"
}
FEATURE_PATHS = {
    "Model 1": "lightgbm_feature_importances.csv",
    "Model 2": "rf_feature_importance.csv"
}

TRAIN_PATHS = {
    "Model 1": "static/lightgbm_train_set.csv",
    "Model 2": "static/train_set.csv"
}
DATASET_PATH = "main.csv"

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="MDM2InPred: Prediction of MDM2 Inhibitors", layout="wide")

# ------------------------------
# MASTER CSS — Premium Teal/Glassmorphism Theme
# ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&family=Merriweather:wght@300;400;700;900&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --teal-dark:   #0d6e6e;
    --teal:        #1a9090;
    --teal-mid:    #22b8b8;
    --teal-light:  #5dd6d6;
    --cyan-soft:   #e0f7f7;
    --cyan-pale:   #f0fafa;
    --white:       #ffffff;
    --gray-100:    #f8fafa;
    --gray-200:    #edf2f2;
    --gray-500:    #6b8f8f;
    --gray-700:    #2d4a4a;
    --gray-900:    #0f2626;
    --shadow-sm:   0 2px 12px rgba(13,110,110,0.08);
    --shadow-md:   0 6px 30px rgba(13,110,110,0.13);
    --shadow-lg:   0 16px 50px rgba(13,110,110,0.18);
    --radius:      16px;
    --radius-sm:   10px;
}

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── APP BACKGROUND ── */
.stApp {
    background: linear-gradient(160deg, #f0fafa 0%, #e8f6f6 40%, #f5fdfd 100%) !important;
    font-family: 'Poppins', sans-serif !important;
    color: var(--gray-700) !important;
}

/* Hide default Streamlit header/menu */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
.stDeployButton { display: none; }
footer { visibility: hidden; }

/* ── ANIMATED BACKGROUND BLOBS ── */
.stApp::before {
    content: '';
    position: fixed;
    width: 520px; height: 520px;
    background: radial-gradient(circle, #5dd6d699, #1a909066);
    border-radius: 50%;
    filter: blur(90px);
    top: -150px; left: -120px;
    z-index: 0;
    pointer-events: none;
    animation: driftBlob1 18s ease-in-out infinite alternate;
}
.stApp::after {
    content: '';
    position: fixed;
    width: 380px; height: 380px;
    background: radial-gradient(circle, #b2efef88, #22b8b855);
    border-radius: 50%;
    filter: blur(80px);
    bottom: 10%; right: -80px;
    z-index: 0;
    pointer-events: none;
    animation: driftBlob2 22s ease-in-out infinite alternate;
}
@keyframes driftBlob1 {
    0%   { transform: translate(0,0) scale(1); }
    50%  { transform: translate(30px,20px) scale(1.06); }
    100% { transform: translate(-20px,40px) scale(0.96); }
}
@keyframes driftBlob2 {
    0%   { transform: translate(0,0) scale(1); }
    50%  { transform: translate(-25px,15px) scale(1.08); }
    100% { transform: translate(20px,-30px) scale(0.94); }
}

/* ── MAIN CONTENT BLOCK ── */
.block-container {
    padding: 0.5rem 2rem 3rem 2rem !important;
    max-width: 1200px !important;
}

/* ── HERO HEADER ── */
.mdm2-hero {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;        
    padding: 0px 32px 36px;
    margin-bottom: 16px;        
    animation: fadeSlideDown 0.9s ease both;
    position: relative;
}

/* Main serif heading */
.mdm2-title {
    font-family: 'Poppins', serif;
    font-size: clamp(1.75rem, 4.2vw, 3rem);
    font-weight: 900;
    line-height: 1.2;
    letter-spacing: 0.01em;
    color: #0B5D5B;
    text-align: center;
    background: linear-gradient(90deg,#00F5FF,#007CF0);
    -webkit-background-clip: text;
    webkit-text-fill-color: transparent;        
    margin: 0 auto;
    max-width: 1400px;
    white-space: nowrap;
    /* Subtle text shadow for depth */
    text-shadow: 0 1px 0 rgba(11,93,91,0.12);
    /* No gradient clip — pure color for serif legibility */
    -webkit-text-fill-color: unset;
    background: none;
}

/* The "Pred" accent within the title */
.mdm2-title-accent {
    color: #1a9090;
    font-style: italic;
    background: none;
    -webkit-background-clip: unset;
    -webkit-text-fill-color: #1a9090;
}

/* Small colon + subtitle section label */
.mdm2-tagline {
    display: none; /* replaced by integrated title text */
}

/* Subtitle */
.mdm2-subtitle {
    margin: 0 auto;
    font-family: 'Poppins', sans-serif;
    font-size: clamp(0.85rem, 1.8vw, 1rem);
    font-weight: 500;
    color: #557777;
    text-align: center;
    white-space: normal;
    max-width: 1400px;
    line-height: 1.75;
    letter-spacing: 0.01em;
}

/* ── NAVBAR (TABS) ── */
.stTabs [role="tablist"] {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    width: 100% !important;
    margin: 0 auto !important;
    margin-top: 16px !important;        
    background: #0B5D5B !important;
    border-radius: 0px !important;
    padding: 0 !important;
    overflow: hidden !important;
    border: 1px solid #0B5D5B !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08) !important;
    gap: 0 !important;
}
.stTabs [role="tab"] {
    flex: 1 !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 14px 0 !important;
    background: #156f6f !important;
    color: white !important;
    font-size: 1.0rem !important;
    font-weight: 500 !important;
    font-family: 'Poppins', sans-serif !important;
    border: none !important;
    border-right: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 0 !important;
    transition: all 0.25s ease !important;
    min-height: 54px !important;
}
.stTabs [role="tab"]:last-child {
    border-right: none !important;
}
.stTabs [role="tab"]:hover {
    background: #1a9090 !important;
    color: white !important;
    transform: none !important;
    box-shadow: inset 0 -3px 0 rgba(255,255,255,0.85) !important;
}
.stTabs [aria-selected="true"] {
    background: #0a5555 !important;
    color: white !important;
    font-weight: 600 !important;
    border-right: 1px solid rgba(255,255,255,0.25) !important;
    box-shadow: inset 0 -4px 0 white !important;
}
.stTabs [role="tab"] p {
    color: inherit !important;
    margin: 0 !important;
    padding: 0 !important;
    font-size: 1.0rem !important;
    font-weight: inherit !important;
}
/* Hide the underline indicator bar */
.stTabs [role="tablist"]::after,
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* Hide the underline indicator bar */
.stTabs [role="tablist"]::after,
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* Remove gap below navbar */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 0 !important;
}
[data-testid="stTabsContent"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
             
/* ── TEAL DIVIDER RULE below navbar ── */
.mdm2-nav-divider {
    width: 100%;
    max-width: 1400px;
    height: 1.5px;
    background: linear-gradient(90deg, transparent 0%, rgba(26,144,144,0.3) 20%, rgba(26,144,144,0.55) 50%, rgba(26,144,144,0.3) 80%, transparent 100%);
    margin: 2px auto 6px;
    border: none;
}

/* ── SECTION DIVIDER ── */
.mdm2-divider {
    height: 1.5px;
    margin: 10px auto;
    display: block;
    background: linear-gradient(90deg, transparent, rgba(93,214,214,0.45), transparent);
    max-width: 1400px;
    opacity: 0.7;
    border: none;
}

/* ── GLASS CARD BASE ── */
.glass-card {
    background: rgba(255,255,255,0.70);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(26,144,144,0.14);
    border-radius: var(--radius);
    box-shadow: var(--shadow-md);
    padding: 36px 38px;
}

/* ── WELCOME CARD (HOME RIGHT) ── */
.welcome-card {
    background: transparent;
    border: none;
    box-shadow: none;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 300px;
}
.welcome-eyebrow {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--teal-mid);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}
.welcome-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px; height: 2px;
    background: var(--teal-mid);
    border-radius: 2px;
}
.welcome-card h2 {
    font-size: clamp(1.4rem, 3vw, 1.9rem);
    font-weight: 700;
    color: var(--gray-900);
    line-height: 1.2;
    letter-spacing: -0.01em;
    margin: 0 0 20px 0;
    padding-top: 10px;
}
.welcome-card p {
    font-size: 1.7rem;
    color: #365c5c;
    line-height: 2.0;
    text-align: justify;
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
}

/* ── STATS ROW ── */
.stats-row {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    padding: 18px 0;
    border-top: 1px solid var(--gray-200);
    border-bottom: 1px solid var(--gray-200);
    margin: 18px 0;
}
.stat-item { text-align: center; flex: 1; min-width: 70px; }
.stat-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--teal-dark);
    font-family: 'Space Mono', monospace;
    line-height: 1;
}
.stat-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--gray-500);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 4px;
}
            
/* ── FEATURE CARDS ── */
.features-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin-top: 5px;
    width: 100%;
}
.feature-card {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(26,144,144,0.12);
    border-radius: var(--radius);
    padding: 20px 16px;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--teal-mid), var(--teal-light));
    opacity: 0;
    transition: opacity 0.3s ease;
}
.feature-card:hover { transform: translateY(-6px); box-shadow: var(--shadow-lg); border-color: rgba(26,144,144,0.28); }
.feature-card:hover::before { opacity: 1; }
.feature-icon-wrap {
    width: 52px; height: 52px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--teal-mid), var(--teal-dark));
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 16px;
    font-size: 1rem;
    box-shadow: 0 6px 18px rgba(26,144,144,0.3);
}
.feature-title { font-size: 0.96rem; font-weight: 700; color: var(--gray-900); margin-bottom: 7px; }
.feature-desc  { font-size: 0.82rem; color: var(--gray-500); line-height: 1.65; }

/* ── HOW IT WORKS STEPS ── */
.how-section {
    background: linear-gradient(135deg, rgba(13,110,110,0.04) 0%, transparent 100%);
    border: 1px solid rgba(26,144,144,0.1);
    border-radius: var(--radius);
    padding: 40px 36px;
    margin-top: 12px;
}
.steps-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 28px;
    position: relative;
}
.steps-row::before {
    content: '';
    position: absolute;
    top: 27px;
    left: calc(12.5% + 14px);
    right: calc(12.5% + 14px);
    height: 2px;
    background: linear-gradient(90deg, var(--teal-mid), var(--teal-light));
    z-index: 0;
}
.step-card { text-align: center; padding: 16px 10px; position: relative; z-index: 1; }
.step-num {
    width: 54px; height: 54px;
    border-radius: 50%;
    background: var(--white);
    border: 2px solid var(--teal-mid);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 14px;
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--teal-dark);
    font-family: 'Space Mono', monospace;
    box-shadow: 0 4px 14px rgba(26,144,144,0.18);
}
.step-title { font-size: 0.9rem; font-weight: 700; color: var(--gray-900); margin-bottom: 5px; }
.step-desc   { font-size: 0.8rem; color: #4f6f6f; line-height: 1.7;font-weight: 500; }

/* ── SECTION HEADINGS ── */
.section-header { text-align: center; margin-bottom: 8px; }
.section-header h3 {
    font-size: clamp(1.2rem, 2.5vw, 1.65rem);
    font-weight: 700;
    color: var(--gray-900);
    letter-spacing: -0.01em;
    margin: 0 0 6px 0;
}
.section-header p { font-size: 0.88rem; color: var(--gray-500); margin: 0; }

/* ── MODULE DESCRIPTION BOXES (Prediction/Converter/Dataset) ── */
.module-desc-box {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(26,144,144,0.10);
    border-top: 4px solid var(--teal-mid);
    border-radius: 18px;
    padding: 26px 28px;
    margin: 20px 0 24px 0;
    box-shadow:
        0 10px 30px rgba(13,110,110,0.08),
        0 2px 10px rgba(13,110,110,0.04);
    position: relative;
    overflow: hidden;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}
.module-desc-box:hover {
    transform: translateY(-6px);
    box-shadow:
        0 18px 40px rgba(13,110,110,0.14),
        0 4px 16px rgba(13,110,110,0.08);

    border-color: rgba(26,144,144,0.22);
}
.module-desc-box .mod-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--teal-dark);
    margin-bottom: 8px;
    font-family: 'Poppins', sans-serif;
    letter-spacing: -0.02em;
}
.module-desc-box .mod-text {
    font-size: 1rem;
    font-weight: 500;
    color: #486868;
    line-height: 1.9;
    text-align: justify;
}

/* ── STREAMLIT WIDGETS OVERRIDES ── */
/* Radio */
div[data-testid="stRadio"] label {
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.9rem !important;
    color: var(--gray-700) !important;
}
div[data-testid="stRadio"] > div { gap: 12px !important; }
div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    border-color: var(--teal-mid) !important;
}

/* Text area */
[data-testid="stTextArea"] textarea {
    background-color: #f0ffff !important;
    color: #003333 !important;
    font-size: 0.9rem !important;
    font-family: "Space Mono", monospace !important;
    border: 1.5px solid rgba(26,144,144,0.35) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--teal-mid) !important;
    box-shadow: 0 0 0 3px rgba(34,184,184,0.15) !important;
}
[data-testid="stTextArea"] label p,
div[data-testid="stTextArea"] label {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: var(--gray-700) !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    background: #f0ffff !important;
    border: 1.5px solid rgba(26,144,144,0.35) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Poppins', sans-serif !important;
    color: var(--gray-700) !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--teal-mid) !important;
    box-shadow: 0 0 0 3px rgba(34,184,184,0.15) !important;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    background-color: #e6ffff !important;
    border: 2px dashed rgba(26,144,144,0.45) !important;
    border-radius: var(--radius-sm) !important;
}

/* Primary Streamlit button */
.stButton > button {
    background: var(--teal-dark) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 26px !important;
    box-shadow: 0 4px 16px rgba(13,110,110,0.3) !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: var(--teal) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(13,110,110,0.38) !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: rgba(26,144,144,0.1) !important;
    color: var(--teal-dark) !important;
    border: 1.5px solid var(--teal-mid) !important;
    border-radius: 50px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 9px 22px !important;
    transition: all 0.25s ease !important;
    margin-top: 14px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--cyan-soft) !important;
    transform: translateY(-2px) !important;
}

/* Dataframe / table */
[data-testid="stDataFrame"] { border-radius: var(--radius-sm) !important; overflow: hidden !important; }

/* Select box */
[data-testid="stSelectbox"] > div > div {
    border: 1.5px solid rgba(26,144,144,0.35) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Success/Warning/Error messages */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Spinner text */
[data-testid="stSpinner"] { font-family: 'Poppins', sans-serif !important; }

/* Markdown text */
.stMarkdown p {
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.92rem !important;
}

/* ── RESULT TABLE ── */
.result-section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--teal-dark);
    font-family: 'Poppins', sans-serif;
    margin: 24px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
table {
    width: 100% !important;
    table-layout: fixed !important;
    border-collapse: collapse !important;
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
    font-family: 'Poppins', sans-serif !important;
}
table th {
    background-color: var(--teal-dark) !important;
    color: white !important;
    font-weight: 600 !important;
    text-align: center !important;
    font-size: 0.85rem !important;
    padding: 12px 8px !important;
    vertical-align: middle !important;
    white-space: normal !important;
    word-wrap: break-word;
    line-height: 1.35;
}
table td {
    padding: 11px !important;
    text-align: center !important;
    border: 1px solid #d8efef !important;
    font-size: 0.88rem !important;
    word-wrap: break-word;
    overflow-wrap: break-word;
    word-break: break-all;
}
table tr:hover td { background-color: #e6ffff !important; }

/* Fixed column width ratios for prediction results table */
table th:nth-child(1), table td:nth-child(1) { width: 11%; }  /* Structure */
table th:nth-child(2), table td:nth-child(2) { width: 22%; }  /* SMILES */
table th:nth-child(3), table td:nth-child(3) { width: 13%; }  /* Predicted pIC50 */
table th:nth-child(4), table td:nth-child(4) { width: 14%; }  /* Bioactivity */
table th:nth-child(5), table td:nth-child(5) { width: 18%; }  /* Prediction */
table th:nth-child(6), table td:nth-child(6) { width: 22%; }  /* Applicability Domain Status */
table td:nth-child(2) {
    text-align: left !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem !important;
    line-height: 1.4;
    max-width: 0;
    word-break: break-all;
}

/* ── EXAMPLE TEXT ── */
.example-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(26,144,144,0.08);
    border: 1px solid rgba(26,144,144,0.2);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.82rem;
    color: var(--teal-dark);
    font-family: 'Space Mono', monospace;
    margin-top: 8px;
    margin-bottom: 18px;
}
.example-pill::before { content: '◉'; color: var(--teal-mid); font-size: 0.7rem; }

/* ── CONTACT / PROFILE CARDS ── */
.profile-card {
    background: rgba(255,255,255,0.75);
    border-radius: var(--radius);
    padding: 24px 18px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    border: 1px solid rgba(26,144,144,0.12);
    transition: all 0.3s ease;
    margin: 8px 0;
}
.profile-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-md); }
.profile-card img {
    width: 110px; height: 110px;
    border-radius: 50%;
    object-fit: cover;
    margin-bottom: 12px;
    border: 3px solid var(--teal-mid);
    box-shadow: 0 4px 14px rgba(26,144,144,0.25);
}
.profile-name  { font-size: 1rem; font-weight: 700; color: var(--teal-dark); margin-bottom: 3px; }
.profile-role  { font-size: 0.82rem; color: var(--gray-500); margin-bottom: 2px; }
.profile-email { font-size: 0.78rem; color: var(--teal); }

/* ── DATASET BUTTONS ── */
.dataset-btn-group .stButton > button {
    background: rgba(26,144,144,0.1) !important;
    color: var(--teal-dark) !important;
    border: 1.5px solid var(--teal-mid) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.85rem !important;
    padding: 10px 18px !important;
    box-shadow: none !important;
}
.dataset-btn-group .stButton > button:hover {
    background: var(--teal-dark) !important;
    color: white !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── CHAT MESSAGES ── */
.chat-bubble {
    padding: 12px 18px;
    border-radius: 14px;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 10px;
    font-family: 'Poppins', sans-serif;
    max-width: 85%;
}
.chat-user {
    background: var(--teal-dark);
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}
.chat-bot {
    background: rgba(255,255,255,0.8);
    border: 1px solid rgba(26,144,144,0.18);
    color: var(--gray-700);
    border-bottom-left-radius: 4px;
    box-shadow: var(--shadow-sm);
}
.chat-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.chat-label-user { color: var(--teal-light); text-align: right; }
.chat-label-bot  { color: var(--teal-mid); }

/* ── VIDEO ── */
[data-testid="stVideo"] { border-radius: var(--radius) !important; overflow: hidden !important; box-shadow: var(--shadow-md) !important; }

/* ── ANIMATIONS ── */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-22px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(0.8); }
}
@keyframes floatY {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-8px); }
}
.float-anim { animation: floatY 5s ease-in-out infinite; }

/* Mol vis card */
.mol-vis-card {
    background: transparent;
    border: none;
    box-shadow: none;
    overflow: hidden;
    max-width: 500px;
    margin: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 280px;
}
            
/* EQUAL HEIGHT HERO COLUMNS */
.hero-equal-height {
    display: flex;
    align-items: center;
    gap: 28px;
    width: 100%;
    max-width: 1200px;
    margin: auto;
}

.hero-left,
.hero-right {
    flex: 1;
    display: flex;
    flex-direction: column;
}

.hero-left > div,
.hero-right > div {
    height: 100%;
}

html, body, .main {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

.block-container {
    padding-bottom: 0rem !important;
}

.custom-footer{
    margin-bottom: 0 !important;
}            

.custom-footer{
    width: 100%;
    background: linear-gradient(
        90deg,
        #0b5d5b 0%,
        #0f6d6a 50%,
        #0b5d5b 100%
    );
    color: white;
    margin-top: 30px;
    margin-bottom: -2rem;
    padding: 14px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-radius: 0;
    font-size: 0.92rem;
    box-shadow:
        0 -2px 12px rgba(0,0,0,0.08);
}

.footer-left{
    font-weight: 600;
    letter-spacing: 0.4px;
}

.footer-center{
    text-align: center;
    line-height: 1.6;
    opacity: 0.95;
}

.footer-right{
    font-size: 0.88rem;
    opacity: 0.95;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Groq Client (OpenAI-compatible)
# ------------------------------
client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

# ------------------------------
# Converter Functions
# ------------------------------
def compute_hat_matrix_inv(X_train):
    X = np.array(X_train, dtype=float)
    XtX = X.T @ X
    return np.linalg.pinv(XtX)

def compute_leverage(X_train, X_query, XtX_inv=None):
    if XtX_inv is None:
        XtX_inv = compute_hat_matrix_inv(X_train)
    X_q = np.array(X_query, dtype=float)
    return np.array([x @ XtX_inv @ x for x in X_q])

def compute_h_star(n_train, p_descriptors):
    return 3.0 * (p_descriptors + 1) / n_train

def compute_applicability_domain(train_df, query_df, feature_cols):
    common = [f for f in feature_cols if f in train_df.columns and f in query_df.columns]
    if not common:
        return ["N/A"] * len(query_df)

    X_train = train_df[common].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_query = query_df[common].apply(pd.to_numeric, errors="coerce").fillna(0).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_query_scaled = scaler.transform(X_query)

    XtX_inv = compute_hat_matrix_inv(X_train_scaled)
    h_star = compute_h_star(n_train=X_train_scaled.shape[0], p_descriptors=X_train_scaled.shape[1])
    h_query = compute_leverage(X_train_scaled, X_query_scaled, XtX_inv)

    return ["Fit" if h <= h_star else "Outlier" for h in h_query]

def pIC50_to_IC50(pIC50):
    return 10 ** (-pIC50)

def IC50_to_pIC50(IC50):
    return -math.log10(IC50)

# ------------------------------
# Helper Functions
# ------------------------------
def filter_valid_smiles(smiles_list):
    valid, invalid = [], []
    for s in smiles_list:
        if Chem.MolFromSmiles(s):
            valid.append(s)
        else:
            invalid.append(s)
    return valid, invalid

def smiles_to_img_tag(smi, size=(140, 140)):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode()
    img_tag = f'<img src="data:image/png;base64,{b64_img}" style="width:{size[0]}px;height:{size[1]}px;display:block;margin:auto;">'
    
    try:
        mol3d = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol3d, maxAttempts=10, randomSeed=42)
        if res == 0:
            AllChem.MMFFOptimizeMolecule(mol3d)
            sdf_block = Chem.MolToMolBlock(mol3d)
        else:
            AllChem.Compute2DCoords(mol)
            sdf_block = Chem.MolToMolBlock(mol)
    except:
        try:
            AllChem.Compute2DCoords(mol)
            sdf_block = Chem.MolToMolBlock(mol)
        except:
            sdf_block = ""
            
    if sdf_block:
        sdf_b64 = base64.b64encode(sdf_block.encode('utf-8')).decode('utf-8')
        download_link = f'<a href="data:chemical/x-mdl-sdfile;base64,{sdf_b64}" download="molecule.sdf" style="font-size:12px;text-decoration:none;color:#0d6e6e;font-weight:600;">download .sdf file</a>'
        
        html_viewer = f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
</head>
<body style="margin:0;padding:0;width:100vw;height:100vh;overflow:hidden;">
    <div id="container" style="width:100%;height:100%;position:relative;"></div>
    <script>
        var sdf_data = atob("{sdf_b64}");
        var viewer = $3Dmol.createViewer("container", {{backgroundColor: "white"}});
        viewer.addModel(sdf_data, "sdf");
        viewer.setStyle({{}}, {{stick: {{}}, sphere: {{scale: 0.3}}}});
        viewer.zoomTo();
        viewer.render();
    </script>
</body>
</html>"""
        view_link = f'<a href="#" class="mol-view-link" data-sdf="{sdf_b64}" style="font-size:12px;text-decoration:none;color:#0d6e6e;font-weight:600;">View</a>'
        
        links = f'<div style="text-align:center;margin-top:5px;">{download_link} | {view_link}</div>'
    else:
        links = ""
        
    return f'<div>{img_tag}{links}</div>'

def generate_descriptors_safe_individual(smiles_list):
    tmpdir = tempfile.mkdtemp()
    # Ensure Java is on PATH for PaDEL-Descriptor subprocess
    java_home = os.environ.get("JAVA_HOME", "")
    if java_home:
        java_bin = os.path.join(java_home, "bin")
        current_path = os.environ.get("PATH", "")
        if java_bin not in current_path:
            os.environ["PATH"] = java_bin + os.pathsep + current_path
    # Remove any JVM env-var overrides: padelpy raises RuntimeError on ANY
    # stderr output, including the JVM's own "Picked up ..." startup messages
    # emitted by JAVA_TOOL_OPTIONS or JDK_JAVA_OPTIONS.
    os.environ.pop("JAVA_TOOL_OPTIONS", None)
    os.environ.pop("JDK_JAVA_OPTIONS", None)
    all_desc = []
    successful_smiles = []
    skipped = []
    skipped_reasons = []
    progress_bar = st.progress(0)
    for i, smi in enumerate(smiles_list):
        try:
            batch_file = os.path.join(tmpdir, f"desc_{i}.csv")
            from_smiles([smi], output_csv=batch_file, fingerprints=True, timeout=600)
            if os.path.exists(batch_file):
                desc_df = pd.read_csv(batch_file)
                all_desc.append(desc_df)
                successful_smiles.append(smi)
            else:
                skipped.append(smi)
                skipped_reasons.append("PaDEL produced no output")
        except Exception as e:
            skipped.append(smi)
            skipped_reasons.append(str(e))
        progress_bar.progress((i + 1) / len(smiles_list))
    shutil.rmtree(tmpdir, ignore_errors=True)
    if skipped:
        reason_sample = skipped_reasons[0] if skipped_reasons else "unknown"
        st.warning(
            f"{len(skipped)} SMILES could not be processed. "
            f"Examples: {skipped[:5]}\n\nReason: {reason_sample}"
        )
    if not all_desc:
        st.error(
            "Descriptor generation failed for all SMILES. "
            "Please check that Java is installed and JAVA_HOME is set correctly, "
            "and that your SMILES strings are valid."
        )
        return None, []
    return pd.concat(all_desc, ignore_index=True), successful_smiles

def run_prediction(model_key, smiles_input, uploaded):
    if smiles_input.strip():
        smiles_list = [line.strip() for line in smiles_input.strip().splitlines()]
    elif uploaded:
        smiles_list = [line.decode("utf-8").strip() for line in uploaded.readlines()]
    else:
        st.error("Please provide SMILES input or upload a file.")
        return

    smiles_list, invalid_smiles = filter_valid_smiles(smiles_list)
    if invalid_smiles:
        st.warning(f"{len(invalid_smiles)} invalid SMILES removed. Examples: {invalid_smiles[:5]}")
    if not smiles_list:
        st.error("No valid SMILES to process!")
        return

    with st.spinner("Preparing prediction results…..."):
        desc_df, successful_smiles = generate_descriptors_safe_individual(smiles_list)

    if desc_df is None or desc_df.empty:
        return

    model = joblib.load(MODEL_PATHS[model_key])
    training_features = pd.read_csv(FEATURE_PATHS[model_key]).columns.tolist()
    common_features = [f for f in training_features if f in desc_df.columns]
    X = desc_df[common_features]
    if X.empty:
        st.error("No matching features found between descriptors and training data!")
        return

    with st.spinner(f"Running prediction with {model_key}…"):
        prediction = model.predict(X)

    with st.spinner("Checking applicability domain…"):
        try:
            train_df = pd.read_csv(TRAIN_PATHS[model_key])
            ad_status = compute_applicability_domain(
                train_df=train_df,
                query_df=desc_df,
                feature_cols=common_features
            )
        except Exception:
            ad_status = ["N/A"] * len(desc_df)

    activity = ["Likely Inhibitor" if p >= 6 else "Likely Non-Inhibitor" for p in prediction]

    structures = [smiles_to_img_tag(s) for s in successful_smiles]

    results_df = pd.DataFrame({
        "SMILES": successful_smiles,
        "Predicted pIC₅₀": prediction,
        "Bioactivity_IC₅₀ (uM)": [10**(-p) * 1e6 for p in prediction],
        "Prediction": activity,
        "Applicability Domain Status": ad_status
    })

    display_df = results_df.copy()
    display_df.insert(0, "Structure", structures)
    display_df["Predicted pIC₅₀"] = display_df["Predicted pIC₅₀"].map(lambda x: f"{x:.4f}")
    display_df["Bioactivity_IC₅₀ (uM)"] = display_df["Bioactivity_IC₅₀ (uM)"].map(lambda x: f"{x:.4f}")

    def style_table(row):
        base_color = '#f9ffff' if row.name % 2 == 0 else '#ffffff'
        base_style = f'background-color: {base_color}; color: #00332e; font-size:15px; text-align:center; vertical-align:middle;'
        styles = [base_style] * len(row)
        
        pred_idx = display_df.columns.get_loc("Prediction")
        if "Applicability Domain Status" in display_df.columns:
            ad_idx = display_df.columns.get_loc("Applicability Domain Status")
        else:
            ad_idx = -1

        if row["Prediction"] == "Likely Inhibitor":
            styles[pred_idx] = 'background-color: #0d6e6e; color: #00e676; font-weight:bold; text-align:center; vertical-align:middle; font-size:15px;'
        elif row["Prediction"] == "Likely Non-Inhibitor":
            styles[pred_idx] = 'background-color: #cce6ff; color: #00701a; font-weight:bold; text-align:center; vertical-align:middle; font-size:15px;'

        if ad_idx != -1:
            if row["Applicability Domain Status"] == "Fit":
                styles[ad_idx] = 'background-color: #d1fae5; color: #065f46; font-weight:bold; text-align:center; vertical-align:middle;'
            elif "Outlier" in str(row["Applicability Domain Status"]):
                styles[ad_idx] = 'background-color: #fee2e2; color: #991b1b; font-weight:bold; text-align:center; vertical-align:middle;'

        return styles

    display_df = display_df.reset_index(drop=True)
    styled_df = display_df.style.hide(axis="index").apply(style_table, axis=1)
    styled_df = styled_df.set_table_styles([
        {'selector': 'th', 'props': [('background-color','#0d6e6e'),('color','white'),('font-weight','bold'),('text-align','center'),('font-size','15px'),('padding','12px')]},
        {'selector': 'td', 'props': [('padding','11px'),('text-align','center'),('border','1px solid #d8efef')]}
    ])

    st.markdown("""
        <div class='result-section-title'>
            <span style='font-size:1.2rem'></span> Prediction Results
        </div>
    """, unsafe_allow_html=True)

    table_html = styled_df.to_html(escape=False)
    st.markdown(
        '<div style="width:100%;overflow-x:auto;overflow-y:auto;border-radius:12px;max-height:600px;">'
        + table_html + '</div>',
        unsafe_allow_html=True
    )

    csv_data = results_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download Prediction Results (CSV)",
        data=csv_data,
        file_name=f"{model_key}_predictions.csv",
        mime="text/csv"
    )

# ------------------------------
# Chatbot Logic
# ------------------------------
APP_CONTEXT = """
This is the MDM2InPred dashboard.

BIOLOGY BACKGROUND (use this to answer conceptual questions directly, without redirecting to a tab unless the user specifically asks how to use the tool):

MDM2 (murine double minute 2) is an E3 ubiquitin ligase and the primary negative regulator of the tumor suppressor protein p53. Under normal cellular conditions, MDM2 binds directly to the transactivation domain of p53, blocking its ability to activate target genes, and also tags p53 for degradation via the ubiquitin-proteasome pathway. This keeps p53 levels low during unstressed conditions.

In response to cellular stress — such as DNA damage, oncogene activation, or hypoxia — this MDM2-p53 interaction is disrupted, allowing p53 to accumulate and activate genes responsible for cell-cycle arrest, DNA repair, senescence, or apoptosis (programmed cell death). This is why p53 is often called the "guardian of the genome."

In many human cancers, MDM2 is amplified or overexpressed (notably in soft-tissue sarcomas, osteosarcomas, and some leukemias), which excessively suppresses p53 activity even when p53 itself is genetically normal (wild-type). This allows cancer cells to evade apoptosis and continue proliferating unchecked.

Small-molecule MDM2 inhibitors are drug candidates designed to block the MDM2-p53 protein-protein interaction, typically by occupying the p53-binding pocket on MDM2. This prevents MDM2 from suppressing p53, allowing p53 to become reactivated and restore its tumor-suppressive functions — making the MDM2-p53 axis an important and actively researched anticancer drug target. This is the biological basis and motivation for the MDM2InPred prediction tool.

Key terms:
- IC50: The concentration of an inhibitor required to reduce a biological activity (e.g., MDM2-p53 binding) by 50%. Lower IC50 = more potent inhibitor.
- pIC50: The negative log10 of IC50 (in molar units), i.e. pIC50 = -log10(IC50). Higher pIC50 = more potent inhibitor. It's commonly used because it produces a more convenient, roughly linear scale for QSAR/ML modeling compared to raw IC50 values.
- SMILES: A text-based notation (Simplified Molecular Input Line Entry System) used to represent a chemical compound's structure as a string, which can be parsed computationally.
- Molecular descriptors: Numerical values computed from a molecule's structure (e.g., size, polarity, atom counts, fingerprints) that are used as input features for machine learning models.
- LightGBM / Random Forest: Two machine learning algorithms used by this tool to predict pIC50 from molecular descriptors. LightGBM is a gradient-boosted decision tree method; Random Forest is an ensemble of decision trees. Both are trained on experimentally validated MDM2 inhibitor data using PaDEL-calculated descriptors.

DASHBOARD MODULES (use this when the user asks how to use a specific feature):
1) Home: Describes MDM2, its interaction with p53, and its role in cancer. Explains the importance of small-molecule MDM2 inhibitors.
2) Prediction: Predicts the pIC50 value of user-provided molecules and classifies them as MDM2 inhibitors or non-inhibitors. User can paste SMILES or upload a .smi file (up to 200 MB). Two ML models: LightGBM and Random Forest. Models are trained on PaDEL descriptors.
3) Converter: Bidirectional conversion between IC50 (in M) and pIC50 using pIC50 = -log10(IC50).
4) Dataset: Access to training, test, and external validation sets for LightGBM and Random Forest models.
5) Help: Basic instructions and video tutorial.
6) Contact: Contact information and team profiles.
"""
SYSTEM_INSTRUCTIONS = """
You are an assistant for the MDM2InPred Streamlit dashboard.
Use the APP CONTEXT to answer questions about how to use each module and general concepts (MDM2, p53, IC50, pIC50, inhibitors, SMILES, LightGBM, Random Forest).
RULES:
- Answer biology and concept questions (e.g. "what is MDM2", "how does p53 work") directly using the BIOLOGY BACKGROUND section in the APP CONTEXT. Do not redirect the user to the Home tab for conceptual questions you can already answer from the context.
- Only redirect to a specific tab when the user asks how to perform an action in the app (e.g. "how do I run a prediction", "where do I upload my file").
- If the user asks for exact internal data not in the context (e.g. specific dataset entries, exact experimental values), say you do not have direct access and ask them to check the Dataset or Prediction tab.
- Do NOT invent experimental results, exact pIC50 values, or dataset entries.
- Be clear, concise, and user-friendly.
"""

def chatbot_reply(user_text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS + "\n\nAPP CONTEXT:\n" + APP_CONTEXT},
                {"role": "user", "content": user_text}
            ],
            temperature=0.2,
            max_tokens=300
        )
        reply = response.choices[0].message.content
        reply = reply.replace("\\[", "$$").replace("\\]", "$$")
        reply = reply.replace("\\(", "$").replace("\\)", "$")
        return reply
    except Exception:
        return "Sorry, I could not generate an answer right now. Please try again later."

# ============================================================
# ── HERO HEADER ──
# ============================================================
st.markdown("""
<div class="mdm2-hero">
    </div>
    <h1 class="mdm2-title">
        MDM2InPred: Prediction of MDM2 Inhibitors
    </h1>
    <div class="mdm2-title-ornament">
    </div>
    <p class="mdm2-subtitle">A machine learning-based web server for predicting MDM2 inhibitory activity (pIC50) and<br>
              classifying compounds as inhibitors or non-inhibitors.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# ── TABS / NAVBAR ──
# ============================================================
tab_home, tab_pred, tab_data, tab_help, tab_contact, tab_chat = st.tabs([
    "Home", "Predict",
    "Dataset", "Help", "Contact", "Ask AI"
])

# ============================================================
# ── HOME TAB ──
# ============================================================
with tab_home:

    # ── Hero two-column: Mol SVG + Welcome Card ──
    st.markdown('<div class="hero-equal-height">', unsafe_allow_html=True)
    col_vis, col_welcome = st.columns([1, 1], gap="medium")

with col_vis:
    st.markdown('<div class="hero-left">', unsafe_allow_html=True)

    with open("3lbk.png", "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <div class="mol-vis-card float-anim">
        <img 
            src="data:image/png;base64,{img_data}" 
            style="
                width: 100%;
                max-width: 300px;
                height: auto;
                border-radius: 20px;
                display: block;
                margin: auto;
                filter: drop-shadow(0 12px 36px rgba(13,110,110,0.25));
            "
            alt="MDM2 Binding Pocket"
        />
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_welcome:
        st.markdown('<div class="hero-right">', unsafe_allow_html=True)
        st.markdown("""
        <div class="welcome-card">
            <h2>Welcome to MDM2InPred</h2>
            <p>
                Murine double minute 2 (MDM2) is a key negative regulator of tumor suppressor p53, is an important
                target in anticancer drug discovery. MDM2InPred, a user-friendly web server for the prediction of 
                MDM2 inhibitory activity of small molecules. It uses advanced machine learning models trained on 
                experimentally validated data. It provides both regression-based pIC50 values and binary inhibitor 
                classification.
            </p>       
        </div>
        """, unsafe_allow_html=True)

with tab_home:

    # ── Feature Cards ──
    st.markdown("""
    <div class="features-grid" style="margin-top: 24px;">
        <div class="feature-card">
            <div class="feature-title">ML-based Prediction</div>
            <div class="feature-desc">Accurate pIC₅₀ prediction using optimized LightGBM and Random Forest models</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">Easy to Use</div>
            <div class="feature-desc">Simple interface for single or batch predictions via SMILES input or file upload</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">Data Transparency</div>
            <div class="feature-desc">Access dataset information, training and validation sets for full reproducibility</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">Free for Academic Use</div>
            <div class="feature-desc">Designed for research and educational purposes at no cost</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Modules Summary ──
    st.markdown("<hr class='mdm2-divider' style='margin-top:0px; margin-bottom:10px;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header" style="margin-bottom:24px;">
        <h3>Platform Modules</h3>
        <p>Everything you need for in silico MDM2 inhibitor discovery</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class="module-desc-box">
            <div class="mod-title">Prediction Module</div>
            <div class="mod-text">
                It enables users to predict the pIC50 value of query compounds and whether they are inhibitors or
                non-inhibitors of MDM2. The user can either paste SMILES strings directly or upload a .smi file.
                LightGBM and Random Forest models are available for selection.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="module-desc-box">
            <div class="mod-title">Converter Module</div>
            <div class="mod-text">
                It enables bidirectional conversion between IC50 (in M) and pIC50 using the formula pIC50 = −log10(IC50).
                Users can convert the predicted pIC50 output to IC50 and vice versa for ease in interpretation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── How It Works ──
    st.markdown("""
    <div class="how-section">
        <div class="section-header">
            <h3>How It Works</h3>
            <p>Four simple steps from molecule to prediction</p>
        </div>
        <div class="steps-row">
            <div class="step-card">
                <div class="step-num">01</div>
                <div class="step-title">Input SMILES</div>
                <div class="step-desc">Enter your compound as a SMILES string or upload a batch .smi file</div>
            </div>
            <div class="step-card">
                <div class="step-num">02</div>
                <div class="step-title">Feature Extraction</div>
                <div class="step-desc">Molecular descriptors and fingerprints are computed automatically via PaDELPy</div>
            </div>
            <div class="step-card">
                <div class="step-num">03</div>
                <div class="step-title">ML Inference</div>
                <div class="step-desc">Random Forest model predicts pIC50 and inhibitor classification</div>
            </div>
            <div class="step-card">
                <div class="step-num">04</div>
                <div class="step-title">Download Results</div>
                <div class="step-desc">View predictions in a styled table and export as CSV</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)



# ============================================================
# ── PREDICTION TAB ──
# ============================================================
with tab_pred:
    st.markdown("""
    <div class="module-desc-box">
        <div class="mod-title">Prediction Module</div>
        <div class="mod-text">
            This module predicts the pIC50 value of query molecules and also predicts whether they are
            inhibitors or non-inhibitors of MDM2. The Random Forest (RF) machine learning algorithm
            has been implemented in the backend for prediction. The result will be visible in tabular format
            and can also be downloaded as a CSV file. For more information, please refer to the Help page.
        </div>
    </div>
    """, unsafe_allow_html=True)

    smiles_input = st.text_area("Paste SMILES string(s) — one per line:", height=120, key="smiles_input")

    st.markdown('<div class="example-pill">COC1=CC=CC=C1O</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Or upload a .smi file (up to 200 MB):", type=["smi"], key="file_upload")

    if st.button("Run Prediction"):
        run_prediction("Model 2", smiles_input, uploaded)
            
    # Inject javascript to handle 3D viewer popups
    components.html("""
    <script>
    const parentDoc = window.parent.document;
    
    // Remove existing listener to prevent duplicates on rerun
    if (window.parent._molViewListener) {
        parentDoc.removeEventListener('click', window.parent._molViewListener);
    }
    
    window.parent._molViewListener = function(e) {
        let target = e.target;
        while (target && target !== parentDoc) {
            if (target.classList && target.classList.contains('mol-view-link')) {
                e.preventDefault();
                e.stopPropagation();
                
                const sdf_b64 = target.getAttribute('data-sdf');
                const html_content = `<!DOCTYPE html>
<html>
<head>
    <title>3D Structure Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"><\\/script>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; background: #000; font-family: sans-serif; }
        #container { width: 100vw; height: 100vh; position: relative; }
        .toolbar { position: absolute; top: 10px; left: 10px; z-index: 10; background: rgba(255,255,255,0.8); padding: 5px 10px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="toolbar">Drag to rotate | Scroll to zoom</div>
    <div id="container"></div>
    <script>
        var sdf_data = atob("${sdf_b64}");
        var viewer = $3Dmol.createViewer("container", {backgroundColor: "#000"});
        viewer.addModel(sdf_data, "sdf");
        viewer.setStyle({}, {stick: {radius: 0.15}, sphere: {scale: 0.3}});
        viewer.zoomTo();
        viewer.render();
    <\\/script>
</body>
</html>`;
                
                const w = window.parent.window.open('', '_blank', 'width=800,height=600,left=200,top=100');
                if (w) {
                    w.document.write(html_content);
                    w.document.close();
                } else {
                    alert("Popup blocker prevented the 3D viewer from opening. Please allow popups for this site.");
                }
                return;
            }
            target = target.parentNode;
        }
    };
    
    parentDoc.addEventListener('click', window.parent._molViewListener);
    </script>
    """, height=0, width=0)

# (Converter tab removed as per request)

# ============================================================
# ── DATASET TAB ──
# ============================================================
with tab_data:
    st.markdown("""
    <div class="module-desc-box">
        <div class="mod-title">Dataset of MDM2InPred</div>
        <div class="mod-text">
            Training, test, and external validation sets used to develop the Random Forest model.
            Click any button to preview the dataset in an interactive table, then download it using the button below the table.
        </div>
    </div>
    """, unsafe_allow_html=True)

    models_list = ["Random Forest"]
    files = {
        "Random Forest": {
            "Training Set": "static/train_set.csv",
            "Test Set": "static/test_set.csv",
            "External Validation Set": "static/validation_set.csv"
        }
    }

    for model in models_list:
        st.markdown(f"<p style='font-weight:700;color:var(--teal-dark);font-size:1.1rem;margin:20px 0 12px;font-family:Poppins,sans-serif;'>{model} Datasets</p>", unsafe_allow_html=True)
        for dataset in ["Training Set", "Test Set", "External Validation Set"]:
            file_path = files[model][dataset]
            with st.expander(f"📂 {dataset}", expanded=False):
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    st.markdown(
                        f"<p style='font-weight:600;color:var(--teal-dark);margin-bottom:8px;"
                        f"font-family:Poppins,sans-serif;font-size:0.9rem;'>"
                        f"{model} — {dataset} &nbsp;|&nbsp; "
                        f"<span style='color:var(--gray-500);font-weight:400;'>{len(df)} rows × {len(df.columns)} columns</span></p>",
                        unsafe_allow_html=True
                    )
                    st.dataframe(df, use_container_width=True)
                    csv_data = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label=f"⬇ Download {dataset} (CSV)",
                        data=csv_data,
                        file_name=os.path.basename(file_path),
                        mime="text/csv",
                        key=f"download-{model}-{dataset}"
                    )
                else:
                    st.error(f"File not found: {file_path}")

# ============================================================
# ── HELP TAB ──
# ============================================================
with tab_help:
    st.markdown("""
    <div class="module-desc-box">
        <div class="mod-title">Help & Tutorial</div>
        <div class="mod-text">
            This dashboard is designed to provide an interactive interface for researchers to screen chemical
            compounds as potential MDM2 inhibitors and non-inhibitors virtually. It includes a prediction module
            for the prediction of MDM2 inhibitors and non-inhibitors. The following video tutorial demonstrates how to navigate the
            dashboard and access its features.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.video("video.mp4")

# ============================================================
# ── CONTACT TAB ──
# ============================================================
with tab_contact:
    def get_base64_image(image_path):
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except Exception:
            return ""

    st.markdown("""
    <div class="section-header" style="margin-bottom:28px;">
        <h3>Contact Us</h3>
        <p>Reach out to the MDM2InPred development team</p>
    </div>
    """, unsafe_allow_html=True)

    # Contact Us profiles
    c1, c2 = st.columns(2)
    with c1:
        head_img_b64 = get_base64_image("images/head.jpeg")
        img_html = f'<img src="data:image/png;base64,{head_img_b64}">' if head_img_b64 else '<div style="width:110px;height:110px;border-radius:50%;background:var(--cyan-soft);display:flex;align-items:center;justify-content:center;margin:0 auto 12px;font-size:2rem;">👤</div>'
        st.markdown(f"""
        <div class="profile-card">
            {img_html}
            <div class="profile-name">Dr. Sarfaraz Alam</div>
            <div class="profile-role">Assistant Professor</div>
            <div class="profile-role">CADDynOmics Lab</div>
            <div class="profile-role">Institute of Advanced Research, The University for Innovation, Gandhinagar</div>
            <div class="profile-email">sarfaraz.alam@iar.ac.in</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        kataria_img_b64 = get_base64_image("images/7.jpg")
        img_html_2 = f'<img src="data:image/png;base64,{kataria_img_b64}">' if kataria_img_b64 else '<div style="width:110px;height:110px;border-radius:50%;background:var(--cyan-soft);display:flex;align-items:center;justify-content:center;margin:0 auto 12px;font-size:2rem;">👤</div>'
        st.markdown(f"""
        <div class="profile-card">
            {img_html_2}
            <div class="profile-name">Dr. Bhavesh Kataria</div>
            <div class="profile-role">Associate Professor</div>
            <div class="profile-role">SCT, Institute of Advanced Research</div>
            <br>
            <div class="profile-email">bhavesh.kataria@iar.ac.in</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header" style="margin: 36px 0 24px;">
        <h3>Our Team</h3>
    </div>
    """, unsafe_allow_html=True)

    team = [
        ("images/1.jpg", "Jharnalipi Soren", "Research Scholar", "jharnalipi018@gmail.com"),
        ("images/6.jpg", "Ruturaj Kardode", "Research Scholar", "rutubioinfo@gmail.com"),
        ("images/2.jpeg", "Meet Bhayani", "Developer", "meetmbhayani@gmail.com"),
        ("images/3.jpeg", "Riya Patel", "Developer", "riya20.surat@gmail.com"),
        ("images/4.jpeg", "Pranjal Oza", "Developer", "pranjaloza7@gmail.com"),
        ("images/5.jpeg", "Raishbhai Mansuri", "Developer", "raishmansuri2003@gmail.com"),
    ]

    row1 = team[:3]
    row2 = team[3:]

    # Row 1 (3 cards)
    cols = st.columns(3)
    for col, (img, name, role, email) in zip(cols, row1):
        with col:
            img_b64 = get_base64_image(img)
            img_tag = f'<img src="data:image/png;base64,{img_b64}">' if img_b64 else '<div style="width:110px;height:110px;border-radius:50%;background:var(--cyan-soft);margin:0 auto 12px;"></div>'
            st.markdown(f"""
            <div class="profile-card">
                {img_tag}
                <div class="profile-name">{name}</div>
                <div class="profile-role">{role}</div>
                <div class="profile-email">{email}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)

    # Row 2 (3 cards)
    cols = st.columns(3)
    for col, (img, name, role, email) in zip(cols, row2):
        with col:
            img_b64 = get_base64_image(img)
            img_tag = f'<img src="data:image/png;base64,{img_b64}">' if img_b64 else '<div style="width:110px;height:110px;border-radius:50%;background:var(--cyan-soft);margin:0 auto 12px;"></div>'
            st.markdown(f"""
            <div class="profile-card">
                {img_tag}
                <div class="profile-name">{name}</div>
                <div class="profile-role">{role}</div>
                <div class="profile-email">{email}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# ── CHATBOT TAB ──
# ============================================================
with tab_chat:
    st.markdown("""
    <div class="module-desc-box" style="border-left-color:#22b8b8;">
        <div class="mod-title">MDM2InPred AI Assistant</div>
        <div class="mod-text">
            This assistant is designed to help you understand and use the MDM2InPred dashboard.
            Ask questions about the <strong>Prediction</strong> module (SMILES input, model selection, CSV output),
            the <strong>Dataset</strong> tab, or general concepts
            such as MDM2, p53, inhibitors, and Random Forest.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_chat_title, col_clear = st.columns([4, 1])
    with col_clear:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Chat history display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["text"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["text"])

    st.markdown("<hr class='mdm2-divider'>", unsafe_allow_html=True)

    user_input = st.chat_input("Type your question here: e.g. How do I use the prediction module?")
    
    if user_input:
        text = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "text": text})
        with st.chat_message("user"):
            st.markdown(text)
        
        with st.spinner("Thinking..."):
            bot_answer = chatbot_reply(text)
            
        st.session_state.chat_history.append({"role": "bot", "text": bot_answer})
        st.rerun()

#--Footer--
st.markdown("""
<div class="custom-footer">
    <div class="footer-left">
        <strong>MDM2InPred v1.0</strong>
    </div>
    <div class="footer-center">
        ©️ 2026 | All rights reserved | For academic and non-commercial use only
    </div>
</div>
""", unsafe_allow_html=True)
