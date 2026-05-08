import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import io
import base64
import datetime
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# ─────────────────────────────────────────────
# PAGE CONFIG  
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FYP_2026_Brain Tumor Detection System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS 
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --teal:      #00a896;
    --teal2:     #007d70;
    --teal-light: rgba(0,168,150,0.1);
    --teal-glow: rgba(0,168,150,0.2);
    --border:    rgba(0,168,150,0.25);
    --card-bg:   #ffffff;
    --card-border: #e2e8f0;
    --text:      #1a2332;
    --muted:     #64748b;
    --red:       #ef4444;
    --amber:     #f59e0b;
    --green:     #10b981;
    --radius:    12px;
    --font:      'Space Grotesk', sans-serif;
    --mono:      'JetBrains Mono', monospace;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: var(--font) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 4rem 2rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--card-border) !important;
    background: #f8fafc !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--teal), var(--teal2)) !important;
    color: #ffffff !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0,168,150,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,168,150,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea textarea {
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 2px var(--teal-glow) !important;
}

/* ── File uploader ── */
.stFileUploader {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
.stFileUploader:hover { border-color: var(--teal) !important; }

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius) !important;
}

/* ── Divider ── */
hr { border-color: var(--card-border) !important; margin: 1.5rem 0 !important; }

/* ── Auth tabs ── */
.auth-tab-active {
    border-bottom: 3px solid #00a896 !important;
    color: #00a896 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)
ADMIN_USERNAME = "dev"
ADMIN_PASSWORD = "dev1234"

# ─────────────────────────────────────────────
# COMPONENT HELPERS
# ─────────────────────────────────────────────

def glass_card(content_html: str, padding: str = "1.5rem"):
    return f"""
    <div style="
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: {padding};
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    ">{content_html}</div>"""


def kpi_card(icon, label, value, sub="", color="#00a896"):
    return f"""
    <div style="
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    ">
        <div style="font-size:2rem; margin-bottom:0.3rem;">{icon}</div>
        <div style="font-size:2.2rem; font-weight:700; color:{color}; font-family:'JetBrains Mono',monospace;">{value}</div>
        <div style="font-size:0.8rem; color:#64748b; margin-top:0.2rem; font-weight:600; letter-spacing:0.5px; text-transform:uppercase;">{label}</div>
        {f'<div style="font-size:0.75rem; color:#94a3b8; margin-top:0.1rem;">{sub}</div>' if sub else ""}
    </div>"""


def risk_badge(level: str):
    styles = {
        "Low":    ("background:#d1fae5;color:#065f46;border:1px solid #6ee7b7;", "✅"),
        "Medium": ("background:#fef3c7;color:#92400e;border:1px solid #fcd34d;", "⚠️"),
        "High":   ("background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;", "🔴"),
    }
    style, icon = styles.get(level, ("", "❓"))
    return f"""<span style="
        {style}
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.4px;
    ">{icon} {level} Risk</span>"""


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <h2 style="font-size:1.6rem; font-weight:700; color:#1a2332; margin:0;">
            {title}
        </h2>
        <p style="color:#64748b; margin:0.2rem 0 0; font-size:0.9rem;">
            {subtitle}
        </p>
        <div style="width:40px; height:3px; background:linear-gradient(90deg,#00a896,#007d70);
        border-radius:2px; margin-top:0.6rem;"></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HASHING UTILITY
# ─────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "neuroscan.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Patient records table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            patient_id   TEXT,
            diagnosis    TEXT,
            confidence   REAL,
            risk_level   TEXT,
            date         TEXT
        )
    """)

    # Users table (for signup)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            email         TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TEXT
        )
    """)

    con.commit()
    con.close()

def insert_record(name, pid, diagnosis, conf, risk):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO records (patient_name,patient_id,diagnosis,confidence,risk_level,date) VALUES (?,?,?,?,?,?)",
        (name, pid, diagnosis, round(conf*100,2), risk, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    con.commit()
    con.close()

def fetch_records():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM records ORDER BY id DESC", con)
    con.close()
    return df

# ── User auth helpers ──────────────────────

def signup_user(name: str, email: str, password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?,?,?,?)",
            (name.strip(), email.strip().lower(),
             hash_password(password),
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        )
        con.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    finally:
        con.close()

def login_user(email: str, password: str) -> tuple[bool, str]:
    """Returns (success, display_name)."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT name, password_hash FROM users WHERE email = ?",
        (email.strip().lower(),)
    )
    row = cur.fetchone()
    con.close()
    if row and row[1] == hash_password(password):
        return True, row[0]
    return False, ""

init_db()

# ─────────────────────────────────────────────
# MODEL UTILITIES
# ─────────────────────────────────────────────
CLASSES  = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMG_SIZE = (224, 224)

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}

    try:
        import tensorflow as tf
        import gdown
        import os

        BASE_DIR = os.path.dirname(os.path.abspath(file))
        models_dir = os.path.join(BASE_DIR, "models")

        os.makedirs(models_dir, exist_ok=True)

        cnn_path = os.path.join(models_dir, "cnn_model.h5")
        resnet_path = os.path.join(models_dir, "resnet50_model.h5")

        # -------------------------------
        # DOWNLOAD CNN MODEL IF MISSING
        # -------------------------------
        if not os.path.exists(cnn_path):
            gdown.download(
                id="1Ovkg2ezfGDqpbEjcqAsbT1-KoYsZM3IW",
                output=cnn_path,
                quiet=False,
                fuzzy=True
            )

        # -----------------------------------
        # DOWNLOAD RESNET50 MODEL IF MISSING
        # -----------------------------------
        if not os.path.exists(resnet_path):
            gdown.download(
                id="1yQXu4eRKFBcwB6af1NgIOsHFt0NqQ7eh",
                output=resnet_path,
                quiet=False,
                fuzzy=True
            )

        # -------------------
        # LOAD CNN MODEL
        # -------------------
        if os.path.exists(cnn_path):
            models["cnn"] = tf.keras.models.load_model(cnn_path)

        # -----------------------
        # LOAD RESNET50 MODEL
        # -----------------------
        if os.path.exists(resnet_path):
            models["resnet"] = tf.keras.models.load_model(resnet_path)

    except Exception as e:
        st.session_state["model_error"] = str(e)

    return models


def preprocess_image(img: Image.Image) -> np.ndarray:
    img_rgb = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def predict(models: dict, img: Image.Image):
    arr = preprocess_image(img)
    has_cnn    = "cnn"    in models
    has_resnet = "resnet" in models

    if has_cnn and has_resnet:
        p_cnn    = models["cnn"].predict(arr, verbose=0)[0]
        p_resnet = models["resnet"].predict(arr, verbose=0)[0]
        probs    = p_cnn * 0.65 + p_resnet * 0.35
    elif has_cnn:
        probs = models["cnn"].predict(arr, verbose=0)[0]
    elif has_resnet:
        probs = models["resnet"].predict(arr, verbose=0)[0]
    else:
        rng   = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(4))

    idx        = int(np.argmax(probs))
    diagnosis  = CLASSES[idx]
    confidence = float(probs[idx])
    return diagnosis, confidence, probs


def get_risk(confidence: float, diagnosis: str) -> str:
    if diagnosis == "no_tumor":
        return "Low"
    if confidence < 0.60:
        return "Low"
    if confidence <= 0.85:
        return "Medium"
    return "High"


def compute_gradcam(model, img: Image.Image):
    try:
        import tensorflow as tf
        arr = preprocess_image(img)
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D,)):
                last_conv = layer.name
                break
        if last_conv is None:
            return None
        grad_model = tf.keras.models.Model(
            inputs  = [model.inputs],
            outputs = [model.get_layer(last_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            inputs = tf.cast(arr, tf.float32)
            conv_out, preds = grad_model(inputs)
            pred_idx = tf.argmax(preds[0])
            class_channel = preds[:, pred_idx]
        grads  = tape.gradient(class_channel, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam    = conv_out[0] @ pooled[..., tf.newaxis]
        cam    = tf.squeeze(cam).numpy()
        cam    = np.maximum(cam, 0)
        cam    = cam / (cam.max() + 1e-8)
        return cam
    except Exception:
        return None


def overlay_gradcam(orig: Image.Image, cam: np.ndarray) -> Image.Image:
    cam_resized = np.array(Image.fromarray(cam).resize(orig.size, Image.BILINEAR))
    heatmap = cm.jet(cam_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    orig_rgb = np.array(orig.convert("RGB"))
    blended  = (orig_rgb * 0.55 + heatmap * 0.45).astype(np.uint8)
    return Image.fromarray(blended)

# ─────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────

def generate_pdf(name, pid, diagnosis, confidence, risk, date_str) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                     fontSize=22, textColor=colors.HexColor("#00d4b4"),
                                     spaceAfter=6)
        h2   = ParagraphStyle("H2", parent=styles["Heading2"],
                               fontSize=13, textColor=colors.HexColor("#0a1628"),
                               spaceAfter=4)
        body = styles["BodyText"]

        story.append(Paragraph("🧠 Diagnostic Report", title_style))
        story.append(Paragraph(f"Generated: {date_str}", body))
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("Patient Information", h2))

        data = [
            ["Field",        "Value"],
            ["Patient Name", name],
            ["Patient ID",   pid],
            ["Date",         date_str],
        ]
        t = Table(data, colWidths=[5*cm, 10*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#00d4b4")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTSIZE",    (0,0), (-1,-1), 11),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f4f8"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.lightgrey),
            ("PADDING",     (0,0), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("Diagnosis Summary", h2))

        risk_color = {"Low":"#06d6a0","Medium":"#ffb703","High":"#ff4d6d"}.get(risk, "#999")
        diag_data = [
            ["Diagnosis",   diagnosis.replace("_"," ").title()],
            ["Confidence",  f"{confidence*100:.1f}%"],
            ["Risk Level",  risk],
        ]
        t2 = Table(diag_data, colWidths=[5*cm, 10*cm])
        t2.setStyle(TableStyle([
            ("FONTSIZE",    (0,0), (-1,-1), 11),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f0f4f8"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.lightgrey),
            ("PADDING",     (0,0), (-1,-1), 8),
            ("TEXTCOLOR",   (1,2), (1,2), colors.HexColor(risk_color)),
            ("FONTNAME",    (1,2), (1,2), "Helvetica-Bold"),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.5*cm))

        disclaimer = ("This report is generated by MRI Insight System and is intended to assist "
                      "qualified medical professionals. It does not replace clinical judgment. "
                      "All results must be reviewed and confirmed by a licensed radiologist or neurologist.")
        story.append(Paragraph("Disclaimer", h2))
        story.append(Paragraph(disclaimer, body))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        text = (f" Diagnostic Report\n"
                f"{'='*45}\n"
                f"Patient Name : {name}\n"
                f"Patient ID   : {pid}\n"
                f"Date         : {date_str}\n"
                f"Diagnosis    : {diagnosis.replace('_',' ').title()}\n"
                f"Confidence   : {confidence*100:.1f}%\n"
                f"Risk Level   : {risk}\n")
        return text.encode()

# ─────────────────────────────────────────────
# AUTHENTICATION PAGE  (Login + Signup tabs)
# ─────────────────────────────────────────────

def auth_page():
    # ── Centered header ──────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem 0 1rem;">
        <div style="font-size:3.5rem;">🧠</div>
        <h1 style="font-size:1.8rem; font-weight:700; color:#1a2332; margin:0.3rem 0 0;
                   letter-spacing:-0.5px;">Brain Tumor Detection System</h1>
        <p style="color:#64748b; font-size:0.82rem; margin:0.3rem 0 0;
                  text-transform:uppercase; letter-spacing:1px;">
            Deep Learning-Based MRI Analysis · FYP 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tab selector stored in session ───────
    if "auth_tab" not in st.session_state:
        st.session_state["auth_tab"] = "Login"

    col_gap1, col_login, col_signup, col_gap2 = st.columns([2, 1, 1, 2])
    with col_login:
        if st.button("🔐  Login", key="tab_login"):
            st.session_state["auth_tab"] = "Login"
    with col_signup:
        if st.button("📝  Sign Up", key="tab_signup"):
            st.session_state["auth_tab"] = "Signup"

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Form area ────────────────────────────
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        if st.session_state["auth_tab"] == "Login":
            _login_form()
        else:
            _signup_form()

        st.markdown("""<p style="text-align:center;color:#94a3b8;font-size:0.72rem;margin-top:1.5rem;">
            Course: <code style='color:#00a896'>Final Year Project</code> /
            <code style='color:#00a896'>2026</code></p>""",
            unsafe_allow_html=True)


def _login_form():
    st.markdown("""
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:16px;
                padding:2rem 2rem 1.5rem;box-shadow:0 4px 20px rgba(0,0,0,0.06);">
        <p style="text-align:center;color:#64748b;font-size:0.78rem;
            text-transform:uppercase;letter-spacing:1px;margin-bottom:1.2rem;font-weight:600;">
            Sign In to Your Account</p>
    </div>
    """, unsafe_allow_html=True)

    email    = st.text_input("Email", placeholder="your@email.com", key="li_email")
    password = st.text_input("Password", placeholder="Enter password",
                             type="password", key="li_pass")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔐  Sign In", key="btn_login"):
        if not email or not password:
            st.error("Please fill in all fields.")
            return

        # ── Admin shortcut ────────────────
        if email.strip() == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state["authenticated"] = True
            st.session_state["display_name"]  = "Debobrata (Admin)"
            st.session_state["is_admin"]       = True
            st.rerun()
            return

        # ── Regular user login ────────────
        ok, name = login_user(email, password)
        if ok:
            st.session_state["authenticated"] = True
            st.session_state["display_name"]  = name
            st.session_state["is_admin"]       = False
            st.rerun()
        else:
            st.error("❌  Invalid email or password.")

    st.markdown("""
    <p style="text-align:center;color:#94a3b8;font-size:0.78rem;margin-top:0.8rem;">
        Don't have an account?
        <span style="color:#00a896;font-weight:600;">Click Sign Up above</span>
    </p>""", unsafe_allow_html=True)


def _signup_form():
    st.markdown("""
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:16px;
                padding:2rem 2rem 1.5rem;box-shadow:0 4px 20px rgba(0,0,0,0.06);">
        <p style="text-align:center;color:#64748b;font-size:0.78rem;
            text-transform:uppercase;letter-spacing:1px;margin-bottom:1.2rem;font-weight:600;">
            Create a New Account</p>
    </div>
    """, unsafe_allow_html=True)

    name     = st.text_input("Full Name",  placeholder="e.g. Debobrata Dey",    key="su_name")
    email    = st.text_input("Gmail",      placeholder="yourname@gmail.com",   key="su_email")
    password = st.text_input("Password",   placeholder="Min. 6 characters",
                             type="password", key="su_pass")
    confirm  = st.text_input("Confirm Password", placeholder="Re-enter password",
                             type="password", key="su_confirm")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("📝  Create Account", key="btn_signup"):
        if not name or not email or not password or not confirm:
            st.error("Please fill in all fields.")
            return
        if password != confirm:
            st.error("❌  Passwords do not match.")
            return
        if "@" not in email:
            st.error("❌  Please enter a valid email address.")
            return

        ok, msg = signup_user(name, email, password)
        if ok:
            st.success(f"✅  {msg} You can now log in.")
            st.session_state["auth_tab"] = "Login"
        else:
            st.error(f"❌  {msg}")

    st.markdown("""
    <p style="text-align:center;color:#94a3b8;font-size:0.78rem;margin-top:0.8rem;">
        Already have an account?
        <span style="color:#00a896;font-weight:600;">Click Login above</span>
    </p>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        display_name = st.session_state.get("display_name", "User")
        is_admin     = st.session_state.get("is_admin", False)
        badge        = " 👑" if is_admin else ""

        st.markdown(f"""
        <div style="text-align:center;padding:1.2rem 0 1.5rem;">
            <div style="font-size:2.2rem;">🧠</div>
            <div style="font-size:1.1rem;font-weight:700;color:#1a2332;letter-spacing:-0.3px;">
                MRI Analysis System</div>
            <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;">
                FYP · 2026</div>
            <div style="margin-top:0.6rem;font-size:0.82rem;color:#00a896;font-weight:600;">
                👤 {display_name}{badge}</div>
        </div>
        <hr style="border-color:#e2e8f0;margin:0 0 1rem;">
        """, unsafe_allow_html=True)

        nav_options = {
            "📊  Dashboard":       "Dashboard",
            "🔬  Diagnosis":       "Diagnosis",
            "📁  Patient History": "Patient History",
        }

        if "page" not in st.session_state:
            st.session_state["page"] = "Dashboard"

        for label, page in nav_options.items():
            active = st.session_state["page"] == page
            bg     = "rgba(0,168,150,0.1)" if active else "transparent"
            border = "#00a896"             if active else "transparent"
            color  = "#007d70"             if active else "#475569"
            st.markdown(f"""
            <a href='#' onclick='return false;' style='text-decoration:none;'>
            <div style="
                background:{bg}; border-left:3px solid {border};
                padding:0.7rem 1rem; border-radius:0 8px 8px 0;
                margin-bottom:0.3rem; color:{color};
                font-weight:{'600' if active else '400'};
                font-size:0.92rem;
            ">{label}</div></a>""", unsafe_allow_html=True)
            if st.button(label, key=f"nav_{page}",
                         help=f"Go to {page}"):
                st.session_state["page"] = page
                st.rerun()

        st.markdown("<br><hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)

        models    = st.session_state.get("models", {})
        cnn_ok    = "✅" if "cnn"    in models else "⚠️"
        resnet_ok = "✅" if "resnet" in models else "⚠️"
        st.markdown(f"""
        <div style="padding:0.8rem;background:#f8fafc;
             border-radius:10px;border:1px solid #e2e8f0;">
            <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;
                 letter-spacing:1px;margin-bottom:0.6rem;font-weight:600;">Model Status</div>
            <div style="font-size:0.82rem;color:#334155;">{cnn_ok} CNN Model Active</div>
            <div style="font-size:0.82rem;color:#334155;margin-top:0.25rem;">{resnet_ok} ResNet50 Model Active</div>
            {'<div style="font-size:0.75rem;color:#f59e0b;margin-top:0.4rem;">⚠️ Demo mode active</div>'
              if not models else ''}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔐  Logout"):
            for key in ["authenticated", "display_name", "is_admin", "models", "page", "auth_tab"]:
                st.session_state.pop(key, None)
            st.rerun()

# ─────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────

def page_dashboard():
    section_header("📊 Dashboard",
        "Real-time overview of MRI diagnostic activity")

    df = fetch_records()
    total       = len(df)
    tumor_cases = len(df[df["diagnosis"] != "no_tumor"]) if total else 0
    no_tumor    = total - tumor_cases
    today_count = len(df[df["date"].str.startswith(datetime.date.today().strftime("%Y-%m-%d"))]) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("👥","Total Patients", total), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("🔴","Tumor Cases", tumor_cases, color="#ef4444"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("✅","No Tumor",   no_tumor,    color="#10b981"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("📅","Scans Today", today_count, color="#f59e0b"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if total == 0:
        st.markdown(glass_card("""
            <div style='text-align:center;padding:2rem;'>
                <div style='font-size:3rem;'>📭</div>
                <h3 style='color:#64748b;margin:0.5rem 0 0;'>No records yet</h3>
                <p style='color:#94a3b8;'>Run your first diagnosis to populate the dashboard.</p>
            </div>"""), unsafe_allow_html=True)
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(glass_card("""<div style='font-size:0.75rem;color:#64748b;
            text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;font-weight:600;'>
            Diagnosis Distribution</div>"""), unsafe_allow_html=True)
        dist = df["diagnosis"].value_counts().reset_index()
        dist.columns = ["Diagnosis","Count"]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8fafc")
        colors_map = {"glioma":"#ef4444","meningioma":"#f59e0b",
                      "no_tumor":"#10b981","pituitary":"#00a896"}
        bar_colors = [colors_map.get(d, "#94a3b8") for d in dist["Diagnosis"]]
        bars = ax.bar(dist["Diagnosis"], dist["Count"], color=bar_colors,
                      width=0.55, edgecolor="none")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(colors="#64748b", labelsize=9)
        ax.set_xlabel("Diagnosis", color="#64748b", fontsize=9)
        ax.set_ylabel("Patients",  color="#64748b", fontsize=9)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(int(bar.get_height())), ha="center", color="#1a2332", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(glass_card("""<div style='font-size:0.75rem;color:#64748b;
            text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;font-weight:600;'>
            Risk Level Breakdown</div>"""), unsafe_allow_html=True)
        risk_dist = df["risk_level"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        fig2.patch.set_facecolor("#ffffff")
        risk_colors = [{"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"}.get(r,"#94a3b8")
                       for r in risk_dist.index]
        wedges, texts, autos = ax2.pie(
            risk_dist.values, labels=risk_dist.index,
            autopct="%1.0f%%", colors=risk_colors,
            startangle=140, pctdistance=0.8,
            wedgeprops=dict(width=0.55, edgecolor="#ffffff", linewidth=2)
        )
        for t in texts:  t.set_color("#475569"); t.set_fontsize(9)
        for a in autos:  a.set_color("#1a2332"); a.set_fontsize(8)
        ax2.set_facecolor("#ffffff")
        plt.tight_layout()
        st.pyplot(fig2, width="stretch")
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🕒 Recent Activity", "Latest patient scans")
    display_df = df.head(10)[["patient_name","patient_id","diagnosis","confidence","risk_level","date"]].copy()
    display_df.columns = ["Patient","ID","Diagnosis","Confidence (%)","Risk","Date"]
    st.dataframe(display_df, width="stretch", hide_index=True)

# ─────────────────────────────────────────────
# DIAGNOSIS PAGE
# ─────────────────────────────────────────────

def page_diagnosis():
    section_header("🔬 MRI Tumor Analysis",
        "Upload an MRI scan to detect and classify brain tumors using deep-learning")

    models = st.session_state.get("models", {})

    col_left, col_right = st.columns([1, 1.1], gap="large")

    with col_left:
        st.markdown(glass_card("""
        <div style='font-size:0.75rem;color:#7a8fa6;text-transform:uppercase;
             letter-spacing:1px;font-weight:600;margin-bottom:1rem;'>Patient Information</div>
        """), unsafe_allow_html=True)
        patient_name = st.text_input("Patient Name", placeholder="e.g. John Cena", key="p_name")
        patient_id   = st.text_input("Patient ID",   placeholder="e.g. DD-012",  key="p_id")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size:0.75rem;color:#7a8fa6;text-transform:uppercase;
             letter-spacing:1px;font-weight:600;margin-bottom:0.5rem;'>MRI Upload</div>""",
            unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop MRI image",
            type=["jpg","jpeg","png"],
            label_visibility="collapsed",
            key="mri_upload"
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded MRI Scan", width="stretch",
                     output_format="JPEG")

    with col_right:
        if uploaded and patient_name and patient_id:
            img = Image.open(uploaded)

            with st.spinner("🔄 Running ensemble inference…"):
                diagnosis, confidence, probs = predict(models, img)
            risk = get_risk(confidence, diagnosis)

            st.markdown(glass_card(f"""
            <div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;
                 letter-spacing:1px;font-weight:600;margin-bottom:1.2rem;'>Analysis Result</div>
            <div style='display:flex;align-items:center;gap:1rem;margin-bottom:1.2rem;'>
                <div>
                    <div style='font-size:0.75rem;color:#64748b;margin-bottom:0.2rem;'>Diagnosis</div>
                    <div style='font-size:1.6rem;font-weight:700;color:#1a2332;letter-spacing:-0.3px;'>
                        {diagnosis.replace("_"," ").title()}</div>
                </div>
            </div>
            <div style='display:flex;gap:1.5rem;margin-bottom:1.2rem;'>
                <div>
                    <div style='font-size:0.7rem;color:#64748b;margin-bottom:0.3rem;'>Confidence</div>
                    <div style='font-size:1.3rem;font-weight:700;color:#00a896;
                         font-family:"JetBrains Mono",monospace;'>{confidence*100:.1f}%</div>
                </div>
                <div>
                    <div style='font-size:0.7rem;color:#64748b;margin-bottom:0.25rem;'>Risk Level</div>
                    {risk_badge(risk)}
                </div>
            </div>
            """), unsafe_allow_html=True)

            st.markdown("""<div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;
                 letter-spacing:1px;font-weight:600;margin:0.5rem 0 0.8rem;'>
                 Class Probability</div>""", unsafe_allow_html=True)
            class_colors = {"glioma":"#ef4444","meningioma":"#f59e0b",
                            "no_tumor":"#10b981","pituitary":"#00a896"}
            for cls, prob in zip(CLASSES, probs):
                pct   = prob * 100
                color = class_colors.get(cls, "#94a3b8")
                st.markdown(f"""
                <div style='margin-bottom:0.6rem;'>
                    <div style='display:flex;justify-content:space-between;
                         font-size:0.78rem;color:#475569;margin-bottom:0.2rem;'>
                        <span>{cls.replace("_"," ").title()}</span>
                        <span style='font-family:"JetBrains Mono",monospace;color:{color};font-weight:600;'>
                            {pct:.1f}%</span>
                    </div>
                    <div style='background:#f1f5f9;border-radius:50px;height:6px;'>
                        <div style='width:{pct}%;height:100%;background:{color};
                             border-radius:50px;'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            record_key = f"saved_{patient_id}_{diagnosis}"
            if record_key not in st.session_state:
                insert_record(patient_name, patient_id, diagnosis, confidence, risk)
                st.session_state[record_key] = True

            st.markdown("<br>", unsafe_allow_html=True)
            date_str  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            pdf_bytes = generate_pdf(patient_name, patient_id, diagnosis, confidence, risk, date_str)
            ext  = "pdf" if pdf_bytes[:4] == b"%PDF" else "txt"
            mime = "application/pdf" if ext == "pdf" else "text/plain"
            st.download_button(
                label="📄  Download PDF Report",
                data=pdf_bytes,
                file_name=f"neuroscan_{patient_id}_{datetime.date.today()}.{ext}",
                mime=mime,
            )

        elif uploaded and not (patient_name and patient_id):
            st.markdown(glass_card("""
            <div style='text-align:center;padding:1.5rem;'>
                <div style='font-size:2.5rem;'>📋</div>
                <p style='color:#64748b;margin:0.5rem 0 0;'>
                    Please fill in Patient Name and ID to proceed.</p>
            </div>"""), unsafe_allow_html=True)
        else:
            st.markdown(glass_card("""
            <div style='text-align:center;padding:3rem 1rem;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>🩻</div>
                <h3 style='color:#64748b;margin:0;font-weight:500;'>🧠 Scan Preview Panel</h3>
                <p style='color:#94a3b8;font-size:0.85rem;margin:0.4rem 0 0;'>
                    No MRI uploaded yet.
                    Upload an image to view prediction results, confidence scores, and risk level.</p>
            </div>"""), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PATIENT HISTORY PAGE
# ─────────────────────────────────────────────

def page_history():
    section_header("📁 Patient Diagnosis History",
        "View, analyze, and manage patient MRI diagnosis records")

    df = fetch_records()

    if df.empty:
        st.markdown(glass_card("""
        <div style='text-align:center;padding:2.5rem;'>
            <div style='font-size:3rem;'>📭</div>
            <h3 style='color:#7a8fa6;margin:0.5rem 0 0;'>No records found</h3>
            <p style='color:#4a5a6a;'>Records will appear here after diagnoses are made.</p>
        </div>"""), unsafe_allow_html=True)
        return

    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        all_diag    = ["All"] + sorted(df["diagnosis"].unique().tolist())
        chosen_diag = st.selectbox("Filter by Diagnosis", all_diag)
    with col_f2:
        all_risk    = ["All"] + sorted(df["risk_level"].unique().tolist())
        chosen_risk = st.selectbox("Filter by Risk Level", all_risk)

    filtered = df.copy()
    if chosen_diag != "All": filtered = filtered[filtered["diagnosis"] == chosen_diag]
    if chosen_risk != "All": filtered = filtered[filtered["risk_level"] == chosen_risk]

    st.markdown("<br>", unsafe_allow_html=True)

    csv = filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download Records (CSV)",
        data=csv,
        file_name=f"neuroscan_records_{datetime.date.today()}.csv",
        mime="text/csv",
    )

    display = filtered[["patient_name","patient_id","diagnosis","confidence","risk_level","date"]].copy()
    display.columns = ["Patient","ID","Diagnosis","Confidence (%)","Risk","Date"]
    st.dataframe(display, width="stretch", hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    colors_map = {"glioma":"#ef4444","meningioma":"#f59e0b",
                  "no_tumor":"#10b981","pituitary":"#00a896"}
    with col1:
        st.markdown("""<div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;
             letter-spacing:1px;font-weight:600;margin-bottom:0.8rem;'>
             Diagnosis Pie Chart</div>""", unsafe_allow_html=True)
        dist = filtered["diagnosis"].value_counts()
        fig, ax = plt.subplots(figsize=(4.5, 3))
        fig.patch.set_facecolor("#ffffff")
        pie_colors = [colors_map.get(d,"#94a3b8") for d in dist.index]
        wedges, texts, autos = ax.pie(
            dist.values, labels=dist.index,
            autopct="%1.0f%%", colors=pie_colors,
            startangle=140, pctdistance=0.78,
            wedgeprops=dict(width=0.6, edgecolor="#ffffff", linewidth=2)
        )
        for t in texts:  t.set_color("#475569"); t.set_fontsize(9)
        for a in autos:  a.set_color("#1a2332"); a.set_fontsize(8)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close()

    with col2:
        st.markdown("""<div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;
             letter-spacing:1px;font-weight:600;margin-bottom:0.8rem;'>
             Avg. Confidence per Diagnosis</div>""", unsafe_allow_html=True)
        avg_conf = filtered.groupby("diagnosis")["confidence"].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(4.5, 3))
        fig2.patch.set_facecolor("#ffffff")
        ax2.set_facecolor("#f8fafc")
        bar_colors2 = [colors_map.get(d,"#94a3b8") for d in avg_conf["diagnosis"]]
        ax2.barh(avg_conf["diagnosis"], avg_conf["confidence"],
                 color=bar_colors2, edgecolor="none", height=0.5)
        for spine in ax2.spines.values(): spine.set_visible(False)
        ax2.tick_params(colors="#64748b", labelsize=9)
        ax2.set_xlabel("Avg Confidence (%)", color="#64748b", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2, width="stretch")
        plt.close()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        auth_page()
        return

    if "models" not in st.session_state:
        with st.spinner("Loading models…"):
            st.session_state["models"] = load_models()

    render_sidebar()

    page = st.session_state.get("page", "Dashboard")
    if page == "Dashboard":
        page_dashboard()
    elif page == "Diagnosis":
        page_diagnosis()
    elif page == "Patient History":
        page_history()


if __name__ == "__main__":
    main()
