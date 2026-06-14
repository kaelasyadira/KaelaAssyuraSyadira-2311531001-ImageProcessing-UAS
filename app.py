import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import time
from tensorflow.keras import layers, models
from PIL import Image, ImageDraw, ImageFont

# --- CONFIG HALAMAN ---
st.set_page_config(page_title="KEMOVA - AI Emotion Detection", page_icon="🔮", layout="wide")

# --- INISIALISASI SESSION STATE UNTUK TEMA ---
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
    
if "emotion_counts" not in st.session_state:
    st.session_state["emotion_counts"] = {e: 0 for e in ['Marah', 'Jijik', 'Takut', 'Bahagia', 'Sedih', 'Terkejut', 'Netral']}

# --- GAYA KUSTOM BARU (Earthy & Elegant Palette dengan Dukungan Dark Mode) ───
def load_custom_style():
    if st.session_state["dark_mode"]:
        # Palet Dark Mode (Elegant Night)
        bg_main = "#1E1A1A"
        bg_section = "#2D2426"
        text_dark = "#EAE3E3"
        text_light = "#A89A9A"
        border_soft = "#443538"
        chalk = "#3D3234"
        status_box_bg = "#231B1C" 
        status_box_border = "transparent"
    else:
        # Palet Light Mode (Earthy & Elegant)
        bg_main = "#F6EFE8"
        bg_section = "#EFE2D6"
        text_dark = "#3D2F2F"
        text_light = "#7C7065"
        border_soft = "#DCCDBF"
        chalk = "#E3D6BF"
        status_box_bg = "#F6EFE8"
        status_box_border = "#E3D6BF"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght=400;500;600;700&family=Manrope:wght=300;400;500;600&display=swap');

        /* ── VARIABLE WARNA DASAR ── */
        :root {{
            --amaranth: #933B5B;
            --thulian-pink: #B5728A;
            --brook-green: #AABAAE;
            --chalk: {chalk};
            --pomelo-olive: #9F9679;
            --bg-main: {bg_main};
            --bg-section: {bg_section};
            --text-dark: {text_dark};
            --text-light: {text_light};
            --border-soft: {border_soft};
        }}

        /* ── CANVAS UTAMA ── */
        html, body, .stApp {{
            background-color: var(--bg-main) !important;
            color: var(--text-dark) !important;
            font-family: 'Manrope', sans-serif !important;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}

        /* Container Layout */
        section.main .block-container {{
            padding: 3rem 4rem 4rem !important;
            max-width: 1140px !important;
        }}

        /* Typography Global Overrides */
        h1, h2, h3, h4 {{
            font-family: 'Cormorant Garamond', serif !important;
            font-weight: 700 !important;
            color: var(--amaranth) !important;
        }}
        h1 {{ 
            font-size: 4rem !important; 
            background: none !important;
            -webkit-text-fill-color: var(--amaranth) !important;
        }}
        h2 {{ font-size: 3rem !important; }}
        h3 {{ font-size: 2.2rem !important; color: var(--text-dark) !important;}}
        p, span, label {{ font-family: 'Manrope', sans-serif !important; color: var(--text-light) !important; }}

        /* ── SIDEBAR NAVIGASI & LOCK BOTTOM FIX ── */
        section[data-testid='stSidebar'] {{
            background-color: var(--bg-section) !important;
            border-right: 1px solid var(--border-soft) !important;
            transition: background-color 0.3s ease;
        }}
        
        /* Memaksa container utama di dalam sidebar menggunakan full height layout */
        section[data-testid='stSidebar'] > div:first-child {{
            padding: 2rem 1.2rem !important;
        }}
        
        /* Mengunci posisi block atas dan bawah sidebar */
        section[data-testid='stSidebar'] .stElementContainer {{
            margin-bottom: 0px !important;
        }}
        
        section[data-testid='stSidebar'] div.stVerticalBlockInsidePage {{
            height: calc(100vh - 3.5rem) !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
        }}

        .sidebar-brand-name {{
            color: var(--amaranth) !important;
            font-family: 'Cormorant Garamond', serif;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-soft);
            text-align: center;
            letter-spacing: 0.05em;
        }}

        /* Tombol Menu Navigasi Sidebar */
        .stRadio > div {{
            gap: 0.5rem !important;
        }}
        .stRadio > div > label {{
            display: flex !important;
            align-items: center !important;
            padding: 0.85rem 1.2rem !important;
            border-radius: 12px !important;
            color: var(--text-dark) !important; 
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            background: transparent !important;
            border: none !important;
        }}
        .stRadio > div > label:hover {{
            background: rgba(147, 59, 91, 0.05) !important; 
            color: var(--amaranth) !important;
        }}
        .stRadio > div > label[data-baseweb="radio"] input:checked + span {{
            color: var(--amaranth) !important;
            font-weight: 600;
        }}
        .stRadio > div [data-testid="stWidgetLabel"] + div div[data-checked="true"] {{
            background: var(--chalk) !important;
            border-radius: 12px !important;
        }}

        /* ── FIX KONTRAST TEKS HALAMAN ANALISIS PERFORMA (DARK MODE) ── */
        /* Warna Label Metric */
        [data-testid="metric-container"] label p {{
            color: var(--text-light) !important;
        }}
        /* Warna Nilai Angka/Teks Utama Metric */
        [data-testid="metric-container"] div[data-testid="stMetricValue"] > div {{
            color: var(--amaranth) !important;
            font-family: 'Cormorant Garamond', serif !important;
            font-size: 2.2rem !important;
            font-weight: 700;
        }}
        /* Kontras Teks Radio Button Pilihan Emosi di Main Page */
        section.main div.stRadio label p {{
            color: var(--text-dark) !important;
            font-weight: 500 !important;
        }}

        /* ── TOGGLE DARK MODE SIDEBAR AT BOTTOM ── */
        div.toggle-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1rem;
            padding: 0 0.5rem;
        }}
        
        div[data-testid="stSidebar"] div[data-testid="stCheckbox"] label p {{
            display: none !important;
        }}
        div[data-testid="stSidebar"] div[data-testid="stCheckbox"] {{
            margin-bottom: 0 !important;
            padding: 0 !important;
        }}

        /* ── HERO CONTAINER ── */
        .exhibit-hero-card {{
            background: var(--bg-section);
            border-radius: 40px;
            padding: 60px;
            margin-bottom: 2.5rem;
            transition: background-color 0.3s ease;
        }}
        .exhibit-hero-card h1 {{
            color: var(--amaranth) !important;
            font-size: 4rem !important;
            margin-bottom: 0.75rem;
        }}
        .exhibit-hero-card p {{
            color: var(--text-light) !important;
        }}

        .exhibit-badge {{
            display: inline-block;
            background: var(--amaranth);
            color: #FFFFFF; 
            padding: 0.35rem 0.85rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }}

        /* ── GLOBAL BUTTON DESIGNS ── */
        div.stButton > button {{
            width: 100% !important;
            background: transparent !important;
            color: #A03F63 !important;
            border: 2px solid #A03F63 !important;
            border-radius: 10px !important;
            padding: 18px 10px !important;
            font-size: 14px !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            white-space: nowrap !important;
            transition: all 0.15s ease-in-out !important;
        }}
        div.stButton > button:hover {{
            background: #A03F63 !important;
            color: white !important;
        }}

        /* ── EMOTION GRID ── */
        .emotion-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
            margin-top: 0.75rem;
            margin-bottom: 3rem;
        }}
        .emotion-grid-card {{
            background: var(--bg-section);
            border: none;
            border-radius: 120px 120px 0 0;
            padding: 30px 20px;
            text-align: center;
            transition: all 0.2s ease;
        }}
        .emotion-grid-card:hover {{
            transform: translateY(-5px);
            background: var(--chalk);
        }}
        .emotion-grid-icon {{ font-size: 1.8rem; margin-bottom: 0.4rem; }}
        .emotion-grid-lbl {{ font-size: 0.85rem; font-weight: 600; color: var(--text-dark); }}

        /* ── STANDAR CARDS ── */
        .glass-card {{
            background: var(--bg-section);
            border: 1px solid var(--border-soft);
            border-radius: 25px;
            padding: 2rem;
            height: 100%;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }}
        .glass-card h4 {{
            color: var(--amaranth) !important;
            font-size: 1.2rem !important;
            margin-bottom: 1rem !important;
            font-weight: 700;
        }}
        .glass-card p {{
            color: var(--text-dark) !important;
        }}

        hr {{ border-color: var(--border-soft) !important; margin: 2.5rem 0 !important; }}
        #MainMenu, footer, header {{ 
            visibility: hidden !important; 
        }}

        /* ── BAGIAN PERBAIKAN TEKS UPLOAD GAMBAR ── */
        div[data-testid="stFileUploader"] section {{
            background-color: var(--bg-section) !important;
            border: 2px dashed var(--border-soft) !important;
        }}
        div[data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] p {{
            color: var(--text-dark) !important; 
            font-weight: 500;
        }}
        div[data-testid="stFileUploader"] small {{
            color: var(--text-light) !important; 
        }}
        div[data-testid="stSidebarCollapseButton"] {{
            display: none !important;
            visibility: hidden !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return status_box_bg, status_box_border

# --- ENGINE MODEL MACHINE LEARNING ---
def build_local_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(7, activation='softmax')
    ])
    return model

@st.cache_resource
def load_my_model():
    local_model = build_local_model()
    try:
        local_model.load_weights('emotion2_model_weights.weights.h5')
    except:
        pass
    return local_model

model = load_my_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EMOTIONS = ['Marah', 'Jijik', 'Takut', 'Bahagia', 'Sedih', 'Terkejut', 'Netral']

# Jalankan Desain Kustom Baru & Ambil Variabel Warna Box Dinamis
status_box_bg, status_box_border = load_custom_style()

# ── SIDEBAR NAVIGATION & LOWER CONTROL CONTAINER ──
list_halaman = ["🏠     Beranda Utama", "🔮     Mulai Deteksi Ekspresi", "📊     Analisis Performa"]

if "page" not in st.session_state:
    st.session_state["page"] = "🏠     Beranda Utama"

if "total_tests" not in st.session_state:
    st.session_state["total_tests"] = 0

index_halaman = list_halaman.index(st.session_state["page"])

with st.sidebar:
    st.markdown("<div>", unsafe_allow_html=True)
    
    try:
        img_logo = Image.open("logo_kemova.png").convert("RGBA")
        size = (105, 105)
        img_logo = img_logo.resize(size, Image.Resampling.LANCZOS)
        
        mask = Image.new("L", size, 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.ellipse((0, 0) + size, fill=255)
        
        output_logo = Image.new("RGBA", size, (0, 0, 0, 0))
        output_logo.paste(img_logo, (0, 0), mask=mask)
        
        st.markdown('<div style="display: flex; justify-content: center; margin-top: -15px; margin-bottom: -5px;">', unsafe_allow_html=True)
        st.image(output_logo, width=105)
        st.markdown('</div>', unsafe_allow_html=True)
            
    except:
        pass
        
    st.markdown('<div class="sidebar-brand-name" style="margin-top: -5px; margin-bottom: 1rem; padding-bottom: 0.5rem;">KEMOVA</div>', unsafe_allow_html=True)

    page = st.radio(
        "NAVIGASI",
        list_halaman,
        index=index_halaman,
        label_visibility="collapsed"
    )
    st.session_state["page"] = page
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="padding: 0.8rem 1rem; border: 1px solid {status_box_border}; border-radius: 12px; background-color: {status_box_bg}; margin-bottom: 0.25rem;">
            <p style="font-size: 0.62rem; color: #A89A9A; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.2rem; margin-top: 0;">STATUS SISTEM</p>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 6px; height: 7px; border-radius: 50%; background: #933B5B; box-shadow: 0 0 6px #933B5B;"></div>
                <span style="font-size: 0.85rem; color: {'#EAE3E3' if st.session_state['dark_mode'] else '#3D2F2F'}; font-weight: 600;">CNN Model Active</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col_lbl, col_sw = st.columns([3, 1])
    with col_lbl:
        st.markdown("""
            <div style="height: 100%; display: flex; align-items: center; padding-top: 0.3rem; padding-left: 0.3rem;">
                <span style="font-size: 0.9rem; font-weight: 500; color: var(--text-dark);">🌙 &nbsp;Dark Mode</span>
            </div>
        """, unsafe_allow_html=True)
    with col_sw:
        dark_toggle = st.toggle("dark_mode_trigger", value=st.session_state["dark_mode"], label_visibility="collapsed")
        if dark_toggle != st.session_state["dark_mode"]:
            st.session_state["dark_mode"] = dark_toggle
            st.rerun()
            
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════
# 🏠 BERANDA UTAMA (Halaman Utama)
# ══════════════════════════════════════════
if page == "🏠     Beranda Utama":
    with st.container():
        st.markdown("""
            <div style="background: var(--bg-section); padding: 80px; border-radius: 300px 300px 0 0; text-align: center; margin-bottom: 2.5rem;">
                <h1>KEMOVA</h1>
                <p style="color: var(--text-dark) !important; font-size: 1.2rem; font-weight: 500;">Deteksi Ekspresi Wajah Berbasis CNN</p>
                <p style="font-family: 'Cormorant Garamond', serif !important; font-style: italic; font-size: 1.3rem !important; color: var(--amaranth) !important; font-weight: 700 !important;">-Kaela EMOtion & Vision AI-</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="exhibit-hero-card">
                <div class="exhibit-badge">Proyek Akhir Image Processing  ·  INFORMATIKA</div>
                <h1>Ekspresimu Nyata, KEMOVA Mendeteksi</h1>
                <p style="max-width:750px; margin-bottom: 2rem;">
                    Hadapkan wajahmu ke kamera atau upload foto terbaikmu, KEMOVA akan memberitahu emosi apa yang terdeteksi — Apakah Bahagia, Terkejut, Netral, Sedih, Marah, Takut, atau Jijik?
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1.5, 1, 1.5])
        with c2:
            st.markdown("""
            <style>
            div.stButton > button[kind="secondary"] {
                width: 100% !important;
                background: transparent !important;
                color: #A03F63 !important;
                border: 2px solid #A03F63 !important;
                border-radius: 10px !important;
                padding: 18px 10px !important;
                font-size: 14px !important;
                font-weight: 700 !important;
                text-transform: uppercase !important;
                white-space: nowrap !important;
            }
            div.stButton > button[kind="secondary"]:hover {
                background: #A03F63 !important;
                color: white !important;
            }
            </style>
            """, unsafe_allow_html=True)

            if st.button("📷 COBA UJI EKSPRESI →", key="btn_exhibit_cta", type="secondary"):
                st.session_state["page"] = "🔮     Mulai Deteksi Ekspresi"
                st.rerun()

    st.markdown("""<hr style="border: none; border-top: 1px solid var(--border-soft); margin: 1.5rem 0;">""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3>🎯 Spektrum Emosi yang Dikenali</h3>', unsafe_allow_html=True)
    st.markdown("""
        <div class="emotion-grid">
            <div class="emotion-grid-card"><div class="emotion-grid-icon">😄</div><div class="emotion-grid-lbl">Bahagia</div></div>
            <div class="emotion-grid-card"><div class="emotion-grid-icon">😲</div><div class="emotion-grid-lbl">Terkejut</div></div>
            <div class="emotion-grid-card"><div class="emotion-grid-icon">😐</div><div class="emotion-grid-lbl">Netral</div></div>
            <div class="emotion-grid-card"><div class="emotion-grid-icon">😢</div><div class="emotion-grid-lbl">Sedih</div></div>
            <div class="emotion-grid-card"><div class="emotion-grid-icon">😠</div><div class="emotion-grid-lbl">Marah</div></div>
            <div class="emotion-grid-card"><div class="emotion-grid-icon">😨</div><div class="emotion-grid-lbl">Takut</div></div>
            <div class="emotion-grid-card"><div class="emotion-grid-icon">🤢</div><div class="emotion-grid-lbl">Jijik</div></div>
        </div>
    """, unsafe_allow_html=True)

    col_dev, col_project = st.columns([2, 3], gap='large')
    with col_dev:
        st.markdown(f"""
            <div class="glass-card">
                <h4>👤 Identitas Mahasiswa</h4>
                <p style="font-size:1.2rem; font-weight:700; color: var(--amaranth) !important; margin-bottom:0.1rem; font-family: 'Cormorant Garamond', serif !important;">Kaela Assyura Syadira</p>
                <p style="font-size:0.9rem!important; color: var(--thulian-pink)!important; font-weight:600!important; margin-bottom:1.25rem!important;">NIM 2311531001</p>
                <div style="display:flex; flex-direction:column; gap:0.75rem; border-top: 1px solid var(--border-soft); padding-top:1rem;">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span style="color: var(--text-light);">Jurusan</span>
                        <span style="color: var(--text-dark); font-weight:600;">Informatika</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span style="color: var(--text-light);">Fakultas</span>
                        <span style="color: var(--text-dark); font-weight:600;">Teknologi Informasi</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span style="color: var(--text-light);">Universitas</span>
                        <span style="color: var(--text-dark); font-weight:600;">Universitas Andalas</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_project:
        st.markdown("""
            <div class="glass-card">
                <h4>🔍 Cara Kerja KEMOVA</h4>
                <p style="font-size:0.88rem; line-height:1.65; color: var(--text-dark) !important;">
                    Sistem mendeteksi area wajah dari gambar atau webcam menggunakan OpenCV Haar Cascade. Kemudian model CNN mengklasifikasikan ekspresi ke dalam 7 kategori emosi lengkap dengan confidence score-nya. Model ini dilatih menggunakan gabungan dataset FER dan AffectNet dengan input citra grayscale berukuran 48×48 piksel.
                </p>
            </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════
# 🔮 HALAMAN PREDIKSI
# ══════════════════════════════════════════
elif page == "🔮     Mulai Deteksi Ekspresi":
    st.markdown("""
        <h2>Kamera Penguji Ekspresi</h2>
        <p style="margin-bottom:2rem;">Silakan pilih metode input untuk melakukan inferensi model Computer Vision.</p>
    """, unsafe_allow_html=True)

    col_input_center, col_spacer = st.columns([2, 1])
    with col_input_center:
        input_option = st.radio(
            "Metode Input Gambar",
            ["📸     Gunakan Live Webcam Kamera", "🖼️     Upload Berkas Foto"],
            horizontal=True,
            label_visibility="collapsed"
        )
    st.markdown("<br>", unsafe_allow_html=True)

    img_array = None

    col_cam_left, col_cam_main, col_cam_right = st.columns([0.5, 5, 0.5])
    with col_cam_main:
        if "Kamera" in input_option:
            img_file = st.camera_input("")
            if img_file is not None:
                file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                img_array = cv2.imdecode(file_bytes, 1)
        else:
            uploaded_file = st.file_uploader(
                "Pilih atau drop berkas gambar Anda ke sini (.jpg, .png)",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_array = cv2.imdecode(file_bytes, 1)

    if img_array is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        col_img, col_res = st.columns([1, 1], gap='large')

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        display_img = img_array.copy()

        if len(faces) > 0:
            face_emotions = []
            start_time = time.time()

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype("float32") / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)

                prediction = model.predict(roi_gray)
                max_index = int(np.argmax(prediction))
                predicted_emotion = EMOTIONS[max_index]
                confidence_score = float(prediction[0][max_index] * 100)
                face_emotions.append((predicted_emotion, confidence_score, prediction[0]))

                cv2.rectangle(display_img, (x, y), (x+w, y+h), (91, 59, 147), 3)
                cv2.putText(display_img, f"{predicted_emotion} ({confidence_score:.1f}%)", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (91, 59, 147), 2)

            inference_time = (time.time() - start_time) * 1000
            best_emotion, best_confidence, best_probs = max(face_emotions, key=lambda item: item[1])

            st.session_state['last_inference'] = inference_time
            st.session_state['last_prediction'] = best_emotion
            st.session_state['last_confidence'] = best_confidence
            st.session_state['total_tests'] = st.session_state.get('total_tests', 0) + 1
            st.session_state['emotion_counts'][best_emotion] += 1

            h_orig, w_orig, _ = img_array.shape
            canvas_info = np.zeros((220, w_orig, 3), dtype=np.uint8)
            canvas_info.fill(245)

            pil_img = Image.fromarray(cv2.cvtColor(canvas_info, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # --- LOGIKA FONT ADAPTIF SERVER LINUX ---
            try:
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except IOError:
                try:
                    font_title = ImageFont.truetype("arialbd.ttf", 20)
                    font_subtitle = ImageFont.truetype("arialbd.ttf", 16)
                    font_body = ImageFont.truetype("arial.ttf", 14)
                except IOError:
                    font_title = font_subtitle = font_body = ImageFont.load_default()

            draw.text((20, 25), "KEMOVA — EMOTION REPORT", font=font_title, fill=(147, 59, 91))
            draw.text((20, 65), f"Hasil Utama: {best_emotion} ({best_confidence:.2f}%)", font=font_subtitle, fill=(61, 47, 47))
            
            y_offset = 115
            x_offset = 20
            for idx, emo in enumerate(EMOTIONS):
                prob_val = float(best_probs[idx] * 100)
                text_emo = f"• {emo}: {prob_val:.1f}%"
                draw.text((x_offset, y_offset), text_emo, font=font_body, fill=(124, 112, 101))
                x_offset += 140
                if (idx + 1) % 4 == 0:
                    y_offset += 35
                    x_offset = 20

            canvas_info_clean = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            download_ready_img = np.vstack((display_img, canvas_info_clean))
            _, img_encoded = cv2.imencode('.jpg', download_ready_img)
            img_bytes = img_encoded.tobytes()

            with col_img:
                st.markdown('<p style="font-size:0.75rem; font-weight:700; text-transform:uppercase; margin-bottom:0.75rem; margin-top:0px; color:var(--text-dark);">⚡ Hasil Deteksi (Bounding Box)</p>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col_res:
                st.markdown('<p style="font-size:0.75rem; font-weight:700; text-transform:uppercase; margin-bottom:0.75rem; margin-top:0px; color:var(--text-dark);">📊 Hasil Analisis & Score</p>', unsafe_allow_html=True)
                st.markdown(f"""
                    <div style="background: var(--bg-section); border: 1px solid var(--border-soft); border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <p style="font-size:0.7rem; color: var(--text-light); font-weight:700; text-transform:uppercase; margin:0;">Emosi Terdeteksi</p>
                            <p style="font-size:1.5rem; font-weight:700; color: var(--amaranth); margin:0; margin-top:0.1rem;">{best_emotion}</p>
                        </div>
                        <div style="text-align: right;">
                            <p style="font-size:0.7rem; color: var(--text-light); font-weight:700; text-transform:uppercase; margin:0;">Confidence Score</p>
                            <p style="font-size:1.5rem; font-weight:700; color: var(--thulian-pink); margin:0; margin-top:0.1rem;">{best_confidence:.2f}%</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                df_probs = pd.DataFrame({
                    'Emosi': EMOTIONS,
                    'Probabilitas (%)': [float(p * 100) for p in best_probs]
                }).sort_values('Probabilitas (%)', ascending=False)
                
                st.dataframe(
                    df_probs.set_index('Emosi'), 
                    use_container_width=True,
                    column_config={
                        "Probabilitas (%)": st.column_config.ProgressColumn(
                            "Probabilitas (%)",
                            help="Persentase kecocokan ekspresi wajah",
                            format="%.2f %%",
                            min_value=0,
                            max_value=100,
                        )
                    }
                )

            pesan_emosi = {
                'Bahagia': "✨ Senyummu terpancar! Bahagia Selalu 😄",
                'Sedih': "It's OK, waktunya sedih dulu, tapi jangan lama-lama yah!💙 ",
                'Marah': "😤 Tarik napas dulu yah...semuanya pasti bisa diselesaikan.",
                'Terkejut': "😲 Wow, ada apa nih?",
                'Takut': "🤗 Gapapa, rasa takut itu wajar. Semangat Melewatinya!",
                'Jijik': "🤢 Sepertinya ada yang kurang menyenangkan hari ini!",
                'Netral': "😐 Tenang.. itulah kamu hari ini.",
            }

            st.markdown(f"""
                <div style="background: var(--bg-section); border-left: 3px solid var(--amaranth); 
                border-radius: 8px; padding: 1rem 1.25rem; margin-top: 1rem;">
                    <p style="font-size:1.20rem; color: var(--text-dark) !important; margin:0; font-style:italic;">
                        {pesan_emosi[best_emotion]}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # ── MASUK KE LEVEL INDENTASI UTAMA (FULL-WIDTH DI BAWAH KEDUA GAMBAR) ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="📥 UNDUH LAPORAN HASIL DETEKSI (.JPG)",
                data=img_bytes,
                file_name=f"KEMOVA_{best_emotion}_{int(best_confidence)}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📊 LIHAT ANALISIS PERFORMA →", key="btn_to_dashboard"):
                st.session_state["page"] = "📊     Analisis Performa"
                st.rerun()
                
        else:
            st.markdown("""
                <div style="background: var(--bg-main); border: 1px dashed var(--border-soft); border-radius: 12px; padding: 2.5rem; text-align: center; margin-top:0.5rem;">
                    <h5 style="color: var(--text-dark); font-weight:700; margin-bottom:0.15rem; font-size:0.95rem; font-family:'Cormorant Garamond', serif;">Wajah Belum Terdeteksi</h5>
                    <p style="color: var(--text-light); font-size:0.85rem; max-width:320px; margin: 0 auto;">Posisikan wajah menghadap lurus ke arah kamera lensa stan pameran.</p>
                </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════
# 📊 DASHBOARD STATISTIK
# ══════════════════════════════════════════
elif page == "📊     Analisis Performa":
    st.markdown("""
        <h2>Dashboard Performa & Evaluasi</h2>
        <p style="margin-bottom:2rem;">Metrik Statistik Performa Model Akurasi.</p>
    """, unsafe_allow_html=True)

    total_uji = st.session_state.get('total_tests', 0)
    waktu_inf = st.session_state.get('last_inference', 0.0)

    c1, c2, c3 = st.columns(3, gap='large')
    c1.metric("Jumlah Data Diuji", f"{total_uji} Sesi")
    c2.metric("Akurasi Model (Avg)", "64.0 %")
    m3_val = f"{waktu_inf:.1f} ms" if waktu_inf > 0 else "0.0 ms"
    c3.metric("Waktu Inferensi", m3_val)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Grafik distribusi emosi terdeteksi
    emotion_df = pd.DataFrame({
        'Emosi': list(st.session_state['emotion_counts'].keys()),
        'Jumlah Terdeteksi': list(st.session_state['emotion_counts'].values())
    })

    st.markdown("""
        <p style="font-size:0.75rem; color: var(--text-dark); font-weight:700; text-transform:uppercase; margin-bottom:0.5rem;">
        🎯 Distribusi Emosi yang Terdeteksi Selama Sesi</p>
    """, unsafe_allow_html=True)

    fig_dist = px.bar(
        emotion_df, 
        x='Emosi', 
        y='Jumlah Terdeteksi', 
        color_discrete_sequence=["#B5728A"],
        height=280
    )
    fig_dist.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<hr>", unsafe_allow_html=True)

    acc_data = pd.DataFrame({
        'Kelas Emosi': ['Bahagia', 'Terkejut', 'Netral', 'Sedih', 'Marah', 'Takut', 'Jijik'],
        'Akurasi (%)': [79, 69, 66, 61, 58, 50, 45]
    }).sort_values('Akurasi (%)', ascending=True)

    analisis_emosi = {
        'Bahagia': {
            'icon': '😄',
            'akurasi': 79,
            'teks': 'Emosi <strong>Bahagia</strong> paling mudah dikenali model dengan akurasi <strong>79%</strong>. Fitur khasnya berupa sudut bibir melengkung ke atas sangat spesifik dan konsisten di berbagai wajah.'
        },
        'Terkejut': {
            'icon': '😲',
            'akurasi': 69,
            'teks': 'Emosi <strong>Terkejut</strong> memiliki akurasi <strong>69%</strong>. Ciri khas alis terangkat dan mulut terbuka cukup mudah dikenali, tetapi kadang mirip dengan ekspresi Takut.'
        },
        'Netral': {
            'icon': '😐',
            'akurasi': 66,
            'teks': 'Emosi <strong>Netral</strong> memiliki akurasi <strong>66%</strong>. Minimnya fitur dominan membuat model cukup sulit membedakannya dari ekspresi Sedih atau emosi lain.'
        },
        'Sedih': {
            'icon': '😢',
            'akurasi': 61,
            'teks': 'Emosi <strong>Sedih</strong> memiliki akurasi <strong>61%</strong>. Fiturnya cukup mirip dengan Netral sehingga model kadang salah mengklasifikasikan keduanya.'
        },
        'Marah': {
            'icon': '😠',
            'akurasi': 58,
            'teks': 'Emosi <strong>Marah</strong> memiliki akurasi <strong>58%</strong>. Ekspresi ini sering tumpang tindih dengan Jijik pada area alis dan bibir yang mirip sehingga membingungkan model.'
        },
        'Takut': {
            'icon': '😨',
            'akurasi': 50,
            'teks': 'Emosi <strong>Takut</strong> memiliki akurasi <strong>50%</strong>. Ekspresinya sangat mirip dengan Terkejut sehingga model sering salah membedakan keduanya.'
        },
        'Jijik': {
            'icon': '🤢',
            'akurasi': 45,
            'teks': 'Emosi <strong>Jijik</strong> memiliki akurasi terendah sebesar <strong>45%</strong>. Dataset kelas ini lebih sedikit dan fitur wajahnya kurang spesifik dibanding emosi lainnya.'
        },
    }

    col_chart, col_note = st.columns([3, 2], gap='large')

    with col_chart:
        st.markdown("""
            <p style="font-size:0.75rem; color: var(--text-dark); font-weight:700; text-transform:uppercase; margin-bottom:0.5rem;">📈 Grafik Tingkat Akurasi Berdasarkan Kelas Dataset</p>
        """, unsafe_allow_html=True)

        selected_emotion = st.radio(
            "Pilih emosi",
            options=['Bahagia', 'Terkejut', 'Netral', 'Sedih', 'Marah', 'Takut', 'Jijik'],
            horizontal=True,
            label_visibility="collapsed"
        )

        fig_acc = px.bar(
            acc_data, 
            x='Akurasi (%)', 
            y='Kelas Emosi', 
            orientation='h',
            color_discrete_sequence=["#933B5B"],
            height=300
        )
        fig_acc.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})

    with col_note:
        info = analisis_emosi[selected_emotion]
        st.markdown(f"""
            <div class="glass-card">
                <h4>💡 Analisis: {info['icon']} {selected_emotion}</h4>
                <p style="font-size:2rem; font-weight:700; color: var(--amaranth) !important; font-family: 'Cormorant Garamond', serif !important; margin-bottom:0.5rem;">{info['akurasi']}%</p>
                <p style="font-size:0.88rem; line-height:1.65; color: var(--text-dark) !important;">
                    {info['teks']}
                </p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<center>
🧠 AI Emotion Detection System<br>
Universitas Andalas 2026
</center>
""", unsafe_allow_html=True)
