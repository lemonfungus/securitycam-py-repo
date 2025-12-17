import streamlit as st
import cv2
import os
import time
from ultralytics import YOLO

# --- 1. PAGE CONFIGURATION & TACTICAL THEME ---
st.set_page_config(page_title="SENTINEL OPS", page_icon="👁️", layout="wide")

# Inject "Tactical Command" CSS
st.markdown("""
<style>
    /* 1. Main Background - Deep Gunmetal */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* 2. Containers - Dark Panels with Red Borders */
    div[data-testid="stForm"], div.css-card {
        background-color: #1e293b;
        padding: 2rem;
        border: 1px solid #334155;
        border-left: 5px solid #ef4444; /* Red Alert Line */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }

    /* 3. Input Fields - Terminal Style */
    .stTextInput input {
        background-color: #0f172a;
        border: 1px solid #475569;
        color: #00ff00; /* Hacker Green Text */
        font-family: 'Courier New', monospace;
    }
    
    /* 4. Buttons - Tactical Grey & Red */
    div.stButton > button {
        background-color: #334155;
        color: #f8fafc;
        border: 1px solid #94a3b8;
        padding: 0.5rem 1rem;
        border-radius: 0px; /* Sharp Edges */
        font-weight: bold;
        letter-spacing: 2px;
        transition: all 0.2s;
        width: 100%;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        background-color: #ef4444; /* Red on Hover */
        border-color: #ef4444;
        color: white;
    }
    
    /* 5. Offline Camera Style */
    .offline-cam p {
        color: #64748b !important;
    }

    /* 6. Headers - Glitch Effect Vibe */
    h1, h2, h3 {
        color: #f8fafc;
        text-transform: uppercase;
        letter-spacing: 3px;
        border-bottom: 2px solid #ef4444;
        display: inline-block;
        padding-bottom: 5px;
    }
    
    /* Custom Status Text */
    .status-online { color: #22c55e; font-weight: bold; }
    .status-offline { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'
if 'camera_source' not in st.session_state:
    st.session_state['camera_source'] = 0

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best.pt")

# --- PAGE 1: LOGIN ---
def show_login():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.form("login_form"):
            st.markdown("<h1 style='border:none; color:#ef4444; font-size: 3em;'>/// SENTINEL</h1>", unsafe_allow_html=True)
            st.markdown("RESTRICTED ACCESS // AUTHORIZED PERSONNEL ONLY")
            st.markdown("---")
            
            st.markdown("OPERATOR ID")
            username = st.text_input("User ID", placeholder="OP-01", label_visibility="collapsed")
            st.markdown("ACCESS KEY")
            password = st.text_input("Password", type="password", placeholder="******", label_visibility="collapsed")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("AUTHENTICATE")
            
            if submit:
                if username == "admin" and password == "1234":
                    st.success(">> IDENTITY CONFIRMED")
                    time.sleep(0.8)
                    st.session_state['page'] = 'camera_select'
                    st.rerun()
                else:
                    st.error(">> ACCESS DENIED // INCIDENT LOGGED")

# --- PAGE 2: GRID SELECTION ---
def show_camera_select():
    st.title("SECTOR SELECTION")
    st.markdown("ESTABLISH UPLINK WITH SURVEILLANCE NODE")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # --- CAM 1 (ONLINE) ---
    with col1:
        st.markdown("""
        <div class="css-card" style="border-left: 5px solid #22c55e;">
            <h3 style="border:none; margin:0;">SECTOR A (MAIN)</h3>
            <p class="status-online">● SIGNAL ACTIVE</p>
            <p style="font-size:0.8em; font-family:monospace;">LAT: 13.7563<br>LON: 100.5018</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("INITIATE FEED", key="cam1"):
            st.session_state['camera_source'] = 0
            st.session_state['page'] = 'dashboard'
            st.rerun()

    # --- CAM 2 (OFFLINE) ---
    with col2:
        st.markdown("""
        <div class="css-card" style="opacity: 0.6; border-left: 5px solid #64748b;">
            <h3 style="border:none; margin:0; color:#94a3b8;">SECTOR B (REAR)</h3>
            <p class="status-offline">● SIGNAL LOST</p>
            <p style="font-size:0.8em; font-family:monospace;">ERR: 404 NO CARRIER</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("INITIATE FEED", key="cam2", disabled=True)

    # --- CAM 3 (OFFLINE) ---
    with col3:
        st.markdown("""
        <div class="css-card" style="opacity: 0.6; border-left: 5px solid #64748b;">
            <h3 style="border:none; margin:0; color:#94a3b8;">SECTOR C (VAULT)</h3>
            <p class="status-offline">● SIGNAL LOST</p>
            <p style="font-size:0.8em; font-family:monospace;">ERR: MAINTENANCE</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("INITIATE FEED", key="cam3", disabled=True)

    # System Check
    st.markdown("<br><br>", unsafe_allow_html=True)
    if os.path.exists(model_path):
        st.code(f"SYSTEM CHECK: AI CORE [ONLINE] >> {os.path.basename(model_path)}")
    else:
        st.error(f"SYSTEM CHECK: AI CORE [CRITICAL FAILURE] >> Missing best.pt")

# --- PAGE 3: DASHBOARD ---
def show_dashboard():
    # Header
    c1, c2 = st.columns([6, 1])
    with c1:
        st.title("LIVE OPS: SECTOR A")
    with c2:
        if st.button("ABORT"):
            st.session_state['page'] = 'camera_select'
            st.rerun()
            
    st.markdown("---")

    col_video, col_log = st.columns([2.5, 1])
    
    with col_video:
        # Video Frame - Simple thin border
        st.markdown('<div style="border: 2px solid #ef4444; padding: 2px;">', unsafe_allow_html=True)
        video_spot = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(">> SENSOR CALIBRATION"):
            conf_threshold = st.slider("CONFIDENCE LEVEL", 0.0, 1.0, 0.5)

    with col_log:
        st.markdown("### INCIDENT LOG")
        log_container = st.empty()
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"CORE DUMP: {e}")
        return

    cap = cv2.VideoCapture(st.session_state['camera_source'])
    if not cap.isOpened():
        st.error("NO VIDEO SOURCE")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("SIGNAL INTERRUPTED")
            break
            
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        detected_names = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            detected_names.append(model.names[cls_id])
        
        # LOGGING - Terminal Style
        log_html = '<div style="background:#0f172a; padding:15px; border:1px solid #334155; height: 400px; overflow-y:auto; font-family:monospace;">'
        if detected_names:
            unique = list(set(detected_names))
            for name in unique:
                # Flashing Green/Red Text
                log_html += f'<div style="margin-bottom:5px; border-bottom:1px dashed #334155;">' \
                            f'<span style="color:#22c55e;">[DETECT]</span> ' \
                            f'<span style="color:#e2e8f0;">TARGET: {name}</span></div>'
        else:
             log_html += '<div style="color:#64748b;">Scanning sector... No targets.</div>'
        log_html += '</div>'
        log_container.markdown(log_html, unsafe_allow_html=True)

        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_spot.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

# --- ROUTER ---
if st.session_state['page'] == 'login':
    show_login()
elif st.session_state['page'] == 'camera_select':
    show_camera_select()
elif st.session_state['page'] == 'dashboard':
    show_dashboard()