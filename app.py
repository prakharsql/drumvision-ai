"""
app.py – Futuristic Streamlit interface for Virtual Air Drums.

Wraps the existing OpenCV/MediaPipe drum kit in a cyberpunk-themed browser UI.
All detection, sound, and animation logic is reused untouched.

Run:  streamlit run app.py
"""

import cv2
import time
import numpy as np
import streamlit as st

from detector import HandDetector
from sound_engine import SoundEngine
from utils import (
    draw_pad, draw_fingertip, draw_hud, darken_frame,
    draw_particles, draw_motion_trail,
    PulseRing, FloatingText, MotionTrail, FPSCounter, ParticleSystem,
)


# ─── Configuration (same as main.py) ─────────────────────────────────────────

CAM_W, CAM_H = 1280, 720
HIT_FLASH_SEC = 0.30

DRUM_PADS = [
    {"name": "CRASH",  "sound": "crash",
     "region": (40,  60,  420, 220), "color": (200, 80,  255)},
    {"name": "RIDE",   "sound": "hihat",
     "region": (450, 60,  830, 220), "color": (255, 160, 30)},
    {"name": "TOM 1",  "sound": "snare",
     "region": (860, 60, 1240, 220), "color": (255, 230, 50)},
    {"name": "HI-HAT", "sound": "hihat",
     "region": (40,  240, 420, 400), "color": (60,  255, 255)},
    {"name": "SNARE",  "sound": "snare",
     "region": (450, 240, 830, 400), "color": (80,  80,  255)},
    {"name": "TOM 2",  "sound": "snare",
     "region": (860, 240,1240, 400), "color": (200, 200, 50)},
    {"name": "KICK",   "sound": "kick",
     "region": (240, 430, 640, 620), "color": (80,  255, 80)},
    {"name": "FLOOR",  "sound": "kick",
     "region": (660, 430,1060, 620), "color": (60,  160, 255)},
]

TRAIL_COLORS = [(255, 255, 0), (0, 255, 255)]


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Virtual Air Drum Kit",
    page_icon="🥁",
    layout="wide",
)


# ─── Futuristic CSS ──────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Fonts ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Rajdhani:wght@400;500;600;700&display=swap');

/* ── Global theme ────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(145deg, #05050f 0%, #0a0a1e 35%, #0d0820 65%, #080515 100%);
    font-family: 'Rajdhani', sans-serif;
}
.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 0 !important;
    max-width: 1400px;
}
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    color: #c0c0d8;
}

/* ── Hide default Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ─────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a18 0%, #0e0820 100%);
    border-right: 1px solid rgba(160, 100, 255, 0.15);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ── Glass cards ─────────────────────────────────────────────────── */
.glass-card {
    background: rgba(15, 12, 30, 0.65);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(160, 100, 255, 0.2);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 0 20px rgba(120, 60, 200, 0.08),
                inset 0 0 30px rgba(120, 60, 200, 0.03);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(180, 120, 255, 0.35);
    box-shadow: 0 0 30px rgba(140, 80, 220, 0.15),
                inset 0 0 40px rgba(140, 80, 220, 0.05);
}
.glass-card h3 {
    font-family: 'Orbitron', sans-serif;
    color: #c8a0ff;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
}
.glass-card p, .glass-card li {
    color: #a0a0b8;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ── Hero title ──────────────────────────────────────────────────── */
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #c084fc, #818cf8, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(160, 100, 255, 0.3);
    margin: 0;
    padding: 8px 0 2px 0;
    animation: heroGlow 3s ease-in-out infinite alternate;
}
@keyframes heroGlow {
    0%   { filter: drop-shadow(0 0 8px rgba(160,100,255,0.3)); }
    100% { filter: drop-shadow(0 0 20px rgba(100,180,255,0.4)); }
}
.hero-sub {
    font-family: 'Rajdhani', sans-serif;
    text-align: center;
    color: #8888aa;
    font-size: 1.1rem;
    letter-spacing: 1px;
    margin-top: 2px;
    margin-bottom: 12px;
}

/* ── Camera panel (applied to Streamlit image container) ─────────── */
[data-testid="stImage"] {
    background: rgba(8, 6, 18, 0.7);
    border: 1px solid rgba(140, 80, 255, 0.25);
    border-radius: 18px;
    padding: 8px;
    box-shadow: 0 0 35px rgba(120, 60, 220, 0.12),
                0 0 80px rgba(80, 40, 160, 0.06);
}
[data-testid="stImage"] img {
    border-radius: 12px;
}

/* ── Status bar ──────────────────────────────────────────────────── */
.status-bar {
    display: flex;
    justify-content: center;
    gap: 40px;
    padding: 10px 24px;
    margin-bottom: 10px;
    background: rgba(12, 10, 25, 0.6);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(120, 80, 200, 0.15);
    border-radius: 12px;
}
.status-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
    color: #a0a0b8;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}
.status-dot.green  { background: #4ade80; box-shadow: 0 0 8px #4ade80; }
.status-dot.blue   { background: #60a5fa; box-shadow: 0 0 8px #60a5fa; }
.status-dot.purple { background: #c084fc; box-shadow: 0 0 8px #c084fc; }
.status-dot.dim    { background: #555; box-shadow: none; animation: none; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.4; }
}
.status-value {
    font-family: 'Orbitron', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: #d0d0e8;
}

/* ── Badge ───────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #c084fc;
    border: 1px solid rgba(160, 100, 255, 0.3);
    border-radius: 20px;
    padding: 4px 16px;
    margin-bottom: 8px;
    text-align: center;
    background: rgba(160, 100, 255, 0.06);
}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-radius: 12px;
    border: 1px solid rgba(160, 100, 255, 0.3);
    background: rgba(120, 60, 220, 0.12);
    color: #c8a0ff;
    padding: 10px 20px;
    transition: all 0.3s ease;
    box-shadow: 0 0 12px rgba(120, 60, 220, 0.08);
}
.stButton > button:hover {
    border-color: rgba(180, 120, 255, 0.6);
    background: rgba(140, 80, 240, 0.2);
    box-shadow: 0 0 25px rgba(140, 80, 240, 0.2);
    color: #e0c0ff;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0px);
}

/* ── Toggle overrides ────────────────────────────────────────────── */
.stToggle label span {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    color: #a0a0b8 !important;
}

/* ── Footer ──────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 16px 0 8px 0;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.8rem;
    color: #555568;
    letter-spacing: 1px;
}
.footer span {
    color: #7c6ea0;
}

/* ── Sidebar section labels ──────────────────────────────────────── */
.sidebar-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.7rem;
    color: #8070a0;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-bottom: 8px;
    margin-top: 8px;
}

/* ── Info box override ───────────────────────────────────────────── */
.stAlert {
    background: rgba(15, 12, 30, 0.5) !important;
    border: 1px solid rgba(120, 80, 200, 0.2) !important;
    border-radius: 12px !important;
    color: #a0a0b8 !important;
}

/* ── Dividers ────────────────────────────────────────────────────── */
hr {
    border-color: rgba(120, 80, 200, 0.12) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────

def _init_state():
    if "initialised" in st.session_state:
        return
    st.session_state.initialised = True
    st.session_state.camera_running = False
    st.session_state.cap = None
    st.session_state.detector = HandDetector()
    st.session_state.sound_engine = SoundEngine("sounds")
    st.session_state.fps_counter = FPSCounter()
    st.session_state.particles = ParticleSystem()
    st.session_state.hand_pad_state = [{}, {}]
    st.session_state.trails = [
        MotionTrail(max_len=25, color=TRAIL_COLORS[0]),
        MotionTrail(max_len=25, color=TRAIL_COLORS[1]),
    ]
    st.session_state.pad_hit_time = {p["name"]: 0.0 for p in DRUM_PADS}
    st.session_state.pad_hover = {p["name"]: False for p in DRUM_PADS}
    st.session_state.active_pulses = []
    st.session_state.active_texts = []
    st.session_state.show_trails = True
    st.session_state.show_particles = True
    st.session_state.show_hud = True
    st.session_state.last_fps = 0.0
    st.session_state.last_hands = 0
    st.session_state.frame_count = 0

_init_state()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="badge">AI Drum System</div>', unsafe_allow_html=True)

    # Controls card
    st.markdown("""
    <div class="glass-card">
        <h3>🎛️ Controls</h3>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.camera_running:
        if st.button("⏹  STOP CAMERA", use_container_width=True):
            st.session_state.camera_running = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.rerun()
    else:
        if st.button("▶  START CAMERA", use_container_width=True, type="primary"):
            st.session_state.camera_running = True
            st.rerun()

    st.markdown('<div class="sidebar-label">Visual Effects</div>', unsafe_allow_html=True)
    st.session_state.show_trails = st.toggle("Motion Trails", value=st.session_state.show_trails)
    st.session_state.show_particles = st.toggle("Particle Effects", value=st.session_state.show_particles)
    st.session_state.show_hud = st.toggle("HUD Overlay", value=st.session_state.show_hud)

    st.divider()

    # Instructions card
    st.markdown("""
    <div class="glass-card">
        <h3>📖 How to Play</h3>
        <ol>
            <li>Click <b>Start Camera</b></li>
            <li>Show your hands to the webcam</li>
            <li>Move fingertips into drum pads</li>
            <li>Drum sounds trigger on entry</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Layout card
    st.markdown("""
    <div class="glass-card">
        <h3>🥁 Kit Layout</h3>
        <p style="font-family: monospace; font-size: 0.8rem; color: #8888aa; line-height: 1.8;">
        CRASH &nbsp;│&nbsp; RIDE &nbsp;&nbsp;│&nbsp; TOM 1<br>
        HIHAT &nbsp;│&nbsp; SNARE │&nbsp; TOM 2<br>
        &nbsp;&nbsp;&nbsp;&nbsp;KICK &nbsp;&nbsp;│&nbsp; FLOOR
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero section ─────────────────────────────────────────────────────────────

st.markdown('<p class="hero-title">🥁 AI Virtual Air Drum Kit</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Control a virtual drum kit with hand gestures &nbsp;·&nbsp; Powered by MediaPipe + OpenCV</p>',
            unsafe_allow_html=True)


# ─── Idle state ───────────────────────────────────────────────────────────────

if not st.session_state.camera_running:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:40px 20px;">
        <h3 style="font-size:1rem;">Ready to Jam</h3>
        <p style="font-size:1.1rem;">Click <b style="color:#c084fc;">Start Camera</b> in the sidebar to begin playing.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        Powered by <span>OpenCV</span> · <span>MediaPipe</span> · <span>Streamlit</span> · <span>Pygame</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Open camera ──────────────────────────────────────────────────────────────

if st.session_state.cap is None or not st.session_state.cap.isOpened():
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not st.session_state.cap.isOpened():
        st.error("❌ Cannot open webcam. Check camera connection.")
        st.session_state.camera_running = False
        st.stop()

# Aliases
cap = st.session_state.cap
detector = st.session_state.detector
sound_engine = st.session_state.sound_engine
fps_counter = st.session_state.fps_counter
particles = st.session_state.particles
hand_pad_state = st.session_state.hand_pad_state
trails = st.session_state.trails
pad_hit_time = st.session_state.pad_hit_time
pad_hover = st.session_state.pad_hover
active_pulses = st.session_state.active_pulses
active_texts = st.session_state.active_texts


# ─── Status bar placeholder ──────────────────────────────────────────────────

status_ph = st.empty()

# ─── Camera frame placeholder ────────────────────────────────────────────────

frame_ph = st.empty()

# ─── Static footer ───────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    Powered by <span>OpenCV</span> · <span>MediaPipe</span> · <span>Streamlit</span> · <span>Pygame</span>
</div>
""", unsafe_allow_html=True)


# ─── Frame loop ───────────────────────────────────────────────────────────────

while st.session_state.camera_running:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Camera feed lost.")
        st.session_state.camera_running = False
        break

    frame = cv2.resize(frame, (CAM_W, CAM_H))
    frame = cv2.flip(frame, 1)
    frame = darken_frame(frame, factor=0.45)

    now = time.perf_counter()
    dt = fps_counter.tick()

    # ── Detect hands ──────────────────────────────────────────────────
    hands = detector.find_hands(frame)
    st.session_state.last_hands = len(hands)
    st.session_state.last_fps = fps_counter.fps

    for p in DRUM_PADS:
        pad_hover[p["name"]] = False

    trail_updated = [False, False]

    if hands:
        for i, hand in enumerate(hands):
            if i > 1:
                break
            x, y = hand
            trail_updated[i] = True
            trails[i].update((x, y))

            for pad in DRUM_PADS:
                x1, y1, x2, y2 = pad["region"]
                pad_name = pad["name"]
                is_inside = x1 < x < x2 and y1 < y < y2

                if is_inside:
                    pad_hover[pad_name] = True

                prev_state = hand_pad_state[i].get(pad_name, False)

                if is_inside and not prev_state:
                    sound_engine.play(pad["sound"])
                    pad_hit_time[pad_name] = now

                    ring_r = max(x2 - x1, y2 - y1) // 2
                    active_pulses.append(
                        PulseRing((x, y), pad["color"],
                                  duration=0.40, max_radius=ring_r))

                    if st.session_state.show_particles:
                        particles.emit(x, y, pad["color"],
                                       count=16, speed=160.0, lifetime=0.45)

                    cx = x1 + (x2 - x1) // 2 - len(pad_name) * 7
                    cy = y1 + (y2 - y1) // 2
                    active_texts.append(
                        FloatingText(pad_name + "!", (cx, cy),
                                     pad["color"], duration=0.55))

                hand_pad_state[i][pad_name] = is_inside

    for i in range(2):
        if not trail_updated[i]:
            trails[i].update(None)

    # ── Draw pads ─────────────────────────────────────────────────────
    for pad in DRUM_PADS:
        elapsed = now - pad_hit_time.get(pad["name"], 0)
        hit_int = max(0.0, 1.0 - elapsed / HIT_FLASH_SEC)
        draw_pad(frame, pad, hover=pad_hover[pad["name"]],
                 hit_intensity=hit_int)

    if st.session_state.show_particles:
        draw_particles(frame, particles, dt)

    active_pulses[:] = [p for p in active_pulses if p.alive]
    for pulse in active_pulses:
        pulse.draw(frame)

    active_texts[:] = [t for t in active_texts if t.alive]
    for txt in active_texts:
        txt.draw(frame)

    if st.session_state.show_trails:
        for trail in trails:
            draw_motion_trail(frame, trail)

    if hands:
        for i, hand in enumerate(hands):
            if i > 1:
                break
            col = TRAIL_COLORS[i] if i < len(TRAIL_COLORS) else (0, 255, 255)
            draw_fingertip(frame, hand, color=col)

    if st.session_state.show_hud:
        draw_hud(frame, fps_counter.fps, len(hands), width=CAM_W)

    # ── Render ────────────────────────────────────────────────────────
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.session_state.frame_count += 1

    # Status bar (update every 10 frames to reduce DOM thrashing)
    if st.session_state.frame_count % 10 == 1:
        n_hands = st.session_state.last_hands
        fps_val = st.session_state.last_fps
        hand_dot = "green" if n_hands > 0 else "dim"
        fps_dot = "blue" if fps_val >= 25 else "purple"

        status_ph.markdown(f"""
        <div class="status-bar">
            <div class="status-item">
                <span class="status-dot {hand_dot}"></span>
                Hands&nbsp;<span class="status-value">{n_hands}</span>
            </div>
            <div class="status-item">
                <span class="status-dot {fps_dot}"></span>
                FPS&nbsp;<span class="status-value">{fps_val:.0f}</span>
            </div>
            <div class="status-item">
                <span class="status-dot green"></span>
                System&nbsp;<span class="status-value">LIVE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Camera frame (single placeholder, no wrapping markdown)
    frame_ph.image(frame_rgb, channels="RGB", use_container_width=True)

    time.sleep(0.001)

# Cleanup
if st.session_state.cap is not None and not st.session_state.camera_running:
    st.session_state.cap.release()
    st.session_state.cap = None
