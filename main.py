"""
main.py – Virtual Air Drums application entry point.

Uses MediaPipe hand detection — play drums with your bare hands!

Launch:  python main.py
Controls:
    Q  – Quit
    R  – Reset trails
"""

import cv2
import time

from detector import HandDetector
from sound_engine import SoundEngine
from utils import (
    draw_pad, draw_fingertip, draw_hud, darken_frame,
    draw_particles, draw_motion_trail,
    PulseRing, FloatingText, MotionTrail, FPSCounter, ParticleSystem,
)


# ─── Configuration ────────────────────────────────────────────────────────────

CAM_W, CAM_H = 1280, 720
HIT_FLASH_SEC = 0.30        # pad flash fade duration

# Neon colour palette
DRUM_PADS = [
    # ── Row 1 ─────────────────────────────────────────────────────────────
    {"name": "CRASH",  "sound": "crash",
     "region": (40,  60,  420, 220), "color": (200, 80,  255)},   # neon purple
    {"name": "RIDE",   "sound": "hihat",
     "region": (450, 60,  830, 220), "color": (255, 160, 30)},    # neon blue (BGR)
    {"name": "TOM 1",  "sound": "snare",
     "region": (860, 60, 1240, 220), "color": (255, 230, 50)},    # neon cyan
    # ── Row 2 ─────────────────────────────────────────────────────────────
    {"name": "HI-HAT", "sound": "hihat",
     "region": (40,  240, 420, 400), "color": (60,  255, 255)},   # neon yellow
    {"name": "SNARE",  "sound": "snare",
     "region": (450, 240, 830, 400), "color": (80,  80,  255)},   # neon red
    {"name": "TOM 2",  "sound": "snare",
     "region": (860, 240,1240, 400), "color": (200, 200, 50)},    # neon teal
    # ── Row 3 (kicks) ─────────────────────────────────────────────────────
    {"name": "KICK",   "sound": "kick",
     "region": (240, 430, 640, 620), "color": (80,  255, 80)},    # neon green
    {"name": "FLOOR",  "sound": "kick",
     "region": (660, 430,1060, 620), "color": (60,  160, 255)},   # neon orange
]

TRAIL_COLORS = [
    (255, 255, 0),     # cyan trail for hand 0
    (0,   255, 255),   # yellow trail for hand 1
]


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        return

    detector = HandDetector()
    sound_engine = SoundEngine("sounds")
    fps_counter = FPSCounter()
    particles = ParticleSystem()

    # Per-hand collision state  [{pad_name: inside?}, ...]
    hand_pad_state = [{}, {}]

    # Per-hand motion trails
    trails = [MotionTrail(max_len=25, color=TRAIL_COLORS[0]),
              MotionTrail(max_len=25, color=TRAIL_COLORS[1])]

    # Animation state
    pad_hit_time  = {p["name"]: 0.0 for p in DRUM_PADS}
    pad_hover     = {p["name"]: False for p in DRUM_PADS}
    active_pulses = []
    active_texts  = []

    cv2.namedWindow("Virtual Air Drums", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Virtual Air Drums", CAM_W, CAM_H)

    print("Virtual Air Drums started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (CAM_W, CAM_H))
        frame = cv2.flip(frame, 1)

        # Darken feed for neon contrast
        frame = darken_frame(frame, factor=0.45)

        now = time.perf_counter()
        dt = fps_counter.tick()       # also returns delta time

        # ── Detect hands ──────────────────────────────────────────────────
        hands = detector.find_hands(frame)

        # Reset hover flags
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

                # ── Collision detection (original logic) ──────────────────
                for pad in DRUM_PADS:
                    x1, y1, x2, y2 = pad["region"]
                    pad_name = pad["name"]

                    is_inside = x1 < x < x2 and y1 < y < y2

                    if is_inside:
                        pad_hover[pad_name] = True

                    prev_state = hand_pad_state[i].get(pad_name, False)

                    if is_inside and not prev_state:
                        # ── Sound trigger (untouched) ─────────────────────
                        sound_engine.play(pad["sound"])

                        # ── Visual hit effects ────────────────────────────
                        pad_hit_time[pad_name] = now

                        # Pulse ring
                        ring_r = max(x2 - x1, y2 - y1) // 2
                        active_pulses.append(
                            PulseRing((x, y), pad["color"],
                                      duration=0.40, max_radius=ring_r))

                        # Particles burst
                        particles.emit(x, y, pad["color"],
                                       count=16, speed=160.0, lifetime=0.45)

                        # Floating label
                        cx = x1 + (x2 - x1) // 2 - len(pad_name) * 7
                        cy = y1 + (y2 - y1) // 2
                        active_texts.append(
                            FloatingText(pad_name + "!", (cx, cy),
                                         pad["color"], duration=0.55))

                    hand_pad_state[i][pad_name] = is_inside

        # Decay trails for missing hands
        for i in range(2):
            if not trail_updated[i]:
                trails[i].update(None)

        # ── Draw drum pads ────────────────────────────────────────────────
        for pad in DRUM_PADS:
            elapsed = now - pad_hit_time.get(pad["name"], 0)
            hit_int = max(0.0, 1.0 - elapsed / HIT_FLASH_SEC)
            draw_pad(frame, pad,
                     hover=pad_hover[pad["name"]],
                     hit_intensity=hit_int)

        # ── Draw particles ────────────────────────────────────────────────
        draw_particles(frame, particles, dt)

        # ── Draw pulse rings ──────────────────────────────────────────────
        active_pulses = [p for p in active_pulses if p.alive]
        for pulse in active_pulses:
            pulse.draw(frame)

        # ── Draw floating texts ───────────────────────────────────────────
        active_texts = [t for t in active_texts if t.alive]
        for txt in active_texts:
            txt.draw(frame)

        # ── Draw motion trails ────────────────────────────────────────────
        for trail in trails:
            draw_motion_trail(frame, trail)

        # ── Draw fingertip markers ────────────────────────────────────────
        if hands:
            for i, hand in enumerate(hands):
                if i > 1:
                    break
                col = TRAIL_COLORS[i] if i < len(TRAIL_COLORS) else (0, 255, 255)
                draw_fingertip(frame, hand, color=col)

        # ── HUD ───────────────────────────────────────────────────────────
        draw_hud(frame, fps_counter.fps, len(hands), width=CAM_W)

        # ── Display + keyboard ────────────────────────────────────────────
        cv2.imshow("Virtual Air Drums", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            for t in trails:
                t.points.clear()

    cap.release()
    cv2.destroyAllWindows()
    sound_engine.quit()


if __name__ == "__main__":
    main()