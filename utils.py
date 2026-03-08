"""
utils.py – Neon visual engine for Virtual Air Drums.

Provides: neon pad rendering, particle hit effects, motion trails,
floating text, pulse rings, FPS counter, and HUD overlay.
All animations are time-based for smooth frame-rate-independent motion.
"""

import cv2
import time
import math
import random
import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════════
#  LOW-LEVEL DRAWING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp_color(color):
    """Clamp each BGR channel to 0–255."""
    return tuple(max(0, min(255, int(c))) for c in color)


def draw_rounded_rect(frame, pt1, pt2, color, radius=16, thickness=2):
    """Draw a rectangle with rounded corners (arcs + lines)."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)

    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0,   0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90,  0, 90, color, thickness, cv2.LINE_AA)

    cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)


def _fill_rounded_rect(overlay, pt1, pt2, color, radius=16):
    """Fill a rounded rectangle on an overlay (for alpha compositing)."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, -1)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, -1)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, -1)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, -1)


def draw_neon_rect(frame, pt1, pt2, color, alpha=0.30, rounded=True):
    """Draw a semi-transparent filled rectangle, optionally rounded."""
    overlay = frame.copy()
    if rounded:
        _fill_rounded_rect(overlay, pt1, pt2, color)
    else:
        cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOW EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_glow(frame, region, color, intensity=0.5):
    """Draw multi-layer expanding neon glow around a rectangular region.

    Parameters
    ----------
    region    : (x1, y1, x2, y2)
    color     : BGR base colour
    intensity : 0.0–1.0
    """
    if intensity <= 0.01:
        return
    x1, y1, x2, y2 = region
    layers = 5
    for i in range(layers, 0, -1):
        expand = i * 3
        layer_alpha = intensity * (0.06 + 0.03 * (layers - i))
        glow_col = _clamp_color(tuple(c * (0.4 + 0.6 * intensity) for c in color))
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (x1 - expand, y1 - expand),
                      (x2 + expand, y2 + expand),
                      glow_col, 2 + i * 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, layer_alpha, frame, 1 - layer_alpha, 0, frame)


def draw_text_glow(frame, text, pos, color, scale=0.85, thickness=2):
    """Draw text with a neon glow halo behind it."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Glow layers (thicker, dimmer behind the text)
    glow_col = _clamp_color(tuple(c * 0.4 for c in color))
    for offset in [3, 2]:
        overlay = frame.copy()
        cv2.putText(overlay, text, (pos[0], pos[1]), font, scale,
                    glow_col, thickness + offset * 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    # Shadow
    cv2.putText(frame, text, (pos[0] + 2, pos[1] + 2), font, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  DRUM PAD RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

def draw_pad(frame, pad, hover=False, hit_intensity=0.0):
    """Render a single drum pad with three visual states.

    States
    ------
    Idle  : soft glow border, dark transparent fill
    Hover : brighter fill + stronger glow
    Hit   : bright flash overlay + expanded glow
    """
    x1, y1, x2, y2 = pad["region"]
    color = pad["color"]
    name = pad["name"]

    # ── Transparent fill ──────────────────────────────────────────────────
    dark = _clamp_color(tuple(c // 6 for c in color))
    fill_alpha = 0.25 + (0.15 if hover else 0.0) + hit_intensity * 0.2
    draw_neon_rect(frame, (x1, y1), (x2, y2), dark, alpha=min(fill_alpha, 0.7))

    # ── Glow border ───────────────────────────────────────────────────────
    glow_str = 0.25 + (0.3 if hover else 0.0) + hit_intensity * 0.5
    draw_glow(frame, pad["region"], color, intensity=min(glow_str, 1.0))

    # ── Rounded outline ───────────────────────────────────────────────────
    outline_col = _clamp_color(
        tuple(c * (1.0 + hit_intensity * 0.5) for c in color))
    t = 2 if not hover else 3
    draw_rounded_rect(frame, (x1, y1), (x2, y2), outline_col, radius=14, thickness=t)

    # ── Hit flash overlay ─────────────────────────────────────────────────
    if hit_intensity > 0.05:
        bright = _clamp_color(
            tuple(c + (255 - c) * hit_intensity * 0.6 for c in color))
        draw_neon_rect(frame, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3),
                       bright, alpha=hit_intensity * 0.35)

    # ── Centred label with glow ───────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.82
    thick = 2
    text_size = cv2.getTextSize(name, font, scale, thick)[0]
    tx = x1 + (x2 - x1 - text_size[0]) // 2
    ty = y1 + (y2 - y1 + text_size[1]) // 2
    txt_col = _clamp_color(
        tuple(c * (0.8 + hit_intensity * 0.4 + (0.1 if hover else 0.0))
              for c in color))
    draw_text_glow(frame, name, (tx, ty), txt_col, scale=scale, thickness=thick)


# ═══════════════════════════════════════════════════════════════════════════════
#  PARTICLE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class Particle:
    """A single glowing particle that moves outward and fades."""

    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "radius")

    def __init__(self, x, y, color, speed=120.0, lifetime=0.5):
        angle = random.uniform(0, 2 * math.pi)
        spd = random.uniform(speed * 0.4, speed)
        self.x = float(x)
        self.y = float(y)
        self.vx = math.cos(angle) * spd
        self.vy = math.sin(angle) * spd
        self.life = lifetime
        self.max_life = lifetime
        self.color = color
        self.radius = random.randint(2, 5)


class ParticleSystem:
    """Manages pools of particles for hit effects."""

    def __init__(self):
        self.particles: list[Particle] = []

    def emit(self, x, y, color, count=14, speed=150.0, lifetime=0.45):
        """Spawn a burst of particles at (x, y)."""
        for _ in range(count):
            self.particles.append(
                Particle(x, y, color, speed=speed, lifetime=lifetime))

    def update_and_draw(self, frame, dt):
        """Advance all particles by *dt* seconds and draw survivors."""
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt
            # Slow down
            p.vx *= 0.96
            p.vy *= 0.96

            t = p.life / p.max_life           # 1→0
            alpha = t * 0.6
            r = max(1, int(p.radius * t))
            col = _clamp_color(tuple(c * t for c in p.color))

            overlay = frame.copy()
            cv2.circle(overlay, (int(p.x), int(p.y)), r, col, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            alive.append(p)
        self.particles = alive


# ═══════════════════════════════════════════════════════════════════════════════
#  PULSE RING ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

class PulseRing:
    """Expanding ring that fades out over its lifetime."""

    def __init__(self, center, color, duration=0.4, max_radius=85):
        self.center = center
        self.color = color
        self.start = time.perf_counter()
        self.duration = duration
        self.max_radius = max_radius

    @property
    def alive(self):
        return (time.perf_counter() - self.start) < self.duration

    def draw(self, frame):
        t = (time.perf_counter() - self.start) / self.duration
        if t >= 1.0:
            return
        radius = int(self.max_radius * t)
        alpha = (1.0 - t) * 0.45
        thickness = max(1, int(3 * (1.0 - t)))
        col = _clamp_color(tuple(c * (1.0 - t * 0.5) for c in self.color))
        overlay = frame.copy()
        cv2.circle(overlay, self.center, radius, col, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ═══════════════════════════════════════════════════════════════════════════════
#  FLOATING TEXT
# ═══════════════════════════════════════════════════════════════════════════════

class FloatingText:
    """Hit label that floats upward and fades out."""

    def __init__(self, text, pos, color, duration=0.6):
        self.text = text
        self.x, self.y = pos
        self.color = color
        self.start = time.perf_counter()
        self.duration = duration

    @property
    def alive(self):
        return (time.perf_counter() - self.start) < self.duration

    def draw(self, frame):
        t = (time.perf_counter() - self.start) / self.duration
        if t >= 1.0:
            return
        fade = 1.0 - t
        y_off = int(35 * t)
        col = _clamp_color(tuple(c * fade for c in self.color))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, self.text, (self.x + 1, self.y - y_off + 1),
                    font, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, self.text, (self.x, self.y - y_off),
                    font, 0.75, col, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTION TRAIL
# ═══════════════════════════════════════════════════════════════════════════════

class MotionTrail:
    """Fading trail of glowing circles behind a fingertip."""

    def __init__(self, max_len=25, color=(0, 255, 255)):
        self.points = deque(maxlen=max_len)
        self.color = color

    def update(self, point):
        """Append (x, y) or None when hand is lost."""
        self.points.append(point)

    def draw(self, frame):
        pts = [p for p in self.points if p is not None]
        n = len(pts)
        if n < 2:
            return
        for i in range(n):
            alpha = (i + 1) / n                     # oldest=0, newest=1
            r = max(1, int(1 + 5 * alpha))
            col = _clamp_color(tuple(c * alpha for c in self.color))
            cv2.circle(frame, pts[i], r, col, -1, cv2.LINE_AA)
        # Connecting lines for smoothness
        for i in range(1, n):
            alpha = (i + 1) / n
            thickness = max(1, int(3 * alpha))
            col = _clamp_color(tuple(c * alpha * 0.6 for c in self.color))
            cv2.line(frame, pts[i - 1], pts[i], col, thickness, cv2.LINE_AA)


def draw_motion_trail(frame, trail):
    """Convenience wrapper – draw a MotionTrail on frame."""
    trail.draw(frame)


# ═══════════════════════════════════════════════════════════════════════════════
#  FINGERTIP MARKER
# ═══════════════════════════════════════════════════════════════════════════════

def draw_fingertip(frame, point, color=(0, 255, 255)):
    """Neon-styled triple-ring marker at a fingertip."""
    x, y = point
    # Outer glow
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), 22, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    # Middle ring
    cv2.circle(frame, (x, y), 14, color, 2, cv2.LINE_AA)
    # Inner bright dot
    bright = _clamp_color(tuple(min(255, c + 100) for c in color))
    cv2.circle(frame, (x, y), 5, bright, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  FPS COUNTER
# ═══════════════════════════════════════════════════════════════════════════════

class FPSCounter:
    """Rolling-average FPS tracker."""

    def __init__(self, avg_over=30):
        self._times = deque(maxlen=avg_over)
        self._last = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        dt = now - self._last
        self._times.append(dt)
        self._last = now
        return dt          # return delta for particle updates

    @property
    def fps(self):
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


# ═══════════════════════════════════════════════════════════════════════════════
#  HUD OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hud(frame, fps_value, num_hands, width=1280):
    """Semi-transparent top bar with title, hand count, and FPS."""
    bar_h = 50

    # Dark bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Bottom accent line (purple gradient feel)
    cv2.line(frame, (0, bar_h), (width, bar_h), (140, 80, 200), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(frame, "VIRTUAL AI DRUM KIT", (18, 34),
                font, 0.72, (220, 160, 255), 2, cv2.LINE_AA)

    # Hands detected
    hand_col = (100, 255, 200) if num_hands > 0 else (120, 120, 120)
    cv2.putText(frame, f"Hands: {num_hands}", (width // 2 - 55, 34),
                font, 0.58, hand_col, 1, cv2.LINE_AA)

    # FPS
    fps_col = (0, 230, 180) if fps_value >= 25 else (0, 150, 255)
    cv2.putText(frame, f"FPS: {fps_value:.0f}", (width - 130, 34),
                font, 0.58, fps_col, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  FRAME UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def darken_frame(frame, factor=0.50):
    """Darken the camera feed for better neon contrast."""
    return cv2.convertScaleAbs(frame, alpha=factor, beta=0)


def draw_particles(frame, particle_system, dt):
    """Convenience wrapper – update and draw all particles."""
    particle_system.update_and_draw(frame, dt)