"""
generate_background.py – Create a dark-themed background image for the drum UI.

Run:  python generate_background.py
Creates: assets/background.png
"""

import os
import cv2
import numpy as np


def main():
    W, H = 1280, 720
    os.makedirs("assets", exist_ok=True)

    # ── Dark gradient base (navy → black) ─────────────────────────────────
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        # navy-blue top → near-black bottom
        b = int(60 * (1 - t))
        g = int(20 * (1 - t))
        r = int(10 * (1 - t))
        img[y, :] = (b, g, r)

    # ── Subtle hexagonal grid pattern ─────────────────────────────────────
    overlay = img.copy()
    hex_r = 50
    dx = int(hex_r * 1.732)   # √3 * r
    dy = int(hex_r * 1.5)
    for row in range(-1, H // dy + 2):
        for col in range(-1, W // dx + 2):
            cx = col * dx + (dx // 2 if row % 2 else 0)
            cy = row * dy
            pts = []
            for k in range(6):
                angle = np.pi / 3 * k + np.pi / 6
                px = int(cx + hex_r * np.cos(angle))
                py = int(cy + hex_r * np.sin(angle))
                pts.append([px, py])
            pts = np.array(pts, np.int32)
            cv2.polylines(overlay, [pts], True, (50, 40, 30), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # ── Glowing accent circles (neon cyan & magenta) ──────────────────────
    accents = [
        (200, 200, (200, 150, 30), 120),     # cyan-ish top-left
        (1080, 520, (150, 40, 180), 100),     # magenta bottom-right
        (640, 360, (100, 80, 50), 180),       # dim centre
    ]
    for cx, cy, color, radius in accents:
        for r in range(radius, 0, -1):
            alpha = (r / radius) ** 2 * 0.12
            overlay = img.copy()
            cv2.circle(overlay, (cx, cy), r, color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # ── Vignette ──────────────────────────────────────────────────────────
    Y, X = np.ogrid[:H, :W]
    cx, cy = W / 2, H / 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1 - (dist / max_dist) ** 1.5 * 0.6
    vignette = np.clip(vignette, 0, 1)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * vignette).astype(np.uint8)

    path = os.path.join("assets", "background.png")
    cv2.imwrite(path, img)
    print(f"  ✓ {path}  ({W}×{H})")


if __name__ == "__main__":
    main()
