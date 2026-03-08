"""
generate_sounds.py – One-time script to synthesise drum .wav files.

Run:  python generate_sounds.py
Creates: sounds/kick.wav, sounds/snare.wav, sounds/hihat.wav, sounds/crash.wav
"""

import os
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 44100
SOUNDS_DIR = "sounds"


def _ensure_dir():
    os.makedirs(SOUNDS_DIR, exist_ok=True)


def _write(name, data):
    """Normalise float data to 16-bit PCM and save."""
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak
    pcm = np.int16(data * 32767 * 0.9)
    path = os.path.join(SOUNDS_DIR, f"{name}.wav")
    wavfile.write(path, SAMPLE_RATE, pcm)
    print(f"  ✓ {path}  ({len(pcm) / SAMPLE_RATE:.2f}s)")


def _envelope(length, attack=0.005, decay_time=None):
    """ADSR-ish envelope: fast attack, exponential decay."""
    t = np.linspace(0, length, int(SAMPLE_RATE * length), endpoint=False)
    env = np.ones_like(t)
    att_samples = int(SAMPLE_RATE * attack)
    env[:att_samples] = np.linspace(0, 1, att_samples)
    if decay_time is None:
        decay_time = length
    decay = np.exp(-t / decay_time * 5)
    return t, env * decay


# ── Individual drums ──────────────────────────────────────────────────────────

def make_kick(duration=0.35):
    """Low-frequency sine sweep + sub thump."""
    t, env = _envelope(duration, attack=0.003, decay_time=0.25)
    # pitch drops from 150 Hz to 50 Hz
    freq = 150 * np.exp(-t / duration * 2) + 40
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    wave = np.sin(phase) * env
    # add a sub-click transient
    click_len = int(SAMPLE_RATE * 0.008)
    click = np.zeros_like(wave)
    click[:click_len] = np.sin(2 * np.pi * 1000 *
                               np.linspace(0, 0.008, click_len)) * 0.4
    click[:click_len] *= np.linspace(1, 0, click_len)
    return wave + click


def make_snare(duration=0.25):
    """Mid-frequency tone + band-limited noise burst."""
    t, env = _envelope(duration, attack=0.002, decay_time=0.12)
    tone = np.sin(2 * np.pi * 200 * t) * 0.5
    noise = np.random.randn(len(t)) * 0.7
    # shape noise envelope faster
    _, noise_env = _envelope(duration, attack=0.001, decay_time=0.08)
    return (tone + noise * noise_env) * env


def make_hihat(duration=0.12):
    """High-frequency band-passed noise."""
    t, env = _envelope(duration, attack=0.001, decay_time=0.05)
    noise = np.random.randn(len(t))
    # emphasise highs with simple high-pass (difference)
    hp = np.diff(noise, prepend=0) * 3
    return hp * env


def make_crash(duration=0.6):
    """Wide-band noise with slow decay + metallic shimmer."""
    t, env = _envelope(duration, attack=0.002, decay_time=0.4)
    noise = np.random.randn(len(t))
    shimmer = np.sin(2 * np.pi * 6500 * t) * 0.15
    shimmer += np.sin(2 * np.pi * 8200 * t) * 0.10
    return (noise * 0.6 + shimmer) * env


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    _ensure_dir()
    print("Generating drum sounds …")
    _write("kick",  make_kick())
    _write("snare", make_snare())
    _write("hihat", make_hihat())
    _write("crash", make_crash())
    print("Done.")


if __name__ == "__main__":
    main()
