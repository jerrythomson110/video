#!/usr/bin/env python3
"""Original 10-second score and synchronized comedy foley for the Bean office short."""
import math
import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
DURATION = 10.0
N = int(SR * DURATION)
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "mr_bean_soundtrack.wav")
rng = np.random.default_rng(4861997)
mix = np.zeros((N, 2), dtype=np.float64)


def pan_gains(pan):
    angle = (max(-1.0, min(1.0, pan)) + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def add(signal, start, gain=1.0, pan=0.0):
    i = max(0, int(round(start * SR)))
    if i >= N:
        return
    sig = np.asarray(signal, dtype=np.float64)
    count = min(len(sig), N - i)
    if count <= 0:
        return
    left, right = pan_gains(pan)
    mix[i:i + count, 0] += sig[:count] * gain * left
    mix[i:i + count, 1] += sig[:count] * gain * right


def pluck(freq, dur=0.42, bright=1.0):
    t = np.arange(int(dur * SR)) / SR
    attack = np.minimum(1.0, t / 0.004)
    body = (
        np.sin(2 * np.pi * freq * t)
        + 0.38 * bright * np.sin(2 * np.pi * freq * 2.002 * t)
        + 0.16 * bright * np.sin(2 * np.pi * freq * 3.01 * t)
        + 0.07 * np.sin(2 * np.pi * freq * 4.02 * t)
    )
    env = attack * (np.exp(-t * 8.7) * 0.82 + np.exp(-t * 2.8) * 0.18)
    pick = rng.normal(0, 1, len(t)) * np.exp(-t * 150) * 0.12
    return (body * env + pick) * 0.5


def bass(freq, dur=0.55):
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(1, t / 0.009) * np.exp(-t * 5.0)
    vibrato = 1 + 0.002 * np.sin(2 * np.pi * 5.1 * t)
    phase = 2 * np.pi * freq * t * vibrato
    return (np.sin(phase) + 0.28 * np.sin(phase * 2) + 0.09 * np.sin(phase * 3)) * env * 0.62


def reed(freq, dur=0.32, wobble=0.0):
    t = np.arange(int(dur * SR)) / SR
    pitch = freq * (1 + wobble * np.sin(2 * np.pi * 6.2 * t) * np.minimum(1, t * 8))
    phase = np.cumsum(2 * np.pi * pitch / SR)
    tone = np.sin(phase) + 0.32 * np.sin(phase * 3) + 0.13 * np.sin(phase * 5)
    env = np.minimum(1, t / 0.025) * np.minimum(1, (dur - t) / 0.08)
    env *= 0.9 + 0.1 * np.sin(2 * np.pi * 4.3 * t)
    breath = rng.normal(0, 1, len(t)) * 0.018
    return (tone * 0.34 + breath) * env


def chime(freq, dur=0.65):
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(1, t / 0.003) * np.exp(-t * 6.8)
    return (
        np.sin(2 * np.pi * freq * t)
        + 0.45 * np.sin(2 * np.pi * freq * 2.76 * t) * np.exp(-t * 6)
        + 0.2 * np.sin(2 * np.pi * freq * 5.4 * t) * np.exp(-t * 11)
    ) * env * 0.45


def click(hard=False):
    dur = 0.055 if hard else 0.04
    t = np.arange(int(dur * SR)) / SR
    noise = rng.normal(0, 1, len(t))
    noise = noise - np.convolve(noise, np.ones(7) / 7, mode="same")
    return noise * np.exp(-t * (95 if hard else 135)) * 0.42 + np.sin(2 * np.pi * 1250 * t) * np.exp(-t * 120) * 0.12


def whoosh(dur=0.36, downward=True):
    t = np.arange(int(dur * SR)) / SR
    noise = rng.normal(0, 1, len(t))
    smooth_noise = np.convolve(noise, np.ones(11) / 11, mode="same")
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.7
    sweep_phase = 2 * np.pi * ((950 * t - 600 * t * t) if downward else (260 * t + 800 * t * t))
    return (smooth_noise * 1.8 + np.sin(sweep_phase) * 0.12) * env


def error_blip(freq=220, dur=0.16):
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(1, t / 0.008) * np.minimum(1, (dur - t) / 0.035)
    wave_ = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * freq * 2 * t)
    return wave_ * env * 0.35


def rattle(intensity=1.0):
    dur = 0.13
    t = np.arange(int(dur * SR)) / SR
    noise = rng.normal(0, 1, len(t))
    low = np.convolve(noise, np.ones(16) / 16, mode="same")
    high = noise - np.convolve(noise, np.ones(5) / 5, mode="same")
    env = np.exp(-t * 34)
    return (low * 1.6 + high * 0.28 + np.sin(2 * np.pi * 95 * t) * 0.3) * env * intensity


def scratch(dur=0.14):
    t = np.arange(int(dur * SR)) / SR
    noise = rng.normal(0, 1, len(t))
    low = np.convolve(noise, np.ones(21) / 21, mode="same")
    band = noise - low
    env = np.sin(np.pi * t / dur) ** 0.8
    return band * env * 0.18


def midi(note):
    return 440.0 * 2 ** ((note - 69) / 12)


# Original, tiptoeing pizzicato motif (not an existing theme).
melody = [
    (0.12, 72), (0.48, 76), (0.84, 79), (1.20, 76),
    (1.55, 74), (1.90, 77), (2.26, 81), (2.62, 77),
    (3.00, 74), (3.50, 78), (4.00, 81), (4.50, 78),
    (5.05, 76), (5.50, 74), (6.00, 71),
]
for i, (start, note) in enumerate(melody):
    add(pluck(midi(note), bright=0.85), start, 0.12 if start < 2.5 else 0.095, pan=-0.18 + (i % 3) * 0.15)
for start, note in [(0.1, 48), (1.1, 53), (2.1, 50), (3.1, 55), (4.1, 53), (5.1, 48), (6.1, 47)]:
    add(bass(midi(note)), start, 0.11, pan=-0.12)
# Soft offbeat chord snaps.
for start, chord in [(0.6, [60, 64, 67]), (1.6, [62, 65, 69]), (2.6, [59, 62, 67]), (3.6, [60, 64, 67]), (4.6, [62, 65, 69]), (5.6, [59, 62, 65])]:
    for j, note in enumerate(chord):
        add(pluck(midi(note), 0.3, 0.65), start + j * 0.018, 0.037, pan=0.25)

# Awkward random computer clicking.
for i, start in enumerate([0.48, 0.77, 1.03, 1.31, 1.58, 1.86, 2.12]):
    add(click(i in [2, 5]), start, 0.2, pan=-0.18 + (i % 2) * 0.32)
    if i in [2, 5]:
        add(click(False), start + 0.045, 0.11, pan=0.12)
# 404 error: two unhelpful descending beeps.
add(error_blip(310), 1.79, 0.19, pan=0.28)
add(error_blip(235), 1.96, 0.19, pan=0.28)
add(error_blip(260), 5.07, 0.13, pan=0.25)
add(error_blip(195), 5.24, 0.13, pan=0.25)

# Customer alerts and their tiny disappearances.
for i, start in enumerate([2.65, 3.37, 4.09]):
    pan = 0.55 - i * 0.12
    add(chime(midi(84 + i * 2)), start, 0.12, pan=pan)
    add(chime(midi(88 + i * 2)), start + 0.075, 0.085, pan=pan)
    add(whoosh(0.34, True), start + 0.91, 0.13, pan=pan)
    add(chime(midi(96 - i * 2), 0.24), start + 1.0, 0.04, pan=0.1)

# Head-scratch foley, deliberately just a little too loud.
for i, start in enumerate([5.38, 5.53, 5.69, 5.85, 6.01]):
    add(scratch(), start, 0.18, pan=-0.15 + (i % 2) * 0.18)
add(pluck(midi(70), 0.28), 6.12, 0.08, pan=-0.25)

# CRT-shaking sequence: rhythmic rattles, plastic knocks, and electrical protest.
add(whoosh(0.22, False), 6.30, 0.14, pan=0.15)
for i, start in enumerate(np.arange(6.42, 8.02, 0.115)):
    strength = 0.65 + 0.35 * math.sin(i * 1.7) ** 2
    add(rattle(strength), float(start), 0.12, pan=math.sin(i * 1.9) * 0.42)
    if i % 4 == 0:
        add(click(True), float(start) + 0.025, 0.12, pan=-math.sin(i) * 0.35)
# Spoon and mug complain at the end of the shake.
add(chime(1740, 0.36), 7.62, 0.055, pan=-0.2)
add(chime(2110, 0.28), 7.82, 0.04, pan=-0.2)

# Sheepish shrug and polished final cadence.
add(reed(midi(62), 0.28, 0.006), 8.20, 0.12, pan=-0.12)
add(reed(midi(60), 0.36, 0.009), 8.47, 0.11, pan=-0.12)
for j, note in enumerate([60, 64, 67, 72]):
    add(pluck(midi(note), 0.7, 0.8), 8.83 + j * 0.055, 0.085, pan=-0.3 + j * 0.2)
add(chime(midi(84), 1.0), 9.03, 0.085, pan=0.2)
add(bass(midi(48), 0.85), 8.83, 0.1, pan=-0.1)

# Very quiet rainy London room tone and old-monitor hum.
t = np.arange(N) / SR
rain = rng.normal(0, 1, N)
rain_low = np.convolve(rain, np.ones(36) / 36, mode="same")
room = rain_low * 0.014 + np.sin(2 * np.pi * 50 * t) * 0.0015
mix[:, 0] += room
mix[:, 1] += np.roll(room, 137) * 0.93

# Short, tasteful room reverb; the dry foley remains intelligible.
dry = mix.copy()
for delay_s, decay, cross in [(0.071, 0.12, True), (0.127, 0.075, False), (0.193, 0.045, True)]:
    d = int(delay_s * SR)
    if cross:
        mix[d:, 0] += dry[:-d, 1] * decay
        mix[d:, 1] += dry[:-d, 0] * decay
    else:
        mix[d:] += dry[:-d] * decay

# Gentle master bus saturation and level control.
mix = np.tanh(mix * 1.35)
rms = math.sqrt(float(np.mean(mix ** 2)) + 1e-12)
if rms > 0:
    mix *= min(3.0, 0.11 / rms)
peak = float(np.max(np.abs(mix)))
if peak > 0.94:
    mix *= 0.94 / peak
fade_in = int(0.06 * SR)
fade_out = int(0.24 * SR)
mix[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
mix[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None]

OUT.parent.mkdir(parents=True, exist_ok=True)
pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2")
with wave.open(str(OUT), "wb") as wav:
    wav.setnchannels(2)
    wav.setsampwidth(2)
    wav.setframerate(SR)
    wav.writeframes(pcm.tobytes())
print(f"Original stereo score + foley: {OUT} | 48 kHz | {DURATION:.2f}s | peak {np.max(np.abs(mix)):.3f}")
