#!/usr/bin/env python3
"""Build the 10-second audio bed for the ProBlack Instagram reel.

Creates an original, subtle upbeat instrumental and mixes it beneath the
registered voiceover with gentle envelope-based ducking.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets", "problack")
VOICEOVER_MP3 = os.path.join(ASSETS, "voiceover.mp3")
OUTPUT_WAV = os.path.join(ASSETS, "final_audio.wav")

SR = 48_000
DURATION = 10.0
N_SAMPLES = int(SR * DURATION)
TWO_PI = 2.0 * np.pi
RNG = np.random.default_rng(20261027)


def add_stereo(mix: np.ndarray, clip: np.ndarray, start_sec: float = 0.0) -> None:
    idx = int(start_sec * SR)
    n = min(len(clip), N_SAMPLES - idx)
    if n > 0 and idx >= 0:
        mix[idx : idx + n] += clip[:n]


def adsr(n: int, attack: float, release: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    a = np.minimum(1.0, t / max(attack, 1 / SR))
    r = np.minimum(1.0, (n / SR - t) / max(release, 1 / SR))
    return (a * r).astype(np.float32)


def warm_pad(freqs: list[float], start: float, dur: float, amp: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    for i, f in enumerate(freqs):
        detune = 1.0018 if i % 2 == 0 else 0.9982
        phase = 0.17 * i
        tone_l = (
            np.sin(TWO_PI * f * t + phase)
            + 0.24 * np.sin(TWO_PI * f * 2.0 * t + phase * 0.5)
            + 0.08 * np.sin(TWO_PI * f * 3.0 * t)
        )
        tone_r = (
            np.sin(TWO_PI * f * detune * t + phase + 0.12)
            + 0.24 * np.sin(TWO_PI * f * detune * 2.0 * t)
            + 0.08 * np.sin(TWO_PI * f * detune * 3.0 * t + 0.3)
        )
        weight = 1.0 / (1.0 + 0.34 * i)
        left += tone_l * weight
        right += tone_r * weight
    norm = sum(1.0 / (1.0 + 0.34 * i) for i in range(len(freqs)))
    env = adsr(n, 0.18, 0.42)
    return np.stack([left / norm * env, right / norm * env], axis=1) * amp


def pluck(freq: float, dur: float, amp: float, pan: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    wave = (
        np.sin(TWO_PI * freq * t)
        + 0.32 * np.sin(TWO_PI * freq * 2.01 * t)
        + 0.10 * np.sin(TWO_PI * freq * 3.98 * t)
    )
    env = np.exp(-t * 8.5) * np.minimum(1.0, t / 0.004)
    gain_l = 1.0 - 0.36 * pan
    gain_r = 0.64 + 0.36 * pan
    return np.stack([wave * env * gain_l, wave * env * gain_r], axis=1) * amp


def soft_kick(amp: float = 0.20) -> np.ndarray:
    dur = 0.32
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    freq = 88.0 * np.exp(-t * 12.0) + 43.0
    phase = TWO_PI * np.cumsum(freq) / SR
    wave = np.sin(phase) * np.exp(-t * 10.5)
    wave += 0.12 * np.sin(2.0 * phase) * np.exp(-t * 18.0)
    return np.stack([wave, wave], axis=1) * amp


def soft_clap(amp: float = 0.055) -> np.ndarray:
    dur = 0.14
    n = int(dur * SR)
    noise = RNG.standard_normal(n).astype(np.float32)
    noise = lfilter([1.0, -0.86], [1.0], noise).astype(np.float32)
    noise = lfilter([0.12], [1.0, -0.88], noise).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.exp(-t * 28.0) * (1.0 - np.exp(-t * 900.0))
    spread = np.zeros((n, 2), dtype=np.float32)
    spread[:, 0] = noise * env
    spread[:, 1] = np.roll(noise, 21) * env
    return spread * amp


def soft_hat(amp: float = 0.022) -> np.ndarray:
    dur = 0.075
    n = int(dur * SR)
    noise = RNG.standard_normal(n).astype(np.float32)
    noise = lfilter([1.0, -0.96], [1.0], noise).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.exp(-t * 46.0) * np.minimum(1.0, t / 0.001)
    pan = float(RNG.uniform(0.15, 0.85))
    return np.stack([noise * env * (1.0 - 0.25 * pan), noise * env * (0.75 + 0.25 * pan)], axis=1) * amp


def bell(freqs: list[float], dur: float, amp: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    for i, f in enumerate(freqs):
        delay = i * 0.055
        local = np.maximum(0.0, t - delay)
        tone = (
            np.sin(TWO_PI * f * local)
            + 0.24 * np.sin(TWO_PI * f * 2.01 * local)
            + 0.08 * np.sin(TWO_PI * f * 3.97 * local)
        )
        env = np.exp(-local * (2.4 + i * 0.35)) * np.minimum(1.0, local / 0.004)
        left += tone * env * (1.0 - 0.08 * i)
        right += tone * env * (0.92 + 0.08 * i)
    return np.stack([left, right], axis=1) * amp / len(freqs)


def build_music() -> np.ndarray:
    music = np.zeros((N_SAMPLES, 2), dtype=np.float32)

    # Four warm, optimistic chord beds at 100 BPM.
    add_stereo(music, warm_pad([146.83, 185.00, 220.00, 277.18, 329.63], 0.00, 2.65, 0.105))
    add_stereo(music, warm_pad([123.47, 146.83, 185.00, 220.00, 277.18], 2.40, 2.80, 0.100))
    add_stereo(music, warm_pad([98.00, 123.47, 146.83, 185.00, 246.94], 4.80, 2.80, 0.105))
    add_stereo(music, warm_pad([110.00, 138.59, 164.81, 185.00, 277.18], 7.20, 2.80, 0.100))

    # Gentle eighth-note arpeggio.
    arp_patterns = [
        [293.66, 369.99, 440.00, 554.37, 440.00, 369.99],
        [246.94, 293.66, 369.99, 440.00, 369.99, 293.66],
        [196.00, 246.94, 293.66, 369.99, 293.66, 246.94],
        [220.00, 277.18, 329.63, 369.99, 329.63, 277.18],
    ]
    step = 0.30
    for k in range(32):
        t = 0.12 + k * step
        pattern = arp_patterns[min(3, int(t // 2.4))]
        f = pattern[k % len(pattern)]
        add_stereo(music, pluck(f, 0.28, 0.036, 0.5 + 0.28 * np.sin(k * 1.7)), t)

    # Soft kick / clap / hats.
    beat = 0.60
    for b in range(16):
        t = b * beat
        if b % 2 == 0:
            add_stereo(music, soft_kick(0.17 if b < 14 else 0.13), t)
        else:
            add_stereo(music, soft_clap(0.050), t)
    for h in range(33):
        add_stereo(music, soft_hat(0.018 if h % 2 == 0 else 0.012), h * 0.30)

    # Restrained logo shimmer at the closing card.
    add_stereo(music, bell([587.33, 880.00, 1174.66, 1479.98], 1.85, 0.075), 8.05)

    fade_in = int(0.30 * SR)
    fade_out = int(0.85 * SR)
    music[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    music[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
    return music


def decode_voiceover() -> np.ndarray:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_wav = tmp.name
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                VOICEOVER_MP3,
                "-ar",
                str(SR),
                "-ac",
                "1",
                "-f",
                "wav",
                temp_wav,
            ],
            check=True,
        )
        sr, data = wavfile.read(temp_wav)
        if sr != SR:
            raise RuntimeError(f"Unexpected voiceover sample rate: {sr}")
        if data.dtype == np.int16:
            voice = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            voice = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float32:
            voice = data.copy()
        else:
            raise RuntimeError(f"Unsupported WAV dtype: {data.dtype}")
        return voice
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    music = build_music()
    voice = decode_voiceover()

    # Remove a little leading silence while keeping natural phrasing.
    peak = float(np.max(np.abs(voice)))
    if peak > 0:
        voice *= 0.78 / peak

    voice_track = np.zeros((N_SAMPLES, 2), dtype=np.float32)
    delay = int(1.55 * SR)
    n = min(len(voice), N_SAMPLES - delay)
    voice_track[delay : delay + n, 0] = voice[:n]
    voice_track[delay : delay + n, 1] = voice[:n]

    # Envelope-follow the narration and duck the music by up to ~5 dB.
    hop = 960  # 20 ms
    frames = 1 + (N_SAMPLES - 1) // hop
    padded = np.pad(voice_track[:, 0], (0, frames * hop - N_SAMPLES))
    energy = np.sqrt(np.mean(padded.reshape(frames, hop) ** 2, axis=1))
    energy = gaussian_filter1d(energy, sigma=10.0, mode="nearest")
    if energy.max() > 0:
        energy /= energy.max()
    energy = np.clip((energy - 0.015) / 0.985, 0.0, 1.0)
    duck = np.interp(np.arange(N_SAMPLES), np.arange(frames) * hop, 1.0 - 0.46 * energy)
    duck = np.clip(duck, 0.54, 1.0).astype(np.float32)

    mixed = music * duck[:, None] + voice_track * 0.92
    mixed = np.tanh(mixed * 1.12) / np.tanh(1.12)

    edge = int(0.04 * SR)
    mixed[:edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)[:, None]
    mixed[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)[:, None]

    pcm = np.clip(mixed * 32767.0, -32768.0, 32767.0).astype(np.int16)
    wavfile.write(OUTPUT_WAV, SR, pcm)
    print(f"Wrote {OUTPUT_WAV}: {DURATION:.2f}s, {SR} Hz, stereo, {os.path.getsize(OUTPUT_WAV):,} bytes")


if __name__ == "__main__":
    main()
