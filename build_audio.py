import os
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import imageio_ffmpeg

SR = 48000
DURATION = 30.0
N_SAMPLES = int(SR * DURATION)

def load_mp3_to_stereo(path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_wav = path + ".tmp.wav"
    subprocess.run([
        ffmpeg, "-y", "-i", path,
        "-ar", str(SR), "-ac", "2", "-f", "wav", tmp_wav
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    sr, data = wavfile.read(tmp_wav)
    os.remove(tmp_wav)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return data

def lowpass(sig, cutoff, sr=SR, order=4):
    b, a = butter(order, cutoff / (0.5 * sr), btype='low')
    return lfilter(b, a, sig, axis=0)

def highpass(sig, cutoff, sr=SR, order=2):
    b, a = butter(order, cutoff / (0.5 * sr), btype='high')
    return lfilter(b, a, sig, axis=0)

def add_stereo(mix, sig_l, sig_r, start_sec):
    idx = int(start_sec * SR)
    n = min(len(sig_l), N_SAMPLES - idx)
    if n <= 0 or idx < 0:
        return
    mix[idx:idx+n, 0] += sig_l[:n]
    mix[idx:idx+n, 1] += sig_r[:n]

def synth_pad_chord(freqs, dur, attack=0.4, release=0.8, amp=0.08, bright=1800.0):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    sig_l = np.zeros_like(t)
    sig_r = np.zeros_like(t)
    for i, f in enumerate(freqs):
        # Rich detuned stereo oscillators
        detune = 1.0025 if i % 2 == 0 else 0.9975
        w1 = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.0 * t)
        w2 = np.sin(2 * np.pi * (f * detune) * t) + 0.35 * np.sin(2 * np.pi * (f * detune) * 2.0 * t)
        # Subtle sawtooth warmth
        saw = 0.15 * (2.0 * (t * f - np.floor(0.5 + t * f)))
        sig_l += (w1 + saw) / len(freqs)
        sig_r += (w2 + saw) / len(freqs)
    env = np.ones_like(t)
    att_n = int(attack * SR)
    rel_n = int(release * SR)
    if att_n > 0:
        env[:att_n] = np.linspace(0, 1, att_n) ** 1.5
    if rel_n > 0 and rel_n < len(t):
        env[-rel_n:] = np.linspace(1, 0, rel_n) ** 1.8
    stereo = np.stack([sig_l * env, sig_r * env], axis=1) * amp
    return lowpass(stereo, bright)

def synth_sub_drop(start_f, end_f, dur, amp=0.32):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    freqs = np.linspace(start_f, end_f, len(t))
    phase = 2 * np.pi * np.cumsum(freqs) / SR
    env = np.exp(-t * 3.2) * np.minimum(1.0, t / 0.015)
    sig = (np.sin(phase) + 0.25 * np.tanh(2.5 * np.sin(phase))) * env * amp
    return sig, sig

def synth_whoosh(dur, low_f=180, high_f=2400, amp=0.14, pan_dir=1):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((n, 2))
    noise = lowpass(highpass(noise, low_f), high_f)
    env = np.sin(np.pi * (t / dur)) ** 2.2
    pan = np.linspace(0.25, 0.85, n) if pan_dir > 0 else np.linspace(0.85, 0.25, n)
    l = noise[:, 0] * env * (1.0 - pan * 0.5) * amp
    r = noise[:, 1] * env * (0.5 + pan * 0.5) * amp
    return l, r

def synth_ui_chime(freqs, dur=0.35, amp=0.11, pan=0.5):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    sig = np.zeros_like(t)
    for idx, f in enumerate(freqs):
        delay = idx * 0.045
        mask = t >= delay
        td = t[mask] - delay
        tone = (np.sin(2 * np.pi * f * td) + 0.25 * np.sin(2 * np.pi * f * 3.0 * td)) * np.exp(-td * 11.0)
        tone *= np.minimum(1.0, td / 0.004)
        sig[mask] += tone
    l = sig * amp * (1.0 - 0.4 * pan)
    r = sig * amp * (0.6 + 0.4 * pan)
    return l, r

def synth_tick(amp=0.07, pitch=1400.0, pan=0.5):
    dur = 0.045
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    sig = np.sin(2 * np.pi * pitch * t) * np.exp(-t * 95.0) * amp
    return sig * (1.0 - 0.4 * pan), sig * (0.6 + 0.4 * pan)

def main():
    mix = np.zeros((N_SAMPLES, 2), dtype=np.float32)

    # =========================================================================
    # SCENE 1 (0 - 4s): CHAOS - Dark tension pulse + accelerating notifications
    # =========================================================================
    pad1 = synth_pad_chord([73.42, 110.0, 146.83, 174.61], 4.0, attack=0.2, release=0.2, amp=0.14, bright=900)
    add_stereo(mix, pad1[:, 0], pad1[:, 1], 0.0)
    l_sub, r_sub = synth_sub_drop(95.0, 48.0, 1.2, amp=0.24)
    add_stereo(mix, l_sub, r_sub, 0.02)

    # Accelerating mechanical ticks and chaotic notification pings (0.2s -> 3.9s)
    tick_times = np.geomspace(0.18, 3.85, 28)
    for i, tt in enumerate(4.02 - tick_times[::-1]):
        if 0.15 <= tt <= 3.92:
            pan = (i % 5) / 4.0
            pitch = 950 + (i * 55) % 900
            tl, tr = synth_tick(amp=0.045 + 0.002 * i, pitch=pitch, pan=pan)
            add_stereo(mix, tl, tr, tt)
            if i % 3 == 0:
                cl, cr = synth_ui_chime([620 + (i * 40) % 500, 580], dur=0.18, amp=0.055, pan=1.0 - pan)
                add_stereo(mix, cl, cr, tt + 0.02)

    # Rising tension whoosh into the 4.0s freeze
    wl, wr = synth_whoosh(1.2, low_f=250, high_f=3200, amp=0.18, pan_dir=1)
    add_stereo(mix, wl, wr, 2.85)

    # =========================================================================
    # SCENE 2 (4 - 8s): THE BRANDRUMS REVEAL - Time freeze + Red/White Pulse
    # =========================================================================
    # 4.00s: Massive time-freeze sub drop + crystalline freeze chime
    dl, dr = synth_sub_drop(120.0, 36.0, 1.6, amp=0.34)
    add_stereo(mix, dl, dr, 4.00)
    fl, fr = synth_ui_chime([1174.66, 1760.0, 2349.32], dur=0.85, amp=0.11, pan=0.5)
    add_stereo(mix, fl, fr, 4.02)

    # 4.35s: Energy pulse sweep + Heroic Brandrums chord (Dmaj9 / Bm11)
    pl, pr = synth_whoosh(1.5, low_f=150, high_f=3800, amp=0.22, pan_dir=-1)
    add_stereo(mix, pl, pr, 4.35)
    pad2 = synth_pad_chord([73.42, 146.83, 220.0, 277.18, 329.63], 3.7, attack=0.35, release=0.6, amp=0.17, bright=2200)
    add_stereo(mix, pad2[:, 0], pad2[:, 1], 4.30)

    # Subtle node activation clicks in Scene 2
    for idx, nt in enumerate([5.5, 6.0, 6.5, 7.0, 7.4]):
        nl, nr = synth_ui_chime([880.0 + idx * 110.0, 1318.5], dur=0.25, amp=0.075, pan=0.2 + 0.15 * idx)
        add_stereo(mix, nl, nr, nt)

    # =========================================================================
    # SCENE 3 (8 - 16s): THE AUTOMATION MACHINE - Driving Apple-style pulse
    # =========================================================================
    pad3a = synth_pad_chord([98.0, 196.0, 246.94, 293.66, 369.99], 4.2, attack=0.3, release=0.5, amp=0.15, bright=2000)
    pad3b = synth_pad_chord([110.0, 220.0, 277.18, 329.63, 440.0], 4.2, attack=0.3, release=0.5, amp=0.16, bright=2400)
    add_stereo(mix, pad3a[:, 0], pad3a[:, 1], 8.0)
    add_stereo(mix, pad3b[:, 0], pad3b[:, 1], 12.0)

    # Precision 16th-note electronic arpeggio pluck (8.0s -> 16.0s)
    arp_notes = [293.66, 369.99, 440.0, 587.33, 440.0, 369.99, 329.63, 440.0]
    step = 0.25
    for k in range(32):
        t_arp = 8.0 + k * step
        f_arp = arp_notes[k % len(arp_notes)]
        al, ar = synth_ui_chime([f_arp], dur=0.22, amp=0.055, pan=0.3 if k % 2 == 0 else 0.7)
        add_stereo(mix, al, ar, t_arp)

    # Synchronized UI action sounds for the 4 headlines:
    # 1. CAPTURE LEADS (8.15s)
    c1l, c1r = synth_ui_chime([587.33, 880.0], dur=0.35, amp=0.12, pan=0.35)
    add_stereo(mix, c1l, c1r, 8.15)
    # 2. RESPOND INSTANTLY (10.0s)
    c2l, c2r = synth_ui_chime([739.99, 1108.73], dur=0.35, amp=0.12, pan=0.65)
    add_stereo(mix, c2l, c2r, 10.00)
    # 3. AUTOMATE WORKFLOWS (12.0s)
    c3l, c3r = synth_ui_chime([880.0, 1318.51], dur=0.38, amp=0.12, pan=0.4)
    add_stereo(mix, c3l, c3r, 12.00)
    # 4. FOLLOW UP AUTOMATICALLY (14.0s)
    c4l, c4r = synth_ui_chime([880.0, 1108.73, 1479.98], dur=0.45, amp=0.13, pan=0.6)
    add_stereo(mix, c4l, c4r, 14.00)

    # =========================================================================
    # SCENE 4 (16 - 22s): TRANSFORMATION - Expansive uplifting chord swell
    # =========================================================================
    wl4, wr4 = synth_whoosh(0.9, low_f=200, high_f=2800, amp=0.16, pan_dir=1)
    add_stereo(mix, wl4, wr4, 15.75)
    dl4, dr4 = synth_sub_drop(85.0, 45.0, 1.2, amp=0.24)
    add_stereo(mix, dl4, dr4, 16.00)

    pad4 = synth_pad_chord([73.42, 146.83, 220.0, 293.66, 369.99, 440.0], 6.0, attack=0.5, release=0.8, amp=0.18, bright=2800)
    add_stereo(mix, pad4[:, 0], pad4[:, 1], 16.0)

    # Completed workflows illuminating one by one
    for idx, wt in enumerate([16.8, 18.0, 19.2, 20.4]):
        il, ir = synth_ui_chime([587.33 * (1.122 ** idx), 880.0 * (1.122 ** idx)], dur=0.4, amp=0.08, pan=0.3 + 0.15 * idx)
        add_stereo(mix, il, ir, wt)

    # =========================================================================
    # SCENE 5 (22 - 26s): BRAND IDENTITY - Sleek halo product resonance
    # =========================================================================
    wl5, wr5 = synth_whoosh(0.8, low_f=180, high_f=2600, amp=0.15, pan_dir=-1)
    add_stereo(mix, wl5, wr5, 21.75)
    pad5 = synth_pad_chord([98.0, 196.0, 293.66, 369.99, 440.0, 587.33], 4.2, attack=0.35, release=0.6, amp=0.17, bright=2600)
    add_stereo(mix, pad5[:, 0], pad5[:, 1], 22.0)

    # =========================================================================
    # SCENE 6 (26 - 30s): POWERFUL CTA + MEMORABLE BRANDRUMS SONIC LOGO
    # =========================================================================
    dl6, dr6 = synth_sub_drop(110.0, 42.0, 1.5, amp=0.28)
    add_stereo(mix, dl6, dr6, 26.00)
    pad6 = synth_pad_chord([73.42, 146.83, 220.0, 293.66, 369.99, 587.33], 4.0, attack=0.2, release=1.4, amp=0.18, bright=3000)
    add_stereo(mix, pad6[:, 0], pad6[:, 1], 26.0)

    # Memorable Sonic Logo at 28.35s (3-note crystalline Brandrums signature: A4 -> D5 -> F#5 + D6 shimmer)
    s1l, s1r = synth_ui_chime([440.0, 880.0], dur=0.5, amp=0.15, pan=0.4)
    s2l, s2r = synth_ui_chime([587.33, 1174.66], dur=0.6, amp=0.16, pan=0.5)
    s3l, s3r = synth_ui_chime([739.99, 1174.66, 1479.98], dur=1.35, amp=0.19, pan=0.5)
    add_stereo(mix, s1l, s1r, 28.35)
    add_stereo(mix, s2l, s2r, 28.55)
    add_stereo(mix, s3l, s3r, 28.78)

    # =========================================================================
    # MIX IN AMERICAN ENGLISH VOICEOVER CLIPS WITH AUTOMATIC DUCKING
    # =========================================================================
    vo_Placements = [
        ("/home/user/video/assets/vo_part1.mp3", 0.30),   # 0.30s - 3.90s (Scene 1)
        ("/home/user/video/assets/vo_part2.mp3", 8.35),   # 8.35s - 13.76s (Scene 3)
        ("/home/user/video/assets/vo_part3.mp3", 17.30),  # 17.30s - 19.94s (Scene 4)
        ("/home/user/video/assets/vo_part4.mp3", 26.05),  # 26.05s - 28.35s (Scene 6)
    ]

    vo_track = np.zeros_like(mix)
    duck_env = np.ones(N_SAMPLES, dtype=np.float32)

    for path, start_t in vo_Placements:
        clip = load_mp3_to_stereo(path)
        # Normalize voice clip cleanly
        peak = np.max(np.abs(clip)) + 1e-6
        clip = (clip / peak) * 0.88
        idx = int(start_t * SR)
        n = min(len(clip), N_SAMPLES - idx)
        vo_track[idx:idx+n] += clip[:n]
        # Duck music smoothly around voiceover
        pad_s = int(0.18 * SR)
        d0 = max(0, idx - pad_s)
        d1 = min(N_SAMPLES, idx + n + pad_s)
        duck_env[d0:d1] = 0.42

    # Smooth ducking envelope
    kernel_len = int(0.15 * SR)
    kernel = np.ones(kernel_len) / kernel_len
    duck_env = np.convolve(duck_env, kernel, mode='same')

    final_mix = mix * duck_env[:, None] + vo_track

    # Soft master limiter to prevent any clipping
    final_mix = np.tanh(final_mix * 1.15) * 0.92
    out_int16 = (np.clip(final_mix, -1.0, 1.0) * 32767).astype(np.int16)
    out_path = "/home/user/video/assets/soundtrack_master.wav"
    wavfile.write(out_path, SR, out_int16)
    print(f"Saved master soundtrack to {out_path} ({len(out_int16)/SR:.2f}s)")

if __name__ == "__main__":
    main()
