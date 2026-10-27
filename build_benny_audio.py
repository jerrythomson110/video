import os
import math
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import imageio_ffmpeg

SR = 48000
DURATION = 60.0
TOTAL_SAMPLES = int(SR * DURATION)
FPS = 30
TOTAL_FRAMES = int(FPS * DURATION)

ASSETS_DIR = "/home/user/video/assets"

def load_mp3_mono(path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_wav = path + ".tmp.wav"
    subprocess.run(
        [ffmpeg, "-y", "-i", path, "-ac", "1", "-ar", str(SR), tmp_wav],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    sr, data = wavfile.read(tmp_wav)
    os.remove(tmp_wav)
    if data.dtype == np.int16:
        arr = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        arr = data.astype(np.float32) / 2147483648.0
    else:
        arr = data.astype(np.float32)
    # Trim leading/trailing silence > 0.008
    idx = np.where(np.abs(arr) > 0.008)[0]
    if len(idx) > 0:
        s0 = max(0, idx[0] - int(0.03 * SR))
        s1 = min(len(arr), idx[-1] + int(0.05 * SR))
        arr = arr[s0:s1]
    peak = np.max(np.abs(arr)) + 1e-9
    return (arr / peak) * 0.92

def add_mono(mix, sig, start_t, pan=0.0, gain=1.0):
    s0 = int(start_t * SR)
    if s0 >= TOTAL_SAMPLES or s0 + len(sig) <= 0:
        return
    if s0 < 0:
        sig = sig[-s0:]
        s0 = 0
    s1 = min(TOTAL_SAMPLES, s0 + len(sig))
    chunk = sig[:s1 - s0] * gain
    l_gain = math.cos((pan + 1.0) * math.pi / 4.0)
    r_gain = math.sin((pan + 1.0) * math.pi / 4.0)
    mix[s0:s1, 0] += chunk * l_gain
    mix[s0:s1, 1] += chunk * r_gain

def synth_pizzicato(freq, dur=0.18, amp=0.22):
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.exp(-t * 24.0) * (1.0 - np.exp(-t * 220.0))
    sig = (
        np.sin(2 * np.pi * freq * t)
        + 0.45 * np.sin(2 * np.pi * freq * 2.0 * t) * np.exp(-t * 35.0)
        + 0.20 * np.sin(2 * np.pi * freq * 3.0 * t) * np.exp(-t * 50.0)
    )
    return sig * env * amp

def synth_key_click(amp=0.35):
    n = int(0.045 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(101)
    noise = rng.standard_normal(n).astype(np.float32)
    b, a = butter(2, [900 / (SR / 2), 4500 / (SR / 2)], btype="band")
    click = lfilter(b, a, noise) * np.exp(-t * 140.0)
    body = np.sin(2 * np.pi * 320.0 * t) * np.exp(-t * 95.0) * 0.4
    return (click + body) * amp

def synth_error_buzzer(dur=0.42, amp=0.38):
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.ones_like(t)
    env[-int(0.06 * SR):] = np.linspace(1.0, 0.0, int(0.06 * SR))
    # Classic harsh dual-tone error buzz (145 Hz + 290 Hz + 435 Hz saw-like)
    sig = (
        0.50 * np.sign(np.sin(2 * np.pi * 155.0 * t))
        + 0.35 * np.sin(2 * np.pi * 310.0 * t)
        + 0.25 * np.sin(2 * np.pi * 620.0 * t)
    )
    b, a = butter(2, 1800 / (SR / 2), btype="low")
    return lfilter(b, a, sig) * env * amp

def synth_popup_beep(pitch_mult=1.0, amp=0.22):
    n = int(0.09 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f = 740.0 * pitch_mult
    env = np.exp(-t * 36.0)
    sig = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 1.5 * t)
    return sig * env * amp

def synth_pencil_scratch(dur=0.32, amp=0.22, seed=1):
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float32)
    b, a = butter(2, [1800 / (SR / 2), 6500 / (SR / 2)], btype="band")
    scr = lfilter(b, a, noise)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 14.0 * t)
    env = np.sin(np.pi * t / dur)
    return scr * mod * env * amp

def synth_pencil_snap(amp=0.65):
    n = int(0.18 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(77)
    noise = rng.standard_normal(n).astype(np.float32)
    b, a = butter(2, [1200 / (SR / 2), 7500 / (SR / 2)], btype="band")
    crack = lfilter(b, a, noise) * np.exp(-t * 85.0)
    wood = np.sin(2 * np.pi * (680.0 * np.exp(-t * 25.0)) * t) * np.exp(-t * 55.0) * 0.6
    return (crack + wood) * amp

def synth_gulp(amp=0.32):
    n = int(0.22 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f = 180.0 + 240.0 * np.sin(np.pi * t / 0.22)
    phase = 2 * np.pi * np.cumsum(f) / SR
    env = np.sin(np.pi * t / 0.22) ** 1.5
    return np.sin(phase) * env * amp

def synth_glass_slip(amp=0.36):
    # Comedic squeak + slide off smartphone glass
    n = int(0.36 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f = 1350.0 + 650.0 * np.sin(2 * np.pi * 6.5 * t) - 900.0 * (t / 0.36)
    phase = 2 * np.pi * np.cumsum(f) / SR
    env = np.sin(np.pi * t / 0.36) * np.exp(-t * 3.5)
    return np.sin(phase) * env * amp

def synth_portal_chime(dur=1.6, amp=0.28):
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    freqs = [523.25, 659.25, 783.99, 1046.50, 1318.51, 1567.98]
    out = np.zeros(n, dtype=np.float32)
    for i, f in enumerate(freqs):
        d_t = np.maximum(0.0, t - i * 0.07)
        env = np.exp(-d_t * 2.8) * (t >= i * 0.07)
        out += np.sin(2 * np.pi * f * d_t) * env * 0.25
    return out * amp

def synth_cinematic_impact(dur=1.8, amp=0.48):
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f = 45.0 + 95.0 * np.exp(-t * 14.0)
    phase = 2 * np.pi * np.cumsum(f) / SR
    sub = np.sin(phase) * np.exp(-t * 2.4)
    rng = np.random.default_rng(303)
    shimmer = rng.standard_normal(n).astype(np.float32)
    b, a = butter(2, [2000 / (SR / 2), 9000 / (SR / 2)], btype="band")
    shimmer = lfilter(b, a, shimmer) * np.exp(-t * 3.2) * 0.25
    return (sub + shimmer) * amp

def synth_whoosh(dur=0.45, f_center=900.0, amp=0.28):
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(int(f_center))
    noise = rng.standard_normal(n).astype(np.float32)
    b, a = butter(2, [max(120.0, f_center * 0.4) / (SR / 2), min(8000.0, f_center * 1.8) / (SR / 2)], btype="band")
    w = lfilter(b, a, noise)
    env = np.sin(np.pi * t / dur) ** 1.6
    return w * env * amp

def synth_warm_chord(freqs, dur=2.2, amp=0.14):
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.sin(np.pi * np.minimum(1.0, t / 0.35) * 0.5) * np.exp(- np.maximum(0.0, t - 0.35) * 1.4)
    out = np.zeros(n, dtype=np.float32)
    for f in freqs:
        out += (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.0 * t)) / len(freqs)
    return out * env * amp

def main():
    print("Building 60.0s Master Comedy Soundtrack & Lip-Sync Envelope for Mr. Benny...")
    mix = np.zeros((TOTAL_SAMPLES, 2), dtype=np.float32)
    vo_track = np.zeros(TOTAL_SAMPLES, dtype=np.float32)

    # Subtle warm British room tone
    rng = np.random.default_rng(42)
    room = rng.standard_normal(TOTAL_SAMPLES).astype(np.float32)
    b, a = butter(2, [80 / (SR / 2), 450 / (SR / 2)], btype="band")
    room = lfilter(b, a, room) * 0.004
    mix[:, 0] += room
    mix[:, 1] += room

    # Load Mr. Benny's 2 British spoken lines
    vo_tech = load_mp3_mono(os.path.join(ASSETS_DIR, "benny_vo_technology.mp3"))
    vo_better = load_mp3_mono(os.path.join(ASSETS_DIR, "benny_vo_much_better.mp3"))
    print(f"  'Technology.' duration: {len(vo_tech)/SR:.2f}s")
    print(f"  'Much better.' duration: {len(vo_better)/SR:.2f}s")

    # Place VO 1 ("Technology.") at 12.55s in Scene 2
    t_vo1 = 12.55
    add_mono(mix, vo_tech, t_vo1, pan=0.0, gain=1.05)
    s0 = int(t_vo1 * SR)
    vo_track[s0:s0 + len(vo_tech)] += vo_tech

    # Place VO 2 ("Much better.") at 47.35s in Scene 5
    t_vo2 = 47.35
    add_mono(mix, vo_better, t_vo2, pan=0.0, gain=1.05)
    s0 = int(t_vo2 * SR)
    vo_track[s0:s0 + len(vo_better)] += vo_better

    # =========================================================================
    # SCENE 1: THE DISASTER (0 - 8s)
    # =========================================================================
    # Tiptoe pizzicato intro (0.1 - 0.7s)
    for bt, f in [(0.15, 261.63), (0.38, 293.66), (0.60, 329.63)]:
        add_mono(mix, synth_pizzicato(f, 0.16, 0.16), bt, pan=-0.1)
    # 0.85s: Benny presses one key
    add_mono(mix, synth_key_click(0.48), 0.85, pan=0.0)
    # 1.05 - 2.1s: Optimistic little musical flourish
    for bt, f in [(1.05, 392.0), (1.20, 493.88), (1.35, 587.33), (1.52, 783.99), (1.72, 659.25), (1.92, 783.99)]:
        add_mono(mix, synth_pizzicato(f, 0.22, 0.18), bt, pan=0.1)
    # 2.5s & 3.2s: Head tilt left, then right (quirky low pizzicato plucks)
    add_mono(mix, synth_pizzicato(196.0, 0.25, 0.20), 2.50, pan=-0.25)
    add_mono(mix, synth_pizzicato(220.0, 0.25, 0.20), 3.20, pan=0.25)
    # 3.9s: Adjust tie, 4.3s: Thumbs-up ding, 4.85s: Click Publish
    add_mono(mix, synth_whoosh(0.22, 1200.0, 0.15), 3.90, pan=0.0)
    add_mono(mix, synth_pizzicato(1046.5, 0.20, 0.22), 4.30, pan=0.15)
    add_mono(mix, synth_key_click(0.45), 4.85, pan=0.0)
    # 5.00s: GIANT ERROR MESSAGE -> Harsh error buzzer!
    add_mono(mix, synth_error_buzzer(0.46, 0.42), 5.00, pan=0.0)
    # 5.46 - 8.00s: Perfectly timed silence! Two tiny eye-blinks (6.15s, 6.36s) + nervous gulp (6.88s)
    add_mono(mix, synth_pizzicato(1567.98, 0.07, 0.12), 6.15, pan=-0.05)
    add_mono(mix, synth_pizzicato(1567.98, 0.07, 0.12), 6.36, pan=0.05)
    add_mono(mix, synth_gulp(0.35), 6.88, pan=0.0)

    # =========================================================================
    # SCENE 2: BENNY BECOMES A DESIGN GENIUS (8 - 17s)
    # =========================================================================
    # 8.1 - 10.1s: Sketching crooked rectangle and circles on paper
    for i, st in enumerate([8.15, 8.60, 9.05, 9.50]):
        add_mono(mix, synth_pencil_scratch(0.34, 0.25, seed=10 + i), st, pan=0.1)
        add_mono(mix, synth_pizzicato([293.66, 349.23, 392.0, 440.0][i], 0.18, 0.15), st + 0.05, pan=-0.15)
    # 10.3 - 11.4s: Places paper against laptop screen & nods proudly
    add_mono(mix, synth_whoosh(0.25, 1600.0, 0.18), 10.25, pan=0.0)
    add_mono(mix, synth_warm_chord([261.63, 329.63, 392.0, 523.25], 1.1, 0.16), 10.35, pan=0.0)
    # 11.55s: PENCIL SNAPS!
    add_mono(mix, synth_pencil_snap(0.72), 11.55, pan=0.0)
    # 11.55 - 12.55s: Dead silence as Benny stares at broken pencil, then 12.55s "Technology."
    # 14.00s: Tiny eyebrow raise bass pluck after "Technology."
    add_mono(mix, synth_pizzicato(146.83, 0.20, 0.18), 14.02, pan=0.0)
    # 14.25 - 16.95s: Cascade of error notifications + frantic clicking!
    for i in range(18):
        et = 14.28 + i * 0.145
        add_mono(mix, synth_popup_beep(0.85 + (i % 5) * 0.12, 0.20), et, pan=((i % 3) - 1) * 0.35)
        add_mono(mix, synth_key_click(0.32), et + 0.06, pan=0.0)

    # =========================================================================
    # SCENE 3: THE BRANDRUMS DISCOVERY (17 - 26s)
    # =========================================================================
    # 17.05 - 18.10s: Frantic typing absurd characters
    for i in range(14):
        add_mono(mix, synth_key_click(0.34), 17.08 + i * 0.072, pan=((i % 2) - 0.5) * 0.2)
    # 18.10 - 18.50s: Pause, then rapid delete (18.50 - 19.05s), then deliberate click (19.25s)
    for i in range(7):
        add_mono(mix, synth_key_click(0.28), 18.52 + i * 0.075, pan=0.1)
    add_mono(mix, synth_key_click(0.50), 19.25, pan=0.0)
    # 19.40s: Room darkens, warm illumination pad + tie adjustment + click at 20.85s
    add_mono(mix, synth_warm_chord([220.0, 277.18, 329.63, 440.0], 1.6, 0.14), 19.40, pan=0.0)
    add_mono(mix, synth_key_click(0.45), 20.85, pan=0.0)
    # 21.00s: Magical digital chime as portal opens!
    add_mono(mix, synth_portal_chime(1.8, 0.32), 21.00, pan=0.0)
    # 22.35s: Pokes portal with finger (sci-fi bloop) -> pulls it out at 22.90s
    add_mono(mix, synth_gulp(0.30), 22.35, pan=0.1)
    add_mono(mix, synth_pizzicato(880.0, 0.12, 0.24), 22.90, pan=0.0)
    # 23.25 - 24.55s: Counts all 5 fingers with 5 ascending pizzicato notes!
    for f_i, f_hz in enumerate([523.25, 587.33, 659.25, 783.99, 1046.50]):
        add_mono(mix, synth_pizzicato(f_hz, 0.16, 0.22), 23.25 + f_i * 0.28, pan=-0.1 + f_i * 0.05)
    # 24.85s: Deep cinematic impact + playful orchestral transition into Scene 4!
    add_mono(mix, synth_cinematic_impact(1.8, 0.46), 24.85, pan=0.0)
    add_mono(mix, synth_portal_chime(1.5, 0.26), 25.05, pan=0.0)

    # =========================================================================
    # SCENE 4: THE IMPOSSIBLE TRANSFORMATION (26 - 41s)
    # =========================================================================
    # 26.0 - 31.0s: Playful orchestral & futuristic studio chords + floating UI whooshes
    studio_chords = [
        (26.0, [261.63, 329.63, 392.00, 523.25]),
        (28.0, [293.66, 369.99, 440.00, 587.33]),
        (30.0, [329.63, 415.30, 493.88, 659.25]),
    ]
    for ct, ch_f in studio_chords:
        add_mono(mix, synth_warm_chord(ch_f, 2.1, 0.13), ct, pan=0.0)
    for wt, wf in [(26.4, 700.0), (27.6, 950.0), (28.8, 1200.0), (30.1, 850.0)]:
        add_mono(mix, synth_whoosh(0.42, wf, 0.24), wt, pan=0.2)

    # 31.2s: Gag 1 — Benny swipes air too aggressively & spins himself around!
    add_mono(mix, synth_whoosh(0.55, 1400.0, 0.36), 31.20, pan=-0.3)
    for s_i in range(6):
        add_mono(mix, synth_pizzicato(587.33 + s_i * 75.0, 0.12, 0.18), 31.45 + s_i * 0.12, pan=math.sin(s_i))
    # 32.8s: Gag 2 — Taps floating smartphone & accidentally ducks as it rotates toward him!
    add_mono(mix, synth_pizzicato(1174.66, 0.12, 0.24), 32.80, pan=0.2)
    add_mono(mix, synth_whoosh(0.50, 520.0, 0.38), 33.10, pan=0.0)
    # 34.2s: Gag 3 — Straightens aligned panel, makes it crooked (34.3s), notices (34.8s), quietly fixes it (35.2s)!
    add_mono(mix, synth_glass_slip(0.24), 34.30, pan=-0.2)
    add_mono(mix, synth_pizzicato(311.13, 0.18, 0.20), 34.75, pan=-0.2)
    add_mono(mix, synth_pizzicato(1046.50, 0.22, 0.24), 35.25, pan=0.0)
    # 36.2s: Digital world locks into sophisticated final design!
    add_mono(mix, synth_warm_chord([261.63, 329.63, 392.0, 523.25, 659.25], 2.4, 0.18), 36.20, pan=0.0)
    add_mono(mix, synth_portal_chime(1.4, 0.22), 36.30, pan=0.0)
    # 37.5 - 40.8s: Removes glasses, wipes on tie (38.3s, 38.7s), puts back on (39.4s), silent nod of approval (40.2s)
    add_mono(mix, synth_pencil_scratch(0.26, 0.16, seed=91), 38.30, pan=0.0)
    add_mono(mix, synth_pencil_scratch(0.26, 0.16, seed=92), 38.70, pan=0.0)
    add_mono(mix, synth_key_click(0.25), 39.40, pan=0.0)
    add_mono(mix, synth_pizzicato(783.99, 0.28, 0.20), 40.20, pan=0.0)

    # =========================================================================
    # SCENE 5: THE PERFECT REVEAL (41 - 51s)
    # =========================================================================
    # Classy pizzicato bass & warm chord (41.0 - 44.5s)
    for bt, f in [(41.1, 130.81), (41.7, 164.81), (42.3, 196.00), (42.9, 261.63), (43.5, 196.00)]:
        add_mono(mix, synth_pizzicato(f, 0.30, 0.20), bt, pan=0.0)
    # 42.6s: Smartphone appears beside laptop with chime
    add_mono(mix, synth_portal_chime(0.9, 0.22), 42.60, pan=-0.25)
    # 45.10s: Handshake with smartphone screen -> hand slides off the glass!
    add_mono(mix, synth_glass_slip(0.42), 45.10, pan=-0.15)
    # 46.40s: Tie-straighten rustle before "Much better." at 47.35s
    add_mono(mix, synth_whoosh(0.20, 1500.0, 0.16), 46.40, pan=0.0)
    # 49.35s: Tiny satisfied smile pizzicato cadence after "Much better."
    add_mono(mix, synth_pizzicato(523.25, 0.20, 0.18), 49.35, pan=-0.1)
    add_mono(mix, synth_pizzicato(659.25, 0.28, 0.22), 49.60, pan=0.1)

    # =========================================================================
    # SCENE 6: BRANDRUMS END CARD (51 - 60s)
    # =========================================================================
    add_mono(mix, synth_warm_chord([261.63, 329.63, 392.0, 523.25], 2.8, 0.18), 51.00, pan=0.0)
    add_mono(mix, synth_portal_chime(1.6, 0.25), 53.85, pan=0.0)
    add_mono(mix, synth_warm_chord([261.63, 329.63, 392.0, 523.25, 783.99], 3.0, 0.16), 54.00, pan=0.0)
    # 56.8 - 60.0s: Final Gag — Benny sneaks back into frame, tries to straighten logo, backs away!
    for st, sf in [(56.85, 392.0), (57.15, 440.0), (57.45, 493.88)]:
        add_mono(mix, synth_pizzicato(sf, 0.14, 0.18), st, pan=0.25)
    # 58.15s: Realizes logo is already perfectly aligned (tiny ding!)
    add_mono(mix, synth_pizzicato(1318.51, 0.22, 0.22), 58.15, pan=0.0)
    # 58.65 - 59.45s: Slowly backs away with sneaky descending pizzicato plucks
    for st, sf in [(58.65, 493.88), (58.95, 440.0), (59.25, 392.0), (59.55, 261.63)]:
        add_mono(mix, synth_pizzicato(sf, 0.16, 0.18), st, pan=0.25)

    # Soft master limiter
    peak = np.max(np.abs(mix)) + 1e-9
    mix = np.tanh((mix / peak) * 1.15) * 0.92
    out_wav = os.path.join(ASSETS_DIR, "benny_soundtrack_master.wav")
    wavfile.write(out_wav, SR, (mix * 32767.0).astype(np.int16))
    print(f"Saved master soundtrack: {out_wav} ({os.path.getsize(out_wav)/1024/1024:.2f} MB)")

    # Build 1,800-frame Lip-Sync Envelope (mouth_open, mouth_wide) from vo_track
    env = np.zeros((TOTAL_FRAMES, 2), dtype=np.float32)
    spf = SR // FPS
    for f_i in range(TOTAL_FRAMES):
        chunk = vo_track[f_i * spf:(f_i + 1) * spf]
        rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
        m_open = min(1.0, max(0.0, (rms - 0.012) * 5.2))
        zc = float(np.mean(np.abs(np.diff(np.sign(chunk)))))
        m_wide = min(1.0, max(0.25, 0.40 + zc * 1.6))
        env[f_i, 0] = m_open
        env[f_i, 1] = m_wide
    env[:, 0] = np.convolve(env[:, 0], [0.2, 0.6, 0.2], mode="same")
    out_npy = os.path.join(ASSETS_DIR, "benny_lipsync_envelope.npy")
    np.save(out_npy, env)
    print(f"Saved lip-sync envelope: {out_npy} ({env.shape})")

if __name__ == "__main__":
    main()
