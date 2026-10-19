import os
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import imageio_ffmpeg

SR = 48000
FPS = 30
DURATION = 30.0
N_SAMPLES = int(SR * DURATION)
TOTAL_FRAMES = int(FPS * DURATION)

def lowpass(sig, cutoff, sr=SR, order=4):
    b, a = butter(order, cutoff / (0.5 * sr), btype='low')
    return lfilter(b, a, sig, axis=0)

def highpass(sig, cutoff, sr=SR, order=2):
    b, a = butter(order, cutoff / (0.5 * sr), btype='high')
    return lfilter(b, a, sig, axis=0)

def bandpass(sig, low_f, high_f, sr=SR, order=3):
    b, a = butter(order, [low_f / (0.5 * sr), high_f / (0.5 * sr)], btype='band')
    return lfilter(b, a, sig, axis=0)

def load_and_compact_vo(path, target_dur):
    """
    Loads an MP3 voice clip, compacts dead silence gaps > 90ms so pacing is tight and natural,
    and applies ffmpeg atempo + subtle warm rasp EQ to fit target_dur cleanly.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    raw_wav = path + ".raw.wav"
    subprocess.run([
        ffmpeg, "-y", "-i", path, "-ar", str(SR), "-ac", "1", "-f", "wav", raw_wav
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    _, data = wavfile.read(raw_wav)
    os.remove(raw_wav)
    sig = data.astype(np.float32) / 32768.0

    # Compact long silence gaps (> 80ms) down to 55ms so speech is snappy
    win = int(0.015 * SR)
    n_wins = len(sig) // win
    kept = []
    silent_run = 0
    for i in range(n_wins):
        chunk = sig[i*win:(i+1)*win]
        rms = np.sqrt(np.mean(chunk * chunk))
        if rms < 0.012:
            silent_run += 1
            if silent_run <= 4:  # keep up to 60ms of pause
                kept.append(chunk)
        else:
            silent_run = 0
            kept.append(chunk)
    compact = np.concatenate(kept) if kept else sig
    cur_dur = len(compact) / float(SR)

    tempo = max(1.0, min(1.95, cur_dur / target_dur))
    comp_wav = path + ".comp.wav"
    out_wav = path + ".tempo.wav"
    wavfile.write(comp_wav, SR, (np.clip(compact, -1.0, 1.0) * 32767).astype(np.int16))

    # Slight pitch/presence boost for energetic young hype-announcer rasp
    subprocess.run([
        ffmpeg, "-y", "-i", comp_wav,
        "-filter:a", f"atempo={tempo:.3f},trebl=g=3:f=3200,acompressor=threshold=-16dB:ratio=3:makeup=2dB" if False else f"atempo={tempo:.3f}",
        "-ar", str(SR), "-ac", "1", out_wav
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    _, stretched = wavfile.read(out_wav)
    os.remove(comp_wav)
    os.remove(out_wav)
    mono = stretched.astype(np.float32) / 32768.0

    # Add subtle studio harmonic rasp & crisp presence
    presence = highpass(mono, 1800.0)
    mono = mono + 0.22 * np.tanh(presence * 3.2)
    peak = np.max(np.abs(mono)) + 1e-6
    mono = (mono / peak) * 0.88
    return np.stack([mono, mono], axis=1)

def add_stereo(mix, sig_l, sig_r, start_sec):
    idx = int(start_sec * SR)
    n = min(len(sig_l), N_SAMPLES - idx)
    if n <= 0 or idx < 0:
        return
    mix[idx:idx+n, 0] += sig_l[:n]
    mix[idx:idx+n, 1] += sig_r[:n]

def synth_snare(amp=0.65, pan=0.5):
    dur = 0.26
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    # Body tone (185 Hz -> 115 Hz) + crisp snare wire noise
    body_f = 115.0 + 75.0 * np.exp(-t * 38.0)
    phase = 2 * np.pi * np.cumsum(body_f) / SR
    body = np.sin(phase) * np.exp(-t * 24.0)
    rng = np.random.default_rng(101)
    wires = rng.standard_normal(n)
    wires = bandpass(wires, 900.0, 9500.0) * np.exp(-t * 15.5)
    snap = np.exp(-t * 120.0) * rng.standard_normal(n) * 0.6
    sig = (body * 0.55 + wires * 0.65 + snap) * amp
    return sig * (1.0 - 0.35 * pan), sig * (0.65 + 0.35 * pan)

def synth_rimshot(amp=0.60, pan=0.55):
    dur = 0.14
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    # Sharp woody stick click + high ring (480 Hz + 1240 Hz + 2350 Hz)
    wood = (
        np.sin(2 * np.pi * 485.0 * t) * 0.6 +
        np.sin(2 * np.pi * 1260.0 * t) * 0.5 +
        np.sin(2 * np.pi * 2380.0 * t) * 0.35
    ) * np.exp(-t * 48.0)
    rng = np.random.default_rng(202)
    crack = bandpass(rng.standard_normal(n), 1500.0, 8000.0) * np.exp(-t * 65.0)
    sig = np.tanh((wood + crack * 0.7) * 1.8) * amp
    return sig * (1.0 - 0.35 * pan), sig * (0.65 + 0.35 * pan)

def synth_kick(amp=0.75):
    dur = 0.35
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    freqs = 46.0 + 115.0 * np.exp(-t * 34.0)
    phase = 2 * np.pi * np.cumsum(freqs) / SR
    beater = np.exp(-t * 95.0) * np.sin(2 * np.pi * 1800.0 * t) * 0.25
    sig = (np.sin(phase) + 0.25 * np.tanh(2.5 * np.sin(phase)) + beater) * np.exp(-t * 10.5) * amp
    return sig, sig

def synth_tom(pitch=140.0, amp=0.55, pan=0.5):
    dur = 0.28
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    freqs = pitch + (pitch * 0.45) * np.exp(-t * 28.0)
    phase = 2 * np.pi * np.cumsum(freqs) / SR
    sig = np.sin(phase) * np.exp(-t * 14.0) * amp
    return sig * (1.0 - 0.4 * pan), sig * (0.6 + 0.4 * pan)

def synth_cymbal_crash(dur=1.85, amp=0.52, pan=0.6):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(303)
    # Metallic inharmonic modes + high shimmer noise
    metal = np.zeros(n, dtype=np.float32)
    for f in [3120.0, 4280.0, 5890.0, 7410.0, 9120.0, 11350.0]:
        metal += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * 0.12
    shimmer = highpass(rng.standard_normal((n, 2)), 2600.0)
    env = np.minimum(1.0, t / 0.008) * (0.65 * np.exp(-t * 2.4) + 0.35 * np.exp(-t * 0.9))
    l = (shimmer[:, 0] * 0.55 + metal) * env * amp * (1.0 - 0.3 * pan)
    r = (shimmer[:, 1] * 0.55 + metal) * env * amp * (0.7 + 0.3 * pan)
    return l, r

def main():
    mix = np.zeros((N_SAMPLES, 2), dtype=np.float32)
    vo_only = np.zeros((N_SAMPLES, 2), dtype=np.float32)

    # 1. Soft charcoal studio room tone (-42 dB warm air, NO music!)
    rng = np.random.default_rng(42)
    room = lowpass(highpass(rng.standard_normal((N_SAMPLES, 2)), 70.0), 950.0) * 0.0045
    mix += room

    # 2. Load and time-fit all 7 dialogue clips to their exact shot windows
    clips = [
        # (path, start_sec, target_dur)
        ("/home/user/video/assets/boom_vo_1.mp3", 1.12, 2.72),   # 1.12s - 3.84s (Shot 2: 1-4s)
        ("/home/user/video/assets/boom_vo_2.mp3", 4.30, 2.95),   # 4.30s - 7.25s (Shot 3: 4-8s)
        ("/home/user/video/assets/boom_vo_3.mp3", 8.42, 4.35),   # 8.42s - 12.77s (Shot 4: 8-13s)
        ("/home/user/video/assets/boom_vo_4.mp3", 13.55, 4.25),  # 13.55s - 17.80s (Shot 5: 13-18s)
        ("/home/user/video/assets/boom_vo_5.mp3", 18.18, 4.60),  # 18.18s - 22.78s (Shot 6: 18-23s)
        ("/home/user/video/assets/boom_vo_6.mp3", 23.12, 3.72),  # 23.12s - 26.84s (Shot 7: 23-27s)
        ("/home/user/video/assets/boom_vo_7.mp3", 27.42, 2.02),  # 27.42s - 29.44s (Shot 8: 27-30s, freeze 29.45-30s)
    ]

    for path, st, td in clips:
        vo_c = load_and_compact_vo(path, td)
        add_stereo(vo_only, vo_c[:, 0], vo_c[:, 1], st)
        print(f"Placed {os.path.basename(path)} at {st:.2f}s -> {st + len(vo_c)/SR:.2f}s")

    # 3. Synthesize exact drum hits, rim-shots, fills & cymbal crashes on the beats described
    # 0-1s: One loud snare hit at 0.22s as Boom slams into frame
    sl, sr_c = synth_snare(amp=0.82, pan=0.5)
    kl, kr = synth_kick(amp=0.55)
    add_stereo(mix, sl, sr_c, 0.22)
    add_stereo(mix, kl, kr, 0.22)

    # 1-4s: Drumstick twirl clicks (1.05s, 1.18s)
    r1l, r1r = synth_rimshot(amp=0.22, pan=0.4)
    add_stereo(mix, r1l, r1r, 1.05)
    add_stereo(mix, r1l * 0.8, r1r * 0.8, 1.16)

    # 4-8s: Hop onto laptop keyboard (4.12s), then on "NOISE" (6.82s) smashes cymbal!
    kl_hop, kr_hop = synth_tom(pitch=165.0, amp=0.35, pan=0.45)
    add_stereo(mix, kl_hop, kr_hop, 4.12)
    cym_l, cym_r = synth_cymbal_crash(dur=1.9, amp=0.68, pan=0.65)
    k_noise_l, k_noise_r = synth_kick(amp=0.72)
    s_noise_l, s_noise_r = synth_snare(amp=0.65, pan=0.55)
    add_stereo(mix, cym_l, cym_r, 6.82)
    add_stereo(mix, k_noise_l, k_noise_r, 6.82)
    add_stereo(mix, s_noise_l, s_noise_r, 6.82)

    # 8-13s: Walks across desk to smartphone, taps with stick (8.28s), rim-shot on each flaw!
    for step_t in [8.04, 8.14, 8.24]:
        tl, tr = synth_rimshot(amp=0.18, pan=0.5)
        add_stereo(mix, tl, tr, step_t)
    # 4 Rim-shots on the 4 flaws: "No logo." / "Stock photo." / "Button goes nowhere." / "Loads like it's 2009."
    for rim_t in [9.12, 10.12, 11.28, 12.55]:
        rl, rr = synth_rimshot(amp=0.62, pan=0.55)
        add_stereo(mix, rl, rr, rim_t)

    # 13-18s: Rapid drum fill (13.00s - 13.48s), then drum hit on each of the 5 phrases!
    fill_hits = [
        (13.00, "snare", 200),
        (13.09, "tom", 190),
        (13.18, "tom", 155),
        (13.27, "snare", 180),
        (13.36, "tom", 120),
        (13.45, "kick", 60),
    ]
    for ft, ftype, fpitch in fill_hits:
        if ftype == "snare":
            fl, fr = synth_snare(amp=0.62, pan=0.45)
        elif ftype == "tom":
            fl, fr = synth_tom(pitch=fpitch, amp=0.60, pan=0.55)
        else:
            fl, fr = synth_kick(amp=0.72)
        add_stereo(mix, fl, fr, ft)

    # 5 Drum hits on each phrase ("Custom logo." / "Mobile first." / "Forms that work." / "SEO baked in." / "Unlimited revisions.")
    for idx_h, hit_t in enumerate([14.15, 14.95, 15.80, 16.65, 17.58]):
        kl_h, kr_h = synth_kick(amp=0.62)
        sl_h, sr_h = synth_snare(amp=0.52, pan=0.4 + 0.05 * idx_h)
        add_stereo(mix, kl_h, kr_h, hit_t)
        add_stereo(mix, sl_h, sr_h, hit_t)

    # 18-23s: Counts on 3 gloved fingers with a snare hit each!
    for snare_t in [18.28, 19.62, 20.98]:
        sl_f, sr_f = synth_snare(amp=0.68, pan=0.5)
        add_stereo(mix, sl_f, sr_f, snare_t)
    # Accent at "Pick one, we do the rest" (22.55s)
    rl_end, rr_end = synth_rimshot(amp=0.48, pan=0.5)
    add_stereo(mix, rl_end, rr_end, 22.55)

    # 23-27s: Subtle stick tap as Boom leans on laptop edge (23.08s)
    tl_l, tr_l = synth_rimshot(amp=0.25, pan=0.35)
    add_stereo(mix, tl_l, tr_l, 23.08)

    # 27-30s: Drum roll (27.00s - 27.38s) into cymbal crash (27.38s), then hits kick drum once (28.65s)!
    for r_i in range(11):
        rt = 27.00 + r_i * 0.034
        amp_r = 0.25 + 0.038 * r_i
        sl_r, sr_r = synth_snare(amp=amp_r, pan=0.45 + 0.1 * (r_i % 2))
        add_stereo(mix, sl_r, sr_r, rt)
    cym2_l, cym2_r = synth_cymbal_crash(dur=1.8, amp=0.62, pan=0.6)
    add_stereo(mix, cym2_l, cym2_r, 27.38)

    # Single punchy kick drum hit at 28.65s ("Brandrums. Make some noise.")
    k_final_l, k_final_r = synth_kick(amp=0.85)
    add_stereo(mix, k_final_l, k_final_r, 28.65)

    # 4. Extract frame-accurate 30fps lip-sync envelope from vo_only!
    samples_per_frame = SR // FPS
    lipsync = np.zeros((TOTAL_FRAMES, 2), dtype=np.float32)
    vo_mono = vo_only[:, 0]
    vo_low = lowpass(vo_mono, 750.0)
    vo_high = highpass(vo_mono, 1400.0)

    for f_idx in range(TOTAL_FRAMES):
        s0 = f_idx * samples_per_frame
        s1 = min(N_SAMPLES, s0 + samples_per_frame)
        chunk = vo_mono[s0:s1]
        chunk_l = vo_low[s0:s1]
        chunk_h = vo_high[s0:s1]
        rms = np.sqrt(np.mean(chunk * chunk))
        rms_l = np.sqrt(np.mean(chunk_l * chunk_l))
        rms_h = np.sqrt(np.mean(chunk_h * chunk_h))

        # Mouth open driven by overall + low vowel energy
        open_val = np.clip((rms - 0.015) * 6.2 + rms_l * 2.5, 0.0, 1.0)
        # Mouth width driven by high-frequency sibilants/spread vowels ('EE', 'S', 'AY')
        wide_val = np.clip(0.45 + (rms_h - rms_l * 0.4) * 3.5, 0.15, 1.0)
        lipsync[f_idx, 0] = open_val
        lipsync[f_idx, 1] = wide_val

    # Smooth lip-sync slightly over 2 frames for natural jaw inertia, zero open during final freeze (29.48s+)
    lipsync[:, 0] = np.convolve(lipsync[:, 0], [0.2, 0.6, 0.2], mode='same')
    lipsync[int(29.48 * FPS):, 0] = 0.0
    np.save("/home/user/video/assets/boom_lipsync_envelope.npy", lipsync)

    # Combine drum/room mix + voiceover
    final_mix = mix + vo_only
    final_mix = np.tanh(final_mix * 1.12) * 0.93
    out_int16 = (np.clip(final_mix, -1.0, 1.0) * 32767).astype(np.int16)
    out_path = "/home/user/video/assets/boom_soundtrack_master.wav"
    wavfile.write(out_path, SR, out_int16)
    print(f"Saved {out_path} ({len(out_int16)/SR:.2f}s) and 900-frame lip-sync envelope!")

if __name__ == "__main__":
    main()
