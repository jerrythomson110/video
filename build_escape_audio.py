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

def synth_clock_tick(is_tock=False, amp=0.14, pan=0.5):
    dur = 0.055
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    f = 1150.0 if not is_tock else 860.0
    sig = (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.4 * t)) * np.exp(-t * 85.0) * amp
    return sig * (1.0 - 0.4 * pan), sig * (0.6 + 0.4 * pan)

def synth_footstep(step_idx=0, amp=0.12, pan=0.5):
    dur = 0.065
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(100 + step_idx)
    noise = rng.standard_normal((n, 2))
    noise = lowpass(highpass(noise, 320), 2200)
    pitch = 420.0 + (step_idx % 3) * 65.0
    pop = np.sin(2 * np.pi * pitch * np.exp(-t * 18.0) * t) * np.exp(-t * 55.0)
    env = np.exp(-t * 48.0)
    l = (noise[:, 0] * 0.45 + pop) * env * amp * (1.0 - 0.5 * pan)
    r = (noise[:, 1] * 0.45 + pop) * env * amp * (0.5 + 0.5 * pan)
    return l, r

def synth_jump_boing(dur=0.28, amp=0.15):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    freqs = np.linspace(260.0, 680.0, len(t))
    phase = 2 * np.pi * np.cumsum(freqs) / SR
    sig = np.sin(phase) * np.exp(-t * 9.5) * amp
    return sig, sig

def synth_sub_impact(start_f=130.0, end_f=38.0, dur=1.4, amp=0.35):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    freqs = np.linspace(start_f, end_f, len(t))
    phase = 2 * np.pi * np.cumsum(freqs) / SR
    env = np.exp(-t * 3.0) * np.minimum(1.0, t / 0.012)
    sig = (np.sin(phase) + 0.3 * np.tanh(2.5 * np.sin(phase))) * env * amp
    return sig, sig

def synth_whoosh(dur, low_f=200, high_f=3400, amp=0.18, pan_dir=1):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(77)
    noise = rng.standard_normal((n, 2))
    noise = lowpass(highpass(noise, low_f), high_f)
    env = np.sin(np.pi * (t / dur)) ** 2.0
    pan = np.linspace(0.2, 0.85, n) if pan_dir > 0 else np.linspace(0.85, 0.2, n)
    l = noise[:, 0] * env * (1.0 - pan * 0.5) * amp
    r = noise[:, 1] * env * (0.5 + pan * 0.5) * amp
    return l, r

def synth_pad_chord(freqs, dur, attack=0.3, release=0.6, amp=0.14, bright=2200.0):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    sig_l = np.zeros_like(t)
    sig_r = np.zeros_like(t)
    for i, f in enumerate(freqs):
        det = 1.0025 if i % 2 == 0 else 0.9975
        w1 = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.0 * t)
        w2 = np.sin(2 * np.pi * (f * det) * t) + 0.35 * np.sin(2 * np.pi * (f * det) * 2.0 * t)
        saw = 0.16 * (2.0 * (t * f - np.floor(0.5 + t * f)))
        sig_l += (w1 + saw) / len(freqs)
        sig_r += (w2 + saw) / len(freqs)
    env = np.ones_like(t)
    att_n = int(attack * SR)
    rel_n = int(release * SR)
    if att_n > 0:
        env[:att_n] = np.linspace(0, 1, att_n) ** 1.4
    if rel_n > 0 and rel_n < len(t):
        env[-rel_n:] = np.linspace(1, 0, rel_n) ** 1.6
    stereo = np.stack([sig_l * env, sig_r * env], axis=1) * amp
    return lowpass(stereo, bright)

def synth_ui_chime(freqs, dur=0.35, amp=0.13, pan=0.5):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    sig = np.zeros_like(t)
    for idx, f in enumerate(freqs):
        delay = idx * 0.045
        mask = t >= delay
        td = t[mask] - delay
        tone = (np.sin(2 * np.pi * f * td) + 0.28 * np.sin(2 * np.pi * f * 3.0 * td)) * np.exp(-td * 10.5)
        tone *= np.minimum(1.0, td / 0.004)
        sig[mask] += tone
    l = sig * amp * (1.0 - 0.4 * pan)
    r = sig * amp * (0.6 + 0.4 * pan)
    return l, r

def main():
    mix = np.zeros((N_SAMPLES, 2), dtype=np.float32)

    # =========================================================================
    # 0 - 5s: THE LEAD THAT ALMOST ESCAPED (Ticking clock + Jump + Footsteps)
    # =========================================================================
    # Playful pizzicato/marimba tension bed
    pizz_notes = [293.66, 349.23, 440.0, 392.0, 293.66, 349.23, 466.16, 440.0]
    for i in range(18):
        pt = 0.15 + i * 0.26
        f = pizz_notes[i % len(pizz_notes)]
        pl, pr = synth_ui_chime([f], dur=0.16, amp=0.06, pan=0.3 + 0.4 * (i % 2))
        add_stereo(mix, pl, pr, pt)

    # Ticking clock (0.8s -> 2.8s)
    for k in range(10):
        ct = 0.80 + k * 0.20
        tl, tr = synth_clock_tick(is_tock=(k % 2 == 1), amp=0.15, pan=0.45)
        add_stereo(mix, tl, tr, ct)

    # 2.55s: Customer sprouts legs & jumps out of website!
    jl, jr = synth_jump_boing(dur=0.28, amp=0.16)
    add_stereo(mix, jl, jr, 2.55)

    # 2.80s -> 4.90s: Playful rapid footsteps running across polished floor (panning left -> right)
    for s in range(18):
        st = 2.82 + s * 0.115
        pan = min(0.9, 0.25 + s * 0.038)
        fl, fr = synth_footstep(step_idx=s, amp=0.14, pan=pan)
        add_stereo(mix, fl, fr, st)

    # =========================================================================
    # 5 - 9s: THE BRANDRUMS "B" HERO INTERCEPTS! (Streak + Impact + Unfold)
    # =========================================================================
    wl, wr = synth_whoosh(0.65, low_f=250, high_f=4200, amp=0.24, pan_dir=-1)
    add_stereo(mix, wl, wr, 4.85)
    il, ir = synth_sub_impact(start_f=135.0, end_f=38.0, dur=1.5, amp=0.36)
    add_stereo(mix, il, ir, 5.05)

    # Heroic Brandrums B chord + ribbon unfold shimmer
    pad_hero = synth_pad_chord([73.42, 146.83, 220.0, 293.66, 369.99, 440.0], 3.9, attack=0.15, release=0.5, amp=0.18, bright=2500)
    add_stereo(mix, pad_hero[:, 0], pad_hero[:, 1], 5.10)
    rl, rr = synth_ui_chime([587.33, 880.0, 1174.66, 1479.98], dur=0.75, amp=0.13, pan=0.5)
    add_stereo(mix, rl, rr, 6.10)

    # =========================================================================
    # 9 - 19s: CONTINUOUS 3D ACTION SEQUENCE (4 Software Stations + Upbeat Groove)
    # =========================================================================
    pad_s3a = synth_pad_chord([98.0, 196.0, 246.94, 293.66, 369.99], 5.2, attack=0.3, release=0.5, amp=0.15, bright=2200)
    pad_s3b = synth_pad_chord([110.0, 220.0, 277.18, 329.63, 440.0], 5.2, attack=0.3, release=0.5, amp=0.16, bright=2600)
    add_stereo(mix, pad_s3a[:, 0], pad_s3a[:, 1], 9.0)
    add_stereo(mix, pad_s3b[:, 0], pad_s3b[:, 1], 14.0)

    # Upbeat 125 BPM electronic arpeggio (9.0s -> 19.0s)
    arp = [293.66, 369.99, 440.0, 587.33, 659.25, 587.33, 440.0, 369.99]
    for k in range(42):
        at = 9.0 + k * 0.24
        al, ar = synth_ui_chime([arp[k % len(arp)]], dur=0.19, amp=0.055, pan=0.3 if k % 2 == 0 else 0.7)
        add_stereo(mix, al, ar, at)

    # Station 1 (9.0-11.5s): CAPTURE (9.1s) & RESPOND (10.3s)
    c1l, c1r = synth_ui_chime([587.33, 880.0], dur=0.35, amp=0.14, pan=0.35)
    add_stereo(mix, c1l, c1r, 9.10)
    c2l, c2r = synth_ui_chime([739.99, 1108.73], dur=0.35, amp=0.14, pan=0.65)
    add_stereo(mix, c2l, c2r, 10.35)

    # Station 2 (11.5-14.0s): Conveyor whoosh + CRM Contact Created click
    w_crm_l, w_crm_r = synth_whoosh(0.55, low_f=300, high_f=2600, amp=0.13, pan_dir=1)
    add_stereo(mix, w_crm_l, w_crm_r, 11.45)
    c_crm_l, c_crm_r = synth_ui_chime([659.25, 987.77, 1318.51], dur=0.4, amp=0.13, pan=0.45)
    add_stereo(mix, c_crm_l, c_crm_r, 12.10)

    # Station 3 (14.0-16.5s): BOOK (Calendar slot selected + confirmation email chime)
    c3l, c3r = synth_ui_chime([783.99, 1174.66], dur=0.38, amp=0.14, pan=0.4)
    add_stereo(mix, c3l, c3r, 14.10)
    c3b_l, c3b_r = synth_ui_chime([1046.50, 1318.51], dur=0.35, amp=0.12, pan=0.6)
    add_stereo(mix, c3b_l, c3b_r, 15.25)

    # Station 4 (16.5-19.0s): FOLLOW UP (Automated sequence lock chime)
    c4l, c4r = synth_ui_chime([880.0, 1108.73, 1479.98], dur=0.45, amp=0.15, pan=0.5)
    add_stereo(mix, c4l, c4r, 16.60)

    # =========================================================================
    # 19 - 25s: ECOSYSTEM RUNNING SMOOTHLY (Uplifting warm chord progression)
    # =========================================================================
    w19l, w19r = synth_whoosh(0.8, low_f=200, high_f=2800, amp=0.15, pan_dir=-1)
    add_stereo(mix, w19l, w19r, 18.80)
    pad_s4 = synth_pad_chord([73.42, 146.83, 220.0, 293.66, 369.99, 440.0, 587.33], 6.0, attack=0.4, release=0.7, amp=0.19, bright=3000)
    add_stereo(mix, pad_s4[:, 0], pad_s4[:, 1], 19.0)

    for idx, gt in enumerate([19.8, 21.0, 22.2, 23.4]):
        gl, gr = synth_ui_chime([587.33 * (1.122 ** idx), 880.0 * (1.122 ** idx)], dur=0.35, amp=0.08, pan=0.3 + 0.15 * idx)
        add_stereo(mix, gl, gr, gt)

    # =========================================================================
    # 25 - 30s: FINAL CONFIDENT SPIN + BRANDRUMS LOGO REVEAL + SONIC LOGO
    # =========================================================================
    w25l, w25r = synth_whoosh(0.95, low_f=250, high_f=3800, amp=0.20, pan_dir=1)
    add_stereo(mix, w25l, w25r, 25.00)
    i26l, i26r = synth_sub_impact(start_f=115.0, end_f=42.0, dur=1.5, amp=0.30)
    add_stereo(mix, i26l, i26r, 25.85)

    pad_s5 = synth_pad_chord([73.42, 146.83, 220.0, 293.66, 369.99, 587.33], 5.0, attack=0.2, release=1.4, amp=0.18, bright=3200)
    add_stereo(mix, pad_s5[:, 0], pad_s5[:, 1], 25.0)

    # Memorable Final Brandrums Sonic Logo (27.4s -> 29.8s)
    s1l, s1r = synth_ui_chime([440.0, 880.0], dur=0.5, amp=0.16, pan=0.4)
    s2l, s2r = synth_ui_chime([587.33, 1174.66], dur=0.6, amp=0.17, pan=0.5)
    s3l, s3r = synth_ui_chime([739.99, 1174.66, 1479.98], dur=1.6, amp=0.21, pan=0.5)
    add_stereo(mix, s1l, s1r, 27.45)
    add_stereo(mix, s2l, s2r, 27.68)
    add_stereo(mix, s3l, s3r, 27.92)

    # =========================================================================
    # MIX IN AMERICAN ENGLISH VOICEOVER CLIPS
    # =========================================================================
    vo_placements = [
        ("/home/user/video/assets/vo2_part1.mp3", 0.95),   # "Leads shouldn't slip away while you're busy." (0.95s - 4.11s)
        ("/home/user/video/assets/vo2_part2.mp3", 9.35),   # "Let AI respond, organize, book, and follow up automatically." (9.35s - 14.60s)
        ("/home/user/video/assets/vo2_part3.mp3", 19.55),  # "Build smarter workflows with Brandrums." (19.55s - 22.40s)
    ]

    vo_track = np.zeros_like(mix)
    duck_env = np.ones(N_SAMPLES, dtype=np.float32)

    for path, start_t in vo_placements:
        clip = load_mp3_to_stereo(path)
        peak = np.max(np.abs(clip)) + 1e-6
        clip = (clip / peak) * 0.90
        idx = int(start_t * SR)
        n = min(len(clip), N_SAMPLES - idx)
        vo_track[idx:idx+n] += clip[:n]
        pad_s = int(0.18 * SR)
        d0 = max(0, idx - pad_s)
        d1 = min(N_SAMPLES, idx + n + pad_s)
        duck_env[d0:d1] = 0.42

    kernel = np.ones(int(0.15 * SR)) / int(0.15 * SR)
    duck_env = np.convolve(duck_env, kernel, mode='same')

    final_mix = mix * duck_env[:, None] + vo_track
    final_mix = np.tanh(final_mix * 1.15) * 0.92
    out_int16 = (np.clip(final_mix, -1.0, 1.0) * 32767).astype(np.int16)
    out_path = "/home/user/video/assets/escape_soundtrack_master.wav"
    wavfile.write(out_path, SR, out_int16)
    print(f"Saved escape soundtrack to {out_path} ({len(out_int16)/SR:.2f}s)")

if __name__ == "__main__":
    main()
