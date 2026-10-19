import os
import sys
import math
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

W, H = 1080, 1920
FPS = 30
DURATION = 30.0
TOTAL_FRAMES = int(FPS * DURATION)

ASSETS_DIR = "/home/user/video/assets"
FONT_DIR = os.path.join(ASSETS_DIR, "fonts")

def get_font(weight=700, size=48):
    path = os.path.join(FONT_DIR, f"inter-latin-{weight}-normal.woff")
    if not os.path.exists(path):
        path = os.path.join(FONT_DIR, "inter-latin-700-normal.woff")
    return ImageFont.truetype(path, size)

def ease_out_cubic(x):
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3

def smoothstep(edge0, edge1, x):
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0 + 1e-9)))
    return t * t * (3.0 - 2.0 * t)

def load_and_fit_9_16(path):
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    target_ratio = W / H
    cur_ratio = iw / ih
    if cur_ratio > target_ratio:
        new_w = int(ih * target_ratio)
        left = (iw - new_w) // 2
        im = im.crop((left, 0, left + new_w, ih))
    elif cur_ratio < target_ratio:
        new_h = int(iw / target_ratio)
        top = (ih - new_h) // 2
        im = im.crop((0, top, iw, top + new_h))
    im = im.resize((W, H), Image.Resampling.LANCZOS)
    return np.array(im, dtype=np.uint8)

def camera_transform(img_rgb, zoom=1.0, pan_x=0.0, pan_y=0.0, roll_deg=0.0):
    cx = W / 2.0 + pan_x
    cy = H / 2.0 + pan_y
    M = cv2.getRotationMatrix2D((cx, cy), roll_deg, zoom)
    M[0, 2] += (W / 2.0) - cx
    M[1, 2] += (H / 2.0) - cy
    return cv2.warpAffine(img_rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def prepare_official_logo_and_b_hero(target_w=760):
    """
    Extracts both:
    1. The full official Brandrums logo (B + randrums)
    2. The standalone expressive 3D red ribbon 'B' character preserving exact silhouette, proportions, and red gradient.
    """
    raw = Image.open(os.path.join(ASSETS_DIR, "brandrums_official_logo.png")).convert("RGBA")
    rw, rh = raw.size
    scale = target_w / rw
    th = int(round(rh * scale))

    up = raw.resize((target_w * 2, th * 2), Image.Resampling.BICUBIC)
    arr = np.array(up, dtype=np.float32)
    alpha = arr[:, :, 3]
    alpha_blur = cv2.GaussianBlur(alpha, (9, 9), 2.2)
    alpha_sharp = np.clip((alpha_blur - 110.0) * (255.0 / 35.0), 0, 255)
    arr[:, :, 3] = alpha_sharp

    clean_rgba = Image.fromarray(arr.astype(np.uint8), "RGBA").resize((target_w, th), Image.Resampling.LANCZOS)
    c_arr = np.array(clean_rgba, dtype=np.float32)

    split_x = int(target_w * 0.295)
    yy, xx = np.mgrid[0:th, 0:target_w]
    norm_y = yy / float(th)

    # Enhance glossy red lacquer highlights on the 'B' ribbon symbol while preserving exact red gradient
    b_rgb = c_arr[:, :split_x, :3]
    b_luma = b_rgb[:, :, 0] / 255.0
    c_arr[:, :split_x, 0] = np.clip(b_rgb[:, :, 0] * 1.12 + 16.0, 95, 255)
    c_arr[:, :split_x, 1] = np.clip(b_rgb[:, :, 1] * 1.1 + (b_luma ** 3) * 42.0, 0, 88)
    c_arr[:, :split_x, 2] = np.clip(b_rgb[:, :, 2] * 1.1 + (b_luma ** 3) * 38.0, 0, 82)

    # Studio-illuminated 'randrums' wordmark
    w_alpha = c_arr[:, split_x:, 3] / 255.0
    grad_y = cv2.Sobel(w_alpha, cv2.CV_32F, 0, 1, ksize=3)
    grad_x = cv2.Sobel(w_alpha, cv2.CV_32F, 1, 0, ksize=3)
    bevel_light = np.clip(-grad_y * 0.7 - grad_x * 0.3, 0.0, 1.0)
    wy = norm_y[:, split_x:]
    wx = (xx[:, split_x:] - split_x) / float(target_w - split_x)
    spec = (1.0 - wy) ** 2.2 * 0.38 + bevel_light * 0.45
    c_arr[:, split_x:, 0] = np.clip(248.0 - wx * 28.0 + spec * 60.0, 0, 255)
    c_arr[:, split_x:, 1] = np.clip(28.0 + spec * 210.0, 0, 245)
    c_arr[:, split_x:, 2] = np.clip(26.0 + spec * 210.0, 0, 245)

    full_logo = c_arr.astype(np.uint8)
    b_only = full_logo[:, :split_x].copy()
    bh, bw = b_only.shape[:2]
    hero_h = 320
    hero_w = int(round(bw * (hero_h / float(bh))))
    b_hero = cv2.resize(b_only, (hero_w, hero_h), interpolation=cv2.INTER_CUBIC)
    return full_logo, b_hero

def render_b_hero_character(
    frame_rgb, b_hero_rgba, cx, cy,
    scale_x=1.0, scale_y=1.0, tilt_deg=0.0, spin_deg=0.0,
    ribbon_unfold=0.0, floor_reflection=True, opacity=1.0
):
    """
    Renders the official Brandrums 'B' ribbon symbol as an expressive, dimensional 3D animated hero character.
    """
    if opacity <= 0.01:
        return frame_rgb

    bh, bw = b_hero_rgba.shape[:2]
    work = b_hero_rgba.copy()

    # Dynamic ribbon curve unfolding wave across the 3 horizontal ribbon folds of the B
    if ribbon_unfold > 0.01:
        pad_u = int(60 * ribbon_unfold)
        unfolded = np.zeros((bh, bw + pad_u * 2, 4), dtype=np.uint8)
        for y in range(bh):
            band_wave = math.sin((y / float(bh)) * math.pi * 3.0)
            shift_x = int(pad_u + band_wave * 32.0 * ribbon_unfold)
            x0 = max(0, min(bw + pad_u * 2 - bw, shift_x))
            unfolded[y, x0:x0+bw] = work[y]
        work = unfolded
        bh, bw = work.shape[:2]

    cos_s = math.cos(math.radians(spin_deg))
    sin_s = math.sin(math.radians(spin_deg))
    spin_w_factor = max(0.18, abs(cos_s))

    sw = max(20, int(bw * scale_x * spin_w_factor))
    sh = max(20, int(bh * scale_y))
    scaled = cv2.resize(work, (sw, sh), interpolation=cv2.INTER_AREA if (scale_x * spin_w_factor) < 1.0 else cv2.INTER_CUBIC)
    if cos_s < 0:
        scaled = cv2.flip(scaled, 1)

    pad = 90
    cw, ch = sw + pad * 2, sh + pad * 2
    canvas = np.zeros((ch, cw, 4), dtype=np.float32)

    alpha = scaled[:, :, 3].astype(np.float32) / 255.0
    rgb = scaled[:, :, :3].astype(np.float32)

    # 1. Volumetric red energy aura
    glow = cv2.GaussianBlur(alpha, (0, 0), sigmaX=24.0, sigmaY=24.0)
    glow = np.clip(glow * (0.85 + 0.35 * ribbon_unfold), 0, 1)
    canvas[pad:pad+sh, pad:pad+sw, 0] += glow * 253.0
    canvas[pad:pad+sh, pad:pad+sw, 1] += glow * 28.0
    canvas[pad:pad+sh, pad:pad+sw, 2] += glow * 22.0
    canvas[pad:pad+sh, pad:pad+sw, 3] = np.maximum(canvas[pad:pad+sh, pad:pad+sw, 3], glow * 175.0)

    # 2. 14-layer 3D extruded glossy red/burgundy body depth
    extrude_dir = 1.0 if sin_s >= 0 else -1.0
    extrude_mag = 0.85 + 0.95 * abs(sin_s)
    n_layers = 14
    for d in range(n_layers, 0, -1):
        ox = pad + int(round(d * extrude_dir * extrude_mag))
        oy = pad + int(round(d * 0.55))
        if ox < 0 or ox + sw > cw or oy < 0 or oy + sh > ch:
            continue
        t_d = 1.0 - (d / float(n_layers))
        r_ext = 95.0 + 95.0 * t_d
        g_ext = 6.0 + 18.0 * t_d
        b_ext = 8.0 + 18.0 * t_d
        a_ext = alpha * 255.0
        reg = canvas[oy:oy+sh, ox:ox+sw]
        m = a_ext > 12
        reg[m, 0] = r_ext
        reg[m, 1] = g_ext
        reg[m, 2] = b_ext
        reg[m, 3] = np.maximum(reg[m, 3], a_ext[m])

    # 3. Front glossy red ribbon surface + studio specular rim
    grad_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
    rim = np.clip((-grad_x - grad_y) * 0.52, 0.0, 1.0)

    face = rgb.copy()
    face[:, :, 0] = np.clip(face[:, :, 0] + rim * 90.0, 0, 255)
    face[:, :, 1] = np.clip(face[:, :, 1] + rim * 165.0, 0, 255)
    face[:, :, 2] = np.clip(face[:, :, 2] + rim * 165.0, 0, 255)

    front = canvas[pad:pad+sh, pad:pad+sw]
    fa = alpha[:, :, None]
    front[:, :, :3] = front[:, :, :3] * (1.0 - fa) + face * fa
    front[:, :, 3] = np.maximum(front[:, :, 3], alpha * 255.0)

    if abs(tilt_deg) > 0.2:
        M = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), tilt_deg, 1.0)
        canvas = cv2.warpAffine(canvas, M, (cw, ch), flags=cv2.INTER_LINEAR)

    if floor_reflection:
        refl = cv2.flip(canvas, 0)
        refl = cv2.GaussianBlur(refl, (25, 25), 8.0)
        refl[:, :, 3] *= 0.26
        ry0 = int(cy + sh // 2 - 20)
        rx0 = int(cx - cw // 2)
        rcx0, rcy0 = max(0, rx0), max(0, ry0)
        rcx1, rcy1 = min(W, rx0 + cw), min(H, ry0 + ch)
        if rcx1 > rcx0 and rcy1 > rcy0:
            sub_r = refl[rcy0 - ry0:rcy1 - ry0, rcx0 - rx0:rcx1 - rx0]
            sub_f = frame_rgb[rcy0:rcy1, rcx0:rcx1].astype(np.float32)
            ra = (sub_r[:, :, 3:4] / 255.0) * opacity
            frame_rgb[rcy0:rcy1, rcx0:rcx1] = np.clip(sub_f * (1.0 - ra) + sub_r[:, :, :3] * ra, 0, 255).astype(np.uint8)

    x0 = int(cx - cw // 2)
    y0 = int(cy - ch // 2)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + cw), min(H, y0 + ch)
    if cx1 > cx0 and cy1 > cy0:
        sub_c = canvas[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        sub_f = frame_rgb[cy0:cy1, cx0:cx1].astype(np.float32)
        a = (sub_c[:, :, 3:4] / 255.0) * opacity
        frame_rgb[cy0:cy1, cx0:cx1] = np.clip(sub_f * (1.0 - a) + sub_c[:, :, :3] * a, 0, 255).astype(np.uint8)

    return frame_rgb

def draw_glowing_customer_character(
    frame_rgb, cx, cy, radius=46,
    mood="calm", legs_amount=0.0, run_phase=0.0, look_dir=1, label_text=None, show_check=False
):
    """
    Renders a true 3D-shaded glowing customer character with smooth spherical normal shading,
    studio specular highlights, expressive eyes, animated 3D capsule legs, and a clean status pill.
    """
    cx, cy = int(cx), int(cy)
    # 1. Volumetric outer glow
    glow = frame_rgb.copy()
    glow_col = (255, 175, 75) if mood in ("impatient", "running") else (255, 225, 145)
    cv2.circle(glow, (cx, cy), int(radius * 1.9), glow_col, -1, lineType=cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (49, 49), 18.0)
    frame_rgb = cv2.addWeighted(frame_rgb, 0.62, glow, 0.38, 0)

    # Floor reflection under customer character
    refl = frame_rgb.copy()
    cv2.ellipse(refl, (cx, cy + int(radius * 1.55)), (int(radius * 0.9), int(radius * 0.28)), 0, 0, 360, glow_col, -1, lineType=cv2.LINE_AA)
    frame_rgb = cv2.addWeighted(frame_rgb, 0.74, refl, 0.26, 0)

    # 2. Construct true 3D shaded sphere on local RGBA patch
    d_size = radius * 2 + 8
    yy, xx = np.mgrid[0:d_size, 0:d_size]
    nx = (xx - (d_size / 2.0)) / float(radius)
    ny = (yy - (d_size / 2.0)) / float(radius)
    r2 = nx * nx + ny * ny
    inside = r2 <= 1.0
    nz = np.sqrt(np.maximum(0.0, 1.0 - r2))

    # Key light from top-left (-0.4, -0.5, 0.76)
    ndotl = np.clip(-0.42 * nx - 0.52 * ny + 0.74 * nz, 0.0, 1.0)
    spec = np.clip(-0.32 * nx - 0.48 * ny + 0.81 * nz, 0.0, 1.0) ** 18.0
    rim = (1.0 - nz) ** 2.2

    sph = np.zeros((d_size, d_size, 4), dtype=np.uint8)
    r_ch = np.clip(225.0 + 30.0 * ndotl + 60.0 * spec + 30.0 * rim, 0, 255)
    g_ch = np.clip(165.0 + 78.0 * ndotl + 65.0 * spec + 20.0 * rim, 0, 255)
    b_ch = np.clip(85.0 + 130.0 * ndotl + 90.0 * spec + 55.0 * rim, 0, 255)
    a_ch = np.clip((1.0 - np.sqrt(r2)) * radius * 1.5, 0.0, 1.0) * 255.0

    sph[:, :, 0] = r_ch.astype(np.uint8)
    sph[:, :, 1] = g_ch.astype(np.uint8)
    sph[:, :, 2] = b_ch.astype(np.uint8)
    sph[:, :, 3] = np.where(inside, a_ch, 0).astype(np.uint8)

    pil_im = Image.fromarray(frame_rgb).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # 3. Animated tiny 3D legs when sprouting / running away!
    if legs_amount > 0.02:
        leg_len = int(radius * 0.98 * legs_amount)
        hip_y = cy + int(radius * 0.68)
        hip_l_x = cx - int(radius * 0.32)
        hip_r_x = cx + int(radius * 0.32)

        swing1 = math.sin(run_phase) * 28.0 * legs_amount
        swing2 = math.sin(run_phase + math.pi) * 28.0 * legs_amount
        lift1 = max(0.0, -math.cos(run_phase)) * 18.0 * legs_amount
        lift2 = max(0.0, -math.cos(run_phase + math.pi)) * 18.0 * legs_amount

        foot1 = (int(hip_l_x + swing1), int(hip_y + leg_len - lift1))
        foot2 = (int(hip_r_x + swing2), int(hip_y + leg_len - lift2))
        knee1 = (int((hip_l_x + foot1[0]) / 2 + 9 * look_dir), int((hip_y + foot1[1]) / 2))
        knee2 = (int((hip_r_x + foot2[0]) / 2 + 9 * look_dir), int((hip_y + foot2[1]) / 2))

        d.line([(hip_l_x, hip_y), knee1, foot1], fill=(255, 240, 200, 255), width=10)
        d.line([(hip_r_x, hip_y), knee2, foot2], fill=(255, 240, 200, 255), width=10)
        f1_x0 = min(foot1[0] - 11, foot1[0] + 18 * look_dir)
        f1_x1 = max(foot1[0] + 11, foot1[0] + 18 * look_dir)
        f2_x0 = min(foot2[0] - 11, foot2[0] + 18 * look_dir)
        f2_x1 = max(foot2[0] + 11, foot2[0] + 18 * look_dir)
        d.rounded_rectangle([f1_x0, foot1[1] - 7, f1_x1, foot1[1] + 9], radius=7, fill=(253, 22, 16, 255), outline=(255, 255, 255, 200), width=1)
        d.rounded_rectangle([f2_x0, foot2[1] - 7, f2_x1, foot2[1] + 9], radius=7, fill=(253, 22, 16, 255), outline=(255, 255, 255, 200), width=1)

    # Paste 3D shaded sphere body
    sph_pil = Image.fromarray(sph, "RGBA")
    ov.paste(sph_pil, (cx - d_size // 2, cy - d_size // 2), sph_pil)

    # 4. Expressive Eyes & Visor Face
    ex_off = int(radius * 0.16) * look_dir
    e_l = (cx - int(radius * 0.28) + ex_off, cy - int(radius * 0.06))
    e_r = (cx + int(radius * 0.28) + ex_off, cy - int(radius * 0.06))
    ew, eh = max(5, int(radius * 0.13)), max(7, int(radius * 0.18))

    if mood == "happy":
        d.arc([e_l[0] - ew, e_l[1] - eh, e_l[0] + ew, e_l[1] + eh], start=190, end=350, fill=(22, 26, 36, 255), width=4)
        d.arc([e_r[0] - ew, e_r[1] - eh, e_r[0] + ew, e_r[1] + eh], start=190, end=350, fill=(22, 26, 36, 255), width=4)
        sm_c = (cx + ex_off, cy + int(radius * 0.18))
        d.arc([sm_c[0] - 14, sm_c[1] - 10, sm_c[0] + 14, sm_c[1] + 12], start=10, end=170, fill=(22, 26, 36, 255), width=4)
    else:
        d.ellipse([e_l[0] - ew, e_l[1] - eh, e_l[0] + ew, e_l[1] + eh], fill=(20, 24, 34, 255))
        d.ellipse([e_r[0] - ew, e_r[1] - eh, e_r[0] + ew, e_r[1] + eh], fill=(20, 24, 34, 255))
        d.ellipse([e_l[0] - 2, e_l[1] - 4, e_l[0] + 2, e_l[1]], fill=(255, 255, 255, 255))
        d.ellipse([e_r[0] - 2, e_r[1] - 4, e_r[0] + 2, e_r[1]], fill=(255, 255, 255, 255))

        if mood == "impatient":
            d.line([(e_l[0] - 10, e_l[1] - 14), (e_l[0] + 8, e_l[1] - 8)], fill=(195, 18, 14, 255), width=4)
            d.line([(e_r[0] - 8, e_r[1] - 8), (e_r[0] + 10, e_r[1] - 14)], fill=(195, 18, 14, 255), width=4)
            d.arc([cx + ex_off - 12, cy + 12, cx + ex_off + 12, cy + 28], start=195, end=345, fill=(22, 26, 36, 255), width=4)
        elif mood == "running":
            d.line([(e_l[0] - 10, e_l[1] - 10), (e_l[0] + 8, e_l[1] - 15)], fill=(195, 18, 14, 255), width=3)
            d.line([(e_r[0] - 8, e_r[1] - 15), (e_r[0] + 10, e_r[1] - 10)], fill=(195, 18, 14, 255), width=3)
            d.ellipse([cx + ex_off - 7, cy + 12, cx + ex_off + 7, cy + 25], fill=(22, 26, 36, 255))

    # 5. Clean status pill above customer (using vector checkmark instead of unicode glyph!)
    if label_text:
        f_lbl = get_font(700, 20)
        bb = d.textbbox((0, 0), label_text, font=f_lbl)
        extra_w = 28 if show_check else 0
        lw = bb[2] - bb[0] + 30 + extra_w
        lh = 38
        lx = cx - lw // 2
        ly = cy - radius - 54
        bg_c = (215, 16, 12, 245) if mood in ("impatient", "running") else (255, 255, 255, 248)
        txt_c = (255, 255, 255, 255) if mood in ("impatient", "running") else (15, 18, 26, 255)
        d.rounded_rectangle([lx, ly, lx + lw, ly + lh], radius=19, fill=bg_c, outline=(253, 14, 10, 255), width=2)
        d.text((lx + 15, ly + 7), label_text, font=f_lbl, fill=txt_c)
        if show_check:
            ck_x = lx + lw - 22
            ck_y = ly + 19
            d.ellipse([ck_x - 10, ck_y - 10, ck_x + 10, ck_y + 10], fill=(253, 14, 10, 255))
            d.line([(ck_x - 5, ck_y), (ck_x - 1, ck_y + 4), (ck_x + 5, ck_y - 4)], fill=(255, 255, 255, 255), width=2)

    return np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)

def draw_white_software_panel(frame_rgb, x0, y0, w, h, station_num, title, lines_with_checks, badge_text, progress=1.0, is_warning=False):
    """
    Renders a crisp, recognizable WHITE 3D software interface panel with vector checkmarks or warning icons.
    """
    if progress <= 0.01:
        return frame_rgb
    ease = ease_out_cubic(progress)
    y_off = int((1.0 - ease) * 35)
    alpha_m = min(1.0, progress * 1.35)

    pil_im = Image.fromarray(frame_rgb).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    py = y0 + y_off
    d.rounded_rectangle([x0 + 8, py + 14, x0 + w + 8, py + h + 14], radius=28, fill=(0, 0, 0, int(165 * alpha_m)))
    d.rounded_rectangle([x0, py, x0 + w, py + h], radius=26, fill=(252, 253, 255, int(248 * alpha_m)), outline=(253, 14, 10, int(225 * alpha_m)), width=3)

    d.rounded_rectangle([x0 + 3, py + 3, x0 + w - 3, py + 62], radius=23, fill=(240, 243, 249, int(250 * alpha_m)))
    d.ellipse([x0 + 24, py + 23, x0 + 40, py + 39], fill=(253, 14, 10, int(255 * alpha_m)))
    d.ellipse([x0 + 48, py + 23, x0 + 64, py + 39], fill=(210, 215, 225, int(255 * alpha_m)))
    d.ellipse([x0 + 72, py + 23, x0 + 88, py + 39], fill=(210, 215, 225, int(255 * alpha_m)))

    f_hdr = get_font(800, 23)
    hdr_prefix = "WEBSITE PORTAL" if station_num == 0 else f"STATION 0{station_num}"
    d.text((x0 + 106, py + 18), f"{hdr_prefix}  •  {title}", font=f_hdr, fill=(18, 22, 32, int(255 * alpha_m)))

    # Right header completed action pill with vector checkmark
    f_bdg = get_font(800, 19)
    bb = d.textbbox((0, 0), badge_text, font=f_bdg)
    bw = bb[2] - bb[0] + (50 if not is_warning else 28)
    bx = x0 + w - bw - 20
    d.rounded_rectangle([bx, py + 15, bx + bw, py + 49], radius=17, fill=(253, 14, 10, int(255 * alpha_m)))
    d.text((bx + 14, py + 20), badge_text, font=f_bdg, fill=(255, 255, 255, int(255 * alpha_m)))
    if not is_warning:
        ckx = bx + bw - 20
        cky = py + 32
        d.line([(ckx - 6, cky), (ckx - 1, cky + 5), (ckx + 6, cky - 5)], fill=(255, 255, 255, int(255 * alpha_m)), width=3)

    f_row_t = get_font(700, 25)
    f_row_s = get_font(500, 21)
    row_y = py + 82
    for idx, (r_title, r_sub, is_highlight) in enumerate(lines_with_checks):
        row_p = smoothstep(idx * 0.18, idx * 0.18 + 0.45, progress)
        if row_p <= 0.01:
            continue
        ra = int(255 * row_p * alpha_m)
        bg_row = (255, 240, 240, int(235 * row_p * alpha_m)) if is_highlight else (244, 246, 252, int(225 * row_p * alpha_m))
        border_row = (253, 14, 10, ra) if is_highlight else (220, 226, 236, ra)
        d.rounded_rectangle([x0 + 24, row_y, x0 + w - 24, row_y + 78], radius=16, fill=bg_row, outline=border_row, width=2)

        cx_orb, cy_orb = x0 + 58, row_y + 39
        d.ellipse([cx_orb - 16, cy_orb - 16, cx_orb + 16, cy_orb + 16], fill=(253, 14, 10, ra))
        if is_warning:
            # Exclamation mark for unanswered inquiry
            d.line([(cx_orb, cy_orb - 8), (cx_orb, cy_orb + 2)], fill=(255, 255, 255, ra), width=3)
            d.ellipse([cx_orb - 2, cy_orb + 6, cx_orb + 2, cy_orb + 10], fill=(255, 255, 255, ra))
        else:
            # Vector checkmark for completed automation step
            d.line([(cx_orb - 7, cy_orb), (cx_orb - 2, cy_orb + 6), (cx_orb + 8, cy_orb - 6)], fill=(255, 255, 255, ra), width=3)

        d.text((x0 + 90, row_y + 12), r_title, font=f_row_t, fill=(14, 18, 28, ra))
        d.text((x0 + 90, row_y + 44), r_sub, font=f_row_s, fill=(85, 95, 115, ra))
        row_y += 90

    return np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)

def draw_hero_headline(frame_rgb, text_lines, y_top, progress=1.0, sub_pill=None, font_size=62):
    if progress <= 0.01:
        return frame_rgb
    ease = ease_out_cubic(progress)
    y_off = int((1.0 - ease) * 28)
    alpha_m = min(1.0, progress * 1.4)

    pil_im = Image.fromarray(frame_rgb).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    font = get_font(900, font_size)
    line_h = int(font_size * 1.2)
    total_h = len(text_lines) * line_h + (56 if sub_pill else 0)

    box_top = max(0, y_top + y_off - 30)
    box_bot = min(H, y_top + y_off + total_h + 36)
    d.rounded_rectangle(
        [48, box_top, W - 48, box_bot],
        radius=28,
        fill=(8, 10, 15, int(205 * alpha_m)),
        outline=(253, 14, 10, int(150 * alpha_m)),
        width=2
    )

    cur_y = y_top + y_off
    for line in text_lines:
        bb = d.textbbox((0, 0), line, font=font)
        tw = bb[2] - bb[0]
        tx = (W - tw) // 2
        d.text((tx + 2, cur_y + 4), line, font=font, fill=(0, 0, 0, int(220 * alpha_m)))
        d.text((tx, cur_y), line, font=font, fill=(255, 255, 255, int(255 * alpha_m)))
        cur_y += line_h

    if sub_pill:
        f_sub = get_font(700, 24)
        sb = d.textbbox((0, 0), sub_pill, font=f_sub)
        sw = sb[2] - sb[0] + 36
        sx = (W - sw) // 2
        sy = cur_y + 8
        d.rounded_rectangle([sx, sy, sx + sw, sy + 40], radius=20, fill=(253, 14, 10, int(240 * alpha_m)))
        d.text((sx + 18, sy + 7), sub_pill, font=f_sub, fill=(255, 255, 255, int(255 * alpha_m)))

    return np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)

class EscapeCommercialRenderer:
    def __init__(self):
        print("Loading 3D story environments & official Brandrums B character...")
        raw_s1 = load_and_fit_9_16(os.path.join(ASSETS_DIR, "story_s1_escape.png"))
        # Smoothly feather-blend the bottom-right floor area so there are zero hard edges or stray symbols
        yy, xx = np.mgrid[0:H, 0:W]
        dist_mask = np.exp(-(((xx - 860.0) / 210.0) ** 2 + ((yy - 1260.0) / 260.0) ** 2))
        blurred_s1 = cv2.GaussianBlur(raw_s1, (99, 99), 40.0)
        dark_floor = (blurred_s1 * 0.25).astype(np.float32)
        m3 = dist_mask[:, :, None]
        self.bg_s1 = np.clip(raw_s1.astype(np.float32) * (1.0 - m3) + dark_floor * m3, 0, 255).astype(np.uint8)

        self.bg_s2 = load_and_fit_9_16(os.path.join(ASSETS_DIR, "story_s2_b_hero_stage.png"))
        self.bg_s3 = load_and_fit_9_16(os.path.join(ASSETS_DIR, "story_s3_conveyor_stations.png"))
        self.bg_s4 = load_and_fit_9_16(os.path.join(ASSETS_DIR, "story_s4_happy_ecosystem.png"))
        self.bg_s5 = load_and_fit_9_16(os.path.join(ASSETS_DIR, "story_s5_final_lockup.png"))

        self.full_logo_rgba, self.b_hero_rgba = prepare_official_logo_and_b_hero(target_w=760)

    def render_s1_escape(self, t):
        """
        0 - 5s: Tiny 3D business website inside miniature office.
        Glowing customer arrives and submits an inquiry. Nobody responds.
        Ticking clock pulses. Customer looks impatient, sprouts tiny animated legs,
        jumps out of the website, and starts running away while miniature business owner chases it!
        """
        prog = t / 5.0
        zoom = 1.02 + 0.08 * prog
        pan_x = -15.0 + 40.0 * smoothstep(2.4, 5.0, t)
        frame = camera_transform(self.bg_s1, zoom=zoom, pan_x=pan_x, pan_y=0.0)

        # Concentric red tick-pulse rings radiating from the 3D alarm clock above the website kiosk (510, 520)
        if t >= 0.75:
            pulse_cycle = ((t - 0.75) * 2.5) % 1.0
            clk_x, clk_y = int(510 - pan_x * 0.9), 520
            rad = int(48 + pulse_cycle * 68)
            ring = frame.copy()
            cv2.circle(ring, (clk_x, clk_y), rad, (253, 25, 18), 4, lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 1.0 - 0.55 * (1.0 - pulse_cycle), ring, 0.55 * (1.0 - pulse_cycle), 0)

        # Customer character animation path
        if t < 1.2:
            p_arr = ease_out_cubic(t / 1.2)
            cust_x = 340 + 165 * p_arr
            cust_y = 790 - math.sin(p_arr * math.pi) * 40
            mood = "calm"
            legs = 0.0
            lbl = "INQUIRY SUBMITTED!"
        elif t < 2.45:
            cust_x = 505 + math.sin(t * 28.0) * 6.0
            cust_y = 790
            mood = "impatient"
            legs = smoothstep(2.05, 2.45, t)
            lbl = "NO RESPONSE... WAITING!"
        elif t < 2.95:
            jp = (t - 2.45) / 0.50
            cust_x = 505 + 135 * jp
            cust_y = 790 + 390 * (jp ** 1.4) - math.sin(jp * math.pi) * 120
            mood = "running"
            legs = 1.0
            lbl = "I'M LEAVING!"
        else:
            rp = (t - 2.95) / 2.05
            cust_x = 640 + 275 * rp
            cust_y = 1180 + 65 * rp - abs(math.sin(t * 18.0)) * 22
            mood = "running"
            legs = 1.0
            lbl = "ESCAPING TO COMPETITOR!"

        frame = draw_glowing_customer_character(
            frame, cust_x, cust_y, radius=48,
            mood=mood, legs_amount=legs, run_phase=t * 26.0, look_dir=1, label_text=lbl, show_check=False
        )

        # White Website Inquiry Panel at top showing the unanswered inquiry state
        frame = draw_white_software_panel(
            frame, x0=70, y0=110, w=940, h=280,
            station_num=0, title="BUSINESS WEBSITE INQUIRY",
            lines_with_checks=[
                ("New Customer Inquiry Submitted", "Waiting for business owner to respond...", True),
                ("Response Timer: DELAYED", "Customer is losing patience & running away!", False),
            ],
            badge_text="NO REPLY!",
            progress=smoothstep(0.15, 0.75, t),
            is_warning=True
        )
        return frame

    def render_s2_b_intercept(self, t):
        """
        5 - 9s: Red light streaks across the room. Official Brandrums B character drops in,
        slides across the polished floor, and intercepts the escaping customer.
        Its red ribbon curves unfold dynamically as it activates the automation system.
        Text: “DON’T LET LEADS ESCAPE.”
        """
        local_t = t - 5.0  # 0.0 to 4.0
        prog = local_t / 4.0
        zoom = 1.08 - 0.06 * ease_out_cubic(prog)
        frame = camera_transform(self.bg_s2, zoom=zoom, pan_x=0.0, pan_y=0.0)

        if local_t < 1.1:
            sp = min(1.0, local_t / 0.55)
            streak = np.zeros_like(frame)
            p_start = (120, 180)
            p_end = (int(120 + 440 * sp), int(180 + 820 * sp))
            cv2.line(streak, p_start, p_end, (253, 14, 10), 42, lineType=cv2.LINE_AA)
            cv2.line(streak, p_start, p_end, (255, 255, 255), 14, lineType=cv2.LINE_AA)
            streak = cv2.GaussianBlur(streak, (35, 35), 14.0)
            frame = cv2.addWeighted(frame, 1.0, streak, max(0.0, 1.0 - local_t * 0.85), 0)

        if local_t < 0.75:
            p_run = local_t / 0.75
            cx_cust = int(300 + 460 * p_run)
            cy_cust = 1130 - abs(math.sin(local_t * 20.0)) * 18
            c_mood = "running"
            c_legs = 1.0 - 0.5 * p_run
            c_dir = 1
            c_lbl = "WAIT—WHO'S THAT?!"
            c_chk = False
        else:
            settle = smoothstep(0.75, 1.5, local_t)
            cx_cust = int(760 - 40 * settle)
            cy_cust = 1130 + int(math.sin(local_t * 4.0) * 8)
            c_mood = "happy"
            c_legs = max(0.0, 0.5 * (1.0 - settle))
            c_dir = -1
            c_lbl = "INSTANTLY CAPTURED"
            c_chk = True

        if local_t < 0.45:
            dp = ease_out_cubic(local_t / 0.45)
            bx = int(220 + 180 * dp)
            by = int(180 + 840 * dp)
            sx, sy = 0.88, 1.18
            tilt = -16.0
            unfold = 0.1
        elif local_t < 1.10:
            slp = ease_out_cubic((local_t - 0.45) / 0.65)
            bx = int(400 + 115 * slp)
            by = 1020
            sx = 1.18 - 0.18 * slp
            sy = 0.84 + 0.16 * slp
            tilt = 14.0 * (1.0 - slp)
            unfold = 0.25 * slp
        else:
            bx = 515
            by = 1010 + int(math.sin(local_t * 3.2) * 10)
            sx, sy = 1.04, 1.04
            tilt = math.sin(local_t * 2.5) * 4.0
            unfold = math.sin(min(1.0, (local_t - 1.1) / 1.8) * math.pi) * 0.85

        if local_t > 1.0:
            ring_t = (local_t - 1.0) / 2.6
            for r_i in range(2):
                rp = (ring_t + r_i * 0.35) % 1.0
                rad = int(120 + rp * 520)
                ring_img = np.zeros_like(frame)
                cv2.ellipse(ring_img, (bx, by + 110), (rad, int(rad * 0.36)), 0, 0, 360, (253, 18, 14), 10, lineType=cv2.LINE_AA)
                cv2.ellipse(ring_img, (bx, by + 110), (max(1, rad - 6), max(1, int(rad * 0.36) - 3)), 0, 0, 360, (255, 255, 255), 3, lineType=cv2.LINE_AA)
                frame = cv2.addWeighted(frame, 1.0, ring_img, (1.0 - rp) * 0.65, 0)

        frame = render_b_hero_character(
            frame, self.b_hero_rgba, bx, by,
            scale_x=sx, scale_y=sy, tilt_deg=tilt, spin_deg=0.0,
            ribbon_unfold=unfold, floor_reflection=True, opacity=1.0
        )

        frame = draw_glowing_customer_character(
            frame, cx_cust, cy_cust, radius=48,
            mood=c_mood, legs_amount=c_legs, run_phase=local_t * 20.0, look_dir=c_dir, label_text=c_lbl, show_check=c_chk
        )

        frame = draw_hero_headline(
            frame,
            ["DON’T LET LEADS ESCAPE."],
            y_top=175,
            progress=smoothstep(0.35, 1.05, local_t),
            sub_pill="BRANDRUMS AI AUTOMATION INTERCEPTS EVERY LEAD",
            font_size=56
        )
        return frame

    def render_s3_stations(self, t):
        """
        9 - 19s: Continuous 3D action sequence through 4 physical interconnected software stations.
        Synchronized text:
        - "CAPTURE"
        - "RESPOND"
        - "BOOK"
        - "FOLLOW UP"
        """
        local_t = t - 9.0  # 0.0 to 10.0
        prog = local_t / 10.0

        zoom = 1.08 + 0.06 * math.sin(prog * math.pi)
        pan_x = math.sin(prog * math.pi * 2.0) * 28.0
        pan_y = -65.0 + 130.0 * prog
        frame = camera_transform(self.bg_s3, zoom=zoom, pan_x=pan_x, pan_y=pan_y)

        waypoints = [
            (720, 410),
            (345, 560),
            (690, 820),
            (330, 1015),
            (660, 1290),
        ]
        seg_f = prog * (len(waypoints) - 1)
        seg_i = min(len(waypoints) - 2, int(seg_f))
        seg_t = ease_out_cubic(seg_f - seg_i)
        p0, p1 = waypoints[seg_i], waypoints[seg_i + 1]
        cx_path = int(p0[0] + (p1[0] - p0[0]) * seg_t)
        cy_path = int(p0[1] + (p1[1] - p0[1]) * seg_t)

        glow_trail = frame.copy()
        cv2.circle(glow_trail, (cx_path, cy_path), 75, (253, 18, 14), -1, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(frame, 0.72, glow_trail, 0.28, 0)

        frame = render_b_hero_character(
            frame, self.b_hero_rgba,
            cx_path - 75, cy_path - 25 + int(math.sin(local_t * 5.0) * 8),
            scale_x=0.62, scale_y=0.62,
            tilt_deg=math.sin(local_t * 4.0) * 5.0,
            spin_deg=0.0, ribbon_unfold=0.12, floor_reflection=False, opacity=1.0
        )
        frame = draw_glowing_customer_character(
            frame, cx_path + 55, cy_path + 10 + int(math.cos(local_t * 5.0) * 6),
            radius=38, mood="happy", legs_amount=0.0, run_phase=0.0, look_dir=-1, label_text="GUIDED BY AI", show_check=True
        )

        stage = min(3, int(local_t // 2.5))
        stage_p = (local_t - stage * 2.5) / 2.5

        if stage == 0:
            hl_word = "CAPTURE"
            sub_word = "STATION 01 • WEBSITE INQUIRY PORTAL"
            p_title = "WEBSITE INQUIRY & AI CHAT"
            rows = [
                ("Visitor Inquiry Received", "“Hi! Looking for availability & pricing this week.”", False),
                ("AI Assistant Responds Instantly", "“Hello! I've got your details—let's book your slot!”", True),
                ("Instant Automated Reply", "Customer receives an immediate helpful response", False),
            ]
            bdg = "CAPTURED"
        elif stage == 1:
            hl_word = "RESPOND"
            sub_word = "STATION 02 • INTELLIGENT CRM STATION"
            p_title = "CRM CONTACT RECORD CREATED"
            rows = [
                ("New Contact Profile Created", "Name, Email, Phone & Service Interest Synced", True),
                ("AI Lead Qualification Complete", "Tagged as Qualified Prospect Automatically", False),
                ("Zero Manual Data Entry", "Moved to Active Sales Pipeline via Red Conveyor", False),
            ]
            bdg = "CRM SYNCED"
        elif stage == 2:
            hl_word = "BOOK"
            sub_word = "STATION 03 • SMART BOOKING CALENDAR"
            p_title = "CALENDAR & CONFIRMATION EMAIL"
            rows = [
                ("Available Appointment Selected", "Preferred Time Slot Locked on Calendar", True),
                ("Confirmation Email Sent", "Instant Calendar Invite & Meeting Link Delivered", True),
                ("Zero Back-and-Forth Calls", "Appointment confirmed automatically", False),
            ]
            bdg = "BOOKED"
        else:
            hl_word = "FOLLOW UP"
            sub_word = "STATION 04 • AUTOMATED FOLLOW-UP ENGINE"
            p_title = "AUTOMATIC FOLLOW-UP SCHEDULED"
            rows = [
                ("Upcoming Reminder Scheduled", "Automated SMS & Email Reminder Queued", True),
                ("Ongoing Nurture Sequence Active", "No lead is ever forgotten or dropped", True),
                ("Complete End-to-End Workflow", "Capture, Respond, Book & Follow Up Complete", False),
            ]
            bdg = "AUTOMATED"

        frame = draw_hero_headline(
            frame, [hl_word], y_top=125,
            progress=smoothstep(0.0, 0.22, stage_p),
            sub_pill=sub_word, font_size=68
        )

        frame = draw_white_software_panel(
            frame, x0=64, y0=1375, w=952, h=405,
            station_num=stage + 1, title=p_title,
            lines_with_checks=rows, badge_text=bdg,
            progress=smoothstep(0.0, 0.28, stage_p),
            is_warning=False
        )
        return frame

    def render_s4_ecosystem(self, t):
        """
        19 - 25s: Complete miniature business ecosystem running smoothly.
        Several new customer characters arrive and enter the automated process.
        Business owner smiles and focuses on running the business.
        Text: “LET YOUR BUSINESS WORK SMARTER.”
        """
        local_t = t - 19.0  # 0.0 to 6.0
        prog = local_t / 6.0

        zoom = 1.14 - 0.14 * ease_out_cubic(prog)
        frame = camera_transform(self.bg_s4, zoom=zoom, pan_x=0.0, pan_y=0.0)

        conveyor_pts = [(680, 440), (410, 520), (470, 680), (260, 890), (415, 1020), (210, 1140)]
        lbls = ["CAPTURED", "BOOKED", "FOLLOWED UP"]
        for c_idx in range(3):
            cp = ((local_t * 0.22) + c_idx * 0.31) % 1.0
            sf = cp * (len(conveyor_pts) - 1)
            si = min(len(conveyor_pts) - 2, int(sf))
            st = sf - si
            cx = int(conveyor_pts[si][0] + (conveyor_pts[si+1][0] - conveyor_pts[si][0]) * st)
            cy = int(conveyor_pts[si][1] + (conveyor_pts[si+1][1] - conveyor_pts[si][1]) * st)
            frame = draw_glowing_customer_character(
                frame, cx, cy, radius=28,
                mood="happy", legs_amount=0.0, run_phase=0.0, look_dir=1, label_text=lbls[c_idx], show_check=True
            )

        frame = render_b_hero_character(
            frame, self.b_hero_rgba,
            cx=545, cy=690 + int(math.sin(local_t * 3.0) * 10),
            scale_x=0.68, scale_y=0.68,
            tilt_deg=math.sin(local_t * 2.2) * 4.0,
            spin_deg=0.0, ribbon_unfold=0.08, floor_reflection=True, opacity=1.0
        )

        frame = draw_hero_headline(
            frame,
            ["LET YOUR BUSINESS", "WORK SMARTER."],
            y_top=135,
            progress=smoothstep(0.15, 0.85, local_t),
            sub_pill="AUTOMATED LEADS • CRM • BOOKING • FOLLOW-UP",
            font_size=56
        )
        return frame

    def render_s5_finale(self, t):
        """
        25 - 30s: The Brandrums character performs a final confident spin.
        Its red ribbons fold elegantly back into the exact original B logo.
        Display official Brandrums logo and exact text:
        “STOP LOSING LEADS.”
        “AUTOMATE YOUR BUSINESS.”
        “BRANDRUMS.COM”
        """
        local_t = t - 25.0  # 0.0 to 5.0

        zoom = 1.08 - 0.08 * ease_out_cubic(min(1.0, local_t / 2.2))
        frame = camera_transform(self.bg_s5, zoom=zoom, pan_x=0.0, pan_y=0.0)

        for idx in range(4):
            pt = ((local_t * 0.65) + idx * 0.25) % 1.0
            yy = int(880 + pt * 920)
            glow = frame.copy()
            cv2.circle(glow, (490, yy), 38, (253, 18, 14), -1, lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(frame, 0.62, glow, 0.38, 0)

        if local_t < 1.45:
            sp = ease_out_cubic(local_t / 1.45)
            spin_deg = (1.0 - sp) * 360.0
            unfold = math.sin(sp * math.pi) * 0.65 * (1.0 - sp)
            bx = int(540 - (540 - 272) * sp)
            by = 250
            frame = render_b_hero_character(
                frame, self.b_hero_rgba,
                cx=bx, cy=by,
                scale_x=0.72, scale_y=0.72,
                tilt_deg=0.0, spin_deg=spin_deg,
                ribbon_unfold=unfold, floor_reflection=False, opacity=1.0
            )
            if local_t > 1.05:
                fade_full = (local_t - 1.05) / 0.40
                from render_brandrums_film import render_3d_logo_on_frame
                frame = render_3d_logo_on_frame(
                    frame, self.full_logo_rgba,
                    center_x=W // 2, center_y=250,
                    scale=0.95, light_sweep=0.35, opacity=fade_full
                )
        else:
            from render_brandrums_film import render_3d_logo_on_frame
            sweep = 0.2 + 0.65 * (((local_t - 1.45) / 3.55) % 1.0)
            frame = render_3d_logo_on_frame(
                frame, self.full_logo_rgba,
                center_x=W // 2, center_y=250,
                scale=0.95, light_sweep=sweep, opacity=1.0
            )

        pil_f = Image.fromarray(frame).convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)

        t1_p = smoothstep(0.45, 1.15, local_t)
        t2_p = smoothstep(0.85, 1.55, local_t)
        t3_p = smoothstep(1.25, 1.95, local_t)

        if t1_p > 0.01:
            d.rounded_rectangle(
                [56, 405, W - 56, 625],
                radius=28,
                fill=(8, 10, 16, int(215 * t1_p)),
                outline=(253, 14, 10, int(165 * t1_p)),
                width=2
            )

        f1 = get_font(900, 52)
        f2 = get_font(800, 44)
        f_url = get_font(900, 56)

        if t1_p > 0.01:
            txt1 = "STOP LOSING LEADS."
            b1 = d.textbbox((0, 0), txt1, font=f1)
            w1 = b1[2] - b1[0]
            y1 = 435 + int((1.0 - ease_out_cubic(t1_p)) * 18)
            d.text(((W - w1) // 2 + 2, y1 + 3), txt1, font=f1, fill=(0, 0, 0, int(220 * t1_p)))
            d.text(((W - w1) // 2, y1), txt1, font=f1, fill=(255, 255, 255, int(255 * t1_p)))

        if t2_p > 0.01:
            txt2 = "AUTOMATE YOUR BUSINESS."
            b2 = d.textbbox((0, 0), txt2, font=f2)
            w2 = b2[2] - b2[0]
            y2 = 525 + int((1.0 - ease_out_cubic(t2_p)) * 18)
            d.text(((W - w2) // 2 + 2, y2 + 3), txt2, font=f2, fill=(0, 0, 0, int(220 * t2_p)))
            d.text(((W - w2) // 2, y2), txt2, font=f2, fill=(253, 65, 60, int(255 * t2_p)))

        if t3_p > 0.01:
            txt3 = "BRANDRUMS.COM"
            b3 = d.textbbox((0, 0), txt3, font=f_url)
            w3 = b3[2] - b3[0]
            pill_w = w3 + 136
            pill_h = 116
            px0 = (W - pill_w) // 2
            py0 = 1615 + int((1.0 - ease_out_cubic(t3_p)) * 24)

            d.rounded_rectangle(
                [px0 - 8, py0 - 8, px0 + pill_w + 8, py0 + pill_h + 8],
                radius=66,
                fill=(253, 14, 10, int(75 * t3_p)),
                outline=(253, 45, 40, int(160 * t3_p)),
                width=2
            )
            d.rounded_rectangle(
                [px0, py0, px0 + pill_w, py0 + pill_h],
                radius=58,
                fill=(220, 10, 8, int(250 * t3_p)),
                outline=(255, 255, 255, int(230 * t3_p)),
                width=3
            )
            tx3 = (W - w3) // 2
            ty3 = py0 + 25
            d.text((tx3 + 2, ty3 + 3), txt3, font=f_url, fill=(60, 0, 0, int(220 * t3_p)))
            d.text((tx3, ty3), txt3, font=f_url, fill=(255, 255, 255, int(255 * t3_p)))

        return np.array(Image.alpha_composite(pil_f, ov).convert("RGB"), dtype=np.uint8)

    def render_frame(self, frame_idx):
        t = frame_idx / float(FPS)
        trans = 0.25
        if t < 5.0:
            return self.render_s1_escape(t)
        elif t < 9.0:
            if t < 5.0 + trans:
                a = (t - 5.0) / trans
                return cv2.addWeighted(self.render_s2_b_intercept(t), a, self.render_s1_escape(4.99), 1.0 - a, 0)
            return self.render_s2_b_intercept(t)
        elif t < 19.0:
            if t < 9.0 + trans:
                a = (t - 9.0) / trans
                return cv2.addWeighted(self.render_s3_stations(t), a, self.render_s2_b_intercept(8.99), 1.0 - a, 0)
            return self.render_s3_stations(t)
        elif t < 25.0:
            if t < 19.0 + trans:
                a = (t - 19.0) / trans
                return cv2.addWeighted(self.render_s4_ecosystem(t), a, self.render_s3_stations(18.99), 1.0 - a, 0)
            return self.render_s4_ecosystem(t)
        else:
            if t < 25.0 + trans:
                a = (t - 25.0) / trans
                return cv2.addWeighted(self.render_s5_finale(t), a, self.render_s4_ecosystem(24.99), 1.0 - a, 0)
            return self.render_s5_finale(t)

def main():
    renderer = EscapeCommercialRenderer()
    if "--preview" in sys.argv:
        os.makedirs("/tmp/escape_previews", exist_ok=True)
        for tt in [3.4, 7.2, 10.2, 15.2, 22.0, 28.0]:
            rgb = renderer.render_frame(int(tt * FPS))
            out_p = f"/tmp/escape_previews/preview_{tt:.1f}s.jpg"
            Image.fromarray(rgb).save(out_p, quality=92)
            print("Saved preview:", out_p)
        return

    out_video = "/home/user/video/brandrums_lead_escape_commercial.mp4"
    audio_path = os.path.join(ASSETS_DIR, "escape_soundtrack_master.wav")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-i", audio_path,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",
        out_video
    ]
    print(f"Rendering {TOTAL_FRAMES} frames ({DURATION}s @ {FPS}fps, {W}x{H})...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for idx in range(TOTAL_FRAMES):
        frame = renderer.render_frame(idx)
        proc.stdin.write(frame.tobytes())
        if (idx + 1) % 150 == 0 or idx == TOTAL_FRAMES - 1:
            print(f"  Rendered {idx + 1}/{TOTAL_FRAMES} frames ({(idx+1)/FPS:.1f}s)")
    proc.stdin.close()
    stderr_out = proc.stderr.read().decode("utf-8", errors="ignore")
    proc.wait()
    if proc.returncode != 0:
        print("FFmpeg error:", stderr_out[-1000:])
        sys.exit(1)
    print(f"Successfully rendered {out_video}!")

if __name__ == "__main__":
    main()
