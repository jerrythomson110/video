import os
import sys
import math
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import imageio_ffmpeg

W, H = 1080, 1920
FPS = 30
DURATION = 30.0
TOTAL_FRAMES = int(FPS * DURATION)

ASSETS_DIR = "/home/user/video/assets"
FONT_DIR = os.path.join(ASSETS_DIR, "fonts")

# Brandrums Official Palette (extracted from brandrums.com & official logo)
BRAND_RED = (253, 14, 10)       # #FD0E0A
BRAND_CRIMSON = (195, 8, 6)     # #C30806
BRAND_DEEP_RED = (110, 0, 0)    # #6E0000
PURE_WHITE = (255, 255, 255)
STUDIO_SILVER = (225, 230, 240)
OBSIDIAN = (10, 12, 16)

def get_font(weight=700, size=48):
    path = os.path.join(FONT_DIR, f"inter-latin-{weight}-normal.woff")
    if not os.path.exists(path):
        path = os.path.join(FONT_DIR, "inter-latin-700-normal.woff")
    return ImageFont.truetype(path, size)

def ease_out_cubic(x):
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3

def ease_in_out_cubic(x):
    x = max(0.0, min(1.0, x))
    return 4 * x * x * x if x < 0.5 else 1 - ((-2 * x + 2) ** 3) / 2

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

def prepare_official_3d_logo(target_w=760):
    """
    Loads the official Brandrums logo extracted from brandrums.com
    and constructs a high-resolution 3D dimensional studio asset
    preserving the exact original geometry, ribbon 'B', and proportions.
    """
    raw = Image.open(os.path.join(ASSETS_DIR, "brandrums_official_logo.png")).convert("RGBA")
    rw, rh = raw.size
    scale = target_w / rw
    th = int(round(rh * scale))

    # Supersample 2x then refine alpha via SDF for razor-sharp contour edges
    up = raw.resize((target_w * 2, th * 2), Image.Resampling.BICUBIC)
    arr = np.array(up, dtype=np.float32)
    alpha = arr[:, :, 3]
    alpha_blur = cv2.GaussianBlur(alpha, (9, 9), 2.2)
    # Sharpen alpha contour around 128 threshold
    alpha_sharp = np.clip((alpha_blur - 110.0) * (255.0 / 35.0), 0, 255)
    arr[:, :, 3] = alpha_sharp

    # Resize down to target_w x th with Lanczos
    clean_rgba = Image.fromarray(arr.astype(np.uint8), "RGBA").resize((target_w, th), Image.Resampling.LANCZOS)
    c_arr = np.array(clean_rgba, dtype=np.float32)

    # Separate the 3D 'B' ribbon icon (x < 0.29 * target_w) and 'randrums' wordmark (x >= 0.29 * target_w)
    # In the official logo, the 'B' has bright red & deep burgundy 3D ribbon folds,
    # while 'randrums' has a dark red gradient. Under dark studio lighting, we illuminate 'randrums'
    # with a studio key light so it gleams in crisp white/silver with a Brandrums crimson core/rim,
    # AND keep a second pure-crimson variant as well.
    split_x = int(target_w * 0.295)
    yy, xx = np.mgrid[0:th, 0:target_w]
    norm_y = yy / float(th)
    norm_x = xx / float(target_w)

    # Enhance the 'B' icon studio lighting while preserving its exact ribbon folds
    b_rgb = c_arr[:, :split_x, :3]
    # Boost highlights on the red ribbon folds so they look like physical anodized red metal
    b_luma = b_rgb[:, :, 0] / 255.0
    b_r = np.clip(b_rgb[:, :, 0] * 1.12 + 18.0, 95, 255)
    b_g = np.clip(b_rgb[:, :, 1] * 1.1 + (b_luma ** 3) * 45.0, 0, 90)
    b_b = np.clip(b_rgb[:, :, 2] * 1.1 + (b_luma ** 3) * 40.0, 0, 85)
    c_arr[:, :split_x, 0] = b_r
    c_arr[:, :split_x, 1] = b_g
    c_arr[:, :split_x, 2] = b_b

    # For 'randrums' wordmark: preserve the official red-crimson gradient AND add a crisp top studio specular sheen
    # so every letter is 100% legible and dimensional against dark studio backgrounds
    w_alpha = c_arr[:, split_x:, 3] / 255.0
    # Compute top bevel normal from vertical alpha gradient
    grad_y = cv2.Sobel(w_alpha, cv2.CV_32F, 0, 1, ksize=3)
    grad_x = cv2.Sobel(w_alpha, cv2.CV_32F, 1, 0, ksize=3)
    bevel_light = np.clip(-grad_y * 0.7 - grad_x * 0.3, 0.0, 1.0)

    # Studio-illuminated Brandrums crimson-to-luminous-coral/white vertical sheen
    wy = norm_y[:, split_x:]
    wx = (xx[:, split_x:] - split_x) / float(target_w - split_x)
    # Bright studio red with crisp silver-white specular top edge
    spec = (1.0 - wy) ** 2.2 * 0.38 + bevel_light * 0.45
    r_word = np.clip(248.0 - wx * 28.0 + spec * 60.0, 0, 255)
    g_word = np.clip(28.0 + spec * 210.0, 0, 245)
    b_word = np.clip(26.0 + spec * 210.0, 0, 245)

    c_arr[:, split_x:, 0] = r_word
    c_arr[:, split_x:, 1] = g_word
    c_arr[:, split_x:, 2] = b_word

    # Also compute full-logo bevel highlight for 3D extrusion
    full_a = c_arr[:, :, 3]
    return c_arr.astype(np.uint8)

def render_3d_logo_on_frame(frame_rgb, logo_rgba, center_x, center_y, scale=1.0, light_sweep=0.5, opacity=1.0, tilt_deg=0.0):
    """
    Renders the official Brandrums logo as a dimensional 3D object with:
    - Volumetric red/white studio backglow
    - 12-layer 3D metallic extrusion depth
    - Specular light sweep across the surface
    - Crisp contact shadow
    """
    if opacity <= 0.01:
        return frame_rgb
    lh, lw = logo_rgba.shape[:2]
    sw = max(16, int(lw * scale))
    sh = max(8, int(lh * scale))
    scaled = cv2.resize(logo_rgba, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

    pad = 80
    canvas_w = sw + pad * 2
    canvas_h = sh + pad * 2
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)

    alpha = scaled[:, :, 3].astype(np.float32) / 255.0
    rgb = scaled[:, :, :3].astype(np.float32)

    # 1. Volumetric red & white studio glow behind logo
    glow_mask = cv2.GaussianBlur(alpha, (0, 0), sigmaX=22.0, sigmaY=22.0)
    glow_mask = np.clip(glow_mask * 0.75, 0, 1)
    gx0, gy0 = pad, pad
    canvas[gy0:gy0+sh, gx0:gx0+sw, 0] += glow_mask * 253.0
    canvas[gy0:gy0+sh, gx0:gx0+sw, 1] += glow_mask * 24.0
    canvas[gy0:gy0+sh, gx0:gx0+sw, 2] += glow_mask * 20.0
    canvas[gy0:gy0+sh, gx0:gx0+sw, 3] = np.maximum(canvas[gy0:gy0+sh, gx0:gx0+sw, 3], glow_mask * 165.0)

    # 2. Deep studio drop shadow
    shad_mask = cv2.GaussianBlur(alpha, (0, 0), sigmaX=14.0, sigmaY=14.0)
    sx, sy = pad + 8, pad + 18
    canvas[sy:sy+sh, sx:sx+sw, 3] = np.maximum(canvas[sy:sy+sh, sx:sx+sw, 3], shad_mask * 210.0)

    # 3. 12-layer 3D physical extrusion (brushed dark crimson/titanium bevel)
    n_layers = 12
    for d in range(n_layers, 0, -1):
        ox = pad + int(round(d * 0.65))
        oy = pad + int(round(d * 0.85))
        t_d = 1.0 - (d / float(n_layers))
        # Metallic crimson-titanium side wall shading
        r_ext = 75.0 + 70.0 * t_d
        g_ext = 12.0 + 22.0 * t_d
        b_ext = 16.0 + 26.0 * t_d
        a_ext = alpha * 255.0
        region = canvas[oy:oy+sh, ox:ox+sw]
        mask = a_ext > 10
        region[mask, 0] = r_ext
        region[mask, 1] = g_ext
        region[mask, 2] = b_ext
        region[mask, 3] = np.maximum(region[mask, 3], a_ext[mask])

    # 4. Front 3D face + animated diagonal studio light sweep
    yy, xx = np.mgrid[0:sh, 0:sw]
    diag = (xx / float(sw)) * 0.75 + (yy / float(sh)) * 0.25
    sweep_band = np.exp(-((diag - light_sweep) ** 2) / (2.0 * (0.085 ** 2)))

    # Edge rim highlight
    grad_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
    rim = np.clip((-grad_x - grad_y) * 0.45, 0.0, 1.0)

    face_rgb = rgb.copy()
    face_rgb[:, :, 0] = np.clip(face_rgb[:, :, 0] + sweep_band * 55.0 + rim * 95.0, 0, 255)
    face_rgb[:, :, 1] = np.clip(face_rgb[:, :, 1] + sweep_band * 85.0 + rim * 140.0, 0, 255)
    face_rgb[:, :, 2] = np.clip(face_rgb[:, :, 2] + sweep_band * 85.0 + rim * 140.0, 0, 255)

    front = canvas[pad:pad+sh, pad:pad+sw]
    f_alpha = alpha[:, :, None]
    front[:, :, :3] = front[:, :, :3] * (1.0 - f_alpha) + face_rgb * f_alpha
    front[:, :, 3] = np.maximum(front[:, :, 3], alpha * 255.0)

    # Optional subtle 3D perspective tilt
    if abs(tilt_deg) > 0.1:
        pts1 = np.float32([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]])
        dx = math.sin(math.radians(tilt_deg)) * 18.0
        dy = abs(dx) * 0.35
        pts2 = np.float32([
            [dx, dy],
            [canvas_w - dx, -dy if tilt_deg > 0 else dy],
            [canvas_w + dx, canvas_h - dy],
            [-dx, canvas_h + dy]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        canvas = cv2.warpPerspective(canvas, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # Alpha composite onto frame_rgb centered at (center_x, center_y)
    x0 = int(center_x - canvas_w // 2)
    y0 = int(center_y - canvas_h // 2)
    cx0 = max(0, x0)
    cy0 = max(0, y0)
    cx1 = min(W, x0 + canvas_w)
    cy1 = min(H, y0 + canvas_h)
    if cx1 <= cx0 or cy1 <= cy0:
        return frame_rgb

    sub_c = canvas[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    sub_f = frame_rgb[cy0:cy1, cx0:cx1].astype(np.float32)
    a_blend = (sub_c[:, :, 3:4] / 255.0) * opacity
    out_region = sub_f * (1.0 - a_blend) + sub_c[:, :, :3] * a_blend
    frame_rgb[cy0:cy1, cx0:cx1] = np.clip(out_region, 0, 255).astype(np.uint8)
    return frame_rgb

def camera_transform(img_rgb, zoom=1.0, pan_x=0.0, pan_y=0.0, roll_deg=0.0):
    """Smooth 3D camera zoom, pan, and subtle roll."""
    cx = W / 2.0 + pan_x
    cy = H / 2.0 + pan_y
    M = cv2.getRotationMatrix2D((cx, cy), roll_deg, zoom)
    M[0, 2] += (W / 2.0) - cx
    M[1, 2] += (H / 2.0) - cy
    return cv2.warpAffine(img_rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def draw_headline_banner(frame_rgb, lines, y_top, progress=1.0, accent_line=True, subtext=None, font_size=56):
    """
    Renders Apple-style studio typography with subtle glass depth, crisp drop shadow,
    and an animated red-and-white laser accent bar.
    """
    if progress <= 0.01:
        return frame_rgb
    ease = ease_out_cubic(progress)
    y_offset = int((1.0 - ease) * 32)
    alpha_mult = min(1.0, progress * 1.4)

    pil_im = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = get_font(800, font_size)
    line_h = int(font_size * 1.22)
    total_h = len(lines) * line_h + (54 if subtext else 0)

    # Frosted dark studio gradient vignette behind text for 100% legibility
    box_top = max(0, y_top + y_offset - 36)
    box_bot = min(H, y_top + y_offset + total_h + 44)
    draw.rounded_rectangle(
        [54, box_top, W - 54, box_bot],
        radius=28,
        fill=(8, 10, 15, int(175 * alpha_mult)),
        outline=(255, 255, 255, int(38 * alpha_mult)),
        width=2
    )

    if accent_line:
        bar_w = int(140 * ease)
        bx0 = (W - bar_w) // 2
        by0 = box_top + 18
        draw.rounded_rectangle([bx0, by0, bx0 + bar_w, by0 + 5], radius=3, fill=(253, 14, 10, int(245 * alpha_mult)))

    cur_y = y_top + y_offset + (12 if accent_line else 0)
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        tx = (W - tw) // 2
        # Soft drop shadow
        draw.text((tx + 2, cur_y + 4), line, font=font, fill=(0, 0, 0, int(210 * alpha_mult)))
        # Highlight key words in Brandrums crimson/white if desired, or crisp studio white
        draw.text((tx, cur_y), line, font=font, fill=(255, 255, 255, int(252 * alpha_mult)))
        cur_y += line_h

    if subtext:
        sub_font = get_font(600, 30)
        sb = draw.textbbox((0, 0), subtext, font=sub_font)
        sw = sb[2] - sb[0]
        draw.text(((W - sw) // 2, cur_y + 10), subtext, font=sub_font, fill=(253, 85, 80, int(235 * alpha_mult)))

    comp = Image.alpha_composite(pil_im, overlay).convert("RGB")
    return np.array(comp, dtype=np.uint8)

def make_3d_glass_card(w, h, title, subtitle, badge_text, is_chaos=False, fill_pct=1.0, highlight_red=False):
    """Creates a high-detail frosted glass 3D UI card image (RGBA)."""
    card = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)

    bg_col = (24, 14, 18, 215) if is_chaos else (14, 18, 26, 220)
    border_col = (253, 45, 40, 165) if (is_chaos or highlight_red) else (255, 255, 255, 95)

    d.rounded_rectangle([4, 4, w - 5, h - 5], radius=24, fill=bg_col, outline=border_col, width=2)
    # Top glass specular highlight line
    d.line([(28, 6), (w - 28, 6)], fill=(255, 255, 255, 130), width=2)

    # Left status indicator orb
    orb_col = (253, 28, 20, 255) if is_chaos else (253, 14, 10, 255)
    d.ellipse([26, 28, 54, 56], fill=orb_col)
    if not is_chaos:
        # Checkmark inside orb
        d.line([(34, 42), (39, 47), (47, 36)], fill=(255, 255, 255, 255), width=3)
    else:
        # Exclamation inside orb
        d.line([(40, 34), (40, 44)], fill=(255, 255, 255, 255), width=3)
        d.ellipse([38, 48, 42, 52], fill=(255, 255, 255, 255))

    f_title = get_font(700, 26)
    f_sub = get_font(500, 21)
    f_badge = get_font(700, 18)

    d.text((72, 24), title, font=f_title, fill=(255, 255, 255, 250))
    d.text((72, 58), subtitle, font=f_sub, fill=(195, 205, 220, 225))

    # Badge pill on top right
    if badge_text:
        bb = d.textbbox((0, 0), badge_text, font=f_badge)
        bw = bb[2] - bb[0] + 24
        bx = w - bw - 22
        by = 24
        b_fill = (180, 15, 12, 230) if is_chaos else (253, 14, 10, 235)
        d.rounded_rectangle([bx, by, bx + bw, by + 30], radius=15, fill=b_fill)
        d.text((bx + 12, by + 5), badge_text, font=f_badge, fill=(255, 255, 255, 255))

    # Progress / workflow bar at bottom
    bar_y = h - 26
    d.rounded_rectangle([26, bar_y, w - 26, bar_y + 8], radius=4, fill=(255, 255, 255, 35))
    prog_w = int((w - 52) * max(0.08, min(1.0, fill_pct)))
    bar_col = (253, 40, 30, 240) if is_chaos else (253, 14, 10, 255)
    d.rounded_rectangle([26, bar_y, 26 + prog_w, bar_y + 8], radius=4, fill=bar_col)

    return np.array(card, dtype=np.uint8)

def warp_and_paste_3d_card(frame_rgb, card_rgba, cx, cy, scale=1.0, yaw_deg=0.0, pitch_deg=0.0, opacity=1.0):
    if opacity <= 0.01 or scale <= 0.05:
        return frame_rgb
    ch, cw = card_rgba.shape[:2]
    sw = max(24, int(cw * scale))
    sh = max(16, int(ch * scale))
    scaled = cv2.resize(card_rgba, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    pad = 40
    pw, ph = sw + pad * 2, sh + pad * 2
    padded = np.zeros((ph, pw, 4), dtype=np.uint8)
    padded[pad:pad+sh, pad:pad+sw] = scaled

    # 3D perspective homography
    pts1 = np.float32([[pad, pad], [pad + sw, pad], [pad + sw, pad + sh], [pad, pad + sh]])
    dx = math.sin(math.radians(yaw_deg)) * (sw * 0.14)
    dy = math.sin(math.radians(pitch_deg)) * (sh * 0.18)
    pts2 = np.float32([
        [pad + dy, pad + dx * 0.35],
        [pad + sw - dy, pad - dx * 0.35],
        [pad + sw + dy, pad + sh + dx * 0.35],
        [pad - dy, pad + sh - dx * 0.35]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(padded, M, (pw, ph), flags=cv2.INTER_LINEAR)

    x0 = int(cx - pw // 2)
    y0 = int(cy - ph // 2)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + pw), min(H, y0 + ph)
    if cx1 <= cx0 or cy1 <= cy0:
        return frame_rgb

    sub_c = warped[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(np.float32)
    sub_f = frame_rgb[cy0:cy1, cx0:cx1].astype(np.float32)
    a = (sub_c[:, :, 3:4] / 255.0) * opacity
    frame_rgb[cy0:cy1, cx0:cx1] = np.clip(sub_f * (1.0 - a) + sub_c[:, :, :3] * a, 0, 255).astype(np.uint8)
    return frame_rgb

def draw_glowing_pathway(frame_rgb, p0, p1, p2, progress=1.0, pulse_t=0.5, color=(253, 14, 10)):
    """Draws a glowing 3D quadratic Bezier fiber-optic data pathway with a traveling white/red energy packet."""
    if progress <= 0.01:
        return frame_rgb
    n_pts = int(65 * min(1.0, progress))
    if n_pts < 2:
        return frame_rgb
    t_vals = np.linspace(0.0, min(1.0, progress), n_pts)
    pts = []
    for t in t_vals:
        x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        pts.append([int(x), int(y)])
    pts_arr = np.array(pts, dtype=np.int32)

    overlay = frame_rgb.copy()
    cv2.polylines(overlay, [pts_arr], False, color, thickness=8, lineType=cv2.LINE_AA)
    frame_rgb = cv2.addWeighted(overlay, 0.35, frame_rgb, 0.65, 0)
    cv2.polylines(frame_rgb, [pts_arr], False, color, thickness=3, lineType=cv2.LINE_AA)
    cv2.polylines(frame_rgb, [pts_arr], False, (255, 225, 225), thickness=1, lineType=cv2.LINE_AA)

    # Traveling energy pulse orb
    pt = pulse_t % 1.0
    if pt <= progress:
        px = int((1 - pt)**2 * p0[0] + 2 * (1 - pt) * pt * p1[0] + pt**2 * p2[0])
        py = int((1 - pt)**2 * p0[1] + 2 * (1 - pt) * pt * p1[1] + pt**2 * p2[1])
        glow = frame_rgb.copy()
        cv2.circle(glow, (px, py), 18, (253, 30, 20), -1, lineType=cv2.LINE_AA)
        frame_rgb = cv2.addWeighted(glow, 0.45, frame_rgb, 0.55, 0)
        cv2.circle(frame_rgb, (px, py), 7, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    return frame_rgb

class BrandrumsFilmRenderer:
    def __init__(self):
        print("Loading 3D studio environment plates & official Brandrums logo...")
        self.bg_s1 = load_and_fit_9_16(os.path.join(ASSETS_DIR, "scene1_3d_chaos_office.png"))
        self.bg_s2 = load_and_fit_9_16(os.path.join(ASSETS_DIR, "scene2_3d_pulse_pedestal.png"))
        self.bg_s3 = load_and_fit_9_16(os.path.join(ASSETS_DIR, "scene3_3d_ecosystem.png"))
        self.bg_s4 = load_and_fit_9_16(os.path.join(ASSETS_DIR, "scene4_3d_thriving_office.png"))
        self.bg_s5 = load_and_fit_9_16(os.path.join(ASSETS_DIR, "scene5_3d_brand_pedestal_dark.png"))
        self.bg_s6 = load_and_fit_9_16(os.path.join(ASSETS_DIR, "scene6_3d_final_cta.png"))

        self.logo_rgba = prepare_official_3d_logo(target_w=760)

        # Prebuild Chaos cards for Scene 1 & early Scene 2
        chaos_specs = [
            ("Missed Call (4)", "Incoming Customer Inquiry • Unanswered", "MISSED", 0.92),
            ("Unread Website Lead", "Waiting 3h 40m for Quote Response", "URGENT", 0.85),
            ("Manual_Leads_v9.xlsx", "Copy-Pasting 48 Rows to CRM...", "MANUAL", 0.78),
            ("Appointment Conflict", "Calendar Double-Booked • No Confirmation", "DELAY", 0.90),
            ("Follow-Up Overdue", "Prospects Waiting on Proposal", "OVERDUE", 0.88),
            ("Voicemail Inbox Full", "After-Hours Customer Calls Dropped", "ALERT", 0.95),
            ("Repetitive Admin Task", "Manual Intake & Invoice Entry Pending", "BUSYWORK", 0.80),
            ("Unanswered Chat", "Visitor Left Website Without Booking", "LOST", 0.94),
        ]
        self.chaos_cards = [
            make_3d_glass_card(540, 118, title, sub, badge, is_chaos=True, fill_pct=pct)
            for (title, sub, badge, pct) in chaos_specs
        ]

        # Prebuild Scene 3 Automation Machine cards
        self.s3_cards = [
            make_3d_glass_card(780, 134, "Website Visitor Inquiry", "AI Assistant captures lead & context in real time", "01 • CAPTURE", is_chaos=False, fill_pct=1.0, highlight_red=True),
            make_3d_glass_card(780, 134, "Instant AI Concierge Reply", "Qualifies intent & answers questions immediately", "02 • RESPOND", is_chaos=False, fill_pct=1.0, highlight_red=True),
            make_3d_glass_card(780, 134, "Automated Workflow & Calendar", "Confirms appointment & completes admin tasks", "03 • AUTOMATE", is_chaos=False, fill_pct=1.0, highlight_red=True),
            make_3d_glass_card(780, 134, "CRM Synced & Follow-Up Active", "Customer record updated • Smart nurture initiated", "04 • FOLLOW UP", is_chaos=False, fill_pct=1.0, highlight_red=True),
        ]

    def render_scene1_chaos(self, t, freeze=False, dissolve=0.0):
        """
        SCENE 1 (0 - 4s): CHAOS
        Dramatic 3D camera race into miniature office building surrounded by swirling chaos notifications.
        Also supports the 4.0s freeze + dissolve in Scene 2.
        """
        eff_t = 4.0 if freeze else t
        prog = eff_t / 4.0
        zoom = 1.02 + 0.24 * (prog ** 1.35)
        shake_x = 0.0 if freeze else math.sin(eff_t * 19.0) * (10.0 * prog)
        shake_y = 0.0 if freeze else math.cos(eff_t * 23.0) * (8.0 * prog)
        roll = 0.0 if freeze else math.sin(eff_t * 9.5) * (1.8 * prog)

        frame = camera_transform(self.bg_s1, zoom=zoom, pan_x=shake_x, pan_y=shake_y, roll_deg=roll)

        # Render swirling 3D chaos cards around the miniature building
        card_positions = [
            (290, 560, -14, 8, 0.15),
            (780, 650, 16, -6, 0.35),
            (310, 860, -12, -8, 0.60),
            (760, 1020, 15, 10, 0.85),
            (330, 1210, -16, 7, 1.15),
            (750, 1380, 14, -9, 1.45),
            (540, 750, -6, 11, 1.75),
            (540, 1130, 8, -10, 2.05),
        ]
        for i, (bx, by, yaw, pitch, spawn_t) in enumerate(card_positions):
            if eff_t < spawn_t:
                continue
            local_t = (eff_t - spawn_t) / max(0.4, 4.0 - spawn_t)
            c_scale = (0.58 + 0.34 * min(1.0, local_t * 1.4)) * (1.0 - 0.35 * dissolve)
            orbit_x = bx + math.sin(eff_t * 2.8 + i) * 38.0 * (1.0 + dissolve * 2.5)
            orbit_y = by + math.cos(eff_t * 3.1 + i * 1.3) * 28.0 - dissolve * 90.0
            c_op = min(1.0, (eff_t - spawn_t) * 3.5) * (1.0 - dissolve)
            frame = warp_and_paste_3d_card(
                frame, self.chaos_cards[i % len(self.chaos_cards)],
                orbit_x, orbit_y,
                scale=c_scale,
                yaw_deg=yaw + math.sin(eff_t * 3.0 + i) * 6.0,
                pitch_deg=pitch,
                opacity=c_op
            )

        if not freeze:
            h_prog = smoothstep(0.35, 1.15, t)
            frame = draw_headline_banner(
                frame,
                ["YOUR BUSINESS IS", "WORKING TOO HARD."],
                y_top=185,
                progress=h_prog,
                accent_line=True,
                subtext="MISSED CALLS • MANUAL BUSYWORK • DELAYED LEADS",
                font_size=56
            )
        return frame

    def render_scene2_reveal(self, t):
        """
        SCENE 2 (4 - 8s): THE BRANDRUMS REVEAL
        Everything freezes at 4.0s -> Glowing red-and-white energy pulse sweeps across ->
        Official 3D Brandrums logo materializes and activates interconnected business nodes.
        """
        local_t = t - 4.0  # 0.0 to 4.0

        # 0.0s - 0.85s: Frozen chaos morphs via glowing red-and-white shockwave into the 3D pulse pedestal
        freeze_frame = self.render_scene1_chaos(4.0, freeze=True, dissolve=smoothstep(0.25, 1.0, local_t))
        ped_zoom = 1.08 - 0.08 * (local_t / 4.0)
        ped_frame = camera_transform(self.bg_s2, zoom=ped_zoom, pan_x=0, pan_y=-15 * (local_t / 4.0))

        blend = smoothstep(0.20, 1.05, local_t)
        frame = cv2.addWeighted(ped_frame, blend, freeze_frame, 1.0 - blend, 0)

        # Glowing Red-and-White Energy Pulse Wave (0.2s -> 1.8s)
        if 0.15 < local_t < 2.2:
            pulse_p = (local_t - 0.15) / 1.65
            radius = int(pulse_p * 1250)
            cx, cy = W // 2, H // 2 - 40
            ring_layer = np.zeros_like(frame)
            cv2.circle(ring_layer, (cx, cy), radius, (253, 14, 10), thickness=46, lineType=cv2.LINE_AA)
            cv2.circle(ring_layer, (cx, cy), max(1, radius - 14), (255, 255, 255), thickness=16, lineType=cv2.LINE_AA)
            ring_layer = cv2.GaussianBlur(ring_layer, (31, 31), 12.0)
            ring_alpha = max(0.0, 1.0 - pulse_p * 0.85)
            frame = cv2.addWeighted(frame, 1.0, ring_layer, ring_alpha * 0.85, 0)

        # Interconnected business system nodes activated by the pulse (1.0s -> 4.0s)
        center_logo = (W // 2, 790)
        nodes = [
            (230, 1180, "WEBSITE AI"),
            (850, 1180, "LEAD ENGINE"),
            (190, 1460, "SMART CRM"),
            (890, 1460, "CALENDAR"),
            (540, 1590, "WORKFLOWS"),
        ]
        net_prog = smoothstep(0.95, 2.4, local_t)
        if net_prog > 0.01:
            for idx, (nx, ny, label) in enumerate(nodes):
                ctrl = ((center_logo[0] + nx) // 2, min(center_logo[1], ny) - 60)
                frame = draw_glowing_pathway(
                    frame, center_logo, ctrl, (nx, ny),
                    progress=net_prog,
                    pulse_t=(local_t * 0.85 + idx * 0.2),
                    color=(253, 18, 14)
                )
            # Draw 3D glass node pills
            pil_f = Image.fromarray(frame).convert("RGBA")
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            f_node = get_font(700, 22)
            for idx, (nx, ny, label) in enumerate(nodes):
                np_i = smoothstep(1.1 + idx * 0.15, 1.8 + idx * 0.15, local_t)
                if np_i <= 0.01:
                    continue
                pw, ph = 220, 56
                x0, y0 = nx - pw // 2, ny - ph // 2
                d.rounded_rectangle(
                    [x0, y0, x0 + pw, y0 + ph],
                    radius=28,
                    fill=(14, 18, 26, int(230 * np_i)),
                    outline=(253, 30, 24, int(220 * np_i)),
                    width=2
                )
                d.ellipse([x0 + 16, y0 + 18, x0 + 36, y0 + 38], fill=(253, 14, 10, int(255 * np_i)))
                d.text((x0 + 48, y0 + 15), label, font=f_node, fill=(255, 255, 255, int(250 * np_i)))
            frame = np.array(Image.alpha_composite(pil_f, ov).convert("RGB"), dtype=np.uint8)

        # Dimensional 3D Official Brandrums Logo Reveal in studio pedestal space
        logo_prog = smoothstep(0.55, 1.55, local_t)
        if logo_prog > 0.01:
            l_scale = 0.82 + 0.18 * ease_out_cubic(logo_prog)
            sweep = (local_t * 0.42) % 1.2
            tilt = math.sin(local_t * 1.8) * 3.5
            frame = render_3d_logo_on_frame(
                frame, self.logo_rgba,
                center_x=W // 2, center_y=760,
                scale=l_scale, light_sweep=sweep,
                opacity=min(1.0, logo_prog * 1.25),
                tilt_deg=tilt
            )

        # Headline: "MEET YOUR NEW AI ADVANTAGE."
        h_prog = smoothstep(0.75, 1.55, local_t)
        frame = draw_headline_banner(
            frame,
            ["MEET YOUR NEW", "AI ADVANTAGE."],
            y_top=185,
            progress=h_prog,
            accent_line=True,
            subtext="INTELLIGENT SYSTEMS • ENGINEERED BY BRANDRUMS",
            font_size=58
        )
        return frame

    def render_scene3_automation_machine(self, t):
        """
        SCENE 3 (8 - 16s): THE AUTOMATION MACHINE
        Continuous 3D camera sequence across the interconnected miniature business pods.
        Synchronized on-screen text:
        - 8-10s: CAPTURE LEADS
        - 10-12s: RESPOND INSTANTLY
        - 12-14s: AUTOMATE WORKFLOWS
        - 14-16s: FOLLOW UP AUTOMATICALLY
        """
        local_t = t - 8.0  # 0.0 to 8.0
        prog = local_t / 8.0

        # Continuous cinematic 3D camera path gliding across the miniature glass pods
        zoom = 1.06 + 0.12 * math.sin(prog * math.pi)
        pan_x = math.sin(prog * math.pi * 2.0) * 34.0
        pan_y = -35.0 + 70.0 * prog
        frame = camera_transform(self.bg_s3, zoom=zoom, pan_x=pan_x, pan_y=pan_y)

        # Animated volumetric light pulses traveling along the 3D fiber-optic pathways between the glass pods
        fiber_paths = [
            [(500, 495), (430, 585), (345, 650)],
            [(585, 505), (630, 595), (725, 665)],
            [(375, 795), (430, 845), (485, 880)],
            [(715, 815), (655, 850), (605, 885)],
            [(495, 965), (470, 1045), (475, 1135)],
            [(615, 965), (665, 1015), (715, 1055)],
            [(435, 1285), (360, 1440), (265, 1680)],
        ]
        for idx, (p0, p1, p2) in enumerate(fiber_paths):
            pt = (local_t * 0.85 + idx * 0.19) % 1.0
            px = int((1 - pt)**2 * p0[0] + 2 * (1 - pt) * pt * p1[0] + pt**2 * p2[0])
            py = int((1 - pt)**2 * p0[1] + 2 * (1 - pt) * pt * p1[1] + pt**2 * p2[1])
            glow = frame.copy()
            cv2.circle(glow, (px, py), 24, (253, 24, 16), -1, lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(glow, 0.50, frame, 0.50, 0)
            cv2.circle(frame, (px, py), 8, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # Determine active stage (0, 1, 2, 3) for every 2.0-second window
        stage = min(3, int(local_t // 2.0))
        stage_t = (local_t - stage * 2.0) / 2.0  # 0.0 -> 1.0 within current 2s stage

        headlines = [
            (["CAPTURE LEADS"], "VISITOR INQUIRY DETECTED • DATA STRUCTURED"),
            (["RESPOND INSTANTLY"], "24/7 AI ASSISTANT • ZERO WAIT TIME"),
            (["AUTOMATE WORKFLOWS"], "CALENDAR CONFIRMED • ADMIN TASKS COMPLETED"),
            (["FOLLOW UP AUTOMATICALLY"], "CRM RECORD UPDATED • SMART NURTURE ACTIVE"),
        ]
        hl_lines, hl_sub = headlines[stage]
        hl_prog = smoothstep(0.0, 0.25, stage_t)
        frame = draw_headline_banner(
            frame, hl_lines, y_top=175,
            progress=hl_prog, accent_line=True,
            subtext=hl_sub, font_size=60
        )

        # Render the stacked 3D glass workflow cards at the lower half of the viewport
        # Previous completed stages remain visible at slightly reduced scale while the active card pops in 3D
        base_y = 1340
        for s_idx in range(stage + 1):
            is_current = (s_idx == stage)
            c_prog = smoothstep(0.0, 0.30, stage_t) if is_current else 1.0
            y_pos = 1260 + (s_idx - stage) * 148 + (0 if is_current else -20)
            if y_pos < 880:
                continue
            c_scale = (0.96 if is_current else 0.84) * ease_out_cubic(c_prog)
            yaw = math.sin(local_t * 2.0 + s_idx) * 4.5 if is_current else 0.0
            op = c_prog if is_current else 0.55
            frame = warp_and_paste_3d_card(
                frame, self.s3_cards[s_idx],
                W // 2, y_pos,
                scale=c_scale, yaw_deg=yaw, pitch_deg=-3.0, opacity=op
            )

        # Subtle Brandrums watermark badge in top corner
        frame = render_3d_logo_on_frame(frame, self.logo_rgba, center_x=W // 2, center_y=96, scale=0.34, light_sweep=(local_t * 0.3) % 1.0, opacity=0.92)
        return frame

    def render_scene4_transformation(self, t):
        """
        SCENE 4 (16 - 22s): TRANSFORMATION
        Camera pulls back to reveal the clean, organized, thriving executive office.
        Completed workflows illuminate one by one.
        Headline: "LESS REPETITIVE WORK. MORE ROOM TO GROW."
        """
        local_t = t - 16.0  # 0.0 to 6.0
        prog = local_t / 6.0

        # Smooth camera pullback from 1.16x -> 1.00x
        zoom = 1.16 - 0.16 * ease_out_cubic(prog)
        pan_y = -24.0 * (1.0 - ease_out_cubic(prog))
        frame = camera_transform(self.bg_s4, zoom=zoom, pan_x=0.0, pan_y=pan_y)

        # Completed workflow modules illuminating one by one in the organized environment (no fake numbers!)
        modules = [
            ("INBOUND LEADS", "Captured & Qualified Automatically", 0.5),
            ("CLIENT CALENDAR", "Appointments Confirmed Seamlessly", 1.5),
            ("CRM PIPELINE", "Customer Records Synced in Real Time", 2.5),
            ("DAILY OPERATIONS", "Repetitive Busywork Completed", 3.5),
        ]

        pil_f = Image.fromarray(frame).convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        f_t = get_font(700, 23)
        f_s = get_font(500, 19)

        for idx, (m_title, m_sub, start_s) in enumerate(modules):
            mp = smoothstep(start_s, start_s + 0.65, local_t)
            if mp <= 0.01:
                continue
            col = idx % 2
            row = idx // 2
            cw_box, ch_box = 460, 102
            x0 = 64 if col == 0 else (W - 64 - cw_box)
            y0 = 1440 + row * 124 + int((1.0 - ease_out_cubic(mp)) * 22)

            # Illuminated crimson-and-white border glow when newly activated
            glow_boost = math.exp(-max(0.0, local_t - start_s) * 2.2)
            border_r = int(200 + 53 * glow_boost)
            border_g = int(45 + 160 * (1.0 - glow_boost))
            border_b = int(45 + 160 * (1.0 - glow_boost))

            d.rounded_rectangle(
                [x0, y0, x0 + cw_box, y0 + ch_box],
                radius=20,
                fill=(10, 14, 22, int(218 * mp)),
                outline=(border_r, border_g, border_b, int(210 * mp)),
                width=2
            )
            # Illuminated status pill
            d.ellipse([x0 + 20, y0 + 26, x0 + 44, y0 + 50], fill=(253, 14, 10, int(255 * mp)))
            d.line([(x0 + 27, y0 + 38), (x0 + 31, y0 + 43), (x0 + 38, y0 + 33)], fill=(255, 255, 255, int(255 * mp)), width=3)
            d.text((x0 + 58, y0 + 20), m_title, font=f_t, fill=(255, 255, 255, int(250 * mp)))
            d.text((x0 + 58, y0 + 54), m_sub, font=f_s, fill=(195, 205, 220, int(225 * mp)))

        frame = np.array(Image.alpha_composite(pil_f, ov).convert("RGB"), dtype=np.uint8)

        # Official logo top header
        frame = render_3d_logo_on_frame(frame, self.logo_rgba, center_x=W // 2, center_y=96, scale=0.34, light_sweep=(local_t * 0.25) % 1.0, opacity=0.92)

        # Headline: "LESS REPETITIVE WORK. MORE ROOM TO GROW."
        h_prog = smoothstep(0.25, 1.10, local_t)
        frame = draw_headline_banner(
            frame,
            ["LESS REPETITIVE WORK.", "MORE ROOM TO GROW."],
            y_top=175,
            progress=h_prog,
            accent_line=True,
            subtext="FOCUS ON HIGH-VALUE STRATEGY WHILE AI RUNS THE REST",
            font_size=52
        )
        return frame

    def render_scene5_brand_identity(self, t):
        """
        SCENE 5 (22 - 26s): BRAND IDENTITY
        Premium Apple-style product shot composition featuring the official Brandrums 3D logo
        above the circular dark glass & brushed titanium studio pedestal.
        Headline: "SMARTER BUSINESS STARTS HERE."
        """
        local_t = t - 22.0  # 0.0 to 4.0
        prog = local_t / 4.0

        zoom = 1.00 + 0.08 * ease_in_out_cubic(prog)
        frame = camera_transform(self.bg_s5, zoom=zoom, pan_x=0.0, pan_y=0.0)

        # Subtle floating animation for the 3D dimensional Brandrums logo levitating above the glowing pedestal
        float_y = 635 + int(math.sin(local_t * 2.2) * 10.0)
        l_prog = smoothstep(0.0, 0.85, local_t)
        l_scale = 0.88 + 0.10 * ease_out_cubic(l_prog)
        sweep = 0.15 + 0.75 * prog

        # Soft crimson reflection on the dark smoked-glass pedestal top surface (500, 795)
        refl_glow = frame.copy()
        cv2.ellipse(refl_glow, (500, 795), (125, 36), 0, 0, 360, (253, 18, 14), -1, lineType=cv2.LINE_AA)
        refl_glow = cv2.GaussianBlur(refl_glow, (51, 51), 18.0)
        frame = cv2.addWeighted(frame, 0.72, refl_glow, 0.28 * l_prog, 0)

        frame = render_3d_logo_on_frame(
            frame, self.logo_rgba,
            center_x=W // 2, center_y=float_y,
            scale=l_scale, light_sweep=sweep,
            opacity=min(1.0, l_prog * 1.3),
            tilt_deg=math.sin(local_t * 1.6) * 2.5
        )

        # Headline: "SMARTER BUSINESS STARTS HERE."
        h_prog = smoothstep(0.25, 1.0, local_t)
        frame = draw_headline_banner(
            frame,
            ["SMARTER BUSINESS", "STARTS HERE."],
            y_top=190,
            progress=h_prog,
            accent_line=True,
            subtext="CUSTOM AI AUTOMATION • WEB & DIGITAL ECOSYSTEMS",
            font_size=58
        )
        return frame

    def render_scene6_final_cta(self, t):
        """
        SCENE 6 (26 - 30s): POWERFUL CALL TO ACTION
        Brandrums logo suspended above the 3D miniature business environment.
        Single elegant red accent travels through the connected systems.
        Exact required text:
        - "STOP DOING EVERYTHING MANUALLY."
        - "LET AI DO THE REPETITIVE WORK."
        - "BUILD YOUR AUTOMATION STRATEGY"
        - "BRANDRUMS.COM"
        """
        local_t = t - 26.0  # 0.0 to 4.0
        prog = local_t / 4.0

        # Camera settles into a clean, balanced final frame
        zoom = 1.06 - 0.06 * ease_out_cubic(min(1.0, local_t / 2.2))
        frame = camera_transform(self.bg_s6, zoom=zoom, pan_x=0.0, pan_y=0.0)

        # Single elegant red accent pulse traveling along the 3D elevated skybridge circuit of the campus
        circuit_pts = [
            (545, 855), (725, 965), (535, 1110), (355, 985), (545, 855)
        ]
        seg_total = len(circuit_pts) - 1
        cycle_pos = (local_t * 0.55) % 1.0
        seg_idx = min(seg_total - 1, int(cycle_pos * seg_total))
        seg_t = (cycle_pos * seg_total) - seg_idx
        p0 = circuit_pts[seg_idx]
        p2 = circuit_pts[seg_idx + 1]
        px = int(p0[0] + (p2[0] - p0[0]) * seg_t)
        py = int(p0[1] + (p2[1] - p0[1]) * seg_t)
        glow = frame.copy()
        cv2.circle(glow, (px, py), 34, (253, 20, 14), -1, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(glow, 0.55, frame, 0.45, 0)
        cv2.circle(frame, (px, py), 10, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # Official 3D Dimensional Brandrums Logo suspended cleanly at top
        l_prog = smoothstep(0.0, 0.7, local_t)
        sweep = 0.2 + 0.65 * ((local_t / 3.5) % 1.0)
        frame = render_3d_logo_on_frame(
            frame, self.logo_rgba,
            center_x=W // 2, center_y=250,
            scale=0.95, light_sweep=sweep,
            opacity=min(1.0, l_prog * 1.4),
            tilt_deg=0.0
        )

        # Render the exact 4 required CTA text lines with Apple-keynote precision
        pil_f = Image.fromarray(frame).convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)

        t1_prog = smoothstep(0.15, 0.75, local_t)
        t2_prog = smoothstep(0.45, 1.05, local_t)
        t3_prog = smoothstep(0.80, 1.45, local_t)
        t4_prog = smoothstep(1.05, 1.75, local_t)

        # Frosted studio card backing for upper headlines
        if t1_prog > 0.01:
            d.rounded_rectangle(
                [56, 405, W - 56, 650],
                radius=28,
                fill=(8, 10, 16, int(195 * t1_prog)),
                outline=(255, 255, 255, int(45 * t1_prog)),
                width=2
            )

        f_line1 = get_font(800, 42)
        f_line2 = get_font(700, 38)
        f_line3 = get_font(700, 32)
        f_url = get_font(900, 54)

        # Line 1: "STOP DOING EVERYTHING MANUALLY."
        if t1_prog > 0.01:
            txt1 = "STOP DOING EVERYTHING MANUALLY."
            b1 = d.textbbox((0, 0), txt1, font=f_line1)
            w1 = b1[2] - b1[0]
            y1 = 445 + int((1.0 - ease_out_cubic(t1_prog)) * 18)
            d.text(((W - w1) // 2 + 2, y1 + 3), txt1, font=f_line1, fill=(0, 0, 0, int(210 * t1_prog)))
            d.text(((W - w1) // 2, y1), txt1, font=f_line1, fill=(255, 255, 255, int(255 * t1_prog)))

        # Line 2: "LET AI DO THE REPETITIVE WORK."
        if t2_prog > 0.01:
            txt2 = "LET AI DO THE REPETITIVE WORK."
            b2 = d.textbbox((0, 0), txt2, font=f_line2)
            w2 = b2[2] - b2[0]
            y2 = 520 + int((1.0 - ease_out_cubic(t2_prog)) * 18)
            d.text(((W - w2) // 2 + 2, y2 + 3), txt2, font=f_line2, fill=(0, 0, 0, int(210 * t2_prog)))
            d.text(((W - w2) // 2, y2), txt2, font=f_line2, fill=(230, 235, 245, int(250 * t2_prog)))

        # Line 3: "BUILD YOUR AUTOMATION STRATEGY"
        if t3_prog > 0.01:
            txt3 = "BUILD YOUR AUTOMATION STRATEGY"
            b3 = d.textbbox((0, 0), txt3, font=f_line3)
            w3 = b3[2] - b3[0]
            y3 = 590 + int((1.0 - ease_out_cubic(t3_prog)) * 16)
            d.text(((W - w3) // 2, y3), txt3, font=f_line3, fill=(253, 75, 70, int(255 * t3_prog)))

        # Line 4: "BRANDRUMS.COM" CTA Hero Pill at lower studio frame
        if t4_prog > 0.01:
            txt4 = "BRANDRUMS.COM"
            b4 = d.textbbox((0, 0), txt4, font=f_url)
            w4 = b4[2] - b4[0]
            pill_w = w4 + 130
            pill_h = 114
            px0 = (W - pill_w) // 2
            py0 = 1585 + int((1.0 - ease_out_cubic(t4_prog)) * 24)

            # Outer crimson glow ring
            d.rounded_rectangle(
                [px0 - 8, py0 - 8, px0 + pill_w + 8, py0 + pill_h + 8],
                radius=64,
                fill=(253, 14, 10, int(65 * t4_prog)),
                outline=(253, 40, 35, int(140 * t4_prog)),
                width=2
            )
            # Main 3D glass & Brandrums crimson pill
            d.rounded_rectangle(
                [px0, py0, px0 + pill_w, py0 + pill_h],
                radius=57,
                fill=(215, 10, 8, int(245 * t4_prog)),
                outline=(255, 255, 255, int(210 * t4_prog)),
                width=3
            )
            # Top specular highlight bar
            d.line([(px0 + 48, py0 + 6), (px0 + pill_w - 48, py0 + 6)], fill=(255, 255, 255, int(180 * t4_prog)), width=2)

            tx4 = (W - w4) // 2
            ty4 = py0 + 25
            d.text((tx4 + 2, ty4 + 3), txt4, font=f_url, fill=(60, 0, 0, int(220 * t4_prog)))
            d.text((tx4, ty4), txt4, font=f_url, fill=(255, 255, 255, int(255 * t4_prog)))

        frame = np.array(Image.alpha_composite(pil_f, ov).convert("RGB"), dtype=np.uint8)
        return frame

    def render_frame(self, frame_idx):
        t = frame_idx / float(FPS)
        # Smooth optical cross-dissolve transitions (0.28s) at scene boundaries
        trans_w = 0.28

        if t < 4.0:
            return self.render_scene1_chaos(t)
        elif t < 8.0:
            return self.render_scene2_reveal(t)
        elif t < 16.0:
            if t < 8.0 + trans_w:
                a = (t - 8.0) / trans_w
                f_prev = self.render_scene2_reveal(8.0 - 0.01)
                f_curr = self.render_scene3_automation_machine(t)
                return cv2.addWeighted(f_curr, a, f_prev, 1.0 - a, 0)
            return self.render_scene3_automation_machine(t)
        elif t < 22.0:
            if t < 16.0 + trans_w:
                a = (t - 16.0) / trans_w
                f_prev = self.render_scene3_automation_machine(16.0 - 0.01)
                f_curr = self.render_scene4_transformation(t)
                return cv2.addWeighted(f_curr, a, f_prev, 1.0 - a, 0)
            return self.render_scene4_transformation(t)
        elif t < 26.0:
            if t < 22.0 + trans_w:
                a = (t - 22.0) / trans_w
                f_prev = self.render_scene4_transformation(22.0 - 0.01)
                f_curr = self.render_scene5_brand_identity(t)
                return cv2.addWeighted(f_curr, a, f_prev, 1.0 - a, 0)
            return self.render_scene5_brand_identity(t)
        else:
            if t < 26.0 + trans_w:
                a = (t - 26.0) / trans_w
                f_prev = self.render_scene5_brand_identity(26.0 - 0.01)
                f_curr = self.render_scene6_final_cta(t)
                return cv2.addWeighted(f_curr, a, f_prev, 1.0 - a, 0)
            return self.render_scene6_final_cta(t)

def main():
    renderer = BrandrumsFilmRenderer()
    if "--preview" in sys.argv:
        os.makedirs("/tmp/previews", exist_ok=True)
        test_times = [2.2, 6.2, 9.0, 13.0, 19.5, 24.2, 28.5]
        for tt in test_times:
            f_idx = int(tt * FPS)
            rgb = renderer.render_frame(f_idx)
            out_p = f"/tmp/previews/preview_{tt:.1f}s.jpg"
            Image.fromarray(rgb).save(out_p, quality=92)
            print("Saved preview:", out_p)
        return

    out_video = "/home/user/video/brandrums_ai_automation_commercial.mp4"
    audio_path = os.path.join(ASSETS_DIR, "soundtrack_master.wav")
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
