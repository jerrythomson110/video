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

def ease_out_back(x):
    x = max(0.0, min(1.0, x))
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * ((x - 1.0) ** 3) + c1 * ((x - 1.0) ** 2)

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

def build_boom_sss_ribbon_body(body_w=380, body_h=440):
    """
    Constructs Boom's chunky stylized folded red-ribbon 'B' torso from the official Brandrums B silhouette,
    shaded with Octane-style subsurface scattering (SSS) soft-touch red rubber (#D50000 -> #9E0000),
    smooth diagonal folded-ribbon creases along the left spine, and a single red rim light on the left contour.
    """
    raw = Image.open(os.path.join(ASSETS_DIR, "brandrums_official_logo.png")).convert("RGBA")
    rw, rh = raw.size
    b_crop = raw.crop((0, 0, int(rw * 0.295), rh)).resize((body_w, body_h), Image.Resampling.LANCZOS)
    arr = np.array(b_crop, dtype=np.float32)

    alpha = arr[:, :, 3]
    mask_bin = (alpha > 90).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(mask_bin)
    cv2.drawContours(solid_mask, contours, -1, 255, -1)
    solid_blur = cv2.GaussianBlur(solid_mask.astype(np.float32), (17, 17), 4.5)
    solid_alpha = np.clip((solid_blur - 95.0) * (255.0 / 45.0), 0, 255)

    # Distance transform for puffy chunky 3D soft-touch rubber normal map
    dist = cv2.distanceTransform((solid_alpha > 128).astype(np.uint8), cv2.DIST_L2, 5)
    dist_norm = np.clip(dist / 78.0, 0.0, 1.0)
    height_map = np.sin(dist_norm * (math.pi / 2.0)) ** 0.85

    yy, xx = np.mgrid[0:body_h, 0:body_w]
    nx = (xx - body_w * 0.46) / (body_w * 0.5)
    ny = (yy - body_h * 0.45) / (body_h * 0.5)

    # Smooth diagonal folded-ribbon creases on the left side & right B-waist (preserving the iconic folded ribbon B look
    # without dark rectangular blocks behind the eyes and mouth!)
    left_weight = np.clip(1.0 - (xx / (body_w * 0.42)), 0.0, 1.0) ** 1.4
    diag_wave = np.sin((yy / float(body_h)) * math.pi * 3.2 + (xx / float(body_w)) * 2.2)
    ribbon_crease = 1.0 - 0.24 * left_weight * np.clip(-diag_wave, 0.0, 1.0)

    # Subtle B-loop indentation on the right side
    loop_top = np.exp(-(((xx - body_w * 0.54) / 72.0) ** 2 + ((yy - body_h * 0.30) / 52.0) ** 2))
    loop_bot = np.exp(-(((xx - body_w * 0.56) / 76.0) ** 2 + ((yy - body_h * 0.72) / 56.0) ** 2))
    b_sculpt = 1.0 - 0.10 * (loop_top + loop_bot)

    # Subsurface scattering red rubber gradient: bright red #D50000 (213, 0, 0) to deep red #9E0000 (158, 0, 0)
    grad_t = np.clip(0.22 * nx + 0.78 * (yy / float(body_h)), 0.0, 1.0)
    base_r = 216.0 * (1.0 - grad_t) + 156.0 * grad_t
    base_g = 14.0 * (1.0 - grad_t) + 2.0 * grad_t
    base_b = 14.0 * (1.0 - grad_t) + 4.0 * grad_t

    gx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=5)
    rim_left = np.clip(-gx * 0.72 - gy * 0.25, 0.0, 1.0) ** 1.3
    sss_edge = np.clip((1.0 - height_map) * 1.25, 0.0, 1.0) * (solid_alpha > 10)
    soft_spec = np.exp(-((nx + 0.22) ** 2 + (ny + 0.34) ** 2) / 0.26) * (height_map ** 1.4)

    shade_mult = ribbon_crease * b_sculpt * (0.68 + 0.38 * height_map)
    r_out = np.clip(base_r * shade_mult + rim_left * 82.0 + soft_spec * 46.0 + sss_edge * 26.0, 0, 255)
    g_out = np.clip(base_g * shade_mult + rim_left * 28.0 + soft_spec * 38.0 + sss_edge * 8.0, 0, 115)
    b_out = np.clip(base_b * shade_mult + rim_left * 24.0 + soft_spec * 36.0 + sss_edge * 8.0, 0, 110)

    out = np.zeros((body_h, body_w, 4), dtype=np.uint8)
    out[:, :, 0] = r_out.astype(np.uint8)
    out[:, :, 1] = g_out.astype(np.uint8)
    out[:, :, 2] = b_out.astype(np.uint8)
    out[:, :, 3] = solid_alpha.astype(np.uint8)
    return out

def draw_3d_glove(d, hx, hy, arm_pose="sticks_ready", is_left=True, finger_count=0):
    """Renders a 3D-shaded white cartoon glove with soft grey-blue spherical shading & puffy fingers."""
    # Ambient shadow under glove
    d.ellipse([hx - 34, hy - 26, hx + 34, hy + 34], fill=(0, 0, 0, 110))
    # Outer 3D shaded white palm sphere (cool shadow base -> warm white key light -> specular highlight)
    d.ellipse([hx - 31, hy - 27, hx + 31, hy + 29], fill=(205, 212, 224, 255), outline=(45, 50, 62, 255), width=2)
    d.ellipse([hx - 27, hy - 24, hx + 25, hy + 22], fill=(242, 245, 252, 255))
    d.ellipse([hx - 18, hy - 18, hx + 8, hy + 2], fill=(255, 255, 255, 255))

    # Puffy cartoon wrist cuff
    d.rounded_rectangle([hx - 23, hy + 15, hx + 23, hy + 31], radius=9, fill=(215, 222, 234, 255), outline=(45, 50, 62, 255), width=2)
    d.rounded_rectangle([hx - 19, hy + 17, hx + 17, hy + 27], radius=6, fill=(246, 248, 255, 255))

    # Special finger poses on left glove
    if is_left and arm_pose == "count_fingers" and finger_count > 0:
        for f_i in range(finger_count):
            fx = hx - 16 + f_i * 16
            d.rounded_rectangle([fx - 8, hy - 60, fx + 8, hy - 14], radius=8, fill=(215, 222, 234, 255), outline=(45, 50, 62, 255), width=2)
            d.rounded_rectangle([fx - 5, hy - 57, fx + 5, hy - 18], radius=5, fill=(252, 254, 255, 255))
    elif is_left and arm_pose == "point_down":
        d.rounded_rectangle([hx - 10, hy + 18, hx + 10, hy + 72], radius=9, fill=(215, 222, 234, 255), outline=(45, 50, 62, 255), width=2)
        d.rounded_rectangle([hx - 6, hy + 22, hx + 6, hy + 68], radius=6, fill=(252, 254, 255, 255))

def render_boom_character(
    frame_rgb, body_rgba, cx, cy, scale=1.0,
    squash_x=1.0, squash_y=1.0, tilt_deg=0.0,
    mouth_open=0.0, mouth_wide=0.6, smile=0.35,
    look_x=0.0, look_y=0.0,
    blink_l=0.0, blink_r=0.0, brow_l=0.0, brow_r=0.0,
    arm_pose="sticks_ready", finger_count=0, hit_phase=0.0,
    show_legs=True
):
    bh, bw = body_rgba.shape[:2]
    pad = 220
    cw, ch = bw + pad * 2, bh + pad * 2
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)

    bx0, by0 = pad, pad

    # 1. Tiny Black Sneakers & Stubby Red Rubber Legs
    if show_legs:
        leg_l_x = bx0 + int(bw * 0.34)
        leg_r_x = bx0 + int(bw * 0.68)
        hip_y = by0 + bh - 24
        ankle_y = hip_y + 56

        # 3D shaded stubby red rubber legs
        for lx_c in [leg_l_x, leg_r_x]:
            d.line([(lx_c, hip_y), (lx_c, ankle_y)], fill=(115, 0, 0, 255), width=30)
            d.line([(lx_c - 3, hip_y), (lx_c - 3, ankle_y)], fill=(195, 10, 10, 255), width=20)

        # Tiny Black 3D Sneakers with white rubber soles, toe caps & laces
        for sx_c, dir_s in [(leg_l_x - 8, -1), (leg_r_x + 8, 1)]:
            sy_c = ankle_y
            d.ellipse([sx_c - 48, sy_c + 16, sx_c + 52, sy_c + 42], fill=(0, 0, 0, 175))
            d.rounded_rectangle([sx_c - 42, sy_c - 14, sx_c + 46, sy_c + 26], radius=18, fill=(20, 22, 26, 255), outline=(65, 70, 82, 255), width=2)
            d.rounded_rectangle([sx_c - 34, sy_c - 10, sx_c + 20, sy_c + 6], radius=8, fill=(42, 46, 54, 255))
            if dir_s < 0:
                d.pieslice([sx_c - 44, sy_c - 10, sx_c - 6, sy_c + 26], start=90, end=270, fill=(240, 244, 250, 255))
            else:
                d.pieslice([sx_c + 8, sy_c - 10, sx_c + 46, sy_c + 26], start=270, end=90, fill=(240, 244, 250, 255))
            d.rounded_rectangle([sx_c - 44, sy_c + 18, sx_c + 48, sy_c + 30], radius=6, fill=(245, 247, 252, 255))
            d.line([(sx_c - 8, sy_c - 6), (sx_c + 8, sy_c - 2)], fill=(240, 240, 245, 255), width=3)
            d.line([(sx_c - 8, sy_c + 2), (sx_c + 8, sy_c - 4)], fill=(240, 240, 245, 255), width=3)

    # 2. 14-Layer 3D Extruded Chunky Back Depth of the 'B' Ribbon Body
    b_arr = body_rgba.copy()
    b_pil = Image.fromarray(b_arr, "RGBA")
    for depth_i in range(14, 0, -1):
        ox = bx0 + int(depth_i * 1.35)
        oy = by0 + int(depth_i * 0.85)
        dark_layer = b_arr.copy()
        shade = 0.38 + 0.032 * (14 - depth_i)
        dark_layer[:, :, 0] = (dark_layer[:, :, 0] * shade).astype(np.uint8)
        dark_layer[:, :, 1] = (dark_layer[:, :, 1] * 0.25).astype(np.uint8)
        dark_layer[:, :, 2] = (dark_layer[:, :, 2] * 0.25).astype(np.uint8)
        dl_pil = Image.fromarray(dark_layer, "RGBA")
        canvas.paste(dl_pil, (ox, oy), dl_pil)

    canvas.paste(b_pil, (bx0, by0), b_pil)

    # 3. Large White Oval 3D Eyes with Black Pupils & Expressive Red Eyelids
    eye_y = by0 + int(bh * 0.34)
    eye_l_x = bx0 + int(bw * 0.33)
    eye_r_x = bx0 + int(bw * 0.61)
    erx, ery = 38, 50

    for idx_e, (ex, ey, blink, brow_ang) in enumerate([
        (eye_l_x, eye_y, blink_l, brow_l),
        (eye_r_x, eye_y, blink_r, brow_r)
    ]):
        # Recessed rubber eye socket AO shadow
        d.ellipse([ex - erx - 7, ey - ery - 6, ex + erx + 7, ey + ery + 8], fill=(95, 0, 0, 210))
        d.ellipse([ex - erx - 3, ey - ery - 3, ex + erx + 3, ey + ery + 4], fill=(145, 4, 4, 255))
        # 3D shaded sclera (cool grey-blue outer rim -> bright white inner dome)
        d.ellipse([ex - erx, ey - ery, ex + erx, ey + ery], fill=(212, 218, 230, 255))
        d.ellipse([ex - erx + 4, ey - ery + 4, ex + erx - 4, ey + ery - 4], fill=(242, 245, 252, 255))
        d.ellipse([ex - erx + 8, ey - ery + 7, ex + erx - 10, ey + ery - 12], fill=(255, 255, 255, 255))

        px = ex + int(look_x * 12)
        py = ey + int(look_y * 10)
        prx, pry = 19, 25
        d.ellipse([px - prx, py - pry, px + prx, py + pry], fill=(10, 12, 16, 255))
        d.ellipse([px - 11, py - 15, px - 1, py - 5], fill=(255, 255, 255, 255))
        d.ellipse([px + 4, py + 6, px + 10, py + 12], fill=(255, 255, 255, 230))

        # Expressive Red Rubber Eyelid
        if blink > 0.02:
            lid_h = int(ery * 2.05 * min(1.0, blink))
            d.pieslice(
                [ex - erx - 2, ey - ery - 2, ex + erx + 2, ey + ery + 2],
                start=180, end=360 if blink < 0.55 else 540,
                fill=(192, 8, 8, 255)
            )
            lid_y = min(ey + ery, ey - ery + lid_h)
            d.line([(ex - erx, lid_y), (ex + erx, lid_y)], fill=(85, 0, 0, 255), width=5)

        # 3D Sculpted Rubber Eyebrow
        brow_y = ey - ery - 18 - int(brow_ang * 20.0)
        tilt_b = (-14 if idx_e == 0 else 14) * brow_ang
        d.line([(ex - 36, brow_y - tilt_b + 3), (ex + 36, brow_y + tilt_b + 3)], fill=(75, 0, 0, 220), width=16)
        d.line([(ex - 36, brow_y - tilt_b), (ex + 36, brow_y + tilt_b)], fill=(155, 4, 4, 255), width=14)
        d.line([(ex - 34, brow_y - tilt_b - 3), (ex + 34, brow_y + tilt_b - 3)], fill=(235, 24, 20, 255), width=6)

    # 4. Wide Flexible Cartoon Mouth with Visible Teeth & Tongue
    mx = bx0 + int(bw * 0.47)
    my = by0 + int(bh * 0.63)
    mw = int(74 * (0.75 + 0.45 * mouth_wide))
    mh = int(12 + 60 * mouth_open)

    if mouth_open > 0.045:
        # Soft AO shadow around lip rim
        d.rounded_rectangle(
            [mx - mw - 10, my - mh // 2 - 8, mx + mw + 10, my + mh + 10],
            radius=min(42, mh // 2 + 16),
            fill=(95, 0, 0, 220)
        )
        # 3D sculpted red rubber lip cushion
        d.rounded_rectangle(
            [mx - mw - 5, my - mh // 2 - 5, mx + mw + 5, my + mh + 5],
            radius=min(38, mh // 2 + 12),
            fill=(165, 4, 4, 255),
            outline=(238, 24, 20, 255),
            width=3
        )
        # Dark oral cavity interior
        d.rounded_rectangle(
            [mx - mw, my - mh // 2, mx + mw, my + mh],
            radius=min(32, mh // 2 + 8),
            fill=(32, 3, 6, 255)
        )
        # 3D Pink/Red Cartoon Tongue
        tg_w = int(mw * 0.72)
        tg_h = max(14, int(mh * 0.56))
        d.ellipse([mx - tg_w, my + mh - tg_h, mx + tg_w, my + mh + tg_h // 3], fill=(218, 58, 74, 255))
        d.ellipse([mx - int(tg_w * 0.65), my + mh - int(tg_h * 0.85), mx + int(tg_w * 0.65), my + mh], fill=(248, 108, 122, 255))

        # Visible White Upper Teeth Bar
        teeth_h = min(22, max(11, int(mh * 0.36)))
        tw_in = int(mw * 0.86)
        d.rounded_rectangle([mx - tw_in, my - mh // 2 + 2, mx + tw_in, my - mh // 2 + 2 + teeth_h], radius=6, fill=(246, 249, 255, 255))
        for tx_div in range(mx - tw_in + 26, mx + tw_in - 10, 26):
            d.line([(tx_div, my - mh // 2 + 2), (tx_div, my - mh // 2 + 2 + teeth_h)], fill=(195, 205, 220, 255), width=2)

        if mouth_open > 0.32:
            lt_w = int(mw * 0.65)
            d.rounded_rectangle([mx - lt_w, my + mh - 10, mx + lt_w, my + mh], radius=4, fill=(238, 242, 250, 255))
    else:
        # Confident Smug Grin with visible white teeth grin bar!
        grin_h = 24
        d.rounded_rectangle([mx - mw - 4, my - 6, mx + mw + 4, my + grin_h + 6], radius=16, fill=(105, 0, 0, 235))
        d.rounded_rectangle([mx - mw, my - 2, mx + mw, my + grin_h], radius=13, fill=(246, 249, 255, 255), outline=(45, 5, 8, 255), width=3)
        for tx_div in range(mx - mw + 26, mx + mw - 10, 26):
            d.line([(tx_div, my), (tx_div, my + grin_h - 2)], fill=(195, 205, 220, 255), width=2)

    # 5. Short Stubby Red Rubber Arms + White Cartoon Gloves + Two Wooden Drumsticks
    sh_l = (bx0 + 28, by0 + int(bh * 0.52))
    sh_r = (bx0 + bw - 22, by0 + int(bh * 0.52))

    if arm_pose == "stool_twirl":
        hand_l = (sh_l[0] - 72, sh_l[1] + 22 + int(math.sin(hit_phase * 4.0) * 14))
        hand_r = (sh_r[0] + 76, sh_r[1] - 18 + int(math.cos(hit_phase * 4.0) * 14))
        stick_l_ang = -35.0 + math.sin(hit_phase * 3.0) * 12.0
        stick_r_ang = (hit_phase * 360.0 * 2.2) % 360.0
    elif arm_pose == "laptop_shrug":
        hand_l = (sh_l[0] - 86, sh_l[1] - 36)
        hand_r = (sh_r[0] + 86, sh_r[1] - 36)
        stick_l_ang = -65.0
        stick_r_ang = 55.0
    elif arm_pose == "cymbal_smash":
        hand_l = (sh_l[0] - 62, sh_l[1] - 65)
        hand_r = (sh_r[0] + 95, sh_r[1] - 50 + int(hit_phase * 95))
        stick_l_ang = -45.0
        stick_r_ang = 25.0 + hit_phase * 55.0
    elif arm_pose == "phone_rimshot":
        hand_l = (sh_l[0] - 76, sh_l[1] + 12)
        hand_r = (sh_r[0] + 76, sh_r[1] + int(math.sin(hit_phase * math.pi) * 45))
        stick_l_ang = -25.0
        stick_r_ang = 12.0 + math.sin(hit_phase * math.pi) * 38.0
    elif arm_pose == "drum_fill":
        hand_l = (sh_l[0] - 58, sh_l[1] + 35 + int(math.sin(hit_phase * 28.0) * 38))
        hand_r = (sh_r[0] + 68, sh_r[1] + 35 + int(math.cos(hit_phase * 28.0) * 38))
        stick_l_ang = -15.0 + math.sin(hit_phase * 28.0) * 35.0
        stick_r_ang = 15.0 + math.cos(hit_phase * 28.0) * 35.0
    elif arm_pose == "count_fingers":
        hand_l = (sh_l[0] - 88, sh_l[1] - 48)
        hand_r = (sh_r[0] + 70, sh_r[1] + 28 + int(math.sin(hit_phase * math.pi) * 36))
        stick_l_ang = None
        stick_r_ang = 22.0 + math.sin(hit_phase * math.pi) * 32.0
    elif arm_pose == "point_down":
        hand_l = (sh_l[0] - 66, sh_l[1] + 55)
        hand_r = (sh_r[0] + 74, sh_r[1] + 15)
        stick_l_ang = None
        stick_r_ang = 35.0
    else:
        hand_l = (sh_l[0] - 66, sh_l[1] + 20)
        hand_r = (sh_r[0] + 70, sh_r[1] + 20)
        stick_l_ang = -30.0
        stick_r_ang = 30.0

    # 3D cylindrical shaded red rubber arms
    for sh_pt, hd_pt in [(sh_l, hand_l), (sh_r, hand_r)]:
        mid_x = (sh_pt[0] + hd_pt[0]) // 2
        mid_y = (sh_pt[1] + hd_pt[1]) // 2 + 18
        d.line([sh_pt, (mid_x, mid_y), hd_pt], fill=(95, 0, 0, 255), width=34)
        d.line([sh_pt, (mid_x, mid_y), hd_pt], fill=(175, 6, 6, 255), width=26)
        d.line([(sh_pt[0], sh_pt[1] - 4), (mid_x, mid_y - 4), (hd_pt[0], hd_pt[1] - 4)], fill=(228, 22, 18, 255), width=12)

    # Wooden Drumsticks & 3D White Cartoon Gloves
    for idx_h, (hd_pt, s_ang) in enumerate([(hand_l, stick_l_ang), (hand_r, stick_r_ang)]):
        hx, hy = hd_pt
        if s_ang is not None:
            rad = math.radians(s_ang)
            dir_mult = -1 if idx_h == 0 else 1
            sx1 = int(hx + math.cos(rad) * 130 * dir_mult)
            sy1 = int(hy + math.sin(rad) * 130)
            sx0 = int(hx - math.cos(rad) * 32 * dir_mult)
            sy0 = int(hy - math.sin(rad) * 32)
            d.line([(sx0, sy0), (sx1, sy1)], fill=(85, 52, 26, 255), width=14)
            d.line([(sx0, sy0), (sx1, sy1)], fill=(218, 175, 115, 255), width=10)
            d.line([(sx0, sy0 - 2), (sx1, sy1 - 2)], fill=(245, 212, 158, 255), width=4)
            d.ellipse([sx1 - 9, sy1 - 9, sx1 + 9, sy1 + 9], fill=(238, 198, 140, 255), outline=(105, 68, 34, 255), width=2)

        draw_3d_glove(d, hx, hy, arm_pose=arm_pose, is_left=(idx_h == 0), finger_count=finger_count)

    c_arr = np.array(canvas, dtype=np.uint8)
    target_w = max(32, int(cw * scale * squash_x))
    target_h = max(32, int(ch * scale * squash_y))
    scaled_c = cv2.resize(c_arr, (target_w, target_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    if abs(tilt_deg) > 0.2:
        M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), tilt_deg, 1.0)
        scaled_c = cv2.warpAffine(scaled_c, M, (target_w, target_h), flags=cv2.INTER_LINEAR)

    x0 = int(cx - target_w // 2)
    y0 = int(cy - target_h // 2)
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + target_w), min(H, y0 + target_h)
    if cx1 > cx0 and cy1 > cy0:
        sub_c = scaled_c[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(np.float32)
        sub_f = frame_rgb[cy0:cy1, cx0:cx1].astype(np.float32)
        a = sub_c[:, :, 3:4] / 255.0
        frame_rgb[cy0:cy1, cx0:cx1] = np.clip(sub_f * (1.0 - a) + sub_c[:, :, :3] * a, 0, 255).astype(np.uint8)

    return frame_rgb

def composite_foreground_drumkit(frame_rgb, bg_drumkit_orig, kick_pulse=0.0):
    """
    Composites the foreground Black Drum Kit (toms, snare, cymbals, and kick drum with 'brandrums' in red)
    in front of Boom's lower torso so Boom sits naturally AT the drum kit!
    """
    # Extract drumkit foreground region (y = 890..1265, x = 260..810) where luminance/contrast differs from dark wall
    dk_patch = bg_drumkit_orig[890:1265, 260:810].copy()
    ph, pw = dk_patch.shape[:2]
    yy, xx = np.mgrid[0:ph, 0:pw]
    # Smooth elliptical mask around the toms, snare, and kick drum
    mask_toms = np.exp(-(((xx - 230.0) / 145.0) ** 2 + ((yy - 110.0) / 85.0) ** 2))
    mask_kick = np.exp(-(((xx - 205.0) / 115.0) ** 2 + ((yy - 255.0) / 105.0) ** 2))
    mask_snare = np.exp(-(((xx - 395.0) / 95.0) ** 2 + ((yy - 165.0) / 80.0) ** 2))
    fg_mask = np.clip((mask_toms + mask_kick * 1.4 + mask_snare) * 1.15, 0.0, 1.0)[:, :, None]

    roi = frame_rgb[890:1265, 260:810].astype(np.float32)
    frame_rgb[890:1265, 260:810] = np.clip(roi * (1.0 - fg_mask) + dk_patch.astype(np.float32) * fg_mask, 0, 255).astype(np.uint8)

    # Stamp the resonant black kick drum head at exact coordinates (464, 1152) with radius 90
    cx, cy = 464, 1152
    r = int(88 * (1.0 + 0.06 * kick_pulse))
    pil_im = Image.fromarray(frame_rgb).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    d.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4], fill=(16, 18, 22, 245), outline=(140, 145, 155, 210), width=3)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(10, 11, 14, 252), outline=(213, 0, 0, int(120 + 135 * kick_pulse)), width=2)

    f_kick = get_font(800, int(27 * (1.0 + 0.05 * kick_pulse)))
    txt = "brandrums"
    bb = d.textbbox((0, 0), txt, font=f_kick)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    d.text((cx - tw // 2, cy - th // 2 - 4), txt, font=f_kick, fill=(213, 14, 10, 255))

    return np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)

def make_laptop_screen_ui(mode="dull_template", beat_step=0):
    sw, sh = 600, 380
    im = Image.new("RGB", (sw, sh), (175, 178, 182) if mode == "dull_template" else (12, 12, 15))
    d = ImageDraw.Draw(im)

    if mode == "dull_template":
        d.rectangle([0, 0, sw, 42], fill=(145, 148, 152))
        d.ellipse([16, 14, 28, 26], fill=(120, 122, 126))
        d.ellipse([36, 14, 48, 26], fill=(120, 122, 126))
        d.rectangle([80, 12, 240, 28], fill=(130, 132, 136))
        d.rectangle([36, 68, sw - 36, 195], fill=(152, 155, 160), outline=(128, 131, 136), width=2)
        d.line([(36, 68), (sw - 36, 195)], fill=(132, 135, 140), width=2)
        d.line([(36, 195), (sw - 36, 68)], fill=(132, 135, 140), width=2)
        for c_i in range(3):
            cx0 = 36 + c_i * 184
            d.rectangle([cx0, 218, cx0 + 160, 335], fill=(155, 158, 162))
            d.rectangle([cx0 + 14, 234, cx0 + 120, 250], fill=(132, 135, 140))
            d.rectangle([cx0 + 14, 262, cx0 + 142, 274], fill=(138, 141, 146))
    else:
        d.rectangle([0, 0, sw, 44], fill=(18, 18, 22))
        logo_col = (213, 0, 0) if beat_step >= 1 else (90, 10, 10)
        d.rounded_rectangle([24, 8, 62, 36], radius=8, fill=logo_col)
        d.rounded_rectangle([24, 60, 365, 225], radius=16, fill=(24, 10, 12), outline=(213, 0, 0), width=3)
        d.rounded_rectangle([44, 82, 125, 165], radius=18, fill=logo_col)
        d.rounded_rectangle([145, 86, 335, 112], radius=8, fill=(245, 245, 250))
        d.rounded_rectangle([145, 126, 295, 146], radius=6, fill=(213, 20, 20))
        d.rounded_rectangle([145, 164, 245, 198], radius=16, fill=(213, 0, 0))

        mob_col = (213, 0, 0) if beat_step >= 2 else (45, 45, 55)
        d.rounded_rectangle([390, 60, 574, 225], radius=18, fill=(18, 20, 26), outline=mob_col, width=3)
        d.rounded_rectangle([435, 76, 529, 208], radius=14, fill=(28, 12, 14), outline=(245, 245, 250), width=2)
        d.rounded_rectangle([450, 96, 514, 136], radius=6, fill=mob_col)

        form_col = (213, 0, 0) if beat_step >= 3 else (45, 45, 55)
        d.rounded_rectangle([24, 242, 285, 358], radius=14, fill=(18, 20, 26), outline=form_col, width=2)
        d.rounded_rectangle([42, 260, 220, 282], radius=6, fill=(38, 42, 52))
        d.rounded_rectangle([42, 292, 220, 314], radius=6, fill=(38, 42, 52))
        d.rounded_rectangle([42, 322, 155, 346], radius=10, fill=form_col)

        seo_col = (213, 0, 0) if beat_step >= 4 else (45, 45, 55)
        d.rounded_rectangle([305, 242, 574, 358], radius=14, fill=(18, 20, 26), outline=seo_col, width=2)
        for b_i, bh_val in enumerate([32, 52, 74, 92]):
            bx_b = 335 + b_i * 42
            by_b = 342 - bh_val
            d.rounded_rectangle([bx_b, by_b, bx_b + 26, 342], radius=5, fill=seo_col if beat_step >= 4 else (65, 65, 75))
        if beat_step >= 5:
            d.ellipse([505, 268, 555, 318], outline=(255, 255, 255), width=4)
            d.ellipse([515, 278, 545, 308], fill=(213, 0, 0))

    return np.array(im, dtype=np.uint8)

def warp_onto_laptop_behind_cymbal(frame_rgb, bg_laptop_orig, tex_rgb):
    """
    Warps tex_rgb onto the exact 1080x1920 laptop display quad [(235, 580), (780, 668), (722, 1185), (182, 1068)],
    and preserves the foreground brass crash cymbal at x=415..1080, y=715..985!
    """
    quad_pts = [(235, 580), (780, 668), (722, 1185), (182, 1068)]
    th, tw = tex_rgb.shape[:2]
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    dst = np.float32(quad_pts)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tex_rgb, M, (W, H), flags=cv2.INTER_LINEAR)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst), 255, lineType=cv2.LINE_AA)
    # Cut out the foreground brass crash cymbal ellipse so the cymbal stays in front of the laptop screen!
    cv2.ellipse(mask, (820, 845), (410, 122), 4, 0, 360, 0, -1, lineType=cv2.LINE_AA)
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (9, 9), 2.5)[:, :, None]

    out = frame_rgb.astype(np.float32) * (1.0 - mask_f) + warped.astype(np.float32) * mask_f
    return np.clip(out, 0, 255).astype(np.uint8)

def warp_onto_phone_screen(frame_rgb, tex_rgb):
    """Warps tex_rgb onto the exact 1080x1920 smartphone front display quad."""
    quad_pts = [(364, 612), (626, 618), (680, 1232), (418, 1248)]
    th, tw = tex_rgb.shape[:2]
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    dst = np.float32(quad_pts)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tex_rgb, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst), 255, lineType=cv2.LINE_AA)
    # Preserve top dynamic island notch
    cv2.rounded_rectangle if hasattr(cv2, "rounded_rectangle") else cv2.ellipse(mask, (482, 628), (38, 12), 2, 0, 360, 0, -1)
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (7, 7), 1.8)[:, :, None]
    out = frame_rgb.astype(np.float32) * (1.0 - mask_f) + warped.astype(np.float32) * mask_f
    return np.clip(out, 0, 255).astype(np.uint8)

def make_cracked_phone_ui(flaw_step=0):
    pw, ph = 360, 720
    im = Image.new("RGB", (pw, ph), (218, 215, 205))
    d = ImageDraw.Draw(im)

    c1 = (213, 0, 0) if flaw_step >= 1 else (140, 140, 140)
    d.rectangle([24, 48, 150, 108], fill=(235, 232, 225), outline=c1, width=4)
    d.line([(24, 48), (150, 108)], fill=c1, width=4)
    d.line([(24, 108), (150, 48)], fill=c1, width=4)

    c2 = (213, 0, 0) if flaw_step >= 2 else (130, 130, 130)
    d.rectangle([24, 135, pw - 24, 320], fill=(175, 182, 190), outline=c2, width=4)
    d.ellipse([56, 165, 96, 205], fill=(235, 200, 90))
    d.polygon([(40, 305), (135, 205), (220, 305)], fill=(120, 130, 140))
    d.polygon([(155, 305), (245, 190), (325, 305)], fill=(95, 105, 115))

    c3 = (213, 0, 0) if flaw_step >= 3 else (120, 120, 120)
    btn_pts = [(48, 375), (295, 402), (288, 462), (42, 435)]
    d.polygon(btn_pts, fill=(160, 155, 150), outline=c3)
    d.line(btn_pts + [btn_pts[0]], fill=c3, width=4)

    c4 = (213, 0, 0) if flaw_step >= 4 else (110, 110, 110)
    d.rectangle([36, 520, pw - 36, 568], fill=(240, 240, 240), outline=c4, width=4)
    for seg_i in range(3):
        sx = 46 + seg_i * 34
        d.rectangle([sx, 530, sx + 24, 558], fill=c4)

    crack_col = (40, 42, 48)
    d.line([(180, 330), (65, 150)], fill=crack_col, width=3)
    d.line([(180, 330), (310, 200)], fill=crack_col, width=3)
    d.line([(180, 330), (45, 530)], fill=crack_col, width=3)
    d.line([(180, 330), (320, 590)], fill=crack_col, width=3)
    d.line([(120, 240), (245, 255)], fill=crack_col, width=2)

    return np.array(im, dtype=np.uint8)

class BoomTikTokAdRenderer:
    def __init__(self):
        print("Initializing Pixar-style 3D 'Boom' TikTok Ad Renderer...")
        self.bg_drumkit = load_and_fit_9_16(os.path.join(ASSETS_DIR, "boom_bg_drumkit.png"))
        self.bg_laptop = load_and_fit_9_16(os.path.join(ASSETS_DIR, "boom_bg_laptop.png"))
        self.bg_phone = load_and_fit_9_16(os.path.join(ASSETS_DIR, "boom_bg_phone_front.png"))
        self.boom_body = build_boom_sss_ribbon_body(body_w=380, body_h=440)
        self.lipsync = np.load(os.path.join(ASSETS_DIR, "boom_lipsync_envelope.npy"))

    def apply_handheld_shake(self, frame_rgb, t, impact_boost=0.0):
        hx = math.sin(t * 5.3) * 4.5 + math.cos(t * 11.7) * 2.2
        hy = math.cos(t * 4.7) * 4.5 + math.sin(t * 9.3) * 2.2
        roll = math.sin(t * 3.9) * 0.28
        if impact_boost > 0.01:
            hx += math.sin(t * 65.0) * 18.0 * impact_boost
            hy += math.cos(t * 72.0) * 16.0 * impact_boost
            roll += math.sin(t * 45.0) * 1.4 * impact_boost
        zoom = 1.015 + 0.025 * impact_boost
        cx = W / 2.0 + hx
        cy = H / 2.0 + hy
        M = cv2.getRotationMatrix2D((cx, cy), roll, zoom)
        M[0, 2] += (W / 2.0) - cx
        M[1, 2] += (H / 2.0) - cy
        return cv2.warpAffine(frame_rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def render_frame(self, frame_idx):
        t = frame_idx / float(FPS)
        m_open = float(self.lipsync[frame_idx, 0])
        m_wide = float(self.lipsync[frame_idx, 1])

        blink = 0.0
        for bt in [2.3, 5.2, 9.6, 12.1, 15.3, 19.1, 22.2, 25.4]:
            if abs(t - bt) < 0.09:
                blink = 1.0 - abs(t - bt) / 0.09

        # =====================================================================
        # SHOT 1 (0 - 1s): Black frame -> loud snare hit at 0.22s -> Boom slams into frame from below, face fills screen!
        # =====================================================================
        if t < 1.0:
            if t < 0.20:
                return np.full((H, W, 3), 12, dtype=np.uint8)
            p = (t - 0.20) / 0.80
            ease = ease_out_back(min(1.0, p * 1.65))
            frame = cv2.GaussianBlur(self.bg_drumkit, (35, 35), 12.0)
            frame = (frame.astype(np.float32) * 0.45).astype(np.uint8)

            cy = int(1980 - (1980 - 960) * ease)
            sq_x = 1.0 + 0.22 * math.exp(-p * 6.0) * math.sin(p * 18.0)
            sq_y = 1.0 - 0.18 * math.exp(-p * 6.0) * math.sin(p * 18.0)
            impact = math.exp(-max(0.0, t - 0.22) * 12.0)

            frame = render_boom_character(
                frame, self.boom_body, cx=540, cy=cy, scale=2.05,
                squash_x=sq_x, squash_y=sq_y, tilt_deg=-4.0 * (1.0 - ease),
                mouth_open=0.45 * impact, mouth_wide=0.85, smile=0.6,
                look_x=0.0, look_y=-0.05, blink_l=0.0, blink_r=0.0,
                brow_l=-0.35, brow_r=-0.35, arm_pose="sticks_ready", show_legs=False
            )
            return self.apply_handheld_shake(frame, t, impact_boost=impact)

        # =====================================================================
        # SHOT 2 (1 - 4s): Medium shot at drum kit, Boom sits behind the kit on his stool, twirls drumstick:
        # "Your business is great. Your website? Nobody can hear it."
        # =====================================================================
        elif t < 4.0:
            local_t = t - 1.0
            pull = ease_out_cubic(min(1.0, local_t / 0.45))
            frame = self.bg_drumkit.copy()

            scale_b = 1.80 - 0.90 * pull
            cy_b = int(960 - (960 - 765) * pull + math.sin(local_t * 4.5) * 6)
            frame = render_boom_character(
                frame, self.boom_body, cx=495, cy=cy_b, scale=scale_b,
                squash_x=1.0, squash_y=1.0, tilt_deg=5.0 + math.sin(local_t * 2.5) * 2.5,
                mouth_open=m_open, mouth_wide=m_wide, smile=0.35,
                look_x=0.05, look_y=0.0, blink_l=blink, blink_r=blink,
                brow_l=0.35 if local_t > 1.5 else 0.1, brow_r=-0.25 if local_t > 1.5 else 0.1,
                arm_pose="stool_twirl", hit_phase=local_t, show_legs=False
            )
            # Composite foreground drumkit + "brandrums" red logo on kick drum in front of Boom's lower body!
            frame = composite_foreground_drumkit(frame, self.bg_drumkit, kick_pulse=0.0)
            return self.apply_handheld_shake(frame, t)

        # =====================================================================
        # SHOT 3 (4 - 8s): Boom hops onto laptop keyboard, points stick at dull grey template website,
        # shrugs, says: "Templates whisper. Brands make NOISE." On "NOISE" (6.82s) smashes cymbal & screen flares red!
        # =====================================================================
        elif t < 8.0:
            local_t = t - 4.0
            frame = self.bg_laptop.copy()

            is_noise_hit = t >= 6.80
            screen_tex = make_laptop_screen_ui("bold_brandrums" if is_noise_hit else "dull_template", beat_step=1)
            frame = warp_onto_laptop_behind_cymbal(frame, self.bg_laptop, screen_tex)

            hop_p = min(1.0, local_t / 0.40)
            hop_y = -int(math.sin(hop_p * math.pi) * 140)
            cx_b = 335
            cy_b = 1020 + hop_y  # Sneakers rest right on the physical laptop keyboard at y=1215!

            pose = "laptop_shrug" if (1.2 < local_t < 2.65) else ("cymbal_smash" if local_t >= 2.65 else "sticks_ready")
            smash_p = smoothstep(2.65, 2.85, local_t)
            cymbal_impact = math.exp(-max(0.0, t - 6.82) * 5.5) if t >= 6.80 else 0.0

            frame = render_boom_character(
                frame, self.boom_body, cx=cx_b, cy=cy_b, scale=0.80,
                squash_x=1.0 + 0.14 * cymbal_impact, squash_y=1.0 - 0.12 * cymbal_impact,
                tilt_deg=-6.0 * smash_p,
                mouth_open=max(m_open, 0.75 * cymbal_impact), mouth_wide=m_wide, smile=0.4,
                look_x=-0.12 if local_t < 1.2 else 0.0, look_y=0.0,
                blink_l=blink, blink_r=blink,
                brow_l=0.45 if local_t < 2.5 else -0.45, brow_r=0.45 if local_t < 2.5 else -0.45,
                arm_pose=pose, hit_phase=smash_p, show_legs=True
            )

            if cymbal_impact > 0.02:
                flare = np.zeros_like(frame)
                cv2.circle(flare, (760, 845), int(620 * (1.2 - 0.4 * cymbal_impact)), (235, 18, 12), -1, lineType=cv2.LINE_AA)
                flare = cv2.GaussianBlur(flare, (99, 99), 38.0)
                frame = cv2.addWeighted(frame, 1.0, flare, min(0.82, cymbal_impact * 0.9), 0)

            return self.apply_handheld_shake(frame, t, impact_boost=cymbal_impact)

        # =====================================================================
        # SHOT 4 (8 - 13s): Boom walks across desk to smartphone, taps it with stick,
        # cracked amateur website appears, rim-shot on each flaw:
        # "No logo. Stock photo. Button goes nowhere. Loads like it's 2009."
        # =====================================================================
        elif t < 13.0:
            local_t = t - 8.0
            frame = self.bg_phone.copy()

            flaw_step = sum(1 for rt in [9.12, 10.12, 11.28, 12.55] if t >= rt)
            rim_impact = max([math.exp(-abs(t - rt) * 18.0) if t >= rt else 0.0 for rt in [8.38, 9.12, 10.12, 11.28, 12.55]])

            if t >= 8.35:
                phone_tex = make_cracked_phone_ui(flaw_step=flaw_step)
                frame = warp_onto_phone_screen(frame, phone_tex)

            walk_p = min(1.0, local_t / 0.40)
            cx_b = int(130 + 115 * walk_p)
            cy_b = 1145 - int(abs(math.sin(local_t * 22.0)) * 26 * (1.0 - walk_p))  # Standing on the wooden desk beside phone!

            frame = render_boom_character(
                frame, self.boom_body, cx=cx_b, cy=cy_b, scale=0.78,
                squash_x=1.0 + 0.08 * rim_impact, squash_y=1.0 - 0.08 * rim_impact,
                tilt_deg=4.5,
                mouth_open=m_open, mouth_wide=m_wide, smile=0.2,
                look_x=0.18, look_y=-0.08, blink_l=blink, blink_r=blink,
                brow_l=-0.35, brow_r=0.45,
                arm_pose="phone_rimshot", hit_phase=rim_impact, show_legs=True
            )
            return self.apply_handheld_shake(frame, t, impact_boost=rim_impact * 0.65)

        # =====================================================================
        # SHOT 5 (13 - 18s): Rapid drum fill (13.0-13.48s), whip-pan, laptop screen flips to bold black-and-red website,
        # Boom lands a drum hit on each word:
        # "Custom logo. Mobile first. Forms that work. SEO baked in. Unlimited revisions."
        # =====================================================================
        elif t < 18.0:
            local_t = t - 13.0
            if local_t < 0.48:
                frame = self.bg_drumkit.copy()
                frame = render_boom_character(
                    frame, self.boom_body, cx=495, cy=765, scale=0.92,
                    squash_x=1.05, squash_y=0.95, tilt_deg=math.sin(local_t * 45.0) * 8.0,
                    mouth_open=0.35, mouth_wide=0.75, smile=0.6,
                    look_x=0.0, look_y=0.1, blink_l=0.0, blink_r=0.0,
                    brow_l=-0.4, brow_r=-0.4, arm_pose="drum_fill", hit_phase=local_t, show_legs=False
                )
                frame = composite_foreground_drumkit(frame, self.bg_drumkit, kick_pulse=abs(math.sin(local_t * 35.0)))
                if local_t > 0.34:
                    kernel = np.zeros((1, 45), dtype=np.float32)
                    kernel[0, :] = 1.0 / 45.0
                    frame = cv2.filter2D(frame, -1, kernel)
                return self.apply_handheld_shake(frame, t, impact_boost=0.5)

            frame = self.bg_laptop.copy()
            hit_times = [14.15, 14.95, 15.80, 16.65, 17.58]
            beat_step = sum(1 for ht in hit_times if t >= ht)
            hit_impact = max([math.exp(-abs(t - ht) * 16.0) if t >= ht else 0.0 for ht in hit_times])

            screen_tex = make_laptop_screen_ui("bold_brandrums", beat_step=max(1, beat_step))
            frame = warp_onto_laptop_behind_cymbal(frame, self.bg_laptop, screen_tex)

            frame = render_boom_character(
                frame, self.boom_body, cx=335, cy=1050, scale=0.80,
                squash_x=1.0 + 0.10 * hit_impact, squash_y=1.0 - 0.10 * hit_impact,
                tilt_deg=-4.0 + hit_impact * 6.0,
                mouth_open=m_open, mouth_wide=m_wide, smile=0.55,
                look_x=0.0, look_y=0.0, blink_l=blink, blink_r=blink,
                brow_l=-0.25, brow_r=-0.25,
                arm_pose="phone_rimshot", hit_phase=hit_impact, show_legs=True
            )
            return self.apply_handheld_shake(frame, t, impact_boost=hit_impact * 0.7)

        # =====================================================================
        # SHOT 6 (18 - 23s): Boom back at the kit, counts on 3 gloved fingers with a snare hit each:
        # "Starter, two ninety-nine. Essential, six forty-nine. Business, twelve ninety-nine. Pick one, we do the rest."
        # =====================================================================
        elif t < 23.0:
            snare_times = [18.28, 19.62, 20.98]
            f_count = sum(1 for st in snare_times if t >= st - 0.08)
            snare_impact = max([math.exp(-abs(t - st) * 16.0) if t >= st else 0.0 for st in snare_times])

            frame = self.bg_drumkit.copy()
            frame = render_boom_character(
                frame, self.boom_body, cx=495, cy=765, scale=0.92,
                squash_x=1.0 + 0.08 * snare_impact, squash_y=1.0 - 0.08 * snare_impact,
                tilt_deg=3.5,
                mouth_open=m_open, mouth_wide=m_wide, smile=0.5,
                look_x=0.0, look_y=0.0, blink_l=blink, blink_r=blink,
                brow_l=0.25, brow_r=0.25,
                arm_pose="count_fingers", finger_count=max(1, f_count), hit_phase=snare_impact, show_legs=False
            )
            frame = composite_foreground_drumkit(frame, self.bg_drumkit, kick_pulse=snare_impact * 0.6)
            return self.apply_handheld_shake(frame, t, impact_boost=snare_impact * 0.6)

        # =====================================================================
        # SHOT 7 (23 - 27s): Boom leans on laptop edge, raises an eyebrow:
        # "Logo, website, whole brand. Done in days, not months. Fifty-plus businesses already made noise with us."
        # =====================================================================
        elif t < 27.0:
            local_t = t - 23.0
            frame = self.bg_laptop.copy()
            screen_tex = make_laptop_screen_ui("bold_brandrums", beat_step=5)
            frame = warp_onto_laptop_behind_cymbal(frame, self.bg_laptop, screen_tex)

            brow_raise = smoothstep(0.15, 0.65, local_t)
            frame = render_boom_character(
                frame, self.boom_body, cx=330, cy=1050, scale=0.84,
                squash_x=1.0, squash_y=1.0, tilt_deg=-7.5,
                mouth_open=m_open, mouth_wide=m_wide, smile=0.55,
                look_x=0.05, look_y=0.0,
                blink_l=blink, blink_r=blink * 0.3,
                brow_l=-0.15, brow_r=0.85 * brow_raise,
                arm_pose="stool_twirl", hit_phase=local_t * 0.4, show_legs=True
            )
            return self.apply_handheld_shake(frame, t)

        # =====================================================================
        # SHOT 8 (27 - 30s): Drum roll (27.0-27.38s) into cymbal crash (27.38s),
        # Boom leans in, points down toward bottom third of frame, winks, hits kick drum once (28.65s):
        # "Free brand audit in 24 hours. Brandrums. Make some noise."
        # Freeze on a smug pose (29.45-30.0s), bottom 25% of frame kept empty for CTA overlay!
        # =====================================================================
        else:
            is_freeze = t >= 29.45
            eff_t = 29.45 if is_freeze else t
            eff_local = eff_t - 27.0

            kick_hit = math.exp(-abs(eff_t - 28.65) * 14.0) if eff_t >= 28.62 else 0.0
            cym_hit = math.exp(-max(0.0, eff_t - 27.38) * 6.0) if eff_t >= 27.38 else 0.0

            frame = self.bg_drumkit.copy()

            if eff_local < 0.38:
                pose = "drum_fill"
                wink_r = 0.0
                scale_b = 0.92
                cy_b = 765
            else:
                lean = smoothstep(0.38, 0.85, eff_local)
                pose = "point_down"
                wink_r = smoothstep(0.65, 0.90, eff_local) if eff_local < 1.35 else (0.38 if is_freeze else 0.35)
                scale_b = 0.92 + 0.14 * lean
                cy_b = int(765 - 25 * lean)

            frame = render_boom_character(
                frame, self.boom_body, cx=495, cy=cy_b, scale=scale_b,
                squash_x=1.0 + 0.09 * kick_hit, squash_y=1.0 - 0.09 * kick_hit,
                tilt_deg=-4.5,
                mouth_open=0.0 if is_freeze else m_open,
                mouth_wide=0.75 if is_freeze else m_wide,
                smile=0.85 if is_freeze else 0.55,
                look_x=0.0, look_y=0.08 if is_freeze else 0.0,
                blink_l=0.32 if is_freeze else 0.0,
                blink_r=wink_r,
                brow_l=-0.25 if is_freeze else -0.15,
                brow_r=0.75 if is_freeze else 0.45,
                arm_pose=pose, hit_phase=eff_local, show_legs=False
            )
            frame = composite_foreground_drumkit(frame, self.bg_drumkit, kick_pulse=kick_hit)

            # Keep bottom 25% of frame (y = 1440..1920) clean/darkened for TikTok CTA overlay
            yy = np.linspace(0.0, 1.0, H - 1400)[:, None, None]
            frame[1400:, :, :] = (frame[1400:, :, :].astype(np.float32) * (1.0 - 0.82 * yy)).astype(np.uint8)

            return frame if is_freeze else self.apply_handheld_shake(frame, t, impact_boost=max(cym_hit * 0.6, kick_hit * 0.8))

def main():
    renderer = BoomTikTokAdRenderer()
    if "--preview" in sys.argv:
        os.makedirs("/tmp/boom_previews", exist_ok=True)
        for tt in [0.55, 2.5, 6.9, 10.5, 15.5, 20.5, 25.0, 29.6]:
            rgb = renderer.render_frame(int(tt * FPS))
            out_p = f"/tmp/boom_previews/preview_{tt:.2f}s.jpg"
            Image.fromarray(rgb).save(out_p, quality=92)
            print("Saved preview:", out_p)
        return

    out_video = "/home/user/video/brandrums_boom_tiktok_ad.mp4"
    audio_path = os.path.join(ASSETS_DIR, "boom_soundtrack_master.wav")
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
