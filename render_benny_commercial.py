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
DURATION = 60.0
TOTAL_FRAMES = int(FPS * DURATION)

ASSETS_DIR = "/home/user/video/assets"
FONT_DIR = os.path.join(ASSETS_DIR, "fonts")

def get_font(weight=700, size=48):
    path = os.path.join(FONT_DIR, f"inter-latin-{weight}-normal.woff")
    if not os.path.exists(path):
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(path, size)

def smoothstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0 + 1e-9)))
    return t * t * (3.0 - 2.0 * t)

def load_fit(path):
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    r = W / H
    if (iw / ih) > r:
        nw = int(ih * r)
        im = im.crop(((iw - nw) // 2, 0, (iw + nw) // 2, ih))
    else:
        nh = int(iw / r)
        im = im.crop((0, (ih - nh) // 2, iw, (ih + nh) // 2))
    return np.array(im.resize((W, H), Image.Resampling.LANCZOS), dtype=np.uint8)

def extract_benny_cutout(master_rgb):
    """
    Extracts Mr. Benny from benny_master_ref.png using exterior flood-fill + morphological hole closing
    so his tweed suit, white shirt, hair, and face are 100% solid with zero interior holes!
    """
    bg_col = np.median(master_rgb[:100, :100].reshape(-1, 3), axis=0)
    diff = np.linalg.norm(master_rgb.astype(np.float32) - bg_col[None, None, :], axis=2)
    is_bg = (diff < 22.0).astype(np.uint8) * 255

    # Flood fill exterior background from top corners & left/right borders
    ff_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
    ext_bg = is_bg.copy()
    for seed_pt in [(2, 2), (W - 3, 2), (2, H // 2), (W - 3, H // 2), (2, H - 3), (W - 3, H - 3)]:
        if ext_bg[seed_pt[1], seed_pt[0]] == 255:
            cv2.floodFill(ext_bg, ff_mask, seed_pt, 128)
    fg = (ext_bg != 128).astype(np.uint8) * 255

    # Also clear the tiny background gaps between his upper arms and torso (around x=318, y=1320 and x=765, y=1320)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg_blur = cv2.GaussianBlur(fg.astype(np.float32) / 255.0, (9, 9), 2.0)

    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[:, :, :3] = master_rgb
    out[:, :, 3] = np.clip(fg_blur * 255.0, 0, 255).astype(np.uint8)
    return out

def extract_real_3d_hand_patch(director_rgb):
    """
    Extracts Mr. Benny's real 3D Pixar raised hand & tweed sleeve from benny_pose_director.png
    (x = 180..440, y = 590..1120) with a soft feathered alpha mask for natural 3D hand animations.
    """
    patch = director_rgb[590:1120, 180:440].copy()
    ph, pw = patch.shape[:2]
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    # Smooth capsule mask around his 3D hand, fingers, wrist cuff, and tweed sleeve
    hand_m = np.exp(-(((xx - 125.0) / 88.0) ** 2 + ((yy - 125.0) / 95.0) ** 2) ** 1.6)
    arm_m = np.exp(-(((xx - 145.0 - (yy - 180.0) * 0.22) / 68.0) ** 2 + ((yy - 320.0) / 175.0) ** 2) ** 1.6)
    alpha = np.clip(np.maximum(hand_m, arm_m), 0.0, 1.0)
    rgba = np.zeros((ph, pw, 4), dtype=np.uint8)
    rgba[:, :, :3] = patch
    rgba[:, :, 3] = (alpha * 255.0).astype(np.uint8)
    return rgba

class MeshRigDeformer:
    """
    2D/3D Mesh-Warp Facial & Body Rig (cv2.remap) for Mr. Benny.
    Animates head tilt/lean/nod, subtle breathing, eyebrow raises, eye gaze shifts,
    natural blinking, nervous throat swallow, and 3D lip-synced jaw/mouth movements!
    """
    def __init__(self, open_mouth_ref=None):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.base_x = xx
        self.base_y = yy
        self.open_mouth_ref = open_mouth_ref
        self.mouth_blend_mask = np.exp(-(((xx - 532.0) / 64.0) ** 2 + ((yy - 622.0) / 24.0) ** 2) ** 1.5)[:, :, None]

    def deform(
        self, img_rgb, layout="mcu",
        t=0.0,
        head_tilt_deg=0.0, head_lean_x=0.0, head_nod_y=0.0,
        brow_l_dy=0.0, brow_r_dy=0.0,
        gaze_dx=0.0, gaze_dy=0.0,
        blink=0.0,
        mouth_open=0.0, mouth_wide=0.5, smile_dy=0.0,
        gulp=0.0,
        glasses_wear=0.0
    ):
        if layout == "thumbsup":
            h_cx, h_cy, h_rx, h_ry = 732.0, 625.0, 185.0, 240.0
            el_x, el_y, er_x, er_y, e_r = 695.0, 618.0, 770.0, 618.0, 22.0
            bl_x, bl_y, br_x, br_y, b_r = 692.0, 584.0, 772.0, 584.0, 42.0
            m_cx, m_cy, m_rx, m_ry = 730.0, 725.0, 52.0, 26.0
            th_x, th_y = 730.0, 815.0
        elif layout == "glasses":
            h_cx, h_cy, h_rx, h_ry = 595.0, 620.0, 240.0, 310.0
            el_x, el_y, er_x, er_y, e_r = 536.0, 612.0, 628.0, 616.0, 32.0
            bl_x, bl_y, br_x, br_y, b_r = 530.0, 558.0, 632.0, 560.0, 62.0
            m_cx, m_cy, m_rx, m_ry = 575.0, 795.0, 58.0, 32.0
            th_x, th_y = 595.0, 910.0
        else:
            # Standard Medium Close-Up ("shock", "pencil", "exec", "master")
            h_cx, h_cy, h_rx, h_ry = 520.0, 460.0, 290.0, 360.0
            el_x, el_y, er_x, er_y, e_r = 442.0, 412.0, 578.0, 412.0, 38.0
            bl_x, bl_y, br_x, br_y, b_r = 438.0, 355.0, 582.0, 355.0, 75.0
            m_cx, m_cy = 532.0, 622.0
            m_rx, m_ry = 66.0, 28.0
            th_x, th_y = 520.0, 795.0

        if mouth_open > 0.03 and self.open_mouth_ref is not None and layout in ("pencil", "exec", "mcu"):
            w_m = min(0.92, mouth_open * 1.15) * self.mouth_blend_mask
            img_rgb = np.clip(img_rgb.astype(np.float32) * (1.0 - w_m) + self.open_mouth_ref.astype(np.float32) * w_m, 0, 255).astype(np.uint8)

        gh, gw = H // 4, W // 4
        gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float32) * 4.0
        dx = np.zeros((gh, gw), dtype=np.float32)
        dy = np.zeros((gh, gw), dtype=np.float32)

        # 1. Subtle continuous breathing in torso/shoulders
        breath = math.sin(t * 2.5) * 4.5
        torso_w = np.exp(-(((gx - h_cx) / (h_rx * 1.6)) ** 2 + ((gy - (h_cy + h_ry * 1.25)) / 240.0) ** 2))
        dy -= breath * torso_w

        # 2. Head tilt, horizontal lean & vertical nod
        if abs(head_tilt_deg) > 0.05 or abs(head_lean_x) > 0.2 or abs(head_nod_y) > 0.2:
            head_w = np.exp(-(((gx - h_cx) / (h_rx * 1.15)) ** 2 + ((gy - h_cy) / (h_ry * 1.15)) ** 2) ** 1.5)
            rad = math.radians(head_tilt_deg)
            rx = gx - h_cx
            ry = gy - (h_cy + h_ry * 0.65)
            rot_dx = rx * (math.cos(rad) - 1.0) - ry * math.sin(rad)
            rot_dy = rx * math.sin(rad) + ry * (math.cos(rad) - 1.0)
            dx -= (rot_dx + head_lean_x) * head_w
            dy -= (rot_dy + head_nod_y) * head_w

        # 3. Expressive Left & Right Eyebrow Raises
        if abs(brow_l_dy) > 0.2:
            bw_l = np.exp(-(((gx - bl_x) / b_r) ** 2 + ((gy - bl_y) / (b_r * 0.55)) ** 2))
            dy -= brow_l_dy * bw_l
        if abs(brow_r_dy) > 0.2:
            bw_r = np.exp(-(((gx - br_x) / b_r) ** 2 + ((gy - br_y) / (b_r * 0.55)) ** 2))
            dy -= brow_r_dy * bw_r

        # 4. Eye Gaze Shifts
        if abs(gaze_dx) > 0.2 or abs(gaze_dy) > 0.2:
            for ex, ey in [(el_x, el_y), (er_x, er_y)]:
                ew = np.exp(-(((gx - ex) / (e_r * 0.85)) ** 2 + ((gy - ey) / (e_r * 0.75)) ** 2) ** 1.5)
                dx -= gaze_dx * ew
                dy -= gaze_dy * ew

        # 5. Nervous Throat Swallow (gulp)
        if abs(gulp) > 0.02:
            tw = np.exp(-(((gx - th_x) / 68.0) ** 2 + ((gy - th_y) / 65.0) ** 2))
            dy -= math.sin(gulp * math.pi * 2.0) * 22.0 * tw

        # 6. Jaw & Mouth Smile/Frown Corner Warp + Lip-Sync Jaw Parting
        if abs(smile_dy) > 0.2:
            for m_side in [-1.0, 1.0]:
                cx_m = m_cx + m_side * m_rx * 0.92
                cw_m = np.exp(-(((gx - cx_m) / (m_rx * 0.65)) ** 2 + ((gy - m_cy) / (m_ry * 1.1)) ** 2))
                dy -= smile_dy * cw_m

        if mouth_open > 0.02:
            # Part lower lip & chin downward and upper lip slightly upward along the real 3D lip seam!
            jaw_w = np.exp(-(((gx - m_cx) / (m_rx * 1.35)) ** 2 + ((gy - (m_cy + m_ry * 1.1)) / (m_ry * 2.2)) ** 2))
            dy -= (mouth_open * 30.0) * jaw_w
            lip_up = np.exp(-(((gx - m_cx) / (m_rx * 1.0)) ** 2 + ((gy - (m_cy - m_ry * 0.35)) / (m_ry * 0.75)) ** 2))
            dy += (mouth_open * 8.0) * lip_up

        full_dx = cv2.resize(dx, (W, H), interpolation=cv2.INTER_LINEAR)
        full_dy = cv2.resize(dy, (W, H), interpolation=cv2.INTER_LINEAR)
        warped = cv2.remap(img_rgb, self.base_x + full_dx, self.base_y + full_dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        need_overlay = (blink > 0.04) or (glasses_wear > 0.05)
        if not need_overlay:
            return warped

        pil_im = Image.fromarray(warped).convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)

        eff_hx = head_lean_x
        eff_hy = head_nod_y

        if blink > 0.04:
            for ex, ey in [(el_x + eff_hx, el_y + eff_hy), (er_x + eff_hx, er_y + eff_hy)]:
                rx_e = int(e_r * 1.18)
                ry_e = int(e_r * 0.92)
                lid_drop = int(ry_e * 2.0 * min(1.0, blink))
                d.pieslice(
                    [int(ex - rx_e), int(ey - ry_e), int(ex + rx_e), int(ey + ry_e)],
                    start=180, end=360 if blink < 0.55 else 540,
                    fill=(216, 168, 142, 248)
                )
                lid_y = min(int(ey + ry_e), int(ey - ry_e + lid_drop))
                d.line([(int(ex - rx_e), lid_y), (int(ex + rx_e), lid_y)], fill=(78, 48, 38, 235), width=4)

        if glasses_wear > 0.05:
            alpha_g = int(255 * min(1.0, glasses_wear))
            y_off = int((1.0 - min(1.0, glasses_wear)) * 180.0)
            g_lx, g_ly = int(el_x + eff_hx), int(el_y + eff_hy + y_off)
            g_rx, g_ry = int(er_x + eff_hx), int(er_y + eff_hy + y_off)
            gr = int(e_r * 1.48)
            d.ellipse([g_lx - gr, g_ly - gr, g_lx + gr, g_ly + gr], fill=(255, 255, 255, int(alpha_g * 0.08)), outline=(60, 52, 44, alpha_g), width=4)
            d.ellipse([g_rx - gr, g_ry - gr, g_rx + gr, g_ry + gr], fill=(255, 255, 255, int(alpha_g * 0.08)), outline=(60, 52, 44, alpha_g), width=4)
            d.line([(g_lx + gr, g_ly), (g_rx - gr, g_ry)], fill=(60, 52, 44, alpha_g), width=4)

        return np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)

def warp_onto_office_laptop(frame_rgb, screen_rgb):
    quad = np.float32([[468, 1068], [875, 1142], [788, 1510], [406, 1362]])
    sh, sw = screen_rgb.shape[:2]
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    M = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(screen_rgb, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(quad), 255, lineType=cv2.LINE_AA)
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (7, 7), 1.8)[:, :, None]
    return np.clip(frame_rgb.astype(np.float32) * (1.0 - mask_f) + warped.astype(np.float32) * mask_f, 0, 255).astype(np.uint8)

def make_office_laptop_ui(mode="bad_mockup", t=0.0, brandrums_logo_rgba=None):
    sw, sh = 640, 420
    im = Image.new("RGB", (sw, sh), (22, 24, 28))
    d = ImageDraw.Draw(im)

    if mode == "blank":
        d.rectangle([0, 0, sw, sh], fill=(16, 18, 22))
    elif mode == "bad_mockup":
        d.rectangle([0, 0, sw, sh], fill=(195, 255, 0))
        d.rectangle([0, 0, sw, 58], fill=(255, 0, 180))
        f_bad = get_font(800, 26)
        d.text((18, 14), "BENNY'S SUPER WEB SITE 1998!!", font=f_bad, fill=(255, 255, 0))
        d.polygon([(22, 78), (290, 64), (275, 210), (35, 225)], fill=(0, 210, 255), outline=(255, 0, 0), width=5)
        d.text((48, 120), " nav?? ", font=f_bad, fill=(255, 0, 120))
        d.polygon([(315, 92), (618, 70), (595, 255), (300, 230)], fill=(255, 135, 0), outline=(0, 40, 255), width=5)
        d.ellipse([80, 235, 250, 395], fill=(165, 40, 255), outline=(255, 255, 0), width=4)
        btn_col = (255, 30, 30) if int(t * 6) % 2 == 0 else (30, 220, 60)
        d.rounded_rectangle([310, 280, 605, 385], radius=18, fill=btn_col, outline=(0, 0, 0), width=5)
        d.text((352, 316), "PUBLISH NOW!", font=get_font(800, 28), fill=(255, 255, 255))
        if t >= 4.0:
            cp = smoothstep(4.0, 4.8, t)
            cx_cur = int(120 + (460 - 120) * cp)
            cy_cur = int(140 + (335 - 140) * cp)
            d.polygon([(cx_cur, cy_cur), (cx_cur, cy_cur + 34), (cx_cur + 12, cy_cur + 24), (cx_cur + 24, cy_cur + 24)], fill=(255, 255, 255), outline=(0, 0, 0))

    elif mode == "giant_error":
        d.rectangle([0, 0, sw, sh], fill=(175, 8, 12))
        d.rectangle([28, 28, sw - 28, sh - 28], fill=(32, 6, 8), outline=(255, 60, 60), width=6)
        d.rectangle([28, 28, sw - 28, 92], fill=(225, 18, 22))
        d.text((52, 44), "CRITICAL DESIGN ERROR 500", font=get_font(800, 30), fill=(255, 255, 255))
        d.polygon([(320, 118), (245, 245), (395, 245)], fill=(255, 210, 0), outline=(255, 255, 255), width=4)
        d.text((310, 158), "!", font=get_font(800, 56), fill=(20, 20, 20))
        d.text((78, 272), "LAYOUT REJECTED BY INTERNET", font=get_font(700, 28), fill=(255, 190, 190))
        d.text((145, 318), " System Confidence: 0.00% ", font=get_font(600, 22), fill=(255, 110, 110))

    elif mode == "search_brandrums":
        if t < 19.25:
            d.rectangle([0, 0, sw, sh], fill=(242, 244, 248))
            d.rounded_rectangle([55, 145, sw - 55, 225], radius=36, fill=(255, 255, 255), outline=(213, 0, 0), width=4)
            raw_str = "asdfghjkl;!!@#help_me"
            if t < 18.10:
                n_ch = int(smoothstep(17.08, 18.05, t) * len(raw_str))
                typed = raw_str[:n_ch]
            elif t < 18.50:
                typed = raw_str
            elif t < 19.00:
                n_ch = int((1.0 - smoothstep(18.50, 18.98, t)) * len(raw_str))
                typed = raw_str[:n_ch]
            else:
                b_str = "brandrums.com"
                n_ch = int(smoothstep(19.00, 19.22, t) * len(b_str))
                typed = b_str[:n_ch]
            d.text((88, 168), typed + "|", font=get_font(700, 30), fill=(20, 22, 28))
            d.text((185, 82), "SEARCH FOR HELP", font=get_font(800, 28), fill=(90, 96, 110))
        else:
            d.rectangle([0, 0, sw, sh], fill=(12, 13, 18))
            d.rectangle([0, 0, sw, 56], fill=(18, 20, 26))
            if brandrums_logo_rgba is not None:
                im_rgba = im.convert("RGBA")
                lg = Image.fromarray(brandrums_logo_rgba).resize((165, 50), Image.Resampling.LANCZOS)
                im_rgba.paste(lg, (24, 6), lg)
                im = im_rgba.convert("RGB")
                d = ImageDraw.Draw(im)
            d.rounded_rectangle([28, 76, sw - 28, 265], radius=18, fill=(24, 12, 16), outline=(213, 14, 10), width=3)
            d.text((54, 102), "BRANDS THAT MAKE NOISE.", font=get_font(800, 30), fill=(255, 255, 255))
            d.text((54, 148), "Websites • Apps • Branding • AI • SEO", font=get_font(600, 21), fill=(220, 190, 190))
            d.rounded_rectangle([54, 192, 295, 242], radius=24, fill=(213, 14, 10))
            d.text((82, 204), "ENTER PORTAL →", font=get_font(800, 21), fill=(255, 255, 255))
            for c_i in range(3):
                cx0 = 28 + c_i * 198
                d.rounded_rectangle([cx0, 284, cx0 + 184, 396], radius=12, fill=(22, 25, 34), outline=(65, 70, 88), width=2)

    return np.array(im, dtype=np.uint8)

def render_camera_move(frame_rgb, zoom=1.0, pan_x=0.0, pan_y=0.0, roll_deg=0.0):
    if abs(zoom - 1.0) < 0.002 and abs(pan_x) < 0.5 and abs(pan_y) < 0.5 and abs(roll_deg) < 0.05:
        return frame_rgb
    cx = W / 2.0 + pan_x
    cy = H / 2.0 + pan_y
    M = cv2.getRotationMatrix2D((cx, cy), roll_deg, zoom)
    M[0, 2] += (W / 2.0) - cx
    M[1, 2] += (H / 2.0) - cy
    return cv2.warpAffine(frame_rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

class BennyComedyCommercialRenderer:
    def __init__(self):
        print("Initializing 60s 9:16 3D Comedy Commercial: 'THE WORST WEBSITE DESIGNER EVER!'...")
        self.bg_office = load_fit(os.path.join(ASSETS_DIR, "benny_bg_office.png"))
        self.bg_studio = load_fit(os.path.join(ASSETS_DIR, "benny_bg_studio.png"))
        self.master_ref = load_fit(os.path.join(ASSETS_DIR, "benny_master_ref.png"))
        self.benny_cutout = extract_benny_cutout(self.master_ref)

        self.pose_thumbsup = load_fit(os.path.join(ASSETS_DIR, "benny_pose_thumbsup.png"))
        self.pose_shock = load_fit(os.path.join(ASSETS_DIR, "benny_pose_shock.png"))
        self.pose_pencil = load_fit(os.path.join(ASSETS_DIR, "benny_pose_pencil.png"))
        self.pose_portal = load_fit(os.path.join(ASSETS_DIR, "benny_pose_portal.png"))
        self.pose_director = load_fit(os.path.join(ASSETS_DIR, "benny_pose_director.png"))
        self.pose_glasses = load_fit(os.path.join(ASSETS_DIR, "benny_pose_glasses.png"))
        self.pose_exec = load_fit(os.path.join(ASSETS_DIR, "benny_pose_exec.png"))
        self.real_hand_rgba = extract_real_3d_hand_patch(self.pose_director)

        raw_logo = Image.open(os.path.join(ASSETS_DIR, "brandrums_official_logo.png")).convert("RGBA")
        self.logo_rgba = np.array(raw_logo, dtype=np.uint8)
        self.lipsync = np.load(os.path.join(ASSETS_DIR, "benny_lipsync_envelope.npy"))
        self.rig = MeshRigDeformer(open_mouth_ref=self.master_ref)

    def get_natural_blink(self, t):
        blink_times = [
            1.6, 3.6, 6.15, 6.36,
            9.2, 11.9, 13.8,
            18.2, 20.3, 23.1,
            27.5, 30.4, 34.9, 38.6,
            42.2, 46.1, 49.5, 52.4, 58.4
        ]
        for bt in blink_times:
            if abs(t - bt) < 0.085:
                return 1.0 - abs(t - bt) / 0.085
        return 0.0

    def render_frame(self, idx):
        t = idx / float(FPS)
        m_open = float(self.lipsync[idx, 0])
        m_wide = float(self.lipsync[idx, 1])
        blink = self.get_natural_blink(t)

        # =====================================================================
        # SCENE 1: THE DISASTER (0 - 8s)
        # =====================================================================
        if t < 8.0:
            if t < 5.0:
                frame = self.pose_thumbsup.copy()
                ui_mode = "blank" if t < 0.85 else "bad_mockup"
                scr = make_office_laptop_ui(ui_mode, t=t, brandrums_logo_rgba=self.logo_rgba)
                frame = warp_onto_office_laptop(frame, scr)

                tilt = 0.0
                lean_x = 0.0
                if 2.2 <= t < 3.85:
                    tilt = math.sin((t - 2.2) / 1.65 * math.pi * 2.0) * 12.5
                    lean_x = math.sin((t - 2.2) / 1.65 * math.pi * 2.0) * 24.0

                nod_y = -10.0 * math.sin((t - 3.9) * math.pi * 2.0) if 3.9 <= t < 4.8 else 0.0
                frame = self.rig.deform(
                    frame, layout="thumbsup", t=t,
                    head_tilt_deg=tilt, head_lean_x=lean_x, head_nod_y=nod_y,
                    brow_l_dy=-8.0 if t > 4.0 else 0.0, brow_r_dy=-8.0 if t > 4.0 else 0.0,
                    gaze_dx=-6.0 if t < 4.2 else 0.0, gaze_dy=6.0 if t < 4.2 else 0.0,
                    blink=blink, smile_dy=-6.0
                )

                z = 1.48 - 0.44 * smoothstep(0.5, 1.35, t)
                px = 140.0 * (1.0 - smoothstep(0.5, 1.35, t))
                py = -160.0 * (1.0 - smoothstep(0.5, 1.35, t))
                return render_camera_move(frame, zoom=z, pan_x=px, pan_y=py)

            else:
                frame = self.pose_shock.copy()
                scr = make_office_laptop_ui("giant_error", t=t)
                frame = warp_onto_office_laptop(frame, scr)

                gaze_y = 14.0 * (1.0 - smoothstep(5.40, 6.05, t))
                collapse = smoothstep(5.3, 6.4, t)
                tiny_smile = smoothstep(7.25, 7.65, t)
                smile_val = 14.0 * collapse - 10.0 * tiny_smile

                gulp_p = (t - 6.75) / 0.35 if 6.75 <= t <= 7.10 else 0.0
                shiver_x = math.sin(t * 55.0) * 4.0 * math.exp(-max(0.0, t - 5.0) * 4.5)

                frame = self.rig.deform(
                    frame, layout="mcu", t=t,
                    head_tilt_deg=0.0, head_lean_x=shiver_x, head_nod_y=0.0,
                    brow_l_dy=-12.0 * (1.0 - 0.5 * collapse), brow_r_dy=-12.0 * (1.0 - 0.5 * collapse),
                    gaze_dx=0.0, gaze_dy=gaze_y,
                    blink=blink, smile_dy=smile_val, gulp=gulp_p
                )

                push = smoothstep(5.0, 7.95, t)
                return render_camera_move(frame, zoom=1.02 + 0.24 * push, pan_x=0.0, pan_y=-165.0 * push)

        # =====================================================================
        # SCENE 2: BENNY BECOMES A DESIGN GENIUS (8 - 17s)
        # =====================================================================
        elif t < 17.0:
            if t < 11.55:
                frame = self.pose_thumbsup.copy()
                scr = make_office_laptop_ui("giant_error", t=t)
                frame = warp_onto_office_laptop(frame, scr)

                nod = math.sin((t - 10.2) * 11.0) * 16.0 if t >= 10.2 else math.sin(t * 8.0) * 5.0
                frame = self.rig.deform(
                    frame, layout="thumbsup", t=t,
                    head_tilt_deg=4.0 * math.sin(t * 3.0), head_nod_y=nod,
                    brow_l_dy=-10.0, brow_r_dy=-10.0, blink=blink, smile_dy=-8.0
                )

                pil_im = Image.fromarray(frame).convert("RGBA")
                ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                d = ImageDraw.Draw(ov)

                compare_p = smoothstep(10.0, 10.45, t)
                px_c = int(310 + (625 - 310) * compare_p)
                py_c = int(1240 + (1260 - 1240) * compare_p + (nod * 0.6 if t >= 10.2 else 0))
                pw_c, ph_c = 380, 280

                d.rounded_rectangle([px_c - pw_c // 2 + 12, py_c - ph_c // 2 + 16, px_c + pw_c // 2 + 12, py_c + ph_c // 2 + 16], radius=10, fill=(0, 0, 0, 125))
                d.rounded_rectangle([px_c - pw_c // 2, py_c - ph_c // 2, px_c + pw_c // 2, py_c + ph_c // 2], radius=8, fill=(252, 248, 236, 255), outline=(185, 175, 155, 255), width=3)

                rect_pts = [
                    (px_c - 145, py_c - 95),
                    (px_c + 155, py_c - 112),
                    (px_c + 138, py_c + 88),
                    (px_c - 160, py_c + 102),
                    (px_c - 145, py_c - 95)
                ]
                r_prog = smoothstep(8.05, 8.85, t)
                n_seg = int(r_prog * 4)
                for s_i in range(n_seg):
                    d.line([rect_pts[s_i], rect_pts[s_i + 1]], fill=(45, 48, 55, 255), width=6)

                circles = [
                    (8.85, px_c - 75, py_c - 25, 38, 28),
                    (9.10, px_c + 15, py_c - 32, 32, 42),
                    (9.35, px_c + 88, py_c - 15, 28, 30),
                    (9.55, px_c - 20, py_c + 45, 75, 24),
                ]
                for ct_start, cx_o, cy_o, rx_o, ry_o in circles:
                    if t >= ct_start:
                        d.ellipse([cx_o - rx_o, cy_o - ry_o, cx_o + rx_o, cy_o + ry_o], outline=(45, 48, 55, 255), width=5)

                if t < 10.05:
                    pen_x = px_c + int(math.sin(t * 16.0) * 110)
                    pen_y = py_c + int(math.cos(t * 12.0) * 65)
                    d.line([(pen_x, pen_y), (pen_x + 145, pen_y - 165)], fill=(230, 175, 45, 255), width=14)
                    d.polygon([(pen_x, pen_y), (pen_x + 12, pen_y - 22), (pen_x + 22, pen_y - 12)], fill=(40, 40, 40, 255))

                return np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)

            elif t < 14.25:
                # 11.55 - 14.25s: Pencil snaps! Benny stares at broken pencil -> looks at camera -> whispers "Technology."
                frame = self.pose_pencil.copy()
                snap_age = t - 11.55

                gaze_y = 14.0 * (1.0 - smoothstep(12.25, 12.52, t))
                brow_raise = smoothstep(13.55, 13.90, t)

                frame = self.rig.deform(
                    frame, layout="pencil", t=t,
                    head_tilt_deg=-2.5 * brow_raise,
                    head_lean_x=0.0,
                    head_nod_y=6.0 if t < 12.3 else 0.0,
                    brow_l_dy=6.0 * brow_raise,
                    brow_r_dy=-22.0 * brow_raise,
                    gaze_dx=0.0, gaze_dy=gaze_y,
                    blink=blink,
                    mouth_open=m_open, mouth_wide=m_wide,
                    smile_dy=8.0
                )

                if 0.0 <= snap_age < 0.42:
                    pil_im = Image.fromarray(frame).convert("RGBA")
                    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    d = ImageDraw.Draw(ov)
                    for sp_i, ang in enumerate([-135, -105, -75, -45, 60, 115]):
                        rad = math.radians(ang)
                        dist = snap_age * (260.0 + sp_i * 40.0)
                        sx = int(495 + math.cos(rad) * dist)
                        sy = int(1090 + math.sin(rad) * dist + 220.0 * (snap_age ** 2))
                        alpha_sp = int(255 * (1.0 - snap_age / 0.42))
                        d.line([(sx, sy), (sx + 14, sy - 10)], fill=(235, 185, 85, alpha_sp), width=5)
                    frame = np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)

                return render_camera_move(frame, zoom=1.04 + 0.05 * smoothstep(12.4, 13.8, t))

            else:
                # 14.25 - 17.0s: Cascade of error notifications + real 3D Pixar hands darting to close them!
                frame = self.pose_shock.copy()
                shake_x = math.sin(t * 36.0) * 10.0
                frame = self.rig.deform(
                    frame, layout="mcu", t=t,
                    head_tilt_deg=math.sin(t * 18.0) * 6.0,
                    head_lean_x=shake_x,
                    brow_l_dy=-18.0, brow_r_dy=-18.0,
                    gaze_dx=math.sin(t * 22.0) * 14.0, gaze_dy=8.0,
                    blink=blink, smile_dy=16.0
                )

                pil_im = Image.fromarray(frame).convert("RGBA")
                ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                d = ImageDraw.Draw(ov)

                n_popups = min(18, int((t - 14.25) / 0.145) + 1)
                rng = np.random.default_rng(404)
                titles = [
                    "ERROR 404: STYLE MISSING", "CSS PANIC OVERFLOW", "FONT REJECTED!",
                    "TOO MANY CIRCLES!", "LAYOUT COLLAPSED", "SYSTEM CONFUSED",
                    "BUTTON BROKEN", "COLORS CLASHING!", "DESIGN EMERGENCY"
                ]
                last_x, last_y = 540, 1200
                for p_i in range(n_popups):
                    px = int(rng.integers(80, 640))
                    py = int(rng.integers(680, 1480))
                    pw, ph = 350, 175
                    d.rounded_rectangle([px + 8, py + 10, px + pw + 8, py + ph + 10], radius=12, fill=(0, 0, 0, 140))
                    d.rounded_rectangle([px, py, px + pw, py + ph], radius=12, fill=(245, 242, 242, 250), outline=(213, 0, 0, 255), width=4)
                    d.rounded_rectangle([px, py, px + pw, py + 44], radius=10, fill=(213, 14, 10, 255))
                    d.text((px + 16, py + 10), titles[p_i % len(titles)], font=get_font(800, 19), fill=(255, 255, 255, 255))
                    d.rectangle([px + pw - 38, py + 8, px + pw - 10, py + 36], fill=(140, 0, 0, 255))
                    d.text((px + pw - 30, py + 8), "X", font=get_font(800, 20), fill=(255, 255, 255, 255))
                    d.text((px + 22, py + 72), "Click [X] to spawn 2 more errors!", font=get_font(600, 18), fill=(40, 40, 45, 255))
                    d.rounded_rectangle([px + 110, py + 118, px + 240, py + 154], radius=8, fill=(213, 14, 10, 255))
                    d.text((px + 142, py + 124), "PANIC", font=get_font(700, 18), fill=(255, 255, 255, 255))
                    last_x, last_y = px + pw - 24, py + 22

                # Animate frantic dual mouse cursors darting to close the windows!
                for c_side, phase_off in [(-1, 0.0), (1, 1.7)]:
                    cx_cur = int(540 + c_side * 160 + math.sin(t * 24.0 + phase_off) * 210)
                    cy_cur = int(1050 + math.cos(t * 20.0 + phase_off) * 250)
                    d.polygon(
                        [(cx_cur, cy_cur), (cx_cur, cy_cur + 46), (cx_cur + 16, cy_cur + 34), (cx_cur + 34, cy_cur + 34)],
                        fill=(255, 255, 255, 255), outline=(20, 20, 20, 255)
                    )

                d.ellipse([last_x - 32, last_y - 32, last_x + 32, last_y + 32], outline=(255, 230, 0, 230), width=5)
                return np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)

        # =====================================================================
        # SCENE 3: THE BRANDRUMS DISCOVERY (17 - 26s)
        # =====================================================================
        elif t < 26.0:
            if t < 21.0:
                frame = self.pose_thumbsup.copy()
                darken = smoothstep(19.25, 20.2, t)
                if darken > 0.01:
                    frame = (frame.astype(np.float32) * (1.0 - 0.38 * darken)).astype(np.uint8)
                    glow = np.zeros_like(frame)
                    cv2.circle(glow, (730, 650), 320, (255, 165, 120), -1)
                    glow = cv2.GaussianBlur(glow, (99, 99), 42.0)
                    frame = cv2.addWeighted(frame, 1.0, glow, 0.35 * darken, 0)

                scr = make_office_laptop_ui("search_brandrums", t=t, brandrums_logo_rgba=self.logo_rgba)
                frame = warp_onto_office_laptop(frame, scr)

                straighten_y = -18.0 * smoothstep(19.5, 20.4, t)
                frame = self.rig.deform(
                    frame, layout="thumbsup", t=t,
                    head_tilt_deg=math.sin(t * 25.0) * 3.0 if t < 18.1 else 0.0,
                    head_nod_y=straighten_y,
                    brow_l_dy=-14.0 * darken, brow_r_dy=-14.0 * darken,
                    gaze_dx=-8.0, gaze_dy=8.0, blink=blink
                )
                return render_camera_move(frame, zoom=1.08 + 0.08 * darken)

            else:
                frame = self.pose_portal.copy()
                poke_p = smoothstep(22.0, 22.45, t) * (1.0 - smoothstep(22.82, 23.02, t))
                frame = self.rig.deform(
                    frame, layout="thumbsup", t=t,
                    head_lean_x=-14.0 * poke_p, head_nod_y=10.0 * poke_p,
                    brow_l_dy=-16.0, brow_r_dy=-16.0, blink=blink
                )

                pil_im = Image.fromarray(frame).convert("RGBA")
                ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                d = ImageDraw.Draw(ov)

                if 22.25 <= t < 22.95:
                    for r_i in range(4):
                        rr = int(28 + r_i * 22 + ((t * 90) % 22))
                        d.ellipse([455 - rr, 815 - rr, 455 + rr, 815 + rr], outline=(255, 220, 130, 210 - r_i * 45), width=5)
                    d.ellipse([425, 785, 488, 848], fill=(255, 235, 175, 245))

                # 23.05 - 24.85s: Benny inspects his real 3D hands & counts all FIVE fingers (1, 2, 3, 4, 5) with relief!
                if 23.05 <= t < 24.90:
                    # Finger positions along his real 3D hand in benny_pose_portal.png
                    finger_pts = [
                        (768, 1510),
                        (825, 1535),
                        (878, 1545),
                        (922, 1540),
                        (962, 1518),
                    ]
                    counted = sum(1 for f_t in [23.25, 23.53, 23.81, 24.09, 24.37] if t >= f_t)
                    for f_idx in range(counted):
                        fx1, fy1 = finger_pts[f_idx]
                        d.ellipse([fx1 - 24, fy1 - 68, fx1 + 24, fy1 - 20], fill=(213, 14, 10, 245), outline=(255, 255, 255, 255), width=3)
                        d.text((fx1 - 9, fy1 - 59), str(f_idx + 1), font=get_font(800, 24), fill=(255, 255, 255, 255))

                if t >= 24.85:
                    exp_p = smoothstep(24.85, 25.95, t)
                    rad_p = int(80 + exp_p * 1450)
                    d.ellipse([610 - rad_p, 1180 - rad_p, 610 + rad_p, 1180 + rad_p], fill=(255, 225, 185, int(245 * math.sin(exp_p * math.pi))), outline=(213, 14, 10, 255), width=28)
                    out_rgb = np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)
                    if exp_p > 0.35:
                        blend_s = smoothstep(0.35, 1.0, exp_p)
                        out_rgb = cv2.addWeighted(out_rgb, 1.0 - blend_s, self.pose_director, blend_s, 0)
                    return out_rgb

                return np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)

        # =====================================================================
        # SCENE 4: THE IMPOSSIBLE TRANSFORMATION (26 - 41s)
        # =====================================================================
        elif t < 41.0:
            if t < 37.3:
                frame = self.pose_director.copy()

                if 31.25 <= t < 32.35:
                    spin_p = smoothstep(31.25, 32.30, t)
                    horiz_scale = math.cos(spin_p * math.pi * 2.0)
                    w_spin = max(24, int(W * abs(horiz_scale)))
                    s_img = cv2.resize(frame, (w_spin, H), interpolation=cv2.INTER_LINEAR)
                    if horiz_scale < 0:
                        s_img = cv2.flip(s_img, 1)
                    bg_blur = cv2.GaussianBlur(self.bg_studio, (45, 45), 15.0)
                    x0 = (W - w_spin) // 2
                    bg_blur[:, x0:x0 + w_spin] = s_img
                    k_w = 25
                    kernel = np.ones((1, k_w), dtype=np.float32) / k_w
                    frame = cv2.filter2D(bg_blur, -1, kernel)

                duck_p = math.sin(smoothstep(33.00, 33.85, t) * math.pi) if 33.00 <= t <= 33.85 else 0.0
                if duck_p > 0.01:
                    frame = self.rig.deform(frame, layout="glasses", t=t, head_nod_y=155.0 * duck_p, brow_l_dy=-18.0, brow_r_dy=-18.0)

                pil_im = Image.fromarray(frame).convert("RGBA")
                ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                d = ImageDraw.Draw(ov)

                float_y = int(math.sin(t * 3.2) * 14)
                if 32.6 <= t <= 34.1:
                    ph_scale = 1.0 + 0.95 * duck_p
                    pw_s = int(170 * ph_scale)
                    ph_s = int(320 * ph_scale)
                    px_s = int(680 - 140 * duck_p)
                    py_s = int(470 + 180 * duck_p)
                    d.rounded_rectangle([px_s - pw_s // 2, py_s - ph_s // 2, px_s + pw_s // 2, py_s + ph_s // 2], radius=28, fill=(16, 18, 24, 245), outline=(213, 14, 10, 255), width=5)
                    d.rounded_rectangle([px_s - pw_s // 2 + 16, py_s - ph_s // 2 + 34, px_s + pw_s // 2 - 16, py_s - 10], radius=14, fill=(213, 14, 10, 220))
                    d.text((px_s - pw_s // 2 + 28, py_s + 20), "BRANDRUMS APP", font=get_font(800, 18), fill=(255, 255, 255, 255))

                if t >= 34.0:
                    panel_ang = 0.0
                    if 34.25 <= t < 35.40:
                        make_crooked = smoothstep(34.25, 34.50, t)
                        fix_it = smoothstep(35.05, 35.38, t)
                        panel_ang = 15.5 * make_crooked * (1.0 - fix_it)
                    panel_im = Image.new("RGBA", (480, 300), (0, 0, 0, 0))
                    pd = ImageDraw.Draw(panel_im)
                    border_col = (255, 195, 0, 255) if abs(panel_ang) > 2.0 else (213, 14, 10, 255)
                    pd.rounded_rectangle([10, 10, 470, 290], radius=20, fill=(14, 16, 22, 238), outline=border_col, width=5)
                    pd.rounded_rectangle([30, 32, 450, 145], radius=12, fill=(28, 12, 16, 255), outline=(213, 14, 10, 200), width=2)
                    pd.text((52, 54), "BRANDRUMS STUDIO 3D", font=get_font(800, 24), fill=(255, 255, 255, 255))
                    pd.text((52, 92), "Websites • Apps • Branding • AI", font=get_font(600, 18), fill=(220, 185, 185, 255))
                    pd.rounded_rectangle([30, 165, 225, 265], radius=10, fill=(213, 14, 10, 230))
                    pd.rounded_rectangle([245, 165, 450, 265], radius=10, fill=(26, 30, 40, 255), outline=(90, 95, 115, 255), width=2)
                    rot_p = panel_im.rotate(-panel_ang, resample=Image.Resampling.BICUBIC, expand=True)
                    ov.paste(rot_p, (85, 930 + float_y), rot_p)

                return np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)

            else:
                frame = self.pose_glasses.copy()
                wipe_osc = math.sin(t * 22.0) * 14.0 if 38.1 <= t < 39.05 else 0.0
                glasses_on = smoothstep(39.15, 39.65, t)
                nod_approve = math.sin((t - 40.0) * 9.5) * 18.0 if t >= 40.0 else 0.0

                frame = self.rig.deform(
                    frame, layout="glasses", t=t,
                    head_tilt_deg=wipe_osc * 0.15,
                    head_nod_y=nod_approve,
                    brow_l_dy=-16.0, brow_r_dy=-16.0,
                    blink=blink,
                    smile_dy=-12.0 * smoothstep(39.8, 40.4, t),
                    glasses_wear=glasses_on
                )
                return render_camera_move(frame, zoom=1.02 + 0.08 * smoothstep(37.3, 40.9, t))

        # =====================================================================
        # SCENE 5: THE PERFECT REVEAL (41 - 51s)
        # =====================================================================
        elif t < 51.0:
            frame = self.pose_exec.copy()
            nod_y = math.sin((t - 43.2) * 10.0) * 12.0 if 43.2 <= t < 43.95 else 0.0
            slip_jerk = 18.0 * math.exp(-max(0.0, t - 45.10) * 7.0) if t >= 45.10 else 0.0
            satisfied_smile = -14.0 * smoothstep(48.75, 49.40, t)

            frame = self.rig.deform(
                frame, layout="exec", t=t,
                head_tilt_deg=2.5 * smoothstep(48.8, 49.5, t) - 5.0 * (slip_jerk / 18.0),
                head_lean_x=-slip_jerk,
                head_nod_y=nod_y + slip_jerk * 0.8,
                brow_l_dy=-10.0 * smoothstep(47.3, 48.0, t),
                brow_r_dy=-14.0 * smoothstep(48.8, 49.4, t),
                gaze_dx=-10.0 if (42.5 <= t < 45.8) else 0.0,
                gaze_dy=8.0 if (42.5 <= t < 45.8) else 0.0,
                blink=blink,
                mouth_open=m_open, mouth_wide=m_wide,
                smile_dy=satisfied_smile
            )

            pil_im = Image.fromarray(frame).convert("RGBA")
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)

            # Handshake Gag with smartphone glass (44.0 - 46.15s): Benny lunges his arm toward the phone & slips off the glass!
            if 44.0 <= t < 46.15:
                reach = smoothstep(44.0, 44.85, t)
                slip_down = smoothstep(45.10, 45.65, t)
                retract = smoothstep(45.65, 46.10, t)
                # Warp his left-screen forearm/hand (around x=280, y=1260) toward the smartphone (x=165, y=1350) & sliding down!
                if 45.08 <= t < 45.75:
                    hy_s = int(1290 + 145 * slip_down)
                    for s_x in [125, 150, 175]:
                        d.line([(s_x, 1270), (s_x + 18, hy_s)], fill=(255, 255, 255, 210), width=5)
                    d.ellipse([115, hy_s - 18, 205, hy_s + 24], outline=(255, 255, 255, 180), width=3)

            out_rgb = np.array(Image.alpha_composite(pil_im, ov).convert("RGB"), dtype=np.uint8)
            mcu_zoom = 1.0 + 0.14 * smoothstep(46.5, 47.3, t)
            return render_camera_move(out_rgb, zoom=mcu_zoom, pan_y=-95.0 * smoothstep(46.5, 47.3, t))

        # =====================================================================
        # SCENE 6: BRANDRUMS END CARD (51 - 60s)
        # =====================================================================
        else:
            if t < 53.8:
                frame = self.pose_director.copy()
                frame = self.rig.deform(frame, layout="glasses", t=t, head_nod_y=math.sin(t * 6.0) * 8.0, smile_dy=-10.0, blink=blink)
                pull = smoothstep(51.0, 53.7, t)
                return render_camera_move(frame, zoom=1.18 - 0.18 * pull)

            im = Image.new("RGBA", (W, H), (18, 20, 26, 255))
            d = ImageDraw.Draw(im)

            d.rounded_rectangle([64, 220, W - 64, 1680], radius=44, fill=(250, 251, 254, 255), outline=(213, 14, 10, 255), width=6)
            d.rounded_rectangle([96, 252, W - 96, 660], radius=32, fill=(18, 20, 26, 255))

            logo_tilt = 0.0
            if 57.65 <= t < 58.25:
                logo_tilt = math.sin((t - 57.65) / 0.60 * math.pi * 2.0) * 1.8
            lg_pil = Image.fromarray(self.logo_rgba).resize((620, 189), Image.Resampling.LANCZOS)
            if abs(logo_tilt) > 0.05:
                lg_pil = lg_pil.rotate(logo_tilt, resample=Image.Resampling.BICUBIC, expand=False)
            im.paste(lg_pil, ((W - 620) // 2, 360), lg_pil)

            f_h1 = get_font(800, 48)
            f_sub = get_font(700, 33)
            f_cta = get_font(800, 40)
            f_url = get_font(800, 44)

            def draw_centered(txt, y_pos, font, col):
                bb = d.textbbox((0, 0), txt, font=font)
                tw = bb[2] - bb[0]
                d.text(((W - tw) // 2, y_pos), txt, font=font, fill=col)

            draw_centered("YOUR IDEA. OUR DIGITAL CRAFT.", 750, f_h1, (18, 20, 26, 255))
            d.rounded_rectangle([420, 835, 660, 845], radius=5, fill=(213, 14, 10, 255))
            draw_centered("Websites | Apps | Branding | AI | Marketing", 895, f_sub, (75, 82, 98, 255))

            d.rounded_rectangle([125, 1025, W - 125, 1145], radius=28, fill=(213, 14, 10, 255))
            draw_centered("LET'S BUILD SOMETHING AMAZING.", 1060, f_cta, (255, 255, 255, 255))

            d.rounded_rectangle([240, 1210, W - 240, 1310], radius=50, fill=(18, 20, 26, 255))
            draw_centered("www.brandrums.com", 1232, f_url, (255, 255, 255, 255))

            # Final Gag (56.6 - 59.7s): Benny sneaks back into the frame right beside the Brandrums logo,
            # tries to straighten the logo, realizes it is already aligned, and slowly backs away with an embarrassed smile!
            if 56.60 <= t <= 59.75:
                sneak_in = smoothstep(56.60, 57.55, t)
                back_away = smoothstep(58.45, 59.70, t)
                vis = sneak_in * (1.0 - back_away)

                bc_h, bc_w = 1040, 585
                b_small = cv2.resize(self.benny_cutout, (bc_w, bc_h), interpolation=cv2.INTER_AREA)
                b_pil = Image.fromarray(b_small, "RGBA")
                lean_ang = 8.0 * math.sin(smoothstep(57.40, 58.45, t) * math.pi)
                if abs(lean_ang) > 0.2:
                    b_pil = b_pil.rotate(lean_ang, resample=Image.Resampling.BICUBIC, expand=False)

                bx = int(W - 20 - (510 * vis))
                by = int(880 + math.sin(t * 14.0) * 10.0 * (1.0 - smoothstep(57.55, 58.45, t)))
                im.paste(b_pil, (bx, by), b_pil)

            return np.array(im.convert("RGB"), dtype=np.uint8)

def main():
    renderer = BennyComedyCommercialRenderer()
    if "--preview" in sys.argv:
        os.makedirs("/tmp/benny_previews", exist_ok=True)
        preview_times = [2.8, 6.5, 10.6, 13.0, 15.8, 20.0, 23.9, 31.8, 34.6, 39.5, 45.2, 48.0, 58.0]
        for pt in preview_times:
            rgb = renderer.render_frame(int(pt * FPS))
            out_p = f"/tmp/benny_previews/preview_{pt:05.2f}s.jpg"
            Image.fromarray(rgb).save(out_p, quality=90)
            print("Saved preview:", out_p)
        return

    out_video = "/home/user/video/brandrums_worst_designer_commercial.mp4"
    audio_path = os.path.join(ASSETS_DIR, "benny_soundtrack_master.wav")
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
        "-crf", "18",
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
        if (idx + 1) % 300 == 0 or idx == TOTAL_FRAMES - 1:
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
