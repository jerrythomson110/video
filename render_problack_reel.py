#!/usr/bin/env python3
"""Render the 10-second 1080x1920 ProBlack Instagram promotional reel.

The phone screen uses the official live ProBlack mobile homepage capture.
The closing card uses the official ProBlack logo with no alterations.
"""

from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets", "problack")
SCENES = os.path.join(ASSETS, "scenes")
TMP = os.path.join(ROOT, "tmp", "problack")
OUTPUT_MP4 = os.path.join(ROOT, "problack_reel.mp4")
POSTER_JPG = os.path.join(ROOT, "problack_reel_poster.jpg")
AUDIO_WAV = os.path.join(ASSETS, "final_audio.wav")
PHONE_UI_PNG = os.path.join(ASSETS, "phone_ui.png")
FINAL_FRAME_PNG = os.path.join(ASSETS, "final_frame.png")

W, H = 1080, 1920
FPS = 30
DURATION = 10.0
N_FRAMES = int(FPS * DURATION)
GOLD = (224, 188, 104, 255)
IVORY = (250, 248, 242, 255)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
RNG = np.random.default_rng(10272026)


@dataclass(frozen=True)
class Shot:
    path: str
    start: float
    end: float
    zoom_start: float
    zoom_end: float
    pan_start: tuple[float, float]
    pan_end: tuple[float, float]


def lerp(a: float, b: float, p: float) -> float:
    return a + (b - a) * p


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def fit_vertical(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    target_ratio = W / H
    ratio = im.width / im.height
    if ratio > target_ratio:
        new_w = int(round(im.height * target_ratio))
        x0 = (im.width - new_w) // 2
        im = im.crop((x0, 0, x0 + new_w, im.height))
    elif ratio < target_ratio:
        new_h = int(round(im.width / target_ratio))
        y0 = (im.height - new_h) // 2
        im = im.crop((0, y0, im.width, y0 + new_h))
    return im.resize((W, H), Image.Resampling.LANCZOS)


def make_phone_ui() -> Image.Image:
    """Composite the exact official mobile homepage capture into the phone screen."""
    base = fit_vertical(os.path.join(SCENES, "02_phone_closeup.png")).convert("RGBA")
    screenshot = Image.open(os.path.join(ASSETS, "website_mobile.png")).convert("RGB")

    # Screen bounds measured from the generated phone image, scaled to 1080x1920.
    sx, sy = 310, 425
    sw, sh = 478, 1177
    content_h = int(round(sw * screenshot.height / screenshot.width))
    content = screenshot.resize((sw, content_h), Image.Resampling.LANCZOS).convert("RGBA")

    screen = Image.new("RGBA", (sw, sh), IVORY)
    top_bar = 72
    screen.alpha_composite(content, (0, top_bar))

    draw = ImageDraw.Draw(screen)
    # Phone status bar and notch; these are device chrome, not website UI.
    draw.rounded_rectangle((sw // 2 - 102, -34, sw // 2 + 102, 47), radius=40, fill=(7, 7, 8, 255))
    draw.ellipse((sw // 2 + 58, 9, sw // 2 + 70, 21), fill=(36, 36, 38, 255))
    draw.rounded_rectangle((sw // 2 - 68, sh - 35, sw // 2 + 68, sh - 28), radius=4, fill=(25, 25, 25, 255))

    # Rounded screen mask matching the generated phone's corners.
    mask = Image.new("L", (sw, sh), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, sw - 1, sh - 1), radius=60, fill=255)
    screen.putalpha(mask)

    # A faint diagonal glass reflection keeps the screen integrated with the scene.
    glare = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glare)
    gd.polygon(
        [(70, 0), (180, 0), (90, sh), (0, sh)],
        fill=(255, 255, 255, 22),
    )
    glare = glare.filter(ImageFilter.GaussianBlur(24))
    glare.putalpha(glare.getchannel("A").point(lambda v: int(v * 0.55)))

    base.alpha_composite(screen, (sx, sy))
    base.alpha_composite(glare, (sx, sy))
    out = base.convert("RGB")
    out.save(PHONE_UI_PNG, optimize=True)
    return out


def draw_spaced_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    spacing: float,
    shadow: tuple[int, int, int, int] | None = None,
) -> None:
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + spacing * (len(text) - 1)
    x = center_x - total / 2
    for ch, cw in zip(text, widths):
        if shadow is not None:
            draw.text((x + 2, y + 3), ch, font=font, fill=shadow)
        draw.text((x, y), ch, font=font, fill=fill)
        x += cw + spacing


def make_final_frame() -> Image.Image:
    """Clean closing shot with the official, unmodified logo and the domain below."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.zeros((H, W, 3), dtype=np.float32)
    base[:] = np.array([7.0, 7.0, 9.0], dtype=np.float32)

    # Warm gold halo behind the logo card.
    radius = np.sqrt(((xx - W * 0.5) / (W * 0.78)) ** 2 + ((yy - H * 0.40) / (H * 0.58)) ** 2)
    halo = np.clip(1.0 - radius, 0.0, 1.0) ** 2.1
    base += halo[..., None] * np.array([34.0, 25.0, 9.0], dtype=np.float32)

    # Subtle corner falloff.
    corner = np.sqrt(((xx - W * 0.5) / (W * 0.72)) ** 2 + ((yy - H * 0.5) / (H * 0.72)) ** 2)
    base *= (0.90 + 0.10 * np.clip(1.0 - corner, 0.0, 1.0))[..., None]

    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    draw = ImageDraw.Draw(im)

    cx, cy = W // 2, 760
    r = 356
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=IVORY, outline=GOLD, width=5)
    draw.ellipse((cx - r + 15, cy - r + 15, cx + r - 15, cy + r - 15), outline=(238, 214, 151, 120), width=2)

    logo = Image.open(os.path.join(ASSETS, "problack_logo.png")).convert("RGBA")
    logo.thumbnail((610, 610), Image.Resampling.LANCZOS)
    im.alpha_composite(logo, (cx - logo.width // 2, cy - logo.height // 2))

    # Restrained gold particles for depth.
    for px, py, pr, alpha in [
        (170, 430, 4, 150), (900, 520, 3, 120), (210, 1010, 3, 110),
        (865, 1110, 5, 130), (130, 1370, 3, 90), (940, 1410, 4, 100),
    ]:
        draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=(224, 188, 104, alpha))

    draw.line((cx - 105, 1160, cx + 105, 1160), fill=GOLD, width=4)
    font = ImageFont.truetype(FONT_PATH, 59)
    draw_spaced_text(
        draw,
        cx,
        1230,
        "problack.co",
        font,
        (241, 214, 145, 255),
        spacing=5.0,
        shadow=(0, 0, 0, 150),
    )

    out = im.convert("RGB")
    out.save(FINAL_FRAME_PNG, optimize=True)
    out.save(POSTER_JPG, quality=92, subsampling=0)
    return out


def render_frame(base: Image.Image, p: float, shot: Shot) -> Image.Image:
    q = smoothstep(p)
    zoom = lerp(shot.zoom_start, shot.zoom_end, q)
    pan_x = lerp(shot.pan_start[0], shot.pan_end[0], q)
    pan_y = lerp(shot.pan_start[1], shot.pan_end[1], q)

    crop_w = min(W, int(round(W / zoom)))
    crop_h = min(H, int(round(H / zoom)))
    cx = W * 0.5 + pan_x
    cy = H * 0.5 + pan_y
    x0 = int(round(np.clip(cx - crop_w / 2, 0, W - crop_w)))
    y0 = int(round(np.clip(cy - crop_h / 2, 0, H - crop_h)))
    box = (x0, y0, x0 + crop_w, y0 + crop_h)
    return base.crop(box).resize((W, H), Image.Resampling.BICUBIC)


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)

    make_phone_ui()
    make_final_frame()

    shots = [
        Shot(os.path.join(SCENES, "01_browsing_person.png"), 0.00, 1.20, 1.015, 1.065, (-8, 5), (10, -6)),
        Shot(PHONE_UI_PNG, 1.20, 3.00, 1.000, 1.040, (0, 0), (0, -12)),
        Shot(os.path.join(SCENES, "03_jewellery.png"), 3.00, 3.75, 1.025, 1.095, (-12, 8), (10, -8)),
        Shot(os.path.join(SCENES, "04_beauty.png"), 3.75, 4.50, 1.090, 1.025, (12, -6), (-8, 8)),
        Shot(os.path.join(SCENES, "05_books.png"), 4.50, 5.25, 1.025, 1.090, (-8, 10), (10, -6)),
        Shot(os.path.join(SCENES, "06_home.png"), 5.25, 6.00, 1.085, 1.020, (10, -8), (-10, 8)),
        Shot(os.path.join(SCENES, "07_owner_packing.png"), 6.00, 7.00, 1.015, 1.075, (-6, 8), (8, -6)),
        Shot(os.path.join(SCENES, "08_customer.png"), 7.00, 8.00, 1.065, 1.015, (8, -6), (-6, 6)),
        Shot(FINAL_FRAME_PNG, 8.00, 10.00, 1.045, 1.000, (0, 0), (0, 0)),
    ]
    bases = [fit_vertical(s.path) for s in shots]
    boundaries = [s.end for s in shots[:-1]]
    fades = [0.22, 0.26, 0.14, 0.14, 0.14, 0.14, 0.20, 0.28]

    # Precompute cinematic vignette and a reusable grain field.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    radial = np.sqrt(((xx - W * 0.5) / (W * 0.82)) ** 2 + ((yy - H * 0.5) / (H * 0.82)) ** 2)
    vignette = 1.0 - 0.115 * np.clip(radial - 0.42, 0.0, 1.0)
    top_fade = np.clip((250.0 - yy) / 250.0, 0.0, 1.0)
    bottom_fade = np.clip((yy - (H - 250.0)) / 250.0, 0.0, 1.0)
    vignette *= 1.0 - 0.075 * (top_fade + bottom_fade)
    vignette = vignette[..., None].astype(np.float32)
    grain_base = RNG.standard_normal((H, W)).astype(np.float32)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_log = os.path.join(TMP, "ffmpeg.log")
    with open(ffmpeg_log, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{W}x{H}",
                "-r",
                str(FPS),
                "-i",
                "-",
                "-i",
                AUDIO_WAV,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-profile:v",
                "high",
                "-level",
                "4.1",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(FPS),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                "-metadata",
                "title=ProBlack 10s Instagram Reel",
                "-shortest",
                OUTPUT_MP4,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log,
            bufsize=10**8,
        )

        assert proc.stdin is not None
        for frame_idx in range(N_FRAMES):
            t = frame_idx / FPS
            shot_idx = next((i for i, b in enumerate(boundaries) if t < b), len(shots) - 1)
            shot = shots[shot_idx]
            p = float(np.clip((t - shot.start) / (shot.end - shot.start), 0.0, 1.0))
            frame = render_frame(bases[shot_idx], p, shot)

            # Subtle crossfades at shot boundaries.
            if shot_idx < len(boundaries):
                boundary = boundaries[shot_idx]
                fade = fades[shot_idx]
                if t >= boundary - fade:
                    alpha = smoothstep((t - (boundary - fade)) / fade)
                    next_shot = shots[shot_idx + 1]
                    next_frame = render_frame(bases[shot_idx + 1], 0.0, next_shot)
                    frame = Image.blend(frame, next_frame, alpha)

            arr = np.asarray(frame, dtype=np.float32)
            arr = (arr - 128.0) * 1.025 + 128.0
            arr[..., 0] += 1.2
            arr[..., 2] -= 1.4
            arr *= vignette
            grain = np.roll(grain_base, (frame_idx * 11) % H, axis=0)
            grain = np.roll(grain, (frame_idx * 7) % W, axis=1)
            arr += grain[..., None] * 1.35

            # Gentle open and close.
            global_in = smoothstep(t / 0.18)
            global_out = smoothstep((DURATION - t) / 0.20)
            arr *= min(global_in, global_out)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            proc.stdin.write(arr.tobytes())
            if frame_idx % 30 == 0:
                print(f"Rendered {frame_idx:03d}/{N_FRAMES} frames ({t:04.2f}s)", flush=True)

        proc.stdin.close()
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with code {return_code}; see {ffmpeg_log}")

    print(f"Wrote {OUTPUT_MP4}: {os.path.getsize(OUTPUT_MP4):,} bytes")
    print(f"Wrote {POSTER_JPG}")


if __name__ == "__main__":
    main()
