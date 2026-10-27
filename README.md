# ProBlack 10-second Instagram Reel

**Deliverable:** [`problack_reel.mp4`](./problack_reel.mp4) — 1080×1920 (9:16), 10.00 seconds, 30 fps, H.264 video with AAC stereo audio.

A matching poster frame is available at [`problack_reel_poster.jpg`](./problack_reel_poster.jpg), and a browser preview is served by `server.py` at port 8080.

## Creative specification

| Time | Visual |
| --- | --- |
| 0.00–1.20s | A shopper comfortably browsing on a phone in a warm, modern interior. |
| 1.20–3.00s | Premium phone close-up showing the official ProBlack mobile marketplace homepage. |
| 3.00–3.75s | Gold-plated jewellery close-up. |
| 3.75–4.50s | Natural beauty/skincare close-up. |
| 4.50–5.25s | Premium books close-up. |
| 5.25–6.00s | Home essentials still life. |
| 6.00–7.00s | Black-owned business owner carefully packing a customer order. |
| 7.00–8.00s | Happy customer receiving/opening the order. |
| 8.00–10.00s | Official ProBlack logo with `problack.co` clearly below. |

The voiceover is exactly: “Discover products you love. Support Black-owned businesses. Shop ProBlack.”

The score is an original, synthesized 100 BPM instrumental with warm chord pads, soft plucks, gentle percussion, and a restrained closing shimmer. It is automatically ducked beneath the narration.

## Asset provenance

The two user-described uploads (a website screenshot and a logo) were not present in the workspace or git history. To avoid inventing a fake interface or logo, the production uses ProBlack’s own published assets:

- `assets/problack/website_mobile.png` — a live mobile homepage capture from `https://problack.co/` rendered at 390×844 CSS pixels via Microlink.
- `assets/problack/problack_logo.png` — the official logo from `https://problack.co/Logo.png`.
- Additional official product/customer/vendor imagery was fetched for reference from `https://problack.co/`.

These files were fetched by [`.github/workflows/fetch_problack_assets.yml`](./.github/workflows/fetch_problack_assets.yml). If you provide the exact original uploaded screenshot and logo, replace the two files above and rerun the renderer to use them verbatim.

The photorealistic scene stills in `assets/problack/scenes/` were generated for this reel. No captions, random logos, or invented website UI were added.

## Reproduce

```bash
python3 -m pip install -r requirements.txt
python3 build_problack_audio.py
python3 render_problack_reel.py
python3 server.py
```

Then open `http://localhost:8080/` locally, or use the Arena live preview for port 8080. The renderer composites the official homepage capture into the phone screen, creates the closing logo card, applies subtle Ken Burns motion and crossfades, grades each frame, and pipes raw RGB frames directly into ffmpeg.

## Key files

- `problack_reel.mp4` — final reel
- `problack_reel_poster.jpg` — poster/closing frame
- `index.html` — premium preview and download page
- `build_problack_audio.py` — original music + voiceover mix
- `render_problack_reel.py` — full 10s video renderer
- `assets/problack/` — official source assets, generated scenes, voiceover, and final audio
