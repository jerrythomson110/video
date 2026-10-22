# Even Mr. Bean Needs a Website!

A 10-second, vertical 3D animated comedy scene set in a messy London office.

## Deliverables

- `mr_bean_needs_a_website.mp4` — 1080 × 1920, 30 fps, 10.00 seconds, H.264 video with 48 kHz stereo AAC audio
- `mr_bean_website_poster.jpg` — end-card poster frame
- `index.html` — self-contained viewing and download page

## Scene

- Articulated, expressive 3D character in a brown suit, white shirt, and red tie
- Old CRT computer with a deliberately broken retro business website
- Customer inquiry cards that arrive and disappear
- Typing, confusion, head scratching, computer shaking, and a final shrug
- Procedural office details: rainy London window, Big Ben, red bus, corkboard, clock, lamp, mug, floppy disks, papers, cables, and Teddy
- Continuous cinematic camera move and an animated end card: **“EVEN MR. BEAN NEEDS A WEBSITE!”**
- Original pizzicato score and synchronized comedy foley

## Render

Requires Node.js, Python 3, NumPy, Pillow, and `imageio-ffmpeg`.

```bash
npm install
python3 -m pip install numpy pillow imageio-ffmpeg
node render_mr_bean.mjs
```

For seven full-resolution review frames without rendering the film:

```bash
node render_mr_bean.mjs --preview
```

The renderer uses Three.js in headless Chromium with SwiftShader. Geometry, textures, character animation, website UI, typography, and sound are generated in code; no external footage is used.

## View locally

```bash
python3 server.py
```

Then open port `8080`. The server supports HTTP byte ranges for video seeking.
