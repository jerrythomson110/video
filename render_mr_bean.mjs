import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import { spawn, execFileSync } from 'node:child_process';

process.env.AWS_EXECUTION_ENV ||= 'AWS_Lambda_nodejs22.x';
const [{ default: chromium }, { default: puppeteer }, { inflate }] = await Promise.all([
  import('@sparticuz/chromium'),
  import('puppeteer-core'),
  import('./node_modules/@sparticuz/chromium/build/esm/lambdafs.js')
]);

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(ROOT, 'mr_bean_needs_a_website.mp4');
const WORK = path.join(ROOT, '.frames');
fs.mkdirSync(WORK, { recursive: true });
const preview = process.argv.includes('--preview');
const W = 1080;
const H = 1920;
const FPS = 30;
const FRAMES = 300;

const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff'
};
const server = http.createServer((req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    let rel = decodeURIComponent(url.pathname);
    if (rel === '/') rel = '/mr_bean_scene.html';
    const file = path.resolve(ROOT, `.${rel}`);
    if (!file.startsWith(ROOT + path.sep) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); res.end('Not found'); return;
    }
    res.writeHead(200, {
      'Content-Type': mime[path.extname(file)] || 'application/octet-stream',
      'Cache-Control': 'no-store',
      'Access-Control-Allow-Origin': '*'
    });
    fs.createReadStream(file).pipe(res);
  } catch (error) {
    res.writeHead(500); res.end(String(error));
  }
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');
const port = server.address().port;

const alLib = '/tmp/al2023/lib/libnspr4.so';
if (!fs.existsSync(alLib)) {
  await inflate(path.join(ROOT, 'node_modules/@sparticuz/chromium/bin/al2023.tar.br'));
}
process.env.LD_LIBRARY_PATH = ['/tmp/al2023/lib', '/tmp', process.env.LD_LIBRARY_PATH || ''].join(':');
process.env.FONTCONFIG_PATH ||= '/tmp/fonts';

let browser;
let ffmpeg;
try {
  browser = await puppeteer.launch({
    args: chromium.args,
    defaultViewport: { width: W, height: H, deviceScaleFactor: 1 },
    executablePath: await chromium.executablePath(),
    headless: 'shell',
    env: process.env,
    protocolTimeout: 240000
  });
  const page = await browser.newPage();
  page.on('pageerror', error => console.error('[scene]', error));
  page.on('console', msg => {
    if (msg.type() === 'error' || msg.type() === 'warning') console.log('[browser]', msg.text());
  });
  await page.goto(`http://127.0.0.1:${port}/mr_bean_scene.html`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction('window.__ready === true', { timeout: 120000 });
  console.log('Scene ready:', await page.evaluate(() => window.sceneInfo()));

  async function capture(t) {
    const data = await page.evaluate(time => window.captureFrame(time), t);
    return Buffer.from(data.slice(data.indexOf(',') + 1), 'base64');
  }

  if (preview) {
    const times = [0.6, 1.9, 3.15, 4.55, 5.72, 7.15, 9.25];
    for (let i = 0; i < times.length; i++) {
      const started = Date.now();
      const file = path.join(WORK, `preview_${String(i + 1).padStart(2, '0')}.jpg`);
      fs.writeFileSync(file, await capture(times[i]));
      console.log(`Preview ${i + 1}/${times.length}  t=${times[i].toFixed(2)}s  ${Date.now() - started}ms  ${file}`);
    }
    fs.writeFileSync(path.join(ROOT, 'mr_bean_website_poster.jpg'), await capture(9.25));
  } else {
    const ffmpegPath = execFileSync('python3', ['-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'], { encoding: 'utf8' }).trim();
    const silentPath = path.join(WORK, 'mr_bean_silent.mp4');
    const audioPath = path.join(WORK, 'mr_bean_soundtrack.wav');
    ffmpeg = spawn(ffmpegPath, [
      '-y', '-loglevel', 'error',
      '-f', 'image2pipe', '-vcodec', 'mjpeg', '-framerate', String(FPS), '-i', 'pipe:0',
      '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '17',
      '-pix_fmt', 'yuv420p', '-r', String(FPS), '-movflags', '+faststart', silentPath
    ], { stdio: ['pipe', 'inherit', 'inherit'] });
    const encodeDone = new Promise((resolve, reject) => {
      ffmpeg.once('exit', code => code === 0 ? resolve() : reject(new Error(`ffmpeg encoder exited ${code}`)));
      ffmpeg.once('error', reject);
    });
    const started = Date.now();
    for (let i = 0; i < FRAMES; i++) {
      const jpeg = await capture(i / FPS);
      if (!ffmpeg.stdin.write(jpeg)) await once(ffmpeg.stdin, 'drain');
      if (i % 15 === 0 || i === FRAMES - 1) {
        const elapsed = (Date.now() - started) / 1000;
        const rate = (i + 1) / Math.max(0.01, elapsed);
        const eta = (FRAMES - i - 1) / Math.max(0.01, rate);
        console.log(`Frame ${String(i + 1).padStart(3)}/${FRAMES}  ${(i / FPS).toFixed(2)}s  ${rate.toFixed(2)} fps  ETA ${eta.toFixed(0)}s`);
      }
    }
    ffmpeg.stdin.end();
    await encodeDone;
    ffmpeg = null;
    console.log('Picture render complete. Building original score and comedy sound design…');
    execFileSync('python3', [path.join(ROOT, 'build_mr_bean_audio.py'), audioPath], { stdio: 'inherit' });
    execFileSync(ffmpegPath, [
      '-y', '-loglevel', 'error', '-i', silentPath, '-i', audioPath,
      '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
      '-t', '10.000', '-movflags', '+faststart', OUT
    ], { stdio: 'inherit' });
    fs.writeFileSync(path.join(ROOT, 'mr_bean_website_poster.jpg'), await capture(9.25));
    console.log(`Finished: ${OUT} (${(fs.statSync(OUT).size / 1048576).toFixed(1)} MB)`);
  }
} finally {
  if (ffmpeg && ffmpeg.exitCode === null) ffmpeg.kill('SIGTERM');
  if (browser) await browser.close();
  server.close();
}
