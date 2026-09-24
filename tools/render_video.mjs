// Render the side-by-side video from recorded benchmark runs. No model is needed: every inference
// result, latency, score and crash on screen comes from results/<device>-run-NNN.json and
// results/summary.json, replayed step for step by static/versus-core.js. Only the playback rate
// (an 8x fast-forward, labelled on screen), the layout and the text around the games are the video's.
//
//     npm run render                      # video/out/lane-runner-gpu-vs-ane.mp4, 1920x1080, 30 fps
//     RUN=3 npm run render                # another recorded run
//     SET=english npm run render          # the runs in results/english/ (upstream's checkpoint for this game)
//
// Needs Google Chrome (driven by playwright-core) and the ffmpeg binary from ffmpeg-static.
import { createServer } from 'node:http';
import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import ffmpeg from 'ffmpeg-static';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const OUT = join(ROOT, 'video', 'out'), RUN = +(process.env.RUN || 1), SET = process.env.SET || '';
const FILE = join(OUT, process.env.OUTPUT || `lane-runner-gpu-vs-ane${SET ? '-' + SET : ''}.mp4`);
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png' };

// static files only, loopback only, nothing outside the repository
const server = createServer((req, res) => {
  const rel = normalize(decodeURIComponent(req.url.split('?')[0])).replace(/^([/\\])+/, '');
  const full = join(ROOT, rel || 'versus.html');
  if (!full.startsWith(ROOT) || !existsSync(full) || !statSync(full).isFile()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'Content-Type': TYPES[extname(full)] || 'application/octet-stream' });
  res.end(readFileSync(full));
}).listen(0, '127.0.0.1');
await new Promise(r => server.once('listening', r));
const url = `http://127.0.0.1:${server.address().port}/versus.html?render&run=${RUN}${SET ? '&set=' + SET : ''}`;

mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
page.on('pageerror', e => { console.error('page error:', e.message); process.exitCode = 1; });
await page.goto(url);
await page.waitForFunction(() => window.vsReady || window.vsError, null, { timeout: 30000 });
const err = await page.evaluate(() => window.vsError);
if (err) throw new Error(err);
await page.evaluate(() => document.fonts.ready);
const { frames, fps } = await page.evaluate(() => ({ frames: window.vsRender.frames, fps: window.vsRender.fps }));

const enc = spawn(ffmpeg, ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-',
  '-c:v', 'libx264', '-profile:v', 'high', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-r', String(fps),
  '-movflags', '+faststart', FILE], { stdio: ['pipe', 'inherit', 'inherit'] });
const done = new Promise((ok, fail) => enc.on('close', c => (c === 0 ? ok() : fail(new Error('ffmpeg exited ' + c)))));

const log = [], stills = new Set([0, Math.round(1.5 * fps), 5 * fps, 9 * fps, 14 * fps, 18 * fps, frames - 1]);
for (let f = 0; f < frames; f++) {
  const at = await page.evaluate(f => window.vsRender.seek(f), f);
  const png = await page.screenshot({ type: 'png' });
  if (!enc.stdin.write(png)) await new Promise(r => enc.stdin.once('drain', r));
  if (stills.has(f)) writeFileSync(join(OUT, `frame${SET ? '-' + SET : ''}-${String(f).padStart(4, '0')}.png`), png);
  log.push({ frame: f, ...at });
  if (f % fps === 0) process.stdout.write(`\rframe ${f}/${frames}`);
}
enc.stdin.end();
await done;
await browser.close();
server.close();
writeFileSync(join(OUT, `render-log${SET ? '-' + SET : ''}.json`), JSON.stringify({ run: RUN, fps, frames, file: FILE, timeline: log }));
console.log(`\nwrote ${FILE}: ${frames} frames, ${(frames / fps).toFixed(1)} s at ${fps} fps, 1920x1080`);
