// JavaScript port of voice_rectifier.py: same features, same model, same matching.

export const STEM = 32;
export const BLOCKS = [[48, 1], [48, 1], [64, 2], [64, 1], [96, 2], [96, 1], [128, 1]];

// ---------------------------------------------------------------- files

function parseNpy(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = bytes[6];
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const start = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(bytes.subarray(start, start + headerLen));
  const descr = header.match(/'descr':\s*'([^']+)'/)[1];
  const shape = header.match(/'shape':\s*\(([^)]*)\)/)[1].split(",").filter(s => s.trim()).map(Number);
  const data = bytes.slice(start + headerLen);
  const array = descr.endsWith("i1") ? new Int8Array(data.buffer) : new Float32Array(data.buffer);
  return { array, shape };
}

export function parseNpz(buffer) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const out = {};
  let pos = 0;
  while (pos + 30 < bytes.length && view.getUint32(pos, true) === 0x04034b50) {
    const size = view.getUint32(pos + 18, true);
    const nameLen = view.getUint16(pos + 26, true);
    const extraLen = view.getUint16(pos + 28, true);
    const name = new TextDecoder().decode(bytes.subarray(pos + 30, pos + 30 + nameLen)).replace(/\.npy$/, "");
    const dataStart = pos + 30 + nameLen + extraLen;
    out[name] = parseNpy(bytes.subarray(dataStart, dataStart + size));
    pos = dataStart + size;
  }
  return out;
}

export function readWav(buffer) {
  const view = new DataView(buffer);
  let pos = 12, fmt = null, data = null;
  while (pos + 8 <= buffer.byteLength) {
    const id = String.fromCharCode(...new Uint8Array(buffer, pos, 4));
    const size = view.getUint32(pos + 4, true);
    if (id === "fmt ") fmt = { tag: view.getUint16(pos + 8, true), channels: view.getUint16(pos + 10, true),
                               rate: view.getUint32(pos + 12, true), bits: view.getUint16(pos + 22, true) };
    if (id === "data") data = { offset: pos + 8, size: Math.min(size, buffer.byteLength - pos - 8) };
    pos += 8 + size + (size & 1);
  }
  const n = Math.floor(data.size / (fmt.bits / 8) / fmt.channels);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let c = 0; c < fmt.channels; c++) {
      const at = data.offset + (i * fmt.channels + c) * fmt.bits / 8;
      sum += fmt.tag === 3 ? view.getFloat32(at, true) : view.getInt16(at, true) / 32768;
    }
    x[i] = sum / fmt.channels;
  }
  return { samples: x, rate: fmt.rate };
}

// ---------------------------------------------------------------- audio

function percentile(values, q) {
  const s = Float64Array.from(values).sort();
  const p = (s.length - 1) * q / 100, lo = Math.floor(p), hi = Math.ceil(p);
  return s[lo] + (s[hi] - s[lo]) * (p - lo);
}

function frameDb(x, start, len) {
  let sum = 0;
  for (let i = start; i < start + len; i++) sum += x[i] * x[i];
  return 10 * Math.log10(sum / len + 1e-10);
}

export function speechRegion(x, cfg) {
  const frame = Math.round(cfg.sample_rate * cfg.trim_frame_ms / 1000);
  let src = x;
  if (x.length < frame) { src = new Float32Array(frame); for (let i = 0; i < frame; i++) src[i] = x[i % x.length]; }
  const n = Math.max(1, Math.floor(src.length / frame));
  const energy = new Float64Array(n);
  for (let i = 0; i < n; i++) energy[i] = frameDb(src, i * frame, frame);
  const peak = Math.max(...energy);
  let threshold = Math.max(peak - cfg.trim_rel_db, percentile(energy, 10) + cfg.trim_floor_db);
  if (threshold >= peak) threshold = peak - 3;
  let first = -1, last = -1;
  for (let i = 0; i < n; i++) if (energy[i] >= threshold) { if (first < 0) first = i; last = i; }
  const pad = Math.round(cfg.sample_rate * cfg.trim_pad_ms / 1000);
  return [Math.max(0, first * frame - pad), Math.min(x.length, (last + 1) * frame + pad)];
}

export function speechSeconds(x, cfg) {
  const [s, e] = speechRegion(x, cfg);
  return (e - s) / cfg.sample_rate;
}

export function prepareClip(x, cfg) {
  const [start, end] = speechRegion(x, cfg);
  const n = Math.round(cfg.clip_seconds * cfg.sample_rate);
  const [lo, hi] = end - start >= n ? [start, end - n] : [end - n, start];
  const offset = Math.floor((lo + hi) / 2);
  const clip = new Float32Array(n);
  for (let i = Math.max(offset, 0); i < Math.min(offset + n, x.length); i++) clip[i - offset] = x[i];
  return clip;
}

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = -2 * Math.PI / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(angle * k), wi = Math.sin(angle * k);
        const a = i + k, b = a + len / 2;
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
}

const melCache = new Map();
function melFilters(cfg) {
  const key = `${cfg.sample_rate}-${cfg.n_fft}-${cfg.n_mels}-${cfg.fmin}-${cfg.fmax}`;
  if (melCache.has(key)) return melCache.get(key);
  const toMel = f => 2595 * Math.log10(1 + f / 700);
  const lo = toMel(cfg.fmin), hi = toMel(cfg.fmax);
  const hz = Array.from({ length: cfg.n_mels + 2 }, (_, i) => 700 * (10 ** ((lo + (hi - lo) * i / (cfg.n_mels + 1)) / 2595) - 1));
  const bins = cfg.n_fft / 2 + 1;
  const filters = new Float32Array(bins * cfg.n_mels);  // [bin][mel]
  for (let m = 0; m < cfg.n_mels; m++) {
    for (let k = 0; k < bins; k++) {
      const f = cfg.sample_rate / 2 * k / (bins - 1);
      const up = (f - hz[m]) / (hz[m + 1] - hz[m]), down = (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1]);
      filters[k * cfg.n_mels + m] = Math.max(0, Math.min(up, down));
    }
  }
  melCache.set(key, filters);
  return filters;
}

export function features(clip, cfg) {
  const x = new Float32Array(clip.length);
  x[0] = clip[0];
  for (let i = 1; i < clip.length; i++) x[i] = clip[i] - cfg.preemphasis * clip[i - 1];
  const frames = 1 + Math.floor((x.length - cfg.win_length) / cfg.hop_length);
  const bins = cfg.n_fft / 2 + 1, filters = melFilters(cfg);
  const out = new Float32Array(cfg.n_mels * frames);
  const re = new Float64Array(cfg.n_fft), im = new Float64Array(cfg.n_fft), power = new Float32Array(bins);
  for (let t = 0; t < frames; t++) {
    re.fill(0); im.fill(0);
    for (let i = 0; i < cfg.win_length; i++) {
      re[i] = Math.fround(x[t * cfg.hop_length + i] * Math.fround(0.5 - 0.5 * Math.cos(2 * Math.PI * i / cfg.win_length)));
    }
    fft(re, im);
    for (let k = 0; k < bins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < cfg.n_mels; m++) {
      let sum = 0;
      for (let k = 0; k < bins; k++) sum += power[k] * filters[k * cfg.n_mels + m];
      out[m * frames + t] = (Math.log(sum + cfg.log_eps) - cfg.feat_mean) / cfg.feat_std;
    }
  }
  return { data: out, height: cfg.n_mels, width: frames };
}

// ---------------------------------------------------------------- model

function conv3x3(x, c, h, w, weight, bias, stride, depthwise) {
  const cout = bias.length, ho = Math.floor((h - 1) / stride) + 1, wo = Math.floor((w - 1) / stride) + 1;
  const out = new Float32Array(cout * ho * wo);
  for (let o = 0; o < cout; o++) {
    const inputs = depthwise ? [o] : Array.from({ length: c }, (_, i) => i);
    for (let i = 0; i < ho; i++) {
      for (let j = 0; j < wo; j++) {
        let sum = bias[o];
        for (let n = 0; n < inputs.length; n++) {
          const ci = inputs[n], wbase = (o * (depthwise ? 1 : c) + n) * 9;
          for (let ki = 0; ki < 3; ki++) {
            const y = i * stride + ki - 1;
            if (y < 0 || y >= h) continue;
            for (let kj = 0; kj < 3; kj++) {
              const xx = j * stride + kj - 1;
              if (xx >= 0 && xx < w) sum += weight[wbase + ki * 3 + kj] * x[(ci * h + y) * w + xx];
            }
          }
        }
        out[(o * ho + i) * wo + j] = sum > 0 ? sum : 0;
      }
    }
  }
  return { x: out, c: cout, h: ho, w: wo };
}

function pointwise(x, c, h, w, weight, bias) {
  const cout = bias.length, size = h * w, out = new Float32Array(cout * size);
  for (let o = 0; o < cout; o++) {
    const row = out.subarray(o * size, (o + 1) * size);
    row.fill(bias[o]);
    for (let ci = 0; ci < c; ci++) {
      const k = weight[o * c + ci], src = x.subarray(ci * size, (ci + 1) * size);
      for (let p = 0; p < size; p++) row[p] += k * src[p];
    }
    for (let p = 0; p < size; p++) if (row[p] < 0) row[p] = 0;
  }
  return out;
}

export function normalize(v) {
  let norm = 0;
  for (const a of v) norm += a * a;
  norm = Math.sqrt(norm) + 1e-12;
  return Float32Array.from(v, a => a / norm);
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export class Model {
  constructor(meta, npzBuffer) {
    this.audio = meta.audio;
    this.thresholds = meta.thresholds;
    this.modelId = meta.model_id;
    const raw = parseNpz(npzBuffer);
    this.w = {};
    for (const key of Object.keys(raw)) {
      if (key.endsWith(".wq")) {
        const q = raw[key].array, scale = raw[key.slice(0, -1) + "s"].array, per = q.length / scale.length;
        this.w[key.slice(0, -3) + ".w"] = Float32Array.from(q, (v, i) => v * scale[Math.floor(i / per)]);
      } else if (key.endsWith(".b")) {
        this.w[key] = raw[key].array;
      }
    }
  }

  static async load(base) {
    const [meta, npz] = await Promise.all([
      fetch(base + "model.json").then(r => r.json()), fetch(base + "model.npz").then(r => r.arrayBuffer())]);
    return new Model(meta, npz);
  }

  forward({ data, height, width }) {
    const w = this.w;
    let t = conv3x3(data, 1, height, width, w["stem.w"], w["stem.b"], 2, false);
    let channels = STEM;
    BLOCKS.forEach(([out, stride], i) => {
      const y = conv3x3(t.x, t.c, t.h, t.w, w[`b${i}.dw.w`], w[`b${i}.dw.b`], stride, true);
      const z = pointwise(y.x, y.c, y.h, y.w, w[`b${i}.pw.w`], w[`b${i}.pw.b`]);
      if (stride === 1 && channels === out) for (let p = 0; p < z.length; p++) z[p] += t.x[p];
      t = { x: z, c: out, h: y.h, w: y.w };
      channels = out;
    });
    const pooled = new Float32Array(2 * t.c);
    const col = new Float64Array(t.w);
    for (let c = 0; c < t.c; c++) {
      col.fill(0);
      for (let i = 0; i < t.h; i++) for (let j = 0; j < t.w; j++) col[j] += t.x[(c * t.h + i) * t.w + j] / t.h;
      let mean = 0, varSum = 0;
      for (const v of col) mean += v / t.w;
      for (const v of col) varSum += (v - mean) ** 2 / t.w;
      pooled[c] = mean;
      pooled[t.c + c] = Math.sqrt(varSum + 1e-5);
    }
    const fcW = w["fc.w"], fcB = w["fc.b"], emb = new Float32Array(fcB.length);
    for (let o = 0; o < emb.length; o++) {
      let s = fcB[o];
      for (let i = 0; i < pooled.length; i++) s += fcW[o * pooled.length + i] * pooled[i];
      emb[o] = s;
    }
    return normalize(emb);
  }

  embed(wave) {
    return this.forward(features(prepareClip(wave, this.audio), this.audio));
  }

  threshold(nExamples, seconds) {
    let group = this.thresholds[0];
    if (seconds !== undefined) group = this.thresholds.find(g => g.max_seconds === null || seconds < g.max_seconds);
    const counts = Object.keys(group.by_examples).map(Number);
    const usable = counts.filter(k => k <= nExamples);
    return group.by_examples[String(usable.length ? Math.max(...usable) : Math.min(...counts))];
  }
}

// ---------------------------------------------------------------- listening

export class VoiceDetector {
  constructor(sampleRate = 16000) {
    Object.assign(this, { frameMs: 20, startDb: 10, stopDb: 6, releaseDb: 35, startFrames: 3,
                          pauseMs: 500, preroll: 10, maxSeconds: 4, minDb: -55 });
    this.flen = sampleRate * this.frameMs / 1000;
    this.buf = new Float32Array(0);
    this.history = [];
    this.floor = null;
    this.pre = [];
    this.level = -100;
    this.reset();
  }

  reset() { Object.assign(this, { inSpeech: false, seg: [], run: 0, silence: 0, peak: -200 }); }

  emit(tooLong) {
    const keep = Math.max(0, this.silence - Math.floor(150 / this.frameMs));
    const frames = keep ? this.seg.slice(0, this.seg.length - keep) : this.seg;
    const audio = new Float32Array(frames.length * this.flen);
    frames.forEach((f, i) => audio.set(f, i * this.flen));
    this.reset();
    this.pre = [];
    return { audio, tooLong };
  }

  feed(samples) {
    const all = new Float32Array(this.buf.length + samples.length);
    all.set(this.buf); all.set(samples, this.buf.length);
    const n = Math.floor(all.length / this.flen), found = [];
    this.buf = all.slice(n * this.flen);
    for (let k = 0; k < n; k++) {
      const frame = all.slice(k * this.flen, (k + 1) * this.flen);
      const e = frameDb(frame, 0, this.flen);
      this.level = e;
      this.history.push(e);
      if (this.history.length > 150) this.history.shift();
      if (this.floor === null && this.history.length < 10) { this.pushPre(frame); continue; }
      this.floor = percentile(this.history, 20);
      if (!this.inSpeech) {
        this.pushPre(frame);
        this.run = e > Math.max(this.floor + this.startDb, this.minDb) ? this.run + 1 : 0;
        if (this.run >= this.startFrames) {
          this.inSpeech = true;
          this.seg = this.pre.slice();
          this.peak = Math.max(...this.seg.map(f => frameDb(f, 0, this.flen)));
        }
        continue;
      }
      this.seg.push(frame);
      this.peak = Math.max(this.peak, e);
      const speaking = e > Math.max(this.floor + this.stopDb, this.peak - this.releaseDb, this.minDb);
      this.silence = speaking ? 0 : this.silence + 1;
      if (this.silence * this.frameMs >= this.pauseMs) found.push(this.emit(false));
      else if (this.seg.length * this.frameMs >= this.maxSeconds * 1000) found.push(this.emit(true));
    }
    return found;
  }

  flush() { return this.inSpeech && this.seg.length ? [this.emit(false)] : []; }

  pushPre(frame) {
    this.pre.push(frame);
    if (this.pre.length > this.preroll) this.pre.shift();
  }
}

export function isSpeech(x, sampleRate = 16000) {
  const flen = sampleRate / 50, n = Math.floor(x.length / flen);
  if (n < 2) return [false, "too short"];
  const energy = Array.from({ length: n }, (_, i) => frameDb(x, i * flen, flen));
  const peak = Math.max(...energy);
  const loud = energy.map((e, i) => (e >= peak - 20 ? i : -1)).filter(i => i >= 0);
  if (loud.length * 20 < 80) return [false, "too short"];
  const bins = flen / 2 + 1, lo = Math.ceil(100 * flen / sampleRate), hi = Math.floor(4000 * flen / sampleRate);
  const cos = new Float64Array(flen * bins), sin = new Float64Array(flen * bins);
  for (let k = 0; k < bins; k++) for (let t = 0; t < flen; t++) {
    cos[k * flen + t] = Math.cos(2 * Math.PI * k * t / flen); sin[k * flen + t] = Math.sin(2 * Math.PI * k * t / flen);
  }
  let flatness = 0, inBand = 0, total = 0;
  const frame = new Float64Array(flen);
  for (const f of loud) {
    for (let t = 0; t < flen; t++) frame[t] = x[f * flen + t] * (0.5 - 0.5 * Math.cos(2 * Math.PI * t / (flen - 1)));
    let logSum = 0, sum = 0;
    for (let k = 0; k < bins; k++) {
      let r = 0, i = 0;
      for (let t = 0; t < flen; t++) { r += frame[t] * cos[k * flen + t]; i -= frame[t] * sin[k * flen + t]; }
      const p = r * r + i * i + 1e-12;
      total += p;
      if (k >= lo && k <= hi) { logSum += Math.log(p); sum += p; inBand += p; }
    }
    const count = hi - lo + 1;
    flatness += Math.exp(logSum / count) / (sum / count) / loud.length;
  }
  if (flatness > 0.30) return [false, "sounds like noise"];
  if (inBand / total < 0.3) return [false, "not in the voice range"];
  return [true, ""];
}

export function match(commands, embedding, seconds, model, margin = 0.03, tolerance = 1.6) {
  // commands: [{name, examples: [{embedding, seconds}]}]
  const scored = commands.filter(c => c.examples.length).map(c => {
    const proto = normalize(c.examples[0].embedding.map((_, i) => c.examples.reduce((s, e) => s + e.embedding[i], 0)));
    return { command: c, score: dot(proto, embedding) };
  }).sort((a, b) => b.score - a.score);
  if (!scored.length) return { name: null, reason: "no commands" };
  const best = scored[0], runnerUp = scored[1] ? scored[1].score : -1;
  const lengths = best.command.examples.map(e => e.seconds);
  const avg = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const bar = model.threshold(lengths.length, avg);
  const info = { best: best.command.name, score: best.score, threshold: bar, scores: scored };
  if (best.score < bar) return { name: null, reason: "not similar enough", ...info };
  if (best.score - runnerUp < margin) return { name: null, reason: `too close to "${scored[1].command.name}"`, ...info };
  if (seconds < Math.min(...lengths) / tolerance - 0.1 || seconds > Math.max(...lengths) * tolerance + 0.1) {
    return { name: null, reason: `wrong length (${seconds.toFixed(1)} s)`, ...info };
  }
  return { name: best.command.name, ...info };
}
