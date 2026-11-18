import { Model, readWav, VoiceDetector, isSpeech, speechSeconds, match, dot, normalize } from "./rectifier.js";

const $ = s => document.querySelector(s);
const RATE = 16000;
const SAMPLES = { bird_house: "bird house", dog_off: "dog off", learn_follow: "learn follow" };
const SPEAKERS = ["A", "B", "C", "D"];
let model;
const commands = [];  // {name, examples: [{embedding, seconds, wave}]}

// ------------------------------------------------------------------ helpers

const cache = {};
async function sample(name) {
  if (!cache[name]) cache[name] = fetch(`samples/${name}.wav`).then(r => r.arrayBuffer()).then(b => readWav(b).samples);
  return cache[name];
}

let player;
function play(wave) {
  player = player || new AudioContext();
  const buffer = player.createBuffer(1, wave.length, RATE);
  buffer.copyToChannel(wave, 0);
  const src = player.createBufferSource();
  src.buffer = buffer;
  src.connect(player.destination);
  src.start();
}

function barCells(embedding) {
  const peak = Math.max(...embedding.map(Math.abs));
  return Array.from(embedding, v => {
    const pct = Math.round(Math.abs(v) / peak * 100);
    const color = v >= 0 ? "var(--accent)" : "var(--blue)";
    return `<i style="background:color-mix(in srgb, ${color} ${pct}%, var(--surface-2))"></i>`;
  }).join("");
}

function drawWave(canvas, wave) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.fillStyle = getComputedStyle(document.body).getPropertyValue("--accent");
  const step = wave.length / w;
  let peak = 1e-6;
  for (const v of wave) peak = Math.max(peak, Math.abs(v));
  for (let x = 0; x < w; x++) {
    let m = 0;
    for (let i = Math.floor(x * step); i < Math.floor((x + 1) * step); i++) m = Math.max(m, Math.abs(wave[i]));
    const bar = Math.max(1, m / peak * (h - 6));
    g.fillRect(x, (h - bar) / 2, 0.8, bar);
  }
}

const fmtTime = () => new Date().toTimeString().slice(0, 8);

// ------------------------------------------------------------------ microphone

class Resampler {  // anti-alias filter + linear interpolation down to 16 kHz
  constructor(rate) {
    this.ratio = rate / RATE;
    const taps = 31, cut = 7600 / rate;
    this.h = Array.from({ length: taps }, (_, k) => {
      const n = k - (taps - 1) / 2, sinc = n === 0 ? 2 * cut : Math.sin(2 * Math.PI * cut * n) / (Math.PI * n);
      return sinc * (0.54 - 0.46 * Math.cos(2 * Math.PI * k / (taps - 1)));
    });
    this.hist = new Float32Array(taps - 1);
    this.tail = new Float32Array(0);
    this.t = 0;
  }

  process(input) {
    const ext = new Float32Array(this.hist.length + input.length);
    ext.set(this.hist); ext.set(input, this.hist.length);
    const data = new Float32Array(this.tail.length + input.length);
    data.set(this.tail);
    for (let i = 0; i < input.length; i++) {
      let s = 0;
      for (let k = 0; k < this.h.length; k++) s += this.h[k] * ext[i + k];
      data[this.tail.length + i] = s;
    }
    this.hist = ext.slice(ext.length - this.hist.length);
    const out = [];
    while (this.t + 1 < data.length) {
      const i = Math.floor(this.t), f = this.t - i;
      out.push(data[i] * (1 - f) + data[i + 1] * f);
      this.t += this.ratio;
    }
    const drop = Math.floor(this.t);
    this.tail = data.slice(drop);
    this.t -= drop;
    return Float32Array.from(out);
  }
}

const mic = {
  async start(onAudio) {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
    let source;
    try {
      this.ctx = new AudioContext({ sampleRate: RATE });
      source = this.ctx.createMediaStreamSource(this.stream);
    } catch {
      if (this.ctx) this.ctx.close();
      this.ctx = new AudioContext();
      source = this.ctx.createMediaStreamSource(this.stream);
    }
    const resampler = this.ctx.sampleRate === RATE ? null : new Resampler(this.ctx.sampleRate);
    this.node = this.ctx.createScriptProcessor(2048, 1, 1);
    this.node.onaudioprocess = e => {
      const block = new Float32Array(e.inputBuffer.getChannelData(0));
      onAudio(resampler ? resampler.process(block) : block);
    };
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    source.connect(this.node); this.node.connect(mute); mute.connect(this.ctx.destination);
  },
  stop() {
    if (this.node) this.node.disconnect();
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    if (this.ctx) this.ctx.close();
    this.node = this.stream = this.ctx = null;
  },
  get on() { return !!this.ctx; },
};

// ------------------------------------------------------------------ hero and grid

async function hero() {
  const rows = [];
  for (const phrase of ["bird_house", "dog_off"]) {
    for (let i = 1; i <= 3; i++) rows.push({ phrase, i, wave: await sample(`${phrase}_${i}`) });
  }
  rows.forEach(r => (r.embedding = model.embed(r.wave)));
  $("#print-rows").innerHTML = rows.map((r, k) => `
    ${k === 3 ? '<div class="group-gap"></div>' : ""}
    <div class="print-row">
      <span class="who"><b>${SAMPLES[r.phrase]}</b> ${SPEAKERS[r.i - 1]}</span>
      <div class="bar">${barCells(r.embedding)}</div>
      <button class="play" aria-label="Play ${SAMPLES[r.phrase]}, speaker ${SPEAKERS[r.i - 1]}" data-k="${k}">
        <svg viewBox="0 0 10 10"><path d="M2 1 L9 5 L2 9 Z"/></svg></button>
    </div>`).join("");
  $("#print-rows").querySelectorAll(".play").forEach(b => (b.onclick = () => play(rows[b.dataset.k].wave)));
  const same = [], diff = [];
  for (let a = 0; a < 6; a++) for (let b = a + 1; b < 6; b++) {
    (rows[a].phrase === rows[b].phrase ? same : diff).push(dot(rows[a].embedding, rows[b].embedding));
  }
  const avg = v => (v.reduce((s, x) => s + x, 0) / v.length).toFixed(2);
  $("#prints-foot").innerHTML = `<span>same phrase <b>${avg(same)}</b></span><span>different phrase <b>${avg(diff)}</b></span>
    <span>similarity, 1.00 = identical</span>`;
}

async function grid() {
  const items = [];
  for (const phrase of Object.keys(SAMPLES)) {
    for (let i = 1; i <= 3; i++) items.push({ label: `${SAMPLES[phrase]} ${SPEAKERS[i - 1]}`, phrase, emb: model.embed(await sample(`${phrase}_${i}`)) });
  }
  let html = `<div></div>` + items.map(it => `<div class="top">${it.label.split(" ").pop()}</div>`).join("");
  const same = [], diff = [];
  items.forEach((a, r) => {
    html += `<div class="lab">${a.label}</div>`;
    items.forEach((b, c) => {
      const s = dot(a.emb, b.emb);
      if (r < c) (a.phrase === b.phrase ? same : diff).push(s);
      const pct = Math.round(Math.max(0, Math.min(1, s)) * 100);
      const ink = pct > 55 ? "var(--bg)" : "var(--muted)";
      html += `<div class="cell" style="background:color-mix(in srgb, var(--accent) ${pct}%, var(--surface-2));color:${ink}"
                 title="${a.label} vs ${b.label}: ${s.toFixed(2)}">${r === c ? "" : s.toFixed(2).replace(/^0/, "")}</div>`;
    });
  });
  $("#sim").innerHTML = html;
  const min = v => Math.min(...v).toFixed(2), max = v => Math.max(...v).toFixed(2);
  $("#sim-summary").innerHTML = `Same phrase, different people: <b>${min(same)}–${max(same)}</b>.
    Different phrases: <b>${min(diff)}–${max(diff)}</b>. Single recordings overlap, so a command is the average
    of several. Matched against those averages, the test clips in the demo score well apart.`;
}

// ------------------------------------------------------------------ commands

function renderCommands(recording) {
  $("#no-cmds").hidden = commands.length > 0 || !!recording;
  const list = recording && !commands.includes(recording) ? [...commands, recording] : commands;
  $("#cmds").innerHTML = list.map((c, k) => `
    <li class="cmd">
      <div class="cmd-top"><span class="cmd-name">${c.name}</span>
        <span class="cmd-meta">${c.examples.length}/3${c.examples.length === 3 ? ` · needs ${model.threshold(3, avgSeconds(c)).toFixed(2)}` : ""}</span>
        ${c === recording ? "" : `<button class="cmd-x" data-k="${k}" aria-label="Remove ${c.name}">×</button>`}</div>
      <div class="takes">${[0, 1, 2].map(i => c.examples[i] ? `<canvas class="take" data-c="${k}" data-i="${i}"></canvas>` : `<span class="take empty"></span>`).join("")}</div>
    </li>`).join("");
  $("#cmds").querySelectorAll("canvas").forEach(cv => {
    const ex = list[cv.dataset.c].examples[cv.dataset.i];
    drawWave(cv, ex.wave);
    cv.title = "Click to play";
    cv.onclick = () => play(ex.wave);
  });
  $("#cmds").querySelectorAll(".cmd-x").forEach(b => (b.onclick = () => { commands.splice(b.dataset.k, 1); renderCommands(); updateListen(); }));
  updateListen();
}

const avgSeconds = c => c.examples.reduce((s, e) => s + e.seconds, 0) / c.examples.length;

function addExample(cmd, wave) {
  const t0 = performance.now();
  cmd.examples.push({ embedding: model.embed(wave), seconds: speechSeconds(wave, model.audio), wave });
  $("#timing").textContent = `last embedding: ${Math.round(performance.now() - t0)} ms`;
}

function similarWarning(cmd) {
  const proto = c => normalize(c.examples[0].embedding.map((_, i) => c.examples.reduce((s, e) => s + e.embedding[i], 0)));
  const bar = model.threshold(1, avgSeconds(cmd));
  const close = commands.filter(o => o !== cmd && o.examples.length && dot(proto(o), proto(cmd)) >= bar);
  return close.length ? `Sounds a lot like "${close[0].name}"; they may get mixed up.` : "";
}

async function record(name) {
  if (listening) await toggleListen();
  const hint = $("#rec-hint");
  const cmd = commands.find(c => c.name === name) || { name, examples: [] };
  if (cmd.examples.length >= 3) cmd.examples = [];
  const detector = new VoiceDetector();
  $("#record-btn").disabled = true;
  hint.className = "hint live";
  hint.textContent = "Starting microphone…";
  let timer;
  try {
    await new Promise((resolve, reject) => {
      const prompt = () => {
        hint.textContent = `Say "${name}" (${cmd.examples.length + 1} of 3)`;
        clearTimeout(timer);
        timer = setTimeout(() => reject(new Error("Didn't hear anything. Check your microphone and try again.")), 9000);
      };
      renderCommands(cmd);
      mic.start(block => {
        for (const u of detector.feed(block)) {
          const [ok, reason] = u.tooLong ? [false, "too long"] : isSpeech(u.audio);
          if (!ok) { hint.textContent = `Didn't catch that (${reason}). Say "${name}" again.`; continue; }
          addExample(cmd, u.audio);
          renderCommands(cmd);
          if (cmd.examples.length === 3) return resolve();
          prompt();
        }
      }).then(() => setTimeout(prompt, 300), reject);
    });
    if (!commands.includes(cmd)) commands.push(cmd);
    hint.className = "hint";
    hint.textContent = similarWarning(cmd) || `Added "${name}". Add another, or start listening.`;
    $("#cmd-name").value = "";
  } catch (err) {
    hint.className = "hint";
    hint.textContent = err.name === "NotAllowedError" ? "Microphone access was blocked." : err.message;
  } finally {
    clearTimeout(timer);
    mic.stop();
    $("#record-btn").disabled = false;
    renderCommands();
  }
}

async function loadSamples() {
  $("#load-samples").textContent = "loading…";
  for (const [key, label] of Object.entries(SAMPLES)) {
    if (commands.some(c => c.name === label)) continue;
    const cmd = { name: label, examples: [] };
    for (let i = 1; i <= 3; i++) addExample(cmd, await sample(`${key}_${i}`));
    commands.push(cmd);
  }
  renderCommands();
  $("#load-samples").textContent = "sample voices loaded";
  $("#rec-hint").textContent = "Each sample command was recorded by three different people (A, B, C).";
  const tests = [...Object.entries(SAMPLES).map(([k, l]) => [`${k}_4`, `"${l}", speaker D`]), ["unknown_1", `"backward tree", not a command`]];
  $("#tests").hidden = false;
  $("#tests").innerHTML = `<span class="hint" style="margin:6px 4px 0 0">Play a test clip:</span>` +
    tests.map(([file, label]) => `<button class="btn" data-file="${file}">${label}</button>`).join("");
  $("#tests").querySelectorAll("button").forEach(b => (b.onclick = async () => {
    const wave = await sample(b.dataset.file);
    play(wave);
    const detector = new VoiceDetector();
    const quiet = Float32Array.from({ length: RATE }, () => (Math.random() - 0.5) * 0.002);
    const found = [...detector.feed(quiet), ...detector.feed(wave), ...detector.feed(quiet), ...detector.flush()];
    found.forEach(u => hear(u, "test"));
  }));
}

// ------------------------------------------------------------------ listening

let listening = false, detector, meterTimer;

function updateListen() {
  $("#listen-btn").disabled = !commands.length;
  $("#listen-btn").textContent = listening ? "Stop listening" : "Start listening";
}

function hear(u, when) {
  let entry;
  if (u.tooLong) entry = { reason: "too long for a command" };
  else {
    const [ok, reason] = isSpeech(u.audio);
    if (!ok) entry = { reason: `ignored: ${reason}` };
    else {
      const t0 = performance.now();
      entry = match(commands, model.embed(u.audio), speechSeconds(u.audio, model.audio), model);
      $("#timing").textContent = `last embedding: ${Math.round(performance.now() - t0)} ms`;
    }
  }
  const li = document.createElement("li");
  const score = entry.score === undefined ? "" : `
    <div class="score ${entry.name ? "hit" : ""}"><i style="width:${Math.max(0, entry.score) * 100}%"></i><b style="left:${entry.threshold * 100}%"></b></div>
    <div class="why">score ${entry.score.toFixed(2)} · needs ${entry.threshold.toFixed(2)}${entry.name ? "" : ` · closest "${entry.best}"`}${entry.name || !entry.reason || entry.reason === "not similar enough" ? "" : ` · ${entry.reason}`}</div>`;
  li.innerHTML = `<time>${when === "test" ? "test" : fmtTime()}</time><div>
    <div class="result ${entry.name ? "hit" : "miss"}">${entry.name || (entry.score === undefined ? entry.reason : "unknown")}</div>${score}</div>`;
  $("#log").prepend(li);
}

async function toggleListen() {
  if (listening) {
    listening = false;
    mic.stop();
    clearInterval(meterTimer);
    $("#meter span").textContent = "microphone off";
    $("#meter .lvl").style.width = "0";
    $("#listen-state").textContent = "";
    updateListen();
    return;
  }
  detector = new VoiceDetector();
  try {
    await mic.start(block => detector.feed(block).forEach(u => hear(u, "now")));
  } catch (err) {
    $("#listen-state").textContent = err.name === "NotAllowedError" ? "mic blocked" : "no mic";
    return;
  }
  listening = true;
  updateListen();
  const pos = db => Math.max(0, Math.min(100, (db + 70) / 60 * 100));
  meterTimer = setInterval(() => {
    $("#meter .lvl").style.width = pos(detector.level) + "%";
    $("#meter .lvl").classList.toggle("speech", detector.inSpeech);
    if (detector.floor !== null) $("#meter .floor").style.left = pos(detector.floor) + "%";
    $("#meter span").textContent = detector.floor === null ? "measuring room noise…" : detector.inSpeech ? "hearing speech" : "listening";
  }, 60);
  $("#listen-state").textContent = "say a command";
}

// ------------------------------------------------------------------ start

document.querySelectorAll("[data-copy], [data-copy-pre]").forEach(b => (b.onclick = () => {
  const text = b.dataset.copy || b.closest(".code").querySelector("pre").innerText.replace(/\s+#.*$/gm, "");
  navigator.clipboard.writeText(text).then(() => { b.textContent = "copied"; setTimeout(() => (b.textContent = "copy"), 1200); });
}));

$("#add-form").onsubmit = e => {
  e.preventDefault();
  const name = $("#cmd-name").value.trim() || `command ${commands.length + 1}`;
  record(name);
};
$("#load-samples").onclick = loadSamples;
$("#listen-btn").onclick = toggleListen;

model = await Model.load("./");
await hero();
await grid();
renderCommands();
