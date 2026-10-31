const $ = id => document.getElementById(id);
const DATA = window.DEMO_DATA || {};
const CAP = 100;
const pages = ["overview", "m1", "m2", "m3", "results"];
const titles = {overview: "Overview", m1: "M1 · LSTM Heuristic Routing", m2: "M2 · Forecast + Optimize", m3: "M3 · Distributed SDN", results: "Results & Comparison"};
const fmt = (v, d = 3) => (v == null || isNaN(v)) ? "--" : Number(v).toFixed(d);

function go(page) {
  pages.forEach(p => $("page-" + p).classList.toggle("active", p === page));
  document.querySelectorAll(".nav").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  $("page-title").textContent = titles[page];
  window.scrollTo({top: 0, behavior: "smooth"});
  stopPlay();
  refreshAll();
}
document.querySelectorAll("[data-page]").forEach(b => b.addEventListener("click", () => go(b.dataset.page)));
document.querySelectorAll("[data-goto]").forEach(b => b.addEventListener("click", () => go(b.dataset.goto)));
$("reset-all").addEventListener("click", () => location.reload());

function bindRange(id, out, unit = "", f) {
  const el = $(id), o = $(out);
  const update = () => o.textContent = (f ? f(el.value) : el.value) + unit;
  el.addEventListener("input", update); update();
}

// ---------------------------------------------------------------- traffic series
function synthetic() {
  let s = 7; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const out = [];
  for (let i = 0; i < 72; i++) {
    let d = 50;
    if (i >= 20 && i < 32) d = 50 + 60 * (i - 20) / 12;
    else if (i >= 32 && i < 52) d = 110;
    else if (i >= 52 && i < 62) d = 110 - 60 * (i - 52) / 10;
    out.push(Math.max(5, d + (rnd() - .5) * 10));
  }
  return out;
}
const SERIES = {abilene: DATA.abilene ? DATA.abilene.values : null, synthetic: synthetic()};
let src = SERIES.abilene ? "abilene" : "synthetic";
function buildSource() {
  const sel = $("src"); sel.innerHTML = "";
  if (SERIES.abilene) sel.add(new Option("Abilene replay (real, scaled)", "abilene"));
  sel.add(new Option("Synthetic spike", "synthetic"));
  sel.value = src;
  sel.addEventListener("change", () => { src = sel.value; refreshAll(); });
}
function srcNote() {
  const t = src === "abilene"
    ? DATA.abilene.note + " Path capacity 100 Mbps."
    : "Synthetic traffic: a ramp to a spike with noise." + (SERIES.abilene ? "" : " Run scripts/export_demo_data.py on the PC that has the Abilene files to switch to real traffic.");
  $("src-note").textContent = t;
}
const D = () => SERIES[src];

// ---------------------------------------------------------------- model of the two-path network
function forecast(kind, hist, now) {
  if (kind === "oracle") return now;
  if (!hist.length) return now;
  const l = hist[hist.length - 1];
  if (kind === "trend") return hist.length > 1 ? Math.max(0, l + (l - hist[hist.length - 2])) : l;
  if (kind === "ma") { const h = hist.slice(-3); return h.reduce((a, b) => a + b, 0) / h.length; }
  return l;
}
const share = (f, a, b) => f <= 0 ? 1 : Math.max(0, Math.min(f, (b + f - a) / 2)) / f;   // share of new demand on A that equalises both paths
const mluOf = (x, d, a, b) => Math.max(a + x * d, b + (1 - x) * d) / CAP;
const stat = m => ({mean: m.reduce((p, c) => p + c, 0) / m.length, peak: Math.max(...m), over: m.filter(v => v > 1).length});

function simStatic(d, a, b) { return d.map(v => mluOf(1, v, a, b)); }
// forecast-informed min-max split every step. Decision uses the forecast, load uses the REAL demand.
function simLP(d, a, b, kind) {
  const m = [], xs = [], fs = [];
  d.forEach((v, t) => {
    const f = forecast(kind, d.slice(0, t), v), x = share(f, a, b);
    fs.push(f); xs.push(x); m.push(mluOf(x, v, a, b));
  });
  return {m, xs, fs};
}
// M1: forecast + hysteresis heuristic (reroute only if predicted gain >= threshold, with a 2-step cooldown)
function simM1(d, a, b, kind, thr) {
  const m = [], xs = [], fs = [], ch = []; let x = 1, last = -99, n = 0;
  d.forEach((v, t) => {
    const f = forecast(kind, d.slice(0, t), v), cand = share(f, a, b);
    const keep = mluOf(x, f, a, b), nw = mluOf(cand, f, a, b);
    let changed = false;
    if (keep > nw && (keep - nw) / keep * 100 >= thr && t - last >= 2) { x = cand; last = t; n++; changed = true; }
    fs.push(f); xs.push(x); ch.push(changed); m.push(mluOf(x, v, a, b));
  });
  return {m, xs, fs, ch, reroutes: n};
}

// ---------------------------------------------------------------- drawing
function ctxFor(id, h) {
  const c = $(id); if (!c) return null;
  c.style.width = "100%";
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth || 600;
  c.width = w * dpr; c.height = h * dpr; c.style.height = h + "px";
  const ctx = c.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
  return {ctx, w, h};
}
function drawBars(id, labels, values, colors, o = {}) {
  const k = ctxFor(id, o.h || 190); if (!k) return;
  const {ctx, w, h} = k, max = o.max || 125, ticks = o.ticks || [0, 25, 50, 75, 100, 125], f = o.f || (v => v.toFixed(1) + "%");
  const pad = {l: 42, r: 15, t: 18, b: 40}, cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;
  ctx.strokeStyle = "#e6edf4"; ctx.fillStyle = "#8090a3"; ctx.font = "10px system-ui"; ctx.textAlign = "left";
  ticks.forEach(v => { const y = pad.t + ch - v / max * ch; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.fillText((o.tickf || (t => t + "%"))(v), 4, y + 3); });
  const g = cw / labels.length;
  labels.forEach((l, i) => {
    const bh = values[i] / max * ch, x = pad.l + i * g + g * .23, y = pad.t + ch - bh;
    ctx.fillStyle = colors[i]; ctx.beginPath(); ctx.roundRect(x, y, g * .54, bh, 5); ctx.fill();
    ctx.fillStyle = "#52657c"; ctx.textAlign = "center"; ctx.fillText(l, x + g * .27, h - 13); ctx.fillText(f(values[i]), x + g * .27, y - 5);
  });
}
function drawLines(id, series, tick) {
  const k = ctxFor(id, 230); if (!k) return;
  const {ctx, w, h} = k, pad = {l: 42, r: 15, t: 30, b: 28}, cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;
  const n = series[0].values.length;
  const top = Math.max(130, Math.ceil(Math.max(...series.flatMap(s => s.values)) * 100 / 10) * 10 + 10);
  const X = i => pad.l + i / (n - 1) * cw, Y = v => pad.t + ch - v * 100 / top * ch;
  ctx.font = "10px system-ui"; ctx.strokeStyle = "#e6edf4"; ctx.fillStyle = "#8090a3"; ctx.textAlign = "left";
  for (let v = 0; v <= top; v += top > 160 ? 50 : 25) { ctx.beginPath(); ctx.moveTo(pad.l, Y(v / 100)); ctx.lineTo(w - pad.r, Y(v / 100)); ctx.stroke(); ctx.fillText(v + "%", 4, Y(v / 100) + 3); }
  ctx.setLineDash([5, 4]); ctx.strokeStyle = "#c0392b"; ctx.beginPath(); ctx.moveTo(pad.l, Y(1)); ctx.lineTo(w - pad.r, Y(1)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#c0392b"; ctx.fillText("capacity", w - pad.r - 44, Y(1) - 4);
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash || []); ctx.beginPath();
    s.values.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))); ctx.stroke();
  });
  ctx.setLineDash([]); ctx.lineWidth = 1;
  if (tick != null) { ctx.strokeStyle = "#8090a3"; ctx.beginPath(); ctx.moveTo(X(tick), pad.t); ctx.lineTo(X(tick), pad.t + ch); ctx.stroke(); }
  let lx = pad.l; ctx.textAlign = "left";
  series.forEach(s => { ctx.fillStyle = s.color; ctx.fillRect(lx, 10, 14, 3); ctx.fillStyle = "#52657c"; ctx.fillText(s.name, lx + 19, 15); lx += 28 + ctx.measureText(s.name).width; });
  ctx.fillStyle = "#8090a3"; ctx.textAlign = "center"; ctx.fillText("time step (5 min)", w / 2, h - 6);
}
const setBar = (id, v) => $(id).style.width = Math.max(0, Math.min(130, v)) + "%";

// ---------------------------------------------------------------- M1
bindRange("m1-a", "m1-a-out", " Mbps"); bindRange("m1-b", "m1-b-out", " Mbps"); bindRange("m1-threshold", "m1-threshold-out", "%");
bindRange("m1-tick", "m1-tick-out", "");
function runM1() {
  const d = D(), a = +$("m1-a").value, b = +$("m1-b").value, kind = $("m1-fc").value, thr = +$("m1-threshold").value;
  srcNote();
  const t = Math.min(+$("m1-tick").value, d.length - 1);
  const st = simStatic(d, a, b), p = simM1(d, a, b, kind, thr);
  const s0 = stat(st), s1 = stat(p.m);
  const x = p.xs[t], A = a + x * d[t], B = b + (1 - x) * d[t];
  $("m1-forecast").textContent = p.fs[t].toFixed(0) + " Mbps";
  $("m1-real").textContent = "Real demand " + d[t].toFixed(0) + " Mbps";
  $("m1-decision").textContent = p.ch[t] ? "REROUTE" : "KEEP ROUTE";
  $("m1-decision-note").textContent = p.ch[t] ? "Predicted gain passed the threshold" : (x === 1 ? "All traffic on shortest path A" : "Keeping the current split");
  $("m1-mlu").textContent = (p.m[t] * 100).toFixed(1) + "%";
  $("m1-change").textContent = "Static " + (st[t] * 100).toFixed(1) + "% → policy " + (p.m[t] * 100).toFixed(1) + "%";
  [["a", A], ["b", B]].forEach(([k, v]) => {
    $("m1-util-" + k).textContent = (v / CAP * 100).toFixed(0) + "%";
    $("m1-edge-" + k).textContent = "Path " + k.toUpperCase() + " · " + (v / CAP * 100).toFixed(0) + "%";
    setBar("m1-bar-" + k, v); $("m1-path-" + k).style.strokeWidth = 5 + Math.min(8, v / CAP * 8);
  });
  drawLines("m1-chart", [
    {name: "Static shortest path", values: st, color: "#c0392b"},
    {name: "Forecast + hysteresis (M1)", values: p.m, color: "#3d87d5"}], t);
  const rec = (DATA.summary || []).filter(r => ["static_shortest", "lstm_hysteresis"].includes(r.method));
  const rs = rec.find(r => r.method === "static_shortest"), rl = rec.find(r => r.method === "lstm_hysteresis");
  $("m1-summary").innerHTML = `<b>This replay (${d.length} steps):</b> mean MLU static ${fmt(s0.mean)} → policy ${fmt(s1.mean)}; peak ${fmt(s0.peak)} → ${fmt(s1.peak)}; steps above 100%: ${s0.over} → ${s1.over}; reroutes: ${p.reroutes}.` +
    (rs && rl ? `<br><b>Measured in Mininet (10 seeds):</b> static ${fmt(rs.mean)} vs LSTM + hysteresis ${fmt(rl.mean)}. The hysteresis policy helps a little; the LP in M2 helps much more.` : "");
  const lines = [
    `Step ${t}: real demand ${d[t].toFixed(0)} Mbps; forecast ${p.fs[t].toFixed(0)} Mbps (${$("m1-fc").selectedOptions[0].text}).`,
    `Static routing puts everything on Path A: Path A ${(a + d[t]).toFixed(0)} Mbps, Path B ${b} Mbps, max utilization ${(st[t] * 100).toFixed(1)}%.`,
    p.ch[t] ? `Hysteresis check passed (predicted gain ≥ ${thr}%): the policy moved to a ${(x * 100).toFixed(0)}% / ${((1 - x) * 100).toFixed(0)}% split.` : `No change this step: the predicted gain was below ${thr}% or the cooldown was active.`,
    `Result: Path A ${A.toFixed(0)} Mbps, Path B ${B.toFixed(0)} Mbps, max utilization ${(p.m[t] * 100).toFixed(1)}%.`];
  $("m1-log").innerHTML = lines.map((l, i) => `<div class="log-line"><b>0${i + 1}</b>${l}</div>`).join("");
}
["m1-a", "m1-b", "m1-fc", "m1-threshold", "m1-tick"].forEach(id => $(id).addEventListener("input", runM1));

// ---------------------------------------------------------------- M2
const PRED = {};
(DATA.prediction || []).forEach(r => PRED[r.model] = r);
const FALLBACK = {Transformer: [.0120, .0244, 6.1535], TCN: [.0121, .0247, 6.1365], LSTM: [.0134, .0259, 6.6974], Persistence: [.0124, .0251, 6.3373], "Moving average": [.0155, .0298, 7.4952]};
const KEY = {Transformer: "transformer", TCN: "tcn", LSTM: "lstm", Persistence: "persistence", "Moving average": "moving_average"};
let model = "Transformer";
const simKind = () => model === "Moving average" ? "ma" : "persistence";
function metrics(name) { const r = PRED[KEY[name]]; return r ? [r.mae, r.rmse, r.smape] : FALLBACK[name]; }
document.querySelectorAll(".model-choice").forEach(b => b.addEventListener("click", () => {
  model = b.dataset.model; document.querySelectorAll(".model-choice").forEach(x => x.classList.toggle("selected", x === b)); runM2();
}));
bindRange("m2-a", "m2-a-out", " Mbps"); bindRange("m2-b", "m2-b-out", " Mbps"); bindRange("m2-tick", "m2-tick-out", "");
function runM2() {
  const d = D(), a = +$("m2-a").value, b = +$("m2-b").value, kind = simKind();
  const t = Math.min(+$("m2-tick").value, d.length - 1);
  const [mae, rmse, smape] = metrics(model);
  $("m2-mae").textContent = mae.toFixed(4); $("m2-rmse").textContent = rmse.toFixed(4); $("m2-smape").textContent = smape.toFixed(2) + "%";
  const persMae = metrics("Persistence")[0];
  $("m2-simnote").innerHTML = ["Persistence", "Moving average"].includes(model)
    ? `The simulation below uses this forecaster on the replayed traffic.`
    : `Trained ${model} checkpoints cannot run in the browser, so the simulation uses a persistence forecast as stand-in. Recorded MAE: ${model} ${mae.toFixed(4)} vs persistence ${persMae.toFixed(4)}.`;
  const st = simStatic(d, a, b), lp = simLP(d, a, b, kind);
  const rows = [["Static shortest path", st], ["Persistence + min-max", simLP(d, a, b, "persistence").m], ["Linear trend + min-max", simLP(d, a, b, "trend").m],
    ["Moving average + min-max", simLP(d, a, b, "ma").m], ["Perfect forecast + min-max", simLP(d, a, b, "oracle").m]];
  const sel = {persistence: 1, ma: 3}[kind];
  $("m2-table").innerHTML = "<thead><tr><th>Policy (this replay)</th><th>Mean MLU</th><th>Peak MLU</th><th>Steps above 100%</th></tr></thead><tbody>" +
    rows.map(([n, m], i) => { const s = stat(m); return `<tr class="${i === sel ? "highlight" : ""}"><td>${n}</td><td>${fmt(s.mean)}</td><td>${fmt(s.peak)}</td><td>${s.over}</td></tr>`; }).join("") + "</tbody>";
  const x = lp.xs[t], toA = x * d[t], toB = d[t] - toA, A = a + toA, B = b + toB;
  $("m2-static-a").textContent = d[t].toFixed(0) + " Mbps";
  $("m2-opt-a").textContent = toA.toFixed(0) + " Mbps"; $("m2-opt-b").textContent = toB.toFixed(0) + " Mbps";
  $("m2-static-mlu").textContent = (st[t] * 100).toFixed(1) + "% MLU"; $("m2-opt-mlu").textContent = (lp.m[t] * 100).toFixed(1) + "% MLU";
  $("m2-static-bar-a").style.width = "100%"; $("m2-static-bar-b").style.width = "0%";
  $("m2-opt-bar-a").style.width = (toA / (d[t] || 1) * 100) + "%"; $("m2-opt-bar-b").style.width = (toB / (d[t] || 1) * 100) + "%";
  drawLines("m2-chart", [
    {name: "Static", values: st, color: "#c0392b"},
    {name: "Forecast + min-max", values: lp.m, color: "#16a57c"},
    {name: "Perfect forecast", values: rows[4][1], color: "#7f8c9a", dash: [3, 3]}], t);
  const means = rows.slice(1).map(r => stat(r[1]).mean), spread = (Math.max(...means) - Math.min(...means));
  const rec = n => (DATA.summary || []).find(r => r.method === n);
  const tr = rec("transformer_minmax_lp"), pe = rec("persistence_minmax_lp"), sa = rec("static_shortest");
  $("m2-explain").innerHTML = `<b>This replay:</b> all four min-max variants are within ${fmt(spread)} of each other in mean MLU, while static is ${fmt(stat(st).mean)}. The split does the work; the forecaster barely matters. This is a small two-path example with a strong spike, so the gain is larger than on the full 12-node network.` +
    (tr && pe && sa ? `<br><b>Measured in Mininet (10 seeds):</b> static ${fmt(sa.mean)}, persistence + LP ${fmt(pe.mean)}, Transformer + LP ${fmt(tr.mean)}.` : "");
}
$("run-m2").addEventListener("click", () => togglePlay("m2"));
["m2-a", "m2-b", "m2-tick"].forEach(id => $(id).addEventListener("input", runM2));

// play buttons advance the time step
let timer = null, playing = null;
function stopPlay() { if (timer) clearInterval(timer); timer = null; if (playing) { $("run-" + playing).textContent = "▶ Play replay"; } playing = null; }
function togglePlay(pg) {
  if (playing === pg) { stopPlay(); return; }
  stopPlay(); playing = pg; $("run-" + pg).textContent = "⏸ Pause";
  const tk = $(pg + "-tick"); if (+tk.value >= +tk.max) tk.value = 0;
  timer = setInterval(() => {
    tk.value = +tk.value + 1; tk.dispatchEvent(new Event("input"));
    if (+tk.value >= +tk.max) stopPlay();
  }, 160);
}
$("run-m1").addEventListener("click", () => togglePlay("m1"));

// ---------------------------------------------------------------- M3
const AGES = [0, 5, 10, 20, 40];
const meanOf = (arr, name) => { const r = (arr || []).find(x => x.setting === name); return r ? r.mean : null; };
function runM3() {
  const age = AGES[+$("m3-delay").value];
  $("m3-delay-out").textContent = age + " s";
  $("sync-label").textContent = age + " s delay"; $("sync-label-2").textContent = age + " s delay";
  $("c1-age").textContent = "0 s"; $("c2-age").textContent = age + " s"; $("c3-age").textContent = age + " s";
  ["c2-state", "c3-state"].forEach(id => { $(id).classList.toggle("stale", age >= 20); $(id).textContent = age >= 20 ? "Stale view" : "Synchronized"; });
  const k = meanOf(DATA.stale_known, "dist_s" + age), k0 = meanOf(DATA.stale_known, "dist_s0"), st = meanOf(DATA.stale_known, "static"), e = meanOf(DATA.stale_est, "dist_s" + age);
  $("m3-quality").textContent = fmt(k);
  $("m3-quality-bar").style.width = (k && st ? k / st * 100 : 0) + "%";
  $("m3-age").textContent = age + " s";
  $("m3-risk").textContent = k != null && k0 != null ? (k - k0 >= 0 ? "+" : "") + (k - k0).toFixed(3) : "--";
  $("m3-mode").textContent = fmt(e);
}
$("m3-delay").addEventListener("input", runM3);

function renderOnos() {
  const o = DATA.onos; if (!o || !o.rows || !o.rows.length) return;
  const names = {A: "Single node", B: "3-node cluster"};
  $("onos-body").innerHTML = o.rows.map(r => `<tr><td>${names[r.cond] || "Cluster " + r.cond}</td><td>${r.delay} ms</td><td>${r.ok}%</td><td>${r.reroute_med.toFixed(1)} ms</td><td>${r.write_vis.toFixed(1)} ms</td></tr>`).join("");
  $("onos-fit").innerHTML = o.fit ? `<b>Measured trend:</b> write visibility ≈ ${o.fit.intercept.toFixed(1)} + ${o.fit.slope.toFixed(2)}·d ms, where d is the added delay per hop (R² ${o.fit.r2.toFixed(3)}). It grows in a straight line with the delay between nodes.` : "";
}

// ---------------------------------------------------------------- results
const NAMES = {static_shortest: "Static", ecmp: "ECMP", lstm_hysteresis: "LSTM", persistence_minmax_lp: "Persist+LP", tcn_minmax_lp: "TCN+LP", transformer_minmax_lp: "Trans+LP"};
function drawResults() {
  const rows = DATA.summary && DATA.summary.length ? Object.keys(NAMES).map(k => [NAMES[k], (DATA.summary.find(r => r.method === k) || {}).mean * 100]) :
    [["Static", 50], ["ECMP", 49.9], ["LSTM", 48.3], ["Persist+LP", 42.7], ["TCN+LP", 42.9], ["Trans+LP", 42.6]];
  drawBars("results-chart", rows.map(r => r[0]), rows.map(r => r[1]), ["#93a4b8", "#93a4b8", "#3d87d5", "#16a57c", "#16a57c", "#16a57c"], {h: 220});
  const b = DATA.bounds || [];
  if (b.length) {
    const order = ["mcf_optimal", "kpath_3", "kpath_3_int", "oblivious_opt", "shortest_path", "vlb"], nm = {mcf_optimal: "Optimal", kpath_3: "K=3 paths", kpath_3_int: "K=3 rounded", oblivious_opt: "Oblivious", shortest_path: "Shortest path", vlb: "VLB"};
    const rr = order.map(k => [nm[k], (b.find(x => x.policy === k) || {}).ratio]).filter(r => r[1] != null);
    drawBars("bounds-chart", rr.map(r => r[0]), rr.map(r => r[1]), rr.map(r => r[0] === "Optimal" ? "#16a57c" : r[0].startsWith("K=3") ? "#3d87d5" : "#93a4b8"),
      {max: 4.2, ticks: [0, 1, 2, 3, 4], tickf: t => t + "×", f: v => v.toFixed(2) + "×", h: 220});
  }
  const est = DATA.estimated || [], t = $("est-table");
  if (est.length) {
    const nm = {mean_mlu: "Mean MLU", p95_mlu: "p95 MLU", peak_mlu: "Peak MLU"};
    t.innerHTML = "<thead><tr><th>Metric</th><th>Static</th><th>Transformer + LP (estimated demand)</th><th>Change</th></tr></thead><tbody>" +
      est.map(r => `<tr><td>${nm[r.metric]}</td><td>${fmt(r.base)}</td><td>${fmt(r.est)}</td><td>${r.pct > 0 ? "+" : ""}${r.pct.toFixed(1)}%</td></tr>`).join("") + "</tbody>";
    const g = DATA.rounding;
    $("est-note").innerHTML = "<b>Takeaway:</b> with demand estimated from flow counters, the mean improves by about half as much as with known demand, but the worst moments get worse." +
      (g ? ` Sending each flow on a single path costs about ${Math.round(g.rounding_mean)}% over the ideal split, and the load actually carried is about ${Math.round(g.real0_mean)}% above what the plan promised.` : "");
  } else t.innerHTML = "<tbody><tr><td>est_paired_alby.csv not found. Run scripts/export_demo_data.py.</td></tr></tbody>";
}

function refreshAll() {
  srcNote(); runM1(); runM2(); runM3();
  if ($("page-results").classList.contains("active")) drawResults();
}
buildSource(); renderOnos(); drawResults(); refreshAll();
window.addEventListener("resize", refreshAll);
