// ROWII Monitor -- live.html replay engine. Reads the embedded #live-data JSON
// (written by scripts/build_live_replay.py) and drives everything: the state
// ribbon/scrubber, the log-mel pan, the p-value chart, the sensor-ring pulsing,
// the KPI strip, the alarm feed, and the Operator/Engineering view toggle. No
// network requests -- every byte the page needs is already inline.
(function () {
  "use strict";

  var DATA = JSON.parse(document.getElementById("live-data").textContent);

  var STATE_COLORS = {
    turbine: "#2563a8",
    pump: "#7c4dbc",
    "phase-shifter": "#1d8a70",
    standstill: "#6b7684",
    transition: "#c07f10",
    invalid: "#aab2bc",
    unknown: "#aab2bc",
  };
  function stateColor(name) { return STATE_COLORS[name] || STATE_COLORS.unknown; }
  function stateLabel(name) {
    if (!name) return "Unknown";
    if (name === "invalid") return "No usable data";
    return name.charAt(0).toUpperCase() + name.slice(1).replace("-", "-");
  }

  // -------------------------------------------------------------- decoding
  function b64ToFloat32(b64, nRows, nCols) {
    var bin = atob(b64);
    var buf = new ArrayBuffer(bin.length);
    var view = new Uint8Array(buf);
    for (var i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
    var arr = new Float32Array(buf);
    if (arr.length !== nRows * nCols) {
      console.warn("live.js: decoded float32 length mismatch", arr.length, nRows * nCols);
    }
    return arr; // row-major (nRows, nCols)
  }

  var audioMat = b64ToFloat32(DATA.features.audio_b64, DATA.features.n_t, DATA.features.n_audio);
  var vibMat = b64ToFloat32(DATA.features.vibration_b64, DATA.features.n_t, DATA.features.n_vibration);

  // -------------------------------------------------------------- format helpers
  function pad2(n) { return n < 10 ? "0" + n : "" + n; }
  function fmtHMS(totalS) {
    totalS = Math.max(0, Math.round(totalS));
    var h = Math.floor(totalS / 3600), m = Math.floor((totalS % 3600) / 60), s = totalS % 60;
    return pad2(h) + ":" + pad2(m) + ":" + pad2(s);
  }
  function fmtDuration(totalS) {
    totalS = Math.max(0, Math.round(totalS));
    var h = Math.floor(totalS / 3600), m = Math.floor((totalS % 3600) / 60), s = totalS % 60;
    if (h > 0) return h + "h " + m + "m " + s + "s";
    if (m > 0) return m + "m " + s + "s";
    return s + "s";
  }
  function simUtcAt(playheadS) {
    var t0 = new Date(DATA.t0_utc);
    var d = new Date(t0.getTime() + playheadS * 1000);
    return d.toISOString().slice(11, 19) + " UTC";
  }
  function fmtNum(v, digits, suffix) {
    if (v === null || v === undefined || (typeof v === "number" && isNaN(v))) return "—";
    return v.toFixed(digits) + (suffix || "");
  }

  // -------------------------------------------------------------- lookups
  function bisectRight(arr, x) {
    var lo = 0, hi = arr.length;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (arr[mid] <= x) lo = mid + 1; else hi = mid;
    }
    return lo;
  }
  function traceIndexAt(playheadS) {
    var i = bisectRight(DATA.trace.t_s, playheadS) - 1;
    return Math.max(0, Math.min(i, DATA.trace.t_s.length - 1));
  }
  function denseIndexAt(playheadS, n) {
    return Math.max(0, Math.min(Math.floor(playheadS), n - 1));
  }
  function segmentAt(playheadS) {
    var segs = DATA.segments;
    for (var i = 0; i < segs.length; i++) {
      if (playheadS >= segs[i].start_s && playheadS < segs[i].end_s) return segs[i];
    }
    return segs[segs.length - 1];
  }
  function featureSnapshotIndexAt(playheadS) {
    var idx = DATA.features.t_idx;
    var i = bisectRight(idx, playheadS) - 1;
    return Math.max(0, Math.min(i, idx.length - 1));
  }

  // -------------------------------------------------------------- DOM refs
  var $ = function (id) { return document.getElementById(id); };
  var duration = DATA.duration_s;

  // -------------------------------------------------------------- static: header/status
  $("headRunLabel").textContent = DATA.run;
  $("unitName").textContent = DATA.unit_name;
  $("eraChip").textContent = "Era " + DATA.sentinel.era + " — thresholds " + DATA.regime;
  var sentinelOk = !DATA.sentinel.s1_fired && !DATA.sentinel.s2_fired;
  $("sentinelLed").className = "led " + (sentinelOk ? "ok" : "warn");
  $("sentinelChip").lastChild.textContent =
    "Sentinel " + (sentinelOk ? "OK" : "fired") + " (no_mode_fits " +
    (DATA.sentinel.s1_rate * 100).toFixed(1) + "% vs " + (DATA.sentinel.s1_threshold * 100).toFixed(1) + "% budget)";

  // -------------------------------------------------------------- static: rings
  $("ringsRow").innerHTML =
    '<div class="ring-cell">' + DATA.rings.generator + "</div>" +
    '<div class="ring-cell">' + DATA.rings.turbine + "</div>";

  // -------------------------------------------------------------- static: ribbon
  var ribbonWrap = $("ribbonWrap");
  DATA.segments.forEach(function (seg) {
    var el = document.createElement("div");
    el.className = "ribbon-seg";
    el.style.left = (100 * seg.start_s / duration) + "%";
    el.style.width = Math.max(0.05, 100 * (seg.end_s - seg.start_s) / duration) + "%";
    el.style.background = stateColor(seg.state_name);
    el.title = stateLabel(seg.state_name) + "  " + fmtHMS(seg.start_s) + "–" + fmtHMS(seg.end_s);
    ribbonWrap.insertBefore(el, $("ribbonPlayhead"));
  });
  DATA.alerts.forEach(function (a) {
    var t = document.createElement("div");
    t.className = "ribbon-tick";
    t.style.left = (100 * a.start_s / duration) + "%";
    t.title = a.candidate_id + " (" + a.klass + ") @ " + fmtHMS(a.start_s);
    ribbonWrap.insertBefore(t, $("ribbonPlayhead"));
  });
  $("ribbonStart").textContent = fmtHMS(0);
  $("ribbonEnd").textContent = fmtHMS(duration);

  // -------------------------------------------------------------- static: log-mel strip
  var logmelSrc = "data:image/png;base64," + DATA.logmel.png_b64;
  $("logmelStrip").src = logmelSrc;
  $("logmelStripEng").src = logmelSrc;
  $("logmelStrip").style.width = DATA.logmel.width_px + "px";
  $("logmelStripEng").style.width = DATA.logmel.width_px + "px";

  // -------------------------------------------------------------- static: p-value chart(s)
  var P_FLOOR = 1e-4;
  function log10(x) { return Math.log(x) / Math.LN10; }
  function pToY(p, h) {
    var clamped = Math.max(p, P_FLOOR);
    var frac = (log10(clamped) - log10(P_FLOOR)) / (0 - log10(P_FLOOR)); // 0 (floor) .. 1 (p=1)
    return h - frac * h;
  }
  function buildPvalueSvg(w, h) {
    var t = DATA.trace, n = t.t_s.length;
    var step = Math.max(1, Math.floor(n / 2200)); // cap path complexity, still dense enough visually
    var pts = [];
    for (var i = 0; i < n; i += step) {
      var x = (t.t_s[i] / duration) * w;
      var y = pToY(t.p_value[i], h);
      pts.push(x.toFixed(1) + "," + y.toFixed(1));
    }
    var dotsRadius = w > 260 ? 1.6 : 1.1;
    var dots = [];
    for (var j = 0; j < n; j++) {
      if (t.alarm[j]) {
        var ax = (t.t_s[j] / duration) * w, ay = pToY(t.p_value[j], h);
        dots.push('<circle cx="' + ax.toFixed(1) + '" cy="' + ay.toFixed(1) + '" r="' + dotsRadius +
          '" fill="var(--alarm)" fill-opacity="0.55"/>');
      }
    }
    var thresholdY = pToY(0.05, h).toFixed(1);
    var gridLines = [1, 0.1, 0.01, 0.001, P_FLOOR].map(function (p) {
      var y = pToY(p, h).toFixed(1);
      return '<line x1="0" y1="' + y + '" x2="' + w + '" y2="' + y +
        '" stroke="var(--hair-2)" stroke-width="1"/>' +
        '<text x="3" y="' + (parseFloat(y) - 2) + '" font-size="8" fill="var(--dim)" font-family="var(--font-mono)">' +
        (p === 1 ? "1" : p.toExponential(0)) + "</text>";
    }).join("");
    return '<svg viewBox="0 0 ' + w + " " + h + '" preserveAspectRatio="none">' +
      gridLines +
      '<line x1="0" y1="' + thresholdY + '" x2="' + w + '" y2="' + thresholdY +
      '" stroke="var(--warn)" stroke-width="1.2" stroke-dasharray="3 2"/>' +
      '<text x="' + (w - 62) + '" y="' + (parseFloat(thresholdY) - 3) +
      '" font-size="8" fill="var(--warn)" font-family="var(--font-mono)">alpha=0.05</text>' +
      '<polyline points="' + pts.join(" ") + '" fill="none" stroke="var(--live)" stroke-width="1"/>' +
      dots.join("") +
      '<line id="__playhead__" x1="0" y1="0" x2="0" y2="' + h + '" stroke="var(--ink)" stroke-width="1.4"/>' +
      "</svg>";
  }
  var pvalueW = 560, pvalueH = 96;
  $("pvalueChartWrap").innerHTML = buildPvalueSvg(pvalueW, pvalueH);
  $("pvalueChartWrapEng").innerHTML = buildPvalueSvg(pvalueW, 150);

  // -------------------------------------------------------------- static: alarm list + ack
  var ACK_PREFIX = "rowii-live-ack:";
  function isAcked(id) { return localStorage.getItem(ACK_PREFIX + id) === "1"; }
  function setAcked(id, val) {
    if (val) localStorage.setItem(ACK_PREFIX + id, "1");
    else localStorage.removeItem(ACK_PREFIX + id);
  }
  var alarmListEl = $("alarmList");
  if (DATA.alerts.length === 0) {
    alarmListEl.innerHTML = '<div class="alarm-empty">No candidate-register episodes for this session.</div>';
  }
  DATA.alerts.forEach(function (a) {
    var row = document.createElement("div");
    row.className = "alarm-row" + (isAcked(a.candidate_id) ? " acked" : "");
    row.dataset.id = a.candidate_id;
    row.dataset.start = a.start_s;
    var mismatch = a.mode_mismatch ? '<span class="mismatch">detector/SCADA mode differ</span> · ' : "";
    row.innerHTML =
      '<span class="alarm-time mono">' + fmtHMS(a.start_s) + "</span>" +
      '<span class="path-badge ' + a.klass + '">' + a.klass + "</span>" +
      '<span class="alarm-state" style="color:' + stateColor(a.state_name) + '">' + stateLabel(a.state_name) + "</span>" +
      '<span class="alarm-why" title="' + a.criterion_text.replace(/"/g, "&quot;") + '">' + mismatch +
      a.criterion_text + "</span>" +
      '<input type="checkbox" title="acknowledge (visual only)" ' + (isAcked(a.candidate_id) ? "checked" : "") + ">";
    row.querySelector("input").addEventListener("change", function (e) {
      setAcked(a.candidate_id, e.target.checked);
      row.classList.toggle("acked", e.target.checked);
      renderKpis(state.playheadS);
    });
    alarmListEl.appendChild(row);
  });

  // -------------------------------------------------------------- static: feature dim labels
  $("audioDim").textContent = DATA.features.n_audio + "-d";
  $("vibDim").textContent = DATA.features.n_vibration + "-d";
  $("fusionDim").textContent = (DATA.features.n_audio + DATA.features.n_vibration) + "-d";
  $("beatsDim").textContent = DATA.features.n_beats + "-d";
  $("s2Threshold").title = "per-state score threshold; the alarm rule itself is p < 0.05 uniformly";

  // -------------------------------------------------------------- feature canvas (bar/heatmap)
  var canvas = $("featureCanvas");
  var ctx = canvas.getContext("2d");
  function resizeCanvas() {
    var rect = canvas.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function divergingColor(z) {
    // z in [-4, 4]; blue for negative (below day-mean), warm for positive.
    var t = Math.max(-4, Math.min(4, z)) / 4; // -1..1
    if (t >= 0) {
      var r = Math.round(37 + t * (185 - 37)), g = Math.round(99 + t * (56 - 99)), b = Math.round(168 + t * (21 - 168));
      return "rgb(" + r + "," + g + "," + b + ")";
    }
    var tt = -t;
    var r2 = Math.round(37 + tt * (91 - 37)), g2 = Math.round(99 + tt * (33 - 99)), b2 = Math.round(168 + tt * (182 - 168));
    return "rgb(" + r2 + "," + g2 + "," + b2 + ")";
  }
  function drawFeatureCanvas(snapIdx) {
    var rect = canvas.getBoundingClientRect();
    var w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);
    var nA = DATA.features.n_audio, nV = DATA.features.n_vibration;
    var total = nA + nV + 2; // +2 for a small gap "slot" between groups
    var barW = w / total;
    var mid = h / 2;
    for (var i = 0; i < nA; i++) {
      var z = audioMat[snapIdx * nA + i];
      ctx.fillStyle = divergingColor(z);
      var bh = Math.min(h / 2 - 2, Math.abs(z) / 4 * (h / 2 - 2));
      ctx.fillRect(i * barW, z >= 0 ? mid - bh : mid, Math.max(1, barW - 0.3), bh);
    }
    var vOffset = nA + 2;
    for (var j = 0; j < nV; j++) {
      var zv = vibMat[snapIdx * nV + j];
      ctx.fillStyle = divergingColor(zv);
      var bhv = Math.min(h / 2 - 2, Math.abs(zv) / 4 * (h / 2 - 2));
      ctx.fillRect((vOffset + j) * barW, zv >= 0 ? mid - bhv : mid, Math.max(1, barW - 0.3), bhv);
    }
    ctx.strokeStyle = "rgba(31,42,55,.35)";
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
    ctx.strokeStyle = "rgba(31,42,55,.5)";
    var divX = (nA + 1) * barW;
    ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(divX, 0); ctx.lineTo(divX, h); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#5b6b7c"; ctx.font = "9px var(--font-ui)";
    ctx.fillText("audio (135)", 2, 10);
    ctx.fillText("vibration (96)", divX + 3, 10);
  }

  // -------------------------------------------------------------- master render
  var state = { playheadS: 0, playing: false, speed: 1, lastTick: null };

  function renderKpis(playheadS) {
    var acknowledgedButOccurred = 0, occurred = 0;
    DATA.alerts.forEach(function (a) {
      if (a.start_s <= playheadS) {
        occurred++;
        if (isAcked(a.candidate_id)) acknowledgedButOccurred++;
      }
    });
    $("activeAlarmsKpi").textContent = occurred - acknowledgedButOccurred;
    $("totalAlarmsKpi").textContent = "of " + occurred + " so far (" + DATA.alerts.length + " today)";
  }

  function render(playheadS) {
    playheadS = Math.max(0, Math.min(playheadS, duration));
    state.playheadS = playheadS;

    // -- clock / transport
    $("simClock").textContent = simUtcAt(playheadS);
    $("transportReadout").textContent = fmtHMS(playheadS) + " / " + fmtHMS(duration);
    var pct = (100 * playheadS / duration);
    $("ribbonPlayhead").style.left = pct + "%";

    // -- state / segment
    var seg = segmentAt(playheadS);
    var since = playheadS - seg.start_s;
    $("stateNameKpi").textContent = stateLabel(seg.state_name);
    $("stateDotKpi").style.background = stateColor(seg.state_name);
    $("stateSinceKpi").textContent = "since " + fmtDuration(since);
    $("s1State").textContent = stateLabel(seg.state_name);
    $("s1Since").textContent = fmtDuration(since);

    // -- ring pulsing (per-stream RMS levels, dense 1 Hz)
    var li = denseIndexAt(playheadS, DATA.levels.n);
    setRingLevel("RAWGeneratorMic__0", DATA.levels.gen_mic[li]);
    setRingLevel("RAWTurbineMic__1", DATA.levels.tur_mic[li]);
    setRingLevel("RAWGeneratorVib__2", DATA.levels.gen_vib[li]);
    setRingLevel("RAWTurbineVib__3", DATA.levels.tur_vib[li]);
    var allValid = DATA.levels.valid[li];
    $("sensorHealth").innerHTML = '<span class="led ' + (allValid ? "ok" : "warn") +
      '"></span>' + (allValid ? "4/4 streams reporting" : "feature gap this second");

    // -- SCADA
    var si = denseIndexAt(playheadS, DATA.scada.n);
    var p = DATA.scada.power_mw[si], n = DATA.scada.speed_rpm[si];
    $("powerKpi").textContent = fmtNum(p, 2, " MW");
    $("speedKpi").textContent = fmtNum(n, 1, " rpm");
    $("scadaPowerMini").textContent = fmtNum(p, 2, " MW");
    $("scadaSpeedMini").textContent = fmtNum(n, 1, " rpm");
    var scadaState = DATA.scada.scada_state[si];
    $("s1ScadaState").textContent = stateLabel(scadaState);
    var agreeStrict = scadaState === seg.state_name;
    $("s1Agree").innerHTML = '<span class="agree-badge ' + (agreeStrict ? "agree" : "disagree") + '">' +
      (agreeStrict ? "agree" : "differ") + "</span>";
    var lb = DATA.scada.load_bin[si];
    var lbLabel = lb < 0 ? "n/a (not turbine/pump)" : ["low", "mid", "high"][lb] || ("bin " + lb);
    $("s1LoadBin").textContent = lbLabel + " load";

    // -- Step 2 trace
    var ti = traceIndexAt(playheadS);
    var pv = DATA.trace.p_value[ti], sc = DATA.trace.score[ti], al = DATA.trace.alarm[ti];
    var nt = DATA.trace.near_transition[ti];
    $("s2Pvalue").textContent = pv.toExponential(3);
    $("s2Score").textContent = sc.toFixed(4);
    var stObj = DATA.states[String(DATA.trace.state[ti])];
    $("s2Threshold").textContent = stObj && stObj.threshold !== null ? stObj.threshold.toFixed(4) : "n/a";
    $("s2NearTransition").textContent = nt ? "yes" : "no";
    $("s2Alarm").innerHTML = al ? '<span style="color:var(--alarm)">ALARM</span>' : "clear";
    var phLine1 = document.querySelector("#pvalueChartWrap #__playhead__");
    var phLine2 = document.querySelector("#pvalueChartWrapEng #__playhead__");
    var px = (playheadS / duration) * pvalueW;
    if (phLine1) { phLine1.setAttribute("x1", px); phLine1.setAttribute("x2", px); }
    if (phLine2) { phLine2.setAttribute("x1", px); phLine2.setAttribute("x2", px); }

    // -- log-mel pan (px-per-second is fixed: one strip column per stride_s seconds)
    var pxPerS = 1.0 / DATA.logmel.stride_s;
    var offset = playheadS * pxPerS - ($("logmelViewport").clientWidth / 2);
    $("logmelStrip").style.transform = "translateX(" + (-offset) + "px)";
    var offsetEng = playheadS * pxPerS - ($("logmelViewportEng").clientWidth / 2);
    $("logmelStripEng").style.transform = "translateX(" + (-offsetEng) + "px)";

    // -- feature snapshot canvas
    var fi = featureSnapshotIndexAt(playheadS);
    drawFeatureCanvas(fi);
    var beatsNorm = DATA.features.beats_norm01[fi];
    $("beatsGaugeFill").style.width = (beatsNorm * 100).toFixed(1) + "%";
    $("beatsGaugeVal").textContent = beatsNorm.toFixed(3) + " (norm., day range)";

    // -- alarm rows: future dim, ack state
    var rows = alarmListEl.children;
    for (var r = 0; r < rows.length; r++) {
      var row = rows[r];
      var startS = parseFloat(row.dataset.start);
      row.classList.toggle("future", startS > playheadS);
    }
    renderKpis(playheadS);
  }

  function setRingLevel(stream, level01) {
    var dots = document.querySelectorAll('[data-stream="' + stream + '"] .mic-dot, [data-stream="' + stream + '"] .vib-dot');
    dots.forEach(function (dot) {
      var opacity = 0.28 + 0.68 * Math.max(0, Math.min(1, level01 || 0));
      dot.setAttribute("fill-opacity", opacity.toFixed(2));
      var isMic = dot.classList.contains("mic-dot");
      if (isMic) {
        var scale = 1 + 0.55 * Math.max(0, Math.min(1, level01 || 0));
        dot.setAttribute("r", (5.6 * scale).toFixed(1));
      }
    });
  }

  // -------------------------------------------------------------- sentinel gauge (engineering, static)
  (function initSentinelGauge() {
    var rate = DATA.sentinel.s1_rate, thr = DATA.sentinel.s1_threshold;
    var maxScale = Math.max(thr * 1.4, rate * 1.15);
    $("sentinelGaugeFill").style.width = (100 * rate / maxScale).toFixed(1) + "%";
    $("sentinelGaugeFill").style.background = DATA.sentinel.s1_fired ? "var(--alarm)" : "var(--live)";
    $("sentinelGaugeMark").style.left = (100 * thr / maxScale).toFixed(1) + "%";
    $("sentinelGaugeVal").textContent = (rate * 100).toFixed(2) + "% / " + (thr * 100).toFixed(2) + "% budget";
    $("s1SentinelRate").textContent = (rate * 100).toFixed(2) + "%";
    $("s1SentinelThreshold").textContent = (thr * 100).toFixed(2) + "%";
    $("s1Decision").textContent = DATA.sentinel.decision + (DATA.sentinel.s1_fired ? "" : " (sentinel did not fire)");
  })();

  // -------------------------------------------------------------- FAR KPI (static)
  (function initFarKpi() {
    var realized = DATA.sentinel.realized_far, nominal = DATA.sentinel.nominal_alpha;
    var ratio = realized / nominal;
    var cls = ratio <= 1.2 ? "var(--live)" : ratio <= 2.2 ? "var(--warn)" : "var(--alarm)";
    var el = $("farKpi");
    el.textContent = (realized * 100).toFixed(1) + "% vs " + (nominal * 100).toFixed(0) + "%";
    el.style.color = cls;
    $("farKpiSub").textContent = DATA.regime + " thresholds, " + DATA.sentinel.far_basis;
  })();

  // -------------------------------------------------------------- playhead loop
  function tick(ts) {
    if (state.playing) {
      if (state.lastTick !== null) {
        var dtS = (ts - state.lastTick) / 1000 * state.speed;
        state.playheadS += dtS;
        if (state.playheadS >= duration) { state.playheadS = duration; state.playing = false; updatePlayButton(); }
      }
      state.lastTick = ts;
      render(state.playheadS);
    } else {
      state.lastTick = null;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  function updatePlayButton() { $("playBtn").textContent = state.playing ? "Pause" : "Play"; }

  // -------------------------------------------------------------- controls
  $("playBtn").addEventListener("click", function () {
    state.playing = !state.playing;
    updatePlayButton();
  });
  Array.prototype.forEach.call(document.querySelectorAll(".transport-speeds button"), function (btn) {
    btn.addEventListener("click", function () {
      state.speed = parseFloat(btn.dataset.speed);
      Array.prototype.forEach.call(document.querySelectorAll(".transport-speeds button"), function (b) {
        b.classList.toggle("active", b === btn);
      });
    });
  });

  function seekFromRibbonEvent(evt) {
    var rect = ribbonWrap.getBoundingClientRect();
    var frac = Math.max(0, Math.min(1, (evt.clientX - rect.left) / rect.width));
    render(frac * duration);
  }
  var dragging = false;
  ribbonWrap.addEventListener("pointerdown", function (e) { dragging = true; seekFromRibbonEvent(e); });
  window.addEventListener("pointermove", function (e) { if (dragging) seekFromRibbonEvent(e); });
  window.addEventListener("pointerup", function () { dragging = false; });

  Array.prototype.forEach.call(document.querySelectorAll("[data-view-btn]"), function (btn) {
    btn.addEventListener("click", function () {
      document.body.dataset.view = btn.dataset.viewBtn;
      Array.prototype.forEach.call(document.querySelectorAll("[data-view-btn]"), function (b) {
        b.classList.toggle("active", b === btn);
      });
      resizeCanvas();
      drawFeatureCanvas(featureSnapshotIndexAt(state.playheadS));
    });
  });

  window.addEventListener("resize", function () {
    resizeCanvas();
    drawFeatureCanvas(featureSnapshotIndexAt(state.playheadS));
    render(state.playheadS);
  });

  // -------------------------------------------------------------- init
  resizeCanvas();
  render(0);
  updatePlayButton();
})();
