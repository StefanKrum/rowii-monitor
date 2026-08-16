// ROWII Monitor -- live.html replay engine. Reads the embedded #live-data JSON
// (written by scripts/build_live_replay.py) and drives everything on the ONE
// integrated dashboard: the two-row state/SCADA ribbon (also the scrubber), the
// log-mel pan, the p-value chart, the sensor-ring pulsing, the KPI strip, the
// alarm feed, and the pipeline-diagnostics band. No network requests -- every
// byte the page needs is already inline (the two mic .m4a files excepted, and
// only once an operator actually selects one in the "Listen" control).
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
    if (name === "n/a") return "n/a"; // impulse-path candidates carry no detector state
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
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function simDateAt(playheadS) {
    // The replay's own simulated DATE, from the same t0_utc + playhead the clock
    // uses -- UTC getters only (never locale-dependent Date formatting), so the
    // date reads identically wherever the page is opened.
    var d = new Date(new Date(DATA.t0_utc).getTime() + playheadS * 1000);
    return pad2(d.getUTCDate()) + " " + MONTHS[d.getUTCMonth()] + " " + d.getUTCFullYear();
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

  // -------------------------------------------------------------- static: ribbon rows
  // Two rows inside ONE `.ribbon-wrap`: the detector's own segments (Step 1) on
  // top, the rule-based SCADA mode underneath, sharing one playhead and one
  // scrub target -- the same detector-vs-SCADA pairing review.html already shows
  // per candidate (candidate_kit.render_state_ribbon_html), here over the whole
  // session. Same fixed state-colour vocabulary for BOTH rows (site_common.py's
  // STATE_COLORS): a SCADA "turbine" block and a detected "turbine" block are
  // the same colour precisely so a disagreement is visible as a colour break.
  var RIBBON_LABEL_MIN_PX = 34;
  /* Mirrors candidate_kit._RIBBON_LABEL_MIN_PX -- below this a block is too
     narrow for in-band text and keeps only its hover title. */
  var RIBBON_RUN_BREAK_PX = 6;
  /* A block of a DIFFERENT state at least this wide is visible as its own colour
     band, so the next same-state block after it starts a new visual run and earns
     its label back (`updateRibbonLabels`). The sub-pixel 1-second invalid blocks
     that punctuate the detector row all day do not. */
  var ribbonWrap = $("ribbonWrap");
  var detectedTrack = $("ribbonTrackDetected");
  var scadaTrack = $("ribbonTrackScada");

  function addRibbonSeg(track, startS, endS, stateName) {
    var el = document.createElement("div");
    el.className = "ribbon-seg";
    el.style.left = (100 * startS / duration) + "%";
    el.style.width = Math.max(0.02, 100 * (endS - startS) / duration) + "%";
    el.style.background = stateColor(stateName);
    el.title = stateLabel(stateName) + "  " + fmtHMS(startS) + "–" + fmtHMS(endS) +
      "  (" + fmtDuration(endS - startS) + ")";
    var label = document.createElement("span");
    label.className = "ribbon-seg-label";
    label.textContent = stateLabel(stateName);
    el.appendChild(label);
    track.appendChild(el);
  }

  DATA.segments.forEach(function (seg) {
    addRibbonSeg(detectedTrack, seg.start_s, seg.end_s, seg.state_name);
  });

  (function buildScadaRow() {
    // The 1 Hz SCADA state series, run-length encoded into contiguous blocks --
    // one div per RUN, not one per second (a full day is ~15.6k seconds and only
    // ~100 runs).
    var s = DATA.scada.scada_state, n = s.length;
    if (!DATA.scada.has_scada || n === 0) {
      var empty = document.createElement("div");
      empty.className = "ribbon-empty";
      empty.textContent = "no SCADA recorded for this session";
      scadaTrack.appendChild(empty);
      return;
    }
    var runStart = 0;
    for (var i = 1; i <= n; i++) {
      if (i === n || s[i] !== s[runStart]) {
        addRibbonSeg(scadaTrack, runStart * duration / n, i * duration / n, s[runStart]);
        runStart = i;
      }
    }
  })();

  DATA.alerts.forEach(function (a) {
    var t = document.createElement("div");
    t.className = "ribbon-tick";
    t.style.left = (100 * a.start_s / duration) + "%";
    t.title = a.candidate_id + " (" + a.klass + ") @ " + fmtHMS(a.start_s);
    detectedTrack.appendChild(t);
  });

  function updateRibbonLabels() {
    // In-band text only where the block is actually wide enough to hold it --
    // measured, so it stays right across a window resize (the ribbon is fluid).
    // A block is also left unlabelled when it would only repeat the label of the
    // block before it (the detector row alternates turbine/1-second-invalid all
    // day, which would otherwise print "Turbine" 29 times): one label per visual
    // run, every block keeping its own hover title either way.
    [detectedTrack, scadaTrack].forEach(function (track) {
      var lastShown = null;
      Array.prototype.forEach.call(track.querySelectorAll(".ribbon-seg"), function (el) {
        var label = el.firstElementChild;
        if (!label) return;
        var name = label.textContent, width = el.offsetWidth;
        var show = width >= RIBBON_LABEL_MIN_PX && name !== lastShown;
        label.style.display = show ? "block" : "none";
        if (show) lastShown = name;
        else if (name !== lastShown && width >= RIBBON_RUN_BREAK_PX) lastShown = null;
      });
    });
  }
  $("ribbonStart").textContent = fmtHMS(0);
  $("ribbonEnd").textContent = fmtHMS(duration);

  // -------------------------------------------------------------- static: log-mel strip
  $("logmelStrip").src = "data:image/png;base64," + DATA.logmel.png_b64;
  $("logmelStrip").style.width = DATA.logmel.width_px + "px";

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
  var pvalueW = 560, pvalueH = 150;
  $("pvalueChartWrap").innerHTML = buildPvalueSvg(pvalueW, pvalueH);

  // -------------------------------------------------------------- static: SCADA trend
  // Active power and shaft speed for the WHOLE session on the ribbon's own x-axis,
  // so the operating-mode blocks above and the process values below are read at the
  // same instant. Each channel is scaled to its own full-day range (the caption
  // states both ranges, since two channels in different units share one box) and
  // drawn in the palette's own ink/dim -- solid = P, dashed = n. Gaps in SCADA
  // coverage break the line rather than being bridged with an invented value.
  var trendW = 560, trendH = 62, TREND_PAD = 6;
  function trendExtent(arr) {
    var lo = Infinity, hi = -Infinity;
    for (var i = 0; i < arr.length; i++) {
      var v = arr[i];
      if (v === null || v === undefined) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo)) return null;
    return [lo, hi - lo < 1e-9 ? lo + 1 : hi];
  }
  function trendPolylines(arr, ext, step, n, dash) {
    var runs = [], cur = [];
    for (var i = 0; i < n; i += step) {
      var v = arr[i];
      if (v === null || v === undefined) {
        if (cur.length > 1) runs.push(cur.join(" "));
        cur = [];
        continue;
      }
      var x = (i / n) * trendW;
      var y = trendH - TREND_PAD - ((v - ext[0]) / (ext[1] - ext[0])) * (trendH - 2 * TREND_PAD);
      cur.push(x.toFixed(1) + "," + y.toFixed(1));
    }
    if (cur.length > 1) runs.push(cur.join(" "));
    return runs.map(function (pts) {
      return '<polyline points="' + pts + '" fill="none" stroke="' + (dash ? "var(--dim)" : "var(--ink)") +
        '" stroke-width="1"' + (dash ? ' stroke-dasharray="3 2"' : "") + "/>";
    }).join("");
  }
  (function buildScadaTrend() {
    var n = DATA.scada.n;
    var pExt = DATA.scada.has_scada ? trendExtent(DATA.scada.power_mw) : null;
    var nExt = DATA.scada.has_scada ? trendExtent(DATA.scada.speed_rpm) : null;
    if (!pExt || !nExt) {
      $("scadaTrend").innerHTML = '<div class="ribbon-empty">no SCADA recorded for this session</div>';
      $("scadaTrendCaption").textContent = "";
      return;
    }
    var step = Math.max(1, Math.floor(n / 1400));
    $("scadaTrend").innerHTML =
      '<svg viewBox="0 0 ' + trendW + " " + trendH + '" preserveAspectRatio="none">' +
      trendPolylines(DATA.scada.speed_rpm, nExt, step, n, true) +
      trendPolylines(DATA.scada.power_mw, pExt, step, n, false) +
      '<line id="__scadaPlayhead__" x1="0" y1="0" x2="0" y2="' + trendH +
      '" stroke="var(--ink)" stroke-width="1.4"/>' +
      "</svg>";
    $("scadaTrendCaption").innerHTML =
      "<span>P " + pExt[0].toFixed(1) + " … " + pExt[1].toFixed(1) + " MW (solid)</span>" +
      "<span>n " + nExt[0].toFixed(1) + " … " + nExt[1].toFixed(1) + " rpm (dashed)</span>";
  })();

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
    row.title = "jump the replay to " + fmtHMS(a.start_s);
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
    // Clicking a row seeks the whole page (ribbon, log-mel, p-value, features,
    // and the selected audio stream) to that episode -- the same render() funnel
    // a ribbon scrub goes through, so audio stays locked to the playhead here too.
    row.addEventListener("click", function (e) {
      if (e.target.tagName === "INPUT") return;
      render(a.start_s);
    });
    alarmListEl.appendChild(row);
  });

  (function initFeedLegend() {
    // The register's own path mix for this session, counted from the real rows
    // above -- each row's `criterion_text` remains the authoritative rule.
    var counts = {}, order = [];
    DATA.alerts.forEach(function (a) {
      if (counts[a.klass] === undefined) { counts[a.klass] = 0; order.push(a.klass); }
      counts[a.klass] += 1;
    });
    order.sort();
    if (order.length === 0) {
      $("feedLegend").textContent = "No candidate-register episodes for this session.";
      return;
    }
    $("feedLegend").innerHTML = "Register paths this session: " + order.map(function (k) {
      return '<span class="path-badge ' + k + '">' + k + "</span> " + counts[k];
    }).join(" &middot; ") + " &mdash; hover a row to read its full criterion sentence.";
  })();

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

  function confidenceFor(seg) {
    // The named state's OWN conformal standing, straight from the monitor's
    // per-state table (`low_confidence`, monitor_notes.md): a state below the
    // conformal floor carries a +inf threshold and can therefore never alarm,
    // which an operator must be able to see AT the state, not only in a report.
    if (seg.state === -1) {
      return { text: "no usable data", cls: "warn",
        title: "This window has no usable sensor data -- it is never scored." };
    }
    var st = DATA.states[String(seg.state)];
    if (!st) {
      return { text: "no snapshot reference", cls: "warn",
        title: "The detected mode has no reference in the calibrated snapshot, so it is never scored " +
          "and can never alarm." };
    }
    if (st.low_confidence) {
      return { text: "low confidence", cls: "warn",
        title: "Too few calibration windows to certify alpha for this state -- its threshold is +inf, " +
          "so it can never alarm." };
    }
    return { text: "threshold certified", cls: "ok",
      title: "This state has enough calibration windows to certify its conformal threshold at the " +
        "nominal alpha." };
  }

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
    $("simDate").textContent = simDateAt(playheadS);
    $("simClock").textContent = simUtcAt(playheadS);
    $("transportReadout").textContent = fmtHMS(playheadS) + " / " + fmtHMS(duration);
    var pct = (100 * playheadS / duration);
    $("ribbonPlayhead").style.left = pct + "%";

    // -- state / segment
    var seg = segmentAt(playheadS);
    var since = playheadS - seg.start_s;
    var segState = DATA.states[String(seg.state)];
    var stateName = segState ? segState.name_label : stateLabel(seg.state_name);
    var conf = confidenceFor(seg);
    $("stateNameKpi").textContent = stateName;
    $("stateDotKpi").style.background = stateColor(seg.state_name);
    $("stateSinceKpi").textContent = "since " + fmtDuration(since);
    var confKpi = $("stateConfKpi");
    confKpi.textContent = conf.text;
    confKpi.className = "conf-chip " + conf.cls;
    confKpi.title = conf.title;
    $("s1State").textContent = stateName;
    $("s1Cluster").textContent = seg.state === -1
      ? "— (invalid window)"
      : seg.state + " (" + (segState ? segState.name : "no reference") + ")";
    $("s1Since").textContent = fmtDuration(since);
    var s1Conf = $("s1Confidence");
    s1Conf.textContent = conf.text;
    s1Conf.title = conf.title;

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
    var px = (playheadS / duration) * pvalueW;
    if (phLine1) { phLine1.setAttribute("x1", px); phLine1.setAttribute("x2", px); }
    var trendPh = document.querySelector("#scadaTrend #__scadaPlayhead__");
    if (trendPh) {
      var tx = (playheadS / duration) * trendW;
      trendPh.setAttribute("x1", tx);
      trendPh.setAttribute("x2", tx);
    }

    // -- log-mel pan (px-per-second is fixed: one strip column per stride_s seconds)
    var pxPerS = 1.0 / DATA.logmel.stride_s;
    var offset = playheadS * pxPerS - ($("logmelViewport").clientWidth / 2);
    $("logmelStrip").style.transform = "translateX(" + (-offset) + "px)";

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

    // -- selectable live audio: keep the selected stream locked to this same
    // playheadS. render() is the single funnel both the rAF playback loop and a
    // ribbon scrub call through, so one hook here covers both.
    syncAudio(playheadS);
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

  // -------------------------------------------------------------- selectable live audio
  // "Listen" control (Sensor panel, left column): Muted (default -- nothing ever
  // autoplays) / Generator mic / Turbine mic. The selected stream's <audio> element
  // is kept in lockstep with the replay's own playhead -- the REPLAY is always the
  // master clock (requestAnimationFrame-driven state.playheadS), never the other
  // way around. `audioOffsetS` below is the exact JS mirror of the pinned Python
  // reference `scripts/build_live_audio.py`'s `audio_offset_s` (see that
  // function's own docstring) -- `tests/test_build_live_audio.py` is this
  // formula's authoritative test, since this repo has no JS test runner.
  var AUDIO = DATA.audio || null;
  var audioEls = { gen: $("audioGen"), tur: $("audioTur") };
  var listenToggleEl = $("listenToggle");
  var listenHintEl = $("listenHint");
  var listenState = { mode: "muted" };

  var SYNC_TOLERANCE_S = 0.35;
  /* Re-seek only once native playback drifts past this -- keeps the <audio>
     element's own clock running smoothly in between (a reseek every frame would
     sound choppy) while still catching ribbon scrubs immediately (a scrub is a
     multi-second jump, far past this tolerance, so the very next render() call
     corrects it without needing a separate "was this a scrub" signal). */
  var END_EPSILON_S = 0.05;
  /* Stay this far clear of `audio.duration` -- on the real 290626-tu extraction
     the audio's own coverage ends within a few ms of the replay's own final
     playhead second (both derived from the same recording session), so a raw
     `currentTime` request could land ON or fractionally past `duration`. */
  var MUTE_AT_OR_ABOVE_SPEED = 16;
  /* Mirrors build_live_audio.MUTE_AT_OR_ABOVE_SPEED -- HTMLMediaElement.
     playbackRate is unreliable at 16x in evergreen browsers (task instruction:
     "browsers do not do 16x audio"). */

  function audioOffsetS(playheadS, streamKey) {
    // Mirrors scripts/build_live_audio.py's audio_offset_s(playhead_s, t0_utc,
    // audio_start_utc) EXACTLY: playhead_s + (t0_utc - audio_start_utc), seconds.
    if (!AUDIO || !AUDIO[streamKey]) return null;
    var t0Ms = new Date(DATA.t0_utc).getTime();
    var audioStartMs = new Date(AUDIO[streamKey].start_utc).getTime();
    return playheadS + (t0Ms - audioStartMs) / 1000;
  }

  function listenPlaybackFor(speed) {
    // Mirrors scripts/build_live_audio.py's audio_playback_for_speed(speed).
    if (speed >= MUTE_AT_OR_ABOVE_SPEED) return { rate: 1.0, muted: true };
    return { rate: speed, muted: false };
  }

  function clearListenMarkers() {
    Array.prototype.forEach.call(document.querySelectorAll(".listen-marker"), function (m) {
      m.parentNode.removeChild(m);
    });
  }

  function setListenMarker(streamKey) {
    // A small dashed ring around every mic-dot of the selected stream's own
    // ring group -- injected client-side (never baked into the build-time SVG,
    // which is static): the selection itself is a runtime, not a build-time,
    // fact. Uniform across every position on that stream's ring, matching
    // setRingLevel's own "one level per stream, applied to every marker on that
    // stream" convention (site_common.py's module docstring: the physical
    // position <-> DAQ-channel mapping was never verified).
    clearListenMarkers();
    if (!streamKey) return;
    var streamName = streamKey === "gen" ? "RAWGeneratorMic__0" : "RAWTurbineMic__1";
    var svgNs = "http://www.w3.org/2000/svg";
    Array.prototype.forEach.call(
      document.querySelectorAll('[data-stream="' + streamName + '"] .mic-dot'),
      function (dot) {
        var marker = document.createElementNS(svgNs, "circle");
        marker.setAttribute("class", "listen-marker");
        marker.setAttribute("cx", dot.getAttribute("cx"));
        marker.setAttribute("cy", dot.getAttribute("cy"));
        marker.setAttribute("r", "10");
        marker.setAttribute("fill", "none");
        marker.setAttribute("stroke", "var(--ink)");
        marker.setAttribute("stroke-width", "1.6");
        marker.setAttribute("stroke-dasharray", "2.2 2.2");
        dot.parentNode.appendChild(marker);
      }
    );
  }

  function applyListenSpeed() {
    var pf = listenPlaybackFor(state.speed);
    if (listenState.mode !== "muted") {
      var el = audioEls[listenState.mode];
      el.playbackRate = pf.rate;
      el.muted = pf.muted;
    }
    var showHint = listenState.mode !== "muted" && pf.muted;
    listenHintEl.textContent = showHint ? "audio muted above 4× (playback rate unsupported)" : "";
    listenHintEl.classList.toggle("warn", showHint);
  }

  function syncAudio(playheadS, opts) {
    opts = opts || {};
    if (listenState.mode === "muted" || !AUDIO) return;
    var el = audioEls[listenState.mode];
    var target = audioOffsetS(playheadS, listenState.mode);
    if (target === null) return;
    var dur = isFinite(el.duration) && el.duration > 0 ? el.duration : null;
    if (target < 0 || (dur !== null && target > dur)) {
      // outside this stream's own extracted coverage -- stay silent rather than
      // guess; real coverage margins are wide on the start side and only a few
      // ms on the end side (see END_EPSILON_S), so this should not trigger
      // anywhere inside [0, duration] on the real payload.
      if (!el.paused) el.pause();
      return;
    }
    var clamped = dur !== null ? Math.min(target, dur - END_EPSILON_S) : target;
    clamped = Math.max(0, clamped);
    var drift = Math.abs(el.currentTime - clamped);
    if (opts.forceSeek || drift > SYNC_TOLERANCE_S) {
      el.currentTime = clamped;
    }
    if (state.playing && el.paused) {
      el.play().catch(function () { /* autoplay policy -- fine, a user gesture already selected this stream */ });
    } else if (!state.playing && !el.paused) {
      el.pause();
    }
  }

  function setListenMode(mode) {
    if (mode === listenState.mode) return;
    if (listenState.mode !== "muted") audioEls[listenState.mode].pause();
    listenState.mode = mode;
    Array.prototype.forEach.call(listenToggleEl.querySelectorAll("button"), function (b) {
      var active = b.dataset.listen === mode;
      b.classList.toggle("active", active);
      b.setAttribute("aria-checked", active ? "true" : "false");
    });
    setListenMarker(mode === "muted" ? null : mode);
    $("listenVolumeRow").style.display = mode === "muted" ? "none" : "flex";
    if (mode !== "muted" && AUDIO && AUDIO[mode]) {
      var el = audioEls[mode];
      if (!el.getAttribute("src")) {
        el.src = AUDIO[mode].file; // lazy -- never fetched until first selected
        el.volume = parseFloat($("listenVolume").value);
      }
      applyListenSpeed();
      syncAudio(state.playheadS, { forceSeek: true });
    } else {
      listenHintEl.textContent = "";
      listenHintEl.classList.remove("warn");
    }
  }

  if (!AUDIO) {
    Array.prototype.forEach.call(listenToggleEl.querySelectorAll("button[data-listen]"), function (b) {
      if (b.dataset.listen !== "muted") b.disabled = true;
    });
    listenHintEl.textContent = "audio unavailable for this build";
  } else {
    Array.prototype.forEach.call(listenToggleEl.querySelectorAll("button"), function (btn) {
      btn.addEventListener("click", function () { setListenMode(btn.dataset.listen); });
    });
    $("listenVolume").addEventListener("input", function (e) {
      if (listenState.mode !== "muted") audioEls[listenState.mode].volume = parseFloat(e.target.value);
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
    // render() only runs on rAF ticks while state.playing is true, so a Pause
    // click would otherwise leave the <audio> element playing on its own --
    // sync explicitly, right here, for both directions.
    syncAudio(state.playheadS, { forceSeek: true });
  });
  Array.prototype.forEach.call(document.querySelectorAll(".transport-speeds button"), function (btn) {
    btn.addEventListener("click", function () {
      state.speed = parseFloat(btn.dataset.speed);
      Array.prototype.forEach.call(document.querySelectorAll(".transport-speeds button"), function (b) {
        b.classList.toggle("active", b === btn);
      });
      applyListenSpeed();
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

  window.addEventListener("resize", function () {
    resizeCanvas();
    drawFeatureCanvas(featureSnapshotIndexAt(state.playheadS));
    updateRibbonLabels();
    render(state.playheadS);
  });

  // -------------------------------------------------------------- init
  resizeCanvas();
  updateRibbonLabels();
  render(0);
  updatePlayButton();

  // -------------------------------------------------------------- verification hook
  // Read-only introspection for browser-automation verification only (this repo
  // has no JS test runner) -- never read by the app itself.
  window.__liveDebug = {
    getPlayheadS: function () { return state.playheadS; },
    getPlaying: function () { return state.playing; },
    getSpeed: function () { return state.speed; },
    getListenMode: function () { return listenState.mode; },
    getT0Utc: function () { return DATA.t0_utc; },
    getDurationS: function () { return duration; },
    getAudioMeta: function () { return AUDIO; },
    audioOffsetS: audioOffsetS,
    listenPlaybackFor: listenPlaybackFor,
    // Drives the exact same render() -> syncAudio() path a ribbon scrub or the
    // rAF playback loop would -- for environments (headless/backgrounded tabs)
    // where requestAnimationFrame is throttled and playheadS cannot be advanced
    // just by waiting; see this repo's own verification notes.
    renderAt: function (playheadS) { render(playheadS); },
  };
})();
