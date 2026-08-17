// ROWII Monitor -- live.html replay engine (v7). Reads the embedded #live-data
// JSON (written by scripts/build_live_replay.py) and drives the monitoring-
// style dashboard: cross-session recording cards, the two independently
// seekable detector/SCADA ribbons (one shared playhead), the detector<->SCADA
// agreement line, the 4-channel SCADA trend (P/n/Q/KS, each with its own
// at-playhead value), the log-mel pan, the p-value chart, the KPI strip, the
// stage-1/2/3 pipeline-diagnostics band (named top-deviating features + a
// strip canvas, the state & drift-sentinel verdict, the anomaly p-value), and
// the alarm register (LISTEN jumps the playhead and switches on the generator
// mic). No network requests -- every byte the page needs is already inline
// (the two mic .m4a files excepted, and only once an operator actually
// selects one in the transport bar's "Listen" control).
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
  var CSS_STATE = {
    turbine: "turbine",
    pump: "pump",
    "phase-shifter": "phase-shifter",
    standstill: "standstill",
    transition: "transition",
    invalid: "unknown",
    unknown: "unknown",
  };
  /* Same lookup keys as STATE_COLORS, but the value is a design.css utility-
     class suffix (bg-turbine, bg-pump, ...) rather than a hex color -- used
     only by the session cards' mini-ribbon (below), which paints with CSS
     classes so it can share the stylesheet's own state palette instead of
     duplicating hex values a second time. There is no bg-invalid utility, so
     "invalid" buckets into the same neutral swatch stateColor already gives
     it via "unknown". */
  function cssState(name) { return CSS_STATE[name] || "unknown"; }
  function stateLabel(name) {
    if (!name) return "Unknown";
    if (name === "invalid") return "No usable data";
    if (name === "n/a") return "n/a"; // impulse-path candidates carry no detector state
    return name.charAt(0).toUpperCase() + name.slice(1);
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
  function fmtHoursMin(totalS) {
    // The session cards' own "04:19 h" date-line format -- distinct from
    // fmtDuration's "4h 19m 45s" prose style, which stays in use for ribbon-
    // segment tooltips and the stage kv rows' "since ..." readouts below.
    totalS = Math.max(0, Math.round(totalS));
    var h = Math.floor(totalS / 3600), m = Math.floor((totalS % 3600) / 60);
    return pad2(h) + ":" + pad2(m) + " h";
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
  function fmtNum(v, digits) {
    if (v === null || v === undefined || (typeof v === "number" && isNaN(v))) return "—";
    // Clamp negative zero AFTER rounding (a -0.03 MW standstill reading rounds to
    // "-0.0", which reads as a bogus negative value) -- the single funnel every
    // trend label, KPI, and trend y-tick in this file prints a number through.
    return (+v.toFixed(digits) + 0).toFixed(digits);
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

  // -------------------------------------------------------------- static: session cards + header
  (function renderSessionCards() {
    var wrap = $("sessionCards");
    DATA.sessions_nav.forEach(function (s) {
      var a = document.createElement("a");
      a.className = "session-card" + (s.id === DATA.session.id ? " active" : "");
      a.href = s.href;
      var hours = fmtHoursMin(s.duration_s);
      a.innerHTML =
        '<div class="dt">' + s.date_label + " · " + hours + "</div>" +
        '<div class="nm">' + s.display_name +
          (s.events ? '<span class="badge ev">EVENTS</span>' : "") + "</div>" +
        '<div class="in">' + s.blurb + " · " + s.n_episodes + " episodes</div>" +
        '<div class="mini-ribbon">' +
          s.ribbon.map(function (seg) {
            return '<i class="bg-' + cssState(seg.state) + '" style="width:' +
              ((seg.end_frac - seg.start_frac) * 100).toFixed(2) + '%"></i>';
          }).join("") +
          s.ticks.map(function (f) {
            return '<span class="mt" style="left:' + (f * 100).toFixed(2) + '%"></span>';
          }).join("") +
        "</div>";
      wrap.appendChild(a);
    });
    $("headRunLabel").textContent = DATA.session.id;
  })();

  // -------------------------------------------------------------- static: ribbon rows
  // Two independently seekable ribbons sharing one playhead position: the
  // detector's own segments (Step 1) on top, the rule-based SCADA mode
  // underneath -- the same detector-vs-SCADA pairing review.html already shows
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
  var ribbonWrapDetected = $("ribbonWrapDetected");
  var ribbonWrapScada = $("ribbonWrapScada");
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
    // ~100 runs). Uses the SAME addRibbonSeg path as the detected track above,
    // so both rows emit .ribbon-seg + .ribbon-seg-label identically.
    var s = DATA.scada.scada_state, n = s.length;
    if (!DATA.scada.has_scada || n === 0) {
      var empty = document.createElement("div");
      empty.style.cssText = "padding:2px 8px;color:var(--faint);font-size:9px;line-height:19px";
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

  // One playhead div PER ribbon (the template's own #ribbonPlayhead stays
  // hidden/unused) so each track carries its own, both moved to the same
  // left % together in render().
  var detectedPlayheadEl = document.createElement("div");
  detectedPlayheadEl.className = "ribbon-playhead";
  ribbonWrapDetected.appendChild(detectedPlayheadEl);
  var scadaPlayheadEl = document.createElement("div");
  scadaPlayheadEl.className = "ribbon-playhead";
  ribbonWrapScada.appendChild(scadaPlayheadEl);

  (function renderAgreementLine() {
    // Detector-vs-SCADA agreement, computed once over the whole session at the
    // same 1 Hz resolution buildScadaRow's own s[] array is already in. The
    // detected side only exists as segments (intervals), not a dense array, so
    // it is densified here by walking the (sorted, contiguous) segment list
    // alongside the second index -- O(n), once, not per frame.
    var scadaStates = DATA.scada.scada_state;
    var detectedStates = new Array(scadaStates.length);
    var segs = DATA.segments, si = 0;
    for (var t = 0; t < scadaStates.length; t++) {
      while (si < segs.length - 1 && t >= segs[si].end_s) si++;
      detectedStates[t] = segs[si].state_name;
    }
    var agreeCount = 0, total = 0;
    for (var i = 0; i < detectedStates.length && i < scadaStates.length; i++) {
      if (scadaStates[i] === null || scadaStates[i] === "unknown") continue;
      total++;
      if (detectedStates[i] === scadaStates[i]) agreeCount++;
    }
    $("agreeLine").textContent = total
      ? "detector ↔ SCADA agreement: " + (100 * agreeCount / total).toFixed(1) + " % of windows"
      : "detector ↔ SCADA agreement: n/a";
  })();

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
      '" stroke="var(--warn-fill)" stroke-width="1.2" stroke-dasharray="3 2"/>' +
      '<text x="' + (w - 62) + '" y="' + (parseFloat(thresholdY) - 3) +
      '" font-size="8" fill="var(--warn-fill)" font-family="var(--font-mono)">alpha=0.05</text>' +
      '<polyline points="' + pts.join(" ") + '" fill="none" stroke="var(--live)" stroke-width="1"/>' +
      dots.join("") +
      '<line id="__playhead__" x1="0" y1="0" x2="0" y2="' + h + '" stroke="var(--ink)" stroke-width="1.4"/>' +
      "</svg>";
  }
  var pvalueW = 560, pvalueH = 150;
  $("pvalueChartWrap").innerHTML = buildPvalueSvg(pvalueW, pvalueH);

  // -------------------------------------------------------------- static: 4-channel SCADA trend
  // P/n/Q/KS for the WHOLE session on the ribbons' own x-axis, each channel its
  // own labelled box (module docstring's "process values read at the same
  // instant as the operating-mode blocks above"). Each channel is scaled to its
  // own full-session range; gaps in SCADA coverage break the line rather than
  // being bridged with an invented value. Colours mirror the state palette the
  // approved v7 mockup itself reuses for these four lines (docs/superpowers/
  // specs/mockups/live-v7.html) via the same CSS custom properties the rest of
  // this file already draws SVG strokes from (buildPvalueSvg above).
  var trendW = 560, TREND_PAD = 6;
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
  function trendPolylines(arr, ext, step, n, color, h) {
    var runs = [], cur = [];
    for (var i = 0; i < n; i += step) {
      var v = arr[i];
      if (v === null || v === undefined) {
        if (cur.length > 1) runs.push(cur.join(" "));
        cur = [];
        continue;
      }
      var x = (i / n) * trendW;
      var y = h - TREND_PAD - ((v - ext[0]) / (ext[1] - ext[0])) * (h - 2 * TREND_PAD);
      cur.push(x.toFixed(1) + "," + y.toFixed(1));
    }
    if (cur.length > 1) runs.push(cur.join(" "));
    return runs.map(function (pts) {
      return '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="1.2"/>';
    }).join("");
  }
  // Grafana-style chart dressing, shared by every trend row: 2-3 light gridlines
  // (always, denser once the row is tall enough to hold a middle one) and a
  // dashed zero line (only when the row's own extent actually spans zero -- pump-
  // day P/n both cross it, turbine-day P/n do not). The zero line reuses
  // trendPolylines' own y-mapping so it lands exactly where the polyline itself
  // would plot a 0 reading.
  function trendGridlinesSvg(h) {
    var fracs = h >= 40 ? [0.25, 0.5, 0.75] : [0.33, 0.67];
    return fracs.map(function (f) {
      var y = (h * f).toFixed(1);
      return '<line x1="0" y1="' + y + '" x2="' + trendW + '" y2="' + y +
        '" stroke="var(--hair-2)" stroke-opacity="0.6" stroke-width="1"/>';
    }).join("");
  }
  function trendZeroLineSvg(ext, h) {
    if (ext[0] > 0 || ext[1] < 0) return "";
    var y = (h - TREND_PAD - ((0 - ext[0]) / (ext[1] - ext[0])) * (h - 2 * TREND_PAD)).toFixed(1);
    return '<line x1="0" y1="' + y + '" x2="' + trendW + '" y2="' + y +
      '" stroke="var(--dim)" stroke-width="1" stroke-dasharray="3 2"/>';
  }
  function trendTicksHtml(ext, digits) {
    // Bare numbers only (no unit) -- the row label to the left already carries
    // the unit (module docstring / task instruction). fmtNum's own -0.0 clamp
    // keeps a near-zero extent edge from ever printing as "-0.0" here too.
    return '<span class="trend-ytick max">' + fmtNum(ext[1], digits) + "</span>" +
      '<span class="trend-ytick min">' + fmtNum(ext[0], digits) + "</span>";
  }
  var TRENDS = [
    { key: "power", el: "trendValP", digits: 1, unit: "MW", series: DATA.scada.power_mw, color: "var(--s-turbine)" },
    { key: "speed", el: "trendValN", digits: 1, unit: "rpm", series: DATA.scada.speed_rpm, color: "var(--dim)" },
    { key: "flow", el: "trendValQ", digits: 1, unit: "m³/s", series: DATA.scada.flow_net_m3s, color: "var(--s-phase)" },
    { key: "ks", el: "trendValKS", digits: 1, unit: "pos", series: DATA.scada.ks_valve, color: "var(--s-pump)" },
  ];
  (function buildTrends() {
    var n = DATA.scada.n;
    var step = Math.max(1, Math.floor(n / 1400));
    TRENDS.forEach(function (t) {
      var box = document.querySelector('.trend-chart[data-channel="' + t.key + '"]');
      var ext = DATA.scada.has_scada ? trendExtent(t.series) : null;
      if (!ext) {
        box.innerHTML = '<div style="padding:6px 8px;color:var(--faint);font-size:9px">no SCADA recorded for this session</div>';
        return;
      }
      var h = box.clientHeight || 34; // template's own inline height (P 48px, others 34px)
      box.innerHTML = '<svg viewBox="0 0 ' + trendW + " " + h + '" preserveAspectRatio="none">' +
        trendGridlinesSvg(h) + trendZeroLineSvg(ext, h) +
        trendPolylines(t.series, ext, step, n, t.color, h) + "</svg>" +
        trendTicksHtml(ext, t.digits);
    });
  })();

  // -------------------------------------------------------------- static: alarm register
  var PATH_BADGE_CLASS = { sustained: "sus", transient: "tra" };
  var alarmListEl = $("alarmList");
  if (DATA.alerts.length === 0) {
    alarmListEl.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--faint);padding:16px 8px">' +
      "No candidate-register episodes for this session.</td></tr>";
  } else {
    DATA.alerts.forEach(function (a) {
      var row = document.createElement("tr");
      row.dataset.start = a.start_s;
      var badgeCls = PATH_BADGE_CLASS[a.klass] || "";
      row.innerHTML =
        '<td class="mono">' + fmtHMS(a.start_s) + "</td>" +
        '<td><span class="badge' + (badgeCls ? " " + badgeCls : "") + '">' + a.klass.toUpperCase() + "</span></td>" +
        "<td>" + stateLabel(a.state_name) + "</td>" +
        "<td>" + a.criterion_text + "</td>" +
        '<td><span class="listen" data-cid="' + a.candidate_id + '" title="jump to ' + fmtHMS(a.start_s) +
        '">LISTEN ▸</span></td>';
      // Only the LISTEN control is clickable (not the whole row): seeks the
      // whole page (ribbon, log-mel, p-value, features, and the selected audio
      // stream) to that episode's start, the same render() funnel a ribbon
      // scrub goes through -- and switches listen on (generator mic) if it was
      // muted, so a click always produces audible feedback.
      row.querySelector(".listen").addEventListener("click", function () {
        render(a.start_s);
        if (listenState.mode === "muted") setListenMode("gen");
      });
      alarmListEl.appendChild(row);
    });
  }
  $("totalAlarmsKpi").innerHTML = String(DATA.session.n_episodes) + ' <span class="u">episodes</span>';

  // -------------------------------------------------------------- stage 1: top features + strip
  var fpCanvas = $("fpStripCanvas");
  var fpCtx = fpCanvas.getContext("2d");
  var lastSnapIdx = -1;

  function topDeviations(snapIdx, k) {
    var out = [];
    var nA = DATA.features.n_audio, nV = DATA.features.n_vibration;
    for (var i = 0; i < nA; i++) out.push({ z: audioMat[snapIdx * nA + i], name: DATA.features.audio_names[i] });
    for (var j = 0; j < nV; j++) out.push({ z: vibMat[snapIdx * nV + j], name: DATA.features.vib_names[j] });
    out.sort(function (a, b) { return Math.abs(b.z) - Math.abs(a.z); });
    return out.slice(0, k);
  }

  function renderTopFeatures(snapIdx) {
    var rows = topDeviations(snapIdx, 4);
    $("topFeatures").innerHTML = rows.map(function (r) {
      var mag = Math.min(Math.abs(r.z) / 4, 1) * 50; // half-bar %
      var side = r.z >= 0 ? "pos" : "neg";
      var sign = r.z >= 0 ? "+" : "−";
      return '<div class="fp-row"><span class="nm">' + r.name + "</span>" +
        '<span class="fp-bar"><i class="' + side + '" style="width:' + mag.toFixed(1) + '%"></i></span>' +
        '<span class="fp-val mono">' + sign + Math.abs(r.z).toFixed(1) + " σ</span></div>";
    }).join("");
    drawFpStrip(snapIdx);
  }

  function drawFpStrip(snapIdx) {
    var rect = fpCanvas.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    fpCanvas.width = Math.max(1, Math.round(rect.width * dpr));
    fpCanvas.height = Math.round(18 * dpr);
    fpCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var nA = DATA.features.n_audio, nV = DATA.features.n_vibration, total = nA + nV;
    var w = rect.width, h = 18, barW = w / (total + 2);
    for (var i = 0; i < total; i++) {
      var z = i < nA ? audioMat[snapIdx * nA + i] : vibMat[snapIdx * nV + (i - nA)];
      var x = (i < nA ? i : i + 2) * barW;
      var bh = Math.min(Math.abs(z) / 4, 1) * (h - 3) + 1;
      fpCtx.fillStyle = Math.abs(z) >= 2 ? "#c07f10" : "#9db4d0";
      fpCtx.fillRect(x, h - bh, Math.max(barW - 0.4, 0.6), bh);
    }
    fpCtx.strokeStyle = "#b9c2cd";
    fpCtx.setLineDash([2, 2]);
    fpCtx.beginPath();
    var divX = (nA + 1) * barW;
    fpCtx.moveTo(divX, 0); fpCtx.lineTo(divX, h); fpCtx.stroke();
    fpCtx.setLineDash([]);
  }

  // -------------------------------------------------------------- master render
  var state = { playheadS: 0, playing: false, speed: 1, lastTick: null };

  function confidenceFor(seg) {
    // The named state's OWN conformal standing, straight from the monitor's
    // per-state table (`low_confidence`, monitor_notes.md): a state below the
    // conformal floor carries a +inf threshold and can therefore never alarm,
    // which an operator must be able to see AT the state, not only in a report.
    if (seg.state === -1) {
      return { text: "no usable data",
        title: "This window has no usable sensor data -- it is never scored." };
    }
    var st = DATA.states[String(seg.state)];
    if (!st) {
      return { text: "no snapshot reference",
        title: "The detected mode has no reference in the calibrated snapshot, so it is never scored " +
          "and can never alarm." };
    }
    if (st.low_confidence) {
      return { text: "low confidence",
        title: "Too few calibration windows to certify alpha for this state -- its threshold is +inf, " +
          "so it can never alarm." };
    }
    return { text: "threshold certified",
      title: "This state has enough calibration windows to certify its conformal threshold at the " +
        "nominal alpha." };
  }

  function renderKpis(playheadS) {
    // "active now" = alarm episodes whose own [start_s, start_s + duration_s]
    // window currently contains the playhead -- independent of any operator
    // acknowledgement (the v7 register has no ack control; see live.js's own
    // task-9 report for that removal's rationale).
    var nActive = 0;
    DATA.alerts.forEach(function (a) {
      if (a.start_s <= playheadS && playheadS <= a.start_s + a.duration_s) nActive++;
    });
    $("activeAlarmsKpi").textContent = nActive + " active now";
  }

  function render(playheadS) {
    playheadS = Math.max(0, Math.min(playheadS, duration));
    state.playheadS = playheadS;

    // -- clock / transport
    $("simDate").textContent = simDateAt(playheadS);
    $("simClock").textContent = simUtcAt(playheadS);
    $("transportReadout").textContent = "t = " + fmtHMS(playheadS) + " / " + fmtHMS(duration) + " h";
    var pct = (100 * playheadS / duration) + "%";
    detectedPlayheadEl.style.left = pct;
    scadaPlayheadEl.style.left = pct;

    // -- state / segment (KPI band + stage-1 kv rows -- unaffected by the
    // sentinel's own availability shape, driven entirely by the timeline)
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
    confKpi.title = conf.title;
    $("s1State").textContent = stateName;
    $("s1Cluster").textContent = seg.state === -1
      ? "— (invalid window)"
      : seg.state + " (" + (segState ? segState.name : "no reference") + ")";
    $("s1Since").textContent = fmtDuration(since);
    var s1Conf = $("s1Confidence");
    s1Conf.textContent = conf.text;
    s1Conf.title = conf.title;

    // -- SCADA at-playhead values: KPI band + 4-channel trend value labels
    var si = denseIndexAt(playheadS, DATA.scada.n);
    $("powerKpi").innerHTML = fmtNum(DATA.scada.power_mw[si], 1) + ' <span class="u">MW</span>';
    $("speedKpi").innerHTML = fmtNum(DATA.scada.speed_rpm[si], 1) + ' <span class="u">rpm</span>';
    TRENDS.forEach(function (t) {
      $(t.el).innerHTML = fmtNum(t.series[si], t.digits) + ' <span class="u">' + t.unit + "</span>";
    });
    $("trendPlayhead").style.left = "calc(90px + (100% - 90px) * " + (playheadS / duration) + ")";

    // -- Stage 3 trace
    var ti = traceIndexAt(playheadS);
    var pv = DATA.trace.p_value[ti], sc = DATA.trace.score[ti], al = DATA.trace.alarm[ti];
    var nt = DATA.trace.near_transition[ti];
    $("s2Pvalue").textContent = "p = " + pv.toExponential(3);
    $("s2Score").textContent = sc.toFixed(4);
    var stObj = DATA.states[String(DATA.trace.state[ti])];
    $("s2Threshold").textContent = stObj && stObj.threshold !== null ? stObj.threshold.toFixed(4) : "n/a";
    $("s2NearTransition").textContent = nt ? "yes" : "no";
    $("s2Alarm").innerHTML = al ? '<span class="text-alarm">ALARM</span>' : "clear";
    var phLine = document.querySelector("#pvalueChartWrap #__playhead__");
    if (phLine) {
      var px = (playheadS / duration) * pvalueW;
      phLine.setAttribute("x1", px);
      phLine.setAttribute("x2", px);
    }

    // -- log-mel pan (px-per-second is fixed: one strip column per stride_s seconds)
    var pxPerS = 1.0 / DATA.logmel.stride_s;
    var offset = playheadS * pxPerS - ($("logmelViewport").clientWidth / 2);
    $("logmelStrip").style.transform = "translateX(" + (-offset) + "px)";

    // -- stage 1 top-deviating features + strip: only redrawn when the
    // FEATURE_SNAPSHOT_STRIDE_S-second snapshot column actually changed --
    // sorting 231 values 60x/s is wasteful.
    var fi = featureSnapshotIndexAt(playheadS);
    if (fi !== lastSnapIdx) {
      lastSnapIdx = fi;
      renderTopFeatures(fi);
    }

    // -- alarm register: dim rows the playhead has not reached yet
    var rows = alarmListEl.children;
    for (var r = 0; r < rows.length; r++) {
      var row = rows[r];
      if (row.dataset.start === undefined) continue; // the "no episodes" placeholder row
      row.classList.toggle("future", parseFloat(row.dataset.start) > playheadS);
    }
    renderKpis(playheadS);

    // -- selectable live audio: keep the selected stream locked to this same
    // playheadS. render() is the single funnel both the rAF playback loop and a
    // ribbon scrub call through, so one hook here covers both.
    syncAudio(playheadS);
  }

  // -------------------------------------------------------------- selectable live audio
  // "Listen" control (transport bar): Muted (default -- nothing ever
  // autoplays) / Generator mic / Turbine mic. The selected stream's <audio>
  // element is kept in lockstep with the replay's own playhead -- the REPLAY
  // is always the master clock (requestAnimationFrame-driven state.playheadS),
  // never the other way around. `audioOffsetS` below is the exact JS mirror of
  // the pinned Python reference `scripts/build_live_audio.py`'s
  // `audio_offset_s` (see that function's own docstring) --
  // `tests/test_build_live_audio.py` is this formula's authoritative test,
  // since this repo has no JS test runner.
  var AUDIO = DATA.audio || null;
  var audioEls = { gen: $("audioGen"), tur: $("audioTur") };
  var listenMicsEl = $("listenMics");
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

  function applyListenSpeed() {
    var pf = listenPlaybackFor(state.speed);
    if (listenState.mode !== "muted") {
      var el = audioEls[listenState.mode];
      el.playbackRate = pf.rate;
      el.muted = pf.muted;
    }
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
    Array.prototype.forEach.call(listenMicsEl.querySelectorAll("button[data-listen]"), function (b) {
      b.classList.toggle("active", b.dataset.listen === mode);
    });
    $("specListenLabel").textContent = { muted: "muted", gen: "generator mic", tur: "turbine mic" }[mode];
    if (mode !== "muted" && AUDIO && AUDIO[mode]) {
      var el = audioEls[mode];
      if (!el.getAttribute("src")) {
        el.src = AUDIO[mode].file; // lazy -- never fetched until first selected
      }
      // Re-applied on EVERY switch, not just the first src assignment: the
      // slider can move while this stream was not the active one (e.g. while
      // muted, or while the other mic was selected), and audioEls[mode].volume
      // must not stay stuck at whatever it was the last time this stream itself
      // was active.
      el.volume = parseFloat($("listenVolume").value);
      applyListenSpeed();
      syncAudio(state.playheadS, { forceSeek: true });
    }
  }

  $("listenVolume").addEventListener("input", function (e) {
    var v = parseFloat(e.target.value);
    $("volPct").textContent = Math.round(v * 100) + " %";
    if (AUDIO && listenState.mode !== "muted") audioEls[listenState.mode].volume = v;
  });

  if (!AUDIO) {
    Array.prototype.forEach.call(listenMicsEl.querySelectorAll("button[data-listen]"), function (b) {
      if (b.dataset.listen !== "muted") b.disabled = true;
    });
  } else {
    Array.prototype.forEach.call(listenMicsEl.querySelectorAll("button"), function (btn) {
      btn.addEventListener("click", function () { setListenMode(btn.dataset.listen); });
    });
  }

  // -------------------------------------------------------------- stage 2: sentinel + FAR KPI (static)
  // `sentinel_payload` (build_live_replay.py) ships one of three honest shapes
  // -- never a fabricated rate standing in for one that was never computed:
  //   "full"         -- a real regime replay exists: gauge + the fired/quiet
  //                      verdict sentence + the real realized FAR in the KPI.
  //   "trigger_only" -- the sentinel's own real recorded decision exists (era,
  //                      s1_rate/threshold, s1_fired, decision) but no regime
  //                      replay (no FAR numbers) was ever computed for this
  //                      session: the gauge still renders from s1_rate/
  //                      threshold (a real measurement) and stays keyed on
  //                      s1_fired specifically (it visualizes s1, nothing
  //                      else), but the verdict states the real recorded
  //                      decision in plain language -- NOT the s1-specific
  //                      "budget exceeded" claim, since a recalibrate
  //                      decision can also come from s2 alone
  //                      (run_once_calibrated.py's own _trigger_verdict, s1 OR
  //                      s2) -- plus the payload's own explanatory note. The
  //                      verdict's OWN .fired class therefore follows
  //                      `decision === "recalibrate"`, not s1_fired: those two
  //                      can legitimately disagree (s1_fired=false, decision=
  //                      "recalibrate" when s2 alone triggered it), and the
  //                      verdict text must never read "recalibrate" in green.
  //                      The FAR KPI stays a dash either way.
  //   "none"         -- this session was never scored by the once-calibrated
  //                      driver at all: the sentinel rows/gauge are hidden,
  //                      the verdict slot carries the payload's own note, and
  //                      the FAR KPI stays a dash.
  // The four state-detection kv rows (s1State/s1Cluster/s1Since/s1Confidence)
  // are driven entirely by the timeline/segment data in render() above and are
  // unaffected by which of these three shapes this session's sentinel is.
  (function renderSentinel() {
    var sent = DATA.sentinel;
    var verdictEl = $("s1Decision");
    var gaugeEl = $("sentinelGauge");
    var rateRow = $("s1SentinelRate").closest(".kv-row");
    var shareRow = $("sentinelGaugeVal").closest(".kv-row");

    function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
    function fullVerdict(fired) {
      return fired
        ? "<b>Sentinel fired</b> — this day exceeded its no-mode-fits budget; thresholds were recalibrated."
        : "<b>Sentinel quiet</b> — once-calibrated thresholds stay frozen. Above its budget, the day would " +
          "be flagged for recalibration.";
    }
    function trigVerdict(decision) {
      return decision === "recalibrate"
        ? "<b>Recorded decision: recalibrate</b> — thresholds were recalibrated for this day."
        : "<b>Recorded decision: frozen</b> — thresholds stayed at their once-calibrated values.";
    }

    if (sent.available === "none") {
      rateRow.style.display = "none";
      gaugeEl.style.display = "none";
      shareRow.style.display = "none";
      verdictEl.textContent = capitalize(sent.note);
      $("farKpiSub").textContent = "no regime replay recorded for this session";
      return;
    }

    // "full" and "trigger_only" both carry era/s1_rate/s1_threshold/s1_fired/decision.
    var rate = sent.s1_rate, thr = sent.s1_threshold;
    var maxScale = Math.max(thr * 1.4, rate * 1.15, 1e-9);
    $("sentinelGaugeFill").style.width = (100 * rate / maxScale).toFixed(1) + "%";
    $("sentinelGaugeMark").style.left = (100 * thr / maxScale).toFixed(1) + "%";
    $("sentinelGaugeVal").textContent = (rate * 100).toFixed(2) + "% / " + (thr * 100).toFixed(2) + "% budget";
    $("s1SentinelRate").textContent = (rate * 100).toFixed(2) + "%";
    $("s1SentinelThreshold").textContent = (thr * 100).toFixed(2) + "%";
    gaugeEl.classList.toggle("fired", sent.s1_fired);
    verdictEl.classList.toggle("fired", sent.s1_fired);

    if (sent.available === "trigger_only") {
      // The verdict text below reports the real recorded `decision`, which
      // can diverge from `s1_fired` (decision is s1 OR s2 -- s2 alone can
      // drive a recalibrate even when s1 itself stayed under budget). Its
      // own .fired class must follow the SAME signal the text reports, not
      // s1_fired, or a "recalibrate" verdict could render in the "quiet"
      // (green) style. The gauge just above stays keyed on s1_fired -- it
      // visualizes s1's own rate vs. threshold specifically, nothing else.
      verdictEl.classList.toggle("fired", sent.decision === "recalibrate");
      verdictEl.innerHTML = trigVerdict(sent.decision) + " " + capitalize(sent.note);
      $("farKpiSub").textContent = "no regime replay recorded for this session";
      return;
    }

    verdictEl.innerHTML = fullVerdict(sent.s1_fired);
    $("farKpi").innerHTML = fmtNum(sent.realized_far * 100, 1) + ' <span class="u">% of windows</span>';
    $("farKpiSub").textContent = "realized · budget α = " + (sent.nominal_alpha * 100).toFixed(0) + " %";
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

  function updatePlayButton() { $("playBtn").textContent = state.playing ? "❚❚ Pause" : "▶ Play"; }

  // -------------------------------------------------------------- controls
  $("playBtn").addEventListener("click", function () {
    state.playing = !state.playing;
    updatePlayButton();
    // render() only runs on rAF ticks while state.playing is true, so a Pause
    // click would otherwise leave the <audio> element playing on its own --
    // sync explicitly, right here, for both directions.
    syncAudio(state.playheadS, { forceSeek: true });
  });
  Array.prototype.forEach.call(document.querySelectorAll(".btn-speed"), function (btn) {
    btn.addEventListener("click", function () {
      state.speed = parseFloat(btn.dataset.speed);
      Array.prototype.forEach.call(document.querySelectorAll(".btn-speed"), function (b) {
        b.classList.toggle("active", b === btn);
      });
      applyListenSpeed();
    });
  });

  function seekFromRibbonEvent(evt, wrapEl) {
    var rect = wrapEl.getBoundingClientRect();
    var frac = Math.max(0, Math.min(1, (evt.clientX - rect.left) / rect.width));
    render(frac * duration);
  }
  var dragging = false, dragWrap = null;
  [ribbonWrapDetected, ribbonWrapScada].forEach(function (wrap) {
    wrap.addEventListener("pointerdown", function (e) { dragging = true; dragWrap = wrap; seekFromRibbonEvent(e, wrap); });
  });
  window.addEventListener("pointermove", function (e) { if (dragging) seekFromRibbonEvent(e, dragWrap); });
  window.addEventListener("pointerup", function () { dragging = false; dragWrap = null; });

  window.addEventListener("resize", function () {
    updateRibbonLabels();
    var fi = featureSnapshotIndexAt(state.playheadS);
    lastSnapIdx = fi;
    renderTopFeatures(fi); // canvas pixel size changed even if the snapshot index didn't
    render(state.playheadS);
  });

  // -------------------------------------------------------------- init
  updateRibbonLabels();
  $("volPct").textContent = Math.round(parseFloat($("listenVolume").value) * 100) + " %";
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
