/* ScholarPath NG front-end. Plain JavaScript, no libraries. */
(function () {
  'use strict';
  var calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var debounce = function (fn, ms) { var t; return function () { clearTimeout(t); t = setTimeout(fn, ms); }; };

  /* toasts, confirmations, busy buttons */
  $$('[data-dismiss]').forEach(function (b) { b.addEventListener('click', function () { b.parentNode.remove(); }); });
  setTimeout(function () { $$('.flash').forEach(function (f) { f.remove(); }); }, 7000);
  $$('form[data-confirm]').forEach(function (f) {
    f.addEventListener('submit', function (e) { if (!window.confirm(f.getAttribute('data-confirm'))) e.preventDefault(); });
  });
  $$('form[data-busy]').forEach(function (f) {
    f.addEventListener('submit', function () {
      var b = $('button[type=submit]', f);
      if (b) { b.textContent = f.getAttribute('data-busy'); setTimeout(function () { b.disabled = true; }, 0); }
    });
  });

  /* landing: reading progress bar */
  var bar = $('#progress');
  if (bar) {
    var onScroll = function () {
      var h = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.transform = 'scaleX(' + (h > 0 ? Math.min(1, window.pageYOffset / h) : 0) + ')';
    };
    window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
  }

  /* landing: numbers count up once */
  var counters = $$('[data-count]');
  if (counters.length && 'IntersectionObserver' in window && !calm) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        io.unobserve(en.target);
        var el = en.target, end = parseFloat(el.getAttribute('data-count')), suffix = el.getAttribute('data-suffix') || '',
          dec = (String(end).split('.')[1] || '').length, t0 = performance.now();
        (function tick(now) {
          var k = Math.min(1, (now - t0) / 900), v = end * (1 - Math.pow(1 - k, 4));
          el.textContent = (dec ? v.toFixed(dec) : Math.round(v).toLocaleString()) + suffix;
          if (k < 1) requestAnimationFrame(tick);
        })(t0);
      });
    }, { threshold: 0.5 });
    counters.forEach(function (c) { io.observe(c); });
  }

  /* landing: the five steps light up in turn while in view */
  var five = $('#five');
  if (five && !calm && 'IntersectionObserver' in window) {
    var lis = $$('li', five), at = -1, timer = null;
    var step = function () { at = (at + 1) % lis.length; lis.forEach(function (li, i) { li.classList.toggle('on', i === at); }); };
    new IntersectionObserver(function (e) {
      if (e[0].isIntersecting && !timer) { step(); timer = setInterval(step, 1800); }
      else if (!e[0].isIntersecting && timer) { clearInterval(timer); timer = null; }
    }, { threshold: 0.4 }).observe(five);
    five.addEventListener('pointerenter', function () { clearInterval(timer); timer = -1; lis.forEach(function (li) { li.classList.remove('on'); }); });
  }

  /* landing: live demo against the real model */
  var demo = $('#demo');
  if (demo) {
    var G = JSON.parse(demo.getAttribute('data-grading')), gpa = $('#d-gpa'), att = $('#d-att'), car = $('#d-car'),
      label = $('#d-label'), scale = $('#d-scale'), labels = $('#d-labels'), seq = 0;
    var type = function () { return $('input[name="demo-type"]:checked', demo).value; };
    var paint = function () {
      var g = G[type()];
      $('#d-gpa-out').textContent = Number(gpa.value).toFixed(2) + ' of ' + g.scale;
      $('#d-att-out').textContent = att.value + '%';
      $('#d-car-out').textContent = car.value + ' of 30';
    };
    var ask = function () {
      var mine = ++seq, q = '?type=' + encodeURIComponent(type()) + '&gpa=' + gpa.value + '&attendance=' + att.value + '&ca=' + car.value;
      fetch(demo.getAttribute('data-url') + q).then(function (r) { return r.json(); }).then(function (d) {
        if (mine !== seq || !d.ok) return;
        if (label.textContent !== d.label) {
          label.textContent = d.label; label.classList.remove('swap'); void label.offsetWidth; label.classList.add('swap');
        }
        scale.style.setProperty('--pos', 6 - d.tier);
        $$('span', labels).forEach(function (s, i) { s.classList.toggle('on', i === 5 - d.tier); });
      }).catch(function () { label.textContent = 'Could not reach the model'; });
    };
    var later = debounce(ask, 120);
    [gpa, att, car].forEach(function (el) { el.addEventListener('input', function () { paint(); later(); }); });
    $$('input[name="demo-type"]', demo).forEach(function (r) {
      r.addEventListener('change', function () {
        var g = G[type()], ratio = gpa.value / gpa.max;
        gpa.max = g.scale; gpa.value = (Math.round(ratio * g.scale * 20) / 20).toFixed(2);
        labels.innerHTML = g.labels.slice().reverse().map(function (l) { return '<span>' + l + '</span>'; }).join('');
        paint(); ask();
      });
    });
    paint(); ask();
  }

  /* ---------- prediction form ---------- */
  var form = $('#predict-form');
  if (form) {
    var cfg = JSON.parse(form.getAttribute('data-config')), by = {}, touched = {};
    cfg.features.forEach(function (f) { by[f.name] = f; });
    var warn = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><path d="M12 3 2 20h20zM12 10v5M12 17.5v.01"/></svg>';
    var val = function (name) {
      if (by[name].choices) { var c = $('input[name="' + name + '"]:checked', form); return c ? c.value : ''; }
      return $('#' + name, form).value.trim();
    };
    var itype = function () { return val('institution_type'); };
    var top = function (f) {
      var t = itype();
      if (f.name === 'previous_gpa' && cfg.scales[t]) return cfg.scales[t];
      if (f.name === 'year_of_study' && cfg.years[t]) return cfg.years[t];
      return f.max;
    };
    /* the same rules as schema.validate_input on the server, which has the final say */
    var problem = function (name) {
      var f = by[name], v = val(name), low = f.label.toLowerCase(), t = itype();
      if (v === '') return (f.choices ? 'Choose ' : 'Enter ') + low + '.';
      if (f.choices) return '';
      var n = Number(v);
      if (!isFinite(n)) return f.label + ' must be a number.';
      if (f.kind === 'int' && n !== Math.floor(n)) return f.label + ' must be a whole number.';
      var hi = top(f);
      if (n < f.min || n > hi) {
        if (name === 'previous_gpa' && cfg.scales[t]) return 'Previous GPA must be between 0 and ' + hi + ' for a ' + t + '.';
        if (name === 'year_of_study' && cfg.years[t]) return 'A ' + t + ' programme runs from year 1 to ' + hi + '.';
        return f.label + ' must be between ' + f.min + ' and ' + hi + '.';
      }
      return '';
    };
    var show = function (name, msg) {
      $('[data-field="' + name + '"]', form).classList.toggle('invalid', !!msg);
      $('#' + name + '-error', form).innerHTML = msg ? warn + msg : '';
      var input = $('#' + name, form); if (input) input.setAttribute('aria-invalid', String(!!msg));
      return !msg;
    };
    var check = function (name, force) { return (!force && !touched[name]) ? true : show(name, problem(name)); };
    var tally = function () {
      var total = 0;
      $$('[data-sec]', form).forEach(function (sec) {
        var names = $$('[data-field]', sec).map(function (el) { return el.getAttribute('data-field'); });
        var ok = names.filter(function (n) { return problem(n) === ''; }).length;
        total += ok;
        $('[data-done]', sec).textContent = ok;
        sec.classList.toggle('done', ok === names.length);
      });
      $('#filled').textContent = total;
      $('#fill').style.width = (total / cfg.features.length * 100) + '%';
      var t = itype(), unit = $('[data-unit="previous_gpa"]', form);
      if (cfg.scales[t]) {
        $('#previous_gpa', form).max = cfg.scales[t]; $('#year_of_study', form).max = cfg.years[t];
        unit.textContent = 'of ' + cfg.scales[t];
        $('#previous_gpa-help').textContent = 'GPA from the last completed semester, out of ' + cfg.scales[t] + ' for a ' + t + '.';
      }
    };
    cfg.features.forEach(function (f) {
      if (f.choices) {
        $$('input[name="' + f.name + '"]', form).forEach(function (r) {
          r.addEventListener('change', function () {
            touched[f.name] = true; check(f.name);
            if (f.name === 'institution_type') { check('previous_gpa'); check('year_of_study'); }
            tally();
          });
        });
      } else {
        var input = $('#' + f.name, form);
        input.addEventListener('blur', function () { touched[f.name] = true; check(f.name); });
        input.addEventListener('input', function () { check(f.name); tally(); });
      }
    });
    form.addEventListener('submit', function (e) {
      var first = null;
      cfg.features.forEach(function (f) { touched[f.name] = true; if (!check(f.name, true) && !first) first = f.name; });
      if (first) {
        e.preventDefault();
        var el = $('#' + first, form) || $('input[name="' + first + '"]', form);
        el.focus({ preventScroll: true });
        $('[data-field="' + first + '"]', form).scrollIntoView({ block: 'center', behavior: calm ? 'auto' : 'smooth' });
        return;
      }
      var b = $('.submit-bar button', form);
      b.firstChild.textContent = 'Working it out '; setTimeout(function () { b.disabled = true; }, 0);
    });
    var bar2 = $('.toolbar', form); if (bar2) bar2.hidden = false;
    var setAll = function (rec) {
      cfg.features.forEach(function (f) {
        var v = rec ? rec[f.name] : '';
        if (f.choices) $$('input[name="' + f.name + '"]', form).forEach(function (r) { r.checked = r.value === v; });
        else $('#' + f.name, form).value = v;
        touched[f.name] = !!rec; show(f.name, '');
      });
      tally();
    };
    $$('[data-sample]', form).forEach(function (b) {
      b.addEventListener('click', function () {
        setAll(cfg.samples[b.getAttribute('data-sample')]);
        var ref = $('#reference', form);
        if (!ref.value || /^Example/.test(ref.value)) ref.value = 'Example, ' + b.textContent.toLowerCase();
      });
    });
    var clr = $('[data-clear]', form);
    if (clr) clr.addEventListener('click', function () { setAll(null); $('#reference', form).value = ''; });
    $$('[data-goto]', form).forEach(function (a) {
      a.addEventListener('click', function (e) {
        var n = a.getAttribute('data-goto'), el = $('#' + n, form) || $('input[name="' + n + '"]', form);
        if (el) { e.preventDefault(); el.focus(); el.scrollIntoView({ block: 'center' }); }
      });
    });
    $$('.field.invalid', form).forEach(function (el) { touched[el.getAttribute('data-field')] = true; });
    var probs = $('#problems'); if (probs) probs.focus();
    tally();
  }

  /* ---------- result: step through the route ---------- */
  var route = $('#route');
  if (route) {
    var rows = $$('tbody tr', route);
    var go = function () { rows.forEach(function (tr, i) { setTimeout(function () { tr.classList.add('on'); }, calm ? 0 : 120 + i * 130); }); };
    if ('IntersectionObserver' in window) {
      var ro = new IntersectionObserver(function (e) { if (e[0].isIntersecting) { ro.disconnect(); go(); } }, { threshold: 0.01 });
      ro.observe(route);
    } else go();
  }

  /* ---------- result: what-if ---------- */
  var wi = $('#whatif');
  if (wi) {
    var base = JSON.parse(wi.getAttribute('data-inputs')), saved = wi.getAttribute('data-label'),
      sliders = $$('[data-w]', wi), out = $('#wi-out'), lab = $('#wi-label'), note = $('#wi-note'), n = 0;
    var current = function () {
      var rec = {}; Object.keys(base).forEach(function (k) { rec[k] = base[k]; });
      sliders.forEach(function (s) { rec[s.getAttribute('data-w')] = Number(s.value); });
      return rec;
    };
    var changed = function () { return sliders.some(function (s) { return Number(s.value) !== Number(base[s.getAttribute('data-w')]); }); };
    var run = function () {
      var mine = ++n;
      if (!changed()) { lab.textContent = saved; note.textContent = 'Same as the saved prediction.'; out.classList.remove('changed'); return; }
      fetch(wi.getAttribute('data-url'), { method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': wi.getAttribute('data-csrf') },
        body: JSON.stringify(current()) })
        .then(function (r) { return r.json(); }).then(function (d) {
          if (mine !== n) return;
          if (!d.ok) { note.textContent = 'That combination is not valid.'; return; }
          lab.textContent = d.label;
          var moved = d.label !== saved;
          out.classList.toggle('changed', moved);
          note.textContent = (moved ? 'Changed from ' + saved + '. ' : 'Still ' + saved + '. ') +
            Math.round(d.confidence * 100) + '% of ' + d.leaf_samples + ' similar records.';
        }).catch(function () { note.textContent = 'Could not reach the model. Check the connection and try again.'; });
    };
    var soon = debounce(run, 140);
    sliders.forEach(function (s) {
      s.addEventListener('input', function () {
        $('output[for="' + s.id + '"]', wi).textContent = s.value + s.getAttribute('data-suffix'); soon();
      });
    });
    $('#wi-reset').addEventListener('click', function () {
      sliders.forEach(function (s) {
        s.value = base[s.getAttribute('data-w')];
        $('output[for="' + s.id + '"]', wi).textContent = s.value + s.getAttribute('data-suffix');
      });
      run();
    });
  }

  /* ---------- admin: draw the learned tree ---------- */
  var tv = $('#tree-view');
  if (tv) {
    var tree = JSON.parse(tv.getAttribute('data-tree')), leaves = [], NS = 'http://www.w3.org/2000/svg';
    var W = 172, H = 56, GX = 14, GY = 54, maxDepth = 0;
    (function layout(n, d) {
      n.depth = d; maxDepth = Math.max(maxDepth, d);
      if (n.yes) { layout(n.yes, d + 1); layout(n.no, d + 1); n.x = (n.yes.x + n.no.x) / 2; }
      else { n.x = leaves.length * (W + GX) + W / 2; leaves.push(n); }
    })(tree, 0);
    var width = leaves.length * (W + GX) - GX, height = (maxDepth + 1) * (H + GY) - GY + 8;
    var svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Top levels of the learned decision tree');
    var el = function (tag, attrs, text) {
      var e = document.createElementNS(NS, tag);
      Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
      if (text != null) e.textContent = text;
      svg.appendChild(e); return e;
    };
    var wrap = function (s, max) {
      var words = s.split(' '), lines = [''];
      words.forEach(function (w) {
        var cur = lines[lines.length - 1];
        if ((cur + ' ' + w).trim().length > max) lines.push(w); else lines[lines.length - 1] = (cur + ' ' + w).trim();
      });
      return lines.slice(0, 2);
    };
    var colors = { 1: '#0B7A47', 2: '#5F9620', 3: '#E0AC00', 4: '#D9661F', 5: '#C8102E' };
    (function draw(n) {
      var y = n.depth * (H + GY) + 4;
      if (n.yes) {
        [['yes', n.yes], ['no', n.no]].forEach(function (pair) {
          var c = pair[1], cy = c.depth * (H + GY) + 4, my = (y + H + cy) / 2;
          el('path', { 'class': 'e', d: 'M' + n.x + ',' + (y + H) + ' C' + n.x + ',' + my + ' ' + c.x + ',' + my + ' ' + c.x + ',' + cy });
          el('text', { 'class': 'yn', x: (n.x + c.x) / 2 + (pair[0] === 'yes' ? -14 : 14), y: my - 4 }, pair[0] === 'yes' ? 'true' : 'false');
        });
      }
      var terminal = n.leaf || n.truncated;
      if (terminal) {
        el('rect', { x: n.x - W / 2, y: y, width: W, height: H, rx: 6, fill: colors[n.tier], stroke: '#141414', 'stroke-width': 1.5 });
        el('text', { 'class': n.tier === 3 ? 'lt dk' : 'lt', x: n.x, y: y + 22 }, 'Mostly tier ' + n.tier + ' (' + Math.round(n.share * 100) + '%)');
        el('text', { 'class': n.tier === 3 ? 'lt dk' : 'lt', x: n.x, y: y + 39, style: 'font-weight:400;font-size:10px' },
          n.samples.toLocaleString() + ' records' + (n.truncated ? ', splits further' : ''));
      } else {
        el('rect', { 'class': 'nb', x: n.x - W / 2, y: y, width: W, height: H, rx: 6 });
        var lines = wrap(n.test, 27);
        lines.forEach(function (l, i) { el('text', { x: n.x, y: y + (lines.length === 1 ? 24 : 18) + i * 14 }, l); });
        el('text', { 'class': 'st', x: n.x, y: y + 48 }, n.samples.toLocaleString() + ' records');
      }
      if (n.yes) { draw(n.yes); draw(n.no); }
    })(tree);
    tv.appendChild(svg);
  }

  /* admin: dataset drop zone and tuning toggle */
  var drop = $('#drop');
  if (drop) {
    var fi = $('input[type=file]', drop), label = $('#drop-name');
    fi.addEventListener('change', function () { if (fi.files[0]) label.textContent = fi.files[0].name; });
    ['dragover', 'dragenter'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('over'); }); });
    ['dragleave', 'drop'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('over'); }); });
    drop.addEventListener('drop', function (e) {
      if (e.dataTransfer.files.length) { fi.files = e.dataTransfer.files; label.textContent = fi.files[0].name; }
    });
  }
  var tune = $('#tune'), manual = $('#manual');
  if (tune && manual) {
    var sync = function () {
      manual.style.opacity = tune.checked ? .4 : 1;
      $$('input', manual).forEach(function (i) { i.disabled = tune.checked; });
    };
    tune.addEventListener('change', sync); sync();
  }
})();
