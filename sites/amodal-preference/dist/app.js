/* Browser-only rating workspace. Method identities never enter this application. */
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const study = window.PREFERENCE_STUDY;
  const core = window.PreferenceCore;
  const showNotice = (message, error = false) => { $('notice').hidden = false; $('notice').textContent = message; $('notice').className = 'notice' + (error ? ' error' : ''); };
  if (!study || !core || !study.cases?.length) {
    showNotice('The frozen study could not load. Keep study.js and all assets beside this page, then reload.', true);
    document.body.classList.add('load-error');
    return;
  }
  const storageKey = 'amodal-preference:' + study.study_id;
  let mountedFiles = null;
  const mountedURLs = new Map();
  const hosted = location.protocol === 'https:' || location.protocol === 'http:';
  let state = {reviewer: {code: '', role: '', prior_exposure: ''}, records: {}, index: 0};
  let storageOK = true, conflict = false, lastStored = null, generation = 0, loaded = false;
  let pendingRaterAction = null;
  async function assetURL(path) {
    if (!mountedFiles) {
      return path;
    }
    if (mountedURLs.has(path)) return mountedURLs.get(path);
    const file = mountedFiles.get(path);
    if (!file) throw new Error('Missing study image: ' + path);
    const expected = window.PREFERENCE_HASHES?.[path];
    if (!expected || !crypto.subtle) throw new Error('Image verification is unavailable. Use the offline reviewer page.');
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', await file.arrayBuffer())), b => b.toString(16).padStart(2, '0')).join('');
    if (digest !== expected) throw new Error('Study image checksum mismatch. Restore the original packet.');
    const url = URL.createObjectURL(file);
    mountedURLs.set(path, url);
    // Keep a small cache as the rater moves through the full census.
    if (mountedURLs.size > 25) {
      const oldest = mountedURLs.keys().next().value;
      URL.revokeObjectURL(mountedURLs.get(oldest)); mountedURLs.delete(oldest);
    }
    return url;
  }
  const empty = () => ({case_id: study.cases[state.index].id, status: 'draft',
    judgements: {overall: null, shape: null, surface: null}, note: '', confidence: '', skip_reason: '',
    created_at: new Date().toISOString(), updated_at: new Date().toISOString()});
  const current = () => state.records[study.cases[state.index].id];
  const payload = () => ({schema: core.SCHEMA, study_id: study.study_id, round: study.round,
    reviewer: state.reviewer, exported_at: new Date().toISOString(),
    complete: study.cases.every(c => state.records[c.id] && state.records[c.id].status !== 'draft'),
    records: study.cases.map(c => state.records[c.id]).filter(Boolean)});
  try {
    lastStored = localStorage.getItem(storageKey);
    if (lastStored) {
      const saved = JSON.parse(lastStored), valid = core.validate(saved.response, study);
      state = {reviewer: valid.reviewer, records: Object.fromEntries(valid.records.map(r => [r.case_id, r])),
        index: Number.isInteger(saved.index) && saved.index >= 0 && saved.index < study.cases.length ? saved.index : 0};
    }
  } catch (error) {
    storageOK = false;
    showNotice('Saved data could not be loaded: ' + error.message + ' Existing storage is preserved. Export this session or restore a valid JSON backup.', true);
  }
  function persist() {
    if (conflict) { showNotice('Another tab changed this study. Export this tab before reloading to resolve the conflict.', true); return; }
    if (!storageOK) { $('save-status').textContent = 'Browser saving unavailable. Export JSON before leaving.'; return; }
    try {
      if (localStorage.getItem(storageKey) !== lastStored) { conflict = true; persist(); return; }
      lastStored = JSON.stringify({response: payload(), index: state.index});
      localStorage.setItem(storageKey, lastStored);
      $('save-status').textContent = 'Saved in this browser · ' + new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) + '. Export after each session.';
    } catch (error) {
      storageOK = false;
      showNotice('Browser saving is unavailable. Your current work is still in memory. Export JSON before closing this page.', true);
      $('save-status').textContent = 'Not saved to browser. Export JSON now.';
    }
  }
  window.addEventListener('storage', event => {
    if (event.key === storageKey && event.newValue !== lastStored) { conflict = true; showNotice('Another tab changed this study. Export this tab before reloading; further saves will not overwrite the other tab.', true); }
  });
  function mutate(callback) {
    const record = current() ? structuredClone(current()) : empty();
    callback(record);
    record.status = 'draft'; record.skip_reason = ''; record.updated_at = new Date().toISOString();
    state.records[record.case_id] = record;
    persist(); updateProgress(); renderChoices();
  }
  function choose(criterion, slot) {
    if (!loaded) return;
    mutate(record => {
      const prior = record.judgements[criterion];
      const winners = prior?.kind === 'preference' ? [...prior.winners] : [];
      const index = winners.indexOf(slot);
      if (index < 0) winners.push(slot); else winners.splice(index, 1);
      record.judgements[criterion] = winners.length ? {kind: 'preference', winners: winners.sort()} : null;
    });
  }
  function setJudgement(criterion, value) { if (loaded) mutate(record => { record.judgements[criterion] = value; }); }
  function addButton(parent, text, handler, pressed = false) {
    const button = document.createElement('button'); button.textContent = text;
    button.type = 'button'; button.setAttribute('aria-pressed', String(pressed));
    button.addEventListener('click', handler); parent.append(button); return button;
  }
  function renderChoices() {
    const record = current();
    core.CRITERIA.forEach(criterion => {
      const parent = $(criterion + '-controls'); parent.replaceChildren();
      const value = record?.judgements[criterion];
      for (const slot of study.slots) addButton(parent, 'Option ' + slot, () => choose(criterion, slot), value?.winners.includes(slot));
      addButton(parent, 'All equal', () => setJudgement(criterion, {kind: 'preference', winners: [...study.slots]}), value?.kind === 'preference' && value.winners.length === study.slots.length);
      addButton(parent, 'Cannot judge', () => setJudgement(criterion, {kind: 'unassessable', winners: []}), value?.kind === 'unassessable');
      if (criterion !== 'overall') addButton(parent, 'Clear', () => setJudgement(criterion, null));
      for (const button of parent.children) button.disabled = !loaded;
    });
    document.querySelectorAll('.candidate').forEach(card => {
      const selected = record?.judgements.overall?.winners.includes(card.dataset.slot) || false;
      card.classList.toggle('selected', selected);
      card.querySelector('.pick').textContent = selected ? 'Selected' : 'Prefer ' + card.dataset.slot;
      card.querySelector('.pick').setAttribute('aria-pressed', String(selected));
      card.querySelector('.image-select').setAttribute('aria-pressed', String(selected));
      card.querySelector('.pick').disabled = !loaded;
      card.querySelector('.image-select').disabled = !loaded;
    });
    $('case-state').textContent = !record ? 'Unreviewed' : ({rated: 'Saved', skipped: 'Skipped', draft: 'Draft · save to finish'})[record.status];
    $('save-next').disabled = !loaded || !record?.judgements.overall;
  }
  function filteredCases() {
    return study.cases.filter(c => {
      const status = state.records[c.id]?.status;
      return $('filter').value === 'all' || ($('filter').value === 'unreviewed' ? !status || status === 'draft' : status === $('filter').value);
    });
  }
  function updateProgress() {
    const records = Object.values(state.records), rated = records.filter(r => r.status === 'rated').length,
      skipped = records.filter(r => r.status === 'skipped').length;
    $('progress-fraction').textContent = `${rated + skipped} / ${study.cases.length}`;
    $('progress').value = rated + skipped; $('progress').max = study.cases.length;
    $('progress-detail').textContent = `${rated} rated · ${skipped} skipped · ${study.cases.length - rated - skipped} remaining`;
    const picker = $('case-picker'); picker.replaceChildren();
    let rows = filteredCases();
    if (!rows.some(c => c.id === study.cases[state.index].id)) rows = [study.cases[state.index], ...rows];
    rows.forEach(c => {
      const option = document.createElement('option'); option.value = c.id;
      option.textContent = `${c.id} · ${c.label} · ${state.records[c.id]?.status || 'unreviewed'}`;
      picker.append(option);
    });
    picker.value = study.cases[state.index].id;
  }
  function loadImage(element, url, token, onReady) {
    if (window.PREFERENCE_LOCAL_IMAGES && !mountedFiles) return;
    element.onload = () => { if (generation === token && element.naturalWidth) onReady(); };
    element.onerror = () => {
      if (generation === token) { loaded = false; renderChoices(); $('folder-bar').hidden = false; $('asset-status').textContent = 'An image did not load. Open the extracted study folder, reload, or skip with “Image failed to load”. Rating is disabled.'; }
    };
    assetURL(url).then(resolved => { if (generation === token) element.src = resolved; }).catch(error => {
      if (generation === token) { loaded = false; renderChoices(); $('asset-status').textContent = error.message; }
    });
  }
  function render() {
    const item = study.cases[state.index], token = ++generation;
    loaded = false; let remaining = 2 + study.slots.length;
    const needsFolder = window.PREFERENCE_LOCAL_IMAGES && !mountedFiles;
    $('asset-status').textContent = needsFolder ? 'Open study folder to load your images and begin rating.' : 'Loading comparison images…';
    $('skip-button').disabled = Boolean(needsFolder);
    const ready = () => { if (--remaining === 0) { loaded = true; $('asset-status').textContent = ''; if (!mountedFiles) $('folder-bar').hidden = true; renderChoices(); } };
    $('case-position').textContent = `${item.id} · CASE ${state.index + 1} OF ${study.cases.length} · BATCH ${Math.floor(state.index / study.batch_size) + 1}`;
    $('target-label').textContent = item.label;
    loadImage($('context-image'), item.files.context, token, ready);
    loadImage($('visible-image'), item.files.visible, token, ready);
    const cards = $('candidates'); cards.replaceChildren();
    cards.style.setProperty('--option-count', study.slots.length);
    for (const slot of study.slots) {
      const card = document.createElement('article'); card.className = 'candidate'; card.dataset.slot = slot;
      const header = document.createElement('div'); header.className = 'candidate-head';
      const title = document.createElement('strong'); title.textContent = 'Option ' + slot;
      const subtitle = document.createElement('span'); subtitle.textContent = 'Anonymous completion'; header.append(title, subtitle);
      const imageButton = document.createElement('button'); imageButton.className = 'image-select'; imageButton.setAttribute('aria-label', 'Prefer option ' + slot);
      imageButton.addEventListener('click', () => choose('overall', slot));
      const img = document.createElement('img'); img.alt = 'Completion option ' + slot; imageButton.append(img);
      const footer = document.createElement('div'); footer.className = 'candidate-foot';
      const pick = addButton(footer, 'Prefer ' + slot, () => choose('overall', slot)); pick.className = 'pick';
      const inspect = addButton(footer, 'Inspect', () => zoom('Option ' + slot, [[slot, item.files[slot]]])); inspect.className = 'quiet';
      card.append(header, imageButton, footer); cards.append(card);
      loadImage(img, item.files[slot], token, ready);
    }
    $('notes').value = current()?.note || ''; $('confidence').value = current()?.confidence || '';
    if (current()?.judgements.shape || current()?.judgements.surface) $('optional-details').open = true;
    $('previous').disabled = state.index === 0; $('next').disabled = state.index === study.cases.length - 1;
    updateProgress(); renderChoices();
    if (!storageOK) $('save-status').textContent = 'Browser saving unavailable. Export JSON before leaving.';
  }
  function go(index) { if (index < 0 || index >= study.cases.length) return; state.index = index; persist(); render(); }
  function nextUnreviewed() {
    for (let offset = 0; offset < study.cases.length; offset++) {
      const i = (state.index + offset) % study.cases.length;
      if (!state.records[study.cases[i].id] || state.records[study.cases[i].id].status === 'draft') { go(i); return; }
    }
    finish();
  }
  function withRater(action) {
    try { core.reviewer(state.reviewer, true); action(); }
    catch (_) { pendingRaterAction = action; openRater(); }
  }
  function saveCase(status, reason = '') {
    if (status === 'rated' && !loaded) return;
    withRater(() => {
      try {
        const record = current() ? structuredClone(current()) : empty();
        record.status = status; record.skip_reason = reason; record.updated_at = new Date().toISOString();
        if (status === 'skipped') { record.judgements = {overall: null, shape: null, surface: null}; record.confidence = ''; }
        state.records[record.case_id] = core.record(record, study);
        persist(); updateProgress();
        const completed = Object.values(state.records).filter(r => r.status !== 'draft').length;
        if (completed === study.cases.length) { renderChoices(); finish(); }
        else { nextUnreviewed(); if (completed > 0 && completed % study.batch_size === 0) showNotice(`${completed} cases saved. A good time for a break and a JSON backup.`); }
      } catch (error) { showNotice(error.message, true); }
    });
  }
  function download(kind) {
    try {
      const data = core.validate(payload(), study);
      const text = kind === 'csv' ? core.csv(data) : JSON.stringify(data, null, 2);
      const url = URL.createObjectURL(new Blob([text], {type: kind === 'csv' ? 'text/csv;charset=utf-8' : 'application/json'}));
      const link = document.createElement('a'); link.href = url;
      const code = (state.reviewer.code || 'draft').replace(/[^a-zA-Z0-9_-]/g, '_');
      link.download = `${study.round}-${code}-${new Date().toISOString().replace(/[:.]/g, '-')}.${kind}`;
      document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { showNotice('Export failed: ' + error.message, true); }
  }
  function finish() {
    const p = payload(), rated = p.records.filter(r => r.status === 'rated').length, skipped = p.records.filter(r => r.status === 'skipped').length;
    $('finish-title').textContent = p.complete ? 'Round complete' : 'Your session is ready to export';
    $('finish-text').textContent = `${rated} rated, ${skipped} skipped, ${study.cases.length - rated - skipped} remaining. Skips and “cannot judge” are retained separately from preferences.`;
    $('finish-dialog').showModal();
  }
  function openRater() {
    $('rater-code').value = state.reviewer.code; $('rater-role').value = state.reviewer.role; $('rater-exposure').value = state.reviewer.prior_exposure;
    $('rater-dialog').showModal();
  }
  async function zoom(title, images) {
    if (!loaded) { showNotice('Load the study images before opening the inspection view.'); return; }
    $('zoom-title').textContent = title; $('zoom-images').replaceChildren();
    for (const [label, path] of images) {
      const figure = document.createElement('figure'), caption = document.createElement('figcaption'), scroll = document.createElement('div'), img = document.createElement('img');
      caption.textContent = label; scroll.className = 'image-scroll'; img.alt = label;
      try { img.src = await assetURL(path); } catch (error) { showNotice(error.message, true); return; }
      scroll.append(img); figure.append(caption, scroll); $('zoom-images').append(figure);
    }
    $('zoom-dialog').showModal();
  }
  $('round-title').textContent = study.title;
  $('round-count').textContent = `${study.cases.length.toLocaleString()} matched objects · ${study.slots.length} methods`;
  $('study-scope').textContent = `This round covers every shared successful output: ${study.eligible.toLocaleString()} of ${study.total_targets.toLocaleString()} targets. ${study.excluded.toLocaleString()} targets without outputs from every method are excluded from preference scoring and tracked separately. This is a single-rater assessment. ${study.display}`;
  for (const round of study.pending) { const el = document.createElement('div'); el.className = 'pending'; const h = document.createElement('h3'), p = document.createElement('p'); h.textContent = round.name; p.textContent = round.reason; el.append(h,p); $('pending-rounds').append(el); }
  $('help-button').onclick = () => $('help-dialog').showModal();
  $('export-button').onclick = finish;
  $('finish-export').onclick = () => download('json'); $('csv-button').onclick = () => download('csv');
  $('rater-button').onclick = () => { pendingRaterAction = null; openRater(); };
  $('rater-form').onsubmit = event => {
    event.preventDefault();
    try { state.reviewer = core.reviewer({code: $('rater-code').value, role: $('rater-role').value, prior_exposure: $('rater-exposure').value}, true); persist(); $('rater-dialog').close(); const action = pendingRaterAction; pendingRaterAction = null; if (action) action(); }
    catch (error) { showNotice(error.message, true); }
  };
  $('notes').oninput = () => mutate(r => { r.note = $('notes').value; });
  $('confidence').onchange = () => mutate(r => { r.confidence = $('confidence').value; });
  $('previous').onclick = () => go(state.index - 1); $('next').onclick = () => go(state.index + 1);
  $('next-unrated').onclick = nextUnreviewed; $('save-next').onclick = () => saveCase('rated');
  $('filter').onchange = updateProgress; $('case-picker').onchange = () => go(study.cases.findIndex(c => c.id === $('case-picker').value));
  $('skip-button').onclick = () => $('skip-dialog').showModal();
  $('skip-form').onsubmit = event => { event.preventDefault(); $('skip-dialog').close(); saveCase('skipped', $('skip-reason').value); };
  $('context-zoom').onclick = () => zoom('Input context', [['Input context', study.cases[state.index].files.context]]);
  $('visible-zoom').onclick = () => zoom('Visible target', [['SAM visible reference', study.cases[state.index].files.visible]]);
  $('scene-zoom').onclick = () => zoom('Full input image', [['Full input image', study.cases[state.index].files.scene]]);
  $('compare-zoom').onclick = () => zoom('Compare at native resolution', study.slots.map(s => ['Option ' + s, study.cases[state.index].files[s]]));
  document.querySelectorAll('[data-close]').forEach(button => { button.onclick = () => $(button.dataset.close).close(); });
  $('import-button').onclick = () => $('import-file').click();
  $('folder-button').onclick = () => $('folder-input').click();
  $('folder-input').onchange = async event => {
    const files = Array.from(event.target.files);
    try {
      const manifest = files.find(f => f.name === 'study.json');
      if (!manifest) throw new Error('Choose the folder containing index.html, study.json, and assets.');
      const selected = JSON.parse(await manifest.text());
      if (selected.study_id !== study.study_id) throw new Error('This folder belongs to a different round.');
      const mapped = new Map();
      for (const file of files) {
        const index = file.webkitRelativePath.lastIndexOf('/assets/');
        if (index >= 0) {
          const path = file.webkitRelativePath.slice(index + 1);
          if (mapped.has(path)) throw new Error('Duplicate image paths. Select one extracted reviewer folder.');
          mapped.set(path, file);
        }
      }
      const needed = study.cases.flatMap(c => Object.values(c.files));
      const missing = needed.filter(path => !mapped.has(path));
      if (missing.length) throw new Error(`${missing.length} images are missing. Extract the entire reviewer packet.`);
      for (const url of mountedURLs.values()) URL.revokeObjectURL(url);
      mountedURLs.clear(); mountedFiles = mapped;
      $('folder-status').textContent = `${needed.length.toLocaleString()} images ready. Each image is checksum-verified when opened. Nothing is uploaded.`;
      $('folder-button').textContent = 'Change image folder';
      render();
    } catch (error) { showNotice(error.message, true); }
    event.target.value = '';
  };
  $('import-file').onchange = async event => {
    const file = event.target.files[0]; event.target.value = ''; if (!file) return;
    try {
      if (file.size > 20 * 1024 * 1024) throw new Error('Backup exceeds the 20 MB limit.');
      const valid = core.validate(JSON.parse(await file.text()), study);
      if (!window.confirm(`Restore ${valid.records.length} case records from ${file.name}? This replaces this browser's current round. Export your current work first if needed.`)) return;
      state = {reviewer: valid.reviewer, records: Object.fromEntries(valid.records.map(r => [r.case_id, r])), index: 0};
      conflict = false;
      try { lastStored = localStorage.getItem(storageKey); storageOK = true; } catch (_) { storageOK = false; }
      persist(); render(); showNotice('Backup restored. This round and its images were verified.');
    } catch (error) { showNotice('Backup was not imported: ' + error.message, true); }
  };
  window.addEventListener('beforeunload', event => { if ((!storageOK || conflict) && Object.keys(state.records).length) { event.preventDefault(); event.returnValue = ''; } });
  if (!hosted) $('folder-bar').hidden = true;
  render();
})();
