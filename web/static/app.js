(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const state = { mode: 'video', video: null, features: null, videoUrl: null,
    models: [], selected: new Set(), busy: false, result: null };
  const palette = ['#5de8d0','#9a91ff','#ffbd7b','#79b6ff','#d4a0f8','#9fe08b'];
  const sanitize = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const formatSize = n => n < 1048576 ? `${(n/1024).toFixed(1)} KB` : `${(n/1048576).toFixed(1)} MB`;
  const setMessage = (text, error = false) => { $('run-message').textContent = text; $('run-message').classList.toggle('error', error); };
  const currentFile = () => state.mode === 'video' ? state.video : state.features;

  function updateButton() {
    $('analyze-btn').disabled = state.busy || !currentFile() || state.selected.size === 0;
    $('button-label').textContent = state.busy ? 'Analyzing video and sound…' : `Analyze with ${state.selected.size || 'selected'} model${state.selected.size === 1 ? '' : 's'}`;
    $('button-icon').textContent = state.busy ? '◌' : '✦';
    if (!state.busy && currentFile() && state.selected.size) setMessage('Ready to analyze. All inference runs locally.');
  }

  function toggleMode(mode) {
    state.mode = mode;
    $('tab-video').classList.toggle('selected', mode === 'video');
    $('tab-features').classList.toggle('selected', mode === 'features');
    $('tab-video').setAttribute('aria-selected', String(mode === 'video'));
    $('tab-features').setAttribute('aria-selected', String(mode === 'features'));
    $('video-input-area').classList.toggle('hidden', mode !== 'video');
    $('features-input-area').classList.toggle('hidden', mode !== 'features');
    updateButton();
  }

  function assignFile(kind, file) {
    if (!file) return;
    const feature = kind === 'features';
    const ext = file.name.split('.').pop().toLowerCase();
    if ((feature && ext !== 'npz') || (!feature && !['mp4','mov','mkv','avi','webm'].includes(ext))) {
      setMessage(feature ? 'Select a .npz feature file.' : 'Select a supported video format.', true); return;
    }
    const limit = feature ? 15*1024*1024 : 100*1024*1024;
    if (file.size > limit) { setMessage(`File too large. Maximum is ${feature ? 15 : 100} MB.`, true); return; }
    state[kind] = file;
    $(`${kind}-summary`).classList.remove('hidden');
    $(`${kind}-name`).textContent = file.name;
    $(`${kind}-size`).textContent = formatSize(file.size);
    if (!feature) {
      if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
      state.videoUrl = URL.createObjectURL(file);
      $('video-preview').src = state.videoUrl;
      $('video-preview').classList.remove('hidden');
    }
    updateButton();
  }

  function clearFile(kind) {
    state[kind] = null;
    $(`${kind}-file`).value = '';
    $(`${kind}-summary`).classList.add('hidden');
    if (kind === 'video') {
      $('video-preview').pause();
      $('video-preview').removeAttribute('src');
      $('video-preview').load();
      $('video-preview').classList.add('hidden');
      if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
      state.videoUrl = null;
    }
    updateButton();
  }

  function renderModels() {
    const box = $('model-list'); box.replaceChildren();
    state.models.forEach(model => {
      const label = document.createElement('label');
      label.className = `model-card${state.selected.has(model.id) ? ' selected' : ''}${model.available ? '' : ' disabled'}`;
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox';
      checkbox.checked = state.selected.has(model.id); checkbox.disabled = !model.available;
      checkbox.setAttribute('aria-label', `Select ${model.name}`);
      checkbox.addEventListener('change', () => { if (checkbox.checked) state.selected.add(model.id); else state.selected.delete(model.id); renderModels(); updateButton(); });
      const main = document.createElement('div'); main.innerHTML = `<div class="model-name">${sanitize(model.name)}</div><div class="model-family">${sanitize(model.family)}</div>`;
      const ready = document.createElement('span'); ready.className = model.available ? 'model-ready' : 'model-unavailable'; ready.textContent = model.available ? '● READY' : '○ NO CHECKPOINT';
      label.append(checkbox,main,ready); box.append(label);
    });
    if (!state.models.length) box.innerHTML = '<p class="loading-text">No models registered.</p>';
  }

  async function bootstrap() {
    try {
      const [rModels, rStatus] = await Promise.all([fetch('/api/models'),fetch('/api/status')]);
      if (!rModels.ok || !rStatus.ok) throw new Error('Backend is not responding.');
      const models = await rModels.json(), status = await rStatus.json();
      state.models = models.models;
      const preferred = state.models.find(m => m.id === 'bigru' && m.available) || state.models.find(m => m.available);
      if (preferred) state.selected.add(preferred.id);
      renderModels();
      $('service-status').textContent = 'Local inference service online';
      const assets = status.assets;
      const missing = Object.entries(assets).filter(([,ready]) => !ready).map(([key]) => key.replaceAll('_',' '));
      const featuresReady = status.ready_for_features;
      const html = `<span class="${featuresReady?'ready':'not-ready'}">● ${featuresReady ? 'Trained model checkpoints available' : 'No model checkpoints detected'}</span><br>`+
        `<span class="${status.ready_for_video ? 'ready' : 'not-ready'}">● ${status.ready_for_video ? 'End-to-end video extraction ready' : 'Video extraction needs setup'}</span>`+
        (missing.length ? `<br>Missing video dependencies: ${missing.map(sanitize).join(', ')}` : '');
      $('requirement-details').innerHTML = html;
      if (!status.ready_for_video) setMessage('Video extraction is not fully configured. You can analyze saved .npz features if a checkpoint is available.');
    } catch (error) {
      $('service-status').textContent = 'Service unavailable';
      $('model-list').textContent = 'Could not load model catalog.';
      setMessage(String(error.message), true);
    }
    updateButton();
  }

  function focusVideo(startSec) {
    const video = $('video-preview');
    if (state.mode !== 'video' || !state.video || video.classList.contains('hidden')) return;
    video.currentTime = Number(startSec);
    video.play().catch(() => {});
    video.scrollIntoView({behavior:'smooth', block:'center'});
  }

  function makeTimeline(events, modelIndex) {
    const track = document.createElement('div'); track.className = 'timeline-track';
    events.forEach((event, i) => {
      const btn = document.createElement('button'); btn.className = 'timeline-event'; btn.type = 'button';
      const start = Math.max(0, Math.min(10, Number(event.start_sec)));
      const end = Math.max(start, Math.min(10, Number(event.end_sec)));
      btn.style.left = `${start * 10}%`;
      btn.style.width = `${(end - start) * 10}%`;
      btn.style.background = palette[(i+modelIndex)%palette.length];
      btn.title = `${event.label} · ${start}s–${end}s · ${(event.confidence*100).toFixed(1)}%`;
      btn.setAttribute('aria-label', btn.title);
      btn.addEventListener('click', () => focusVideo(start));
      track.append(btn);
    });
    return track;
  }

  function renderResult(data) {
    state.result = data;
    $('result-placeholder').classList.add('hidden'); $('result-content').classList.remove('hidden');
    $('results-meta').textContent = data.input_type === 'real_video' ? 'REAL VIDEO INFERENCE' : 'PRECOMPUTED FEATURE INFERENCE';
    const entries = Object.entries(data.models || {});
    $('metric-models').textContent = String(entries.length);
    $('metric-events').textContent = String(entries.reduce((n,[,r]) => n + r.event_count, 0));
    $('metric-duration').textContent = `${data.duration_sec.toFixed(1)}s`;
    $('metric-threshold').textContent = data.threshold.toFixed(2);
    const container = $('model-results'); container.replaceChildren();
    entries.forEach(([id, result], modelIndex) => {
      const card = document.createElement('article'); card.className = 'result-model-card';
      const header = document.createElement('div'); header.className = 'result-header';
      header.innerHTML = `<div><h3>${sanitize(result.name)}</h3><small>${sanitize(result.family||id)} · ${sanitize(result.checkpoint||'')}</small></div><span class="result-count">${result.event_count} DETECTED</span>`;
      card.append(header);
      const ticks = document.createElement('div'); ticks.className='timeline-labels'; ticks.innerHTML='<span>00:00</span><span>00:02</span><span>00:04</span><span>00:06</span><span>00:08</span><span>00:10</span>'; card.append(ticks);
      card.append(makeTimeline(result.events,modelIndex));
      if (!result.events.length) {
        const empty = document.createElement('p'); empty.className = 'empty-result'; empty.textContent = 'No event exceeded the selected threshold.'; card.append(empty);
      } else {
        const grid = document.createElement('div'); grid.className='events-grid';
        result.events.forEach((event,i) => {
          const btn=document.createElement('button');btn.className='event-card';btn.type='button';
          btn.innerHTML = `<div class="event-top"><span class="event-name"><span class="event-swatch" style="background:${palette[(i+modelIndex)%palette.length]}"></span>${sanitize(event.label)}</span><span class="event-confidence">${(event.confidence*100).toFixed(1)}%</span></div><div class="event-time">◷ ${Number(event.start_sec).toFixed(0)}s – ${Number(event.end_sec).toFixed(0)}s · Click to seek</div>`;
          btn.addEventListener('click',()=>focusVideo(event.start_sec));grid.append(btn);
        });
        card.append(grid);
      }
      container.append(card);
    });
    $('results').scrollIntoView({behavior:'smooth',block:'start'});
  }

  async function analyze() {
    if (state.busy || !currentFile() || !state.selected.size) return;
    state.busy = true; updateButton(); setMessage('Running local model inference…');
    const params = new FormData(); params.append('file', currentFile());
    params.append('model_ids', Array.from(state.selected).join(','));
    params.append('threshold', $('threshold').value);
    const endpoint = state.mode === 'video' ? '/api/analyze-video' : '/api/analyze-features';
    try {
      const res = await fetch(endpoint,{method:'POST',body:params});
      let data; try{data=await res.json();}catch{throw new Error(`Server returned HTTP ${res.status}`);}
      if (!res.ok) {const detail=Array.isArray(data.detail)?data.detail.map(x=>x.msg).join('; '):data.detail; throw new Error(detail || `HTTP ${res.status}`);}
      renderResult(data); setMessage('Analysis complete. Results represent model predictions, not verified labels.');
    } catch (error) {setMessage(error.message || 'Analysis failed. Check the server log.',true);}
    finally {state.busy=false;updateButton();}
  }

  $('tab-video').addEventListener('click',()=>toggleMode('video'));
  $('tab-features').addEventListener('click',()=>toggleMode('features'));
  $('video-file').addEventListener('change',e=>assignFile('video',e.target.files[0]));
  $('features-file').addEventListener('change',e=>assignFile('features',e.target.files[0]));
  $('clear-video').addEventListener('click',()=>clearFile('video'));
  $('clear-features').addEventListener('click',()=>clearFile('features'));
  ['video','features'].forEach(kind=>{
    const drop=$(kind==='video'?'drop-zone':'features-drop-zone');
    drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('drag-over');});
    drop.addEventListener('dragleave',()=>drop.classList.remove('drag-over'));
    drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('drag-over');assignFile(kind,e.dataTransfer.files[0]);});
    drop.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$(kind+'-file').click();}});
  });
  $('threshold').addEventListener('input',e=>{$('threshold-label').textContent=`${Math.round(Number(e.target.value)*100)}%`;});
  $('analyze-btn').addEventListener('click',analyze);
  bootstrap();
})();
