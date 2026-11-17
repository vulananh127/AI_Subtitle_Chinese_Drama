const $ = s => document.querySelector(s);
const video = $('#video'), listEl = $('#list');
let cues = [], sel = null, seq = 1, ws = null, regions = null, file = null;
let loop = false, seg = null, hist = [], hi = -1, pend = null;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const byId = id => cues.find(c => c.id === id);
const setStatus = t => $('#status').textContent = t;

function fmt(t){ t=Math.max(0,t); const m=Math.floor(t/60), s=t-m*60; return String(m).padStart(2,'0')+':'+s.toFixed(3).padStart(6,'0'); }
function parse(v){ const p=String(v).trim().split(':').map(Number); if(p.some(isNaN)) return NaN; return p.reduce((a,x)=>a*60+x,0); }
function ts(t,sep){ const ms=Math.round(t*1000), h=Math.floor(ms/3600000), m=Math.floor(ms/60000)%60, s=Math.floor(ms/1000)%60, r=ms%1000;
  const z=(n,l=2)=>String(n).padStart(l,'0'); return `${z(h)}:${z(m)}:${z(s)}${sep}${z(r,3)}`; }

/* ---------- Tải video + waveform ---------- */
let proj = null, rev = 0;
function loadCues(p){
  cues = p.cues; rev = p.rev; sel = null; seg = null; hist = []; hi = -1;
  seq = Math.max(seq, ...cues.map(c => parseInt(String(c.id).replace(/\D/g, '')) || 0)) + 1;
  renderList(); drawRegions(); snap();
}
async function openProject(p){
  proj = p; file = { name: p.title }; video.pause();
  if(ws){ ws.destroy(); ws = null; } regions = null;
  loadCues(p);
  video.src = p.video_url;
  const pk = await loadPeaks(p); if(proj !== p) return;   // người dùng đã chuyển sang dự án khác
  regions = WaveSurfer.Regions.create();
  ws = WaveSurfer.create({ container:'#wave', media:video, height:88, normalize:true, hideScrollbar:true, minPxPerSec:+$('#zoom').value,
    waveColor:css('--wave'), progressColor:css('--prog'), cursorColor:css('--accent'), plugins:[regions],
    ...(pk ? { peaks:[pk.data], duration: pk.duration } : {}) });   // có peaks từ server thì không phải giải mã video
  ws.on('decode', drawRegions); ws.on('decode', syncView); ws.on('scroll', syncView); ws.on('zoom', syncView); ws.on('redrawcomplete', syncView); ws.on('ready', syncView);
  ws.on('error', e => setStatus('Không dựng được waveform: ' + ((e && e.message) || e)));
  regions.on('region-updated', r => { const c=byId(r.id); if(!c) return; c.start=r.start; c.end=r.end; syncTimes(c); snap(); });
  regions.on('region-clicked', (r,ev) => { ev.stopPropagation(); select(r.id,true); playCue(byId(r.id)); });
  ['run','exp','imp','hist','addnow','playsel','split','merge'].forEach(i => $('#'+i).disabled = false);
  $('#fname').textContent = p.title; $('#fn').value = ''; setSaved(''); setStatus(''); lastAuto = ''; ovSig = '';
  $('#viewlang').innerHTML = `<option value="text">Xem: ${esc(langName(p.tgt_lang))}</option><option value="src">Xem: ${esc(langName(p.src_lang))}</option>`;
}
$('#zoom').oninput = e => { if(ws){ ws.zoom(+e.target.value); requestAnimationFrame(syncView); } };

const ppsNow = () => { const sc = scroller(); return sc && video.duration ? sc.scrollWidth / video.duration : +$('#zoom').value; };
const tierOf = pps => pps < 4 ? 0 : pps < 60 ? 1 : 2;   // 0: không vẽ khối cue (đã có bản đồ cue trên thước) · 1: khối không chữ · 2: khối có chữ
let lodTier = -1;
function drawRegions(){
  if(!regions) return; regions.clearRegions();
  lodTier = tierOf(ppsNow()); if(lodTier === 0) return;
  cues.forEach(c => regions.addRegion({ id:c.id, start:c.start, end:c.end, content: lodTier === 2 ? c.text.slice(0, 40) : '', color:regColor(c), drag:true, resize:true }));
}

/* ---------- Dịch sơ (ASR + MT) qua web backend -> model server ---------- */
$('#run').onclick = async () => {
  if(cues.length && !confirm('Phụ đề tự động sẽ thay thế toàn bộ cue hiện có (vẫn khôi phục được trong Lịch sử). Tiếp tục?')) return;
  const btn = $('#run'); btn.disabled = true; setStatus('Đang gửi tới model…');
  try{
    if(!(await flush())) throw new Error('Chưa lưu được thay đổi nên đã hủy việc tạo phụ đề tự động.');
    await api(`/api/projects/${proj.id}/translate`, { method:'POST' });
    for(;;){
      await sleep(1500);
      const j = await api(`/api/projects/${proj.id}/job`);
      if(j.status === 'failed') throw new Error(j.error || 'Job thất bại');
      if(j.status === 'done') break;
      setStatus(`${j.stage || 'Đang xử lý'}… ${Math.round((j.progress || 0) * 100)}%`);
    }
    loadCues(await api(`/api/projects/${proj.id}`)); setSaved('Đã lưu'); setStatus(`Đã tạo ${cues.length} cue.`);
  }catch(err){ setStatus('Lỗi: ' + err.message); }
  btn.disabled = false;
};

/* ---------- Danh sách cue ---------- */
/* ---------- Trạng thái duyệt + nguồn gốc cue ---------- */
const FMT_DEF = { pos:'bot', align:'center', b:false, i:false, color:'#ffffff' };
const fmtOf = c => ({ ...FMT_DEF, ...(c.fmt || {}) });
const hasFmt = c => { const f = fmtOf(c); return f.pos !== 'bot' || f.align !== 'center' || f.b || f.i || f.color !== FMT_DEF.color; };
function setFmt(c, patch){
  const f = { ...fmtOf(c), ...patch }, o = {};   // chỉ lưu giá trị khác mặc định
  if(f.pos !== 'bot') o.pos = f.pos; if(f.align !== 'center') o.align = f.align; if(f.b) o.b = true; if(f.i) o.i = true;
  if(f.color !== FMT_DEF.color) o.color = f.color;
  c.fmt = o; const row = listEl.querySelector(`[data-id="${c.id}"]`); row && paintRow(row, c); snap();
}
function fmtChange(el, c){
  if(el.classList.contains('f-pos')) setFmt(c, { pos: el.value });
  else if(el.classList.contains('f-al')) setFmt(c, { align: el.value });
  else if(el.classList.contains('f-c')) setFmt(c, { color: el.value });
}
function fmtClick(el, c){
  const b = el.closest('.f-b, .f-i'); if(!b) return;
  setFmt(c, b.classList.contains('f-b') ? { b: !fmtOf(c).b } : { i: !fmtOf(c).i });
}
const tagOf = c => ({ model:'Model', import:'Import', manual:'Thủ công' }[c.origin] || '') + (c.mt && c.text !== c.mt ? ' · đã sửa' : '');
const regColor = c => c.ok ? css(c.id === sel ? '--regoksel' : '--regok') : css(c.id === sel ? '--regsel' : '--reg');
function paintRow(el, c){
  el.classList.toggle('ok', !!c.ok);
  const b = el.querySelector('.okb'); b.setAttribute('aria-pressed', !!c.ok); b.textContent = c.ok ? 'Đã duyệt' : 'Duyệt';
  el.querySelector('.tag').textContent = tagOf(c);
  const ff = fmtOf(c); el.querySelector('.f-pos').value = ff.pos; el.querySelector('.f-al').value = ff.align;
  el.querySelector('.f-b').setAttribute('aria-pressed', ff.b); el.querySelector('.f-i').setAttribute('aria-pressed', ff.i); el.querySelector('.f-c').value = ff.color;
  const f = $('#filt').value; el.classList.toggle('hide', (f === 'todo' && c.ok) || (f === 'done' && !c.ok));
}
function updateProg(){ $('#prog').textContent = `Đã duyệt ${cues.filter(c => c.ok).length}/${cues.length}`; drawRuler(); }
function setOk(c, v){
  c.ok = v; paintRow(listEl.querySelector(`[data-id="${c.id}"]`), c); updateProg();
  const r = regions && regions.getRegions().find(r => r.id === c.id); r && r.setOptions({ color: regColor(c) });
  snap();
}

function renderList(){
  listEl.hidden = !cues.length; $('#empty').hidden = !!cues.length;
  listEl.innerHTML = '';
  cues.forEach((c,i) => {
    const el = document.createElement('div');
    el.className = 'cue' + (c.id===sel?' sel':''); el.dataset.id = c.id;
    el.innerHTML = `<div class="times"><div class="no">#${i+1}</div><input class="t s" aria-label="Bắt đầu"><input class="t e" aria-label="Kết thúc"><span class="tag"></span></div>
      <div class="txt"><textarea class="src" rows="1" placeholder="Văn bản nguồn (ASR)"></textarea><textarea class="tx" rows="2" placeholder="Bản dịch"></textarea></div>
      <div class="act"><button data-a="ok" class="okb" aria-pressed="false">Duyệt</button><button data-a="asr">Nghe lại</button><button data-a="mt">Dịch lại</button><button data-a="add">Thêm sau</button><button data-a="del" class="del">Xóa</button></div>
      <div class="fmt"><div class="tag">Định dạng</div>
       <select class="f-pos" aria-label="Vị trí"><option value="bot">Dưới</option><option value="mid">Giữa</option><option value="top">Trên</option></select>
       <select class="f-al" aria-label="Căn lề"><option value="center">Căn giữa</option><option value="left">Căn trái</option><option value="right">Căn phải</option></select>
       <div class="f-sty"><button class="f-b" aria-pressed="false" title="Đậm"><b>B</b></button><button class="f-i" aria-pressed="false" title="Nghiêng"><i>I</i></button><input type="color" class="f-c" title="Màu chữ"></div></div>`;
    el.querySelector('.s').value = fmt(c.start); el.querySelector('.e').value = fmt(c.end);
    el.querySelector('.src').value = c.src; el.querySelector('.tx').value = c.text; paintRow(el, c);
    listEl.appendChild(el);
  });
  updateProg();
}
listEl.addEventListener('click', e => {
  const row = e.target.closest('.cue'); if(!row) return; const c = byId(row.dataset.id);
  if(e.target.closest('.fmt')){ fmtClick(e.target, c); return; }
  const a = e.target.dataset.a;
  if(a==='asr') redoAsr(c, e.target);
  else if(a==='mt') redoMt(c, e.target);
  else if(a==='add') addCue(c.end, c.id);
  else if(a==='del') delCue(c.id);
  else if(a==='ok') setOk(c, !c.ok);
  if(!a && !e.target.classList.contains('t')){ const first = sel !== c.id; select(c.id, false); if(first) playCue(c); }
});
listEl.addEventListener('input', e => {
  const c = byId(e.target.closest('.cue').dataset.id);
  if(e.target.classList.contains('tx')){ c.text = e.target.value; const r = regions.getRegions().find(r=>r.id===c.id); r && r.setOptions({content: lodTier === 2 ? c.text.slice(0,40) : ''}); }
  if(e.target.classList.contains('src')) c.src = e.target.value;
  e.target.closest('.cue').querySelector('.tag').textContent = tagOf(c);
  snapSoon();
});
listEl.addEventListener('change', e => {
  const row = e.target.closest('.cue'), c = byId(row.dataset.id);
  if(e.target.closest('.fmt')){ fmtChange(e.target, c); return; }
  if(!e.target.classList.contains('t')) return;
  const s = parse(row.querySelector('.s').value), en = parse(row.querySelector('.e').value);
  if(isNaN(s) || isNaN(en) || en <= s){ setStatus('Thời gian không hợp lệ (kết thúc phải sau bắt đầu).'); syncTimes(c); return; }
  c.start = s; c.end = en; cues.sort((a,b)=>a.start-b.start); renderList(); drawRegions(); select(c.id,false); snap();
});
function syncTimes(c){ const row = listEl.querySelector(`[data-id="${c.id}"]`); if(!row) return; row.querySelector('.s').value = fmt(c.start); row.querySelector('.e').value = fmt(c.end); }

// Chỉ cuộn bên trong danh sách cue, không dùng scrollIntoView (có thể cuộn/giật cả trang khi đang phát)
function showRow(row){
  if(!row) return; const t = row.offsetTop, b = t + row.offsetHeight;
  if(t < listEl.scrollTop) listEl.scrollTop = t - 6;
  else if(b > listEl.scrollTop + listEl.clientHeight) listEl.scrollTop = b - listEl.clientHeight + 6;
}
function select(id, seek, follow){
  sel = id;
  listEl.querySelectorAll('.cue').forEach(r => r.classList.toggle('sel', r.dataset.id===id));
  regions && regions.getRegions().forEach(r => { const k = byId(r.id); k && r.setOptions({ color: regColor(k) }); });
  const c = byId(id); if(!c) return;
  if(!follow && ws && ws.setScrollTime) ws.setScrollTime(Math.max(0, c.start - 1));
  if(seek){ video.currentTime = c.start; showRow(listEl.querySelector(`[data-id="${id}"]`)); }
}
function addCue(start, afterId){
  const d = video.duration || start+2, c = { id:'c'+(seq++), start:Math.min(start,d), end:Math.min(start+2,d), src:'', text:'', mt:'', ok:false, origin:'manual' };
  if(c.end <= c.start){ c.start = Math.max(0,d-2); c.end = d; }
  cues.push(c); cues.sort((a,b)=>a.start-b.start); renderList(); drawRegions(); select(c.id,false); snap();
  listEl.querySelector(`[data-id="${c.id}"] .tx`)?.focus();
}
function delCue(id){ cues = cues.filter(c=>c.id!==id); if(sel===id) sel=null; renderList(); drawRegions(); snap(); }
/* Phát riêng một cue: dừng ở cuối cue, hoặc lặp lại nếu bật chế độ lặp (kiểm tra mỗi frame để chính xác) */
function playCue(c){ if(!c) return; seg = c.id; video.currentTime = c.start; video.play(); }
function tick(){
  if(seg && !video.paused){
    const c = byId(seg), t = video.currentTime;
    if(!c || t < c.start - 0.05 || t > c.end + 0.3) seg = null;      // người dùng tua ra ngoài cue
    else if(t >= c.end - 0.02){
      if(loop) video.currentTime = c.start;
      else { video.pause(); video.currentTime = c.end; seg = null; }
    }
  }
  syncLive();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
function toggleLoop(){ loop = !loop; $('#loop').setAttribute('aria-pressed', loop); }

/* ---------- Undo / redo (lưu ảnh chụp toàn bộ cue) ---------- */
function snap(){ clearTimeout(pend); pend = null; const s = JSON.stringify(cues); if(hist[hi] === s) return;
  hist = hist.slice(0, hi+1); hist.push(s); if(hist.length > 200) hist.shift(); hi = hist.length - 1; drawRuler(); if(hi > 0) scheduleSave(); }
function snapSoon(){ clearTimeout(pend); pend = setTimeout(snap, 600); }
function restore(){ cues = JSON.parse(hist[hi]); if(!byId(sel)) sel = null; renderList(); drawRegions(); if(sel) select(sel, false); scheduleSave(); }
function undo(){ if(pend) snap(); if(hi > 0){ hi--; restore(); setStatus('Đã hoàn tác'); } }
function redo(){ if(hi < hist.length - 1){ hi++; restore(); setStatus('Đã làm lại'); } }

/* ---------- Tách / gộp / điều hướng ---------- */
function splitCue(){
  const c = byId(sel); if(!c) return setStatus('Chọn một cue để tách.');
  let t = video.currentTime; if(t <= c.start + 0.2 || t >= c.end - 0.2) t = (c.start + c.end) / 2;
  const ta = document.activeElement; let a = c.text, b = '';
  if(ta && ta.classList.contains('tx') && ta.closest('.cue').dataset.id === c.id){ const p = ta.selectionStart; a = c.text.slice(0,p).trim(); b = c.text.slice(p).trim(); }
  const n = { id:'c'+(seq++), start:t, end:c.end, src:'', text:b, mt:'', ok:false, origin:'manual', fmt:{ ...(c.fmt || {}) } };
  c.end = t; c.text = a; c.ok = false; cues.push(n); cues.sort((x,y)=>x.start-y.start);
  renderList(); drawRegions(); select(c.id, false); snap();
}
function mergeNext(){
  const i = cues.findIndex(c => c.id === sel);
  if(i < 0 || i >= cues.length-1) return setStatus('Không có cue kế tiếp để gộp.');
  const a = cues[i], b = cues[i+1];
  a.ok = false; a.end = Math.max(a.end, b.end); a.text = (a.text+' '+b.text).trim(); a.src = (a.src+' '+b.src).trim();
  cues.splice(i+1, 1); renderList(); drawRegions(); select(a.id, false); snap();
}
function step(d){
  if(!cues.length) return; let i = cues.findIndex(c => c.id === sel);
  i = i < 0 ? 0 : Math.min(cues.length-1, Math.max(0, i+d));
  select(cues[i].id, true); playCue(cues[i]);
}
function setEdge(w){
  const c = byId(sel); if(!c) return; const t = video.currentTime;
  if(w==='s' && t < c.end - 0.05) c.start = t; else if(w==='e' && t > c.start + 0.05) c.end = t; else return;
  syncTimes(c); const r = regions.getRegions().find(r => r.id === c.id); r && r.setOptions({ start:c.start, end:c.end }); snap();
}

/* ---------- Phím tắt ---------- */
const keys = {
  Enter: () => { const c = byId(sel); if(c) (seg === c.id && !video.paused) ? video.pause() : playCue(c); },
  Space: () => video.paused ? video.play() : video.pause(),
  ArrowUp: () => step(-1), ArrowDown: () => step(1),
  ArrowLeft: () => video.currentTime -= 2, ArrowRight: () => video.currentTime += 2,
  KeyA: () => { const c = byId(sel); if(!c) return; setOk(c, !c.ok); if(c.ok) step(1); },
  KeyL: toggleLoop, KeyS: splitCue, KeyM: mergeNext, KeyN: () => addCue(video.currentTime),
  Delete: () => sel && delCue(sel), Backspace: () => sel && delCue(sel),
  BracketLeft: () => setEdge('s'), BracketRight: () => setEdge('e')
};
document.addEventListener('keydown', e => {
  if($('#app').hidden) return;
  const mod = e.ctrlKey || e.metaKey;
  if(mod && e.code === 'KeyZ'){ e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
  if(mod && e.code === 'KeyY'){ e.preventDefault(); redo(); return; }
  if(e.code === 'Escape'){ document.activeElement.blur(); return; }
  if(!video.src) return;
  const typing = /INPUT|TEXTAREA|SELECT|BUTTON/.test(document.activeElement.tagName);
  if(e.altKey || (!typing && !mod && e.code === 'Space')){ const f = keys[e.code]; if(f){ e.preventDefault(); f(); } }
});
$('#addnow').onclick = () => addCue(video.currentTime);
$('#playsel').onclick = keys.Enter;
$('#undo').onclick = undo; $('#redo').onclick = redo; $('#split').onclick = splitCue; $('#merge').onclick = mergeNext; $('#loop').onclick = toggleLoop;
$('#helpbtn').onclick = () => $('#help').showModal(); $('#hclose').onclick = () => $('#help').close();

/* ---------- Phụ đề trên video + cue đang phát ---------- */
let liveId = null;
function centerRow(row){
  listEl.scrollTo({ top: Math.max(0, row.offsetTop - 6), behavior:'smooth' });
}
let ovSig = '';
function renderOverlay(act){   // cue đến trước hiển thị trên, cue chồng thời gian hiển thị ngay bên dưới
  const field = $('#viewlang').value || 'text';
  const sig = field + '§' + act.map(c => c.id + '|' + (c[field] || '') + '|' + JSON.stringify(c.fmt || 0)).join('§');
  if(sig === ovSig) return; ovSig = sig;
  const ov = $('#ov'), zones = {}; ov.textContent = '';
  for(const c of act){
    const t = (c[field] || '').trim(); if(!t) continue;
    const f = fmtOf(c), z = zones[f.pos] || (zones[f.pos] = Object.assign(document.createElement('div'), { className: 'ovz ' + f.pos }));
    const d = document.createElement('div'); d.className = 'ovc'; d.textContent = t; d.style.textAlign = f.align;
    if(f.b) d.style.fontWeight = '700'; if(f.i) d.style.fontStyle = 'italic'; if(f.color !== FMT_DEF.color) d.style.color = f.color;
    z.appendChild(d);
  }
  Object.values(zones).forEach(z => ov.appendChild(z));
}
$('#viewlang').onchange = () => { ovSig = ''; syncLive(); };
function syncLive(){
  const t = video.currentTime, act = cues.filter(c => t >= c.start && t < c.end).sort((x, y) => x.start - y.start);
  renderOverlay(act);
  const key = act.map(c => c.id).join(',');
  if(key === liveId && (!key || listEl.querySelector('.live'))) return;
  liveId = key;
  listEl.querySelectorAll('.live').forEach(r => r.classList.remove('live'));
  let first = null;
  for(const c of act){ const row = listEl.querySelector(`[data-id="${c.id}"]`); if(row){ row.classList.add('live'); first = first || row; } }
  if(first){ select(first.dataset.id, false, true); centerRow(first); }   // danh sách cuộn theo cue đến trước
}
['timeupdate','seeked','play'].forEach(ev => video.addEventListener(ev, syncLive));

/* ---------- Ngôn ngữ: 'src' = văn bản gốc (ASR), 'text' = bản dịch (MT) ---------- */
const langName = code => (typeof LANG !== 'undefined' && LANG[code]) || code;
const codeOf = field => field === 'src' ? proj.src_lang : proj.tgt_lang;
function langSelect(sel){
  sel.innerHTML = `<option value="src">${esc(langName(proj.src_lang))} — văn bản gốc (ASR)</option><option value="text">${esc(langName(proj.tgt_lang))} — bản dịch (MT)</option>`;
}

/* ---------- Xuất phụ đề: chọn ngôn ngữ + định dạng, hỏi giữ/bỏ vị trí và kiểu chữ ---------- */
let exFmt = 'srt', lastAuto = '';
const baseName = () => proj ? proj.title : 'phu-de';
const hex6 = c => c.replace('#', '').toLowerCase();
function vttSettings(c){
  const f = fmtOf(c); let s = '';
  if(f.pos === 'top') s += ' line:10%'; else if(f.pos === 'mid') s += ' line:50%';
  if(f.align !== 'center') s += ' align:' + f.align;
  return s;
}
function cueText(c, field, keep){
  let t = c[field] || ''; if(!keep) return t;
  const f = fmtOf(c);
  if(f.b) t = `<b>${t}</b>`; if(f.i) t = `<i>${t}</i>`;
  if(f.color !== FMT_DEF.color) t = `<c.color-${hex6(f.color)}>${t}</c>`;
  return t;
}
const withText = field => cues.filter(c => (c[field] || '').trim());
function build(kind, field, keep){
  const arr = withText(field).sort((x, y) => x.start - y.start);
  if(kind === 'vtt'){
    const colors = [...new Set(arr.map(c => fmtOf(c).color).filter(x => x !== FMT_DEF.color))];
    let out = 'WEBVTT\n\n';
    if(keep && colors.length) out += 'STYLE\n' + colors.map(x => `::cue(.color-${hex6(x)}) { color: ${x}; }`).join('\n') + '\n\n';
    return out + arr.map(c => `${ts(c.start, '.')} --> ${ts(c.end, '.')}${keep ? vttSettings(c) : ''}\n${cueText(c, field, keep)}`).join('\n\n') + '\n';
  }
  return arr.map((c, i) => `${i + 1}\n${ts(c.start, ',')} --> ${ts(c.end, ',')}\n${c[field]}`).join('\n\n') + '\n';   // SRT: văn bản thuần
}
function showExport(){
  const field = $('#ex-lang').value, auto = `${baseName()}.${codeOf(field)}.${exFmt}`, cur = $('#fn').value.trim();
  $('#fn').value = (!cur || cur === lastAuto) ? auto : cur.replace(/\.(srt|vtt)$/i, '') + '.' + exFmt;   // giữ tên người dùng đã gõ, chỉ đổi đuôi
  lastAuto = auto;
  document.querySelectorAll('[data-f]').forEach(b => b.setAttribute('aria-pressed', b.dataset.f === exFmt));
  $('#out').value = build(exFmt, field, exFmt === 'vtt');
}
function askFmt(kind, list){   // Promise: true = giữ, false = bỏ, null = hủy
  const n = list.filter(hasFmt).length;
  if(!n) return Promise.resolve(false);   // không cue nào có định dạng riêng thì không cần hỏi
  return new Promise(res => {
    const d = $('#fmtq');
    $('#fmtq-msg').textContent = kind === 'srt'
      ? `Có ${n} cue được đặt vị trí / kiểu chữ riêng. File SRT không lưu các giá trị này nên chúng sẽ bị bỏ.`
      : `Có ${n} cue được đặt vị trí / kiểu chữ riêng. File VTT lưu được các giá trị này. Bạn muốn giữ hay bỏ?`;
    $('#fmtq-keep').hidden = kind === 'srt'; $('#fmtq-drop').textContent = kind === 'srt' ? 'Xuất SRT thuần (bỏ định dạng)' : 'Bỏ định dạng';
    const done = v => { d.close(); res(v); };
    $('#fmtq-keep').onclick = () => done(true); $('#fmtq-drop').onclick = () => done(false); $('#fmtq-cancel').onclick = () => done(null);
    d.oncancel = () => res(null);
    d.showModal();
  });
}
async function exportNow(action){
  const field = $('#ex-lang').value, list = withText(field);
  if(!list.length) return alert(`Chưa có văn bản ${langName(codeOf(field))} để xuất.`);
  const keep = await askFmt(exFmt, list); if(keep === null) return;
  const text = build(exFmt, field, keep); $('#out').value = text;
  if(action === 'copy'){
    try{ await navigator.clipboard.writeText(text); }catch(e){ $('#out').select(); document.execCommand('copy'); }
    $('#copy').textContent = 'Đã sao chép'; setTimeout(() => $('#copy').textContent = 'Sao chép', 1500); return;
  }
  const name = ($('#fn').value.trim().replace(/[\\/:*?"<>|]+/g, '_').replace(/\.(srt|vtt)$/i, '') || baseName()) + '.' + exFmt;
  const url = URL.createObjectURL(new Blob([text], { type:'text/plain;charset=utf-8' }));
  const link = document.createElement('a'); link.href = url; link.download = name;
  document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('#exp').onclick = () => { langSelect($('#ex-lang')); $('#ex-lang').value = 'text'; showExport(); $('#dlg').showModal(); };
$('#ex-lang').onchange = showExport;
document.querySelectorAll('[data-f]').forEach(b => b.onclick = () => { exFmt = b.dataset.f; showExport(); });
$('#dl').onclick = () => exportNow('dl'); $('#copy').onclick = () => exportNow('copy'); $('#close').onclick = () => $('#dlg').close();

/* ---------- Nhập SRT/VTT: chọn ngôn ngữ của file (gốc → cập nhật ASR, đích → cập nhật MT) ---------- */
const TM = /(?:(\d+):)?(\d+):(\d+)[,.](\d+)/;
const toSec = m => (+(m[1] || 0)) * 3600 + (+m[2]) * 60 + (+m[3]) + parseFloat('0.' + m[4]);
function parseSub(txt){
  return txt.replace(/^\uFEFF/, '').replace(/\r/g, '').split(/\n\s*\n/).map(block => {
    const L = block.split('\n'), i = L.findIndex(l => l.includes('-->')); if(i < 0) return null;
    const [s, e] = L[i].split('-->').map(x => x.trim().match(TM)); if(!s || !e) return null;
    return { start: toSec(s), end: toSec(e), text: L.slice(i + 1).join('\n').trim() };
  }).filter(Boolean);
}
function applyImport(list, field, mode){
  const mk = o => ({ id:'c' + (seq++), start:o.start, end:o.end, src: field === 'src' ? o.text : '', text: field === 'text' ? o.text : '', mt:'', ok:false, origin:'import' });
  if(mode === 'replace' || !cues.length){ cues = list.map(mk); return { matched: 0, added: list.length }; }
  const used = new Set(); let matched = 0, added = 0;
  for(const o of list){   // ghép theo mức chồng thời gian lớn nhất, mỗi cue hiện có chỉ ghép một lần
    let best = null, bestOv = 0;
    for(const c of cues){ if(used.has(c.id)) continue; const ov = Math.min(c.end, o.end) - Math.max(c.start, o.start); if(ov > bestOv){ bestOv = ov; best = c; } }
    if(best && bestOv >= 0.3 * Math.min(o.end - o.start, best.end - best.start)){ used.add(best.id); best[field] = o.text; best.ok = false; matched++; }
    else { const n = mk(o); cues.push(n); used.add(n.id); added++; }
  }
  cues.sort((x, y) => x.start - y.start);
  return { matched, added };
}
$('#imp').onclick = () => {
  langSelect($('#imp-lang')); $('#imp-lang').value = cues.some(c => c.src) ? 'text' : 'src';
  $('#imp-modef').hidden = !cues.length; $('#imp-file').value = ''; $('#imp-msg').textContent = ''; $('#impd').showModal();
};
$('#imp-cancel').onclick = () => $('#impd').close();
$('#imp-go').onclick = async () => {
  const f = $('#imp-file').files[0], msg = t => $('#imp-msg').textContent = t;
  if(!f) return msg('Hãy chọn một file SRT hoặc VTT.');
  const field = $('#imp-lang').value, list = parseSub(await f.text());
  if(!list.length) return msg('Không đọc được cue nào trong file này.');
  const dur = video.duration; if(!isFinite(dur)) return msg('Video chưa tải xong, hãy thử lại sau giây lát.');
  const bad = list.map((o, i) => ({ o, i })).filter(x => x.o.end > dur + 0.5);   // dung sai 0,5 giây
  if(bad.length) return msg(`Không thể nhập: file có ${bad.length} cue vượt quá độ dài video (${fmt(dur)}). Cue đầu tiên lỗi: #${bad[0].i + 1}, kết thúc lúc ${fmt(bad[0].o.end)}. Hãy kiểm tra xem đây có đúng file của video này không.`);
  const mode = cues.length ? document.querySelector('input[name=impmode]:checked').value : 'replace';
  if(mode === 'replace' && cues.length && !confirm(`Thay ${cues.length} cue hiện có bằng ${list.length} cue từ “${f.name}”? (có thể hoàn tác)`)) return;
  const r = applyImport(list, field, mode);
  renderList(); drawRegions(); dirtyReason = 'import'; snap(); $('#impd').close();
  setStatus(`Đã nhập ${list.length} cue ${langName(codeOf(field))} từ ${f.name}` + (mode === 'merge' ? ` (ghép ${r.matched}, thêm mới ${r.added})` : '') + '.');
};

/* ---------- Tự động lưu lên server (debounce 1,5s) ---------- */
let saveT = null, saving = false, saveFailed = false, dirtyReason = 'edit';
const setSaved = t => $('#saved').textContent = t;
function scheduleSave(){ if(!proj) return; setSaved('Chưa lưu…'); clearTimeout(saveT); saveT = setTimeout(saveNow, 1500); }
async function saveNow(){
  clearTimeout(saveT); saveT = null;
  if(!proj) return;
  if(saving){ saveT = setTimeout(saveNow, 400); return; }
  saving = true; const body = { cues, rev, reason: dirtyReason }; dirtyReason = 'edit';
  try{ rev = (await api(`/api/projects/${proj.id}/cues`, { method:'PUT', json: body })).rev; saveFailed = false; if(!saveT) setSaved('Đã lưu'); }
  catch(err){
    saveFailed = true;
    if(err.status === 409) setSaved('Xung đột — hãy tải lại trang');
    else { setSaved('Lưu lỗi, đang thử lại…'); saveT = setTimeout(saveNow, 5000); }
  }
  saving = false;
}
async function flush(){   // chờ lưu xong trước khi chuyển màn hình / chạy model; trả về false nếu người dùng ở lại
  for(let i = 0; (saveT || saving) && i < 80; i++){
    if(saveT && !saving){ clearTimeout(saveT); saveT = null; await saveNow(); if(saveFailed) break; }
    else await sleep(100);
  }
  if(!saveFailed && !saveT && !saving) return true;
  clearTimeout(saveT); saveT = null;
  if(confirm('Chưa lưu được thay đổi lên server. Vẫn tiếp tục và bỏ các thay đổi chưa lưu?')){ saveFailed = false; return true; }
  scheduleSave(); return false;
}
window.addEventListener('beforeunload', e => { if(saveT || saving){ e.preventDefault(); e.returnValue = ''; } });

/* ---------- Nghe lại (chỉ ASR) / Dịch lại (chỉ MT) cho từng cue ---------- */
function refreshRow(c){
  const row = listEl.querySelector(`[data-id="${c.id}"]`); if(!row) return;
  row.querySelector('.src').value = c.src; row.querySelector('.tx').value = c.text; paintRow(row, c); updateProg();
  const r = regions && regions.getRegions().find(r => r.id === c.id); r && r.setOptions({ content: lodTier === 2 ? c.text.slice(0, 40) : '', color: regColor(c) });
}
async function redoCue(btn, label, job){
  if(!proj) return;
  const old = btn.textContent; btn.disabled = true; btn.textContent = label;
  try{ await job(); }catch(err){ setStatus('Lỗi: ' + err.message); }
  btn.disabled = false; btn.textContent = old;
}
const redoAsr = (c, btn) => redoCue(btn, 'Đang nghe…', async () => {
  const r = await api(`/api/projects/${proj.id}/redo-asr`, { method:'POST', json:{ start:c.start, end:c.end } });
  c.src = r.text; c.ok = false; refreshRow(c); snap(); setStatus(`Đã nghe lại cue #${cues.indexOf(c) + 1}.`);
});
const redoMt = (c, btn) => redoCue(btn, 'Đang dịch…', async () => {
  if(!c.src.trim()) return setStatus('Cue chưa có văn bản gốc để dịch. Hãy bấm “Nghe lại” hoặc nhập văn bản gốc.');
  const r = await api(`/api/projects/${proj.id}/redo-mt`, { method:'POST', json:{ text:c.src } });
  c.text = r.text; c.ok = false; refreshRow(c); snap(); setStatus(`Đã dịch lại cue #${cues.indexOf(c) + 1}.`);
});

/* ---------- Thước thời gian + thanh cuộn ngang, đều theo độ thu phóng của waveform chính ---------- */
const scroller = () => ws && ws.getWrapper().parentElement;
const NICE = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
function rulerLabel(t, step){
  const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), s = t % 60;
  const ss = step < 1 ? s.toFixed(1).padStart(4, '0') : String(Math.floor(s)).padStart(2, '0');
  return (h ? h + ':' + String(m).padStart(2, '0') : m) + ':' + ss;
}
function drawRuler(){
  const cv = $('#ruler'), W = cv.clientWidth, H = cv.clientHeight; if(!W) return;
  const dpr = window.devicePixelRatio || 1; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const g = cv.getContext('2d'); if(!g) return;
  g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const sc = scroller(), dur = video.duration; if(!sc || !isFinite(dur) || !dur) return;
  const pps = sc.scrollWidth / dur, t0 = sc.scrollLeft / pps, t1 = t0 + W / pps;   // pixel/giây thực tế và khoảng thời gian đang hiển thị
  const step = NICE.find(n => n * pps >= 90) || 7200, minor = step / 5;           // nhãn cách nhau ≥ 90px
  for(const c of cues){   // bản đồ cue: vàng = chưa duyệt, xanh = đã duyệt (thấy được cả khi thu nhỏ toàn bộ video)
    const x0 = (c.start - t0) * pps, x1 = (c.end - t0) * pps; if(x1 < 0 || x0 > W) continue;
    g.fillStyle = c.ok ? '#2e9e5b' : '#f0aa14'; g.fillRect(x0, 0, Math.max(1, x1 - x0), 4);
  }
  g.strokeStyle = g.fillStyle = css('--muted'); g.font = '11px "Be Vietnam Pro", system-ui, sans-serif'; g.textBaseline = 'top';
  for(let k = Math.max(0, Math.floor(t0 / minor)); k * minor <= t1 + minor; k++){
    const t = k * minor, x = Math.round((t - t0) * pps) + .5, major = k % 5 === 0;
    if(!major && minor * pps < 6) continue;
    g.beginPath(); g.moveTo(x, H); g.lineTo(x, major ? H - 12 : H - 6); g.stroke();
    if(major) g.fillText(rulerLabel(Math.round(t * 1000) / 1000, step), x + 3, 6);
  }
}
function syncBar(){   // độ dài thanh trượt = phần waveform đang nhìn thấy / toàn bộ; thu phóng càng lớn thanh càng ngắn
  const sc = scroller(), bar = $('#hbar'), th = $('#hthumb'); if(!sc) return;
  const max = sc.scrollWidth - sc.clientWidth, W = bar.clientWidth;
  bar.classList.toggle('off', max <= 1);
  if(max <= 1){ th.style.width = '100%'; th.style.left = '0px'; return; }
  const w = Math.max(24, W * sc.clientWidth / sc.scrollWidth);
  th.style.width = w + 'px'; th.style.left = ((W - w) * sc.scrollLeft / max) + 'px';
}
function syncView(){ syncBar(); if(regions && tierOf(ppsNow()) !== lodTier) drawRegions(); drawRuler(); }
video.addEventListener('loadedmetadata', syncView);
window.addEventListener('resize', syncView);
if(window.ResizeObserver) new ResizeObserver(syncView).observe($('#wavebox'));

function rulerSeek(e){   // bấm hoặc kéo trên thước để chuyển con trỏ thời gian
  const sc = scroller(); if(!sc || !isFinite(video.duration)) return;
  const pps = sc.scrollWidth / video.duration, x = e.clientX - $('#ruler').getBoundingClientRect().left;
  video.currentTime = Math.min(video.duration, Math.max(0, (sc.scrollLeft + x) / pps));
}
$('#ruler').onpointerdown = e => { e.target.setPointerCapture(e.pointerId); rulerSeek(e); e.target.onpointermove = rulerSeek; };
$('#ruler').onpointerup = $('#ruler').onpointercancel = e => { e.target.onpointermove = null; };

(() => {   // kéo thanh trượt để cuộn waveform
  const bar = $('#hbar'), th = $('#hthumb'); let drag = null;
  const moveTo = left => {
    const sc = scroller(); if(!sc) return;
    const W = bar.clientWidth, w = th.offsetWidth, max = sc.scrollWidth - sc.clientWidth; if(max <= 0 || W <= w) return;
    sc.scrollLeft = Math.min(1, Math.max(0, left / (W - w))) * max;
  };
  th.onpointerdown = e => { e.stopPropagation(); th.setPointerCapture(e.pointerId); drag = { x:e.clientX, left:th.offsetLeft }; };
  th.onpointermove = e => { if(drag) moveTo(drag.left + e.clientX - drag.x); };
  th.onpointerup = th.onpointercancel = () => { drag = null; };
  bar.onpointerdown = e => { if(e.target === bar) moveTo(e.clientX - bar.getBoundingClientRect().left - th.offsetWidth / 2); };
})();

/* ---------- Waveform từ peaks tính sẵn ở server ---------- */
const metaReady = () => new Promise(res => {
  if(video.readyState >= 1 && isFinite(video.duration)) return res();
  const t = setTimeout(res, 8000);
  video.addEventListener('loadedmetadata', () => { clearTimeout(t); res(); }, { once:true });
});
async function loadPeaks(p){
  setStatus('Đang tải waveform…');
  try{
    await metaReady(); if(!isFinite(video.duration)) throw new Error('chưa đọc được độ dài video');
    const r = await fetch(`/api/projects/${p.id}/peaks`, { credentials:'same-origin' });
    if(!r.ok) throw new Error('HTTP ' + r.status);
    const u8 = new Uint8Array(await r.arrayBuffer()); if(!u8.length) throw new Error('dữ liệu rỗng');
    const f = new Float32Array(u8.length * 2);   // mỗi mẫu → (+v, −v) để vẽ waveform đối xứng
    for(let i = 0; i < u8.length; i++){ const v = u8[i] / 255; f[2 * i] = v; f[2 * i + 1] = -v; }
    setStatus(''); return { data: f, duration: video.duration };
  }catch(err){ setStatus('Không lấy được waveform từ server (' + err.message + '), thử giải mã trên trình duyệt…'); return null; }
}