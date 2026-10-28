const $ = s => document.querySelector(s);
const video = $('#video'), listEl = $('#list');
let cues = [], sel = null, seq = 1, ws = null, regions = null, file = null;
let loop = false, seg = null, hist = [], hi = -1, pend = null;
// Client gọi API cùng origin (backend phục vụ cả trang web và /api). Địa chỉ model GPU cấu hình trong backend.
// DEMO_FALLBACK: khi chưa có backend thì dùng dữ liệu mẫu. Đặt false khi chạy thật để thấy lỗi kết nối.
const API_BASE = '', DEMO_FALLBACK = true;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const byId = id => cues.find(c => c.id === id);
const setStatus = t => $('#status').textContent = t;

function fmt(t){ t=Math.max(0,t); const m=Math.floor(t/60), s=t-m*60; return String(m).padStart(2,'0')+':'+s.toFixed(3).padStart(6,'0'); }
function parse(v){ const p=String(v).trim().split(':').map(Number); if(p.some(isNaN)) return NaN; return p.reduce((a,x)=>a*60+x,0); }
function ts(t,sep){ const ms=Math.round(t*1000), h=Math.floor(ms/3600000), m=Math.floor(ms/60000)%60, s=Math.floor(ms/1000)%60, r=ms%1000;
  const z=(n,l=2)=>String(n).padStart(l,'0'); return `${z(h)}:${z(m)}:${z(s)}${sep}${z(r,3)}`; }

/* ---------- Tải video + waveform ---------- */
$('#pick').onclick = () => $('#file').click();
$('#file').onchange = e => {
  file = e.target.files[0]; if(!file) return;
  if(ws){ ws.destroy(); }
  cues = []; sel = null; seg = null; hist = []; hi = -1; renderList(); snap();
  video.src = URL.createObjectURL(file);
  regions = WaveSurfer.Regions.create();
  ws = WaveSurfer.create({ container:'#wave', media:video, height:88, normalize:true, hideScrollbar:true, minPxPerSec:+$('#zoom').value,
    waveColor:css('--wave'), progressColor:css('--prog'), cursorColor:css('--accent'), plugins:[regions] });
  ws.on('decode', drawRegions);
  regions.on('region-updated', r => { const c=byId(r.id); if(!c) return; c.start=r.start; c.end=r.end; syncTimes(c); snap(); });
  regions.on('region-clicked', (r,ev) => { ev.stopPropagation(); select(r.id,true); playCue(byId(r.id)); });
  ['run','exp','addnow','playsel','split','merge'].forEach(i => $('#'+i).disabled = false);
  $('#fname').textContent = file.name; $('#fn').value = ''; setStatus('');
};
$('#zoom').oninput = e => ws && ws.zoom(+e.target.value);

function drawRegions(){
  if(!regions) return; regions.clearRegions();
  cues.forEach(c => regions.addRegion({ id:c.id, start:c.start, end:c.end, content:c.text.slice(0,40), color:c.id===sel?css('--regsel'):css('--reg'), drag:true, resize:true }));
}

/* ---------- Dịch sơ (ASR + MT) ---------- */
$('#run').onclick = async () => {
  if(cues.length && !confirm('Bản dịch sơ sẽ thay thế toàn bộ cue hiện có. Tiếp tục?')) return;
  const btn=$('#run'); btn.disabled=true; setStatus('Đang nhận dạng và dịch…');
  try{
    let out;
    out = await runJob().catch(err => {
      if(err.noBackend && DEMO_FALLBACK){ setStatus('Chưa có backend, đang dùng dữ liệu mẫu.'); return null; }
      throw err;
    });
    if(!out){
      await new Promise(r => setTimeout(r, 800));
      const d = video.duration || 20; out = [];
      for(let t=0, i=1; t<d; t+=4, i++) out.push({ start:t, end:Math.min(t+3.6,d), src:`(ASR) câu nguồn số ${i}`, text:`(MT) bản dịch sơ số ${i}` });
    }
    cues = out.map(o => ({ id:'c'+(seq++), start:+o.start, end:+o.end, src:o.src||'', text:o.text||'' }));
    renderList(); drawRegions(); snap(); setStatus(`Đã tạo ${cues.length} cue.`);
  }catch(err){ setStatus('Lỗi: ' + err.message); }
  btn.disabled=false;
};

// Giao thức với server GPU: POST /api/jobs (multipart) -> {job_id}; GET /api/jobs/{id} -> {status, stage, progress, cues, error}
async function runJob(){
  const base = API_BASE;
  const fd = new FormData(); fd.append('video', file); fd.append('src_lang', $('#sl').value); fd.append('tgt_lang', $('#tl').value);
  setStatus('Đang tải video lên server…');
  let r;
  try{ r = await fetch(base + '/api/jobs', { method:'POST', body:fd }); }
  catch(e){ throw Object.assign(new Error('Không kết nối được server'), { noBackend:true }); }
  if(r.status === 404) throw Object.assign(new Error('Không tìm thấy API /api/jobs'), { noBackend:true });
  if(!r.ok) throw new Error('Server trả lỗi ' + r.status);
  const { job_id } = await r.json();
  for(;;){
    await new Promise(res => setTimeout(res, 1500));
    const s = await (await fetch(`${base}/api/jobs/${job_id}`)).json();
    if(s.status === 'failed') throw new Error(s.error || 'Job thất bại');
    if(s.status === 'done') return s.cues;   // [{start, end, src, text}]
    setStatus(`${s.stage || 'Đang xử lý'}… ${Math.round((s.progress||0)*100)}%`);
  }
}

/* ---------- Danh sách cue ---------- */
function renderList(){
  listEl.hidden = !cues.length; $('#empty').hidden = !!cues.length;
  listEl.innerHTML = '';
  cues.forEach((c,i) => {
    const el = document.createElement('div');
    el.className = 'cue' + (c.id===sel?' sel':''); el.dataset.id = c.id;
    el.innerHTML = `<div class="times"><input class="t s" aria-label="Bắt đầu"><input class="t e" aria-label="Kết thúc"></div>
      <div class="txt"><textarea class="src" rows="1" placeholder="Văn bản nguồn (ASR)"></textarea><textarea class="tx" rows="2" placeholder="Bản dịch"></textarea></div>
      <div class="act"><button data-a="play">Phát</button><button data-a="add">Thêm sau</button><button data-a="del" class="del">Xóa</button></div>`;
    el.querySelector('.s').value = fmt(c.start); el.querySelector('.e').value = fmt(c.end);
    el.querySelector('.src').value = c.src; el.querySelector('.tx').value = c.text;
    listEl.appendChild(el);
  });
}
listEl.addEventListener('click', e => {
  const row = e.target.closest('.cue'); if(!row) return; const c = byId(row.dataset.id);
  const a = e.target.dataset.a;
  if(a==='play') playCue(c);
  else if(a==='add') addCue(c.end, c.id);
  else if(a==='del') delCue(c.id);
  if(!a && !e.target.classList.contains('t')){ const first = sel !== c.id; select(c.id, false); if(first) playCue(c); }
});
listEl.addEventListener('input', e => {
  const c = byId(e.target.closest('.cue').dataset.id);
  if(e.target.classList.contains('tx')){ c.text = e.target.value; const r = regions.getRegions().find(r=>r.id===c.id); r && r.setOptions({content:c.text.slice(0,40)}); }
  if(e.target.classList.contains('src')) c.src = e.target.value;
  snapSoon();
});
listEl.addEventListener('change', e => {
  const row = e.target.closest('.cue'), c = byId(row.dataset.id);
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
  regions && regions.getRegions().forEach(r => r.setOptions({ color: r.id===id ? css('--regsel') : css('--reg') }));
  const c = byId(id); if(!c) return;
  if(!follow && ws && ws.setScrollTime) ws.setScrollTime(Math.max(0, c.start - 1));
  if(seek){ video.currentTime = c.start; showRow(listEl.querySelector(`[data-id="${id}"]`)); }
}
function addCue(start, afterId){
  const d = video.duration || start+2, c = { id:'c'+(seq++), start:Math.min(start,d), end:Math.min(start+2,d), src:'', text:'' };
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
  hist = hist.slice(0, hi+1); hist.push(s); if(hist.length > 200) hist.shift(); hi = hist.length - 1; }
function snapSoon(){ clearTimeout(pend); pend = setTimeout(snap, 600); }
function restore(){ cues = JSON.parse(hist[hi]); if(!byId(sel)) sel = null; renderList(); drawRegions(); if(sel) select(sel, false); }
function undo(){ if(pend) snap(); if(hi > 0){ hi--; restore(); setStatus('Đã hoàn tác'); } }
function redo(){ if(hi < hist.length - 1){ hi++; restore(); setStatus('Đã làm lại'); } }

/* ---------- Tách / gộp / điều hướng ---------- */
function splitCue(){
  const c = byId(sel); if(!c) return setStatus('Chọn một cue để tách.');
  let t = video.currentTime; if(t <= c.start + 0.2 || t >= c.end - 0.2) t = (c.start + c.end) / 2;
  const ta = document.activeElement; let a = c.text, b = '';
  if(ta && ta.classList.contains('tx') && ta.closest('.cue').dataset.id === c.id){ const p = ta.selectionStart; a = c.text.slice(0,p).trim(); b = c.text.slice(p).trim(); }
  const n = { id:'c'+(seq++), start:t, end:c.end, src:'', text:b };
  c.end = t; c.text = a; cues.push(n); cues.sort((x,y)=>x.start-y.start);
  renderList(); drawRegions(); select(c.id, false); snap();
}
function mergeNext(){
  const i = cues.findIndex(c => c.id === sel);
  if(i < 0 || i >= cues.length-1) return setStatus('Không có cue kế tiếp để gộp.');
  const a = cues[i], b = cues[i+1];
  a.end = Math.max(a.end, b.end); a.text = (a.text+' '+b.text).trim(); a.src = (a.src+' '+b.src).trim();
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
  KeyL: toggleLoop, KeyS: splitCue, KeyM: mergeNext, KeyN: () => addCue(video.currentTime),
  Delete: () => sel && delCue(sel), Backspace: () => sel && delCue(sel),
  BracketLeft: () => setEdge('s'), BracketRight: () => setEdge('e')
};
document.addEventListener('keydown', e => {
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
function syncLive(){
  const t = video.currentTime, c = cues.find(c => t >= c.start && t < c.end), id = c ? c.id : null;
  $('#ov').textContent = c ? c.text : '';
  if(id === liveId && (!id || listEl.querySelector('.live'))) return;
  liveId = id;
  listEl.querySelector('.live')?.classList.remove('live');
  const row = id && listEl.querySelector(`[data-id="${id}"]`);
  if(row){ row.classList.add('live'); select(id, false, true); centerRow(row); }
}
['timeupdate','seeked','play'].forEach(ev => video.addEventListener(ev, syncLive));

/* ---------- Xuất ---------- */
function build(f){
  const sorted = [...cues].sort((a,b)=>a.start-b.start);
  if(f==='vtt') return 'WEBVTT\n\n' + sorted.map(c => `${ts(c.start,'.')} --> ${ts(c.end,'.')}\n${c.text}`).join('\n\n') + '\n';
  return sorted.map((c,i) => `${i+1}\n${ts(c.start,',')} --> ${ts(c.end,',')}\n${c.text}`).join('\n\n') + '\n';
}
let fmtSel = 'srt';
const baseName = () => file ? file.name.replace(/\.[^.]+$/, '') : 'phu-de';
function showExport(f){
  fmtSel = f; $('#out').value = build(f);
  const cur = $('#fn').value.trim().replace(/\.(srt|vtt)$/i, '');   // giữ tên người dùng đã gõ, chỉ đổi đuôi
  $('#fn').value = (cur || baseName()) + '.' + f;
}
$('#exp').onclick = () => { showExport(fmtSel); $('#dlg').showModal(); };
document.querySelectorAll('[data-f]').forEach(b => b.onclick = () => showExport(b.dataset.f));
$('#dl').onclick = () => {
  const name = ($('#fn').value.trim().replace(/[\\/:*?"<>|]+/g, '_').replace(/\.(srt|vtt)$/i, '') || baseName()) + '.' + fmtSel;
  const url = URL.createObjectURL(new Blob([$('#out').value], { type:'text/plain;charset=utf-8' }));
  const link = document.createElement('a'); link.href = url; link.download = name;
  document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('#close').onclick = () => $('#dlg').close();
$('#copy').onclick = async () => {
  const o = $('#out'); o.select();
  try{ await navigator.clipboard.writeText(o.value); }catch(e){ document.execCommand('copy'); }
  $('#copy').textContent = 'Đã sao chép'; setTimeout(()=>$('#copy').textContent='Sao chép',1500);
};
