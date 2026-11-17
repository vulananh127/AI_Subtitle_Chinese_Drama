/* shell.js — trang giới thiệu, đăng nhập, thanh tab dọc, danh sách dự án (chạy sau app.js) */
const sleep = ms => new Promise(r => setTimeout(r, ms));
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
const localTime = s => new Date(s.replace(' ', 'T') + 'Z').toLocaleString('vi-VN');
const LANG = { zh:'Tiếng Trung (giản thể)', vi:'Tiếng Việt' };
const REASON = { edit:'Chỉnh sửa', import:'Nhập SRT', model:'Tạo phụ đề tự động (AI)', restore:'Khôi phục' };
let me = null, cfg = {};

/* where: landing | projects | editor */
function show(where){
  $('#landing').hidden = where !== 'landing'; $('#shell').hidden = where === 'landing';
  $('#v-projects').hidden = where !== 'projects'; $('#app').hidden = where !== 'editor';
  $('#accmenu').hidden = true;
}
const showLanding = () => show('landing');

async function api(url, o = {}){
  const init = { method: o.method || 'GET', headers: {}, credentials: 'same-origin' };
  if(o.json){ init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(o.json); }
  const r = await fetch(url, init);
  let data = null; try{ data = await r.json(); }catch(_){}
  if(!r.ok){
    if(r.status === 401 && !o.noAuth && me){ me = null; proj = null; showLanding(); }
    const e = new Error((data && data.error) || ('Lỗi ' + r.status)); e.status = r.status; throw e;
  }
  return data;
}

/* ---------- Đăng nhập / đăng ký (cửa sổ nhỏ) ---------- */
function setTab(t){
  for(const k of ['login', 'reg']){ $('#tab-' + k).setAttribute('aria-selected', k === t); $('#pane-' + k).hidden = k !== t; }
  $('#aerr').textContent = '';
}
function openAuth(tab){ setTab(tab || 'login'); $('#authm').showModal(); initGoogle(0); }
function initGoogle(n){
  if(!cfg.google_client_id){ $('#gor').textContent = 'Đăng nhập bằng tài khoản'; return; }
  if(!window.google || !google.accounts){ if(n < 30) setTimeout(() => initGoogle(n + 1), 300); return; }
  google.accounts.id.initialize({ client_id: cfg.google_client_id, callback: r => doAuth('/api/auth/google', { credential: r.credential }) });
  google.accounts.id.renderButton($('#gbtn'), { theme:'outline', size:'large', width:280, text:'signin_with' });
}
async function doAuth(url, body){
  try{ me = await api(url, { method:'POST', json: body, noAuth: true }); $('#authm').close(); await showProjects(); }
  catch(e){ $('#aerr').textContent = e.message; }
}
$('#lp-login').onclick = $('#lp-cta').onclick = () => openAuth('login');
$('#tab-login').onclick = () => setTab('login'); $('#tab-reg').onclick = () => setTab('reg');
$('#authx').onclick = () => $('#authm').close();
$('#li-go').onclick = () => doAuth('/api/auth/login', { email: $('#li-em').value, password: $('#li-pw').value });
$('#li-pw').onkeydown = e => { if(e.key === 'Enter') $('#li-go').click(); };
$('#rg-go').onclick = () => doAuth('/api/auth/register', { email: $('#rg-em').value, password: $('#rg-pw').value, name: $('#rg-nm').value });

/* ---------- Thanh tab dọc ---------- */
$('#nav-new').onclick = () => { $('#np-msg').textContent = ''; $('#newm').showModal(); };
$('#nav-proj').onclick = () => showProjects();
$('#nav-acc').onclick = e => { e.stopPropagation(); $('#accmenu').hidden = !$('#accmenu').hidden; };
document.addEventListener('click', () => { $('#accmenu').hidden = true; });
$('#acc-info').onclick = async () => {
  const u = await api('/api/me');
  $('#infot').innerHTML = [['Tên hiển thị', u.name], ['Email', u.email], ['Đăng nhập bằng', (u.methods || []).join(', ')], ['Tham gia từ', localTime(u.created_at)]]
    .map(([k, v]) => `<tr><td class="tag">${k}</td><td>${esc(v)}</td></tr>`).join('');
  $('#infom').showModal();
};
$('#info-close').onclick = () => $('#infom').close();
$('#acc-out').onclick = async () => {
  if(!(await flush())) return;
  await api('/api/auth/logout', { method:'POST', noAuth:true }).catch(() => {});
  me = null; proj = null; video.pause(); showLanding();
};

/* ---------- Danh sách dự án ---------- */
async function showProjects(){
  if(!(await flush())) return;
  video.pause(); proj = null; show('projects');
  $('#nav-proj').classList.add('active');
  const list = await api('/api/projects');
  $('#plist').innerHTML = list.length ? list.map(p => `<div class="prow" data-id="${p.id}">
    <div class="name" title="${esc(p.title)}">${esc(p.title)}</div>
    <div>${esc(LANG[p.src_lang] || p.src_lang)} → ${esc(LANG[p.tgt_lang] || p.tgt_lang)}</div>
    <div>${p.n} cue</div><div class="tag">${esc(localTime(p.updated_at))}</div>
    <div class="bt"><button data-a="open" class="primary">Mở</button><button data-a="del" class="del">Xóa</button></div></div>`).join('')
    : '<p class="tag">Chưa có dự án nào. Bấm “Tạo” ở thanh bên trái để bắt đầu.</p>';
}
async function openById(id){ const p = await api('/api/projects/' + id); show('editor'); openProject(p); }
$('#plist').onclick = async e => {
  const row = e.target.closest('.prow'), a = e.target.dataset.a; if(!row || !a) return;
  if(a === 'open') openById(row.dataset.id);
  if(a === 'del' && confirm('Xóa dự án này cùng video và toàn bộ lịch sử? Không thể khôi phục.')){
    await api('/api/projects/' + row.dataset.id, { method:'DELETE' }); showProjects();
  }
};

/* ---------- Tạo dự án ---------- */
$('#np-cancel').onclick = () => $('#newm').close();
$('#np-go').onclick = () => {
  const f = $('#np-file').files[0]; if(!f) return $('#np-msg').textContent = 'Hãy chọn một file video.';
  const fd = new FormData(); fd.append('video', f); fd.append('title', $('#np-title').value.trim());
  fd.append('src_lang', $('#np-sl').value); fd.append('tgt_lang', $('#np-tl').value);
  const x = new XMLHttpRequest(); x.open('POST', '/api/projects'); $('#np-go').disabled = true;
  x.upload.onprogress = e => { if(e.lengthComputable) $('#np-msg').textContent = e.loaded === e.total ? 'Đang xử lý trên server…' : `Đang tải lên… ${Math.round(e.loaded / e.total * 100)}%`; };
  x.onload = () => {
    $('#np-go').disabled = false; let d = {}; try{ d = JSON.parse(x.responseText); }catch(_){}
    if(x.status === 201){ $('#np-file').value = ''; $('#np-title').value = ''; $('#newm').close(); openById(d.id); }
    else $('#np-msg').textContent = d.error || ('Lỗi ' + x.status);
  };
  x.onerror = () => { $('#np-go').disabled = false; $('#np-msg').textContent = 'Mất kết nối khi tải lên.'; };
  x.send(fd);
};

/* ---------- Trong trang chỉnh sửa: quay về + lịch sử ---------- */
$('#back').onclick = () => showProjects();
$('#hist').onclick = async () => {
  if(!(await flush())) return;
  const rows = await api(`/api/projects/${proj.id}/revisions`);
  $('#hlist').innerHTML = rows.map(r => `<div style="display:flex;gap:8px;align-items:center;padding:6px 0;border-bottom:1px solid var(--line)"><div style="flex:1"><strong>${esc(REASON[r.reason] || r.reason)}</strong> · ${esc(r.name || '')}
    <div class="tag">${esc(localTime(r.created_at))} · ${r.n_cues} cue, đã duyệt ${r.n_ok}</div></div>
    <button data-rid="${r.id}">Khôi phục</button></div>`).join('') || '<p class="tag">Chưa có lịch sử.</p>';
  $('#histd').showModal();
};
$('#hlist').onclick = async e => {
  const id = e.target.dataset.rid; if(!id || !confirm('Khôi phục phiên bản này? Bản hiện tại vẫn nằm trong lịch sử.')) return;
  const p = await api(`/api/projects/${proj.id}/revisions/${id}/restore`, { method:'POST' });
  $('#histd').close(); loadCues(p); setSaved('Đã khôi phục');
};
$('#hclose2').onclick = () => $('#histd').close();

/* ---------- Khởi động ---------- */
(async () => {
  cfg = await api('/api/config', { noAuth:true }).catch(() => ({}));
  try{ me = await api('/api/me', { noAuth:true }); await showProjects(); } catch(_){ showLanding(); }
})();
