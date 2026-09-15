/* Страница жителя: загрузка фото -> анализ -> карточка результата. */

const $ = (id) => document.getElementById(id);
let DICT = null, ZONES = [], FILE = null, LAST = null;

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const fill = (sel, obj, selected) => {
  sel.innerHTML = Object.entries(obj)
    .map(([k, v]) => `<option value="${esc(k)}"${k === selected ? ' selected' : ''}>${esc(v)}</option>`)
    .join('');
};

async function init() {
  try {
    const [d, z] = await Promise.all([
      fetch('/api/dictionaries').then(r => r.json()),
      fetch('/api/zones').then(r => r.json()),
    ]);
    DICT = d; ZONES = z;
  } catch {
    showError('Сервис недоступен. Проверьте, что сервер запущен.');
    return;
  }

  $('zone').innerHTML =
    ZONES.map(z => `<option value="${esc(z.zone_id)}">${esc(z.zone_id)} — ${esc(z.zone_name)}</option>`).join('')
    + '<option value="">— указать вручную —</option>';

  fill($('road'), DICT.road_types, 'main_road');
  fill($('traffic'), DICT.traffic_levels, 'medium');
  fill($('weather'), DICT.weather, 'dry');

  $('flags').innerHTML = Object.entries(DICT.context_flags).map(([k, v]) =>
    `<label><input type="checkbox" value="${esc(k)}">${esc(v)}</label>`).join('');

  $('zone').onchange = () => {
    const manual = $('zone').value === '';
    $('manual').classList.toggle('hidden', !manual);
    $('flagsWrap').classList.toggle('hidden', !manual);
  };
}

/* ---------- выбор файла ---------- */
const dz = $('dz');
dz.onclick = () => $('file').click();
dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('over'); };
dz.ondragleave = () => dz.classList.remove('over');
dz.ondrop = (e) => {
  e.preventDefault(); dz.classList.remove('over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
};
$('file').onchange = (e) => e.target.files[0] && setFile(e.target.files[0]);

function setFile(f) {
  if (!f.type.startsWith('image/')) return showError('Это не изображение.');
  if (f.size > 10 * 1024 * 1024) return showError('Файл больше 10 МБ.');
  FILE = f;
  const img = $('preview');
  img.src = URL.createObjectURL(f);
  img.style.display = 'block';
  $('send').disabled = false;
  $('err').classList.add('hidden');
  $('result').classList.add('hidden');
}

function showError(msg) {
  $('err').textContent = msg;
  $('err').classList.remove('hidden');
}

/* ---------- отправка ---------- */
$('send').onclick = async () => {
  if (!FILE) return;
  const btn = $('send');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Анализируем снимок…';
  $('err').classList.add('hidden');

  const fd = new FormData();
  fd.append('file', FILE);
  fd.append('zone_id', $('zone').value);
  fd.append('road_type', $('road').value);
  fd.append('traffic_level', $('traffic').value);
  fd.append('weather', $('weather').value);
  fd.append('anonymize', $('anon').checked);
  fd.append('context_flags',
    [...$('flags').querySelectorAll('input:checked')].map(i => i.value).join(','));

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: fd });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      let msg = e.detail || `Ошибка сервера (${res.status})`;
      if (Array.isArray(e.detail)) {
        // 422 от pydantic приходит списком — разворачиваем в читаемый текст
        msg = e.detail.map(d => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join('; ');
      }
      if (e.hint) msg += ` — ${e.hint}`;
      throw new Error(msg);
    }
    LAST = await res.json();
    renderResult(LAST);
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Проанализировать';
  }
};

/* ---------- результат ---------- */
function renderResult(r) {
  const d = r.detection, p = r.priority, q = r.quality;
  const noDefect = d.defect_type === 'none';
  const cls = noDefect ? 'none' : p.level;

  const maxAbs = Math.max(...p.factors.map(f => Math.abs(f.contribution)), 1);
  const factors = p.factors.map(f => {
    const w = Math.round(Math.abs(f.contribution) / maxAbs * 100);
    const neg = f.contribution < 0 ? ' neg' : '';
    return `<div class="factor">
      <div class="txt">${esc(f.explanation)}</div>
      <div class="bar"><i class="${neg.trim()}" style="width:${w}%"></i></div>
      <div class="val">${f.contribution >= 0 ? '+' : ''}${f.contribution.toFixed(2)}</div>
    </div>`;
  }).join('');

  $('result').innerHTML = `
  <div class="card">
    <div class="verdict ${cls}">
      <div>
        <div class="t">${noDefect ? 'Дефект не выявлен' : esc(d.defect_label)}</div>
        <div class="s">${noDefect
          ? 'Покрытие на снимке в нормативном состоянии'
          : 'Приоритет ремонта: ' + esc(p.label) + ' · срок реакции ' + p.sla_days + ' дн.'}</div>
      </div>
      <div style="font-size:30px;font-weight:700">${noDefect ? '✓' : p.score.toFixed(1)}</div>
    </div>

    <img src="${esc(r.image_url)}" alt="Снимок с отмеченным дефектом"
         style="width:100%;border-radius:11px;margin:16px 0">

    <div class="metrics">
      <div class="metric"><div class="v">${(p.final_confidence * 100).toFixed(0)}%</div>
        <div class="l">Уверенность</div></div>
      <div class="metric"><div class="v">${noDefect ? '—' : d.severity + '/5'}</div>
        <div class="l">Тяжесть дефекта</div></div>
      <div class="metric"><div class="v">${r.processing_ms} мс</div>
        <div class="l">Время анализа</div></div>
      <div class="metric"><div class="v mono">${esc(r.report_id)}</div>
        <div class="l">Номер заявки</div></div>
    </div>

    ${d.description ? `<h3>Описание</h3><p style="margin:0">${esc(d.description)}</p>` : ''}
    ${!noDefect && d.location_text ? `<p class="small muted" style="margin:6px 0 0">
        Расположение: ${esc(d.location_text)}</p>` : ''}

    ${d.evidence?.length ? `<h3>На чём основан вывод</h3>
      <ul class="evidence">${d.evidence.map(e => `<li>${esc(e)}</li>`).join('')}</ul>` : ''}

    ${p.needs_manual_review ? `<div class="note warn">
      <b>Требуется ручная проверка.</b> ${esc(p.review_reason || '')}</div>` : ''}

    ${q.issues?.length ? `<div class="note info">
      <b>Качество снимка:</b> ${esc(q.issues.join('; '))}</div>` : ''}

    ${r.privacy_blurred?.length ? `<div class="note info">
      🔒 Анонимизировано в кадре: ${esc(r.privacy_blurred.join(', '))}</div>` : ''}

    ${noDefect ? '' : `
      <h3>Почему такой приоритет</h3>
      ${factors}
      <p class="small muted" style="margin-top:10px">
        Итог: <b>${p.score.toFixed(2)}</b> балла. Пороги: ниже
        ${DICT.thresholds.medium} — низкий, ${DICT.thresholds.medium}–${DICT.thresholds.high} — средний,
        от ${DICT.thresholds.high} — высокий. Приоритет считает детерминированная формула,
        а не языковая модель.
      </p>

      <h3>Тот же дефект в другом месте</h3>
      <div class="row">
        <div class="field">
          <label for="wiRoad">Перенести на участок типа</label>
          <select id="wiRoad"></select>
        </div>
        <div class="field">
          <label for="wiTraffic">Интенсивность движения</label>
          <select id="wiTraffic"></select>
        </div>
      </div>
      <div id="wiOut" class="metrics"></div>
    `}
  </div>`;

  $('result').classList.remove('hidden');
  $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });

  if (!noDefect) {
    fill($('wiRoad'), DICT.road_types, 'yard');
    fill($('wiTraffic'), DICT.traffic_levels, 'low');
    $('wiRoad').onchange = $('wiTraffic').onchange = whatIf;
    whatIf();
  }
}

/* Пересчёт приоритета в другом контексте — модель повторно не вызывается. */
async function whatIf() {
  if (!LAST) return;
  const res = await fetch('/api/what-if', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      report_id: LAST.report_id,
      road_type: $('wiRoad').value,
      traffic_level: $('wiTraffic').value,
      context_flags: [],
    }),
  });
  if (!res.ok) return;
  const alt = await res.json();
  const cur = LAST.priority;

  $('wiOut').innerHTML = `
    <div class="metric">
      <div class="v"><span class="badge ${cur.level}">${esc(cur.label)}</span></div>
      <div class="l">Сейчас: ${esc(LAST.context.road_type_label)} · ${cur.score.toFixed(2)} балла</div>
    </div>
    <div class="metric">
      <div class="v"><span class="badge ${alt.level}">${esc(alt.label)}</span></div>
      <div class="l">Если: ${esc(DICT.road_types[$('wiRoad').value])} · ${alt.score.toFixed(2)} балла</div>
    </div>
    <div class="metric">
      <div class="v">${cur.sla_days} → ${alt.sla_days} дн.</div>
      <div class="l">Изменение срока реакции</div>
    </div>`;
}

init();
