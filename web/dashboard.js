/* Дашборд акимата: очередь ремонта, карта, смена статусов. */

const $ = (id) => document.getElementById(id);
const COLORS = { high: '#c62828', medium: '#ef6c00', low: '#2e7d32' };
const STATUSES = { new: 'Новая', in_progress: 'В работе', done: 'Выполнено', rejected: 'Отклонена' };

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

let map = null, layer = null;

function initMap() {
  if (typeof L === 'undefined') { $('mapCard').classList.add('hidden'); return; }
  map = L.map('map').setView([44.848, 65.512], 12);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, attribution: '© OpenStreetMap',
  }).addTo(map);
  layer = L.layerGroup().addTo(map);
}

async function load() {
  const qs = new URLSearchParams();
  if ($('fLevel').value) qs.set('level', $('fLevel').value);
  if ($('fStatus').value) qs.set('status', $('fStatus').value);

  let rows = [], stats = {};
  try {
    [rows, stats] = await Promise.all([
      fetch('/api/reports?' + qs).then(r => r.json()),
      fetch('/api/reports/stats').then(r => r.json()),
    ]);
  } catch {
    $('empty').textContent = 'Сервис недоступен. Проверьте, что сервер запущен.';
    $('empty').classList.remove('hidden');
    return;
  }

  renderKpi(stats);
  renderRows(rows);
  renderMap(rows);
}

function renderKpi(s) {
  const b = s.by_level || {};
  $('kpi').innerHTML = `
    <div class="metric"><div class="v">${s.defects ?? 0}</div><div class="l">Выявлено дефектов</div></div>
    <div class="metric"><div class="v" style="color:${COLORS.high}">${b.high ?? 0}</div>
      <div class="l">Высокий приоритет</div></div>
    <div class="metric"><div class="v" style="color:${COLORS.medium}">${b.medium ?? 0}</div>
      <div class="l">Средний приоритет</div></div>
    <div class="metric"><div class="v" style="color:${COLORS.low}">${b.low ?? 0}</div>
      <div class="l">Низкий приоритет</div></div>
    <div class="metric"><div class="v">${s.manual_review ?? 0}</div><div class="l">На ручной проверке</div></div>
    <div class="metric"><div class="v">${s.open ?? 0}</div><div class="l">В работе и новых</div></div>`;
}

function renderRows(rows) {
  $('empty').classList.toggle('hidden', rows.length > 0);
  $('rows').innerHTML = rows.map(r => `
    <tr>
      <td><img class="thumb" src="${esc(r.image_url)}" alt="Снимок заявки ${esc(r.report_id)}" loading="lazy"></td>
      <td class="mono small">${esc(r.report_id)}</td>
      <td>
        <div>${esc(r.zone_name || r.zone_id)}</div>
        <div class="small muted">${esc(r.road_type_label)}</div>
      </td>
      <td>
        ${esc(r.defect_label)}
        ${r.needs_manual_review ? '<span class="badge gray" title="Низкая уверенность">проверка</span>' : ''}
      </td>
      <td><span class="badge ${esc(r.priority_level)}">${esc(r.priority_label)}</span></td>
      <td class="num"><b>${r.score.toFixed(2)}</b></td>
      <td class="num">${(r.confidence * 100).toFixed(0)}%</td>
      <td class="small">${esc(r.due_date)}</td>
      <td>
        <select data-id="${esc(r.report_id)}" class="st" style="width:auto;padding:6px 9px;font-size:13px">
          ${Object.entries(STATUSES).map(([k, v]) =>
            `<option value="${k}"${k === r.status ? ' selected' : ''}>${v}</option>`).join('')}
        </select>
      </td>
    </tr>`).join('');

  document.querySelectorAll('select.st').forEach(sel => {
    sel.onchange = async () => {
      await fetch(`/api/reports/${sel.dataset.id}/status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: sel.value }),
      });
      load();
    };
  });
}

function renderMap(rows) {
  if (!map) return;
  layer.clearLayers();
  const pts = rows.filter(r => r.lat && r.lon);
  pts.forEach(r => {
    L.circleMarker([r.lat, r.lon], {
      radius: Math.min(22, 6 + r.score * 0.8),
      color: COLORS[r.priority_level] || '#666',
      fillColor: COLORS[r.priority_level] || '#666',
      fillOpacity: .55, weight: 2,
    }).bindPopup(
      `<b>${esc(r.defect_label)}</b><br>${esc(r.zone_name || r.zone_id)}<br>` +
      `Приоритет: ${esc(r.priority_label)} (${r.score.toFixed(2)})<br>` +
      `Срок до ${esc(r.due_date)}<br>` +
      `<img src="${esc(r.image_url)}" style="width:190px;border-radius:6px;margin-top:6px">`
    ).addTo(layer);
  });
  if (pts.length) {
    map.fitBounds(L.latLngBounds(pts.map(r => [r.lat, r.lon])).pad(0.25));
  }
}

$('refresh').onclick = load;
$('fLevel').onchange = load;
$('fStatus').onchange = load;

initMap();
load();
setInterval(load, 15000);   // заявки от жителей появляются в очереди сами
