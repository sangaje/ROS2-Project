const stateUrl = '/api/status';
const yoloStatusUrl = '/api/yolo_status';
const mapImg = new Image();
const riskImg = new Image();
let latest = null;
let latestYolo = null;
let mapSeq = -1;
let riskSeq = -1;
let mapReady = false;
let riskReady = false;
const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
const roleColors = {leader: '#58a6ff', follower: '#63d297'};

function statusClass(status) {
  return String(status || 'NO DATA').toLowerCase().replaceAll(' ', '-');
}

function fmt(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : '--';
}

function ageText(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)}s` : '--';
}

function pctText(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : '--';
}

function yesNo(value) {
  if (value === true) return 'YES';
  if (value === false) return 'NO';
  return '--';
}

function escapeHtml(value) {
  return String(value ?? '--')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function yawDeg(rad) {
  return Number.isFinite(rad) ? `${(rad * 180.0 / Math.PI).toFixed(0)} deg` : '--';
}

function setPill(id, label, status) {
  const el = document.getElementById(id);
  el.className = `pill ${statusClass(status)}`;
  el.innerHTML = `<span class="dot"></span>${label} ${status || 'NO DATA'}`;
}

function configureStream(s) {
  const omx = document.getElementById('omxStream');
  if (omx.dataset.configured !== '1') {
    const port = s.omx_debug.port;
    const path = s.omx_debug.stream_path || '/stream.mjpg';
    omx.src = `${location.protocol}//${location.hostname}:${port}${path}`;
    omx.dataset.configured = '1';
  }

  const yolo = s.yolo_server || {};
  const yoloPort = yolo.port || 5005;
  const streams = [
    ['scoutRawStream', yolo.raw_stream_path || '/stream/raw.mjpg'],
    ['scoutYoloStream', yolo.overlay_stream_path || '/stream/yolo.mjpg'],
  ];
  streams.forEach(([id, path]) => {
    const img = document.getElementById(id);
    if (img.dataset.configured === '1') return;
    img.src = `${location.protocol}//${location.hostname}:${yoloPort}${path}`;
    img.dataset.configured = '1';
  });
}

function updateImages(s) {
  if (s.map.seq !== mapSeq && s.map.status !== 'NO DATA') {
    mapSeq = s.map.seq;
    mapReady = false;
    mapImg.src = `/api/map.png?v=${mapSeq}&t=${Date.now()}`;
  }
  if (s.risk.seq !== riskSeq && s.risk.status !== 'NO DATA') {
    riskSeq = s.risk.seq;
    riskReady = false;
    riskImg.src = `/api/risk.png?v=${riskSeq}&t=${Date.now()}`;
  }
}

mapImg.onload = () => { mapReady = true; draw(); };
riskImg.onload = () => { riskReady = true; draw(); };
mapImg.onerror = () => { mapReady = false; draw(); };
riskImg.onerror = () => { riskReady = false; draw(); };

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const w = Math.max(320, Math.floor(rect.width * ratio));
  const h = Math.max(360, Math.floor(rect.height * ratio));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function mapViewport(meta) {
  const scale = Math.min(canvas.width / meta.width, canvas.height / meta.height);
  return {
    scale,
    x: (canvas.width - meta.width * scale) * 0.5,
    y: (canvas.height - meta.height * scale) * 0.5,
    w: meta.width * scale,
    h: meta.height * scale,
  };
}

function worldToCell(meta, x, y) {
  const o = meta.origin;
  const dx = x - o.x;
  const dy = y - o.y;
  const c = Math.cos(o.yaw || 0.0);
  const s = Math.sin(o.yaw || 0.0);
  return {
    x: (c * dx + s * dy) / meta.resolution,
    y: (-s * dx + c * dy) / meta.resolution,
  };
}

function cellToCanvas(meta, vp, cell) {
  return {
    x: vp.x + cell.x * vp.scale,
    y: vp.y + (meta.height - cell.y) * vp.scale,
  };
}

function drawRobot(meta, vp, robot) {
  if (!robot.position || !Number.isFinite(robot.yaw_rad)) return;
  const cell = worldToCell(meta, robot.position.x, robot.position.y);
  if (cell.x < -1 || cell.x > meta.width + 1 || cell.y < -1 || cell.y > meta.height + 1) return;
  const p = cellToCanvas(meta, vp, cell);
  const color = roleColors[robot.role] || '#d0d7de';
  const stale = robot.status !== 'ONLINE';
  const radius = Math.max(5, Math.min(11, vp.scale * 0.18));
  const yawGrid = robot.yaw_rad - (meta.origin.yaw || 0.0);
  ctx.save();
  ctx.globalAlpha = stale ? 0.48 : 1.0;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  ctx.lineTo(p.x + Math.cos(yawGrid) * radius * 2.4, p.y - Math.sin(yawGrid) * radius * 2.4);
  ctx.stroke();
  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillStyle = '#f6f8fa';
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 3;
  const label = `${robot.name} (${robot.role})`;
  ctx.strokeText(label, p.x + radius + 5, p.y - radius - 5);
  ctx.fillText(label, p.x + radius + 5, p.y - radius - 5);
  ctx.restore();
}

function topicAge(s, key) {
  const t = s.omx[`${key}_received_wall_sec`];
  return t ? s.server_time_sec - t : null;
}

function metricCard(label, value, status = 'OK', sub = '') {
  const cls = statusClass(status);
  return `<div class="metric-card ${cls}">
    <div class="metric-label">${escapeHtml(label)}</div>
    <div class="metric-value">${escapeHtml(value)}</div>
    ${sub ? `<div class="metric-sub">${escapeHtml(sub)}</div>` : ''}
  </div>`;
}

function formatPoint(p) {
  if (!p) return '--';
  return `(${fmt(p.x)}, ${fmt(p.y)}, ${fmt(p.z)})`;
}

function updateOmxPanel(s) {
  const err = s.omx.aim_error_norm || {};
  const cmd = s.omx.leader_cmd_vel || {};
  const detected = s.omx.target_detected === true;
  const fireDisabled = s.omx.fire_disabled;
  const cards = [
    ['State', s.omx.state || '--', s.omx.state ? 'OK' : 'NO DATA', ageText(topicAge(s, 'state'))],
    ['Status', s.omx.status || '--', s.omx.status ? 'OK' : 'NO DATA', ageText(topicAge(s, 'status'))],
    ['Target', yesNo(s.omx.target_detected), detected ? 'OK' : 'NO DATA', ageText(topicAge(s, 'target_detected'))],
    ['Aim', pctText(s.omx.aim_progress), s.omx.aim_progress ? 'OK' : 'NO DATA', `err ${fmt(err.magnitude, 3)}`],
    ['Queue', s.omx.queue_size ?? '--', s.omx.queue_size ? 'OK' : 'NO DATA', ageText(topicAge(s, 'queue_size'))],
    ['Fire', s.omx.fire_status || '--', s.omx.fire_status ? 'OK' : 'NO DATA', ageText(topicAge(s, 'fire_status'))],
    ['Fire Lock', fireDisabled === true ? 'DISABLED' : fireDisabled === false ? 'ARMED' : '--', fireDisabled === false ? 'OK' : fireDisabled === true ? 'STALE' : 'NO DATA', ageText(topicAge(s, 'fire_disabled'))],
    ['Nav Result', s.omx.waffle_nav_result || '--', s.omx.waffle_nav_result ? 'OK' : 'NO DATA', ageText(topicAge(s, 'waffle_nav_result'))],
    ['Waffle', s.omx.waffle_status || '--', s.omx.waffle_status ? 'OK' : 'NO DATA', ageText(topicAge(s, 'waffle_status'))],
    ['Cmd Vel', `x ${fmt(cmd.linear_x)} / z ${fmt(cmd.angular_z)}`, s.omx.leader_cmd_vel ? 'OK' : 'NO DATA', ageText(topicAge(s, 'leader_cmd_vel'))],
  ];
  document.getElementById('omxCards').innerHTML = cards
    .map(([label, value, status, sub]) => metricCard(label, value, status, sub))
    .join('');
}

function updateYoloPanel(y) {
  const data = y && y.data ? y.data : {};
  const detections = Array.isArray(data.detections) ? data.detections : [];
  const cards = [
    ['Server', y ? y.status : 'NO DATA', y ? y.status : 'NO DATA', y ? `${fmt(y.latency_ms, 0)} ms` : ''],
    ['Raw FPS', fmt(data.raw_fps, 1), data.ok ? 'OK' : 'NO DATA', `${data.raw_frames ?? 0} frames`],
    ['YOLO FPS', fmt(data.yolo_fps, 1), data.yolo_frames ? 'OK' : 'NO DATA', `${data.yolo_frames ?? 0} frames`],
    ['People', data.people ?? '--', data.people > 0 ? 'OK' : 'NO DATA', `${detections.length} detections`],
    ['Latency', `${fmt(data.latency_ms, 1)} ms`, data.yolo_frames ? 'OK' : 'NO DATA', `pred ${fmt(data.predict_ms, 1)} ms`],
    ['Frame Age', `${fmt(data.raw_frame_age_sec, 2)} s`, data.raw_frame_age_sec < 2 ? 'OK' : 'STALE', `${data.image_width || '--'}x${data.image_height || '--'}`],
  ];
  document.getElementById('yoloCards').innerHTML = cards
    .map(([label, value, status, sub]) => metricCard(label, value, status, sub))
    .join('');
}

function updateEvents(s) {
  const names = [
    'fire',
    'target_processed',
    'target_lost',
    'target_blocked',
    'target_not_found',
    'nav_goal',
    'nav_cancel',
    'patrol_complete',
  ];
  document.getElementById('eventRows').innerHTML = names.map(name => {
    const e = s.events[name] || {};
    const last = e.last ? formatPoint(e.last) : '';
    const cls = statusClass(e.status);
    return `<div class="topic-row event-row">
      <span>${escapeHtml(e.topic || name)}<small>${escapeHtml(last)}</small></span>
      <span class="status-${cls}">#${e.count || 0} ${escapeHtml(e.status || 'NO DATA')} ${ageText(e.age_sec)}</span>
    </div>`;
  }).join('');
}

function gridMetaText(grid) {
  const m = grid.metadata;
  if (!m) return '--';
  return `${m.width}x${m.height} @ ${fmt(m.resolution, 3)}m frame=${m.frame_id}`;
}

function updateMapMeta(s) {
  const rows = [
    ['Map', s.map.status, s.map.age_sec, gridMetaText(s.map)],
    ['Risk', s.risk.status, s.risk.age_sec, gridMetaText(s.risk)],
    ['Risk Overlay', s.risk.metadata_matches_map ? 'OK' : 'STALE', null, s.risk.metadata_matches_map ? 'metadata match' : 'metadata mismatch'],
  ];
  document.getElementById('mapMetaRows').innerHTML = rows.map(row => {
    const cls = statusClass(row[1]);
    return `<div class="topic-row"><span>${escapeHtml(row[0])}<small>${escapeHtml(row[3])}</small></span><span class="status-${cls}">${escapeHtml(row[1])} ${ageText(row[2])}</span></div>`;
  }).join('');
}

function draw() {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#08090a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!latest || !latest.map.metadata) {
    ctx.fillStyle = '#9aa8b1';
    ctx.font = '15px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('Waiting for /map', 18, 32);
    return;
  }
  const meta = latest.map.metadata;
  const vp = mapViewport(meta);
  if (document.getElementById('layerMap').checked && mapReady) {
    ctx.drawImage(mapImg, vp.x, vp.y, vp.w, vp.h);
  }
  if (document.getElementById('layerRisk').checked && riskReady && latest.risk.metadata_matches_map) {
    ctx.save();
    ctx.globalAlpha = Number(document.getElementById('riskOpacity').value);
    ctx.drawImage(riskImg, vp.x, vp.y, vp.w, vp.h);
    ctx.restore();
  }
  if (document.getElementById('layerRobots').checked) {
    latest.robots.forEach(robot => drawRobot(meta, vp, robot));
  }
  ctx.strokeStyle = '#4b5560';
  ctx.lineWidth = 1;
  ctx.strokeRect(vp.x, vp.y, vp.w, vp.h);
}

function updateTables(s) {
  document.getElementById('robotRows').innerHTML = s.robots.map(r => {
    const pos = r.position || {};
    const sc = statusClass(r.status);
    return `<tr>
      <td>${r.name}</td>
      <td class="role ${r.role}">${r.role}</td>
      <td class="status-${sc}">${r.status}</td>
      <td>${fmt(pos.x)}</td>
      <td>${fmt(pos.y)}</td>
      <td>${yawDeg(r.yaw_rad)}</td>
      <td>${ageText(r.age_sec)}</td>
    </tr>`;
  }).join('');
  const fixedRows = [
    [s.map.topic, s.map.status, s.map.age_sec],
    [s.risk.topic, s.risk.status, s.risk.age_sec],
    [s.fleet.coordination_status.topic, s.fleet.coordination_status.status, s.fleet.coordination_status.age_sec],
    [s.fleet.collision_warning.topic, s.fleet.collision_warning.status, s.fleet.collision_warning.age_sec],
  ];
  const seen = new Set(fixedRows.map(row => row[0]));
  const dynamicRows = Object.entries(s.topics || {})
    .filter(([topic]) => !seen.has(topic))
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([topic, info]) => [topic, info.status, info.age_sec, info.type, info.value]);
  const rows = fixedRows.map(row => [...row, '', null]).concat(dynamicRows);
  document.getElementById('topicRows').innerHTML = rows.map(row => {
    const cls = statusClass(row[1]);
    const value = row[4] === null || row[4] === undefined ? '' : JSON.stringify(row[4]);
    const detail = [row[3], value].filter(Boolean).join(' | ');
    return `<div class="topic-row">
      <span>${escapeHtml(row[0])}${detail ? `<small>${escapeHtml(detail)}</small>` : ''}</span>
      <span class="status-${cls}">${escapeHtml(row[1])} ${ageText(row[2])}</span>
    </div>`;
  }).join('');
}

function updateTop(s) {
  const leader = s.robots.find(r => r.name === 'leader') || {};
  const online = s.robots.filter(r => r.status === 'ONLINE').length;
  setPill('leaderPill', 'Leader', leader.status);
  setPill('mapPill', 'Map', s.map.status);
  setPill('riskPill', 'Risk', s.risk.status);
  setPill('yoloPill', 'YOLO', latestYolo ? latestYolo.status : 'NO DATA');
  const fireStatus = s.omx.fire_disabled === false ? 'ARMED' : s.omx.fire_status ? s.omx.fire_status : 'NO DATA';
  setPill('firePill', 'Fire', fireStatus);
  const rp = document.getElementById('robotPill');
  rp.className = `pill ${online ? 'online' : 'no-data'}`;
  rp.innerHTML = `<span class="dot"></span>Robots ${online}/${s.robots.length}`;
  document.getElementById('mapWarning').textContent = (!s.risk.metadata_matches_map && s.risk.status !== 'NO DATA')
    ? 'Risk overlay metadata does not match /map, so overlay rendering is suppressed.'
    : '';
}

async function refresh() {
  try {
    const [stateResp, yoloResp] = await Promise.all([
      fetch(stateUrl, {cache: 'no-store'}),
      fetch(yoloStatusUrl, {cache: 'no-store'}),
    ]);
    const s = await stateResp.json();
    latestYolo = await yoloResp.json();
    latest = s;
    configureStream(s);
    updateImages(s);
    updateTop(s);
    updateTables(s);
    updateOmxPanel(s);
    updateYoloPanel(latestYolo);
    updateEvents(s);
    updateMapMeta(s);
    draw();
  } catch (err) {
    console.warn('dashboard refresh failed', err);
  }
}

['layerMap', 'layerRisk', 'layerRobots', 'riskOpacity'].forEach(id => {
  document.getElementById(id).addEventListener('input', draw);
  document.getElementById(id).addEventListener('change', draw);
});
window.addEventListener('resize', draw);
refresh();
setInterval(refresh, 500);
