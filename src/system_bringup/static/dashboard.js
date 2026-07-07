const stateUrl = '/api/status';
const mapImg = new Image();
const riskImg = new Image();
let latest = null;
let mapSeq = -1;
let riskSeq = -1;
let mapReady = false;
let riskReady = false;
const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
const roleColors = {leader: '#58a6ff', follower: '#63d297', member: '#f2cc60'};

function statusClass(status) {
  return String(status || 'NO DATA').toLowerCase().replaceAll(' ', '-');
}

function fmt(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : '--';
}

function ageText(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)}s` : '--';
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
  const img = document.getElementById('omxStream');
  if (img.dataset.configured === '1') return;
  const port = s.omx_debug.port;
  const path = s.omx_debug.stream_path || '/stream.mjpg';
  img.src = `${location.protocol}//${location.hostname}:${port}${path}`;
  img.dataset.configured = '1';
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
  const rows = [
    [s.map.topic, s.map.status, s.map.age_sec],
    [s.risk.topic, s.risk.status, s.risk.age_sec],
    [s.fleet.coordination_status.topic, s.fleet.coordination_status.status, s.fleet.coordination_status.age_sec],
    [s.fleet.collision_warning.topic, s.fleet.collision_warning.status, s.fleet.collision_warning.age_sec],
    ['/omx/state', s.omx.state ? 'OK' : 'NO DATA', s.omx.state_received_wall_sec ? s.server_time_sec - s.omx.state_received_wall_sec : null],
    ['/omx/status', s.omx.status ? 'OK' : 'NO DATA', s.omx.status_received_wall_sec ? s.server_time_sec - s.omx.status_received_wall_sec : null],
  ];
  document.getElementById('topicRows').innerHTML = rows.map(row => {
    const cls = statusClass(row[1]);
    return `<div class="topic-row"><span>${row[0]}</span><span class="status-${cls}">${row[1]} ${ageText(row[2])}</span></div>`;
  }).join('');
}

function updateTop(s) {
  const leader = s.robots.find(r => r.name === 'leader') || {};
  const online = s.robots.filter(r => r.status === 'ONLINE').length;
  setPill('leaderPill', 'Leader', leader.status);
  setPill('mapPill', 'Map', s.map.status);
  setPill('riskPill', 'Risk', s.risk.status);
  const rp = document.getElementById('robotPill');
  rp.className = `pill ${online ? 'online' : 'no-data'}`;
  rp.innerHTML = `<span class="dot"></span>Robots ${online}/${s.robots.length}`;
  document.getElementById('mapWarning').textContent = (!s.risk.metadata_matches_map && s.risk.status !== 'NO DATA')
    ? 'Risk overlay metadata does not match /map, so overlay rendering is suppressed.'
    : '';
}

async function refresh() {
  try {
    const s = await (await fetch(stateUrl, {cache: 'no-store'})).json();
    latest = s;
    configureStream(s);
    updateImages(s);
    updateTop(s);
    updateTables(s);
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
