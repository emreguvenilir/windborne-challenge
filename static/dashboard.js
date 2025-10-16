let map = L.map('map').setView([20, 0], 2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

let markers = {};
let activePrediction = null;
let balloonData = [];

// --- Fetch balloon data ---
async function fetchBalloons() {
  const res = await fetch('/api/balloons');
  balloonData = await res.json();

  // Clear previous markers
  Object.values(markers).forEach(m => map.removeLayer(m));
  markers = {};

  balloonData.forEach(b => {
    if (!b.latitude || !b.longitude) return;

    // Blue icon for current position
    const balloonIcon = L.divIcon({
      html: '<i class="fa-solid fa-location-arrow" style="color:#00bfff; font-size:18px;"></i>',
      className: 'balloon-icon',
      iconSize: [20, 20],
      iconAnchor: [10, 10]
    });

    const marker = L.marker([b.latitude, b.longitude], { icon: balloonIcon }).addTo(map);
    marker.on('click', () => showPrediction(b));
    markers[b.balloon_id] = marker;
  });

  console.log(`Loaded ${balloonData.length} balloons`);
}

// --- Show balloon details and predicted path on click ---
function showPrediction(b) {
  showBalloonDetails(b);

  // Remove previous prediction (if any)
  if (activePrediction) {
    map.removeLayer(activePrediction.marker);
    map.removeLayer(activePrediction.line);
    activePrediction = null;
  }

  // Only draw prediction if available
  if (b.pred_latitude && b.pred_longitude) {
    const predIcon = L.divIcon({
      html: '<i class="fa-solid fa-location-arrow" style="color:#ff4b4b; font-size:18px;"></i>',
      className: 'predicted-icon',
      iconSize: [20, 20],
      iconAnchor: [10, 10]
    });

    const predMarker = L.marker([b.pred_latitude, b.pred_longitude], { icon: predIcon }).addTo(map);
    const line = L.polyline(
      [[b.latitude, b.longitude], [b.pred_latitude, b.pred_longitude]],
      { color: '#ff4b4b', weight: 2, opacity: 0.7, dashArray: '4, 4' }
    ).addTo(map);

    activePrediction = { marker: predMarker, line };
  }
}

// --- Balloon info sidebar ---
function showBalloonDetails(b) {
  const info = document.getElementById('balloon-info');

  const latColor = b.pred_latitude > b.latitude ? 'green' : 'red';
  const lonColor = b.pred_longitude > b.longitude ? 'green' : 'red';
  const altColor = b.pred_altitude > b.altitude ? 'green' : 'red';

info.innerHTML = `
    <h3><i class="fa-solid fa-balloon"></i> Balloon #${b.balloon_id}</h3>
    <p style="font-size:1.1rem;"><i class="fa-solid fa-location-dot"></i> 
         <b>Lat:</b> ${b.latitude?.toFixed(2)}, 
         <b>Lon:</b> ${b.longitude?.toFixed(2)}</p>
    <p style="font-size:1.2rem;"><i class="fa-solid fa-mountain"></i> 
         <b>Altitude:</b> ${b.altitude?.toFixed(2)} km</p>
    <p style="font-size:1.2rem;"><i class="fa-solid fa-temperature-half"></i> 
         <b>Temperature:</b> ${b.temperature_2m?.toFixed(1)} °C</p>
    <p style="font-size:1.2rem;"><i class="fa-solid fa-wind"></i> 
         <b>Wind Speed:</b> ${b.wind_speed_10m?.toFixed(1)} m/s</p>
    <p style="font-size:1.2rem;"><i class="fa-solid fa-cloud"></i> 
         <b>Cloud Cover:</b> ${b.cloud_cover?.toFixed(0)}%</p>
    <p style="font-size:1.2rem;"><i class="fa-solid fa-compass"></i> 
         <b>Wind Direction:</b> ${b.wind_direction_10m?.toFixed(0)}°</p>

    <h4 style="margin-top:0.6rem; font-size:1.1rem;">Predicted Location (Next Hour)</h4>
    <p style="font-size:1.1rem;">
        <b>Lat:</b> <span style="color:${latColor}">${b.pred_latitude?.toFixed(2)}</span>
    </p>
    <p style="font-size:1.1rem;">
        <b>Lon:</b> <span style="color:${lonColor}">${b.pred_longitude?.toFixed(2)}</span>
    </p>
    <p style="font-size:1.1rem;">
        <b>Alt:</b> <span style="color:${altColor}">${b.pred_altitude?.toFixed(2)} km</span>
    </p>
`;
}

// --- Fetch model metrics ---
async function fetchModelMetrics() {
  try {
    const res = await fetch('/api/model_metrics');
    if (!res.ok) throw new Error('Failed to load metrics');
    const data = await res.json();

    const safe = (v) => (v != null ? v.toFixed(3) : '—');

    $('#last-update').text(data.last_updated || '—');
    $('#overall-mse').text(safe(data.overall?.mse));
    $('#overall-mae').text(safe(data.overall?.mae));
    $('#overall-corr').text(safe(data.overall?.corr));

    const key = data.key_features || {};
    $('#lat-mse').text(safe(key.latitude?.mse));
    $('#lat-mae').text(safe(key.latitude?.mae));
    $('#lat-corr').text(safe(key.latitude?.corr));

    $('#lon-mse').text(safe(key.longitude?.mse));
    $('#lon-mae').text(safe(key.longitude?.mae));
    $('#lon-corr').text(safe(key.longitude?.corr));

    $('#alt-mse').text(safe(key.altitude?.mse));
    $('#alt-mae').text(safe(key.altitude?.mae));
    $('#alt-corr').text(safe(key.altitude?.corr));

    if (data.loss_curve_path) {
      $('#loss-curve-img').attr('src', data.loss_curve_path + '?t=' + Date.now());
    }
  } catch (err) {
    console.error(err);
    $('#metrics').html('<p style="color:red;">Error loading model metrics</p>');
  }
}

// --- Initial fetches ---
fetchBalloons();
fetchModelMetrics();

// Update frequencies
setInterval(fetchBalloons, 300000);       // refresh map every 5 min
setInterval(fetchModelMetrics, 21600000); // refresh model metrics every 6 hours