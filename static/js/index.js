
"use strict";

/*
 * Benue Weather Intelligence
 *
 * Weather provider:
 *   Open-Meteo forecast API
 *   Open-Meteo geocoding API
 *
 * Backend contract:
 *   GET  /api/health
 *   GET  /api/lgas
 *   GET  /api/reference
 *   GET  /api/metrics
 *   POST /api/predict
 *   POST /api/batch
 *   GET  /api/batch/<id>/series?lga=<LGA>
 *   GET  /download/<id>.csv
 *
 * Configure a different backend URL before this file loads:
 *   window.BENUE_API_BASE_URL = "https://your-api.example.com";
 *
 * An empty API base URL means the frontend and backend share an origin.
 */

const CONFIG = Object.freeze({
  API_BASE_URL: String(window.BENUE_API_BASE_URL || "").replace(/\/+$/, ""),
  WEATHER_URL: "https://api.open-meteo.com/v1/forecast",
  GEOCODING_URL: "https://geocoding-api.open-meteo.com/v1/search",
  TIMEZONE: "Africa/Lagos",
  WEATHER_REFRESH_MS: 10 * 60 * 1000,
  REQUEST_TIMEOUT_MS: 20000,
  DEFAULT_LOCATION: "Makurdi",
  DEFAULT_COORDINATES: { lat: 7.7322, lon: 8.5391 },
  BENUE_CENTER: { lat: 7.19, lon: 8.13 },
  MAP_ZOOM: 7
});

const BENUE_LOCATIONS = [
  { name: "Ado", lat: 7.55, lon: 8.49 },
  { name: "Agatu", lat: 7.98, lon: 7.92 },
  { name: "Apa", lat: 7.25, lon: 7.83 },
  { name: "Buruku", lat: 7.42, lon: 9.20 },
  { name: "Gboko", lat: 7.32, lon: 9.00 },
  { name: "Guma", lat: 7.78, lon: 8.87 },
  { name: "Gwer East", lat: 7.45, lon: 8.87 },
  { name: "Gwer West", lat: 7.62, lon: 8.25 },
  { name: "Katsina-Ala", lat: 7.17, lon: 9.28 },
  { name: "Konshisha", lat: 7.15, lon: 8.98 },
  { name: "Kwande", lat: 6.80, lon: 9.50 },
  { name: "Logo", lat: 7.28, lon: 9.20 },
  { name: "Makurdi", lat: 7.7322, lon: 8.5391 },
  { name: "Obi", lat: 7.20, lon: 8.36 },
  { name: "Ogbadibo", lat: 7.03, lon: 7.99 },
  { name: "Ohimini", lat: 7.13, lon: 7.94 },
  { name: "Oju", lat: 6.85, lon: 8.42 },
  { name: "Okpokwu", lat: 7.10, lon: 7.81 },
  { name: "Otukpo", lat: 7.19, lon: 8.13 },
  { name: "Tarka", lat: 7.35, lon: 8.99 },
  { name: "Ukum", lat: 7.00, lon: 9.67 },
  { name: "Ushongo", lat: 7.20, lon: 9.00 },
  { name: "Vandeikya", lat: 6.78, lon: 9.07 }
];

const WEATHER_CODES = {
  0: ["Clear sky", "☀"],
  1: ["Mainly clear", "🌤"],
  2: ["Partly cloudy", "⛅"],
  3: ["Overcast", "☁"],
  45: ["Fog", "🌫"],
  48: ["Depositing rime fog", "🌫"],
  51: ["Light drizzle", "🌦"],
  53: ["Moderate drizzle", "🌦"],
  55: ["Dense drizzle", "🌧"],
  56: ["Light freezing drizzle", "🌧"],
  57: ["Dense freezing drizzle", "🌧"],
  61: ["Slight rain", "🌦"],
  63: ["Moderate rain", "🌧"],
  65: ["Heavy rain", "🌧"],
  66: ["Light freezing rain", "🌧"],
  67: ["Heavy freezing rain", "🌧"],
  71: ["Slight snowfall", "🌨"],
  73: ["Moderate snowfall", "🌨"],
  75: ["Heavy snowfall", "❄"],
  77: ["Snow grains", "🌨"],
  80: ["Slight rain showers", "🌦"],
  81: ["Moderate rain showers", "🌧"],
  82: ["Violent rain showers", "⛈"],
  85: ["Slight snow showers", "🌨"],
  86: ["Heavy snow showers", "❄"],
  95: ["Thunderstorm", "⛈"],
  96: ["Thunderstorm with hail", "⛈"],
  99: ["Severe thunderstorm with hail", "⛈"]
};

const state = {
  location: { ...BENUE_LOCATIONS.find(item => item.name === CONFIG.DEFAULT_LOCATION) },
  weather: null,
  hourlyHistory: [],
  daily: [],
  hourly: [],
  maps: { dashboard: null, full: null },
  markers: { dashboard: [], full: [] },
  backendOnline: false,
  lgas: BENUE_LOCATIONS.map(item => item.name),
  metrics: null,
  reference: null,
  lastPrediction: null,
  lastBatchId: null,
  lastBatchSeries: null,
  lastBatchDownloadUrl: null,
  toastTimer: null,
  weatherRequestId: 0,
  weatherInterval: null
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value == null ? "" : String(value);
}

function setHTML(id, value) {
  const element = document.getElementById(id);
  if (element) element.innerHTML = value;
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatNumber(value, digits = 1) {
  const number = numberOrNull(value);
  return number === null ? "—" : number.toFixed(digits);
}

function formatTemperature(value) {
  const number = numberOrNull(value);
  return number === null ? "--°C" : `${Math.round(number)}°C`;
}

function apiURL(path) {
  return `${CONFIG.API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

async function requestJSON(url, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(
    () => controller.abort(),
    CONFIG.REQUEST_TIMEOUT_MS
  );

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...(options.headers || {})
      }
    });

    const text = await response.text();
    let data = null;

    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { message: text };
      }
    }

    if (!response.ok) {
      const message =
        data?.detail ||
        data?.error ||
        data?.message ||
        `Request failed (${response.status})`;
      throw new Error(message);
    }

    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("The request timed out. Please try again.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function showToast(message, type = "info") {
  const toast = $("#toast");
  if (!toast) return;

  const icons = { info: "ℹ", success: "✓", error: "!" };
  setText("toastIcon", icons[type] || icons.info);
  setText("toastMessage", message);
  toast.classList.add("show");

  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => toast.classList.remove("show"), 4200);
}

function setButtonBusy(button, busy, busyLabel = "Working…") {
  if (!button) return;

  if (busy) {
    if (!button.dataset.originalContent) {
      button.dataset.originalContent = button.innerHTML;
    }
    button.disabled = true;
    button.innerHTML = `<span class="loading-spinner" aria-hidden="true"></span> ${escapeHTML(busyLabel)}`;
  } else {
    button.disabled = false;
    if (button.dataset.originalContent) {
      button.innerHTML = button.dataset.originalContent;
      delete button.dataset.originalContent;
    }
  }
}

function setBackendStatus(online, detail = "") {
  state.backendOnline = online;
  const dot = $("#backendStatusDot");

  if (dot) {
    dot.classList.toggle("online", online);
    dot.classList.toggle("offline", !online);
  }

  setText("backendStatusText", online ? "Backend connected" : "Backend unavailable");
  setText("backendStatusDetail", detail || (online ? "API responding" : "Check API server"));
  setText("performanceModelStatus", online ? "Online" : "Offline");
}

function localDateString(date = new Date()) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: CONFIG.TIMEZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).formatToParts(date);

  const part = type => parts.find(item => item.type === type)?.value;
  return `${part("year")}-${part("month")}-${part("day")}`;
}

function localHourString(date = new Date()) {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: CONFIG.TIMEZONE,
    hour: "2-digit",
    hourCycle: "h23"
  }).format(date);
}

function initDateTimeFields() {
  const today = localDateString();
  const hour = localHourString();

  ["predictionDate", "performanceDate"].forEach(id => {
    const input = document.getElementById(id);
    if (input) input.value = today;
  });

  ["predictionHour", "performanceHour"].forEach(id => {
    const select = document.getElementById(id);
    if (!select) return;

    select.innerHTML = Array.from({ length: 24 }, (_, index) => {
      const value = String(index).padStart(2, "0");
      return `<option value="${value}">${value}:00</option>`;
    }).join("");

    select.value = hour;
  });

  setText("currentDateLabel", new Intl.DateTimeFormat("en-NG", {
    timeZone: CONFIG.TIMEZONE,
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric"
  }).format(new Date()));
}

function weatherDescription(code) {
  const entry = WEATHER_CODES[Number(code)];
  return entry ? entry[0] : "Weather conditions unavailable";
}

function weatherIcon(code) {
  const entry = WEATHER_CODES[Number(code)];
  return entry ? entry[1] : "🌤";
}

function formatLocalHour(isoString) {
  if (!isoString) return "--";

  const match = String(isoString).match(/T(\d{2}):(\d{2})/);
  if (!match) return String(isoString);

  const hour = Number(match[1]);
  const minute = match[2];
  const suffix = hour >= 12 ? "PM" : "AM";
  const hour12 = hour % 12 || 12;

  return `${hour12}:${minute}${suffix}`;
}

function formatLocalDay(isoString) {
  if (!isoString) return "Unknown day";

  const match = String(isoString).match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!match) return String(isoString);

  const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  return new Intl.DateTimeFormat("en-NG", {
    weekday: "long",
    timeZone: CONFIG.TIMEZONE
  }).format(date);
}

function getPreviousHourlyRecord(hourly, currentTime) {
  if (!hourly?.time?.length) return null;

  const currentTimestamp = Date.parse(String(currentTime || "").replace(" ", "T"));
  let bestIndex = -1;
  let bestTimestamp = -Infinity;

  hourly.time.forEach((time, index) => {
    const timestamp = Date.parse(String(time).replace(" ", "T"));
    if (!Number.isFinite(timestamp)) return;

    if (Number.isFinite(currentTimestamp) && timestamp < currentTimestamp && timestamp > bestTimestamp) {
      bestTimestamp = timestamp;
      bestIndex = index;
    }
  });

  if (bestIndex < 0) {
    bestIndex = Math.max(0, hourly.time.length - 2);
  }

  return {
    time: hourly.time[bestIndex],
    temperature: hourly.temperature_2m?.[bestIndex] ?? null,
    humidity: hourly.relative_humidity_2m?.[bestIndex] ?? null,
    windSpeed: hourly.wind_speed_10m?.[bestIndex] ?? null,
    precipitationProbability: hourly.precipitation_probability?.[bestIndex] ?? null,
    precipitation: hourly.precipitation?.[bestIndex] ?? null
  };
}

function normalizeWeatherResponse(data, location) {
  if (!data?.current) {
    throw new Error("The weather provider did not return current weather data.");
  }

  const current = data.current;
  const hourly = data.hourly || {};
  const daily = data.daily || {};

  const todayIndex = Math.max(0, (daily.time || []).findIndex(date => date === String(current.time || "").slice(0, 10)));
  const selectedDailyIndex = todayIndex >= 0 ? todayIndex : 0;

  return {
    location,
    latitude: data.latitude,
    longitude: data.longitude,
    timezone: data.timezone || CONFIG.TIMEZONE,
    current: {
      time: current.time,
      temperature: current.temperature_2m,
      feelsLike: current.apparent_temperature,
      humidity: current.relative_humidity_2m,
      windSpeed: current.wind_speed_10m,
      windDirection: current.wind_direction_10m,
      weatherCode: current.weather_code,
      precipitation: current.precipitation,
      isDay: current.is_day
    },
    hourly,
    daily,
    previous: getPreviousHourlyRecord(hourly, current.time),
    today: {
      low: daily.temperature_2m_min?.[selectedDailyIndex] ?? null,
      high: daily.temperature_2m_max?.[selectedDailyIndex] ?? null
    }
  };
}

async function fetchOpenMeteoWeather(location) {
  const params = new URLSearchParams({
    latitude: String(location.lat),
    longitude: String(location.lon),
    timezone: CONFIG.TIMEZONE,
    forecast_days: "7",
    past_days: "1",
    current: [
      "temperature_2m",
      "relative_humidity_2m",
      "apparent_temperature",
      "is_day",
      "precipitation",
      "weather_code",
      "wind_speed_10m",
      "wind_direction_10m"
    ].join(","),
    hourly: [
      "temperature_2m",
      "relative_humidity_2m",
      "apparent_temperature",
      "precipitation_probability",
      "precipitation",
      "weather_code",
      "wind_speed_10m",
      "wind_direction_10m"
    ].join(","),
    daily: [
      "weather_code",
      "temperature_2m_max",
      "temperature_2m_min",
      "precipitation_probability_max",
      "precipitation_sum",
      "wind_speed_10m_max"
    ].join(","),
    wind_speed_unit: "kmh"
  });

  const data = await requestJSON(`${CONFIG.WEATHER_URL}?${params}`, {
    headers: { Accept: "application/json" }
  });

  return normalizeWeatherResponse(data, location);
}

async function loadWeather(location = state.location) {
  const requestId = ++state.weatherRequestId;
  state.location = { ...location };

  try {
    const weather = await fetchOpenMeteoWeather(location);
    if (requestId !== state.weatherRequestId) return;

    state.weather = weather;
    state.hourlyHistory = weather.previous ? [weather.previous] : [];
    state.hourly = weather.hourly;
    state.daily = weather.daily;

    renderWeatherDashboard();
    renderHourlyForecast($("#hourlyForecast"), 6);
    renderHourlyForecast($("#fullHourlyForecast"), 24);
    renderDailyForecast();
    updateLocationWeatherCards();
    renderLocationSummary();
    updateMapMarkers();

    if (location.name) {
      setText("currentLocationName", `${location.name}, Benue State`);
      const search = $("#locationSearch");
      if (search) search.value = "";
    }
  } catch (error) {
    if (requestId !== state.weatherRequestId) return;
    console.error(" weather error:", error);
    setText("weatherCondition", "Live weather unavailable");
    setText("lastUpdated", error.message);
    setHTML("hourlyForecast", `<div class="error-state">Could not load live weather. Check your internet connection and try refreshing.</div>`);
    setHTML("fullHourlyForecast", `<div class="error-state">Could not load the hourly forecast.</div>`);
    setHTML("dailyForecast", `<div class="error-state">Could not load the daily forecast.</div>`);
    showToast("Unable to load Open-Meteo weather data.", "error");
  }
}

function renderWeatherDashboard() {
  const weather = state.weather;
  if (!weather) return;

  const current = weather.current;
  const currentCode = current.weatherCode;
  const now = current.time
    ? `${current.time.replace("T", " ")} (${weather.timezone})`
    : "Time unavailable";

  setText("currentTemperature", numberOrNull(current.temperature) === null ? "--" : Math.round(current.temperature));
  setText("weatherCondition", weatherDescription(currentCode));
  setText("feelsLike", formatTemperature(current.feelsLike));
  setText("lowTemperature", formatTemperature(weather.today.low));
  setText("highTemperature", formatTemperature(weather.today.high));
  setText("lastUpdated", `Weather time: ${now}`);
  setText("metricHumidity", current.humidity == null ? "--%" : `${Math.round(current.humidity)}%`);
  setText("metricWind", current.windSpeed == null ? "-- km/h" : `${formatNumber(current.windSpeed)} km/h`);
  setText("metricWindDirection", `Direction: ${current.windDirection == null ? "--" : `${Math.round(current.windDirection)}°`}`);
  setText("metricRainChance", getCurrentRainProbability() == null ? "--%" : `${Math.round(getCurrentRainProbability())}%`);
  setText("metricFeelsLike", formatTemperature(current.feelsLike));

  const prediction = state.lastPrediction;
  const confidence = getConfidenceValue(prediction);
  setText("modelConfidence", confidence === null ? "--" : `${Math.round(confidence)}%`);
  const confidenceBar = $("#confidenceBar");
  if (confidenceBar) confidenceBar.style.width = `${confidence === null ? 0 : Math.min(100, Math.max(0, confidence))}%`;

  const insightTitle = `${weatherDescription(currentCode)} in ${weather.location.name}`;
  setText("weatherInsightTitle", insightTitle);

  const pieces = [];
  if (current.temperature != null) pieces.push(`The current temperature is ${formatNumber(current.temperature)}°C`);
  if (current.humidity != null) pieces.push(`relative humidity is ${Math.round(current.humidity)}%`);
  if (current.windSpeed != null) pieces.push(`and winds are around ${formatNumber(current.windSpeed)} km/h`);
  setText("weatherInsight", `${pieces.join(", ")}. Forecasts are provided by Open-Meteo. Model output is available separately through Quick Prediction.`);
}

function getCurrentRainProbability() {
  const hourly = state.weather?.hourly;
  if (!hourly?.time?.length) return null;

  const now = state.weather?.current?.time;
  let index = now ? hourly.time.indexOf(now) : -1;
  if (index < 0) index = Math.max(0, hourly.time.length - 1);

  return hourly.precipitation_probability?.[index] ?? null;
}

function getConfidenceValue(result) {
  if (!result || typeof result !== "object") return null;

  const candidates = [
    result.confidence,
    result.model_confidence,
    result.confidence_score,
    result.metrics?.confidence
  ];

  for (const value of candidates) {
    const parsed = numberOrNull(value);
    if (parsed !== null) {
      return parsed >= 0 && parsed <= 1 ? parsed * 100 : parsed;
    }
  }

  return null;
}

function renderHourlyForecast(container, limit = 6) {
  if (!container) return;

  const hourly = state.weather?.hourly;
  if (!hourly?.time?.length) {
    container.innerHTML = `<div class="empty-state">Hourly weather is unavailable.</div>`;
    return;
  }

  const currentTime = state.weather.current.time;
  let start = currentTime ? hourly.time.indexOf(currentTime) : -1;

  if (start < 0 && currentTime) {
    start = hourly.time.findIndex(time => time >= currentTime);
  }
  if (start < 0) start = 0;

  const count = Math.min(limit, hourly.time.length - start);
  const cards = [];

  for (let offset = 0; offset < count; offset++) {
    const i = start + offset;
    const time = hourly.time[i];
    const temperature = hourly.temperature_2m?.[i];
    const code = hourly.weather_code?.[i];
    const rain = hourly.precipitation_probability?.[i];

    cards.push(`
      <article class="hour-card">
        <span class="hour-time">${escapeHTML(formatLocalHour(time))}</span>
        <span class="hour-icon" aria-hidden="true">${weatherIcon(code)}</span>
        <strong>${temperature == null ? "--" : `${Math.round(temperature)}°`}</strong>
        <small>${escapeHTML(weatherDescription(code))}</small>
        <small class="rain-label">☂ ${rain == null ? "--" : `${Math.round(rain)}%`}</small>
      </article>
    `);
  }

  container.innerHTML = cards.join("");
}

function renderDailyForecast() {
  const container = $("#dailyForecast");
  const daily = state.weather?.daily;
  if (!container) return;

  if (!daily?.time?.length) {
    container.innerHTML = `<div class="empty-state">Daily forecast is unavailable.</div>`;
    return;
  }

  container.innerHTML = daily.time.map((date, i) => {
    const code = daily.weather_code?.[i];
    const high = daily.temperature_2m_max?.[i];
    const low = daily.temperature_2m_min?.[i];
    const rain = daily.precipitation_probability_max?.[i];

    return `
      <div class="daily-row">
        <div>
          <div class="daily-day">${escapeHTML(i === 0 ? "Today" : formatLocalDay(date))}</div>
          <div class="daily-date">${escapeHTML(date)}</div>
        </div>
        <div class="daily-weather"><span>${weatherIcon(code)}</span><span>${escapeHTML(weatherDescription(code))}</span></div>
        <div class="daily-rain">☂ ${rain == null ? "--" : `${Math.round(rain)}%`}</div>
        <div class="daily-temperatures">${high == null ? "--" : `${Math.round(high)}°`} <span>${low == null ? "--" : `${Math.round(low)}°`}</span></div>
      </div>
    `;
  }).join("");
}

/* ---------- Map setup ---------- */

function createMap(containerId, mapKey, center, zoom) {
  if (!window.L) {
    console.error("Leaflet did not load.");
    const element = document.getElementById(containerId);
    if (element) element.innerHTML = `<div class="empty-state">Map library unavailable. Check the Leaflet script connection.</div>`;
    return null;
  }

  const element = document.getElementById(containerId);
  if (!element) return null;

  if (state.maps[mapKey]) {
    state.maps[mapKey].invalidateSize();
    return state.maps[mapKey];
  }

  const map = L.map(containerId, {
    scrollWheelZoom: false,
    zoomControl: true
  }).setView([center.lat, center.lon], zoom);

  /*
   * CARTO's light basemap is used instead of the original OSM standard
   * tile URL to avoid the previous tile-layer configuration.
   * It still requires internet access and attribution.
   */
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    subdomains: "abcd",
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
  }).addTo(map);

  state.maps[mapKey] = map;
  addMarkersToMap(map, mapKey);

  window.setTimeout(() => map.invalidateSize(), 150);
  return map;
}

function initMaps() {
  createMap("weatherMap", "dashboard", CONFIG.BENUE_CENTER, CONFIG.MAP_ZOOM);
  if ($("#fullWeatherMap")) {
    createMap("fullWeatherMap", "full", CONFIG.BENUE_CENTER, CONFIG.MAP_ZOOM);
  }
}

function addMarkersToMap(map, mapKey) {
  if (!map || !window.L) return;

  state.markers[mapKey].forEach(marker => marker.remove());
  state.markers[mapKey] = BENUE_LOCATIONS.map(location => {
    const marker = L.marker([location.lat, location.lon]).addTo(map);
    marker.bindPopup(`
      <strong>${escapeHTML(location.name)}</strong>
      <p style="margin:5px 0 0;color:#778198;font-size:12px">Benue State, Nigeria</p>
      <button type="button" class="map-select-button" data-map-location="${escapeHTML(location.name)}">View weather</button>
    `);
    marker.on("click", () => {
      window.setTimeout(() => {
        const buttons = $$(".map-select-button");
        buttons.forEach(button => {
          if (button.dataset.mapLocation === location.name) {
            button.onclick = () => selectLocation(location.name);
          }
        });
      }, 0);
    });
    return marker;
  });
}

function updateMapMarkers() {
  ["dashboard", "full"].forEach(key => {
    const map = state.maps[key];
    if (!map) return;

    state.markers[key].forEach((marker, index) => {
      const location = BENUE_LOCATIONS[index];
      const isSelected = location.name === state.location.name;
      marker.setOpacity(isSelected ? 1 : 0.78);
      if (isSelected) marker.setZIndexOffset(1000);
      else marker.setZIndexOffset(0);
    });
  });
}

function selectLocation(name) {
  const location = BENUE_LOCATIONS.find(item => item.name.toLowerCase() === String(name).toLowerCase());
  if (!location) {
    showToast(`No coordinates configured for ${name}.`, "error");
    return;
  }

  state.location = { ...location };
  updateMapMarkers();
  loadWeather(state.location);
  showSection("dashboard");
  showToast(`Loading weather for ${location.name}…`);
}

function locateUser() {
  if (!navigator.geolocation) {
    showToast("Geolocation is not supported by this browser.", "error");
    return;
  }

  navigator.geolocation.getCurrentPosition(
    position => {
      const location = {
        name: "Your location",
        lat: position.coords.latitude,
        lon: position.coords.longitude
      };

      state.location = location;
      loadWeather(location);

      Object.values(state.maps).forEach(map => {
        if (map) map.setView([location.lat, location.lon], 10);
      });

      setText("currentLocationName", "Your current location");
      showToast("Location detected. Loading weather.", "success");
    },
    error => {
      const message = error.code === 1
        ? "Location access was denied. You can select an LGA instead."
        : "Unable to determine your location.";
      showToast(message, "error");
    },
    { enableHighAccuracy: false, timeout: 10000, maximumAge: 300000 }
  );
}

function renderLocationSummary() {
  const container = $("#mapLocationSummary");
  if (!container || !state.weather) return;

  const current = state.weather.current;
  container.innerHTML = `
    <div><span>Selected location</span><strong>${escapeHTML(state.location.name)}</strong></div>
    <div><span>Current temperature</span><strong>${escapeHTML(formatTemperature(current.temperature))}</strong></div>
    <div><span>Conditions</span><strong>${escapeHTML(weatherDescription(current.weatherCode))}</strong></div>
  `;
}

/* ---------- Location search and cards ---------- */

async function searchLocations(query) {
  const term = String(query || "").trim();
  if (!term) {
    renderSearchResults([]);
    return;
  }

  const localMatches = BENUE_LOCATIONS.filter(item =>
    item.name.toLowerCase().includes(term.toLowerCase())
  );

  renderSearchResults(localMatches);

  if (localMatches.some(item => item.name.toLowerCase() === term.toLowerCase())) return;

  try {
    const params = new URLSearchParams({
      name: term,
      count: "8",
      language: "en",
      format: "json"
    });

    const response = await requestJSON(`${CONFIG.GEOCODING_URL}?${params}`);
    const results = (response.results || []).filter(item =>
      item.country_code === "NG" &&
      (item.admin1 || "").toLowerCase().includes("benue")
    ).map(item => ({
      name: item.name,
      lat: item.latitude,
      lon: item.longitude
    }));

    const combined = [...localMatches];
    results.forEach(item => {
      if (!combined.some(existing => existing.name.toLowerCase() === item.name.toLowerCase())) {
        combined.push(item);
      }
    });

    renderSearchResults(combined);
  } catch (error) {
    console.warn("Geocoding lookup failed:", error);
    if (!localMatches.length) {
      const results = $("#searchResults");
      if (results) {
        results.innerHTML = `<div class="empty-state">No matching local LGA found. Try one of the listed Benue LGAs.</div>`;
        results.classList.add("open");
      }
    }
  }
}

function renderSearchResults(locations) {
  const container = $("#searchResults");
  if (!container) return;

  if (!locations.length) {
    container.classList.remove("open");
    container.innerHTML = "";
    return;
  }

  container.innerHTML = locations.slice(0, 10).map(location => `
    <button type="button" class="search-result"
      data-location-name="${escapeHTML(location.name)}"
      data-lat="${Number(location.lat)}"
      data-lon="${Number(location.lon)}">
      <span>${escapeHTML(location.name)}</span>
      <small>View weather →</small>
    </button>
  `).join("");

  container.classList.add("open");
}

function renderLocations() {
  const container = $("#locationsGrid");
  if (!container) return;

  container.innerHTML = BENUE_LOCATIONS.map(location => `
    <article class="location-card">
      <div class="metric-icon blue">⌖</div>
      <h3>${escapeHTML(location.name)}</h3>
      <p>Benue State, Nigeria</p>
      <div class="location-weather" data-location-temperature="${escapeHTML(location.name)}">--°C</div>
      <p data-location-condition="${escapeHTML(location.name)}">Select to load weather</p>
      <button class="secondary-button" type="button" data-select-location="${escapeHTML(location.name)}">View location</button>
    </article>
  `).join("");
}

async function updateLocationWeatherCards() {
  const currentLocationTemperature = document.querySelector(
    `[data-location-temperature="${CSS.escape(state.location.name)}"]`
  );

  if (currentLocationTemperature && state.weather?.current) {
    currentLocationTemperature.textContent = formatTemperature(state.weather.current.temperature);
    const condition = document.querySelector(
      `[data-location-condition="${CSS.escape(state.location.name)}"]`
    );
    if (condition) condition.textContent = weatherDescription(state.weather.current.weatherCode);
  }
}

/* ---------- Navigation and mobile menu ---------- */

const SECTION_TITLES = {
  dashboard: "Weather Dashboard",
  forecast: "Weather Forecast",
  map: "Weather Map",
  ai: "Quick Prediction",
  performance: "Model Performance",
  locations: "Benue Locations",
  about: "About This Platform"
};

function closeMobileMenu() {
  $("#sidebar")?.classList.remove("open");
  $("#sidebarOverlay")?.classList.remove("open");
  $("#mobileMenu")?.setAttribute("aria-expanded", "false");
}

function openMobileMenu() {
  $("#sidebar")?.classList.add("open");
  $("#sidebarOverlay")?.classList.add("open");
  $("#mobileMenu")?.setAttribute("aria-expanded", "true");
}

function showSection(name) {
  const target = document.getElementById(`${name}Section`);
  if (!target) return;

  $$(".page-section").forEach(section => section.classList.toggle("active", section === target));
  $$(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.section === name));
  setText("pageTitle", SECTION_TITLES[name] || "Benue Weather Intelligence");

  closeMobileMenu();

  if (name === "map") {
    initMaps();
    window.setTimeout(() => {
      state.maps.full?.invalidateSize();
      if (state.maps.full) {
        state.maps.full.setView([state.location.lat, state.location.lon], Math.max(8, CONFIG.MAP_ZOOM));
      }
    }, 180);
  }

  if (name === "dashboard") {
    window.setTimeout(() => state.maps.dashboard?.invalidateSize(), 120);
  }

  if (name === "performance") {
    loadMetrics();
    loadReference();
  }
}

/* ---------- Backend API ---------- */

async function checkBackendHealth() {
  try {
    const data = await requestJSON(apiURL("/api/health"));
    setBackendStatus(true, data?.status || data?.message || "Health check passed");
    return data;
  } catch (error) {
    setBackendStatus(false, error.message);
    console.warn("Backend health check failed:", error);
    return null;
  }
}

function extractLgaNames(data) {
  const list = Array.isArray(data)
    ? data
    : data?.lgas || data?.locations || data?.items || data?.data || [];

  return list.map(item => {
    if (typeof item === "string") return item;
    return item?.name || item?.lga || item?.label || item?.value || null;
  }).filter(Boolean);
}

async function loadLGAs() {
  try {
    const data = await requestJSON(apiURL("/api/lgas"));
    const names = extractLgaNames(data);
    if (!names.length) return;

    state.lgas = [...new Set(names)];
    const fallback = state.lgas.find(name => name.toLowerCase() === "makurdi") || state.lgas[0];

    ["predictionLga", "performanceLga", "seriesLga"].forEach(id => {
      const select = document.getElementById(id);
      if (!select) return;

      const previous = select.value;
      select.innerHTML = state.lgas.map(name =>
        `<option value="${escapeHTML(name)}">${escapeHTML(name)}</option>`
      ).join("");

      select.value = state.lgas.includes(previous) ? previous : fallback;
    });
  } catch (error) {
    console.warn("Could not load LGA list:", error);
    // The bundled Benue LGA list remains available as a frontend fallback.
  }
}

function findMetric(data, aliases) {
  if (!data || typeof data !== "object") return null;

  for (const alias of aliases) {
    if (data[alias] !== undefined && data[alias] !== null) return data[alias];
  }

  for (const value of Object.values(data)) {
    if (value && typeof value === "object") {
      const result = findMetric(value, aliases);
      if (result !== null) return result;
    }
  }

  return null;
}

function displayMetric(value, digits = 3) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(digits) : "—";
  if (typeof value === "string" || typeof value === "boolean") return String(value);
  return "Available";
}

async function loadMetrics() {
  try {
    const data = await requestJSON(apiURL("/api/metrics"));
    state.metrics = data;

    setText("metricMae", displayMetric(findMetric(data, ["mae", "MAE", "mean_absolute_error"])));
    setText("metricRmse", displayMetric(findMetric(data, ["rmse", "RMSE", "root_mean_squared_error"])));
    setText("metricR2", displayMetric(findMetric(data, ["r2", "r2_score", "r_squared", "r2Score", "R2"])));
    setText("performanceModelStatus", state.backendOnline ? "Online" : "Available");

    const confidence = findMetric(data, ["confidence", "model_confidence"]);
    if (confidence !== null) {
      const parsed = numberOrNull(confidence);
      if (parsed !== null) {
        const normalized = parsed >= 0 && parsed <= 1 ? parsed * 100 : parsed;
        setText("modelConfidence", `${Math.round(normalized)}%`);
        const bar = $("#confidenceBar");
        if (bar) bar.style.width = `${Math.min(100, Math.max(0, normalized))}%`;
      }
    }
  } catch (error) {
    console.warn("Metrics endpoint failed:", error);
    setText("metricMae", "—");
    setText("metricRmse", "—");
    setText("metricR2", "—");
  }
}

function renderReference(data) {
  const container = $("#referenceContent");
  if (!container) return;

  const entries = [];

  function walk(value, path = "") {
    if (value === null || value === undefined) return;

    if (typeof value !== "object") {
      entries.push([path || "Value", String(value)]);
      return;
    }

    if (Array.isArray(value)) {
      entries.push([path || "Items", value.map(item =>
        typeof item === "object" ? JSON.stringify(item) : String(item)
      ).join(", ")]);
      return;
    }

    Object.entries(value).forEach(([key, child]) => {
      const nextPath = path ? `${path} / ${key}` : key;
      if (child && typeof child === "object") walk(child, nextPath);
      else entries.push([nextPath, String(child)]);
    });
  }

  walk(data);

  if (!entries.length) {
    container.innerHTML = `<div class="empty-state">No reference data was returned.</div>`;
    return;
  }

  container.innerHTML = entries.slice(0, 35).map(([key, value]) => `
    <div class="reference-row">
      <span>${escapeHTML(key)}</span>
      <strong>${escapeHTML(value)}</strong>
    </div>
  `).join("");
}

async function loadReference() {
  const container = $("#referenceContent");
  try {
    const data = await requestJSON(apiURL("/api/reference"));
    state.reference = data;
    renderReference(data);
  } catch (error) {
    console.warn("Reference endpoint failed:", error);
    if (container) {
      container.innerHTML = `<div class="error-state">Model reference data is unavailable: ${escapeHTML(error.message)}</div>`;
    }
  }
}

/*
 * Centralized prediction payload.
 *
 * Align these property names with your backend's actual /api/predict schema.
 * The names here follow the user-requested fields and use snake_case.
 * The form's rain setting is sent as use_rain: "auto" | "0" | "1".
 */
function buildPredictionPayload(form, mode) {
  const formData = new FormData(form);
  const lga = String(formData.get("lga") || "").trim();
  const temperature = numberOrNull(formData.get("temperature"));
  const humidity = numberOrNull(formData.get("humidity"));
  const windSpeed = numberOrNull(formData.get("wind_speed"));
  const temperaturePrevious = numberOrNull(formData.get("temperature_1hr_ago"));
  const humidityPrevious = numberOrNull(formData.get("humidity_1hr_ago"));
  const date = String(formData.get("date") || "");
  const hour = Number(formData.get("hour"));
  const useRain = String(formData.get("use_rain") || "auto");

  if (!lga) throw new Error("Select an LGA.");
  if (temperature === null) throw new Error("Enter a valid temperature.");
  if (humidity === null || humidity < 0 || humidity > 100) {
    throw new Error("Humidity must be between 0 and 100.");
  }
  if (windSpeed === null || windSpeed < 0) {
    throw new Error("Wind speed must be zero or greater.");
  }
  if (temperaturePrevious === null) {
    throw new Error("Enter the temperature from one hour ago.");
  }
  if (humidityPrevious === null || humidityPrevious < 0 || humidityPrevious > 100) {
    throw new Error("Previous humidity must be between 0 and 100.");
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    throw new Error("Select a valid date.");
  }
  if (!Number.isInteger(hour) || hour < 0 || hour > 23) {
    throw new Error("Select a valid hour.");
  }
  if (!["auto", "0", "1"].includes(useRain)) {
    throw new Error("Invalid precipitation option.");
  }

  const payload = {
    lga,
    temperature,
    humidity,
    wind_speed: windSpeed,
    temperature_1hr_ago: temperaturePrevious,
    humidity_1hr_ago: humidityPrevious,
    date,
    hour,
    use_rain: useRain
  };

  if (useRain === "1") {
    const r1 = numberOrNull(formData.get("rain_last_1h"));
    const r6 = numberOrNull(formData.get("rain_last_6h"));
    const r24 = numberOrNull(formData.get("rain_last_24h"));
    if (r1 === null || r6 === null || r24 === null || r1 < 0) {
      throw new Error("Enter the rainfall (mm) for the last 1 h, 6 h and 24 h.");
    }
    if (!(r1 <= r6 && r6 <= r24)) {
      throw new Error("Rainfall totals must increase: last 1 h \u2264 6 h \u2264 24 h.");
    }
    Object.assign(payload, { rain_last_1h: r1, rain_last_6h: r6, rain_last_24h: r24 });
  }

  return payload;
}

function extractPredictionValue(data) {
  if (data === null || data === undefined) return null;
  if (typeof data !== "object") return data;

  const keys = [
    "prediction",
    "predicted_value",
    "predicted_rainfall",
    "rainfall_prediction",
    "rainfall",
    "prediction_value",
    "result",
    "output",
    "y_pred",
    "forecast"
  ];

  for (const key of keys) {
    if (data[key] !== undefined && data[key] !== null) {
      const value = data[key];

      if (value && typeof value === "object") {
        const nested = extractPredictionValue(value);
        if (nested !== null) return nested;
      } else {
        return value;
      }
    }
  }

  if (data.data && typeof data.data === "object") {
    return extractPredictionValue(data.data);
  }

  return null;
}

function renderPredictionResult(data, payload) {
  const container = $("#predictionOutput");
  const meta = $("#predictionMeta");
  if (!container) return;

  const prediction = extractPredictionValue(data);
  const label = data?.prediction_label || data?.label || data?.class_name || data?.category;
  const confidence = getConfidenceValue(data);

  let predictionDisplay = "Prediction returned";
  if (typeof prediction === "number") {
    predictionDisplay = prediction.toLocaleString(undefined, { maximumFractionDigits: 4 });
  } else if (typeof prediction === "string") {
    predictionDisplay = prediction;
  } else if (prediction && typeof prediction === "object") {
    predictionDisplay = JSON.stringify(prediction);
  } else if (label) {
    predictionDisplay = String(label);
  }

  const isNumeric = typeof prediction === "number" || (
    typeof prediction === "string" && prediction.trim() !== "" && Number.isFinite(Number(prediction))
  );

  container.innerHTML = `
    <div class="prediction-success">
      <div class="result-label">MODEL OUTPUT</div>
      <div class="prediction-value">${escapeHTML(predictionDisplay)}${isNumeric ? " <small>mm/h</small>" : ""}</div>
      <p>${escapeHTML(label && String(label) !== predictionDisplay ? `Classification: ${label}` : "Expected rainfall for the hour (probability × typical amount if it rains).")}</p>
      <div class="result-details">
        <div class="result-detail"><span>Location</span><strong>${escapeHTML(payload.lga)}</strong></div>
        <div class="result-detail"><span>Input time</span><strong>${escapeHTML(payload.date)} · ${String(payload.hour).padStart(2, "0")}:00</strong></div>
        <div class="result-detail"><span>Temperature</span><strong>${formatNumber(payload.temperature)}°C</strong></div>
        <div class="result-detail"><span>Humidity</span><strong>${formatNumber(payload.humidity)}%</strong></div>
        <div class="result-detail"><span>Wind speed</span><strong>${formatNumber(payload.wind_speed)} km/h</strong></div>
        ${data?.probability == null ? "" : `<div class="result-detail"><span>Rain probability</span><strong>${Math.round(data.probability * 100)}%</strong></div>`}
        ${data?.risk ? `<div class="result-detail"><span>Risk level</span><strong>${escapeHTML(data.risk)}</strong></div>` : ""}
        ${data?.amount_if_wet_mm == null ? "" : `<div class="result-detail"><span>If it rains</span><strong>${formatNumber(data.amount_if_wet_mm)} mm</strong></div>`}
        ${data?.model ? `<div class="result-detail"><span>Model</span><strong>${escapeHTML(data.model)}</strong></div>` : ""}
        ${confidence === null ? "" : `<div class="result-detail"><span>Reported confidence</span><strong>${Math.round(confidence)}%</strong></div>`}
      </div>
    </div>
  `;

  if (meta) {
    meta.textContent = `Response received ${new Date().toLocaleTimeString()} · Endpoint: POST /api/predict`;
  }

  state.lastPrediction = data;
  renderWeatherDashboard();
}

async function submitPrediction(form, button, resultTarget) {
  let payload;

  try {
    payload = buildPredictionPayload(form, "single");
  } catch (error) {
    showToast(error.message, "error");
    if (resultTarget) {
      resultTarget.innerHTML = `<div class="inline-result error">${escapeHTML(error.message)}</div>`;
    }
    return;
  }

  setButtonBusy(button, true, "Running model…");

  try {
    const data = await requestJSON(apiURL("/api/predict"), {
      method: "POST",
      body: JSON.stringify(payload)
    });

    if (data && typeof data === "object" && data.error) {
      throw new Error(data.error);
    }

    state.lastPrediction = data;

    if (form.id === "predictionForm") {
      renderPredictionResult(data, payload);
    } else if (resultTarget) {
      const prediction = extractPredictionValue(data);
      resultTarget.className = "inline-result success";
      resultTarget.innerHTML = `
        <strong>Prediction received.</strong>
        <div class="result-detail"><span>Output</span><strong>${escapeHTML(typeof prediction === "object" ? JSON.stringify(prediction) : prediction ?? "See response details")}</strong></div>
      `;
    }

    setBackendStatus(true);
    showToast("Prediction completed successfully.", "success");
    renderWeatherDashboard();
  } catch (error) {
    console.error("Prediction request failed:", error);
    if (form.id === "predictionForm") {
      setHTML("predictionOutput", `<div class="error-state">Prediction failed: ${escapeHTML(error.message)}</div>`);
      setText("predictionMeta", "Check that the backend is running and accepts the submitted JSON fields.");
    } else if (resultTarget) {
      resultTarget.className = "inline-result error";
      resultTarget.textContent = `Evaluation failed: ${error.message}`;
    }
    showToast("Could not complete the prediction.", "error");
  } finally {
    setButtonBusy(button, false);
  }
}

/*
 * Current and previous observations are populated separately.
 * Open-Meteo's hourly series includes a past day in this configuration.
 * If the previous-hour observation is unavailable, the form is left blank
 * rather than inventing a historical value.
 */
function fillFormFromWeather(formId) {
  const form = document.getElementById(formId);
  if (!form || !state.weather) {
    showToast("Live weather is not loaded yet.", "error");
    return;
  }

  const weather = state.weather;
  const current = weather.current;
  const previous = weather.previous;

  const values = {
    temperature: current.temperature,
    humidity: current.humidity,
    wind_speed: current.windSpeed,
    temperature_1hr_ago: previous?.temperature,
    humidity_1hr_ago: previous?.humidity,
    date: localDateString(),
    hour: localHourString()
  };

  Object.entries(values).forEach(([name, value]) => {
    const field = form.elements.namedItem(name);
    if (!field) return;

    if (value === null || value === undefined || value === "") {
      if (name === "temperature_1hr_ago" || name === "humidity_1hr_ago") {
        field.value = "";
      }
      return;
    }

    field.value = typeof value === "number" ? String(Number(value.toFixed(2))) : String(value);
  });

  const lgaSelect = form.elements.namedItem("lga");
  if (lgaSelect && [...lgaSelect.options].some(option => option.value === weather.location.name)) {
    lgaSelect.value = weather.location.name;
  }

  if (!previous?.temperature || previous?.humidity == null) {
    showToast("Current weather filled. Previous-hour values were unavailable; enter them manually.", "info");
  } else {
    showToast("Current and previous-hour weather inputs filled.", "success");
  }
}

/* ---------- Batch upload and time series ---------- */

async function submitBatch(form) {
  const fileInput = $("#batchFile");
  const file = fileInput?.files?.[0];
  if (!file) {
    showToast("Choose a CSV file before uploading.", "error");
    return;
  }

  if (!file.name.toLowerCase().endsWith(".csv") && file.type !== "text/csv") {
    showToast("Please select a CSV dataset.", "error");
    return;
  }

  const button = $("#runBatchButton");
  const result = $("#batchResult");
  const formData = new FormData();

  formData.append("file", file);
  formData.append("use_rain", $("#batchRain")?.value || "auto");

  setButtonBusy(button, true, "Uploading dataset…");
  if (result) result.textContent = "Uploading dataset and running batch evaluation…";

  try {
    const data = await requestJSON(apiURL("/api/batch"), {
      method: "POST",
      body: formData
    });

    const batchId = data?.id ?? data?.batch_id ?? data?.batchId ?? data?.job_id;
    if (batchId === null || batchId === undefined || batchId === "") {
      throw new Error("The batch API response did not include a batch ID.");
    }

    state.lastBatchId = String(batchId);
    state.lastBatchDownloadUrl = apiURL(`/download/${encodeURIComponent(state.lastBatchId)}.csv`);

    if ($("#seriesBatchId")) $("#seriesBatchId").value = state.lastBatchId;
    if ($("#downloadBatchButton")) $("#downloadBatchButton").disabled = false;

    const summary = data?.message || data?.status || "Batch evaluation completed.";
    const downloadUrl = state.lastBatchDownloadUrl;

    if (result) {
      result.innerHTML = `
        <div class="prediction-success">
          <strong>Batch submitted successfully</strong>
          <div class="result-detail"><span>Batch ID</span><strong>${escapeHTML(state.lastBatchId)}</strong></div>
          <p>${escapeHTML(summary)}</p>
          <p><a href="${escapeHTML(downloadUrl)}" download>Download batch CSV ↓</a></p>
        </div>
      `;
    }

    showToast("Batch evaluation submitted successfully.", "success");
    await loadBatchSeries();
  } catch (error) {
    console.error("Batch upload failed:", error);
    if (result) result.innerHTML = `<div class="error-state">Batch evaluation failed: ${escapeHTML(error.message)}</div>`;
    showToast("Batch upload failed.", "error");
  } finally {
    setButtonBusy(button, false);
  }
}

function normalizeSeriesResponse(data) {
  if (Array.isArray(data)) return data;
  if (!data || typeof data !== "object") return [];

  const possible = data.series || data.results || data.data || data.items || data.predictions;
  if (Array.isArray(possible)) return possible;

  // Support APIs returning a column-oriented structure.
  const time = data.time || data.times || data.timestamps || data.date_time;
  const prediction = data.prediction || data.predictions || data.predicted_value || data.y_pred;
  if (Array.isArray(time) && Array.isArray(prediction)) {
    return time.map((timestamp, index) => ({
      time: timestamp,
      prediction: prediction[index],
      actual: Array.isArray(data.actual) ? data.actual[index] : undefined
    }));
  }

  return [];
}

function rowTimestamp(row) {
  return row?.time ?? row?.timestamp ?? row?.datetime ?? row?.date_time ?? row?.date ?? row?.hour ?? "—";
}

function rowPrediction(row) {
  if (row === null || row === undefined) return null;
  if (typeof row !== "object") return row;

  return row.prediction ??
    row.predicted_value ??
    row.predicted_rainfall ??
    row.rainfall_prediction ??
    row.y_pred ??
    row.output ??
    row.result ??
    null;
}

function rowActual(row) {
  if (!row || typeof row !== "object") return null;
  return row.actual ?? row.observed ?? row.target ?? row.actual_value ?? row.y_true ?? null;
}

async function loadBatchSeries() {
  const batchId = String($("#seriesBatchId")?.value || state.lastBatchId || "").trim();
  const lga = String($("#seriesLga")?.value || CONFIG.DEFAULT_LOCATION).trim();

  if (!batchId) {
    showToast("Enter a batch ID first.", "error");
    return;
  }

  const button = $("#loadSeriesButton");
  setButtonBusy(button, true, "Loading series…");

  try {
    const params = new URLSearchParams({ lga });
    const data = await requestJSON(
      apiURL(`/api/batch/${encodeURIComponent(batchId)}/series?${params}`)
    );

    const series = normalizeSeriesResponse(data);
    if (!series.length) {
      throw new Error("The series endpoint returned no recognizable time-series rows.");
    }

    state.lastBatchId = batchId;
    state.lastBatchSeries = series;
    state.lastBatchDownloadUrl = apiURL(`/download/${encodeURIComponent(batchId)}.csv`);

    if ($("#downloadBatchButton")) $("#downloadBatchButton").disabled = false;

    renderBatchSeries(series, { batchId, lga, response: data });
    showToast(`Loaded ${series.length} series rows for ${lga}.`, "success");
  } catch (error) {
    console.error("Loading batch series failed:", error);
    setText("seriesSummary", `Could not load series: ${error.message}`);
    setHTML("seriesTableBody", `<tr><td colspan="10">${escapeHTML(error.message)}</td></tr>`);
    clearSeriesChart();
    showToast("Could not load batch series.", "error");
  } finally {
    setButtonBusy(button, false);
  }
}

function renderBatchSeries(series, details) {
  const head = $("#seriesTableHead");
  const body = $("#seriesTableBody");
  const summary = $("#seriesSummary");

  const keys = [...new Set(series.flatMap(row =>
    row && typeof row === "object" ? Object.keys(row) : []
  ))];

  const timeKey = keys.find(key => ["time", "timestamp", "datetime", "date_time", "date"].includes(key));
  const predictionKey = keys.find(key => ["prediction", "predicted_value", "predicted_rainfall", "rainfall_prediction", "y_pred"].includes(key));
  const actualKey = keys.find(key => ["actual", "observed", "target", "actual_value", "y_true"].includes(key));

  if (head) {
    const columns = timeKey || predictionKey
      ? [timeKey, predictionKey, actualKey].filter(Boolean)
      : keys.slice(0, 8);

    head.innerHTML = `<tr>${columns.map(key => `<th>${escapeHTML(key.replaceAll("_", " "))}</th>`).join("")}</tr>`;
  }

  if (body) {
    body.innerHTML = series.slice(0, 300).map(row => {
      if (row && typeof row === "object") {
        const columns = timeKey || predictionKey
          ? [timeKey, predictionKey, actualKey].filter(Boolean)
          : keys.slice(0, 8);
        return `<tr>${columns.map(key => `<td>${escapeHTML(row[key] ?? "—")}</td>`).join("")}</tr>`;
      }

      return `<tr><td>${escapeHTML(row)}</td></tr>`;
    }).join("");
  }

  if (summary) {
    const metrics = details.response?.metrics || {};
    summary.textContent = `Batch ${details.batchId} · ${details.lga} · ${series.length} records` +
      (Object.keys(metrics).length ? ` · Metrics: ${JSON.stringify(metrics)}` : "");
  }

  const points = series.map(row => ({
    label: String(rowTimestamp(row)),
    prediction: numberOrNull(rowPrediction(row)),
    actual: numberOrNull(rowActual(row))
  }));

  drawSeriesChart(points);
}

function clearSeriesChart() {
  const canvas = $("#seriesChart");
  if (!canvas) return;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function drawSeriesChart(points) {
  const canvas = $("#seriesChart");
  if (!canvas) return;

  const rect = canvas.getBoundingClientRect();
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(320, Math.floor(rect.width));
  const height = 220;

  canvas.width = width * ratio;
  canvas.height = height * ratio;
  canvas.style.height = `${height}px`;

  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const predictions = points.filter(point => point.prediction !== null);
  const actuals = points.filter(point => point.actual !== null);

  if (!predictions.length && !actuals.length) {
    context.fillStyle = "#8792a7";
    context.font = "12px DM Sans, sans-serif";
    context.fillText("No numeric prediction values available to chart.", 12, 30);
    return;
  }

  const allValues = [...predictions.map(point => point.prediction), ...actuals.map(point => point.actual)];
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const range = max - min || 1;

  const pad = { left: 42, right: 16, top: 20, bottom: 36 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  context.strokeStyle = "#e9edf5";
  context.fillStyle = "#8792a7";
  context.lineWidth = 1;
  context.font = "10px DM Sans, sans-serif";

  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotHeight * i) / 4;
    const value = max - (range * i) / 4;

    context.beginPath();
    context.moveTo(pad.left, y);
    context.lineTo(width - pad.right, y);
    context.stroke();
    context.fillText(value.toFixed(1), 3, y + 4);
  }

  function drawLine(key, color) {
    let started = false;
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();

    points.forEach((point, index) => {
      const value = point[key];
      if (value === null) {
        started = false;
        return;
      }

      const x = pad.left + (points.length <= 1 ? plotWidth / 2 : (plotWidth * index) / (points.length - 1));
      const y = pad.top + ((max - value) / range) * plotHeight;

      if (!started) {
        context.moveTo(x, y);
        started = true;
      } else {
        context.lineTo(x, y);
      }
    });

    context.stroke();
  }

  drawLine("prediction", "#6558e8");
  drawLine("actual", "#22b98b");

  context.fillStyle = "#6558e8";
  context.fillRect(pad.left, height - 16, 9, 3);
  context.fillStyle = "#66738b";
  context.fillText("Prediction", pad.left + 14, height - 12);

  if (actuals.length) {
    context.fillStyle = "#22b98b";
    context.fillRect(pad.left + 95, height - 16, 9, 3);
    context.fillStyle = "#66738b";
    context.fillText("Actual", pad.left + 109, height - 12);
  }
}

function downloadBatchCSV() {
  const batchId = String($("#seriesBatchId")?.value || state.lastBatchId || "").trim();
  if (!batchId) {
    showToast("No batch ID is available to download.", "error");
    return;
  }

  const url = apiURL(`/download/${encodeURIComponent(batchId)}.csv`);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${batchId}.csv`;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

/* ---------- Event handlers ---------- */

function bindEvents() {
  $$(".nav-item").forEach(button => {
    button.addEventListener("click", () => showSection(button.dataset.section));
  });

  $$("[data-goto]").forEach(button => {
    button.addEventListener("click", () => showSection(button.dataset.goto));
  });

  $("#mobileMenu")?.addEventListener("click", event => {
    event.stopPropagation();
    if ($("#sidebar")?.classList.contains("open")) closeMobileMenu();
    else openMobileMenu();
  });

  $("#sidebarOverlay")?.addEventListener("click", closeMobileMenu);

  document.addEventListener("click", event => {
    const sidebar = $("#sidebar");
    const menuButton = $("#mobileMenu");

    if (
      sidebar?.classList.contains("open") &&
      !sidebar.contains(event.target) &&
      !menuButton?.contains(event.target)
    ) {
      closeMobileMenu();
    }

    const results = $("#searchResults");
    const searchBar = $(".search-bar");
    if (results?.classList.contains("open") && searchBar && !searchBar.contains(event.target)) {
      results.classList.remove("open");
    }
  });

  document.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      closeMobileMenu();
      $("#searchResults")?.classList.remove("open");
    }
  });

  $("#refreshButton")?.addEventListener("click", async () => {
    await Promise.allSettled([
      loadWeather(state.location),
      checkBackendHealth(),
      loadMetrics()
    ]);
    showToast("Refresh finished.", "success");
  });

  $("#searchButton")?.addEventListener("click", () => searchLocations($("#locationSearch")?.value));
  $("#locationSearch")?.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      event.preventDefault();
      searchLocations(event.currentTarget.value);
    }
  });

  $("#locationSearch")?.addEventListener("input", event => {
    const value = event.currentTarget.value.trim();
    if (!value) $("#searchResults")?.classList.remove("open");
  });

  $("#currentLocationButton")?.addEventListener("click", locateUser);
  $("#mapLocateButton")?.addEventListener("click", locateUser);

  $("#predictionForm")?.addEventListener("submit", event => {
    event.preventDefault();
    submitPrediction(
      event.currentTarget,
      $("#runPredictionButton"),
      $("#predictionMeta")
    );
  });

  $("#performanceForm")?.addEventListener("submit", event => {
    event.preventDefault();
    submitPrediction(
      event.currentTarget,
      $("#runPerformanceButton"),
      $("#performanceResult")
    );
  });

  $("#fillLiveWeatherButton")?.addEventListener("click", () => fillFormFromWeather("predictionForm"));
  $("#fillPerformanceLiveButton")?.addEventListener("click", () => fillFormFromWeather("performanceForm"));

  $("#batchForm")?.addEventListener("submit", event => {
    event.preventDefault();
    submitBatch(event.currentTarget);
  });

  $("#loadSeriesButton")?.addEventListener("click", loadBatchSeries);
  $("#downloadBatchButton")?.addEventListener("click", downloadBatchCSV);

  $("#refreshMetricsButton")?.addEventListener("click", async () => {
    const button = $("#refreshMetricsButton");
    setButtonBusy(button, true, "Refreshing…");
    await Promise.allSettled([checkBackendHealth(), loadMetrics(), loadReference()]);
    setButtonBusy(button, false);
    showToast("Model metrics refreshed.", "success");
  });

  $("#toastClose")?.addEventListener("click", () => $("#toast")?.classList.remove("show"));

  document.addEventListener("click", event => {
    const searchResult = event.target.closest(".search-result");
    if (searchResult) {
      const location = {
        name: searchResult.dataset.locationName,
        lat: Number(searchResult.dataset.lat),
        lon: Number(searchResult.dataset.lon)
      };
      $("#searchResults")?.classList.remove("open");
      selectLocation(location.name);

      // Use geocoding coordinates when a result is not in the built-in list.
      if (!BENUE_LOCATIONS.some(item => item.name.toLowerCase() === location.name.toLowerCase())) {
        state.location = location;
        loadWeather(location);
      }
      return;
    }

    const locationButton = event.target.closest("[data-select-location]");
    if (locationButton) {
      selectLocation(locationButton.dataset.selectLocation);
    }

    const mapButton = event.target.closest(".map-select-button");
    if (mapButton) {
      selectLocation(mapButton.dataset.mapLocation);
    }
  });

  window.addEventListener("resize", () => {
    Object.values(state.maps).forEach(map => map?.invalidateSize());
    if (state.lastBatchSeries) {
      drawSeriesChart(state.lastBatchSeries.map(row => ({
        label: String(rowTimestamp(row)),
        prediction: numberOrNull(rowPrediction(row)),
        actual: numberOrNull(rowActual(row))
      })));
    }
  });
}

/* ---------- Startup ---------- */

function bindRainFields() {
  $$('select[name="use_rain"]').forEach(select => {
    const form = select.closest("form");
    const sync = () => $$(".rain-field", form).forEach(field => { field.hidden = select.value !== "1"; });
    select.addEventListener("change", sync);
    sync();
  });
}

async function initializeApp() {
  initDateTimeFields();
  bindEvents();
  bindRainFields();
  renderLocations();
  initMaps();

  // Weather is useful even when the model API is temporarily offline.
  const weatherPromise = loadWeather(state.location);

  const backendPromises = Promise.allSettled([
    checkBackendHealth(),
    loadLGAs(),
    loadMetrics(),
    loadReference()
  ]);

  await Promise.allSettled([weatherPromise, backendPromises]);

  // Revalidate live conditions periodically without requiring a page reload.
  if (state.weatherInterval) window.clearInterval(state.weatherInterval);
  state.weatherInterval = window.setInterval(() => {
    loadWeather(state.location);
  }, CONFIG.WEATHER_REFRESH_MS);
}

document.addEventListener("DOMContentLoaded", initializeApp);
