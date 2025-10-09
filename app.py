# Adapted from:
# Naeima (2022). Correlation-between-Female-Fertility-and-Employment-Status.
# GitHub repository: https://github.com/Naeima/Correlation-between-Female-Fertility-and-Employment-Status
# dashboard_ev_merged_rcsp_zones_weather.py
# Chargers coloured by FLOOD MODEL ZONE + Routing risk from ALL flood layers + Weather
# Light/Incremental mode: fast startup with no WFS/WMS feature fetch until you opt in.

import io, os, time, json, tempfile, requests, math, heapq, inspect
from io import StringIO, BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass, asdict
from typing import List, Tuple
import numpy as np

import dash
from dash import Dash, dcc, html, Input, Output, State  
from flask import Response
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import split as shp_split

import folium
from folium.plugins import MarkerCluster, Draw, BeautifyIcon
from folium.raster_layers import WmsTileLayer

# Optional graph libs for exact optimiser
try:
    import osmnx as ox
    import networkx as nx
    HAS_OSMNX = True
except Exception:
    HAS_OSMNX = False
# ---- Geocoding helpers: UK postcodes and general addresses ----
def geocode_postcode_uk(pc: str):
    """Fast UK postcode lookup (Postcodes.io). Returns (lat, lon) or None."""
    try:
        pc = (pc or "").strip()
        if not pc:
            return None
        url = f"https://api.postcodes.io/postcodes/{requests.utils.quote(pc)}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            j = r.json()
            res = (j.get("result") or {})
            lat = res.get("latitude"); lon = res.get("longitude")
            if lat is not None and lon is not None:
                return float(lat), float(lon)
    except Exception:
        pass
    return None


def geocode_text_osm(q: str):
    """General forward geocoder (Nominatim). Returns (lat, lon) or None."""
    try:
        q = (q or "").strip()
        if not q:
            return None
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": q, "format": "jsonv2", "limit": 1, "addressdetails": 0}
        r = requests.get(url, params=params, headers={"User-Agent": "CLEETS-EV/1.0"}, timeout=15)
        if r.status_code == 200:
            arr = r.json()
            if arr:
                return float(arr[0]["lat"]), float(arr[0]["lon"])
    except Exception:
        pass
    return None


def geocode_start_end(start_pc_or_text: str, end_pc_or_text: str):
    """Try UK postcode first; fall back to OSM text search."""
    s = geocode_postcode_uk(start_pc_or_text) or geocode_text_osm(start_pc_or_text)
    e = geocode_postcode_uk(end_pc_or_text) or geocode_text_osm(end_pc_or_text)
    return s, e

# =========================
# Config
# =========================
EV_GDRIVE_FILE_ID = "1RFtC5hSEIrg5yG1rkmfD8JasAK6h212K"  # https://drive.google.com/file/d/1RFtC5hSEIrg5yG1rkmfD8JasAK6h212K/view?usp=sharing
LOCAL_EV_CSV = "south_wales_ev.csv"
CACHE_DIR = ".cache_wfs"; os.makedirs(CACHE_DIR, exist_ok=True)

# Logo from Google Drive (provided by you)
LOGO_GDRIVE_FILE_OR_URL = "https://drive.google.com/file/d/1QLQPln4dRyWXh65E5ua_rC3CGTChwKxc/view?usp=sharing"
LOGO_CACHE_PATH = os.path.join(CACHE_DIR, "cleets_logo-01.png")

OWS_BASE = "https://datamap.gov.wales/geoserver/ows"

FRAW_WMS = {
    "FRAW – Rivers":  "inspire-nrw:NRW_FLOOD_RISK_FROM_RIVERS",
    "FRAW – Sea":     "inspire-nrw:NRW_FLOOD_RISK_FROM_SEA",
    "FRAW – Surface": "inspire-nrw:NRW_FLOOD_RISK_FROM_SURFACE_WATER_SMALL_WATERCOURSES",
}
FMFP_WMS = {
    "FMfP – Rivers & Sea": "inspire-nrw:NRW_FLOODZONE_RIVERS_SEAS_MERGED",
    "FMfP – Surface/Small Watercourses": "inspire-nrw:NRW_FLOODZONE_SURFACE_WATER_AND_SMALL_WATERCOURSES",
}
LIVE_WMS = {
    "Live – Warning Areas": "inspire-nrw:NRW_FLOOD_WARNING",
    "Live – Alert Areas":   "inspire-nrw:NRW_FLOOD_WATCH_AREAS",
}
CONTEXT_WMS = {"Historic Flood Extents": "inspire-nrw:NRW_HISTORIC_FLOODMAP"}

FRAW_WFS = {
    "FRAW Rivers":  "inspire-nrw:NRW_FLOOD_RISK_FROM_RIVERS",
    "FRAW Sea":     "inspire-nrw:NRW_FLOOD_RISK_FROM_SEA",
    "FRAW Surface": "inspire-nrw:NRW_FLOOD_RISK_FROM_SURFACE_WATER_SMALL_WATERCOURSES",
}
FMFP_WFS = {
    "FMfP Rivers & Sea": "inspire-nrw:NRW_FLOODZONE_RIVERS_SEAS_MERGED",
    "FMfP Surface/Small": "inspire-nrw:NRW_FLOODZONE_SURFACE_WATER_AND_SMALL_WATERCOURSES",
}
LIVE_WFS = {
    "Warnings": "inspire-nrw:NRW_FLOOD_WARNING",
    "Alerts":   "inspire-nrw:NRW_FLOOD_WATCH_AREAS",
}

SIM_DEFAULTS = dict(
    start_lat=51.4816, start_lon=-3.1791,  # Cardiff
    end_lat=51.6214,   end_lon=-3.9436,    # Swansea
    battery_kwh=64.0, init_soc=90.0, reserve_soc=10.0, target_soc=80.0,
    kwh_per_km=0.18, max_charger_offset_km=1.5, min_leg_km=20.0,
    route_buffer_m=30, wfs_pad_m=800
)

# RCSP knobs
SOC_STEP = 0.05
CHARGE_STEP = 0.10
DEFAULT_POWER_KW = 50.0
BASE_RISK_PENALTY_PER_KM = 60.0   # sec/km
EXTREME_RISK_PENALTY_PER_KM = 240.0
EXTREME_BUFFER_M = 60.0
MAX_GRAPH_BBOX_DEG = 1.0
ROUTE_BUFFER_M = 30                     # used when segmenting by flood risk

ZONE_COLORS = {"Outside": "#2E7D32", "Zone 3":"#D32F2F", "Zone 2":"#FFC107", "Zone 1":"#2E7D32"}

@dataclass
class EVParams:
    battery_kWh: float = 75.0
    start_soc: float = 0.80
    reserve_soc: float = 0.10
    target_soc: float = 0.80
    kWh_per_km: float = 0.20
    max_charge_kW: float = 120.0

@dataclass
class StopInfo:
    lat: float
    lon: float
    name: str = "Charger"
    postcode: str = ""
    ZoneLabel: str = "Outside"
    ZoneColor: str = ZONE_COLORS["Outside"]
    Operational: bool = True
    soc_before: float = 0.0
    soc_after: float = 0.0
    energy_kWh: float = 0.0
    charge_time_min: float = 0.0

# -------------------------
# Vehicle presets (category → model → specs)
# kWh/km are typical mixed-cycle estimates; tweak freely.
# -------------------------
VEHICLE_PRESETS = {
    "Hatchback": {
        "Nissan Leaf 40":   {"battery_kWh": 40.0, "kWh_per_km": 0.16, "max_charge_kW": 50.0},
        "Renault Zoe R135": {"battery_kWh": 52.0, "kWh_per_km": 0.15, "max_charge_kW": 50.0},
        "VW ID.3 Pro":      {"battery_kWh": 58.0, "kWh_per_km": 0.16, "max_charge_kW": 120.0},
    },
    "Sedan": {
        "Tesla Model 3 RWD":   {"battery_kWh": 57.5, "kWh_per_km": 0.145, "max_charge_kW": 170.0},
        "Hyundai Ioniq 6":     {"battery_kWh": 77.4, "kWh_per_km": 0.14,  "max_charge_kW": 220.0},
        "Polestar 2 Long":     {"battery_kWh": 82.0, "kWh_per_km": 0.17,  "max_charge_kW": 155.0},
    },
    "SUV / Crossover": {
        "Kia EV6 RWD":         {"battery_kWh": 77.4, "kWh_per_km": 0.18, "max_charge_kW": 230.0},
        "Hyundai Kona 64":     {"battery_kWh": 64.0, "kWh_per_km": 0.155,"max_charge_kW": 75.0},
        "VW ID.4 Pro":         {"battery_kWh": 77.0, "kWh_per_km": 0.19, "max_charge_kW": 125.0},
    },
    "Van / MPV": {
        "VW ID. Buzz":         {"battery_kWh": 77.0, "kWh_per_km": 0.23, "max_charge_kW": 170.0},
        "Peugeot e-Traveller": {"battery_kWh": 75.0, "kWh_per_km": 0.26, "max_charge_kW": 100.0},
        "Renault Kangoo E-Tech":{"battery_kWh": 45.0, "kWh_per_km": 0.20, "max_charge_kW": 80.0},
    },
}


# =========================
# Utilities
# =========================
def _gd_url(x):
    if "drive.google.com" in x:
        if "/file/d/" in x:
            fid = x.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={fid}"
        return x
    return f"https://drive.google.com/uc?export=download&id={x}"

def _requests_session():
    sess = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5,
                  status_forcelist=(429,500,502,503,504),
                  allowed_methods=frozenset(["GET","HEAD"]))
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.headers.update({"User-Agent":"Mozilla/5.0 (EV-Dashboard)"})
    return sess

def read_csv_resilient_gdrive(file_id_or_url: str, **kw):
    url = _gd_url(file_id_or_url)
    sess = _requests_session()
    try:
        r = sess.get(url, timeout=30, stream=True); r.raise_for_status()
        token = next((v for k,v in r.cookies.items() if k.startswith("download_warning")), None)
        if token:
            r = sess.get(url, params={"confirm":token}, timeout=30, stream=True); r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content), **({"low_memory":False}|kw))
    except Exception as e:
        try:
            import gdown
            with tempfile.TemporaryDirectory() as td:
                out = os.path.join(td,"data.csv")
                gdown.download(url, out, quiet=True, fuzzy=True)
                return pd.read_csv(out, **({"low_memory":False}|kw))
        except Exception:
            pass
        if "export=download" not in url:
            r = requests.get(url.replace("/uc?","/uc?export=download&"), timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.BytesIO(r.content), **({"low_memory":False}|kw))
        raise RuntimeError(f"Google Drive fetch failed: {e}")

def read_bytes_resilient_gdrive(file_id_or_url: str) -> bytes:
    url = _gd_url(file_id_or_url)
    sess = _requests_session()
    r = sess.get(url, timeout=30, stream=True); r.raise_for_status()
    token = next((v for k,v in r.cookies.items() if k.startswith("download_warning")), None)
    if token:
        r = sess.get(url, params={"confirm":token}, timeout=30, stream=True); r.raise_for_status()
    return r.content

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

def get_postcode(lat, lon):
    """Reverse geocode using Nominatim (polite: ~1 req/sec)."""  # :contentReference[oaicite:4]{index=4}
    try:
        url = (f"https://nominatim.openstreetmap.org/reverse"
               f"?format=jsonv2&lat={lat}&lon={lon}&zoom=18&addressdetails=1")
        resp = requests.get(url, headers={"User-Agent": "ons-evapp/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            return data.get("address", {}).get("postcode", "")
        return ""
    except Exception:
        return ""
    
def bbox_expand(bounds, pad_m):
    minx,miny,maxx,maxy = bounds
    pad_deg = max(0.002, pad_m/111_320.0)
    return (minx-pad_deg, miny-pad_deg, maxx+pad_deg, maxy+pad_deg)

def _bbox_for(df_like, pad_m=SIM_DEFAULTS["wfs_pad_m"]):
    if isinstance(df_like, gpd.GeoDataFrame) and not df_like.empty:
        minx,miny,maxx,maxy = df_like.total_bounds
    else:
        minx,miny,maxx,maxy = gdf_ev.total_bounds
    return bbox_expand((minx,miny,maxx,maxy), pad_m)

def _cache_path(layer,bbox):
    safe = f"{layer}_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}".replace(":","_").replace(",","_")
    return os.path.join(CACHE_DIR, f"{safe}.geojson")

def fetch_wfs_layer_cached(layer, bbox, ttl_h=48):
    p = _cache_path(layer, bbox)
    if os.path.exists(p) and time.time()-os.path.getmtime(p) < ttl_h*3600:
        try:
            gj = json.load(open(p,"r",encoding="utf-8"))
            return gpd.GeoDataFrame.from_features(gj.get("features",[]), crs="EPSG:4326")
        except Exception:
            pass
    from urllib.parse import urlencode
    params = {
        "service":"WFS","request":"GetFeature","version":"2.0.0",
        "typenames":layer,"outputFormat":"application/json","srsName":"EPSG:4326",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326"
    }
    url = f"{OWS_BASE}?{urlencode(params)}"
    try:
        gj = requests.get(url, timeout=30).json()
        with open(p,"w",encoding="utf-8") as f: json.dump(gj,f)
        return gpd.GeoDataFrame.from_features(gj.get("features",[]), crs="EPSG:4326")
    except Exception:
        if os.path.exists(p):
            try:
                gj = json.load(open(p,"r",encoding="utf-8"))
                return gpd.GeoDataFrame.from_features(gj.get("features",[]), crs="EPSG:4326")
            except Exception:
                pass
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

# =========================
# Flood model zones → label/colour
# =========================
ZONE_COLORS = {
    "Zone 3": "#D32F2F", "High": "#D32F2F",
    "Zone 2": "#FFC107", "Medium": "#FFC107",
    "Zone 1": "#2E7D32", "Low": "#2E7D32", "Very Low": "#2E7D32",
    "Outside": "#2E7D32", "Unknown": "#2E7D32"
}
ZONE_PRIORITY = ["Zone 3", "High", "Zone 2", "Medium", "Zone 1", "Low", "Very Low", "Outside", "Unknown"]
_PRI = {z:i for i,z in enumerate(ZONE_PRIORITY)}

def make_beautify_icon(color_hex: str):
    border_map = {"#D32F2F":"#B71C1C", "#FFC107":"#FF8F00", "#2E7D32":"#1B5E20"}
    border = border_map.get(color_hex, "#1B5E20")
    return BeautifyIcon(
        icon="bolt", icon_shape="marker",
        background_color=color_hex, border_color=border, border_width=3,
        text_color="white", inner_icon_style="font-size:22px;padding-top:2px;"
    )

def zone_to_icon(z: str) -> str:
    z = (z or "").strip()
    if z in ("Zone 3", "High"):   return "red"
    if z in ("Zone 2", "Medium"): return "orange"
    return "green"

def _norm_zone(props: dict, layer_name: str) -> str:
    txt = " ".join([str(v) for v in props.values() if v is not None]).lower()
    if "zone 3" in txt: return "Zone 3"
    if "zone 2" in txt: return "Zone 2"
    if "zone 1" in txt: return "Zone 1"
    if "very low" in txt: return "Very Low"
    if "high" in txt:     return "High"
    if "medium" in txt:   return "Medium"
    if "low" in txt:      return "Low"
    return "Unknown"

def fetch_model_zones_gdf(ev_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    bbox = _bbox_for(ev_gdf, pad_m=SIM_DEFAULTS.get("wfs_pad_m", 800))
    chunks = []
    for title, layer in {**FMFP_WFS, **FRAW_WFS}.items():
        g = fetch_wfs_layer_cached(layer, bbox)
        if g.empty: continue
        props_df = g.drop(columns=["geometry"], errors="ignore")
        zlabs = [_norm_zone(r.to_dict(), title) for _, r in props_df.iterrows()]
        g = g.assign(zone=zlabs, color=[ZONE_COLORS.get(z,"#2E7D32") for z in zlabs], model=title)
        try: g["geometry"] = g["geometry"].buffer(0)
        except Exception: pass
        try: g = g.explode(index_parts=False).reset_index(drop=True)
        except Exception: pass
        chunks.append(g[["zone","color","model","geometry"]])
    if not chunks:
        return gpd.GeoDataFrame(columns=["zone","color","model","geometry"], geometry="geometry", crs="EPSG:4326")
    G = pd.concat(chunks, ignore_index=True)
    return gpd.GeoDataFrame(G, geometry="geometry", crs="EPSG:4326")

def compute_model_zones_for_points(ev_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    zones = fetch_model_zones_gdf(ev_gdf)
    out = ev_gdf[["ROW_ID"]].copy()
    out["ZoneLabel"] = "Outside"; out["ZoneColor"] = ZONE_COLORS["Outside"]
    if zones.empty or ev_gdf.empty:
        return out[["ROW_ID","ZoneLabel","ZoneColor"]]
    try:
        ev_m = ev_gdf.to_crs("EPSG:27700"); zn_m = zones.to_crs("EPSG:27700")
    except Exception:
        ev_m = ev_gdf.to_crs("EPSG:3857");  zn_m = zones.to_crs("EPSG:3857")
    try:
        joined = gpd.sjoin(ev_m[["ROW_ID","geometry"]], zn_m, how="left", predicate="within")
    except Exception:
        joined = gpd.sjoin(ev_m[["ROW_ID","geometry"]], zn_m, how="left", predicate="intersects")
    if joined.empty:
        return out[["ROW_ID","ZoneLabel","ZoneColor"]]
    joined["pri"] = joined["zone"].map(_PRI).fillna(_PRI["Unknown"])
    idx = joined.sort_values(["ROW_ID","pri"]).groupby("ROW_ID", as_index=False).first()
    lut = idx.set_index("ROW_ID")
    out.loc[out["ROW_ID"].isin(lut.index), "ZoneLabel"] = lut["zone"]
    out.loc[out["ROW_ID"].isin(lut.index), "ZoneColor"] = lut["zone"].map(ZONE_COLORS).fillna("#2E7D32")
    return out[["ROW_ID","ZoneLabel","ZoneColor"]]

# =========================
# Load EV data
# =========================
if os.path.exists(LOCAL_EV_CSV):
    df = pd.read_csv(LOCAL_EV_CSV, low_memory=False)
else:
    df = read_csv_resilient_gdrive(EV_GDRIVE_FILE_ID)

TARGET_AREAS = [
    "Blaenau Gwent","Bridgend","Caerphilly","Cardiff","Carmarthenshire","Merthyr Tydfil",
    "Monmouthshire","Neath Port Talbot","Newport","Pembrokeshire","Rhondda Cynon Taf",
    "Swansea","The Vale Of Glamorgan","Torfaen"
]
area_col = 'country' if 'country' in df.columns else ('adminArea' if 'adminArea' in df.columns else 'town')
df[area_col] = df[area_col].astype(str).str.strip().str.title()

df['Latitude']  = pd.to_numeric(df.get('latitude', df.get('Latitude')), errors='coerce')
df['Longitude'] = pd.to_numeric(df.get('longitude', df.get('Longitude')), errors='coerce')
df = df.dropna(subset=['Latitude','Longitude'])
df['country'] = df[area_col]

def classify_availability(s):
    s = str(s).lower().strip()
    if any(k in s for k in ["available","in service","operational","working","ok","service"]): return True
    if any(k in s for k in ["not operational","fault","out of service","offline","unavailable","down"]): return False
    return None

_df_status = df.get('chargeDeviceStatus', pd.Series(index=df.index))
df['Available'] = _df_status.apply(classify_availability)
df['AvailabilityLabel'] = df['Available'].map({True:"Operational", False:"Not operational"}).fillna("Unknown")
df['Operator'] = df.get('deviceControllerName', df.get('Operator','Unknown'))
df['Postcode'] = df.get('postcode', df.get('Postcode','N/A'))
df['dateCreated'] = pd.to_datetime(df.get('dateCreated', df.get('DateCreated')), errors='coerce', dayfirst=True)

df['geometry'] = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf_ev = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
gdf_ev['ROW_ID'] = gdf_ev.index.astype(int)
country_OPTIONS = sorted([t for t in gdf_ev['country'].dropna().astype(str).unique() if t])

# =========================
# Routing helpers
# =========================

# --- Robust OSRM helpers (road polyline + basic turn text) ---
def _requests_session_osrm():
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.6, status_forcelist=(429,500,502,503,504),
              allowed_methods=frozenset(["GET"]))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": "CLEETS-EV/1.0"})
    return s

def _osrm_try(base_url, sl, so, el, eo, want_steps=True):
    url = f"{base_url}/route/v1/driving/{so},{sl};{eo},{el}"
    params = {"overview":"full","geometries":"geojson","alternatives":"false","steps": "true" if want_steps else "false"}
    sess = _requests_session_osrm()
    r = sess.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data.get("routes"):
        raise RuntimeError("No routes")
    rt = data["routes"][0]
    coords = rt["geometry"]["coordinates"]
    ln = LineString([(x, y) for x, y in coords])
    dist_m = float(rt["distance"])
    dur_s  = float(rt["duration"])
    steps = []
    if want_steps:
        for leg in rt.get("legs", []):
            for st in leg.get("steps", []):
                nm = st.get("name") or ""
                m  = st.get("maneuver", {})
                kind = m.get("modifier") or m.get("type") or ""
                t = " ".join([w for w in [kind, nm] if w]).strip()
                if t:
                    steps.append(t)
    return ln, dist_m, dur_s, steps

def osrm_route(sl, so, el, eo):
    for base in ("https://router.project-osrm.org", "https://routing.openstreetmap.de/routed-car"):
        try:
            ln, d, t, steps = _osrm_try(base, sl, so, el, eo, want_steps=True)
            return ln, d, t, steps, base
        except Exception:
            continue
    raise RuntimeError("OSRM routing failed on both endpoints")

# def osrm_route(sl, so, el, eo):
#     url = f"https://router.project-osrm.org/route/v1/driving/{so},{sl};{eo},{el}"
#     params = {"overview":"full","geometries":"geojson","alternatives":"false","steps":"false"}
#     try:
#         r = requests.get(url, params=params, timeout=15).json()
#         if r.get("routes"):
#             coords = r["routes"][0]["geometry"]["coordinates"]
#             ln = LineString([(c[0],c[1]) for c in coords])
#             return ln, float(r["routes"][0]["distance"]), float(r["routes"][0]["duration"]), "OSRM"
#     except Exception:
#         pass
#     ln = LineString([(so,sl),(eo,el)])
#     d_km = haversine_km(sl, so, el, eo)
#     return ln, d_km*1000.0, d_km/70.0*3600.0, "Great-circle (approx)"

def get_flood_union(bounds, include_live=True, include_fraw=True, include_fmfp=True, pad_m=SIM_DEFAULTS["wfs_pad_m"]):
    bbox = bbox_expand(bounds, pad_m); chunks=[]
    if include_fmfp:
        for lyr in FMFP_WFS.values():
            g = fetch_wfs_layer_cached(lyr, bbox)
            if not g.empty: chunks.append(g[['geometry']])
    if include_fraw:
        for lyr in FRAW_WFS.values():
            g = fetch_wfs_layer_cached(lyr, bbox)
            if not g.empty: chunks.append(g[['geometry']])
    if include_live:
        for lyr in LIVE_WFS.values():
            g = fetch_wfs_layer_cached(lyr, bbox)
            if not g.empty: chunks.append(g[['geometry']])
    if not chunks: return None
    G = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True), geometry='geometry', crs='EPSG:4326')
    try: G["geometry"] = G["geometry"].buffer(0)
    except Exception: pass
    try: G = G.explode(index_parts=False).reset_index(drop=True)
    except Exception: pass
    try: return G.to_crs('EPSG:27700').union_all()
    except Exception: return G.to_crs('EPSG:27700').unary_union

# def segment_route_by_risk(line_wgs84, risk_union_metric, buffer_m=30):
#     if risk_union_metric is None: return [line_wgs84], []
#     try: line_m = gpd.GeoSeries([line_wgs84], crs='EPSG:4326').to_crs('EPSG:27700').iloc[0]
#     except Exception: line_m = gpd.GeoSeries([line_wgs84], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
#     hit = risk_union_metric.buffer(buffer_m)
#     try: pieces = list(shp_split(line_m, hit.boundary))
#     except Exception: pieces = [line_m]
#     safe_m, risk_m = [], []
#     for seg in pieces: (risk_m if seg.intersects(hit) else safe_m).append(seg)
#     safe = gpd.GeoSeries(safe_m, crs='EPSG:27700').to_crs('EPSG:4326').tolist() if safe_m else []
#     risk = gpd.GeoSeries(risk_m, crs='EPSG:27700').to_crs('EPSG:4326').tolist() if risk_m else []
#     return safe, risk

def segment_route_by_risk(line_wgs84, risk_union_metric, buffer_m=ROUTE_BUFFER_M):
    """Split route into safe vs risk segments against a metric-union geometry."""
    if risk_union_metric is None:
        return [line_wgs84], []
    try:
        line_m = gpd.GeoSeries([line_wgs84], crs='EPSG:4326').to_crs('EPSG:27700').iloc[0]
    except Exception:
        line_m = gpd.GeoSeries([line_wgs84], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
    hit = risk_union_metric.buffer(buffer_m)
    try:
        pieces = list(shp_split(line_m, hit.boundary))
    except Exception:
        pieces = [line_m]
    safe_m, risk_m = [], []
    for seg in pieces:
        (risk_m if seg.intersects(hit) else safe_m).append(seg)
    safe = gpd.GeoSeries(safe_m, crs='EPSG:27700').to_crs('EPSG:4326').tolist() if safe_m else []
    risk = gpd.GeoSeries(risk_m, crs='EPSG:27700').to_crs('EPSG:4326').tolist() if risk_m else []
    return safe, risk

# =========================
# Graph + charger fetch
# =========================
def fetch_graph_and_chargers(center_lat, center_lon, dist_m=15000):
    """OSMnx graph + OSM charging POIs (amenity=charging_station)."""  # :contentReference[oaicite:5]{index=5}
    G = ox.graph_from_point((center_lat, center_lon), dist=dist_m, network_type="drive", simplify=True)
    try:
        pois = ox.geometries_from_point((center_lat, center_lon),
                                        tags={"amenity": "charging_station"}, dist=dist_m)
        chargers = []
        if not pois.empty:
            for _, row in pois.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                pt = geom.centroid if hasattr(geom, "centroid") else geom
                chargers.append({
                    "Latitude":  float(pt.y),
                    "Longitude": float(pt.x),
                    "Name": row.get("name", "Charging station"),
                    "ROW_ID": len(chargers)+1,
                    "AvailabilityLabel": "Operational"
                })
        chargers_df = pd.DataFrame(chargers, columns=["ROW_ID","Latitude","Longitude","Name","AvailabilityLabel"])
    except Exception:
        chargers_df = pd.DataFrame(columns=["ROW_ID","Latitude","Longitude","Name","AvailabilityLabel"])
    return G, chargers_df


# RCSP optimiser


def _build_graph_bbox(north, south, east, west):
    """
    Return a drive network graph for the given bbox, compatible with OSMnx
    versions ≤1.8 (positional), ≥1.9 (keyword-only), and 2.x (bbox tuple).
    """
    # 1) Newer OSMnx (≥1.9): keyword-only north/south/east/west
    try:
        return ox.graph_from_bbox(
            north=north, south=south, east=east, west=west,
            network_type="drive", simplify=True
        )
    except TypeError:
        pass

    # 2) OSMnx variants expecting a single bbox tuple as first positional arg
    try:
        return ox.graph_from_bbox(
            (north, south, east, west),
            network_type="drive", simplify=True
        )
    except TypeError:
        pass

    # 3) Legacy OSMnx (≤1.8): positional arguments
    try:
        return ox.graph_from_bbox(north, south, east, west, "drive", True)
    except TypeError as e:
        raise RuntimeError(f"OSMnx graph_from_bbox signature not recognised: {e}")


def _graph_two_points(sl, so, el, eo, dist_m=15000):
    # Build two local graphs around start/end and merge them
    G1 = ox.graph_from_point((sl, so), dist=dist_m, network_type="drive", simplify=True)
    G2 = ox.graph_from_point((el, eo), dist=dist_m, network_type="drive", simplify=True)
    try:
        return nx.compose(G1, G2)
    except Exception:
        G = nx.MultiDiGraph()
        G.update(G1); G.update(G2)
        return G


def rcsp_optimize(start_lat, start_lon, end_lat, end_lon,
                  battery_kwh, init_soc, reserve_soc, target_soc,
                  kwh_per_km, chargers_df, flood_union_m, extreme=False,
                  risk_penalty_per_km=None, max_seconds=5.0):
    if not HAS_OSMNX:
        raise RuntimeError("OSMnx not installed")

    # ---- Build bounding box
    minlat, maxlat = sorted([float(start_lat), float(end_lat)])
    minlon, maxlon = sorted([float(start_lon), float(end_lon)])
    pad = 0.05
    south, north = minlat - pad, maxlat + pad
    west,  east  = minlon - pad, maxlon + pad

    # Diagonal distance across bbox (km) → decide which graph strategy to use
    diag_km = haversine_km(south, west, north, east)

    # Guard absurdly large areas by degrees too
    if (east - west) > MAX_GRAPH_BBOX_DEG or (north - south) > MAX_GRAPH_BBOX_DEG:
        # You can either raise or force the two-point strategy:
        # raise RuntimeError("Area too large for local graph; use OSRM fallback")
        G = _graph_two_points(start_lat, start_lon, end_lat, end_lon, dist_m=20000)
    else:
        # Long trips → cheaper to stitch two local graphs than a huge bbox
        if diag_km > 60:
            G = _graph_two_points(start_lat, start_lon, end_lat, end_lon, dist_m=20000)
        else:
            G = _build_graph_bbox(north, south, east, west)

    # Speeds and travel time
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # ---- Project to metric CRS and get GeoDataFrames
    Gm = ox.project_graph(G, to_crs="EPSG:27700")
    nodes_m, edges_m = ox.graph_to_gdfs(Gm, nodes=True, edges=True,
                                        node_geometry=True, fill_edge_geometry=True)
    nodes,    edges  = ox.graph_to_gdfs(G,  nodes=True, edges=True,
                                        node_geometry=True, fill_edge_geometry=True)

    # ---- Risk mark (buffer in metric CRS)
    if flood_union_m is not None:
        try:
            buf = EXTREME_BUFFER_M if extreme else 0.0
            edges_m["risk"] = edges_m.geometry.buffer(0).intersects(flood_union_m.buffer(buf))
        except Exception:
            edges_m["risk"] = False
    else:
        edges_m["risk"] = False

    # Length (m) in metric CRS
    edges_m["length_m"] = edges_m.geometry.length.astype(float)

    # Join WGS84 attrs (incl. travel_time) with metric risk/length
    edges_join = edges.join(edges_m[["risk", "length_m"]])
    edges_lookup = {}
    for (u, v, k), row in edges_join.iterrows():
        L = float(row.get("length_m", 0.0))
        if not L or not math.isfinite(L):
            L = float(getattr(row, "length", 0.0)) * 111_000.0  # conservative
        T = float(row.get("travel_time", L / 13.9))             # ~50 km/h fallback
        R = bool(row.get("risk", False))
        edges_lookup[(u, v, k)] = (L, T, R)

    # ---- Nearest graph nodes (API-compatible)
    try:
        nn = ox.nearest_nodes
    except AttributeError:
        from osmnx.distance import nearest_nodes as nn

    u = nn(G, start_lon, start_lat)
    v = nn(G, end_lon,   end_lat)

    # ---- Map chargers to nearest nodes
    chargers = {}
    if isinstance(chargers_df, (pd.DataFrame, gpd.GeoDataFrame)) and not chargers_df.empty:
        for _, r in chargers_df.iterrows():
            try:
                nid = nn(G, float(r["Longitude"]), float(r["Latitude"]))
                p_kw = r.get('power_kW', None)
                try:
                    p_kw = float(p_kw) if p_kw is not None and p_kw==p_kw else None
                except Exception:
                    p_kw = None
                chargers[nid] = dict(power_kW=(p_kw if p_kw and p_kw>0 else DEFAULT_POWER_KW),
                                     operational=(str(r.get("AvailabilityLabel","")) == "Operational"))
            except Exception:
                continue

    # ---- RCSP over (node, discretised SOC)
    step = SOC_STEP
    Q = [round(i * step, 2) for i in range(0, int(1 / step) + 1)]
    def q_to_idx(q): return max(0, min(len(Q)-1, int(round(q/step))))
    reserve_q = float(reserve_soc)/100.0
    tgt_q     = float(target_soc)/100.0
    init_q    = float(init_soc)/100.0

    INF = 10**18
    best = {}
    pred = {}
    hq = []
    start_key = (u, q_to_idx(init_q))
    best[start_key] = 0.0
    heapq.heappush(hq, (0.0, u, q_to_idx(init_q)))

    # Pareto sets per node for dominance: list of (qi, cost)
    pareto = {}

    t0 = time.time()

    # Collapse multiedges to fastest representative
    adj_min = {}
    for (uu, vv, kk), (L, T, R) in edges_lookup.items():
        cur = adj_min.get((uu, vv))
        if (cur is None) or (T < cur[1]):
            adj_min[(uu, vv)] = (L, T, R)
    adj = {}
    for (uu, vv), (L, T, R) in adj_min.items():
        adj.setdefault(uu, []).append((vv, L, T, R))

    risk_penalty = (risk_penalty_per_km if (risk_penalty_per_km is not None) else (EXTREME_RISK_PENALTY_PER_KM if extreme else BASE_RISK_PENALTY_PER_KM))

    while hq:
        # Timeout check
        if (time.time() - t0) > max_seconds:
            raise TimeoutError(f"RCSP exceeded time budget of {max_seconds:.1f}s")
        cost, node, qi = heapq.heappop(hq)
        # Dominance: if an existing label at this node has >= SOC and <= cost, skip
        pls = pareto.get(node, [])
        dominated = False
        for qj_existing, cj in pls:
            if (qj_existing >= qi) and (cj <= cost + 1e-9):
                dominated = True
                break
        if dominated:
            continue
        if best.get((node, qi), INF) < cost - 1e-9:
            continue
        if node == v and Q[qi] >= reserve_q:
            # update pareto set with this terminal label
            ps = pareto.setdefault(node, [])
            ps.append((qi, cost))
            break

        # Drive transitions
        for (vv, L, T, R) in adj.get(node, []):
            E_kWh = (L/1000.0) * float(kwh_per_km)
            dq = E_kWh / float(battery_kwh)
            if Q[qi] - dq < reserve_q - 1e-9:
                continue
            qj = q_to_idx(max(reserve_q, Q[qi] - dq))
            new_cost = cost + T + (risk_penalty * (L/1000.0) if R else 0.0)
            key = (vv, qj)
            if new_cost + 1e-9 < best.get(key, INF):
                best[key] = new_cost
                pred[key] = (node, qi, "drive", dict(L=L, T=T, R=R))
                heapq.heappush(hq, (new_cost, vv, qj))
                ps = pareto.setdefault(vv, [])
                ps[:] = [(qidx, cval) for (qidx, cval) in ps if not (qj >= qidx and new_cost <= cval + 1e-9)]
                add_ok = True
                for (qidx, cval) in ps:
                    if (qidx >= qj) and (cval <= new_cost + 1e-9):
                        add_ok = False; break
                if add_ok: ps.append((qj, new_cost))

        # Charge transitions (only at charger nodes)
        ch = chargers.get(node)
        if ch and ch.get("operational", True):
            p_kw = float(ch.get("power_kW", DEFAULT_POWER_KW))
            max_target = max(tgt_q, Q[qi])
            for dq_step in [CHARGE_STEP, 2*CHARGE_STEP, 3*CHARGE_STEP]:
                q_next = min(1.0, Q[qi] + dq_step)
                if q_next <= Q[qi] or q_next < max_target - 1e-9:
                    continue
                added_kWh = float(battery_kwh) * (q_next - Q[qi])
                charge_time_s = 3600.0 * (added_kWh / max(1e-6, p_kw))
                key = (node, q_to_idx(q_next))
                new_cost = cost + charge_time_s
                if new_cost + 1e-9 < best.get(key, INF):
                    best[key] = new_cost
                    pred[key] = (node, qi, "charge",
                                 dict(p_kW=p_kw, added_kWh=added_kWh, dt=charge_time_s))
                    qn = q_to_idx(q_next)
                    heapq.heappush(hq, (new_cost, node, qn))
                    ps = pareto.setdefault(node, [])
                    ps[:] = [(qidx, cval) for (qidx, cval) in ps if not (qn >= qidx and new_cost <= cval + 1e-9)]
                    add_ok = True
                    for (qidx, cval) in ps:
                        if (qidx >= qn) and (cval <= new_cost + 1e-9):
                            add_ok = False; break
                    if add_ok: ps.append((qn, new_cost))

    # ---- Goal
    goal = None
    goal_cost = INF
    for qi in range(len(Q)-1, -1, -1):
        k = (v, qi)
        if k in best and best[k] < goal_cost:
            goal, goal_cost = k, best[k]
    if goal is None:
        raise RuntimeError("No feasible RCSP solution in bbox.")

    # ---- Reconstruct
    path_nodes = []
    charges = []
    k = goal
    while k in pred:
        prev, pqi, act, info = pred[k]
        if act == "charge":
            charges.append((k[0], Q[pqi], Q[k[1]], info))
        path_nodes.append(k[0])
        k = (prev, pqi)
    path_nodes.append(u)
    path_nodes.reverse()
    charges.reverse()

    lats = [G.nodes[n]['y'] for n in path_nodes]
    lons = [G.nodes[n]['x'] for n in path_nodes]
    line = LineString([(lon, lat) for lon, lat in zip(lons, lats)])

    # ---- Segment by risk
    safe_lines, risk_lines = segment_route_by_risk(
        line, flood_union_m, buffer_m=(EXTREME_BUFFER_M if extreme else ROUTE_BUFFER_M)
    )

    # ---- Map planned charges back to charger rows (optional)
    planned_stops = []
    if isinstance(chargers_df, (pd.DataFrame, gpd.GeoDataFrame)) and not chargers_df.empty and charges:
        cols = {c.lower(): c for c in chargers_df.columns}
        lat_col = cols.get("latitude")
        lon_col = cols.get("longitude")
        rowid_col = cols.get("row_id") or "ROW_ID"
        if lat_col and lon_col and (rowid_col in chargers_df.columns):
            lats_arr = pd.to_numeric(chargers_df[lat_col], errors="coerce").to_numpy()
            lons_arr = pd.to_numeric(chargers_df[lon_col], errors="coerce").to_numpy()
            for (nid, q_before, q_after, info) in charges:
                try:
                    lat = float(G.nodes[nid]["y"]); lon = float(G.nodes[nid]["x"])
                except Exception:
                    continue
                d2 = (lats_arr - lat)**2 + (lons_arr - lon)**2
                idx = int(np.nanargmin(d2)) if np.isfinite(d2).any() else None
                if idx is None: continue
                row = chargers_df.iloc[idx]
                rid = int(row[rowid_col]) if pd.notna(row.get(rowid_col, None)) else int(idx)
                planned_stops.append(dict(
                    ROW_ID=rid,
                    Operational=(str(row.get("AvailabilityLabel","")) == "Operational"),
                    soc_before=q_before, soc_after=q_after,
                    energy_kwh=float(battery_kwh) * (q_after - q_before),
                    charge_time_min=float(info.get("dt", 0.0)) / 60.0
                ))

    return line, safe_lines, risk_lines, planned_stops, goal_cost


# =========================
# Planner used by the Dash UI
# =========================
def line_to_latlon_list(line: LineString) -> List[Tuple[float,float]]:
    return [(lat, lon) for lon, lat in line.coords]

def plan_rcsp_route(sl, so, el, eo, ev: EVParams, extreme=False, risk_penalty_per_km=None, rcsp_timeout_s=5.0):
    """Wrapper that calls rcsp_optimize and converts its outputs to UI-friendly objects."""
    if not HAS_OSMNX:
        raise RuntimeError("OSMnx not available")

    ctr_lat, ctr_lon = (sl+el)/2.0, (so+eo)/2.0
    _G, chargers_df = fetch_graph_and_chargers(ctr_lat, ctr_lon, dist_m=20000)
    try:
        # Prefer ONS/NCR set within a radius if available (uses global gdf_ev prepared earlier)
        if 'gdf_ev' in globals():
            import geopandas as gpd
            from shapely.geometry import Point
            center_pt = gpd.GeoSeries([Point(ctr_lon, ctr_lat)], crs="EPSG:4326")
            radius_m = 20000
            ev_local = gdf_ev.to_crs('EPSG:27700')
            mask = ev_local.geometry.distance(center_pt.to_crs('EPSG:27700').iloc[0]) <= radius_m
            subset = gdf_ev.loc[mask.values]
            if not subset.empty:
                out = subset.copy()
                out['Latitude'] = out.geometry.y
                out['Longitude'] = out.geometry.x
                # Normalize power column names to 'power_kW' if present
                for cand in ['power_kw','Power_kW','rated_power_kw','RatedPowerKW','connector_power_kw']:
                    if cand in out.columns:
                        out = out.rename(columns={cand:'power_kW'})
                        break
                # Minimal schema for RCSP
                keep = ['Latitude','Longitude','Name','ROW_ID','AvailabilityLabel','power_kW']
                for col in keep:
                    if col not in out.columns:
                        out[col] = None
                chargers_df = out[keep].reset_index(drop=True)
    except Exception:
        pass

    # Build flood union (metric CRS) from WFS layers over the OD bounding box
    try:
        bounds = (min(sl, el), min(so, eo), max(sl, el), max(so, eo))  # (south, west, north, east)
        flood_union_m = get_flood_union(bounds, include_live=True, include_fraw=True, include_fmfp=True, pad_m=SIM_DEFAULTS["wfs_pad_m"])
    except Exception:
        flood_union_m = None

    try:
        line, safe_lines, risk_lines, planned_stops, goal_cost = rcsp_optimize(
            sl, so, el, eo,
            ev.battery_kWh, ev.start_soc*100, ev.reserve_soc*100, ev.target_soc*100,
            ev.kWh_per_km, chargers_df, flood_union_m, extreme=extreme,
            risk_penalty_per_km=risk_penalty_per_km, max_seconds=rcsp_timeout_s
        )
        used_src = "RCSP"
        dist_m = float(line.length)*111000.0 if hasattr(line, "length") else None
        dur_s = goal_cost
        step_text = []
    except Exception as e:
        line, dist_m, dur_s, step_text, used_src = osrm_route(float(sl), float(so), float(el), float(eo))
        safe_lines, risk_lines = segment_route_by_risk(line, flood_union_m, buffer_m=ROUTE_BUFFER_M)
        planned_stops = []
        goal_cost = dur_s

    coords = line_to_latlon_list(line)

    stops: List[StopInfo] = []
    for st in planned_stops:
        try:
            row = chargers_df.loc[chargers_df["ROW_ID"].eq(st["ROW_ID"])].iloc[0]
            s = StopInfo(
                lat=float(row["Latitude"]),
                lon=float(row["Longitude"]),
                name=str(row.get("Name","Charger")),
                postcode=get_postcode(float(row["Latitude"]), float(row["Longitude"])),
                ZoneLabel=row.get("ZoneLabel","Outside"),
                ZoneColor=row.get("ZoneColor", ZONE_COLORS["Outside"]),
                Operational=bool(st.get("Operational", True)),
                soc_before=float(st.get("soc_before", 0.0)),
                soc_after=float(st.get("soc_after", 0.0)),
                energy_kWh=float(st.get("energy_kwh", 0.0)),
                charge_time_min=float(st.get("charge_time_min", 0.0)),
            )
            stops.append(s)
            time.sleep(1)  # be polite to Nominatim
        except Exception:
            continue

    total_cost_min = float(goal_cost)/60.0
    return coords, stops, total_cost_min

# =========================
# Folium helpers — tiles & overlays
# =========================
def add_wms_group(fmap, title_to_layer: dict, visible=True, opacity=0.55):
    for title, layer in title_to_layer.items():
        try:
            WmsTileLayer(url=OWS_BASE, layers=layer, name=f"{title} (WMS)",
                         fmt="image/png", transparent=True, opacity=opacity, version="1.3.0", show=visible).add_to(fmap)
        except Exception:
            pass

def add_live_wfs_popups(fmap, df_like):
    bbox = _bbox_for(df_like if isinstance(df_like, gpd.GeoDataFrame) and not df_like.empty else gdf_ev)
    for title, layer in LIVE_WFS.items():
        g = fetch_wfs_layer_cached(layer, bbox)
        if g.empty: continue
        folium.GeoJson(g.to_json(), name=f"{title} (WFS)",
                       style_function=lambda _: {'fillOpacity':0.15,'weight':2}).add_to(fmap)

def add_base_tiles(m):
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap",
        attr="© OpenStreetMap contributors",
        control=True,
        overlay=False, max_zoom=19
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        name="CartoDB Positron",
        attr="© OpenStreetMap contributors, © CARTO",
        control=True, overlay=False, max_zoom=19
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        name="CartoDB Dark Matter",
        attr="© OpenStreetMap contributors, © CARTO",
        control=True, overlay=False, max_zoom=19
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        name="OSM Humanitarian",
        attr="© OpenStreetMap contributors, Tiles courtesy of HOT",
        control=True, overlay=False, max_zoom=19
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Esri WorldImagery",
        attr="Tiles © Esri & contributors",
        control=True, overlay=False, max_zoom=19
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        name="OpenTopoMap",
        attr="© OpenStreetMap contributors, SRTM; style © OpenTopoMap (CC-BY-SA)",
        control=True, overlay=False, max_zoom=17
    ).add_to(m)

# =========================
# Rendering (icons by zone)
# =========================

def _row_to_tooltip_html(row, title=None):
    s = f"<b>{title or 'Data Point'}</b><br>"
    for k, v in row.items():
        # Drop columns where the value is NaN, None, or empty string
        if pd.isna(v) or v is None or str(v).strip() == "":
            continue
        s += f"<b>{k}:</b> {v}<br>"
    return s

def render_map_html_ev(df_map, show_fraw, show_fmfp, show_live, show_ctx, light=False):
    m = folium.Map(location=[51.6,-3.2], zoom_start=9, tiles=None, control_scale=True)
    add_base_tiles(m)

    red_group   = folium.FeatureGroup(name="Chargers: Zone 3 / High (red)", show=True).add_to(m)
    amber_group = folium.FeatureGroup(name="Chargers: Zone 2 / Medium (amber)", show=True).add_to(m)
    green_group = folium.FeatureGroup(name="Chargers: Zone 1 / Low–Outside (green)", show=True).add_to(m)

    red_cluster   = MarkerCluster(name="Cluster: Zone 3 / High").add_to(red_group)
    amber_cluster = MarkerCluster(name="Cluster: Zone 2 / Medium").add_to(amber_group)
    green_cluster = MarkerCluster(name="Cluster: Zone 1 / Low–Outside").add_to(green_group)

    Draw(export=False, position='topleft',
         draw_options={'polygon': {'allowIntersection': False, 'showArea': True},
                       'rectangle': True, 'polyline': False, 'circle': False,
                       'circlemarker': False, 'marker': False},
         edit_options={'edit': True}).add_to(m)

    for _, row in df_map.iterrows():
        zlabel = (row.get("ZoneLabel") or "Outside")
        if zlabel in ("Zone 3","High"):
            color_hex = ZONE_COLORS["Zone 3"]; group_cluster = red_cluster
        elif zlabel in ("Zone 2","Medium"):
            color_hex = ZONE_COLORS["Zone 2"]; group_cluster = amber_cluster
        else:
            color_hex = ZONE_COLORS["Outside"]; group_cluster = green_cluster

        title = f"{row.get('Operator','')} ({row.get('country','')})"
        try:
            tooltip_html = _row_to_tooltip_html(row, title=title)
            tooltip_obj = folium.Tooltip(tooltip_html, sticky=True)
        except Exception:
            tooltip_obj = title

        marker = folium.Marker([row['Latitude'], row['Longitude']],
                               tooltip=tooltip_obj,
                               icon=make_beautify_icon(color_hex))
        group_cluster.add_child(marker)

    if show_fraw: add_wms_group(m, FRAW_WMS, True, 0.50)
    if show_fmfp: add_wms_group(m, FMFP_WMS, True, 0.55)
    if show_ctx:  add_wms_group(m, CONTEXT_WMS, False, 0.45)
    if show_live:
        add_wms_group(m, LIVE_WMS, True, 0.65)
        if not light:
            add_live_wfs_popups(m, df_map)

    folium.LayerControl(collapsed=True).add_to(m)

    legend_html = f"<div style='position: fixed; bottom:20px; left:20px; z-index:9999; background:white; padding:10px 12px; border:1px solid #ccc; border-radius:6px; font-size:13px;'>" \
                  f"<b>Chargers by Flood Model Zone</b>" \
                  f"<div style='margin-top:6px'><span style='display:inline-block;width:12px;height:12px;background:{ZONE_COLORS['Zone 3']};margin-right:6px;border:1px solid #555;'></span> Zone 3 / High</div>" \
                  f"<div><span style='display:inline-block;width:12px;height:12px;background:{ZONE_COLORS['Zone 2']};margin-right:6px;border:1px solid #555;'></span> Zone 2 / Medium</div>" \
                  f"<div><span style='display:inline-block;width:12px;height:12px;background:{ZONE_COLORS['Outside']};margin-right:6px;border:1px solid #555;'></span> Zone 1 / Low–Very Low / Outside</div>" \
                  f"</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    return m.get_root().render()

def render_map_html_route(full_line, route_safe, route_risk, start, end, chargers,
                          all_chargers_df=None,  # <--- new argument
                          animate=False, speed_kmh=50, show_live_backdrops=False):
    m = folium.Map(location=[(start[0]+end[0])/2, (start[1]+end[1])/2], zoom_start=11,
                   tiles=None, control_scale=True)
    add_base_tiles(m)

    # --- Plot all charging stations (clustered by risk zone) ---
    if all_chargers_df is not None:
        red_group   = folium.FeatureGroup(name="All Chargers: Zone 3 / High (red)", show=True).add_to(m)
        amber_group = folium.FeatureGroup(name="All Chargers: Zone 2 / Medium (amber)", show=True).add_to(m)
        green_group = folium.FeatureGroup(name="All Chargers: Zone 1 / Low–Outside (green)", show=True).add_to(m)

        red_cluster   = MarkerCluster(name="Cluster: Zone 3 / High").add_to(red_group)
        amber_cluster = MarkerCluster(name="Cluster: Zone 2 / Medium").add_to(amber_group)
        green_cluster = MarkerCluster(name="Cluster: Zone 1 / Low–Outside").add_to(green_group)

        for _, row in all_chargers_df.iterrows():
            zlabel = (row.get("ZoneLabel") or "Outside")
            if zlabel in ("Zone 3","High"):
                color_hex = ZONE_COLORS["Zone 3"]; group_cluster = red_cluster
            elif zlabel in ("Zone 2","Medium"):
                color_hex = ZONE_COLORS["Zone 2"]; group_cluster = amber_cluster
            else:
                color_hex = ZONE_COLORS["Outside"]; group_cluster = green_cluster

            title = f"{row.get('Operator','')} ({row.get('country','')})"
            try:
                tooltip_html = _row_to_tooltip_html(row, title=title)
                tooltip_obj = folium.Tooltip(tooltip_html, sticky=True)
            except Exception:
                tooltip_obj = title

            marker = folium.Marker([row['Latitude'], row['Longitude']],
                                   tooltip=tooltip_obj,
                                   icon=make_beautify_icon(color_hex))
            group_cluster.add_child(marker)

    # Route lines
    def add_lines(lines, color, name):
        if not lines: return
        fg = folium.FeatureGroup(name=name).add_to(m)
        for ln in lines:
            coords = [(lat,lon) for lon,lat in ln.coords]
            folium.PolyLine(coords, color=color, weight=6, opacity=0.9).add_to(fg)

    # Whole route (faint for context)
    if isinstance(full_line, LineString):
        coords_full = [(lat,lon) for lon,lat in full_line.coords]
        folium.PolyLine(coords_full, color="#999999", weight=3, opacity=0.5, tooltip="Planned route").add_to(m)

    add_lines(route_safe, "#2b8cbe", "Route – safe")
    add_lines(route_risk, "#e31a1c", "Route – flood risk")
    folium.Marker(start, tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(end,   tooltip="End",   icon=folium.Icon(color="blue")).add_to(m)

    # Highlight planned stops (simulate stops) on top (with bold border)
    cluster = MarkerCluster(name="Planned route stops").add_to(m)
    for st in chargers:
        try:
            row = gdf_ev.loc[gdf_ev['ROW_ID'].eq(st.get('ROW_ID',-1))].iloc[0]
        except Exception:
            continue
        zlabel = (row.get('ZoneLabel') or 'Outside')
        if zlabel in ('Zone 3','High'):   color_hex = ZONE_COLORS['Zone 3']
        elif zlabel in ('Zone 2','Medium'): color_hex = ZONE_COLORS['Zone 2']
        else:                               color_hex = ZONE_COLORS['Outside']
        title = f"{row.get('Operator','')} ({row.get('country','')})"
        try:
            tooltip_html = _row_to_tooltip_html(row, title=title)
            tooltip_obj = folium.Tooltip(tooltip_html, sticky=True)
        except Exception:
            tooltip_obj = title
        # Mark planned stops with a bold border
        folium.Marker([row['Latitude'], row['Longitude']], tooltip=tooltip_obj,
                      icon=make_beautify_icon(color_hex)).add_to(cluster)

    if show_live_backdrops:
        add_wms_group(m, LIVE_WMS, visible=True, opacity=0.65)

    folium.LayerControl(collapsed=True).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom:20px; left:20px; z-index:9999; background:white; padding:10px 12px; border:1px solid #ccc; border-radius:6px; font-size:13px;">
      <b>Charger icon colour — Flood model zone</b>
      <div style="margin-top:6px"><span style="display:inline-block;width:12px;height:12px;background:#D32F2F;margin-right:6px;"></span> Red: Zone 3 / High</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#FFC107;margin-right:6px;"></span> Amber: Zone 2 / Medium</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#2E7D32;margin-right:6px;"></span> Green: Zone 1 / Low–Very Low / Outside</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m.get_root().render()


# =========================
# Weather (Open-Meteo default; Met Office optional)
# =========================
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_get(url, headers_tuple=(), params_tuple=()):
    headers = dict(headers_tuple) if headers_tuple else {}
    params = dict(params_tuple) if params_tuple else {}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _ts():
    return str(int(time.time()//300))

METOFFICE_KEY = os.environ.get("METOFFICE_KEY", "").strip()
METOFFICE_SITE_API = os.environ.get("METOFFICE_SITE_API", "").strip()  # optional

def get_weather(lat=51.48, lon=-3.18):
    try:
        if METOFFICE_KEY and METOFFICE_SITE_API:
            headers = (("apikey", METOFFICE_KEY),)
            params = (("latitude", str(lat)), ("longitude", str(lon)),)
            data = cached_get(METOFFICE_SITE_API, headers, params)
            return {"provider":"Met Office", "raw": data}
        else:
            url = "https://api.open-meteo.com/v1/forecast"
            params = (
                ("latitude", str(lat)), ("longitude", str(lon)),
                ("current", "temperature_2m,precipitation,wind_speed_10m"),
                ("hourly", "temperature_2m,precipitation_probability,wind_speed_10m"),
                ("timezone", "Europe/London"),
                ("_ts", _ts()),
            )
            data = cached_get(url, (), params)
            return {"provider":"Open-Meteo", "raw": data}
    except Exception as e:
        return {"provider":"error", "error": str(e)}

def _parse_metoffice_timeseries(raw):
    try:
        feats = raw.get("features") or []
        if feats and isinstance(feats, list):
            ts = feats[0].get("properties", {}).get("timeSeries") or []
            times = [r.get("time") for r in ts if "time" in r][:24]
            temps = [r.get("screenTemperature") for r in ts][:24]
            pops  = [r.get("precipitationProbability") or r.get("precipProb") for r in ts][:24]
            if times and temps:
                return {"time": times, "temp": temps, "pop": pops or [None]*len(times)}
    except Exception:
        pass
    return {}

# =========================
# Dash app + header logo + KML download
# =========================

app = Dash(__name__)
app.layout = html.Div([
    html.H1("OK"),
    dcc.Dropdown(id="x")
])

server = app.server

# Serve logo from Google Drive (cached to disk)
@server.route("/__logo")
def _serve_logo():
    try:
        if not os.path.exists(LOGO_CACHE_PATH) or (time.time()-os.path.getmtime(LOGO_CACHE_PATH)) > 7*24*3600:
            content = read_bytes_resilient_gdrive(LOGO_GDRIVE_FILE_OR_URL)
            with open(LOGO_CACHE_PATH, "wb") as f: f.write(content)
        with open(LOGO_CACHE_PATH, "rb") as f:
            return Response(f.read(), mimetype="image/png")
    except Exception:
        # tiny transparent placeholder
        return Response(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06'
                        b'\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc``\x00\x00\x00\x02\x00\x01'
                        b'\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82', mimetype="image/png")

WALES_LOCS = {
    "Cardiff": (51.4816, -3.1791),
    "Swansea": (51.6214, -3.9436),
    "Newport": (51.5842, -2.9977),
    "Aberystwyth": (52.4153, -4.0829),
    "Bangor": (53.2290, -4.1294)
}

def preload_zones_json() -> str:
    p = "cache_model_zones.parquet"
    if os.path.exists(p):
        try:
            cache = pd.read_parquet(p)
            if {"ROW_ID","ZoneLabel","ZoneColor"}.issubset(cache.columns) and not cache.empty:
                return cache[["ROW_ID","ZoneLabel","ZoneColor"]].to_json(orient="records")
        except Exception:
            pass
    return "[]"

# Build KML from route+stops
def _kml_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _kml_argb_from_hex(rgb: str) -> str:
    """
    Convert #rrggbb to KML aabbggrr (opaque).
    Falls back to green if input is malformed.
    """
    try:
        if not (isinstance(rgb, str) and rgb.startswith("#") and len(rgb) == 7):
            raise ValueError
        rr, gg, bb = rgb[1:3], rgb[3:5], rgb[5:7]
        return f"ff{bb}{gg}{rr}"
    except Exception:
        return "ff327d2e"  # fallback for "#2E7D32"

def build_kml(route_data: dict) -> str:
    name = "EV Journey Simulator"
    linestring = ""
    coords = (route_data or {}).get("route") or []

    if coords:
        coord_str = " ".join(f"{p['lon']:.6f},{p['lat']:.6f},0" for p in coords)
        linestring = f"""
  <Placemark>
    <name>Planned route</name>
    <Style><LineStyle><color>ff8a2be2</color><width>4</width></LineStyle></Style>
    <LineString><tessellate>1</tessellate><coordinates>{coord_str}</coordinates></LineString>
  </Placemark>"""

    def mk_pt(title, lat, lon, color_hex=None):
        kml_color = _kml_argb_from_hex(color_hex or "#2E7D32")
        return f"""
  <Placemark>
    <name>{_kml_escape(title)}</name>
    <Style><IconStyle><color>{kml_color}</color></IconStyle></Style>
    <Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>
  </Placemark>"""

    pts = []
    s = (route_data or {}).get("start")
    e = (route_data or {}).get("end")
    if s:
        pts.append(mk_pt("Start", float(s["lat"]), float(s["lon"]), "#2E7D32"))
    if e:
        pts.append(mk_pt("End", float(e["lat"]), float(e["lon"]), "#1f78b4"))

    for i, st in enumerate((route_data or {}).get("stops") or [], 1):
        title = st.get("name") or f"Stop {i}"
        color_hex = st.get("ZoneColor") or "#2E7D32"
        pts.append(mk_pt(title, float(st["lat"]), float(st["lon"]), color_hex))

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{_kml_escape(name)}</name>{linestring}
  {''.join(pts)}
</Document>
</kml>"""
# Layout
from dash import html, dcc
# Layout
from dash import html, dcc

app.layout = html.Div([
    # Header
    html.Div([
        html.Img(
            src="/__logo",
            style={"height": "200px", "marginRight": "8px"}
        ),
        html.H1(
            "CLEETS-SMART: Sustainable Mobility and Resilient Transport",
            style={"margin": "4px"}
        )
    ], style={
        "display": "flex",
        "alignItems": "center",
        "gap": "100px",
        "marginBottom": "8px"
    }),

    # Section A – Chargers & Flood Overlays
    html.H2("A) Chargers & Flood Overlays", style={"margin": "24px 7px 8px"}),
    html.Div([
        html.Div([
            html.Label("country(s)"),
            dcc.Dropdown(
                id="f-country",
                options=[{"label": t, "value": t} for t in country_OPTIONS],
                value=[], multi=True, placeholder="All countrys"
            )
        ], style={"minWidth": "260px"}),

        html.Div([
            html.Label("country contains"),
            dcc.Input(id="f-country-like", type="text", placeholder="substring", debounce=True)
        ], style={"minWidth": "220px"}),

        html.Div([
            html.Label("Operational"),
            dcc.Checklist(
                id="f-op",
                options=[
                    {"label": "Operational", "value": "op"},
                    {"label": "Not operational", "value": "down"},
                    {"label": "Unknown", "value": "unk"}
                ],
                value=["op", "down", "unk"],
                inputStyle={"marginRight": "6px"}
            )
        ], style={"minWidth": "320px"}),

        html.Div([
            html.Label("Show overlays"),
            dcc.Checklist(
                id="layers",
                options=[
                    {"label": "FRAW", "value": "fraw"},
                    {"label": "FMfP", "value": "fmfp"},
                    {"label": "Live warnings", "value": "live"},
                    {"label": "Context", "value": "ctx"}
                ],
                value=["fraw", "fmfp"],
                inputStyle={"marginRight": "6px"}
            )
        ], style={"minWidth": "360px"}),

        html.Div([
            html.Label("Start-up mode"),
            dcc.Checklist(
                id="light",
                options=[{"label": "Light mode (fast start)", "value": "on"}],
                value=["on"]
            )
        ], style={"minWidth": "260px"}),

        html.Button("Compute/Update zones", id="btn-zones", n_clicks=0,
                    style={"height": "38px", "marginLeft": "8px"}),
        html.Button("Refresh overlays", id="btn-refresh", n_clicks=0,
                    style={"height": "38px", "marginLeft": "8px"}),
    ], style={
        "display": "flex",
        "gap": "12px",
        "alignItems": "end",
        "flexWrap": "wrap",
        "margin": "6px 0 12px"
    }),

    # Section B – Journey Simulator
    html.H2("B) Journey Simulator"),
    html.Div([
        # Postcode inputs
        html.Div([
            html.Label("Start & End (postcode)"),
            dcc.Input(
                id="start_pc", type="text",
                placeholder="e.g. CF10 3AT or 'Cardiff Castle'",
                debounce=True, style={"width": "260px"}
            ),
            dcc.Input(
                id="end_pc", type="text",
                placeholder="e.g. SA1 3SN or 'Swansea Marina'",
                debounce=True, style={"width": "260px"}
            ),
            html.Button(
                "Use postcodes", id="btn-geocode", n_clicks=0,
                style={"marginLeft": "8px"}
            ),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, 260px) 140px",
            "gap": "8px",
            "marginBottom": "6px"
        }),

        # Lat/lon row
        html.Div([
            html.Label("Start lat,lon"),
            dcc.Input(id="sla", type="number", value=51.4816, step=0.0001),
            dcc.Input(id="slo", type="number", value=-3.1791, step=0.0001),
            html.Label("End lat,lon"),
            dcc.Input(id="ela", type="number", value=51.6164, step=0.0001),
            dcc.Input(id="elo", type="number", value=-3.9409, step=0.0001),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(4, 180px)",
            "gap": "8px"
        }),

        # Vehicle selection
        html.Div([
            html.Label("Vehicle category", style={"whiteSpace": "nowrap"}),
            dcc.Dropdown(
                id="veh_cat",
                options=[{"label": c, "value": c} for c in VEHICLE_PRESETS.keys()],
                value="Hatchback", clearable=False,
                style={"flex": "1", "minWidth": "220px"}
            ),
            html.Label("Vehicle model", style={"marginLeft": "16px", "whiteSpace": "nowrap"}),
            dcc.Dropdown(
                id="veh_model",
                options=[{"label": m, "value": m}
                         for m in VEHICLE_PRESETS["Hatchback"].keys()],
                value="VW ID.3 Pro", clearable=False,
                style={"flex": "1", "minWidth": "260px"}
            ),
        ], style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "gap": "12px",
            "marginTop": "6px",
            "flexWrap": "nowrap",
            "width": "100%"
        }),

        # Battery parameters
        html.Div([
            html.Label("Battery kWh"), dcc.Input(id="batt", type="number", value=75.0, step=1),
            html.Label("Start SoC (0..1)"), dcc.Input(id="si", type="number", value=0.8, step=0.05),
            html.Label("Reserve SoC"), dcc.Input(id="sres", type="number", value=0.1, step=0.05),
            html.Label("Target SoC"), dcc.Input(id="stgt", type="number", value=0.8, step=0.05),
            html.Label("kWh/km"), dcc.Input(id="kwhkm", type="number", value=0.20, step=0.01),
            html.Label("Max charge kW"), dcc.Input(id="pmax", type="number", value=120.0, step=5),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(6, 160px)",
            "gap": "8px",
            "marginTop": "8px"
        }),

        dcc.Checklist(
            id="show_leg_details",
            options=[{"label": "Show per-leg details", "value": "details"}],
            value=["details"], inline=True
        ),

        dcc.RadioItems(
            id="units",
            options=[
                {"label": "km / kWh / %", "value": "metric"},
                {"label": "miles / kWh / %", "value": "imperial"}
            ],
            value="metric", inline=True
        ),

        html.Button("Optimise", id="simulate", n_clicks=0, style={"marginTop": "10px"}),
        html.Button("Download KML", id="btn-kml", n_clicks=0, style={"marginLeft": "8px"}),

        html.Div(id="status", style={"marginTop": "10px"}),
        html.Div(id="explain", style={"marginTop": "10px", "whiteSpace": "pre-line"}),

        dcc.Loading(
            html.Iframe(
                id="map",
                srcDoc=(
                    "<html><body style='font-family:sans-serif;padding:10px'>Loading…</body></html>"
                ),
                style={
                    "width": "100%",
                    "height": "620px",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px"
                }
            )
        ),

        html.Div(id="itinerary", style={"marginTop": "10px"}),
    ]),

    # Section C – Weather
    html.H2("C) Weather for Wales"),
    html.Div([
        html.Div([
            html.Label("Location"),
            dcc.Dropdown(
                id="wales-location",
                value="Cardiff",
                options=[{"label": k, "value": k} for k in WALES_LOCS.keys()],
                clearable=False,
                style={"width": "220px"}
            )
        ], style={"marginRight": "16px"}),

        dcc.Interval(id="wx-refresh", interval=5 * 60 * 1000, n_intervals=0)
    ], style={
        "display": "flex",
        "alignItems": "center",
        "gap": "12px"
    }),

    html.Div(
        id="weather-split",
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "12px",
            "alignItems": "stretch"
        }
    ),

    # Persistent storage + background refreshers
    dcc.Store(id="store-zones", data=preload_zones_json()),
    dcc.Store(id="overlay-refresh-token"),
    dcc.Store(id="store-route"),
    dcc.Download(id="dl-kml"),
    dcc.Interval(id="init", interval=250, n_intervals=0, max_intervals=1)
])

# -------------------------
# Callbacks
# -------------------------
@app.callback(
    Output("overlay-refresh-token", "data"),
    Input("btn-refresh", "n_clicks"),
    State("overlay-refresh-token", "data"),
    prevent_initial_call=True
)
def _bump_refresh(_n, tok):
    return (tok or 0) + 1

@app.callback(
    Output("store-zones", "data"),
    Input("btn-zones", "n_clicks"),
    prevent_initial_call=True
)
def _compute_zones(_n):
    zones = compute_model_zones_for_points(gdf_ev)
    try: zones.to_parquet("cache_model_zones.parquet", index=False)
    except Exception: pass
    return zones.to_json(orient="records")

@app.callback(
    Output("sla","value"), Output("slo","value"),
    Output("ela","value"), Output("elo","value"),
    Input("btn-geocode","n_clicks"),
    State("start_pc","value"), State("end_pc","value"),
    prevent_initial_call=True
)
def _fill_latlon_from_postcodes(_n, start_pc, end_pc):
    s, e = geocode_start_end(start_pc or "", end_pc or "")
    # Keep existing values if a side fails to geocode
    sla = dash.no_update; slo = dash.no_update
    ela = dash.no_update; elo = dash.no_update
    if s: sla, slo = float(s[0]), float(s[1])
    if e: ela, elo = float(e[0]), float(e[1])
    return sla, slo, ela, elo

@app.callback(
    Output("veh_model", "options"),
    Output("veh_model", "value"),
    Input("veh_cat", "value"),
)
def _update_vehicle_models(cat):
    cat = cat or next(iter(VEHICLE_PRESETS))
    models = list(VEHICLE_PRESETS.get(cat, {}).keys())
    opts = [{"label": m, "value": m} for m in models]
    # keep first model as default
    val = models[0] if models else dash.no_update
    return opts, val

@app.callback(
    Output("batt", "value"),
    Output("kwhkm", "value"),
    Output("pmax", "value"),
    # Optionally nudge target SoC to 0.8 on selection:
    # Output("stgt", "value"),
    Input("veh_cat", "value"),
    Input("veh_model", "value"),
    prevent_initial_call=True
)
def _apply_vehicle_preset(cat, model):
    try:
        spec = VEHICLE_PRESETS.get(cat, {}).get(model, {})
        batt  = float(spec.get("battery_kWh", dash.no_update))
        kwhkm = float(spec.get("kWh_per_km", dash.no_update))
        pmax  = float(spec.get("max_charge_kW", dash.no_update))
        return batt, kwhkm, pmax  # , 0.8
    except Exception:
        return dash.no_update, dash.no_update, dash.no_update  # , dash.no_update

@app.callback(
    Output("map", "srcDoc"),
    Output("itinerary", "children"),
    Output("store-route", "data"),
    # EV filters + overlays
    Input("f-country", "value"),
    Input("f-country-like", "value"),
    Input("f-op", "value"),
    Input("layers", "value"),
    Input("light", "value"),
    Input("store-zones", "data"),
    Input("overlay-refresh-token", "data"),
    # Simulation trigger + params — USE EXISTING IDS FROM LAYOUT
    Input("simulate", "n_clicks"),
    State("sla","value"), State("slo","value"),
    State("ela","value"), State("elo","value"),
    State("batt","value"),  State("si","value"),
    State("sres","value"),  State("stgt","value"),
    State("kwhkm","value"), State("pmax","value"),
    State("show_leg_details","value"),
    State("units","value"),
    # Trigger once on load
    Input("init","n_intervals"),
)
def _update_map(countrys, country_like, op_vals, layers_vals, light_vals, zones_json, _tok,
                sim_clicks, sla, slo, ela, elo, batt, si, sres, stgt, kwhkm, pmax,
                show_leg_details, units, _init_n):


    light = "on" in (light_vals or [])

    # Base dataset + zone join (skip heavy WFS in light mode unless user pressed Compute)
    d = gdf_ev.copy()
    zones_df = None
    if zones_json and zones_json != "[]":
        try:
            zextra = pd.read_json(StringIO(zones_json))
            if {"ROW_ID","ZoneLabel","ZoneColor"}.issubset(zextra.columns) and not zextra.empty:
                zones_df = zextra[["ROW_ID","ZoneLabel","ZoneColor"]]
        except Exception:
            pass
    if zones_df is None:
        if light:
            zones_df = pd.DataFrame({"ROW_ID": gdf_ev["ROW_ID"],
                                     "ZoneLabel": "Outside",
                                     "ZoneColor": ZONE_COLORS["Outside"]})
        else:
            zones_df = compute_model_zones_for_points(gdf_ev)

    d = d.merge(zones_df, on="ROW_ID", how="left")
    d["ZoneLabel"] = d["ZoneLabel"].fillna("Outside")
    d["ZoneColor"] = d["ZoneColor"].fillna(ZONE_COLORS["Outside"])

    # Filters
    if countrys:
        d = d[d['country'].isin(countrys)]
    else:
        d = gdf_ev.copy()  # Always plot all if nothing selected
    if country_like:
        s = str(country_like).strip().lower()
        if s: d = d[d['country'].str.lower().str.contains(s, na=False)]
    op_vals = set(op_vals or [])
    if op_vals and len(op_vals) < 3:
        mask = pd.Series(False, index=d.index)
        if "op" in op_vals:   mask |= d['AvailabilityLabel'].eq("Operational")
        if "down" in op_vals: mask |= d['AvailabilityLabel'].eq("Not operational")
        if "unk" in op_vals:  mask |= d['AvailabilityLabel'].eq("Unknown") | d['AvailabilityLabel'].isna()
        d = d[mask]

    # Overlays
    layers_vals = set(layers_vals or [])
    show_fraw = "fraw" in layers_vals
    show_fmfp = "fmfp" in layers_vals
    show_live = "live" in layers_vals
    show_ctx  = "ctx"  in layers_vals

    itinerary_children = html.Div()
    route_store = {}

    # Route mode
    
    if sim_clicks:
        try:
            # Optional heavy risk union (skip in light mode)
            flood_union_m = None
            if not light:
                bounds = (min(slo,elo), min(sla,ela), max(slo,elo), max(sla,ela))
                flood_union_m = get_flood_union(bounds, include_live=True, include_fraw=True, include_fmfp=True,
                                                pad_m=SIM_DEFAULTS["wfs_pad_m"])

            # ---- Preferred: RCSP over OSM graph (true OSM road line + flood-aware penalties) ----
            if HAS_OSMNX:
                line, safe_lines, risk_lines, stops, total_cost = rcsp_optimize(
                    float(sla), float(slo), float(ela), float(elo),
                    float(batt), float(si), float(sres), float(stgt),
                    float(kwhkm), d if not d.empty else gdf_ev, flood_union_m, extreme=False
                )

                html_str = render_map_html_route(
                    full_line=line, route_safe=safe_lines, route_risk=risk_lines,
                    start=(float(sla), float(slo)), end=(float(ela), float(elo)),
                    chargers=stops, all_chargers_df=None,
                    animate=False, speed_kmh=45, show_live_backdrops=False
                )

                rows = [f"**Routing & charging plan** — RCSP on OSM road network; generalised cost ≈ {total_cost/60:.1f} min"]
                if stops:
                    rows.append("---")
                    for i, st in enumerate(stops, 1):
                        row = gdf_ev.loc[gdf_ev['ROW_ID'].eq(st['ROW_ID'])].iloc[0]
                        rows.append(
                            f"**Stop {i}** — {row.get('Operator','')} ({row.get('country','')}) {row.get('Postcode','')}"
                            f" • Zone: {row.get('ZoneLabel','Outside')} • +{st['energy_kwh']:.1f} kWh"
                            f" to {int(100*st['soc_after'])}% in ~{st['charge_time_min']:.0f} min"
                        )
                itinerary_children = dcc.Markdown("\n\n".join(rows))

                coords_latlng = [{"lat":lat, "lon":lon} for lon,lat in list(line.coords)]
                route_store = dict(
                    start={"lat":float(sla), "lon":float(slo)},
                    end={"lat":float(ela), "lon":float(elo)},
                    route=coords_latlng,
                    stops=[{
                        "lat": gdf_ev.loc[gdf_ev['ROW_ID'].eq(s['ROW_ID'])].iloc[0]['Latitude'],
                        "lon": gdf_ev.loc[gdf_ev['ROW_ID'].eq(s['ROW_ID'])].iloc[0]['Longitude'],
                        "name": gdf_ev.loc[gdf_ev['ROW_ID'].eq(s['ROW_ID'])].iloc[0].get('Operator',"Charger"),
                        "ZoneColor": gdf_ev.loc[gdf_ev['ROW_ID'].eq(s['ROW_ID'])].iloc[0].get('ZoneColor', ZONE_COLORS["Outside"])
                    } for s in stops],
                    created_ts=time.time()
                )
                return html_str, itinerary_children, route_store

            # ---- Fallback: OSRM road polyline + greedy charge suggestions ----
            line, dist_m, dur_s, step_text, src = osrm_route(float(sla), float(slo), float(ela), float(elo))

            stops = osrm_greedy_charge_plan(
                line,
                battery_kwh=float(batt),
                init_soc=float(si),
                reserve_soc=float(sres),
                target_soc=float(stgt),
                kwh_per_km=float(kwhkm),
                chargers_df=d if not d.empty else gdf_ev,
                max_charger_offset_km=float(SIM_DEFAULTS["max_charger_offset_km"])
            )

            safe_lines, risk_lines = segment_route_by_risk(
                line, flood_union_m, buffer_m=ROUTE_BUFFER_M
            )

            html_str = render_map_html_route(
                full_line=line, route_safe=safe_lines, route_risk=risk_lines,
                start=(float(sla), float(slo)), end=(float(ela), float(elo)),
                chargers=stops, all_chargers_df=d,  # show all chargers plus planned ones
                animate=False, speed_kmh=45, show_live_backdrops=False
            )

            msg = [f"**Routing & charging plan** — OSRM ({src}) • {dist_m/1000.0:.1f} km • {dur_s/3600.0:.2f} h"]
            if step_text:
                msg.append("---")
                msg.extend([f"- {t}" for t in step_text[:12]])
            if stops:
                msg.append("---")
                for i, st in enumerate(stops, 1):
                    row = gdf_ev.loc[gdf_ev['ROW_ID'].eq(st['ROW_ID'])].iloc[0]
                    msg.append(
                        f"**Stop {i}** — {row.get('Operator','')} ({row.get('country','')}) {row.get('Postcode','')}"
                        f" • +{st['energy_kwh']:.1f} kWh to {int(100*st['soc_after'])}% in ~{st['charge_time_min']:.0f} min"
                    )
            itinerary_children = dcc.Markdown("\n".join(msg))

            coords_latlng = [{"lat":lat, "lon":lon} for lon,lat in list(line.coords)]
            route_store = dict(
                start={"lat":float(sla), "lon":float(slo)},
                end={"lat":float(ela), "lon":float(elo)},
                route=coords_latlng,
                stops=[{
                    "lat": gdf_ev.loc[gdf_ev['ROW_ID'].eq(s['ROW_ID'])].iloc[0]['Latitude'],
                    "lon": gdf_ev.loc[gdf_ev['ROW_ID'].eq(s['ROW_ID'])].iloc[0]['Longitude'],
                    "name": gdf_ev.loc[gdf_ev['ROW_ID'].eq(s['ROW_ID'])].iloc[0].get('Operator',"Charger"),
                    "ZoneColor": gdf_ev.loc[gdf_ev['ROW_ID'].eq(s['ROW_ID'])].iloc[0].get('ZoneColor', ZONE_COLORS["Outside"])
                } for s in stops],
                created_ts=time.time()
            )

            return html_str, itinerary_children, route_store

        except Exception as e:
            itinerary_children = dcc.Markdown(f"**Routing error:** {e}")
            return render_map_html_ev(d, show_fraw, show_fmfp, show_live, show_ctx, light=light), itinerary_children, {}

    # EV overview
    html_str = render_map_html_ev(d, show_fraw, show_fmfp, show_live, show_ctx, light=light)
    return html_str, itinerary_children, {}
 

# Weather split
@app.callback(
    Output("weather-split", "children"),
    Input("wales-location", "value"),
    Input("wx-refresh", "n_intervals")
)
def _wx_split(loc, _n):
    lat, lon = WALES_LOCS.get(loc, (51.60, -3.20))
    data = get_weather(lat, lon)
    prov = data.get("provider","?")
    raw = data.get("raw") or {}

    left_children = [html.H3(f"{loc} – Current ({prov})")]
    if prov == "Open-Meteo":
        cur = raw.get("current", {})
        if cur:
            left_children += [
                html.Div(f"Temperature: {cur.get('temperature_2m','?')} °C"),
                html.Div(f"Precipitation: {cur.get('precipitation','?')} mm"),
                html.Div(f"Wind: {cur.get('wind_speed_10m','?')} m/s"),
            ]
    elif prov == "Met Office":
        left_children += [html.Pre(json.dumps(raw, indent=2)[:1200])]
    elif prov == "error":
        left_children += [html.Div("Weather error: " + data.get("error",""))]

    left = html.Div(style={"border":"1px solid #eee","borderRadius":"10px","padding":"10px"}, children=left_children)

    try:
        import plotly.graph_objects as go
        if prov == "Open-Meteo":
            hrs = raw.get("hourly", {})
            times = hrs.get("time") or []
            temps = hrs.get("temperature_2m") or []
            pops  = hrs.get("precipitation_probability") or []
            times = times[:24]; temps = temps[:24]; pops = pops[:24]
            df2 = pd.DataFrame({"time": times, "temp": temps, "pop": pops})
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df2["time"], y=df2["pop"], name="Precip %", yaxis="y2", opacity=0.5))
            fig.add_trace(go.Scatter(x=df2["time"], y=df2["temp"], name="Temp °C"))
            fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=320,
                              xaxis_title="", yaxis_title="Temp (°C)",
                              yaxis2=dict(title="Precip (%)", overlaying="y", side="right"))
        elif prov == "Met Office":
            ts = _parse_metoffice_timeseries(raw)
            fig = go.Figure()
            if ts:
                df2 = pd.DataFrame({"time": ts.get("time",[]), "temp": ts.get("temp",[]), "pop": ts.get("pop",[])})
                if any(df2.get("pop", [])):
                    fig.add_trace(go.Bar(x=df2["time"], y=df2["pop"], name="Precip %", yaxis="y2", opacity=0.5))
                fig.add_trace(go.Scatter(x=df2["time"], y=df2["temp"], name="Temp °C"))
                fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=320,
                                  xaxis_title="", yaxis_title="Temp (°C)",
                                  yaxis2=dict(title="Precip (%)", overlaying="y", side="right"))
            else:
                fig.update_layout(title="Met Office: timeseries not found", height=320)
        else:
            import plotly.graph_objects as go
            fig = go.Figure(); fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=320)
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure(); fig.update_layout(title=f"Weather chart error: {e}", height=320)

    right = html.Div(style={"border":"1px solid #eee","borderRadius":"10px","padding":"10px"},
                     children=[html.H3("Next 24h forecast"), dcc.Graph(figure=fig, config={"displayModeBar": False})])

    return [left, right]

# ============ KML Download ============
@app.callback(
    Output("dl-kml", "data"),
    Input("btn-kml", "n_clicks"),
    State("store-route", "data"),
    prevent_initial_call=True
)
def _download_kml(_n, route_data):
    if not route_data or not (route_data.get("route") and route_data.get("start") and route_data.get("end")):
        return dash.no_update
    kml = build_kml(route_data)
    return dict(content=kml, filename="ev_journey.kml", type="application/vnd.google-earth.kml+xml")

def simulate(n, sla, slo, ela, elo, batt, si, sres, stgt, kwhkm, pmax, show_leg_details, units):
    if not HAS_OSMNX:
        return go.Figure(), "OSMnx not available or not installed.", ""
    try:
        coords, stops, cost_min = plan_rcsp_route(
            float(sla), float(slo), float(ela), float(elo),
            EVParams(
                battery_kWh=batt, start_soc=si, reserve_soc=sres,
                target_soc=stgt, kWh_per_km=kwhkm, max_charge_kW=pmax
            ),
            extreme=False
        )
        if not coords or len(coords) < 2:
            return go.Figure(), "No route could be computed. Check start/end.", ""
        center = ((float(sla)+float(ela))/2, (float(slo)+float(elo))/2)
        fig = make_fig_geo(coords, stops, center)
        status = f"Exact RCSP route. Total generalised cost ≈ {cost_min:.1f} min; stops: {len(stops)}."
        # Journey narrative (with per-leg SoC/energy)
        use_miles = (units == "imperial")
        dist_func = (lambda km: km*0.621371) if use_miles else (lambda km: km)
        dist_unit = "mi" if use_miles else "km"
        fmt_dist = lambda d: f"{dist_func(d):.1f} {dist_unit}"
        fmt_kwh = lambda e: f"{e:.1f} kWh"
        fmt_soc = lambda x: f"{int(100*x)}%"

        journey = []
        total_dist = 0.0
        total_energy = 0.0
        journey.append(f"**Start:** ({sla:.4f}, {slo:.4f}), SoC: {fmt_soc(si)}, Battery: {batt:.1f} kWh")
        prev = (sla, slo)
        prev_soc = si

        # pair stops with the path: simple legs start->stop1->...->end
        waypoints = [(s.lat, s.lon, s) for s in stops] + [(ela, elo, None)]
        for i, (lt, ln, s) in enumerate(waypoints, 1):
            leg_dist = haversine_km(prev[0], prev[1], lt, ln)
            leg_energy = leg_dist * kwhkm
            total_dist += leg_dist
            total_energy += leg_energy
            if show_leg_details and "details" in (show_leg_details or []):
                journey.append(
                    f"\n**Leg {i}**: {fmt_dist(leg_dist)} → ({lt:.4f},{ln:.4f})"
                )
            if s is not None:
                pct_added = int(100 * (s.soc_after - s.soc_before))
                journey.append(
                    f"  - **Stop {i}: {s.name}** ({s.lat:.4f},{s.lon:.4f})"
                    f"\n    Postcode: `{s.postcode}`"
                    f"\n    Arrived {fmt_soc(s.soc_before)}; charged {fmt_kwh(s.energy_kWh)} "
                    f"({pct_added}% of battery) in ~{s.charge_time_min:.0f} min; left {fmt_soc(s.soc_after)}."
                )
                prev = (s.lat, s.lon)
                prev_soc = s.soc_after
            else:
                journey.append(f"\n**End:** ({ela:.4f}, {elo:.4f})")

        journey.append(f"\n**Total distance:** {fmt_dist(total_dist)}")
        journey.append(f"**Total drive energy:** {fmt_kwh(total_energy)}")
        journey.append(f"**Total charging time:** {sum(s.charge_time_min for s in stops):.0f} min")
        journey.append(f"**Number of charging stops:** {len(stops)}")

        return fig, status, dcc.Markdown("\n\n".join(journey))
    except Exception as e:
        return go.Figure(), f"Route computation failed: {e}", ""
    
# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run_server(debug=True)

# =========================
# Calibration / Sensitivity
# =========================
def calibrate_risk_penalty(od_pairs, battery_kwh, init_soc, reserve_soc, target_soc, kwh_per_km,
                           chargers_df, flood_union_m, lambda_grid=(0.0,30.0,60.0,120.0,240.0),
                           exposure_weight=1.0, time_weight=1.0, max_seconds=3.0):
    """
    Grid-search calibration for risk penalty λ (seconds/km).
    Scores each λ by a weighted sum of (route_time, risk_exposed_km).
    Returns the λ with minimum average score across OD pairs.
    """
    best_lambda, best_score = None, float("inf")
    for lam in lambda_grid:
        tot_score = 0.0; n=0
        for (sl,so,el,eo) in od_pairs:
            try:
                line, safe, risk, stops, cost = rcsp_optimize(
                    sl, so, el, eo, battery_kwh, init_soc, reserve_soc, target_soc,
                    kwh_per_km, chargers_df, flood_union_m, extreme=False,
                    risk_penalty_per_km=lam, max_seconds=max_seconds
                )
                risk_km = sum(seg.length for seg in risk) * 111.0 if risk else 0.0
                score = time_weight*float(cost)/3600.0 + exposure_weight*risk_km
                tot_score += score; n += 1
            except Exception:
                continue
        if n>0 and tot_score/n < best_score:
            best_score, best_lambda = tot_score/n, lam
    return best_lambda, best_score

def sensitivity_on_params(od_pairs, ev_params, chargers_df, flood_union_m, lambdas=(0.0,60.0,240.0),
                          reserves=(0.1,0.2), kwhs=(0.15,0.18,0.22)):
    """Run a coarse sensitivity sweep over key parameters. Returns a list of results rows."""
    rows=[]
    for lam in lambdas:
        for rs in reserves:
            for k in kwhs:
                succ=0; avg_time=0.0; avg_risk=0.0; cnt=0
                for (sl,so,el,eo) in od_pairs:
                    try:
                        line, safe, risk, stops, cost = rcsp_optimize(
                            sl, so, el, eo,
                            ev_params.battery_kWh, ev_params.start_soc*100, rs*100, ev_params.target_soc*100,
                            k, chargers_df, flood_union_m, extreme=False, risk_penalty_per_km=lam, max_seconds=3.0
                        )
                        risk_km = sum(seg.length for seg in risk) * 111.0 if risk else 0.0
                        succ+=1; avg_time+=cost; avg_risk+=risk_km; cnt+=1
                    except Exception:
                        pass
                if cnt>0:
                    rows.append(dict(lambda_sec_per_km=lam, reserve=rs, kwh_per_km=k,
                                     success=succ, avg_time_s=avg_time/max(1,cnt), avg_risk_km=avg_risk/max(1,cnt)))
    return rows
