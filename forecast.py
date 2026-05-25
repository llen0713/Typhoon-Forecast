import os
import io
import requests # type: ignore
import pandas as pd # type: ignore
from datetime import datetime, timezone, timedelta
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.lines as mlines # type: ignore
import matplotlib.ticker as mticker # type: ignore
import re

try:
    from PIL import Image # type: ignore
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d as _scipy_interp1d  # type: ignore
    from scipy.ndimage import gaussian_filter1d as _scipy_gf1d  # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy 未安裝，無法繪製不確定性圓錐")

try:
    from shapely.geometry import Point as _ShapelyPoint        # type: ignore
    from shapely.ops import unary_union as _shapely_union      # type: ignore
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

# Target timestamp — auto-derived from current UTC time.
# FNV3 data is available ~6h45m after cycle time, so we subtract that
# and round down to the nearest 6-hour boundary (00/06/12/18Z).
def _latest_cycle() -> datetime:
    adjusted = datetime.now(timezone.utc) - timedelta(hours=6, minutes=45)
    cycle_hour = (adjusted.hour // 6) * 6
    return adjusted.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

_cycle = _latest_cycle()
YEAR   = _cycle.strftime("%Y")
MONTH  = _cycle.strftime("%m")
DAY    = _cycle.strftime("%d")
HOUR   = _cycle.strftime("%H")
MINUTE = "00"

# Output directories
ENSEMBLE_DIR = "deepmind_weather_downloads_2026"
MEAN_DIR = "deepmind_weather_ensemble_mean_downloads_2026"
CYCLOGENESIS_DIR = "deepmind_weather_cyclogenesis_2026"

os.makedirs(ENSEMBLE_DIR, exist_ok=True)
os.makedirs(MEAN_DIR, exist_ok=True)
os.makedirs(CYCLOGENESIS_DIR, exist_ok=True)

# Base URLs
ENSEMBLE_URL = (
    "https://deepmind.google.com/science/weatherlab/download/cyclones/FNV3/ensemble/paired/csv/FNV3_{year}_{month}_{day}T{hour}_{minute}_paired.csv"
)
ENSEMBLE_MEAN_URL = (
    "https://deepmind.google.com/science/weatherlab/download/cyclones/FNV3/ensemble_mean/paired/csv/FNV3_{year}_{month}_{day}T{hour}_{minute}_paired.csv"
)
CYCLOGENESIS_URL = (
    "https://deepmind.google.com/science/weatherlab/download/cyclones/FNV3/ensemble/cyclogenesis/csv/FNV3_{year}_{month}_{day}T{hour}_{minute}_cyclogenesis.csv"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CopilotDownloader/1.0)"}

# 輸出目錄
OUTPUT_DIR = "docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILENAME = f"FNV3_{YEAR}_{MONTH}_{DAY}T{HOUR}_{MINUTE}_paired.csv"
CSV_PATH = os.path.join(ENSEMBLE_DIR, FILENAME)
CYCLOGENESIS_FILENAME = f"FNV3_{YEAR}_{MONTH}_{DAY}T{HOUR}_{MINUTE}_cyclogenesis.csv"
CYCLOGENESIS_CSV_PATH = os.path.join(CYCLOGENESIS_DIR, CYCLOGENESIS_FILENAME)


def _auto_detect_track_ids(mean_dir: str) -> list[str]:
    """從 MEAN_DIR 內最新的 CSV 自動偵測有效颱風 TRACK_ID。
    只保留 WP[0-8]X20XX，排除 WP9X（擾動）。
    """
    if not os.path.isdir(mean_dir):
        return []
    csvs = sorted(
        [os.path.join(mean_dir, f) for f in os.listdir(mean_dir)
         if f.startswith("FNV3_") and f.endswith(".csv")],
        key=os.path.getmtime,
        reverse=True,
    )
    if not csvs:
        print("[AUTO-DETECT] MEAN_DIR 內無可用 CSV")
        return []
    latest = csvs[0]
    print(f"[AUTO-DETECT] 讀取: {os.path.basename(latest)}")
    try:
        with open(latest, 'r', encoding='utf-8') as f:
            content = ''.join(line for line in f if not line.startswith('#'))
        df = pd.read_csv(io.StringIO(content))
        col = next((c for c in df.columns if c.lower() == 'track_id'), None)
        if col is None:
            print("[AUTO-DETECT] 找不到 track_id 欄位")
            return []
        ids = df[col].dropna().astype(str).unique().tolist()
        valid = sorted(tid for tid in ids if re.match(r'^WP[0-8]\d20\d{2}$', tid))
        print(f"[AUTO-DETECT] 偵測到颱風: {valid}")
        return valid
    except Exception as e:
        print(f"[AUTO-DETECT] 讀取失敗: {e}")
        return []


def _jtwc_url_key(track_id: str) -> str:
    """WP042026 → 'wp0426'"""
    m = re.match(r'^WP(\d{2})(\d{4})$', track_id)
    if not m:
        return track_id.lower()
    return f"wp{m.group(1)}{m.group(2)[2:]}"


def _build_jtwc_urls(track_ids: list[str]) -> tuple[dict, dict]:
    base = "https://www.metoc.navy.mil/jtwc/products"
    forecast, text = {}, {}
    for tid in track_ids:
        key = _jtwc_url_key(tid)
        forecast[tid] = f"{base}/{key}.gif"
        text[tid] = f"{base}/{key}web.txt"
    return forecast, text

# 可選底圖：Cartopy
try:
    import cartopy.crs as ccrs # Type: ignore
    import cartopy.feature as cfeature # Type: ignore
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER # type: ignore
    HAS_CARTOPY = True
    print("[INFO] Cartopy 已載入")
except ImportError:
    HAS_CARTOPY = False
    print("[WARN] 未安裝 Cartopy，將使用簡易經緯度圖")


def _pick_tick_step(span: float) -> float:
    if span <= 20:
        return 2
    if span <= 40:
        return 5
    if span <= 80:
        return 10
    return 20


def _configure_cartopy_gridlines(gl, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    x_step = _pick_tick_step(abs(lon_max - lon_min))
    y_step = _pick_tick_step(abs(lat_max - lat_min))

    gl.xlocator = mticker.MultipleLocator(x_step)
    gl.ylocator = mticker.MultipleLocator(y_step)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}


def _fit_extent_to_aspect(extent, target_ar: float, use_360: bool):
    min_lon, max_lon, min_lat, max_lat = extent
    w, h = max_lon - min_lon, max_lat - min_lat
    if h <= 0:
        h = 1e-6

    if w / h > target_ar:
        extra = (w / target_ar - h) / 2
        min_lat -= extra
        max_lat += extra
    else:
        extra = (target_ar * h - w) / 2
        min_lon -= extra
        max_lon += extra

    # 比例調整後再夾制邊界，避免超出經度範圍造成 Cartopy 版面異常
    if use_360:
        min_lon = max(0.0, min_lon)
        max_lon = min(360.0, max_lon)
    else:
        min_lon = max(-180.0, min_lon)
        max_lon = min(180.0, max_lon)
    min_lat = max(-90.0, min_lat)
    max_lat = min(90.0, max_lat)

    return (min_lon, max_lon, min_lat, max_lat)

# 一致的顏色與尺寸設定
COLOR_MAP = {
    'TD': '#CCCCCC', 'TS': '#00FFFF', 'Cat1': '#00FF00', 'Cat2': '#FFFF00',
    'Cat3': '#FFA500', 'Cat4': '#FF0000', 'Cat5': '#800080', 'Unknown': 'gray'
}

# MSLP 色彩分級（對應西太平洋潛勢預報）
MSLP_COLOR_BINS = [
    (935,         '#CC0000', '≤935 hPa'),
    (955,         '#FF6600', '936–955 hPa'),
    (978,         '#FFB800', '956–978 hPa'),
    (988,         '#FFD966', '979–988 hPa'),
    (1000,        '#4488FF', '989–1000 hPa'),
    (float('inf'),'#AACCFF', '>1000 hPa'),
]

def _mslp_to_color(mslp: float) -> tuple[str, bool]:
    """返回 (顏色, 是否填實) — 全段均為實心"""
    if pd.isna(mslp):
        return '#AACCFF', True
    for threshold, color, _ in MSLP_COLOR_BINS:
        if mslp <= threshold:
            return color, True
    return '#AACCFF', True

FIG_AR = 1.40
FIG_H = 8
FIG_W = FIG_AR * FIG_H
FIG_DPI = 300

# 目標颱風 Track ID — 從最新 CSV 自動偵測，無需手動修改
TARGET_TRACK_IDS = _auto_detect_track_ids(MEAN_DIR)
JTWC_FORECAST_URLS, JTWC_TEXT_URLS = _build_jtwc_urls(TARGET_TRACK_IDS)


def download_jtwc_image(track_id: str, output_dir: str = "Typhoon_Analysis_Forecast") -> str:
    """下載 JTWC 預報圖
    
    Args:
        track_id: 風暴追蹤 ID
        output_dir: 輸出目錄
        
    Returns:
        下載的圖片檔案路徑，如果失敗則返回 None
    """
    if track_id not in JTWC_FORECAST_URLS:
        print(f"[JTWC] 無對應的 JTWC URL: {track_id}")
        return None
    
    url = JTWC_FORECAST_URLS[track_id]
    output_path = os.path.join(output_dir, f"jtwc_{track_id}.gif")
    
    try:
        print(f"[JTWC] 正在下載: {url}")
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"[JTWC] 下載完成: {output_path}")
        return output_path
    except Exception as e:
        print(f"[JTWC] 下載失敗: {e}")
        return None


def ss_category(kt):
    if pd.isna(kt):
        return 'Unknown'
    try:
        kt = float(kt)
    except Exception:
        return 'Unknown'
    if kt < 34: return 'TD'
    elif kt < 64: return 'TS'
    elif kt < 83: return 'Cat1'
    elif kt < 96: return 'Cat2'
    elif kt < 113: return 'Cat3'
    elif kt < 137: return 'Cat4'
    else: return 'Cat5'


def get_24h_markers(df, init_time, max_hour=360):
    if df.empty or init_time is None:
        return pd.DataFrame()
    df = df.copy()
    df['fh'] = (df['valid_time'] - init_time).dt.total_seconds() / 3600.0
    targets = np.arange(0, max_hour + 1, 24)
    result = pd.DataFrame()
    for t in targets:
        match = df[abs(df['fh'] - t) < 1.5]
        if not match.empty:
            best = match.loc[abs(match['fh'] - t).idxmin()]
            result = pd.concat([result, best.to_frame().T])
    return result


def get_6h_markers(df, init_time, max_hour=360):
    if df.empty or init_time is None:
        return pd.DataFrame()
    df = df.copy()
    df['fh'] = (df['valid_time'] - init_time).dt.total_seconds() / 3600.0
    targets = np.arange(0, max_hour + 1, 6)
    result = pd.DataFrame()
    for t in targets:
        match = df[abs(df['fh'] - t) < 1.0]
        if not match.empty:
            best = match.loc[abs(match['fh'] - t).idxmin()]
            result = pd.concat([result, best.to_frame().T])
    return result


def build_24h_summary(pts: pd.DataFrame, init_time: pd.Timestamp) -> str:
    """組合 24 小時標記的摘要字串，顯示在右下角。
    內容格式：+24h: 80kt (Cat1)
    """
    if pts is None or pts.empty:
        return ""
    lines = []
    for _, pt in pts.iterrows():
        wind = pt.get('wind', np.nan)
        cat = ss_category(wind)
        fh = (pt['valid_time'] - init_time).total_seconds() / 3600.0
        wind_str = f"{int(wind)}kt" if not pd.isna(wind) else "N/A"
        lines.append(f"+{int(fh)}h: {wind_str} ({cat})")
    return "\n".join(lines)


def _format_intensity_label(wind) -> str:
    """Format wind intensity for compact point annotations."""
    if pd.isna(wind):
        return "N/A"
    try:
        w = float(wind)
    except Exception:
        return "N/A"
    return f"{int(round(w))}kt ({ss_category(w)})"


def _detect_dateline_crossing(lon_vals):
    """檢測是否跨越國際換日線"""
    lon_vals = np.array(lon_vals)
    lon_vals = lon_vals[~np.isnan(lon_vals)]
    if len(lon_vals) < 2:
        return False
    lon_sorted = np.sort(lon_vals)
    lon_diffs = np.diff(lon_sorted)
    return np.max(lon_diffs) > 180


def _normalize_lon_values(lon_vals, use_360: bool):
    """將經度正規化到 [-180, 180) 或 [0, 360) 以利跨換日線繪圖。"""
    lons = np.asarray(lon_vals, dtype=float)
    lons = ((lons + 180.0) % 360.0) - 180.0
    if use_360:
        lons = lons % 360.0
    return lons


def _split_track_segments(lons, lats, jump_threshold: float = 180.0):
    """在經度跳躍過大處切段，避免軌跡跨整張圖連線。"""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    valid = (~np.isnan(lons)) & (~np.isnan(lats))
    lons = lons[valid]
    lats = lats[valid]
    if len(lons) == 0:
        return []

    split_idx = np.where(np.abs(np.diff(lons)) > jump_threshold)[0] + 1
    lon_segments = np.split(lons, split_idx)
    lat_segments = np.split(lats, split_idx)
    return [(seg_lon, seg_lat) for seg_lon, seg_lat in zip(lon_segments, lat_segments) if len(seg_lon) > 0]

def _auto_extent(lat_vals, lon_vals, pad_deg=3.0):
    """計算地圖範圍，自動選擇較緊湊的經度表達方式。"""
    lon_vals = np.asarray(lon_vals, dtype=float)
    lat_vals = np.asarray(lat_vals, dtype=float)
    lon_vals = lon_vals[~np.isnan(lon_vals)]
    lat_vals = lat_vals[~np.isnan(lat_vals)]
    
    if len(lon_vals) == 0 or len(lat_vals) == 0:
        return (-180, 180, -90, 90), False
    
    lat_min, lat_max = np.min(lat_vals), np.max(lat_vals)

    lons_180 = _normalize_lon_values(lon_vals, use_360=False)
    span_180 = np.max(lons_180) - np.min(lons_180)

    lons_360 = _normalize_lon_values(lon_vals, use_360=True)
    span_360 = np.max(lons_360) - np.min(lons_360)

    use_360 = span_360 < span_180
    if use_360:
        lon_min = np.min(lons_360) - pad_deg
        lon_max = np.max(lons_360) + pad_deg
        lon_min = max(lon_min, 0)
        lon_max = min(lon_max, 360)
    else:
        lon_min = np.min(lons_180) - pad_deg
        lon_max = np.max(lons_180) + pad_deg
        lon_min = max(lon_min, -180)
        lon_max = min(lon_max, 180)

    return (lon_min, lon_max, lat_min - pad_deg, lat_max + pad_deg), use_360


def load_forecast_dataframe(csv_path: str, track_id: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """讀取 paired CSV 並過濾指定 track_id，僅保留必要欄位。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到資料檔：{csv_path}")
    df = pd.read_csv(csv_path, comment="#")
    # 標準欄位存在性
    for col in ["track_id", "sample", "valid_time", "lat", "lon"]:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要欄位：{col}")
    # 欄位修正: 一致使用 wind 欄位（若存在）
    if 'maximum_sustained_wind_speed_knots' in df.columns:
        df = df.rename(columns={'maximum_sustained_wind_speed_knots': 'wind'})

    sub = df[df["track_id"].astype(str) == str(track_id)].copy()
    if sub.empty:
        raise ValueError(f"CSV 中未找到 track_id={track_id} 的資料")
    # 轉型與排序
    sub["valid_time"] = pd.to_datetime(sub["valid_time"], errors="coerce", utc=True)
    # 取初始化時間（欄位存在時）
    init_time = None
    if 'init_time' in sub.columns:
        try:
            init_time = pd.to_datetime(sub['init_time'].iloc[0], utc=True)
        except Exception:
            init_time = None
    sub["sample"] = sub["sample"].astype(float)
    sub = sub.dropna(subset=["valid_time", "lat", "lon"]).sort_values(["sample", "valid_time"]).reset_index(drop=True)
    # 若沒有 init_time 欄位則以第一筆 valid_time 當作初始（近似）
    if init_time is None and not sub.empty:
        init_time = sub['valid_time'].min()
    return sub, init_time

def scrape_jtwc_text_product(track_id: str) -> dict:

    url = JTWC_TEXT_URLS.get(track_id)
    if not url:
        print(f"[JTWC] 無對應 web.txt URL: {track_id}")
        return {}
    try:
        print(f"[JTWC] 下載 web.txt: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text

        info: dict[str, object] = {}

        # 名稱（支援 SUBJ/TYPHOON、1. TYPHOON、TROPICAL STORM/CYCLONE 等格式）
        name_patterns = [
            r"SUBJ/\s*(?:SUPER\s+)?(?:TROPICAL\s+)?(?:DEPRESSION|STORM|CYCLONE|TYPHOON)\s+\d+[A-Z]?\s*\(([A-Z\-]+)\)",
            r"\b(?:SUPER\s+)?(?:TROPICAL\s+)?(?:DEPRESSION|STORM|CYCLONE|TYPHOON)\s+\d+[A-Z]?\s*\(([A-Z\-]+)\)",
        ]
        for pat in name_patterns:
            m_name = re.search(pat, text, re.IGNORECASE)
            if m_name:
                info['name'] = m_name.group(1).upper()
                break

        # 時間，抓 Z 時戳
        m_time = re.search(r"\b(\d{6})Z\b", text)
        if m_time:
            info['update_time'] = m_time.group(1) + " Z"

        # 位置（支援 "POSITION NEAR ..."、"--- NEAR ..." 等格式）
        m_pos = re.search(
            r"(?:POSITION\s+NEAR|---\s+NEAR|NEAR)\s+([0-9.]+)([NS])\s+([0-9.]+)([EW])",
            text,
            re.IGNORECASE,
        )
        if m_pos:
            lat_val = float(m_pos.group(1))
            lat_hem = m_pos.group(2).upper()
            lon_val = float(m_pos.group(3))
            lon_hem = m_pos.group(4).upper()
            info['latitude'] = f"{lat_val:.1f}°{lat_hem}"
            info['longitude'] = f"{lon_val:.1f}°{lon_hem}"

        # 最大風速
        m_wind = re.search(r"MAX\s+SUSTAINED\s+WINDS\s*[:\-]?\s*(\d+)\s*KT", text, re.IGNORECASE)
        if m_wind:
            info['max_winds_kt'] = int(m_wind.group(1))

        # Gusts
        m_gusts = re.search(r"GUSTS\s+(?:TO\s+)?(\d+)\s*KT", text, re.IGNORECASE)
        if m_gusts:
            info['gusts'] = int(m_gusts.group(1))

        # 34/50/64 KT 風圈: RADIUS OF XXX KT WINDS - NE/SE/SW/NW quadrants
        def _parse_radii(block: str) -> dict:
            res = {}
            ne_match = re.search(r"(\d+)\s*NM\s+NORTHEAST", block, re.IGNORECASE)
            if ne_match:
                res['NE'] = ne_match.group(1)
            se_match = re.search(r"(\d+)\s*NM\s+SOUTHEAST", block, re.IGNORECASE)
            if se_match:
                res['SE'] = se_match.group(1)
            sw_match = re.search(r"(\d+)\s*NM\s+SOUTHWEST", block, re.IGNORECASE)
            if sw_match:
                res['SW'] = sw_match.group(1)
            nw_match = re.search(r"(\d+)\s*NM\s+NORTHWEST", block, re.IGNORECASE)
            if nw_match:
                res['NW'] = nw_match.group(1)
            return res

        for kt in (34, 50, 64):
            radii_pattern = rf"RADIUS\s+OF\s+0?{kt}\s+KT\s+WINDS\s*-?\s*(.*?)(?:\n\n|$)"
            m_block = re.search(radii_pattern, text, re.IGNORECASE | re.DOTALL)
            if m_block:
                radii = _parse_radii(m_block.group(1))
                if radii:
                    info[f'wind_radii_{kt}kt'] = radii

        if info:
            print(f"[JTWC] 解析成功: {info}")
        else:
            print("[JTWC] 警告: 未能從 web.txt 解析資料")
        return info

    except Exception as e:
        print(f"[JTWC] 錯誤: {e}")
        return {}

def extract_current_info(df: pd.DataFrame, target_time: pd.Timestamp) -> dict:
    """Extract current condition from first (earliest) forecast row."""
    if df.empty:
        return {}
    df = df.copy().sort_values('valid_time').reset_index(drop=True)
    row = df.iloc[0]
    wind = row.get('wind', np.nan)
    pressure = row.get('minimum_sea_level_pressure_hpa', np.nan)
    rmw = row.get('radius_of_maximum_winds_km', np.nan)
    r34_ne = row.get('radius_34_knot_winds_ne_km', np.nan)
    r34_se = row.get('radius_34_knot_winds_se_km', np.nan)
    r34_sw = row.get('radius_34_knot_winds_sw_km', np.nan)
    r34_nw = row.get('radius_34_knot_winds_nw_km', np.nan)
    
    return {
        'valid_time': row.get('valid_time'),
        'lat': row.get('lat', np.nan),
        'lon': row.get('lon', np.nan),
        'wind': wind,
        'category': ss_category(wind),
        'pressure': pressure,
        'rmw': rmw,
        'r34_ne': r34_ne,
        'r34_se': r34_se,
        'r34_sw': r34_sw,
        'r34_nw': r34_nw,
    }

def generate_frame_sequence(df: pd.DataFrame, mean_df: pd.DataFrame, init_time: pd.Timestamp, track_id: str, output_dir: str, max_frames: int = 72) -> list:
    """生成預報軌跡演變的幀序列，用於網頁動畫生成。與靜態地圖保持一致的風格。
    
    Returns:
        包含所有生成幀的文件路徑列表
    """
    # 計算時間步
    all_times = sorted(pd.concat([df, mean_df])['valid_time'].unique())
    if len(all_times) > max_frames:
        step = len(all_times) // max_frames
        all_times = all_times[::step]
    
    print(f"[FRAME] 正在生成 {len(all_times)} 幀序列...")

    # 計算地圖範圍（處理國際換日線），並套用與靜態圖一致的比例
    all_lons = list(df['lon']) + list(mean_df['lon'])
    all_lats = list(df['lat']) + list(mean_df['lat'])
    extent, use_360 = _auto_extent(all_lats, all_lons, pad_deg=3)
    min_lon, max_lon, min_lat, max_lat = extent
    if min_lat == max_lat:
        min_lat -= 2
        max_lat += 2
    if min_lon == max_lon:
        min_lon -= 2
        max_lon += 2

    target_ar = 1.40
    extent = _fit_extent_to_aspect((min_lon, max_lon, min_lat, max_lat), target_ar, use_360)

    frame_paths = []
    os.makedirs(output_dir, exist_ok=True)

    # 起始位置（全序列共用）
    init_pt = mean_df.sort_values('valid_time').iloc[0]
    init_lon_star = _normalize_lon_values([init_pt['lon']], use_360=use_360)[0]
    init_lat_star = float(init_pt['lat'])

    for frame_idx, current_time in enumerate(all_times):
        # ── 建立畫布 ──────────────────────────────────────────────────────────
        if HAS_CARTOPY:
            fig = plt.figure(figsize=(10, 7))
            if use_360:
                ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
            else:
                ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines(resolution='50m', linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.8)
            ax.add_feature(cfeature.LAND, facecolor='#f0e8d4', alpha=0.9)
            ax.add_feature(cfeature.OCEAN, facecolor='#cce8f4', alpha=0.9)
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
            gl.right_labels = False
            gl.top_labels = False
            _configure_cartopy_gridlines(gl, extent)
            kw = dict(transform=ccrs.PlateCarree())
        else:
            fig = plt.figure(figsize=(10, 7), dpi=100, facecolor='white')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#cce8f4')
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.grid(True, linewidth=0.5, alpha=0.4, linestyle='--')
            ax.set_xlabel("Longitude", fontsize=10)
            ax.set_ylabel("Latitude", fontsize=10)
            kw = {}

        # ── 不確定性圓錐（截至 current_time）──────────────────────────────────
        # Pass full df so circle radii are computed from the complete ensemble
        # (consistent across frames), and limit drawing via max_fh.
        fh_now = (pd.to_datetime(current_time) - init_time).total_seconds() / 3600.0
        cone_stop_fh = _compute_cone_stop_fh(df, init_time, max_fh=fh_now)
        _draw_uncertainty_cone(ax, df, mean_df, init_time, use_360, kw, max_fh=fh_now)

        # ── 起始位置星形標記 ──────────────────────────────────────────────────
        ax.scatter([init_lon_star], [init_lat_star], marker='*', color='gold', s=200,
                   ec='darkorange', zorder=7, linewidth=0.8, **kw)

        # ── 集合成員軌跡 ──────────────────────────────────────────────────────
        for sid, g in df.groupby("sample"):
            g = g[g['valid_time'] <= current_time].sort_values('valid_time')
            if not g.empty:
                lons = _normalize_lon_values(g["lon"].to_numpy(), use_360=use_360)
                lats = g["lat"].to_numpy()
                for seg_lon, seg_lat in _split_track_segments(lons, lats):
                    ax.plot(seg_lon, seg_lat, color='gray', linewidth=0.6, alpha=0.6, zorder=0.9, **kw)
                last_pt = g.iloc[-1]
                wind = last_pt.get('wind', np.nan)
                cat = ss_category(wind)
                last_lon = _normalize_lon_values([last_pt['lon']], use_360=use_360)[0]
                ax.scatter(last_lon, last_pt['lat'], color=COLOR_MAP.get(cat, 'gray'),
                           s=15, marker='o', edgecolor='darkgray', alpha=0.8, zorder=1.1, linewidth=0.5, **kw)

        # ── 平均軌跡 + 24h 標記 ───────────────────────────────────────────────
        mean_subset = mean_df[mean_df['valid_time'] <= current_time].sort_values('valid_time')
        if cone_stop_fh is not None:
            cone_stop_time = init_time + pd.to_timedelta(cone_stop_fh, unit='h')
            mean_subset = mean_subset[mean_subset['valid_time'] <= cone_stop_time]
        if not mean_subset.empty:
            mean_lons = _normalize_lon_values(mean_subset["lon"].to_numpy(), use_360=use_360)
            mean_lats = mean_subset["lat"].to_numpy()
            for seg_lon, seg_lat in _split_track_segments(mean_lons, mean_lats):
                ax.plot(seg_lon, seg_lat, 'r-', lw=2.5, zorder=4, **kw)
            last_mean_pt = mean_subset.iloc[-1]
            last_mean_lon = _normalize_lon_values([last_mean_pt['lon']], use_360=use_360)[0]
            last_fh = int(round((last_mean_pt['valid_time'] - init_time).total_seconds() / 3600.0))
            last_intensity = _format_intensity_label(last_mean_pt.get('wind', np.nan))
            ax.scatter([last_mean_lon], [last_mean_pt['lat']], marker='s', color='red',
                       s=40, ec='black', zorder=5, linewidth=1, **kw)
            ax.text(last_mean_lon + 0.4, last_mean_pt['lat'] + 0.4, f'+{last_fh}h\n{last_intensity}',
                    fontsize=6, color='darkred', fontweight='bold', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='red', alpha=0.90, linewidth=0.6),
                    clip_on=True, **kw)

            pts_24h_mean = get_24h_markers(mean_subset, init_time)
            if not pts_24h_mean.empty:
                marker_lons = _normalize_lon_values(pts_24h_mean['lon'].to_numpy(), use_360=use_360)
                ax.scatter(marker_lons, pts_24h_mean['lat'], marker='s', color='red', s=30, ec='black', zorder=5, **kw)
                # +Nh 時間標籤
                for i, (_, pt) in enumerate(pts_24h_mean.iterrows()):
                    fh = int((pt['valid_time'] - init_time).total_seconds() / 3600)
                    if fh == 0 or fh == last_fh:
                        continue
                    ax.text(marker_lons[i] + 0.4, pt['lat'] + 0.4, f'+{fh}h',
                            fontsize=6, color='darkred', fontweight='bold', zorder=6,
                            clip_on=True, **kw)
                summary = build_24h_summary(pts_24h_mean, init_time)
                if summary:
                    n_lines = len(summary.splitlines())
                    y_anchor = 0.27 + min(n_lines, 8) * 0.015
                    ax.text(0.985, y_anchor, summary, transform=ax.transAxes, fontsize=6, ha='right', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.85, linewidth=0.6),
                            zorder=6)

        # ── 圖例 ──────────────────────────────────────────────────────────────
        handles = [
            mlines.Line2D([], [], color='gray', label='Ensemble Members', lw=1),
            mlines.Line2D([], [], color='red', marker='s', markeredgecolor='black', label='FNV3 Mean', lw=2),
            mlines.Line2D([], [], color='#88CC78', lw=4, alpha=0.55, label='Uncertainty Cone'),
            mlines.Line2D([], [], color='gold', marker='*', ms=9, ls='', markeredgecolor='darkorange', label='Init Position'),
        ]
        l1 = ax.legend(handles=handles, loc='upper left', title='Track Source', fontsize=8,
                       frameon=True, framealpha=0.92, borderpad=0.6, labelspacing=0.3,
                       handlelength=0.9, handletextpad=0.7)
        l1.get_title().set_multialignment('center')
        l1.get_title().set_ha('center')
        ax.add_artist(l1)

        int_handles = []
        for cat in ['TD', 'Cat2', 'Cat5', 'TS', 'Cat3', 'Cat1', 'Cat4']:
            int_handles.append(mlines.Line2D([], [], color=COLOR_MAP[cat], marker='o', ms=6, ls='',
                                             markeredgecolor='black', label=cat))
        int_leg = ax.legend(handles=int_handles, loc='lower right', bbox_to_anchor=(0.995, 0.005),
                            title='Intensity', fontsize=7, frameon=True, framealpha=0.92, ncol=3,
                            borderpad=0.6, labelspacing=0.25, handlelength=0.75, handletextpad=0.35,
                            markerscale=0.7, borderaxespad=0.5)
        int_leg.get_title().set_multialignment('center')
        int_leg.get_title().set_ha('center')
        ax.add_artist(int_leg)

        # ── 標題 ──────────────────────────────────────────────────────────────
        time_str = pd.to_datetime(current_time).strftime('%Y-%m-%d %H:%M UTC')
        ax.set_title(
            f"FNV3 {track_id} Track Forecast  (Init: {pd.to_datetime(init_time).strftime('%Y-%m-%d %H:%M UTC')})\n{time_str}",
            fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout(pad=0.6)
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.12)
        frame_paths.append(frame_path)
        plt.close(fig)

        print(f"[FRAME] 已生成第 {frame_idx + 1}/{len(all_times)} 幀")

    return frame_paths


def create_gif_from_frames(frame_paths: list[str], gif_path: str, duration_ms: int = 180, loop: int = 0) -> str:
    """將幀序列輸出為 GIF 動圖。"""
    if not frame_paths:
        print("[GIF] 無可用幀，略過 GIF 生成")
        return None
    if not HAS_PIL:
        print("[GIF] 未安裝 Pillow，略過 GIF 生成")
        return None

    sorted_paths = sorted(frame_paths)
    images = []
    try:
        for p in sorted_paths:
            with Image.open(p) as im:
                images.append(im.convert("P", palette=Image.ADAPTIVE).copy())

        first, rest = images[0], images[1:]
        first.save(
            gif_path,
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=loop,
            optimize=False,
            disposal=2,
        )
        print(f"[GIF] 已輸出動圖：{gif_path}")
        return gif_path
    except Exception as e:
        print(f"[GIF] 生成失敗: {e}")
        return None


def _compute_cone_stop_fh(df: pd.DataFrame, init_time: pd.Timestamp, max_fh: float = None) -> float | None:
    """Compute the effective cone stop hour used by both cone and mean-track display."""
    if df.empty or init_time is None:
        return None

    work = df.copy()
    work['fh'] = (work['valid_time'] - init_time).dt.total_seconds() / 3600.0
    data_max_fh = float(work['fh'].max())
    if data_max_fh < 12:
        return None

    if max_fh is None:
        max_fh = data_max_fh
    else:
        max_fh = min(float(max_fh), data_max_fh)
    if max_fh < 12:
        return None

    # If max_fh lands on 12h but not 24h, move back to the previous 24h boundary.
    # Example: 36h -> 24h, so cone ending aligns with 24h cadence.
    cone_end_fh = float(max_fh)
    rem12 = np.mod(cone_end_fh, 12.0)
    rem24 = np.mod(cone_end_fh, 24.0)
    if np.isclose(rem12, 0.0, atol=1e-6) and (not np.isclose(rem24, 0.0, atol=1e-6)):
        cone_end_fh = max(0.0, cone_end_fh - 12.0)

    last_ok_t = None
    for t in np.arange(0, cone_end_fh + 1, 12):
        sub = work[abs(work['fh'] - t) < 3.5]
        if len(sub) < 25:
            break
        if sub.empty:
            continue
        last_ok_t = float(t)

    return last_ok_t




def _draw_uncertainty_cone(ax, df: pd.DataFrame, mean_df: pd.DataFrame,
                           init_time: pd.Timestamp, use_360: bool, kw: dict,
                           max_fh: float = None):
    """CWA-style uncertainty cone using the true geometric union of circles.

    Approach: build one circle per 12-h step (90th-pct ensemble spread,
    non-decreasing radius), then take their Shapely unary_union.  The union
    boundary is by construction non-self-intersecting and monotonically grows
    as more circles are added — no analytical envelope needed.

    max_fh : limit drawing to this forecast hour (animation mode); circles are
             always computed from the full df so radii stay consistent across frames.
    """
    if not HAS_SCIPY or df.empty or mean_df.empty or init_time is None:
        return

    cone_stop_fh = _compute_cone_stop_fh(df, init_time, max_fh=max_fh)
    if cone_stop_fh is None:
        return

    work = df.copy()
    work['fh'] = (work['valid_time'] - init_time).dt.total_seconds() / 3600.0

    mean_s = mean_df.sort_values('valid_time').copy()
    mean_s['fh'] = (mean_s['valid_time'] - init_time).dt.total_seconds() / 3600.0
    mean_s['lon_n'] = _normalize_lon_values(mean_s['lon'].to_numpy(), use_360=use_360)

    # ── 1. Discrete circles at 12-h steps ────────────────────────────────────
    t_kn, cx_kn, cy_kn, r_kn = [], [], [], []
    for t in np.arange(0, cone_stop_fh + 1, 12):
        sub = work[abs(work['fh'] - t) < 3.5]
        if sub.empty:
            continue
        lons = _normalize_lon_values(sub['lon'].to_numpy(), use_360=use_360)
        lats = sub['lat'].to_numpy()
        mt   = mean_s[abs(mean_s['fh'] - t) < 3.5]
        cx   = float(mt.iloc[0]['lon_n']) if not mt.empty else float(np.nanmean(lons))
        cy   = float(mt.iloc[0]['lat'])   if not mt.empty else float(np.nanmean(lats))
        r    = float(np.percentile(np.sqrt((lons - cx)**2 + (lats - cy)**2), 90))
        t_kn.append(t); cx_kn.append(cx); cy_kn.append(cy); r_kn.append(r)

    if len(t_kn) < 2:
        return

    t_arr  = np.array(t_kn,  dtype=float)
    cx_arr = np.array(cx_kn, dtype=float)
    cy_arr = np.array(cy_kn, dtype=float)
    r_arr  = np.array(r_kn,  dtype=float)
    for i in range(1, len(r_arr)):          # enforce non-decreasing radius
        r_arr[i] = max(r_arr[i], r_arr[i - 1])

    # ── 2. Smooth splines for centre and radius ───────────────────────────────
    cs_x = CubicSpline(t_arr, cx_arr, bc_type='not-a-knot')
    cs_y = CubicSpline(t_arr, cy_arr, bc_type='not-a-knot')
    cs_r = PchipInterpolator(t_arr, r_arr)   # monotone → radius stays non-decreasing

    # ── 3. Build cone polygon ─────────────────────────────────────────────────
    if HAS_SHAPELY:
        # Use a FIXED step (np.arange) so the same t values are sampled in every frame.
        # linspace(0, t_arr[-1], N) shifts all intermediate t positions as t_arr[-1]
        # grows, causing the early cone to drift. With arange(0, ..., STEP_H) the
        # circles at t=0, STEP_H, 2*STEP_H, ... are identical across all frames —
        # only new circles at the trailing end are added as max_fh increases.
        STEP_H = 1.0   # hours — fixed step guarantees stability of the early cone

        # Adaptive minimum radius: ensures consecutive circles always overlap even
        # for fast-moving typhoons (needs 2*MIN_R > max_track_speed * STEP_H).
        if len(t_arr) >= 2:
            spd = np.hypot(np.diff(cx_arr), np.diff(cy_arr)) / np.diff(t_arr)
            MAX_SPD = float(np.max(spd))
        else:
            MAX_SPD = 0.0
        MIN_R = max(0.10, MAX_SPD * STEP_H * 0.6)

        t_dense  = np.arange(t_arr[0], t_arr[-1] + STEP_H * 0.5, STEP_H)
        cx_dense = np.interp(t_dense, t_arr, cx_arr)
        cy_dense = np.interp(t_dense, t_arr, cy_arr)
        r_dense  = np.maximum(np.interp(t_dense, t_arr, r_arr), MIN_R)

        geo_circles = [
            _ShapelyPoint(float(cx), float(cy)).buffer(float(r), resolution=64)
            for cx, cy, r in zip(cx_dense, cy_dense, r_dense)
        ]
        cone = _shapely_union(geo_circles)
        if cone.geom_type == 'MultiPolygon':
            cone = max(cone.geoms, key=lambda g: g.area)
        raw_lons, raw_lats = map(np.array, cone.exterior.xy)

        # Resample to uniform arc-length, then Gaussian-smooth the C0 circle-arc
        # junctions → visually smooth boundary without self-intersection.
        # Shapely exterior.xy already closes the ring (last==first); do NOT append
        # raw_lons[0] again or arc gets a zero-length final segment and interp1d fails.
        pts_cl = np.column_stack([raw_lons, raw_lats])
        seg = np.hypot(np.diff(pts_cl[:, 0]), np.diff(pts_cl[:, 1]))
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        N_s = 600
        t_u = np.linspace(0.0, arc[-1], N_s, endpoint=False)
        xs_u = _scipy_interp1d(arc, pts_cl[:, 0])(t_u)
        ys_u = _scipy_interp1d(arc, pts_cl[:, 1])(t_u)
        # Sigma in PHYSICAL arc-length (degrees), not sample count.
        # A fixed SMOOTH_DEG ensures the same physical smoothing scale in every frame
        # regardless of the total cone perimeter → no drift in the early cone.
        SMOOTH_DEG = 0.20                              # smooth over 0.2° of arc
        sigma = max(2, round(SMOOTH_DEG * N_s / arc[-1]))
        # Roll the array so the Gaussian wrap seam falls at the cone's tail
        # (0h initial position), not on the leading cap arc.  If the Shapely
        # exterior starts mid-cap, mode='wrap' blends opposite sides of the arc
        # across the array boundary → concave dent at the leading edge.
        _roll_idx = int(np.argmin(np.hypot(xs_u - float(cx_dense[0]),
                                           ys_u - float(cy_dense[0]))))
        xs_u = np.roll(xs_u, -_roll_idx)
        ys_u = np.roll(ys_u, -_roll_idx)
        poly_lons = list(_scipy_gf1d(xs_u, sigma=sigma, mode='wrap'))
        poly_lats = list(_scipy_gf1d(ys_u, sigma=sigma, mode='wrap'))

    else:
        # Analytical envelope fallback (may self-intersect on sharply curved tracks)
        N    = 400
        t_f  = np.linspace(t_arr[0], t_arr[-1], N)
        cx_f = cs_x(t_f); cy_f = cs_y(t_f); r_f = np.maximum(cs_r(t_f), 0.0)
        xt_f = cs_x(t_f, 1); yt_f = cs_y(t_f, 1); rt_f = cs_r(t_f, 1)
        v_f  = np.where(np.hypot(xt_f, yt_f) < 1e-9, 1e-9, np.hypot(xt_f, yt_f))
        kappa = np.clip(rt_f / v_f, -1.0, 1.0)
        eta   = np.sqrt(np.maximum(0.0, 1.0 - kappa**2))
        along_x = -r_f * kappa * xt_f / v_f
        along_y = -r_f * kappa * yt_f / v_f
        perp_x  =  r_f * eta   * yt_f / v_f
        perp_y  =  r_f * eta   * xt_f / v_f
        x_L = cx_f + along_x - perp_x
        y_L = cy_f + along_y + perp_y
        x_R = cx_f + along_x + perp_x
        y_R = cy_f + along_y - perp_y

        def _norm_angle(a, phi, side):
            if side == 'L':
                while a <= phi:            a += 2 * np.pi
                while a > phi + 2*np.pi:  a -= 2 * np.pi
            else:
                while a > phi:             a -= 2 * np.pi
                while a <= phi - 2*np.pi: a += 2 * np.pi
            return a

        phi_tail = np.arctan2(yt_f[-1], xt_f[-1])
        aL_tail  = _norm_angle(np.arctan2(y_L[-1]-cy_f[-1], x_L[-1]-cx_f[-1]), phi_tail, 'L')
        aR_tail  = _norm_angle(np.arctan2(y_R[-1]-cy_f[-1], x_R[-1]-cx_f[-1]), phi_tail, 'R')
        tail_a    = np.linspace(aL_tail, aR_tail, 64)
        tail_lons = cx_f[-1] + r_f[-1] * np.cos(tail_a)
        tail_lats = cy_f[-1] + r_f[-1] * np.sin(tail_a)
        poly_lons  = list(x_L) + list(tail_lons) + list(x_R[::-1])
        poly_lats  = list(y_L) + list(tail_lats) + list(y_R[::-1])
        r0 = float(r_f[0])
        if r0 > 0.05:
            phi_start = np.arctan2(yt_f[0], xt_f[0])
            aL_start  = _norm_angle(np.arctan2(y_L[0]-cy_f[0], x_L[0]-cx_f[0]), phi_start, 'L')
            aR_start  = _norm_angle(np.arctan2(y_R[0]-cy_f[0], x_R[0]-cx_f[0]), phi_start, 'R')
            start_a    = np.linspace(aR_start, aL_start - 2*np.pi, 48)
            poly_lons  = list(cx_f[0] + r0*np.cos(start_a)) + poly_lons
            poly_lats  = list(cy_f[0] + r0*np.sin(start_a)) + poly_lats

    # ── 3. Draw ───────────────────────────────────────────────────────────────
    ax.fill(poly_lons, poly_lats, color='#88CC78', alpha=0.42, zorder=0.35, **kw)
    ax.plot(np.append(poly_lons, poly_lons[:1]),
            np.append(poly_lats, poly_lats[:1]),
            color='#CC3300', lw=1.6, alpha=1.0, zorder=0.50, **kw)


def plot_forecast_map(df: pd.DataFrame, mean_df: pd.DataFrame, init_time: pd.Timestamp, track_id: str, save_path: str):
    """將各 Ensemble member 的預報路徑畫在地圖上，並標示 Ensemble 平均路徑。"""
    # 計算範圍（處理國際換日線）
    all_lons = list(df['lon']) + list(mean_df['lon'])
    all_lats = list(df['lat']) + list(mean_df['lat'])
    extent, use_360 = _auto_extent(all_lats, all_lons, pad_deg=3)
    min_lon, max_lon, min_lat, max_lat = extent
    if min_lat == max_lat:
        min_lat -= 2; max_lat += 2
    if min_lon == max_lon:
        min_lon -= 2; max_lon += 2
    target_ar = 1.40
    extent = _fit_extent_to_aspect((min_lon, max_lon, min_lat, max_lat), target_ar, use_360)

    cone_stop_fh = _compute_cone_stop_fh(df, init_time)
    if cone_stop_fh is not None:
        cone_stop_time = init_time + pd.to_timedelta(cone_stop_fh, unit='h')
        mean_plot_df = mean_df[mean_df['valid_time'] <= cone_stop_time].copy()
    else:
        mean_plot_df = mean_df.copy()

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(10, 7))
        # 跨越換日線時，使用中心在 180° 的投影
        if use_360:
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        else:
            ax = plt.axes(projection=ccrs.PlateCarree())

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor='#f0e8d4', alpha=0.9)
        ax.add_feature(cfeature.OCEAN, facecolor='#cce8f4', alpha=0.9)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.right_labels = False
        gl.top_labels = False
        _configure_cartopy_gridlines(gl, extent)

        # 不確定性圓錐（ensemble spread）
        _draw_uncertainty_cone(ax, df, mean_df, init_time, use_360, dict(transform=ccrs.PlateCarree()))

        # 畫每個 member（灰色線）與 6h 強度標記（彩色點）
        for sid, g in df.groupby("sample"):
            g = g.sort_values('valid_time')
            kw = dict(transform=ccrs.PlateCarree())
            lons = _normalize_lon_values(g["lon"].to_numpy(), use_360=use_360)
            lats = g["lat"].to_numpy()
            for seg_lon, seg_lat in _split_track_segments(lons, lats):
                ax.plot(seg_lon, seg_lat, color='gray', linewidth=0.4, alpha=0.5, zorder=0.9, **kw)
            pts_6h = get_6h_markers(g, init_time)
            if not pts_6h.empty:
                marker_lons = _normalize_lon_values(pts_6h['lon'].to_numpy(), use_360=use_360)
                for idx, (_, pt) in enumerate(pts_6h.iterrows()):
                    wind = pt.get('wind', np.nan)
                    cat = ss_category(wind)
                    ax.scatter(marker_lons[idx], pt['lat'], color=COLOR_MAP.get(cat, 'gray'), s=8, marker='o', edgecolor='none', alpha=0.8, zorder=1.1, **kw)

        # 起始位置星形標記
        if not mean_df.empty:
            init_pt = mean_df.sort_values('valid_time').iloc[0]
            init_lon_star = _normalize_lon_values([init_pt['lon']], use_360=use_360)[0]
            ax.scatter([init_lon_star], [init_pt['lat']], marker='*', color='gold', s=220,
                       ec='darkorange', zorder=7, linewidth=0.8, transform=ccrs.PlateCarree())

        # 平均路徑（紅線）與 24h 標記（紅色方塊）及標註
        mean_lons = _normalize_lon_values(mean_plot_df["lon"].to_numpy(), use_360=use_360)
        mean_lats = mean_plot_df["lat"].to_numpy()
        for seg_lon, seg_lat in _split_track_segments(mean_lons, mean_lats):
            ax.plot(seg_lon, seg_lat, 'r-', lw=2.5, zorder=4, transform=ccrs.PlateCarree())
        last_mean_pt = mean_plot_df.sort_values('valid_time').iloc[-1]
        last_mean_lon = _normalize_lon_values([last_mean_pt['lon']], use_360=use_360)[0]
        last_fh = int(round((last_mean_pt['valid_time'] - init_time).total_seconds() / 3600.0))
        last_intensity = _format_intensity_label(last_mean_pt.get('wind', np.nan))
        ax.scatter([last_mean_lon], [last_mean_pt['lat']], marker='s', color='red', s=40,
                   ec='black', zorder=5, linewidth=1, transform=ccrs.PlateCarree())
        ax.text(last_mean_lon + 0.4, last_mean_pt['lat'] + 0.4, f'+{last_fh}h\n{last_intensity}',
                fontsize=6, color='darkred', fontweight='bold', zorder=6,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='red', alpha=0.90, linewidth=0.6),
                transform=ccrs.PlateCarree(), clip_on=True)
        pts_24h_mean = get_24h_markers(mean_plot_df, init_time)
        if not pts_24h_mean.empty:
            marker_lons = _normalize_lon_values(pts_24h_mean['lon'].to_numpy(), use_360=use_360)
            ax.scatter(marker_lons, pts_24h_mean['lat'], marker='s', color='red', s=30, ec='black', zorder=5, transform=ccrs.PlateCarree())
            # +24h、+48h… 文字標籤
            for i, (_, pt) in enumerate(pts_24h_mean.iterrows()):
                fh = int((pt['valid_time'] - init_time).total_seconds() / 3600)
                if fh == 0 or fh == last_fh:
                    continue
                lbl_lon = marker_lons[i]
                ax.text(lbl_lon + 0.4, pt['lat'] + 0.4, f'+{fh}h',
                        fontsize=6, color='darkred', fontweight='bold', zorder=6,
                        transform=ccrs.PlateCarree(), clip_on=True)
            # 右下角摘要框
            summary = build_24h_summary(pts_24h_mean, init_time)
            if summary:
                n_lines = len(summary.splitlines())
                y_anchor = 0.27 + min(n_lines, 8) * 0.015
                ax.text(0.985, y_anchor, summary, transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='red', alpha=0.85, linewidth=0.6),
                        zorder=6)

        # 圖例
        handles = [
            mlines.Line2D([], [], color='gray', label='Ensemble Members', lw=1),
            mlines.Line2D([], [], color='red', marker='s', markeredgecolor='black', label='FNV3 Mean', lw=2),
            mlines.Line2D([], [], color='#88CC78', lw=4, alpha=0.55, label='Uncertainty Cone'),
            mlines.Line2D([], [], color='gold', marker='*', ms=10, ls='', markeredgecolor='darkorange', label='Init Position'),
        ]
        l1 = ax.legend(
            handles=handles,
            loc='upper left',
            title='Track Source',
            fontsize=9,
            frameon=True,
            framealpha=0.92,
            borderpad=0.6,
            labelspacing=0.3,
            handlelength=0.9,
            handletextpad=0.7
        )
        l1.get_title().set_multialignment('center')
        l1.get_title().set_ha('center')
        ax.add_artist(l1)

        int_handles = []
        labels = ['TD', 'Cat2', 'Cat5', 'TS', 'Cat3', 'Cat1', 'Cat4']
        for cat in labels:
            int_handles.append(mlines.Line2D([],[], color=COLOR_MAP[cat], marker='o', ms=7, ls='', markeredgecolor='black', label=cat))
        int_leg = ax.legend(
            handles=int_handles,
            loc='lower right',
            bbox_to_anchor=(0.995, 0.005),
            title='Intensity',
            fontsize=8,
            frameon=True,
            framealpha=0.92,
            ncol=3,
            borderpad=0.6,
            labelspacing=0.25,
            handlelength=0.75,
            handletextpad=0.35,
            markerscale=0.75,
            borderaxespad=0.5
        )
        int_leg.get_title().set_multialignment('center')
        int_leg.get_title().set_ha('center')
        ax.add_artist(int_leg)

        ax.set_title(f"FNV3 {track_id} Track Forecast (Init: {pd.to_datetime(init_time).strftime('%Y-%m-%d %H:%M UTC')})", fontsize=16, fontweight='bold', pad=15)
        ax.text(0.995, 0.992, 'By Pillar', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, style='italic',
                color='black', fontweight='bold', zorder=10)
        plt.tight_layout(pad=0.6)
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight', facecolor='white', pad_inches=0.12)
        plt.close(fig)
        print(f"[INFO] 已儲存地圖：{save_path}")
        return

    # 無 Cartopy：經緯度折線圖
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    ax.set_facecolor('#cce8f4')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # 不確定性圓錐
    _draw_uncertainty_cone(ax, df, mean_df, init_time, use_360, {})

    for sid, g in df.groupby("sample"):
        g = g.sort_values('valid_time')
        lons = _normalize_lon_values(g["lon"].to_numpy(), use_360=use_360)
        lats = g["lat"].to_numpy()
        for seg_lon, seg_lat in _split_track_segments(lons, lats):
            ax.plot(seg_lon, seg_lat, color='gray', linewidth=0.4, alpha=0.5, zorder=0.9)
        pts_6h = get_6h_markers(g, init_time)
        if not pts_6h.empty:
            marker_lons = _normalize_lon_values(pts_6h['lon'].to_numpy(), use_360=use_360)
            for idx, (_, pt) in enumerate(pts_6h.iterrows()):
                wind = pt.get('wind', np.nan)
                cat = ss_category(wind)
                ax.scatter(marker_lons[idx], pt['lat'], color=COLOR_MAP.get(cat, 'gray'), s=8, marker='o', edgecolor='none', alpha=0.8, zorder=1.1)

    # 起始位置星形標記
    if not mean_df.empty:
        init_pt = mean_df.sort_values('valid_time').iloc[0]
        init_lon_star = _normalize_lon_values([init_pt['lon']], use_360=use_360)[0]
        ax.scatter([init_lon_star], [init_pt['lat']], marker='*', color='gold', s=220,
                   ec='darkorange', zorder=7, linewidth=0.8)

    mean_lons = _normalize_lon_values(mean_plot_df["lon"].to_numpy(), use_360=use_360)
    mean_lats = mean_plot_df["lat"].to_numpy()
    for seg_lon, seg_lat in _split_track_segments(mean_lons, mean_lats):
        ax.plot(seg_lon, seg_lat, 'r-', lw=2.5, zorder=4)
    last_mean_pt = mean_plot_df.sort_values('valid_time').iloc[-1]
    last_mean_lon = _normalize_lon_values([last_mean_pt['lon']], use_360=use_360)[0]
    last_fh = int(round((last_mean_pt['valid_time'] - init_time).total_seconds() / 3600.0))
    last_intensity = _format_intensity_label(last_mean_pt.get('wind', np.nan))
    ax.scatter([last_mean_lon], [last_mean_pt['lat']], marker='s', color='red', s=40,
               ec='black', zorder=5, linewidth=1)
    ax.text(last_mean_lon + 0.4, last_mean_pt['lat'] + 0.4, f'+{last_fh}h\n{last_intensity}',
            fontsize=6, color='darkred', fontweight='bold', zorder=6,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='red', alpha=0.90, linewidth=0.6),
            clip_on=True)
    pts_24h_mean = get_24h_markers(mean_plot_df, init_time)
    if not pts_24h_mean.empty:
        marker_lons = _normalize_lon_values(pts_24h_mean['lon'].to_numpy(), use_360=use_360)
        ax.scatter(marker_lons, pts_24h_mean['lat'], marker='s', color='red', s=30, ec='black', zorder=5)
        # +24h、+48h… 文字標籤
        for i, (_, pt) in enumerate(pts_24h_mean.iterrows()):
            fh = int((pt['valid_time'] - init_time).total_seconds() / 3600)
            if fh == 0 or fh == last_fh:
                continue
            ax.text(marker_lons[i] + 0.4, pt['lat'] + 0.4, f'+{fh}h',
                    fontsize=6, color='darkred', fontweight='bold', zorder=6, clip_on=True)
        # 右下角摘要框
        summary = build_24h_summary(pts_24h_mean, init_time)
        if summary:
            n_lines = len(summary.splitlines())
            y_anchor = 0.30 + min(n_lines, 8) * 0.015
            ax.text(0.985, y_anchor, summary, transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='red', alpha=0.85, linewidth=0.6),
                zorder=6)

    handles = [
        mlines.Line2D([], [], color='gray', label='Ensemble Members', lw=1),
        mlines.Line2D([], [], color='red', marker='s', markeredgecolor='black', label='FNV3 Mean', lw=2),
        mlines.Line2D([], [], color='#88CC78', lw=4, alpha=0.55, label='Uncertainty Cone'),
        mlines.Line2D([], [], color='gold', marker='*', ms=10, ls='', markeredgecolor='darkorange', label='Init Position'),
    ]
    l1 = ax.legend(
        handles=handles,
        loc='upper left',
        title='Track Source',
        fontsize=9,
        frameon=True,
        framealpha=0.92,
        borderpad=0.6,
        labelspacing=0.3,
        handlelength=0.9,
        handletextpad=0.7
    )
    l1.get_title().set_multialignment('center')
    l1.get_title().set_ha('center')
    ax.add_artist(l1)

    int_handles = []
    labels = ['TD', 'Cat2', 'Cat5', 'TS', 'Cat3', 'Cat1', 'Cat4']
    for cat in labels:
        int_handles.append(mlines.Line2D([],[], color=COLOR_MAP[cat], marker='o', ms=7, ls='', markeredgecolor='black', label=cat))
    int_leg = ax.legend(
        handles=int_handles,
        loc='lower right',
        bbox_to_anchor=(0.995, 0.005),
        title='Intensity',
        fontsize=8,
        frameon=True,
        framealpha=0.92,
        ncol=3,
        borderpad=0.6,
        labelspacing=0.25,
        handlelength=0.75,
        handletextpad=0.35,
        markerscale=0.75,
        borderaxespad=0.5
    )
    int_leg.get_title().set_multialignment('center')
    int_leg.get_title().set_ha('center')
    ax.add_artist(int_leg)

    ax.set_title(f"FNV3 {track_id} Track Forecast (Init: {pd.to_datetime(init_time).strftime('%Y-%m-%d %H:%M UTC')})", fontsize=16, fontweight='bold', pad=18)
    ax.text(0.995, 0.992, 'By Pillar', transform=ax.transAxes,
            ha='right', va='top', fontsize=7, style='italic',
            color='black', fontweight='bold', zorder=10)
    plt.tight_layout(pad=0.6)
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight', facecolor='white', pad_inches=0.12)
    plt.close(fig)
    print(f"[INFO] 已儲存地圖：{save_path}")


def download(url_tmpl: str, out_dir: str, label: str) -> str:
    url = url_tmpl.format(year=YEAR, month=MONTH, day=DAY, hour=HOUR, minute=MINUTE)
    fname = f"FNV3_{YEAR}_{MONTH}_{DAY}T{HOUR}_{MINUTE}_paired.csv"
    out_path = os.path.join(out_dir, fname)
    print(f"[{label}] GET {url}")
    resp = requests.get(url, headers=HEADERS, stream=True, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} for {label}: {url}")
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"[{label}] saved to {out_path}")
    return out_path


def plot_genesis_potential_map(csv_path: str, save_path: str):
    """繪製西太平洋 FNV3 Ensemble 潛勢預報總覽圖（以 MSLP 著色）。"""
    if not os.path.exists(csv_path):
        print(f"[GENESIS] 找不到潛勢 CSV：{csv_path}")
        return None

    df = pd.read_csv(csv_path, comment='#')
    if df.empty:
        print("[GENESIS] 潛勢 CSV 無資料")
        return None

    if 'maximum_sustained_wind_speed_knots' in df.columns:
        df = df.rename(columns={'maximum_sustained_wind_speed_knots': 'wind'})
    df['valid_time'] = pd.to_datetime(df['valid_time'], errors='coerce', utc=True)

    # 從最小 lead_time 反推初始化時間，格式化為 YYYY-MM-DD-HHZ
    if 'lead_time_hours' in df.columns:
        first = df.sort_values('lead_time_hours').iloc[0]
        init_dt = pd.to_datetime(first['valid_time'], utc=True) - pd.to_timedelta(float(first['lead_time_hours']), unit='h')
        init_time_str = init_dt.strftime('%Y-%m-%d-%HZ')
    elif 'init_time' in df.columns:
        init_time_str = str(df['init_time'].iloc[0]) + '-00Z'
    else:
        init_time_str = ''

    # 只保留西太平洋範圍
    WP_LON_MIN, WP_LON_MAX = 95.0, 170.0
    WP_LAT_MIN, WP_LAT_MAX = 3.0, 53.0
    df_wp = df[(df['lon'] >= WP_LON_MIN) & (df['lon'] <= WP_LON_MAX) &
               (df['lat'] >= WP_LAT_MIN) & (df['lat'] <= WP_LAT_MAX)].copy()

    if df_wp.empty:
        print("[GENESIS] 西太平洋範圍內無潛勢資料")
        return None

    EXTENT = (98, 162, 3, 52)
    global_min_mslp = float(df_wp['minimum_sea_level_pressure_hpa'].min())

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(13, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.8, color='#555555')
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='#777777')
        ax.add_feature(cfeature.LAND, facecolor='#f0e8d4', alpha=0.95)
        ax.add_feature(cfeature.OCEAN, facecolor='#cce8f4', alpha=0.95)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.right_labels = False
        gl.top_labels = False
        gl.xlocator = mticker.MultipleLocator(10)
        gl.ylocator = mticker.MultipleLocator(10)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        kw = dict(transform=ccrs.PlateCarree())
    else:
        fig = plt.figure(figsize=(13, 8), dpi=150, facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#cce8f4')
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.grid(True, linewidth=0.5, alpha=0.4, linestyle='--')
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_ylabel('Latitude', fontsize=9)
        kw = {}

    # 繪製各 track_id + sample 的 ensemble 軌跡
    for (tid, sid), g in df_wp.groupby(['track_id', 'sample']):
        g = g.sort_values('valid_time')
        lons = g['lon'].to_numpy()
        lats = g['lat'].to_numpy()
        mslps = g['minimum_sea_level_pressure_hpa'].to_numpy()

        # 軌跡連線（淡灰）
        ax.plot(lons, lats, color='#888888', linewidth=0.4, alpha=0.35, zorder=1, **kw)

        # MSLP 著色圓點（全部實心）
        for lon, lat, mslp in zip(lons, lats, mslps):
            color, _ = _mslp_to_color(float(mslp))
            ax.scatter(lon, lat, color=color, s=10, marker='o',
                       edgecolor='none', alpha=0.85, zorder=2, **kw)

    # 圖例（MSLP 色階）
    legend_handles = []
    for _, color, label in MSLP_COLOR_BINS:
        h = mlines.Line2D([], [], marker='o', ms=7, ls='',
                          color=color, markeredgecolor='none', label=label)
        legend_handles.append(h)

    leg = ax.legend(handles=legend_handles, loc='upper left',
                    title='min. MSLP\n' + '─' * 13, title_fontsize=8,
                    fontsize=8, frameon=True, framealpha=0.92,
                    borderpad=0.6, labelspacing=0.3, handletextpad=0.5)
    leg.get_title().set_multialignment('center')
    leg.get_title().set_ha('center')

    # 右上角顯示全域最低 MSLP
    ax.text(0.99, 0.99, f'min. MSLP: {global_min_mslp:.1f} hPa',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            color='#CC0000', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CC0000', alpha=0.85, linewidth=0.8))

    ax.set_title(
        f'FNV3 Ensemble Forecast for Tropical Cyclone (0–360 hours)  '
        f'Data sourced from Google DeepMind\n'
        f'Initial time: {init_time_str}',
        fontsize=11, fontweight='bold', pad=12
    )
    ax.text(0.995, 0.008, 'By Pillar', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7, style='italic',
            color='black', fontweight='bold', zorder=10,
            **({} if not HAS_CARTOPY else {}))
    plt.tight_layout(pad=0.6)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.12)
    plt.close(fig)
    print(f'[GENESIS] 已儲存潛勢預報圖：{save_path}')
    return save_path


def main():
    # 若檔案不存在，才下載 Ensemble 檔；否則直接讀取
    if not os.path.exists(CSV_PATH):
        url = (
            "https://deepmind.google.com/science/weatherlab/download/cyclones/FNV3/ensemble/paired/csv/"
            f"FNV3_{YEAR}_{MONTH}_{DAY}T{HOUR}_{MINUTE}_paired.csv"
        )
        print(f"[DL] GET {url}")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(CSV_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[DL] saved to {CSV_PATH}")

    # 下載 Ensemble Mean 檔
    mean_csv_path = os.path.join(MEAN_DIR, FILENAME)
    if not os.path.exists(mean_csv_path):
        mean_url = (
            "https://deepmind.google.com/science/weatherlab/download/cyclones/FNV3/ensemble_mean/paired/csv/"
            f"FNV3_{YEAR}_{MONTH}_{DAY}T{HOUR}_{MINUTE}_paired.csv"
        )
        print(f"[DL-MEAN] GET {mean_url}")
        resp = requests.get(mean_url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(mean_csv_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[DL-MEAN] saved to {mean_csv_path}")

    # 下載 Cyclogenesis 潛勢 CSV
    # 優先使用與當期日期相符的檔案，不存在時嘗試下載，下載失敗才退而用目錄中最新檔
    CYCLOGENESIS_CSV_PATH_USED = None
    if os.path.exists(CYCLOGENESIS_CSV_PATH):
        CYCLOGENESIS_CSV_PATH_USED = CYCLOGENESIS_CSV_PATH
        print(f"[GENESIS] 使用當期潛勢檔: {CYCLOGENESIS_CSV_PATH_USED}")
    else:
        cyc_url = CYCLOGENESIS_URL.format(
            year=YEAR, month=MONTH, day=DAY, hour=HOUR, minute=MINUTE)
        print(f"[GENESIS] 下載潛勢 CSV: {cyc_url}")
        try:
            resp = requests.get(cyc_url, headers=HEADERS, stream=True, timeout=60)
            resp.raise_for_status()
            with open(CYCLOGENESIS_CSV_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"[GENESIS] saved to {CYCLOGENESIS_CSV_PATH}")
            CYCLOGENESIS_CSV_PATH_USED = CYCLOGENESIS_CSV_PATH
        except Exception as e:
            print(f"[GENESIS] 下載失敗: {e}，嘗試使用目錄中最新檔")
            fallback = sorted([
                f for f in os.listdir(CYCLOGENESIS_DIR) if f.endswith('_cyclogenesis.csv')
            ])
            if fallback:
                CYCLOGENESIS_CSV_PATH_USED = os.path.join(CYCLOGENESIS_DIR, fallback[-1])
                print(f"[GENESIS] 退而使用: {CYCLOGENESIS_CSV_PATH_USED}")

    # 生成西太平洋潛勢預報圖
    genesis_map_path = None
    if CYCLOGENESIS_CSV_PATH_USED and os.path.exists(CYCLOGENESIS_CSV_PATH_USED):
        print("[GENESIS] 正在繪製西太平洋潛勢預報圖...")
        genesis_save = os.path.join(OUTPUT_DIR, "WP_Genesis_Potential.png")
        genesis_map_path = plot_genesis_potential_map(CYCLOGENESIS_CSV_PATH_USED, genesis_save)

    # 處理所有目標颱風
    storms = []
    
    for TARGET_TRACK_ID in TARGET_TRACK_IDS:
        print(f"\n[RUN] === 處理颱風 {TARGET_TRACK_ID} ===")
        print(f"[RUN] 讀取 {CSV_PATH}，並繪製 {TARGET_TRACK_ID} 預報地圖…")
        
        try:
            df, init_time = load_forecast_dataframe(CSV_PATH, TARGET_TRACK_ID)
            
            # 讀取 Ensemble Mean 檔案
            print(f"[RUN] 讀取 Ensemble Mean: {mean_csv_path}")
            mean_df, _ = load_forecast_dataframe(mean_csv_path, TARGET_TRACK_ID)

            read_time = pd.Timestamp.now(tz='UTC')
            current_info = extract_current_info(mean_df, read_time)
            
            # 從 JTWC web.txt 抓取實時資料
            if TARGET_TRACK_ID in JTWC_TEXT_URLS:
                print(f"[JTWC] 嘗試從 JTWC web.txt 抓取 {TARGET_TRACK_ID} 資料...")
                jtwc_data = scrape_jtwc_text_product(TARGET_TRACK_ID)

                if jtwc_data:
                    current_info['jtwc'] = jtwc_data
                    # 目前強度優先採用 JTWC web.txt 的最大持續風速
                    max_wind = jtwc_data.get('max_winds_kt')
                    if max_wind is not None:
                        try:
                            current_info['wind'] = float(max_wind)
                            current_info['category'] = ss_category(current_info['wind'])
                            print(f"[JTWC] 目前強度已更新為 web.txt: {int(float(max_wind))} kt")
                        except Exception:
                            print("[JTWC] 警告: max_winds_kt 格式異常，保留原始強度")

                    print(f"[JTWC] OK: 解析並寫入實時資料")
                else:
                    print(f"[JTWC] FAIL web.txt 抓取失敗")
            else:
                print(f"[JTWC] 未設定 web.txt URL，略過: {TARGET_TRACK_ID}")
            
            save_path = os.path.join(OUTPUT_DIR, f"{TARGET_TRACK_ID}_Forecast_Map.png")
            plot_forecast_map(df, mean_df, init_time, TARGET_TRACK_ID, save_path)
            
            # 下載 JTWC 預報圖
            print(f"[JTWC] 正在下載 {TARGET_TRACK_ID} 的 JTWC 預報圖...")
            download_jtwc_image(TARGET_TRACK_ID, OUTPUT_DIR)
            
            # 生成幀序列（用於網頁動畫）
            print("[ANIMATION] 正在生成幀序列...")
            frames_dir = os.path.join(OUTPUT_DIR, f"animation_frames_{TARGET_TRACK_ID}")
            frame_paths = generate_frame_sequence(df, mean_df, init_time, TARGET_TRACK_ID, frames_dir, max_frames=72)

            # 將幀序列輸出成 GIF
            gif_path = os.path.join(OUTPUT_DIR, f"{TARGET_TRACK_ID}_Forecast_Animation.gif")
            gif_output = create_gif_from_frames(frame_paths, gif_path, duration_ms=180, loop=0)
            
            # 將此颱風資料加入列表（包含各自的幀序列路徑）
            storms.append({
                'track_id': TARGET_TRACK_ID,
                'forecast_map_path': save_path,
                'current_info': current_info,
                'frames_dir': frames_dir,  # 每個颱風專屬的幀序列資料夾
                'forecast_gif_path': gif_output,
            })
            
            print(f"[DONE] 完成颱風 {TARGET_TRACK_ID} 的處理")
            
        except Exception as e:
            print(f"[ERROR] 處理 {TARGET_TRACK_ID} 時發生錯誤: {e}")
            continue
    
    # 生成預報網站 HTML（支援多顆颱風；列表順序決定上下位置）
    print("\n[HTML] 正在生成預報網站...")
    from generate_forecast_website import generate_forecast_html
    html_path = os.path.join(OUTPUT_DIR, "index.html")
    generate_forecast_html(storms, html_path, frames_dir=None,
                           genesis_map_path=genesis_map_path)
    
    print("\n[DONE] 所有颱風處理完成")



if __name__ == "__main__":
    main()