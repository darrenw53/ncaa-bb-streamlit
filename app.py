import io
import re
import difflib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Optional (only needed if you enable GitHub fallback)
try:
    import requests
except Exception:
    requests = None


# ============================================================
# CONFIG: REPO ROOT AUTO-LOADING
# ============================================================
# This now searches the REPO ROOT (same folder as app.py)
# Put files like:
#   kenpom_1.10.26.xlsx
#   Schedule_1.10.26.xlsx
REPO_ROOT = Path(__file__).parent

# Optional GitHub fallback (if you want it):
GITHUB_OWNER = ""   # e.g. "DarrenWitter"
GITHUB_REPO = ""    # e.g. "NCAA_BB"
GITHUB_BRANCH = "main"
GITHUB_DATA_DIR = ""  # repo root


# -----------------------------
# Constants (match your sheet)
# -----------------------------
AVG_EFF = 102.0
AVG_TEMPO = 64.8


# -----------------------------
# Trapezoid / Fraud tagging
# -----------------------------
TRAP_POLY_X = [64.0, 66.0, 73.0, 75.0]
TRAP_POLY_Y = [26.0, 38.0, 38.0, 26.0]

FRAUD_PACE_MIN = 71.0
FRAUD_NET_MAX = 26.0


def point_in_polygon(px: float, py: float, poly_x: list[float], poly_y: list[float]) -> bool:
    """Ray-casting point-in-polygon. Works for convex/non-convex polygons."""
    inside = False
    j = len(poly_x) - 1
    for i in range(len(poly_x)):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        intersects = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def tag_team_style(adjT: float, net: float) -> tuple[bool, bool, str]:
    """Returns (in_trapezoid, is_fraud, tag_string)."""
    if adjT is None or net is None or np.isnan(adjT) or np.isnan(net):
        return False, False, ""
    in_trap = point_in_polygon(float(adjT), float(net), TRAP_POLY_X, TRAP_POLY_Y)
    is_fraud = (float(adjT) >= FRAUD_PACE_MIN) and (float(net) < FRAUD_NET_MAX) and (not in_trap)
    tag = "🏆 TRAP" if in_trap else ("🚨 FRAUD" if is_fraud else "")
    return in_trap, is_fraud, tag


# =============================
# Share-card helpers (HTML)
# =============================
def fmt_num(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def pick_side_from_edge(edge_spread):
    # Edge_Spread = Model(Home_Margin) + Spread_Home
    if edge_spread is None or (isinstance(edge_spread, float) and np.isnan(edge_spread)):
        return ""
    return "HOME" if edge_spread > 0 else "VISITOR"


def pick_total_from_edge(edge_total):
    # Edge_Total = Model_Total - Vegas_Total
    if edge_total is None or (isinstance(edge_total, float) and np.isnan(edge_total)):
        return ""
    return "OVER" if edge_total > 0 else "UNDER"


def confidence_label(edge_abs):
    if edge_abs >= 6:
        return "🔥 Strong"
    if edge_abs >= 4:
        return "✅ Solid"
    if edge_abs >= 2:
        return "👍 Lean"
    return "—"


def build_share_card_html(spread_df, total_df, title, subtitle):
    """
    Renders a branded, shareable HTML card.
    """
    def safe_get(r, key):
        return r[key] if key in r else np.nan

    def table_html(df, kind):
        if df is None or len(df) == 0:
            return f"<div class='empty'>No {kind} plays today.</div>"

        rows = []
        for _, r in df.iterrows():
            vtag = safe_get(r, 'Visitor_Tag')
            htag = safe_get(r, 'Home_Tag')
            tag_txt = ""
            if isinstance(vtag, str) and vtag.strip():
                tag_txt += f" {vtag.strip()}"
            if isinstance(htag, str) and htag.strip():
                tag_txt += f" / {htag.strip()}" if tag_txt else f" {htag.strip()}"
            matchup = f"{safe_get(r,'Visitor')} @ {safe_get(r,'Home')}" + (
                f" <span style='opacity:0.8;font-weight:700'>[{tag_txt.strip()}]</span>" if tag_txt else ""
            )
            score = f"{fmt_num(safe_get(r,'Pred_Away'))} - {fmt_num(safe_get(r,'Pred_Home'))}"

            if kind == "spread":
                vegas = fmt_num(safe_get(r, "Spread_Home"), 1)
                edge = safe_get(r, "Edge_Spread")
                edge_txt = fmt_num(edge, 2)
                side = pick_side_from_edge(edge)
                conf = confidence_label(abs(edge) if pd.notna(edge) else 0)
                rows.append(
                    f"<tr>"
                    f"<td class='col-match'>{matchup}</td>"
                    f"<td class='col-score'>{score}</td>"
                    f"<td class='col-vegas'>{vegas}</td>"
                    f"<td class='col-edge'>{edge_txt}</td>"
                    f"<td class='col-pick'>{side}</td>"
                    f"<td class='col-conf'>{conf}</td>"
                    f"</tr>"
                )
            else:
                vegas = fmt_num(safe_get(r, "DK_Total"), 1)
                edge = safe_get(r, "Edge_Total")
                edge_txt = fmt_num(edge, 2)
                side = pick_total_from_edge(edge)
                conf = confidence_label(abs(edge) if pd.notna(edge) else 0)
                rows.append(
                    f"<tr>"
                    f"<td class='col-match'>{matchup}</td>"
                    f"<td class='col-score'>{score}</td>"
                    f"<td class='col-vegas'>{vegas}</td>"
                    f"<td class='col-edge'>{edge_txt}</td>"
                    f"<td class='col-pick'>{side}</td>"
                    f"<td class='col-conf'>{conf}</td>"
                    f"</tr>"
                )

        header = (
            "<tr>"
            "<th>Matchup</th><th>Model Score</th><th>Vegas</th><th>Edge</th><th>Pick</th><th>Confidence</th>"
            "</tr>"
        )
        return f"<table>{header}{''.join(rows)}</table>"

    brand_line = "DW’s Master Lock Picks of The Day"
    brand_tag = "NCAA • Model Edges • Daily Card"

    html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
          padding: 24px;
          background: #ffffff;
        }}

        .card {{
          max-width: 980px;
          margin: 0 auto;
          border: 1px solid #e6e6e6;
          border-radius: 18px;
          overflow: hidden;
          box-shadow: 0 10px 28px rgba(0,0,0,0.06);
          background: #fff;
        }}

        .hero {{
          position: relative;
          padding: 22px 22px 18px 22px;
          background: radial-gradient(1200px circle at 10% 20%, rgba(255,255,255,0.35), rgba(255,255,255,0) 40%),
                      linear-gradient(135deg, #0f172a 0%, #111827 35%, #1f2937 100%);
          color: #fff;
        }}
        .hero:before {{
          content: "";
          position: absolute;
          inset: -2px;
          background: linear-gradient(90deg, rgba(56,189,248,0.35), rgba(168,85,247,0.35), rgba(34,197,94,0.35));
          filter: blur(22px);
          opacity: 0.55;
          z-index: 0;
        }}
        .hero-inner {{
          position: relative;
          z-index: 1;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          flex-wrap: wrap;
        }}
        .hero-left {{
          display: flex;
          align-items: center;
          gap: 14px;
          min-width: 260px;
        }}
        .lock-badge {{
          width: 46px;
          height: 46px;
          border-radius: 14px;
          display: grid;
          place-items: center;
          background: rgba(255,255,255,0.10);
          border: 1px solid rgba(255,255,255,0.20);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.10);
          font-size: 22px;
        }}
        .hero-title {{
          margin: 0;
          font-size: 24px;
          font-weight: 900;
          letter-spacing: -0.4px;
          line-height: 1.05;
        }}
        .hero-tag {{
          margin: 6px 0 0 0;
          font-size: 12px;
          color: rgba(255,255,255,0.80);
          font-weight: 700;
          letter-spacing: 0.2px;
        }}
        .hero-right {{
          text-align: right;
          min-width: 240px;
        }}
        .pill {{
          display: inline-block;
          padding: 6px 12px;
          border-radius: 999px;
          background: rgba(255,255,255,0.10);
          border: 1px solid rgba(255,255,255,0.18);
          font-size: 12px;
          font-weight: 800;
          margin-left: 8px;
        }}
        .subtitle {{
          margin: 10px 0 0 0;
          color: rgba(255,255,255,0.78);
          font-size: 12.5px;
          line-height: 1.35;
          max-width: 520px;
          margin-left: auto;
        }}

        .content {{
          padding: 18px 22px 18px 22px;
        }}
        .section {{
          margin-top: 18px;
        }}
        .section h2 {{
          font-size: 18px;
          margin: 0 0 10px 0;
          letter-spacing: -0.2px;
        }}

        table {{
          width: 100%;
          border-collapse: collapse;
          overflow: hidden;
          border-radius: 14px;
        }}
        th, td {{
          padding: 10px 10px;
          border-bottom: 1px solid #eee;
          font-size: 14px;
          vertical-align: top;
        }}
        th {{
          text-align: left;
          background: #fafafa;
          font-weight: 900;
          font-size: 12.5px;
        }}
        tr:hover td {{
          background: #fcfcfc;
        }}
        .empty {{
          padding: 12px;
          color: #666;
          border: 1px dashed #ddd;
          border-radius: 14px;
        }}
        .footer {{
          margin-top: 16px;
          font-size: 12px;
          color: #777;
          line-height: 1.35;
        }}

        .col-match {{ width: 34%; font-weight: 800; }}
        .col-score {{ width: 16%; }}
        .col-vegas {{ width: 12%; }}
        .col-edge {{ width: 12%; font-weight: 900; }}
        .col-pick {{ width: 12%; font-weight: 900; }}
        .col-conf {{ width: 14%; }}
      </style>
    </head>
    <body>
      <div class="card">
        <div class="hero">
          <div class="hero-inner">
            <div class="hero-left">
              <div class="lock-badge">🔒</div>
              <div>
                <p class="hero-title">{brand_line}</p>
                <p class="hero-tag">{brand_tag}</p>
              </div>
            </div>

            <div class="hero-right">
              <span class="pill">{title}</span>
              <span class="pill">Subscriber Card</span>
              <p class="subtitle">{subtitle}</p>
            </div>
          </div>
        </div>

        <div class="content">
          <div class="section">
            <h2>Spread Plays</h2>
            {table_html(spread_df, "spread")}
          </div>

          <div class="section">
            <h2>Total Plays</h2>
            {table_html(total_df, "total")}
          </div>

          <div class="footer">
            <div><b>Notes:</b> Edges are model vs DraftKings lines/totals.</div>
            <div>Entertainment only. If odds are missing or teams don’t map cleanly, plays may not appear.</div>
          </div>
        </div>
      </div>
    </body>
    </html>
    """
    return html


# ============================================================
# AUTO-LOADING HELPERS (repo root + optional GitHub API)
# ============================================================
def _extract_date_key_from_filename(name: str):
    """
    Parse dates in filenames like:
      kenpom_1.10.26.xlsx
      Schedule_01.10.2026.xlsx
      kenpom_2026-01-10.xlsx
    Returns a comparable tuple (year, month, day) or None.
    """
    s = name.lower()

    # 2026-01-10
    m = re.search(r"(20\d{2})[-_.](\d{1,2})[-_.](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (y, mo, d)

    # 1.10.26 or 01.10.2026
    m = re.search(r"(\d{1,2})[.](\d{1,2})[.](\d{2,4})", s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        return (y, mo, d)

    return None


def find_latest_local_file(pattern: str, folder: Path) -> Path | None:
    """
    Finds the newest file matching pattern.
    Uses:
      1) date parsed from filename if possible
      2) otherwise filesystem mtime
    """
    files = sorted(folder.glob(pattern))
    if not files:
        return None

    scored = []
    for p in files:
        dk = _extract_date_key_from_filename(p.name)
        scored.append((dk if dk is not None else (0, 0, 0), p))

    any_dated = any(k != (0, 0, 0) for k, _ in scored)
    if any_dated:
        scored = [sp for sp in scored if sp[0] != (0, 0, 0)]
        scored.sort(key=lambda x: x[0])
        return scored[-1][1]

    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def github_list_dir(owner: str, repo: str, path: str, branch: str, token: str | None):
    """
    Lists files in a GitHub directory via API.
    If path == "" -> lists repo root.
    """
    if requests is None:
        raise RuntimeError("requests is not installed, cannot use GitHub API fallback.")

    path = (path or "").strip().strip("/")
    base = f"https://api.github.com/repos/{owner}/{repo}/contents"
    url = base + (f"/{path}" if path else "")

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {"ref": branch}

    r = requests.get(url, headers=headers, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API error {r.status_code}: {r.text}")
    return r.json()


def github_download_raw(download_url: str, token: str | None) -> bytes:
    if requests is None:
        raise RuntimeError("requests is not installed, cannot use GitHub API fallback.")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(download_url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub download error {r.status_code}: {r.text}")
    return r.content


def find_latest_github_file(owner: str, repo: str, folder: str, pattern_regex: str, branch: str, token: str | None):
    """
    Finds newest by filename date (preferred). If no date match, uses name sort.
    Returns (name, bytes) or (None, None)
    """
    items = github_list_dir(owner, repo, folder, branch, token)
    files = [x for x in items if isinstance(x, dict) and x.get("type") == "file"]
    files = [f for f in files if re.match(pattern_regex, f.get("name", ""), flags=re.IGNORECASE)]
    if not files:
        return None, None

    scored = []
    for f in files:
        name = f.get("name", "")
        dk = _extract_date_key_from_filename(name)
        scored.append((dk if dk is not None else (0, 0, 0), f))

    any_dated = any(k != (0, 0, 0) for k, _ in scored)
    if any_dated:
        scored = [sp for sp in scored if sp[0] != (0, 0, 0)]
        scored.sort(key=lambda x: x[0])
        best = scored[-1][1]
    else:
        best = sorted(files, key=lambda x: x.get("name", ""))[-1]

    content = github_download_raw(best["download_url"], token)
    return best["name"], content


# -----------------------------
# Robust weekly KenPom loader
# -----------------------------
def load_kenpom_excel(uploaded_file: io.BytesIO) -> pd.DataFrame:
    raw = pd.read_excel(uploaded_file, header=None)

    header_row_idx = None
    scan_rows = min(20, len(raw))
    for i in range(scan_rows):
        row = raw.iloc[i].astype(str).str.strip().tolist()
        has_stats = ("ORtg" in row) and ("DRtg" in row) and ("AdjT" in row)
        has_teamish = any(("team" in c.lower()) or ("school" in c.lower()) for c in row)
        if has_stats and has_teamish:
            header_row_idx = i
            break

    if header_row_idx is None:
        preview = raw.head(6).astype(str).values.tolist()
        raise ValueError(
            "Could not find a header row containing Team/School + ORtg/DRtg/AdjT. "
            f"Top rows preview: {preview}"
        )

    df = pd.read_excel(uploaded_file, header=header_row_idx)
    df.columns = [str(c).strip() for c in df.columns]

    team_col_candidates = [
        c for c in df.columns
        if c.lower() in ["team", "teamname", "school", "team name", "team_name"]
        or "team" in c.lower()
        or "school" in c.lower()
    ]
    if not team_col_candidates:
        raise ValueError(f"No team column found. Columns detected: {df.columns.tolist()}")

    team_col = team_col_candidates[0]

    for col in ["ORtg", "DRtg", "AdjT"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found. Found: {df.columns.tolist()}")

    # --- HARD-CODED SoS COLUMNS by position (Excel letters) ---
    # Excel: N (Net SoS), P (Off SoS), R (Def SoS)
    # pandas iloc is 0-indexed: N=13, P=15, R=17
    try:
        sos_net = pd.to_numeric(df.iloc[:, 13], errors="coerce")  # Column N
        sos_off = pd.to_numeric(df.iloc[:, 15], errors="coerce")  # Column P
        sos_def = pd.to_numeric(df.iloc[:, 17], errors="coerce")  # Column R
    except Exception:
        sos_net = pd.Series(np.nan, index=df.index)
        sos_off = pd.Series(np.nan, index=df.index)
        sos_def = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({
        "Team": df[team_col].astype(str).str.strip(),
        "ORtg": pd.to_numeric(df["ORtg"], errors="coerce"),
        "DRtg": pd.to_numeric(df["DRtg"], errors="coerce"),
        "AdjT": pd.to_numeric(df["AdjT"], errors="coerce"),
        "SOS_NET": sos_net,
        "SOS_OFF": sos_off,
        "SOS_DEF": sos_def,
    })

    # --- Blended SoS (stable, low variance) ---
    out["SOS_BLEND"] = (0.50 * out["SOS_NET"]) + (0.25 * out["SOS_OFF"]) + (0.25 * out["SOS_DEF"])

    # Derived metrics (tagging only; does NOT alter predictions)
    out["NetRtg"] = out["ORtg"] - out["DRtg"]
    tags = out.apply(lambda r: tag_team_style(r["AdjT"], r["NetRtg"]), axis=1)
    out["In_Trapezoid"] = [t[0] for t in tags]
    out["Fraud"] = [t[1] for t in tags]
    out["Style_Tag"] = [t[2] for t in tags]

    out = out.dropna(subset=["Team", "ORtg", "DRtg", "AdjT"])
    out = out[out["Team"] != ""].drop_duplicates(subset=["Team"]).reset_index(drop=True)

    return out


# -----------------------------
# Schedule loader
# -----------------------------
def load_schedule_excel(uploaded_file: io.BytesIO) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {c.lower(): c for c in df.columns}
    visitor_col = None
    home_col = None

    for key in ["visitor", "away", "visiting", "road"]:
        if key in col_map:
            visitor_col = col_map[key]
            break
    for key in ["home", "hometeam", "home team"]:
        if key in col_map:
            home_col = col_map[key]
            break

    if visitor_col is None:
        for c in df.columns:
            if "visitor" in c.lower() or "away" in c.lower():
                visitor_col = c
                break
    if home_col is None:
        for c in df.columns:
            if "home" in c.lower():
                home_col = c
                break

    if visitor_col is None or home_col is None:
        raise ValueError(
            f"Schedule file must contain Visitor/Away and Home columns. Found: {df.columns.tolist()}"
        )

    df = df.rename(columns={visitor_col: "Visitor", home_col: "Home"})
    df["Visitor"] = df["Visitor"].astype(str).str.strip()
    df["Home"] = df["Home"].astype(str).str.strip()

    df = df.dropna(subset=["Visitor", "Home"])
    df = df[(df["Visitor"] != "") & (df["Home"] != "")]
    df = df.reset_index(drop=True)

    return df


# -----------------------------
# Team mapping (Schedule -> KenPom)
# -----------------------------
ALIAS_MAP = {
    "american university": "American",
    "app state": "Appalachian St.",
    "siu edwardsville": "SIUE",
    "ut martin": "Tennessee Martin",
    "long beach state": "Long Beach St.",
    "delaware state": "Delaware St.",
    "georgia state": "Georgia St.",
    "idaho state": "Idaho St.",
    "illinois state": "Illinois St.",
    "indiana state": "Indiana St.",
    "jackson state": "Jackson St.",
    "murray state": "Murray St.",
    "norfolk state": "Norfolk St.",
    "oklahoma state": "Oklahoma St.",
    "tennessee state": "Tennessee St.",
    "southeast missouri state": "Southeast Missouri",
    "cal state bakersfield": "Cal St. Bakersfield",
}
STOPWORDS = {"university", "college", "the", "at", "of"}


def normalize_team_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\bst\.\b", "st", s)
    s = re.sub(r"\bst\b", "state", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = [p for p in s.split() if p not in STOPWORDS]
    return " ".join(parts)


def build_kp_lookup(df_kp: pd.DataFrame):
    kp_teams = df_kp["Team"].astype(str).str.strip().tolist()
    kp_set = set(kp_teams)
    norm_to_team = {}
    for t in kp_teams:
        k = normalize_team_name(t)
        norm_to_team.setdefault(k, t)
    return kp_teams, kp_set, norm_to_team


def map_to_kenpom(team_name: str, kp_set: set[str], norm_to_team: dict, fuzzy_cutoff: float = 0.86):
    """
    Returns: (mapped_name or None, score 0..1, method)
    """
    original = str(team_name).strip()
    if not original or original.lower() == "nan":
        return None, 0.0, "empty"

    # exact
    if original in kp_set:
        return original, 1.0, "exact"

    # alias
    alias_key = original.lower()
    if alias_key in ALIAS_MAP and ALIAS_MAP[alias_key] in kp_set:
        return ALIAS_MAP[alias_key], 1.0, "alias"

    # normalized exact
    n = normalize_team_name(original)
    if n in norm_to_team:
        return norm_to_team[n], 0.98, "normalized"

    # fuzzy on normalized keys
    norm_keys = list(norm_to_team.keys())
    best = difflib.get_close_matches(n, norm_keys, n=1, cutoff=fuzzy_cutoff)
    if best:
        best_key = best[0]
        score = difflib.SequenceMatcher(None, n, best_key).ratio()
        return norm_to_team[best_key], float(score), "fuzzy"

    return None, 0.0, "unmatched"


# -----------------------------
# Odds parsing helpers
# -----------------------------
def build_team_abbrev_candidates(team_name: str) -> set[str]:
    name = str(team_name).strip().upper()
    words = re.findall(r"[A-Z0-9]+", name)
    joined = "".join(words)

    cands = set()
    if joined:
        cands.update([joined, joined[:3], joined[:4], joined[:5]])

    if words:
        initials = "".join(w[0] for w in words if w)
        if initials:
            cands.update([initials, initials[:3], initials[:4]])

    if len(words) >= 2:
        two_each = "".join(w[:2] for w in words if len(w) >= 2)
        if two_each:
            cands.update([two_each, two_each[:3], two_each[:4], two_each[:5]])
        three_each = "".join(w[:3] for w in words if len(w) >= 3)
        if three_each:
            cands.update([three_each, three_each[:3], three_each[:4], three_each[:5]])

    if words:
        cands.update([words[0][:3], words[0][:4], words[0][:5]])

    return {c for c in cands if c and c != "NAN"}


def parse_dk_odds(odds_text) -> dict:
    if odds_text is None or (isinstance(odds_text, float) and np.isnan(odds_text)):
        return {"DK_Fav": np.nan, "DK_Line_Fav": np.nan, "DK_Total": np.nan}

    s = str(odds_text).strip()
    if not s or s.lower() == "nan":
        return {"DK_Fav": np.nan, "DK_Line_Fav": np.nan, "DK_Total": np.nan}

    m_line = re.search(r"Line:\s*([A-Za-z0-9]+)\s*([+\-]?\d+(\.\d+)?)", s)
    fav, line = np.nan, np.nan
    if m_line:
        fav = m_line.group(1).upper().strip()
        line = float(m_line.group(2))

    m_ou = re.search(r"O\/U[:\s]*([+\-]?\d+(\.\d+)?)", s)
    total = np.nan
    if m_ou:
        total = float(m_ou.group(1))

    return {"DK_Fav": fav, "DK_Line_Fav": line, "DK_Total": total}


def derive_home_spread(visitor: str, home: str, dk_fav: str, dk_line_fav: float) -> float:
    if dk_fav is None or (isinstance(dk_fav, float) and np.isnan(dk_fav)):
        return np.nan
    if dk_line_fav is None or (isinstance(dk_line_fav, float) and np.isnan(dk_line_fav)):
        return np.nan

    fav = str(dk_fav).upper().strip()
    home_cands = build_team_abbrev_candidates(home)
    away_cands = build_team_abbrev_candidates(visitor)

    fav_is_home = fav in home_cands
    fav_is_away = fav in away_cands

    if fav_is_home and not fav_is_away:
        return float(dk_line_fav)
    if fav_is_away and not fav_is_home:
        return float(-dk_line_fav)

    return np.nan


# -----------------------------
# SoS modifier (margin-only) — DOUBLE current effect (4× vs original)
# -----------------------------
def sos_margin_adjustment_pts(
    sos_home: float,
    sos_away: float,
    possessions: float,
    sos_weight: float,
    sos_share: float = 0.80,      # was 0.40 (2×); now doubled again
    max_margin_pts: float = 20.0,  # was 10.0 (2×); now doubled again
) -> float:
    """
    Returns a margin-only adjustment in points applied to (home - away).
    Uses blended SoS differential (home - away), scaled by sos_weight.
    Capped to keep it controlled. If SoS missing, returns 0.
    """
    if sos_weight is None or np.isnan(sos_weight) or sos_weight <= 0:
        return 0.0
    if any(pd.isna(x) for x in [sos_home, sos_away, possessions]):
        return 0.0

    sos_diff = float(sos_home) - float(sos_away)  # home - away
    adj_pts = sos_diff * sos_share * float(sos_weight) * (float(possessions) / 100.0)
    return float(np.clip(adj_pts, -max_margin_pts, max_margin_pts))


# -----------------------------
# Core prediction logic (spreadsheet structure)
# -----------------------------
def predict_matchup(
    team_away: str,
    team_home: str,
    df_kp: pd.DataFrame,
    home_edge_points: float = 0.0,
    off_scale: float = 1.0,
    def_scale: float = 1.0,
    tempo_scale: float = 1.0,
    sos_weight: float = 0.0,
):
    away_row = df_kp.loc[df_kp["Team"] == team_away].iloc[0]
    home_row = df_kp.loc[df_kp["Team"] == team_home].iloc[0]

    OR_away = away_row["ORtg"] * off_scale
    DR_away = away_row["DRtg"] * def_scale
    T_away = away_row["AdjT"] * tempo_scale

    OR_home = home_row["ORtg"] * off_scale
    DR_home = home_row["DRtg"] * def_scale
    T_home = home_row["AdjT"] * tempo_scale

    possessions = (T_away * T_home) / AVG_TEMPO

    away_pts_neutral = (OR_away / AVG_EFF) * (DR_home / AVG_EFF) * AVG_EFF * (possessions / 100.0)
    home_pts_neutral = (OR_home / AVG_EFF) * (DR_away / AVG_EFF) * AVG_EFF * (possessions / 100.0)

    away_pts = away_pts_neutral
    home_pts = home_pts_neutral + home_edge_points

    # --- HARD-CODED SoS (blended) : margin-only adjustment ---
    sos_home = home_row.get("SOS_BLEND", np.nan)
    sos_away = away_row.get("SOS_BLEND", np.nan)

    margin_adj_pts = sos_margin_adjustment_pts(
        sos_home=sos_home,
        sos_away=sos_away,
        possessions=possessions,
        sos_weight=sos_weight,
    )

    # Split margin-only adjustment so total stays stable
    home_pts += (margin_adj_pts / 2.0)
    away_pts -= (margin_adj_pts / 2.0)

    margin_home = home_pts - away_pts
    total_pts = home_pts + away_pts

    return {
        "Away": team_away,
        "Home": team_home,
        "Possessions": float(np.round(possessions, 1)),
        "Pred_Away": float(np.round(away_pts, 1)),
        "Pred_Home": float(np.round(home_pts, 1)),
        "Home_Margin": float(np.round(margin_home, 1)),
        "Total": float(np.round(total_pts, 1)),
        "SoS_Away": float(sos_away) if pd.notna(sos_away) else np.nan,
        "SoS_Home": float(sos_home) if pd.notna(sos_home) else np.nan,
        "SoS_MarginAdj_Pts": float(np.round(margin_adj_pts, 2)),
    }


def run_schedule(
    schedule_df: pd.DataFrame,
    df_kp: pd.DataFrame,
    home_edge_points: float,
    off_scale: float,
    def_scale: float,
    tempo_scale: float,
    spread_edge_threshold: float,
    total_edge_threshold: float,
    fuzzy_cutoff: float,
    sos_weight: float,
):
    kp_teams, kp_set, norm_to_team = build_kp_lookup(df_kp)

    results_rows = []
    for _, r in schedule_df.iterrows():
        visitor_raw = r["Visitor"]
        home_raw = r["Home"]
        base = r.to_dict()

        # --- Map schedule team names to KenPom team names ---
        v_map, v_score, v_method = map_to_kenpom(visitor_raw, kp_set, norm_to_team, fuzzy_cutoff=fuzzy_cutoff)
        h_map, h_score, h_method = map_to_kenpom(home_raw, kp_set, norm_to_team, fuzzy_cutoff=fuzzy_cutoff)

        base["Visitor_Mapped"] = v_map if v_map else ""
        base["Home_Mapped"] = h_map if h_map else ""
        base["Visitor_MapScore"] = float(np.round(v_score, 3))
        base["Home_MapScore"] = float(np.round(h_score, 3))
        base["Visitor_MapMethod"] = v_method
        base["Home_MapMethod"] = h_method

        # Parse odds regardless (so unmapped rows still show vegas)
        odds_text = base.get("Odds by draft kings", np.nan)
        dk = parse_dk_odds(odds_text)
        base.update(dk)
        base["Spread_Home"] = derive_home_spread(visitor_raw, home_raw, dk.get("DK_Fav"), dk.get("DK_Line_Fav"))

        if not v_map or not h_map:
            base.update({
                "Possessions": np.nan,
                "Pred_Away": np.nan,
                "Pred_Home": np.nan,
                "Home_Margin": np.nan,
                "Total": np.nan,
                "SoS_Away": np.nan,
                "SoS_Home": np.nan,
                "SoS_MarginAdj_Pts": np.nan,
                "Edge_Spread": np.nan,
                "Edge_Total": np.nan,
                "Spread_Play": False,
                "Total_Play": False,
                "Visitor_NetRtg": np.nan,
                "Home_NetRtg": np.nan,
                "Visitor_InTrapezoid": False,
                "Home_InTrapezoid": False,
                "Visitor_Fraud": False,
                "Home_Fraud": False,
                "Visitor_Tag": "",
                "Home_Tag": "",
                "Map_Status": "UNMAPPED (needs alias/manual fix)"
            })
            results_rows.append(base)
            continue

        base["Map_Status"] = "OK"

        # --- Style tags (no impact on calculations) ---
        try:
            v_style = df_kp.loc[df_kp["Team"] == v_map, ["NetRtg", "AdjT", "In_Trapezoid", "Fraud", "Style_Tag"]].iloc[0]
            h_style = df_kp.loc[df_kp["Team"] == h_map, ["NetRtg", "AdjT", "In_Trapezoid", "Fraud", "Style_Tag"]].iloc[0]
            base["Visitor_NetRtg"] = float(np.round(v_style["NetRtg"], 2))
            base["Home_NetRtg"] = float(np.round(h_style["NetRtg"], 2))
            base["Visitor_InTrapezoid"] = bool(v_style["In_Trapezoid"])
            base["Home_InTrapezoid"] = bool(h_style["In_Trapezoid"])
            base["Visitor_Fraud"] = bool(v_style["Fraud"])
            base["Home_Fraud"] = bool(h_style["Fraud"])
            base["Visitor_Tag"] = str(v_style["Style_Tag"])
            base["Home_Tag"] = str(h_style["Style_Tag"])
        except Exception:
            base["Visitor_NetRtg"] = np.nan
            base["Home_NetRtg"] = np.nan
            base["Visitor_InTrapezoid"] = False
            base["Home_InTrapezoid"] = False
            base["Visitor_Fraud"] = False
            base["Home_Fraud"] = False
            base["Visitor_Tag"] = ""
            base["Home_Tag"] = ""

        # --- Predict using mapped names ---
        pred = predict_matchup(
            team_away=v_map,
            team_home=h_map,
            df_kp=df_kp,
            home_edge_points=home_edge_points,
            off_scale=off_scale,
            def_scale=def_scale,
            tempo_scale=tempo_scale,
            sos_weight=sos_weight,
        )
        base.update(pred)

        # Edge_Spread = Model(Home_Margin) + Spread_Home
        if isinstance(base.get("Spread_Home"), (int, float)) and not np.isnan(base["Spread_Home"]):
            base["Edge_Spread"] = float(np.round(base["Home_Margin"] + base["Spread_Home"], 2))
        else:
            base["Edge_Spread"] = np.nan

        # Edge_Total = Model(Total) - Vegas(O/U)
        if isinstance(base.get("DK_Total"), (int, float)) and not np.isnan(base["DK_Total"]):
            base["Edge_Total"] = float(np.round(base["Total"] - base["DK_Total"], 2))
        else:
            base["Edge_Total"] = np.nan

        base["Spread_Play"] = bool(
            isinstance(base.get("Edge_Spread"), (int, float)) and not np.isnan(base["Edge_Spread"])
            and abs(base["Edge_Spread"]) >= spread_edge_threshold
        )
        base["Total_Play"] = bool(
            isinstance(base.get("Edge_Total"), (int, float)) and not np.isnan(base["Edge_Total"])
            and abs(base["Edge_Total"]) >= total_edge_threshold
        )

        results_rows.append(base)

    results_df = pd.DataFrame(results_rows)

    preferred = [
        "Visitor", "Home", "Visitor_Tag", "Home_Tag", "Visitor_NetRtg", "Home_NetRtg",
        "Visitor_InTrapezoid", "Home_InTrapezoid", "Visitor_Fraud", "Home_Fraud",
        "Visitor_Mapped", "Home_Mapped", "Visitor_MapScore", "Home_MapScore",
        "Visitor_MapMethod", "Home_MapMethod", "Map_Status",
        "TIME", "TV", "location",
        "DK_Fav", "DK_Line_Fav", "Spread_Home", "DK_Total",
        "Pred_Away", "Pred_Home", "Home_Margin", "Total",
        "SoS_Away", "SoS_Home", "SoS_MarginAdj_Pts",
        "Edge_Spread", "Edge_Total", "Spread_Play", "Total_Play",
        "Odds by draft kings",
    ]
    cols = [c for c in preferred if c in results_df.columns] + [c for c in results_df.columns if c not in preferred]
    return results_df[cols]


# -----------------------------
# Streamlit App (repo root auto-load)
# -----------------------------
def main():
    st.set_page_config(page_title="SignalAI NCAA Predictor", layout="wide")

    # ============================================================
    # STEP 3 HEADER: LOGO + TITLE ROW (repo root image)
    # ============================================================
    logo_path = REPO_ROOT / "SignalAI_Logo.png"

    col1, col2 = st.columns([1, 6])
    with col1:
        if logo_path.exists():
            st.image(str(logo_path), width=120)
        else:
            # Safe fallback (won't crash the app)
            st.write("")

    with col2:
        st.markdown("## **SignalAI NCAA Predictor**")
        st.caption("AI-powered NCAA model edges")

    st.sidebar.header("Data source (Repo Root)")

    # Auto-load from repo root
    kp_path = find_latest_local_file("kenpom_*.xls*", REPO_ROOT)
    sched_path = find_latest_local_file("Schedule_*.xls*", REPO_ROOT)

    auto_kp_bytes = None
    auto_sched_bytes = None
    auto_kp_label = None
    auto_sched_label = None

    if kp_path and kp_path.exists():
        auto_kp_bytes = kp_path.read_bytes()
        auto_kp_label = f"Local (repo root): {kp_path.name}"
    if sched_path and sched_path.exists():
        auto_sched_bytes = sched_path.read_bytes()
        auto_sched_label = f"Local (repo root): {sched_path.name}"

    # Optional GitHub fallback
    use_github_fallback = st.sidebar.toggle(
        "Enable GitHub fallback (if local files missing)",
        value=False,
        help="Uses GitHub API to find and download the newest Excel files from the repo root. "
             "Useful for private repos or if files aren't present in the build image."
    )

    if use_github_fallback and (auto_kp_bytes is None or auto_sched_bytes is None):
        if not (GITHUB_OWNER and GITHUB_REPO):
            st.sidebar.error("Set GITHUB_OWNER and GITHUB_REPO at the top of app.py to use GitHub fallback.")
        else:
            gh_token = None
            try:
                gh_token = st.secrets.get("GH_TOKEN", None)
            except Exception:
                gh_token = None

            with st.sidebar.spinner("Checking GitHub for newest files..."):
                try:
                    if auto_kp_bytes is None:
                        name, b = find_latest_github_file(
                            owner=GITHUB_OWNER,
                            repo=GITHUB_REPO,
                            folder=GITHUB_DATA_DIR,  # "" => repo root
                            pattern_regex=r"^kenpom_.*\.xls[x]?$",
                            branch=GITHUB_BRANCH,
                            token=gh_token,
                        )
                        if b:
                            auto_kp_bytes = b
                            auto_kp_label = f"GitHub (root): {name}"

                    if auto_sched_bytes is None:
                        name, b = find_latest_github_file(
                            owner=GITHUB_OWNER,
                            repo=GITHUB_REPO,
                            folder=GITHUB_DATA_DIR,  # "" => repo root
                            pattern_regex=r"^Schedule_.*\.xls[x]?$",
                            branch=GITHUB_BRANCH,
                            token=gh_token,
                        )
                        if b:
                            auto_sched_bytes = b
                            auto_sched_label = f"GitHub (root): {name}"
                except Exception as e:
                    st.sidebar.error(f"GitHub fallback failed: {e}")

    # Manual upload override (still available)
    st.sidebar.markdown("---")
    st.sidebar.caption("Auto-load is used if found. Upload overrides auto-load.")
    kp_uploaded = st.sidebar.file_uploader(
        "KenPom Excel (.xlsx) — optional override",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        key="kp_upload",
    )
    sched_uploaded = st.sidebar.file_uploader(
        "Schedule Excel (.xlsx) — optional override (schedule mode)",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        key="sched_upload_override",
    )

    # Decide final KenPom bytes
    if kp_uploaded is not None:
        kp_bytes = kp_uploaded.getvalue()
        kp_source = "Manual upload"
    else:
        kp_bytes = auto_kp_bytes
        kp_source = auto_kp_label or "Not found"

    if kp_bytes is None:
        st.info(
            "No KenPom file found in repo root.\n\n"
            "Fix: commit a file named like `kenpom_1.10.26.xlsx` into the repo root (same folder as app.py),\n"
            "or upload manually in the sidebar."
        )
        return

    # ------------------------------------------------------------
    # COMMENTED OUT PER YOUR REQUEST (keep in code, don't show UI)
    # st.caption(f"KenPom source: **{kp_source}**")
    # ------------------------------------------------------------

    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.header("Adjustments")
    home_edge_points = st.sidebar.slider("Home edge (pts added to HOME score)", -10.0, 10.0, 3.0, 0.5)
    off_scale = st.sidebar.slider("Offense scale (ORtg multiplier)", 0.80, 1.20, 1.00, 0.01)
    def_scale = st.sidebar.slider(
        "Defense scale (DRtg multiplier)",
        0.80, 1.20, 1.00, 0.01,
        help=">1.00 makes DRtg larger (worse defense) -> increases opponent points."
    )
    tempo_scale = st.sidebar.slider("Tempo scale (AdjT multiplier)", 0.80, 1.20, 1.00, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Strength of Schedule (hardcoded columns N/P/R) — 4× impact")
    sos_weight = st.sidebar.slider(
        "SoS weight (0 = off, 1 = max effect)",
        0.0, 1.0, 0.20, 0.05,
        help="Applies a capped margin-only adjustment using SoS differential from KenPom columns "
             "N=Net SoS, P=Off SoS, R=Def SoS. This build is ~4× the original impact."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Bet Flag Thresholds")
    spread_edge_threshold = st.sidebar.slider("Spread play threshold X", 0.0, 20.0, 2.0, 0.5)
    total_edge_threshold = st.sidebar.slider("Total play threshold Y", 0.0, 30.0, 3.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Team Mapping")
    fuzzy_cutoff = st.sidebar.slider(
        "Fuzzy match strictness (higher = stricter)",
        0.70, 0.95, 0.86, 0.01
    )
    st.sidebar.caption("If teams fail to map, lower strictness slightly or add to ALIAS_MAP.")

    # Load KenPom
    try:
        df_kp = load_kenpom_excel(io.BytesIO(kp_bytes))
    except Exception as e:
        st.error(f"Error loading KenPom file: {e}")
        st.stop()

    sos_ok = int(df_kp["SOS_BLEND"].notna().sum())
    if sos_ok == 0:
        st.warning(
            "SoS_BLEND is missing for all teams. "
            "Your hardcoded columns (N/P/R) may not exist in this upload after the header row. "
            "SoS weight will have no effect."
        )
    else:
        # ------------------------------------------------------------
        # COMMENTED OUT PER YOUR REQUEST (keep in code, don't show UI)
        # st.caption(f"SoS loaded for {sos_ok} teams (SOS_BLEND from columns N/P/R).")
        # ------------------------------------------------------------
        pass

    mode = st.radio("Select evaluation mode:", ["Single matchup", "Run full daily schedule"], horizontal=True)

    if mode == "Single matchup":
        teams = sorted(df_kp["Team"].unique().tolist())
        c1, c2 = st.columns(2)
        with c1:
            away_team = st.selectbox("Visiting (Away) Team", teams, index=0)
        with c2:
            home_team = st.selectbox("Home Team", teams, index=1 if len(teams) > 1 else 0)

        if away_team == home_team:
            st.warning("Please choose two different teams.")
            return

        result = predict_matchup(
            team_away=away_team,
            team_home=home_team,
            df_kp=df_kp,
            home_edge_points=home_edge_points,
            off_scale=off_scale,
            def_scale=def_scale,
            tempo_scale=tempo_scale,
            sos_weight=sos_weight,
        )

        st.markdown("## Predicted Score")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{result['Away']} (Away)", f"{result['Pred_Away']:.1f}")
        m2.metric(f"{result['Home']} (Home)", f"{result['Pred_Home']:.1f}")
        m3.metric("Total Points", f"{result['Total']:.1f}")
        m4.metric("Home Margin (Home - Away)", f"{result['Home_Margin']:.1f}")

        # with st.expander("SoS modifier (debug)"):
            # st.write({
                # "SoS_Away (blend)": result.get("SoS_Away", np.nan),
                # "SoS_Home (blend)": result.get("SoS_Home", np.nan),
                # "SoS_MarginAdj_Pts (home - away)": result.get("SoS_MarginAdj_Pts", np.nan),
                # "SoS_weight": sos_weight,
                # "Note": "Margin-only adjustment is split: +adj/2 to home and -adj/2 to away. Totals stay stable."
            })

        with st.expander("KenPom data preview"):
            st.dataframe(df_kp.head(25), use_container_width=True)

    else:
        st.markdown("## Run Full Daily Schedule")

        # Schedule bytes: manual override wins, else repo root auto-load
        if sched_uploaded is not None:
            sched_bytes = sched_uploaded.getvalue()
            sched_source = "Manual upload"
        else:
            sched_bytes = auto_sched_bytes
            sched_source = auto_sched_label or "Not found"

        st.caption(f"Schedule source: **{sched_source}**")

        if sched_bytes is None:
            st.info(
                "No schedule file found in repo root.\n\n"
                "Fix: commit a file named like `Schedule_1.10.26.xlsx` into the repo root (same folder as app.py),\n"
                "or upload manually in the sidebar."
            )
            return

        try:
            schedule_df = load_schedule_excel(io.BytesIO(sched_bytes))
        except Exception as e:
            st.error(f"Error loading schedule file: {e}")
            st.stop()

        st.write(f"Games found: **{len(schedule_df)}**")
        with st.expander("Schedule preview"):
            st.dataframe(schedule_df.head(50), use_container_width=True)

        if st.button("Run all games", type="primary"):
            results_df = run_schedule(
                schedule_df=schedule_df,
                df_kp=df_kp,
                home_edge_points=home_edge_points,
                off_scale=off_scale,
                def_scale=def_scale,
                tempo_scale=tempo_scale,
                spread_edge_threshold=spread_edge_threshold,
                total_edge_threshold=total_edge_threshold,
                fuzzy_cutoff=fuzzy_cutoff,
                sos_weight=sos_weight,
            )

            st.markdown("### Full Results (mapping + edges + flags)")
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv_bytes,
                file_name="schedule_predictions_full.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

