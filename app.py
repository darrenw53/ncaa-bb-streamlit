import io
import re
import difflib
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Optional (only needed if you enable GitHub fallback)
try:
    import requests
except Exception:
    requests = None


# ============================================================
# CONFIG: REPO ROOT AUTO-LOADING
# ============================================================
REPO_ROOT = Path(__file__).parent

LOGO_FILENAME = "SignalAI_Logo.png"
LOGO_PATH = REPO_ROOT / LOGO_FILENAME

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
# Helpers
# =============================
def fmt_num(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def pick_side_from_edge(edge_spread):
    if edge_spread is None or (isinstance(edge_spread, float) and np.isnan(edge_spread)):
        return ""
    return "HOME" if edge_spread > 0 else "VISITOR"


def pick_total_from_edge(edge_total):
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


def tier_from_edge_abs(edge_abs: float) -> str:
    """Tier string based on your confidence thresholds."""
    if edge_abs is None or (isinstance(edge_abs, float) and np.isnan(edge_abs)):
        return ""
    if edge_abs >= 6:
        return "Strong"
    if edge_abs >= 4:
        return "Solid"
    if edge_abs >= 2:
        return "Lean"
    return ""


def units_from_tier(tier: str) -> float:
    """Recommended units per tier (simple v1)."""
    t = (tier or "").lower().strip()
    if t == "strong":
        return 1.5
    if t == "solid":
        return 1.0
    if t == "lean":
        return 0.5
    return 0.0


# =============================
# Win probability helpers
# =============================
def win_prob_from_margin(margin_pts: float, scale: float = 7.5) -> float:
    """
    Converts a predicted margin (Home - Away) into a win probability for HOME.
    """
    if margin_pts is None or (isinstance(margin_pts, float) and np.isnan(margin_pts)):
        return np.nan
    try:
        x = float(margin_pts) / float(scale)
        return float(1.0 / (1.0 + np.exp(-x)))
    except Exception:
        return np.nan


def pct_str(p: float, nd: int = 1) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return f"{100.0 * float(p):.{nd}f}%"


# =============================
# Login gate (Option 1) — logo + password page
# =============================
def require_login(
    logo_path: Path,
    app_name: str = "SignalAI NCAA Predictor",
    subscribe_url: str = "",
) -> bool:
    """
    Simple password gate for Streamlit.
    Uses st.secrets["APP_PASSWORD"].

    Returns True if user is authenticated for this session.
    """
    if st.session_state.get("is_authed", False):
        return True

    # Read password from Streamlit secrets (recommended)
    app_password = None
    try:
        app_password = st.secrets.get("APP_PASSWORD", None)
    except Exception:
        app_password = None

    if not app_password:
        st.set_page_config(page_title=app_name, layout="wide")
        st.error("APP_PASSWORD is not set. Add it to Streamlit secrets to enable login.")
        st.stop()

    # Login screen
    st.set_page_config(page_title=app_name, layout="wide")

    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        if logo_path and logo_path.exists():
            st.image(str(logo_path), use_container_width=True)

        st.markdown(
            f"""
            <div style="
                padding: 18px 18px 8px 18px;
                border: 1px solid rgba(0,0,0,0.08);
                border-radius: 16px;
                box-shadow: 0 10px 26px rgba(0,0,0,0.05);
                background: white;
                ">
              <div style="font-size:22px;font-weight:900;letter-spacing:-0.2px;margin-bottom:6px;">
                {app_name}
              </div>
              <div style="opacity:0.75;font-weight:700;margin-bottom:14px;">
                Subscriber Login
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("login_form", clear_on_submit=False):
            pw = st.text_input("Password", type="password", placeholder="Enter subscriber password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            if str(pw).strip() == str(app_password).strip():
                st.session_state["is_authed"] = True
                st.session_state["auth_error"] = ""
                st.rerun()
            else:
                st.session_state["auth_error"] = "Invalid password. Please try again."

        err = st.session_state.get("auth_error", "")
        if err:
            st.error(err)

        if subscribe_url:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="opacity:0.75;font-weight:700;">
                  Need access? Subscribe here:
                </div>
                """,
                unsafe_allow_html=True
            )
            st.link_button("Subscribe", subscribe_url)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.caption("Entertainment only. Not financial advice.")

    return False


# =============================
# Subscriber Guide (in-app help page)
# =============================
def _get_query_param(name: str, default: str = "") -> str:
    """Compatibility wrapper for Streamlit query params."""
    try:
        # Newer Streamlit
        v = st.query_params.get(name, default)
        if isinstance(v, list):
            return v[0] if v else default
        return str(v) if v is not None else default
    except Exception:
        try:
            # Older Streamlit
            v = st.experimental_get_query_params().get(name, [default])
            return v[0] if v else default
        except Exception:
            return default


def render_subscriber_guide(app_name: str = "SignalAI NCAA Predictor"):
    st.markdown(f"# {app_name} — Subscriber Guide")
    st.caption("Quick, practical instructions for subscribers. For entertainment purposes only.")

    st.markdown("---")
    st.markdown("## Getting started")
    st.markdown(
        """

- **Log in** with the subscriber password.

- Use the left sidebar to adjust sliders (home edge, offense/defense/tempo, SoS, thresholds, unit size).

- Choose a mode at the top: **Single matchup**, **Run full daily schedule**, or **Team dashboard**.
""")

    st.markdown("## What the main numbers mean")
    st.markdown(
        """

- **Pred_Away / Pred_Home**: the model’s predicted score.

- **Home_Margin**: predicted (Home − Away).

- **Total**: predicted combined points.

- **Win%**: model win probability for each team (based on predicted margin).
""")

    st.markdown("## Understanding edges and picks")
    st.markdown(
        """

- **Spread_Home**: the posted spread expressed as *home team spread* (negative = home favored).

- **Edge_Spread**: how far the model is from the spread.

  - Positive → model likes **HOME** against the spread.

  - Negative → model likes **VISITOR** against the spread.

- **DK_Total**: posted total points.

- **Edge_Total**: model total − posted total.

  - Positive → lean **OVER**.

  - Negative → lean **UNDER**.
""")

    st.markdown("## Confidence tiers and staking")
    st.markdown(
        """

- A play becomes **flagged** when its absolute edge clears your thresholds:

  - **Spread_Play** if |Edge_Spread| ≥ your spread threshold.

  - **Total_Play** if |Edge_Total| ≥ your total threshold.

- **Confidence label** is based on the *largest* edge (spread or total):

  - 👍 **Lean** (≥ 2)

  - ✅ **Solid** (≥ 4)

  - 🔥 **Strong** (≥ 6)

- **Recommended Units** (flagged plays only):

  - Lean = 0.5u, Solid = 1.0u, Strong = 1.5u

- **Stake_$** = Rec_Units × your selected **Unit size ($)**.
""")

    st.markdown("## Strength of Schedule (SoS) slider")
    st.markdown(
        """

- **SoS weight** adds a *margin-only* adjustment based on SoS differential.

- It nudges the predicted margin (and therefore win%) without changing your inputs.

- If you want SoS to have **no influence**, set **SoS weight = 0**.
""")

    st.markdown("## Mode walkthroughs")
    with st.expander("Single matchup", expanded=True):
        st.markdown(
            """

1) Pick the **Away** and **Home** team.

2) Read the predicted score, total, margin, and win%.

3) Adjust sliders to see how predictions move.
""")

    with st.expander("Run full daily schedule", expanded=True):
        st.markdown(
            """

1) Choose display options:

   - **Show only flagged plays**: shows only plays that clear your thresholds.

   - **Require full data**: filters out games missing lines/totals.

   - **Max cards**: how many matchup cards to display.

2) Click **Run all games**.

3) Review:

   - **Exposure summary** (flagged plays only)

   - **Cards view** (quick read)

   - **Full Results table** (everything)

4) Download the results CSV if you want to archive or share.
""")

    with st.expander("Team dashboard", expanded=True):
        st.markdown(
            """

1) Select a team.

2) Review the team profile (ORtg/DRtg/Net/Tempo/SoS).

3) See that team’s games from the most recent schedule run.

   - If it says no schedule results are loaded, go run **Run full daily schedule** first.
""")

    st.markdown("## Tags: TRAP and FRAUD")
    st.markdown(
        """

- Some teams may display style tags:

  - **🏆 TRAP**: fits the trapezoid profile.

  - **🚨 FRAUD**: flagged as fraud by the pace/net filters.

These are informational — use them however you like in your process.
""")

    st.markdown("---")
    st.caption("Entertainment only. Not financial advice.")


# =============================
# Share-card helpers (HTML) — kept as-is
# =============================
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
    s = name.lower()

    m = re.search(r"(20\d{2})[-_.](\d{1,2})[-_.](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (y, mo, d)

    m = re.search(r"(\d{1,2})[.](\d{1,2})[.](\d{2,4})", s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        return (y, mo, d)

    return None


def find_latest_local_file(pattern: str, folder: Path) -> Path | None:
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
    "Connecticut": "UConn",
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
    original = str(team_name).strip()
    if not original or original.lower() == "nan":
        return None, 0.0, "empty"

    if original in kp_set:
        return original, 1.0, "exact"

    alias_key = original.lower()
    if alias_key in ALIAS_MAP and ALIAS_MAP[alias_key] in kp_set:
        return ALIAS_MAP[alias_key], 1.0, "alias"

    n = normalize_team_name(original)
    if n in norm_to_team:
        return norm_to_team[n], 0.98, "normalized"

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
# SoS modifier (margin-only) — 4× impact
# -----------------------------
def sos_margin_adjustment_pts(
    sos_home: float,
    sos_away: float,
    possessions: float,
    sos_weight: float,
    sos_share: float = 0.80,
    max_margin_pts: float = 20.0,
) -> float:
    if sos_weight is None or np.isnan(sos_weight) or sos_weight <= 0:
        return 0.0
    if any(pd.isna(x) for x in [sos_home, sos_away, possessions]):
        return 0.0

    sos_diff = float(sos_home) - float(sos_away)
    adj_pts = sos_diff * sos_share * float(sos_weight) * (float(possessions) / 100.0)
    return float(np.clip(adj_pts, -max_margin_pts, max_margin_pts))


# -----------------------------
# Core prediction logic
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

    sos_home = home_row.get("SOS_BLEND", np.nan)
    sos_away = away_row.get("SOS_BLEND", np.nan)

    margin_adj_pts = sos_margin_adjustment_pts(
        sos_home=sos_home,
        sos_away=sos_away,
        possessions=possessions,
        sos_weight=sos_weight,
    )

    home_pts += (margin_adj_pts / 2.0)
    away_pts -= (margin_adj_pts / 2.0)

    margin_home = home_pts - away_pts
    total_pts = home_pts + away_pts

    home_win_prob = win_prob_from_margin(margin_home, scale=7.5)
    away_win_prob = (1.0 - home_win_prob) if pd.notna(home_win_prob) else np.nan

    return {
        "Away": team_away,
        "Home": team_home,
        "Possessions": float(np.round(possessions, 1)),
        "Pred_Away": float(np.round(away_pts, 1)),
        "Pred_Home": float(np.round(home_pts, 1)),
        "Home_Margin": float(np.round(margin_home, 1)),
        "Total": float(np.round(total_pts, 1)),
        "Home_WinProb": float(np.round(home_win_prob, 6)) if pd.notna(home_win_prob) else np.nan,
        "Away_WinProb": float(np.round(away_win_prob, 6)) if pd.notna(away_win_prob) else np.nan,
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
    _, kp_set, norm_to_team = build_kp_lookup(df_kp)

    results_rows = []
    for _, r in schedule_df.iterrows():
        visitor_raw = r["Visitor"]
        home_raw = r["Home"]
        base = r.to_dict()

        v_map, v_score, v_method = map_to_kenpom(visitor_raw, kp_set, norm_to_team, fuzzy_cutoff=fuzzy_cutoff)
        h_map, h_score, h_method = map_to_kenpom(home_raw, kp_set, norm_to_team, fuzzy_cutoff=fuzzy_cutoff)

        base["Visitor_Mapped"] = v_map if v_map else ""
        base["Home_Mapped"] = h_map if h_map else ""
        base["Visitor_MapScore"] = float(np.round(v_score, 3))
        base["Home_MapScore"] = float(np.round(h_score, 3))
        base["Visitor_MapMethod"] = v_method
        base["Home_MapMethod"] = h_method

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
                "Home_WinProb": np.nan,
                "Away_WinProb": np.nan,
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

        if isinstance(base.get("Spread_Home"), (int, float)) and not np.isnan(base["Spread_Home"]):
            base["Edge_Spread"] = float(np.round(base["Home_Margin"] + base["Spread_Home"], 2))
        else:
            base["Edge_Spread"] = np.nan

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
        "Pred_Away", "Pred_Home", "Home_WinProb", "Away_WinProb", "Home_Margin", "Total",
        "SoS_Away", "SoS_Home", "SoS_MarginAdj_Pts",
        "Edge_Spread", "Edge_Total", "Spread_Play", "Total_Play",
        "Odds by draft kings",
    ]
    cols = [c for c in preferred if c in results_df.columns] + [c for c in results_df.columns if c not in preferred]
    return results_df[cols]


# =============================
# Schedule Cards (Option B) — RENDER VIA components.html
# =============================
SIG_CARDS_CSS = """
<style>
  body { margin: 0; padding: 0; background: transparent; }
  .sig-card {
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    padding: 14px 14px 12px 14px;
    background: #ffffff;
    box-shadow: 0 10px 26px rgba(0,0,0,0.05);
  }
  .sig-card-top {
    display:flex;
    align-items:flex-start;
    justify-content:space-between;
    gap:10px;
    margin-bottom: 8px;
  }
  .sig-matchup {
    font-weight: 900;
    letter-spacing: -0.2px;
    font-size: 15px;
    line-height: 1.2;
  }
  .sig-tags {
    margin-top: 4px;
    font-size: 12px;
    opacity: 0.85;
    font-weight: 800;
  }
  .sig-pill {
    display:inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid rgba(0,0,0,0.10);
    background: rgba(0,0,0,0.03);
    font-weight: 900;
    font-size: 12px;
    margin-left: 6px;
    white-space: nowrap;
  }
  .sig-row {
    display:flex;
    gap:10px;
    flex-wrap: wrap;
    margin-top: 8px;
  }
  .sig-kv {
    flex: 1 1 160px;
    border-radius: 12px;
    padding: 10px 10px;
    border: 1px solid rgba(0,0,0,0.06);
    background: rgba(0,0,0,0.02);
  }
  .sig-k {
    font-size: 11px;
    font-weight: 900;
    opacity: 0.70;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .sig-v {
    font-size: 14px;
    font-weight: 900;
    letter-spacing: -0.2px;
  }
  .sig-subv {
    margin-top: 3px;
    font-size: 12px;
    font-weight: 800;
    opacity: 0.85;
  }
  .sig-badge {
    display:inline-block;
    padding: 3px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 900;
    border: 1px solid rgba(0,0,0,0.10);
    background: rgba(0,0,0,0.03);
    margin-right: 6px;
    margin-top: 6px;
  }
</style>
"""


def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def _render_html_component(html: str, height: int = 260):
    doc = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    {SIG_CARDS_CSS}
  </head>
  <body>
    {html}
  </body>
</html>
"""
    components.html(textwrap.dedent(doc).strip(), height=height, scrolling=False)


def render_schedule_cards(
    df: pd.DataFrame,
    spread_edge_threshold: float,
    total_edge_threshold: float,
    show_only_flagged: bool,
    require_full_data: bool,
    max_cards: int,
    unit_size_dollars: int,
):
    if df is None or len(df) == 0:
        st.info("No schedule results to display.")
        return

    df2 = df.copy()

    if "Map_Status" in df2.columns:
        df2 = df2[df2["Map_Status"].astype(str).str.upper().eq("OK")]

    if require_full_data:
        for col in ["Pred_Away", "Pred_Home"]:
            if col in df2.columns:
                df2 = df2[df2[col].notna()]
        has_spread = ("Spread_Home" in df2.columns) and df2["Spread_Home"].notna()
        has_total = ("DK_Total" in df2.columns) and df2["DK_Total"].notna()
        df2 = df2[has_spread | has_total]

    if show_only_flagged:
        sp_ok = ("Spread_Play" in df2.columns) and df2["Spread_Play"].fillna(False)
        tot_ok = ("Total_Play" in df2.columns) and df2["Total_Play"].fillna(False)
        df2 = df2[sp_ok | tot_ok]

    if len(df2) == 0:
        st.warning("No games match the current filters (mapped/full-data/flagged).")
        return

    def _row_best_abs(r):
        es = r.get("Edge_Spread", np.nan)
        et = r.get("Edge_Total", np.nan)
        a = abs(es) if pd.notna(es) else 0.0
        b = abs(et) if pd.notna(et) else 0.0
        return max(a, b)

    df2["_best_abs"] = df2.apply(_row_best_abs, axis=1)
    df2 = df2.sort_values("_best_abs", ascending=False).drop(columns=["_best_abs"])
    df2 = df2.head(int(max_cards))

    cols = st.columns(2)
    col_idx = 0

    for _, r in df2.iterrows():
        visitor = _safe_str(r.get("Visitor", ""))
        home = _safe_str(r.get("Home", ""))
        matchup = f"{visitor} @ {home}".strip()

        vtag = _safe_str(r.get("Visitor_Tag", "")).strip()
        htag = _safe_str(r.get("Home_Tag", "")).strip()
        tag_txt = ""
        if vtag:
            tag_txt += vtag
        if htag:
            tag_txt += (" / " if tag_txt else "") + htag

        pred_away = r.get("Pred_Away", np.nan)
        pred_home = r.get("Pred_Home", np.nan)
        score_txt = f"{fmt_num(pred_away,1)} - {fmt_num(pred_home,1)}"

        spread_home = r.get("Spread_Home", np.nan)
        dk_total = r.get("DK_Total", np.nan)
        vegas_spread_txt = fmt_num(spread_home, 1) if pd.notna(spread_home) else "—"
        vegas_total_txt = fmt_num(dk_total, 1) if pd.notna(dk_total) else "—"

        edge_spread = r.get("Edge_Spread", np.nan)
        edge_total = r.get("Edge_Total", np.nan)

        side_pick = pick_side_from_edge(edge_spread) if pd.notna(edge_spread) else ""
        total_pick = pick_total_from_edge(edge_total) if pd.notna(edge_total) else ""

        abs_sp = abs(edge_spread) if pd.notna(edge_spread) else 0.0
        abs_to = abs(edge_total) if pd.notna(edge_total) else 0.0
        primary_abs = max(abs_sp, abs_to)

        tier = tier_from_edge_abs(primary_abs)
        rec_units = units_from_tier(tier)
        stake_dollars = rec_units * float(unit_size_dollars)

        spread_conf = confidence_label(primary_abs)

        badges = []
        if bool(r.get("Spread_Play", False)):
            badges.append("✅ Spread Play")
        if bool(r.get("Total_Play", False)):
            badges.append("✅ Total Play")

        stake_line = ""
        if (bool(r.get("Spread_Play", False)) or bool(r.get("Total_Play", False))) and rec_units > 0:
            stake_line = f"Stake: {fmt_num(rec_units,1)}u (${int(round(stake_dollars))})"
        else:
            stake_line = "Stake: —"

        hwp = r.get("Home_WinProb", np.nan)
        awp = r.get("Away_WinProb", np.nan)
        winprob_txt = f"Win%: Away {pct_str(awp,1)} • Home {pct_str(hwp,1)}"

        html = f"""
<div class="sig-card">
  <div class="sig-card-top">
    <div>
      <div class="sig-matchup">{matchup}</div>
      {"<div class='sig-tags'>" + tag_txt + "</div>" if tag_txt else ""}
    </div>
    <div>
      <span class="sig-pill">{spread_conf if (bool(r.get("Spread_Play", False)) or bool(r.get("Total_Play", False))) else "Subscriber Card"}</span>
    </div>
  </div>

  <div class="sig-row">
    <div class="sig-kv">
      <div class="sig-k">Model Score</div>
      <div class="sig-v">{score_txt}</div>
      <div class="sig-subv">Home Margin: {fmt_num(r.get("Home_Margin", np.nan),1)} • Total: {fmt_num(r.get("Total", np.nan),1)} • {winprob_txt}</div>
    </div>

    <div class="sig-kv">
      <div class="sig-k">Spread</div>
      <div class="sig-v">Vegas: {vegas_spread_txt}</div>
      <div class="sig-subv">Edge: {fmt_num(edge_spread,2) if pd.notna(edge_spread) else "—"} • Pick: {side_pick or "—"}</div>
    </div>

    <div class="sig-kv">
      <div class="sig-k">Total</div>
      <div class="sig-v">Vegas: {vegas_total_txt}</div>
      <div class="sig-subv">Edge: {fmt_num(edge_total,2) if pd.notna(edge_total) else "—"} • Pick: {total_pick or "—"}</div>
    </div>
  </div>

  <div style="margin-top:8px;">
    {"".join([f"<span class='sig-badge'>{b}</span>" for b in badges]) if badges else "<span class='sig-badge'>Mapped • Full data</span>"}
    <span class="sig-badge">{stake_line}</span>
    <span class="sig-badge">Unit: ${int(unit_size_dollars)} • Thresholds: Spread ≥ {fmt_num(spread_edge_threshold,1)} • Total ≥ {fmt_num(total_edge_threshold,1)}</span>
  </div>
</div>
"""
        html = textwrap.dedent(html).strip()

        with cols[col_idx]:
            _render_html_component(html, height=285)

        col_idx = 1 - col_idx


def add_staking_columns(results_df: pd.DataFrame, unit_size_dollars: int) -> pd.DataFrame:
    """
    Adds:
      - Primary_Edge_Abs
      - Tier
      - Rec_Units
      - Stake_$
    Applies only to flagged plays (spread or total). Otherwise 0.
    """
    if results_df is None or len(results_df) == 0:
        return results_df

    df = results_df.copy()

    def _primary_abs(r):
        es = r.get("Edge_Spread", np.nan)
        et = r.get("Edge_Total", np.nan)
        a = abs(es) if pd.notna(es) else 0.0
        b = abs(et) if pd.notna(et) else 0.0
        return max(a, b)

    df["Primary_Edge_Abs"] = df.apply(_primary_abs, axis=1)

    def _is_flagged(r):
        return bool(r.get("Spread_Play", False)) or bool(r.get("Total_Play", False))

    df["_Flagged"] = df.apply(_is_flagged, axis=1)

    df["Tier"] = df["Primary_Edge_Abs"].apply(lambda x: tier_from_edge_abs(x))
    df["Rec_Units"] = df["Tier"].apply(units_from_tier)

    # Flagged-only: else 0
    df.loc[~df["_Flagged"], "Rec_Units"] = 0.0
    df["Stake_$"] = (df["Rec_Units"].astype(float) * float(unit_size_dollars)).round(2)

    df = df.drop(columns=["_Flagged"])
    return df


# =============================
# TEAM DASHBOARD (Option A)
# =============================
def build_team_dashboard_table(results_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Filters schedule results down to games involving `team`, and adds a few
    convenience columns for the dashboard view.
    """
    if results_df is None or len(results_df) == 0:
        return pd.DataFrame()

    df = results_df.copy()
    df["Visitor"] = df.get("Visitor", "").astype(str)
    df["Home"] = df.get("Home", "").astype(str)

    mask = (df["Visitor"] == team) | (df["Home"] == team) | (df.get("Visitor_Mapped", "") == team) | (df.get("Home_Mapped", "") == team)
    df = df[mask].copy()

    if len(df) == 0:
        return df

    # Identify which side the team is on (visitor/home)
    df["Team_Side"] = np.where(df["Home"] == team, "HOME",
                       np.where(df["Visitor"] == team, "VISITOR",
                       np.where(df.get("Home_Mapped", "") == team, "HOME",
                       np.where(df.get("Visitor_Mapped", "") == team, "VISITOR", ""))))

    # Basic pick labels for convenience
    df["Spread_Pick"] = df["Edge_Spread"].apply(lambda x: pick_side_from_edge(x) if pd.notna(x) else "")
    df["Total_Pick"] = df["Edge_Total"].apply(lambda x: pick_total_from_edge(x) if pd.notna(x) else "")

    # A clean matchup string
    df["Matchup"] = df["Visitor"].astype(str) + " @ " + df["Home"].astype(str)

    # Sort most relevant first: flagged, then biggest edge
    def _best_abs(r):
        es = r.get("Edge_Spread", np.nan)
        et = r.get("Edge_Total", np.nan)
        a = abs(es) if pd.notna(es) else 0.0
        b = abs(et) if pd.notna(et) else 0.0
        return max(a, b)

    df["_best_abs"] = df.apply(_best_abs, axis=1)
    df["_flag"] = (df.get("Spread_Play", False).fillna(False) | df.get("Total_Play", False).fillna(False)).astype(int)
    df = df.sort_values(["_flag", "_best_abs"], ascending=[False, False]).drop(columns=["_best_abs", "_flag"])

    # Columns to show
    show_cols = [
        "Matchup", "Team_Side",
        "Pred_Away", "Pred_Home", "Home_Margin", "Total",
        "Home_WinProb", "Away_WinProb",
        "Spread_Home", "Edge_Spread", "Spread_Pick", "Spread_Play",
        "DK_Total", "Edge_Total", "Total_Pick", "Total_Play",
        "Tier", "Rec_Units", "Stake_$",
        "Visitor_Tag", "Home_Tag",
        "TIME", "TV", "location",
        "Map_Status",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    return df[show_cols]


def render_team_dashboard(team: str, df_kp: pd.DataFrame, results_df: pd.DataFrame | None):
    st.markdown("## Team Dashboard")

    # Team profile from KenPom
    row = df_kp.loc[df_kp["Team"] == team]
    if row.empty:
        st.warning("Team not found in KenPom table.")
        return
    r = row.iloc[0]

    # Header
    tag = str(r.get("Style_Tag", "") or "")
    sub = []
    if tag:
        sub.append(tag)
    if bool(r.get("In_Trapezoid", False)):
        sub.append("In Trapezoid")
    if bool(r.get("Fraud", False)):
        sub.append("Fraud Flag")
    subtitle = " • ".join([s for s in sub if s])

    st.markdown(f"### {team} {('— ' + subtitle) if subtitle else ''}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ORtg", fmt_num(r.get("ORtg", np.nan), 1))
    c2.metric("DRtg", fmt_num(r.get("DRtg", np.nan), 1))
    c3.metric("NetRtg", fmt_num(r.get("NetRtg", np.nan), 1))
    c4.metric("AdjT", fmt_num(r.get("AdjT", np.nan), 1))
    c5.metric("SOS Blend", fmt_num(r.get("SOS_BLEND", np.nan), 2))

    with st.expander("SoS detail (from columns N/P/R)"):
        d1, d2, d3 = st.columns(3)
        d1.metric("SOS_NET", fmt_num(r.get("SOS_NET", np.nan), 2))
        d2.metric("SOS_OFF", fmt_num(r.get("SOS_OFF", np.nan), 2))
        d3.metric("SOS_DEF", fmt_num(r.get("SOS_DEF", np.nan), 2))

    st.markdown("---")
    st.markdown("### Today’s Games (if schedule was run)")

    if results_df is None or len(results_df) == 0:
        st.info("No schedule results loaded yet. Go to **Run full daily schedule** and click **Run all games** once. Then come back here.")
        return

    team_games = build_team_dashboard_table(results_df, team)
    if team_games.empty:
        st.warning("No games found for this team in the current schedule results.")
        return

    # Quick summary for this team
    sp_ct = int(team_games.get("Spread_Play", pd.Series([False]*len(team_games))).fillna(False).sum()) if "Spread_Play" in team_games.columns else 0
    tot_ct = int(team_games.get("Total_Play", pd.Series([False]*len(team_games))).fillna(False).sum()) if "Total_Play" in team_games.columns else 0

    s1, s2, s3 = st.columns(3)
    s1.metric("Games on board", str(len(team_games)))
    s2.metric("Flagged spreads", str(sp_ct))
    s3.metric("Flagged totals", str(tot_ct))

    st.dataframe(team_games, use_container_width=True)


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    # ✅ Password gate + logo login page (Option 1)
    subscribe_url = ""  # Optional: paste your Squarespace checkout link here
    if not require_login(logo_path=LOGO_PATH, app_name="SignalAI NCAA Predictor", subscribe_url=subscribe_url):
        return

    st.set_page_config(page_title="SignalAI NCAA Predictor", layout="wide")

    # -----------------------------
    # In-app routing (subscriber guide)
    # -----------------------------
    page = _get_query_param("page", "app").lower().strip()
    if page in {"help", "guide", "instructions"}:
        st.sidebar.markdown("### Navigation")
        st.sidebar.markdown("[← Back to app](?page=app)")
        st.sidebar.markdown("---")
        render_subscriber_guide(app_name="SignalAI NCAA Predictor")
        return

    h1, h2 = st.columns([1, 5], vertical_alignment="center")
    with h1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=110)
    with h2:
        st.title("SignalAI NCAA Predictor")

    # Optional logout
    if st.sidebar.button("Log out"):
        st.session_state["is_authed"] = False
        st.session_state["auth_error"] = ""
        st.rerun()

    # Subscriber help link
    st.sidebar.markdown("[📘 Subscriber Guide](?page=help)")

    # session cache for schedule results (so Team Dashboard can use it)
    if "schedule_results_df" not in st.session_state:
        st.session_state["schedule_results_df"] = None

    st.sidebar.header("Data source (Repo Root)")

    kp_path = find_latest_local_file("kenpom_*.xls*", REPO_ROOT)
    sched_path = find_latest_local_file("Schedule_*.xls*", REPO_ROOT)

    auto_kp_bytes = None
    auto_sched_bytes = None

    if kp_path and kp_path.exists():
        auto_kp_bytes = kp_path.read_bytes()
    if sched_path and sched_path.exists():
        auto_sched_bytes = sched_path.read_bytes()

    use_github_fallback = st.sidebar.toggle(
        "Enable GitHub fallback (if local files missing)",
        value=False,
        help="Uses GitHub API to find and download the newest Excel files from the repo root."
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
                        _, b = find_latest_github_file(
                            owner=GITHUB_OWNER,
                            repo=GITHUB_REPO,
                            folder=GITHUB_DATA_DIR,
                            pattern_regex=r"^kenpom_.*\.xls[x]?$",
                            branch=GITHUB_BRANCH,
                            token=gh_token,
                        )
                        if b:
                            auto_kp_bytes = b

                    if auto_sched_bytes is None:
                        _, b = find_latest_github_file(
                            owner=GITHUB_OWNER,
                            repo=GITHUB_REPO,
                            folder=GITHUB_DATA_DIR,
                            pattern_regex=r"^Schedule_.*\.xls[x]?$",
                            branch=GITHUB_BRANCH,
                            token=gh_token,
                        )
                        if b:
                            auto_sched_bytes = b
                except Exception as e:
                    st.sidebar.error(f"GitHub fallback failed: {e}")

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

    if kp_uploaded is not None:
        kp_bytes = kp_uploaded.getvalue()
    else:
        kp_bytes = auto_kp_bytes

    if kp_bytes is None:
        st.info(
            "No KenPom file found in repo root.\n\n"
            "Fix: commit `kenpom_*.xlsx` into the repo root (same folder as app.py), or upload manually."
        )
        return

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
        help="Applies a capped margin-only adjustment using SoS differential from KenPom columns N/P/R."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Bet Flag Thresholds")
    spread_edge_threshold = st.sidebar.slider("Spread play threshold X", 0.0, 20.0, 2.0, 0.5)
    total_edge_threshold = st.sidebar.slider("Total play threshold Y", 0.0, 30.0, 3.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Unit Size (Flagged Plays Only)")
    unit_size_dollars = st.sidebar.slider(
        "Unit size ($)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Stake amounts apply only to flagged plays (Spread_Play or Total_Play). "
             "Tiers: Lean=0.5u, Solid=1.0u, Strong=1.5u."
    )

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

    # ✅ Added Team Dashboard mode (Option A) without removing anything
    mode = st.radio(
        "Select evaluation mode:",
        ["Single matchup", "Run full daily schedule", "Team dashboard"],
        horizontal=True
    )

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
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(f"{result['Away']} (Away)", f"{result['Pred_Away']:.1f}")
        m2.metric(f"{result['Home']} (Home)", f"{result['Pred_Home']:.1f}")
        m3.metric("Total Points", f"{result['Total']:.1f}")
        m4.metric("Home Margin (Home - Away)", f"{result['Home_Margin']:.1f}")
        m5.metric("Home Win %", pct_str(result.get("Home_WinProb", np.nan), 1))
        m6.metric("Away Win %", pct_str(result.get("Away_WinProb", np.nan), 1))

        with st.expander("KenPom data preview"):
            st.dataframe(df_kp.head(25), use_container_width=True)

    elif mode == "Run full daily schedule":
        st.markdown("## Run Full Daily Schedule")

        if sched_uploaded is not None:
            sched_bytes = sched_uploaded.getvalue()
        else:
            sched_bytes = auto_sched_bytes

        if sched_bytes is None:
            st.info(
                "No schedule file found in repo root.\n\n"
                "Fix: commit `Schedule_*.xlsx` into the repo root (same folder as app.py), or upload manually."
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

        st.markdown("### Display Options")
        o1, o2, o3 = st.columns(3)
        with o1:
            show_only_flagged = st.toggle("Show only flagged plays", value=True)
        with o2:
            require_full_data = st.toggle("Require full data", value=True)
        with o3:
            max_cards = st.number_input("Max cards", min_value=4, max_value=200, value=40, step=4)

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

            # Add staking columns (flagged-only)
            results_df = add_staking_columns(results_df, unit_size_dollars=unit_size_dollars)

            # cache so Team Dashboard can use it
            st.session_state["schedule_results_df"] = results_df.copy()

            # Daily exposure summary (flagged-only)
            flagged = results_df[
                (results_df.get("Spread_Play", False) == True) | (results_df.get("Total_Play", False) == True)
            ].copy()

            if len(flagged) > 0 and "Rec_Units" in flagged.columns:
                total_units = float(flagged["Rec_Units"].fillna(0).sum())
                total_stake = float(flagged["Stake_$"].fillna(0).sum())
                strong_ct = int((flagged["Tier"].astype(str).str.lower() == "strong").sum()) if "Tier" in flagged.columns else 0
                solid_ct = int((flagged["Tier"].astype(str).str.lower() == "solid").sum()) if "Tier" in flagged.columns else 0
                lean_ct = int((flagged["Tier"].astype(str).str.lower() == "lean").sum()) if "Tier" in flagged.columns else 0

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Flagged plays", f"{len(flagged)}")
                s2.metric("Total units", f"{total_units:.1f}u")
                s3.metric("Total $ at risk", f"${total_stake:,.0f}")
                s4.metric("Tier mix", f"S:{strong_ct}  So:{solid_ct}  L:{lean_ct}")
            else:
                st.info("No flagged plays found with the current thresholds (or lines missing).")

            st.markdown("### Today’s Board (Cards)")
            render_schedule_cards(
                df=results_df,
                spread_edge_threshold=spread_edge_threshold,
                total_edge_threshold=total_edge_threshold,
                show_only_flagged=show_only_flagged,
                require_full_data=require_full_data,
                max_cards=max_cards,
                unit_size_dollars=unit_size_dollars,
            )

            st.markdown("### Full Results (mapping + edges + flags + staking)")
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv_bytes,
                file_name="schedule_predictions_full.csv",
                mime="text/csv",
            )

    else:
        # ✅ TEAM DASHBOARD (Option A)
        teams = sorted(df_kp["Team"].unique().tolist())
        default_team = teams[0] if teams else ""

        # If we already have cached schedule results, try to default to a team that appears in today's board
        cached = st.session_state.get("schedule_results_df", None)
        if isinstance(cached, pd.DataFrame) and len(cached) > 0:
            # try to pick a team that is on the schedule if possible
            todays = pd.unique(pd.concat([cached.get("Visitor", pd.Series(dtype=str)), cached.get("Home", pd.Series(dtype=str))], ignore_index=True))
            todays = [t for t in todays if isinstance(t, str) and t.strip() in set(teams)]
            if todays:
                default_team = todays[0]

        team = st.selectbox("Select a team", teams, index=(teams.index(default_team) if default_team in teams else 0))
        render_team_dashboard(team=team, df_kp=df_kp, results_df=st.session_state.get("schedule_results_df", None))


if __name__ == "__main__":
    main()


