"""
미국 개별주 옵션 스캐너 — Max Pain / Call Wall / Put Wall (월물·주물 비교)
-----------------------------------------------------------------------
실행:  streamlit run app.py
데이터: yfinance (Yahoo Finance, OI는 보통 전일 종가 기준)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="옵션 스캐너 — Max Pain / Walls", layout="wide")
R = 0.045  # 무위험금리 근사(감마용)


# ============================================================ 만기 유틸
def is_monthly(expiry: str) -> bool:
    """표준 월물 = 매월 셋째 금요일(15~21일 사이의 금요일)."""
    d = datetime.strptime(expiry, "%Y-%m-%d")
    return d.weekday() == 4 and 15 <= d.day <= 21


def exp_label(e: str) -> str:
    return f"{e}  ·  {'월물' if is_monthly(e) else '주물'}"


def years_to_exp(e: str) -> float:
    d = datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return max((d - datetime.now(timezone.utc)).days, 0) / 365.0 + 1e-4


def nearest(exps, monthly: bool):
    pool = [e for e in exps if is_monthly(e) == monthly]
    return pool[0] if pool else None  # yfinance는 오름차순 정렬


# ============================================================ 데이터
@st.cache_data(ttl=300, show_spinner=False)
def get_spot(ticker: str):
    tk = yf.Ticker(ticker)
    fi = getattr(tk, "fast_info", None)
    for k in ("last_price", "lastPrice", "regular_market_price"):
        try:
            v = fi[k] if isinstance(fi, dict) else getattr(fi, k, None)
            if v:
                return float(v)
        except Exception:
            pass
    h = tk.history(period="1d")
    return float(h["Close"].iloc[-1]) if len(h) else np.nan


@st.cache_data(ttl=300, show_spinner=False)
def get_expirations(ticker: str):
    return list(yf.Ticker(ticker).options)


@st.cache_data(ttl=300, show_spinner=False)
def get_chain(ticker: str, expiries: tuple):
    tk = yf.Ticker(ticker)
    calls, puts = [], []
    for e in expiries:
        try:
            ch = tk.option_chain(e)
            c, p = ch.calls.copy(), ch.puts.copy()
            c["expiry"], p["expiry"] = e, e
            calls.append(c)
            puts.append(p)
        except Exception:
            pass
    cols = ["strike", "openInterest", "volume", "impliedVolatility", "expiry"]
    cdf = pd.concat(calls)[cols] if calls else pd.DataFrame(columns=cols)
    pdf = pd.concat(puts)[cols] if puts else pd.DataFrame(columns=cols)
    for df in (cdf, pdf):
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
        df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
        df["T"] = df["expiry"].map(years_to_exp)
    return cdf, pdf


@st.cache_data(ttl=900, show_spinner=False)
def get_ohlc(ticker: str, period: str = "4mo"):
    return yf.Ticker(ticker).history(period=period)


def atr_wilder(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def money(x, cur):
    if x != x:
        return "—"
    sym = "₩" if cur == "KRW" else "$"
    dec = 0 if cur == "KRW" else 2
    return f"{sym}{x:,.{dec}f}"


# ============================================================ 분석
def compute_max_pain(calls, puts):
    c = calls.groupby("strike")["openInterest"].sum()
    p = puts.groupby("strike")["openInterest"].sum()
    strikes = np.array(sorted(set(c.index).union(p.index)), dtype=float)
    if strikes.size == 0:
        return np.nan, pd.DataFrame(columns=["price", "total_value"])
    coi = c.reindex(strikes, fill_value=0).values.astype(float)
    poi = p.reindex(strikes, fill_value=0).values.astype(float)
    P, K = strikes.reshape(-1, 1), strikes.reshape(1, -1)
    total = ((np.maximum(P - K, 0) * coi) + (np.maximum(K - P, 0) * poi)).sum(axis=1) * 100.0
    df = pd.DataFrame({"price": strikes, "total_value": total})
    return float(strikes[int(np.argmin(total))]), df


def oi_walls(calls, puts):
    c = calls.groupby("strike")["openInterest"].sum()
    p = puts.groupby("strike")["openInterest"].sum()
    cw = float(c.idxmax()) if len(c) and c.max() > 0 else np.nan
    pw = float(p.idxmax()) if len(p) and p.max() > 0 else np.nan
    return cw, pw


def by_strike(calls, puts):
    c = calls.groupby("strike")["openInterest"].sum()
    p = puts.groupby("strike")["openInterest"].sum()
    ks = sorted(set(c.index).union(p.index))
    return pd.DataFrame({"strike": ks,
                         "call": [c.get(k, 0) for k in ks],
                         "put": [p.get(k, 0) for k in ks]})


def bs_gamma(S, K, T, r, sigma):
    sigma = np.where(np.asarray(sigma, float) <= 0, np.nan, sigma)
    T = np.maximum(T, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    pdf = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi)
    return pdf / (S * sigma * np.sqrt(T))


def gamma_by_strike(side, spot):
    g = bs_gamma(spot, side["strike"].values, side["T"].values,
                 R, side["impliedVolatility"].values)
    notfor = np.nan_to_num(g) * side["openInterest"].values * 100 * spot * spot * 0.01
    return pd.Series(notfor, index=side["strike"].values).groupby(level=0).sum()


@st.cache_data(ttl=300, show_spinner=False)
def scan_one(ticker: str):
    ticker = ticker.upper().strip()
    out = {"티커": ticker}
    try:
        exps = get_expirations(ticker)
        if not exps:
            out["비고"] = "옵션 없음"
            return out
        spot = get_spot(ticker)
        out["현재가"] = spot
        for tag, monthly in [("월", True), ("주", False)]:
            e = nearest(exps, monthly)
            if e is None:
                continue
            calls, puts = get_chain(ticker, (e,))
            mp, _ = compute_max_pain(calls, puts)
            cw, pw = oi_walls(calls, puts)
            out[f"{tag}_만기"] = e
            out[f"{tag}_MaxPain"] = mp
            out[f"{tag}_콜월"] = cw
            out[f"{tag}_풋월"] = pw
    except Exception as ex:
        out["비고"] = f"오류: {ex}"
    return out


# ============================================================ 넷 유동성 (FRED)
FRED_IDS = ["WALCL", "WTREGEN", "RRPONTSYD", "SOFR", "IORB",
            "DGS10", "DGS2", "DTWEXBGS", "VIXCLS"]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fred_series(series_id, api_key, start="2014-01-01"):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key,
              "file_type": "json", "observation_start": start}
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    data = {o["date"]: (float(o["value"]) if o["value"] not in (".", "") else np.nan)
            for o in obs}
    s = pd.Series(data)
    s.index = pd.to_datetime(s.index)
    return s.dropna()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_liquidity(api_key):
    raw = {sid: fred_series(sid, api_key) for sid in FRED_IDS}
    start = min((s.index.min() for s in raw.values() if len(s)), default=pd.Timestamp("2014-01-01"))
    idx = pd.date_range(start, pd.Timestamp.today().normalize(), freq="D")
    d = {sid: raw[sid].reindex(idx).ffill() for sid in raw}
    # 단위 정렬: WALCL·WTREGEN=백만$, RRPONTSYD=십억$ → 백만$로
    d["NETLIQ"] = d["WALCL"] - d["WTREGEN"] - d["RRPONTSYD"] * 1000.0
    return d


def zscore_last(series, window=756):
    s = series.dropna()
    if len(s) < 30:
        return np.nan
    w = s.tail(window)
    sd = w.std()
    return float((s.iloc[-1] - w.mean()) / sd) if sd and sd == sd else np.nan


def liquidity_table(d):
    chg = lambda s, n=91: s - s.shift(n)  # 약 13주 변화
    # (이름, 최신값, 단위, z, 가중치, 부호[+ = 값↑이 유동성↑])
    spec = [
        ("넷유동성 (레벨)", d["NETLIQ"].iloc[-1] / 1e6, "T$", zscore_last(d["NETLIQ"]), 1.0, +1),
        ("넷유동성 (13주 변화)", chg(d["NETLIQ"]).iloc[-1] / 1e6, "T$", zscore_last(chg(d["NETLIQ"])), 2.0, +1),
        ("SOFR−IORB 스프레드", (d["SOFR"] - d["IORB"]).iloc[-1], "%p", zscore_last(d["SOFR"] - d["IORB"]), 1.5, -1),
        ("VIX", d["VIXCLS"].iloc[-1], "", zscore_last(d["VIXCLS"]), 1.0, -1),
        ("브로드 달러 (13주 변화)", chg(d["DTWEXBGS"]).iloc[-1], "idx", zscore_last(chg(d["DTWEXBGS"])), 1.0, -1),
    ]
    rows = []
    for name, val, unit, z, w, sign in spec:
        sz = (z * sign) if z == z else np.nan
        rows.append({"지표": name, "_val": val, "단위": unit, "z": z,
                     "유동성기여(z)": sz, "가중치": w})
    df = pd.DataFrame(rows)
    v = df.dropna(subset=["유동성기여(z)"])
    comp = float((v["유동성기여(z)"] * v["가중치"]).sum() / v["가중치"].sum()) if len(v) else np.nan
    return df, comp


def classify_liquidity(comp, t_rich=0.5, t_norm=-0.5, t_low=-1.25):
    if comp != comp:
        return ("판정 불가", "#666666", "⚪")
    if comp >= t_rich:
        return ("풍부함", "#2e7d32", "🟢")
    if comp >= t_norm:
        return ("보통", "#f9a825", "🟡")
    if comp >= t_low:
        return ("적음", "#ef6c00", "🟠")
    return ("위험", "#c62828", "🔴")


# ============================================================ 차트
def oi_bar(view, spot, mp, cw, pw, title):
    fig = go.Figure()
    fig.add_bar(x=view.strike, y=view.call, name="Call OI", marker_color="#2e86de")
    fig.add_bar(x=view.strike, y=-view.put, name="Put OI", marker_color="#e74c3c")
    fig.update_layout(barmode="relative", height=430, title=title,
                      xaxis_title="행사가", yaxis_title="OI (위:콜 / 아래:풋)",
                      legend=dict(orientation="h"), margin=dict(t=40, b=10))
    ymax = max(view.call.max(), view.put.max()) if len(view) else 1
    for x, nm, col in [(spot, "현재가", "#888"), (mp, "MaxPain", "#e6a817"),
                       (cw, "CallWall", "#2e86de"), (pw, "PutWall", "#e74c3c")]:
        if x == x and view.strike.min() <= x <= view.strike.max():
            fig.add_vline(x=x, line_dash="dash", line_color=col, line_width=1.4)
            fig.add_annotation(x=x, y=ymax, text=nm, showarrow=False, yshift=8,
                               font=dict(color=col, size=10))
    return fig


def detail_panel(col, ticker, spot, expiry):
    calls, puts = get_chain(ticker, (expiry,))
    if calls.empty and puts.empty:
        col.warning("체인이 비어 있습니다.")
        return
    mp, _ = compute_max_pain(calls, puts)
    cw, pw = oi_walls(calls, puts)
    col.markdown(f"**{exp_label(expiry)}**")
    a, b, c, d = col.columns(4)
    a.metric("MaxPain", f"{mp:,.1f}" if mp == mp else "—")
    b.metric("콜월", f"{cw:,.1f}" if cw == cw else "—")
    c.metric("풋월", f"{pw:,.1f}" if pw == pw else "—")
    d.metric("현재가", f"{spot:,.1f}" if spot == spot else "—")
    oi = by_strike(calls, puts)
    if spot == spot:
        oi = oi[(oi.strike >= spot * 0.6) & (oi.strike <= spot * 1.4)]
    col.plotly_chart(oi_bar(oi, spot, mp, cw, pw, ""), use_container_width=True)


# ============================================================ UI
st.title("미국 개별주 옵션 스캐너 — Max Pain · Call Wall · Put Wall")
st.caption("데이터: Yahoo Finance(약 15분 지연, OI는 보통 전일 종가 기준). 보조 지표일 뿐 매매 신호가 아닙니다.")

raw = st.text_area("티커 (쉼표 또는 줄바꿈으로 여러 개)",
                   value="MU, NVDA, AAPL, TSLA", height=80)
tickers = [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]
tickers = list(dict.fromkeys(tickers))  # 중복 제거, 순서 유지

tab_scan, tab_detail = st.tabs(["📊 스캐너", "🔍 종목 상세 (월물 / 주물)"])

# ---------------------------------------------------------- 스캐너
with tab_scan:
    if not tickers:
        st.info("티커를 입력하세요.")
    else:
        prog = st.progress(0.0, text="스캔 중...")
        rows = []
        for i, t in enumerate(tickers, 1):
            rows.append(scan_one(t))
            prog.progress(i / len(tickers), text=f"스캔 중... {t}")
        prog.empty()
        df = pd.DataFrame(rows)

        def dist(level, spot):
            if level != level or spot != spot or spot == 0:
                return ""
            return f"{(level/spot-1)*100:+.1f}%"

        show = pd.DataFrame({"티커": df.get("티커")})
        show["현재가"] = df.get("현재가")
        for tag in ("월", "주"):
            show[f"{tag}·만기"] = df.get(f"{tag}_만기")
            for nm in ("MaxPain", "콜월", "풋월"):
                col = f"{tag}_{nm}"
                show[f"{tag}·{nm}"] = [
                    (f"{lv:,.1f} ({dist(lv, sp)})" if lv == lv else "—")
                    for lv, sp in zip(df.get(col, [np.nan] * len(df)),
                                      df.get("현재가", [np.nan] * len(df)))
                ]
        if "비고" in df:
            show["비고"] = df["비고"]
        show["현재가"] = show["현재가"].map(lambda v: f"{v:,.2f}" if v == v else "—")

        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("괄호 값 = 현재가 대비 거리. 콜월=저항·풋월=지지로 흔히 해석. "
                   "월물은 OI가 집중돼 벽이 뚜렷하고, 주물은 단기 플로우 성격.")

# ---------------------------------------------------------- 상세
with tab_detail:
    if not tickers:
        st.info("위에서 티커를 입력하세요.")
    else:
        sel_t = st.selectbox("종목 선택", tickers)
        try:
            exps = get_expirations(sel_t)
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            exps = []
        if not exps:
            st.error("옵션 만기를 찾지 못했습니다.")
        else:
            spot = get_spot(sel_t)
            months = [e for e in exps if is_monthly(e)]
            weeks = [e for e in exps if not is_monthly(e)]
            cL, cR = st.columns(2)
            with cL:
                st.subheader("월물")
                if months:
                    me = st.selectbox("월물 만기", months, key="m",
                                      format_func=lambda x: x)
                    detail_panel(cL, sel_t, spot, me)
                else:
                    st.info("월물 없음")
            with cR:
                st.subheader("주물")
                if weeks:
                    we = st.selectbox("주물 만기", weeks, key="w",
                                      format_func=lambda x: x)
                    detail_panel(cR, sel_t, spot, we)
                else:
                    st.info("주물 없음")

            with st.expander("감마 프로파일(GEX) — 선택한 월물 기준"):
                if months:
                    calls, puts = get_chain(sel_t, (st.session_state.get("m", months[0]),))
                    dealer = st.radio("딜러 포지션 가정",
                                      ["롱콜·숏풋 (SpotGamma식)", "단순 합산"],
                                      horizontal=True, key="dealer")
                    sign = -1.0 if dealer.startswith("롱콜") else 1.0
                    cg, pg = gamma_by_strike(calls, spot), gamma_by_strike(puts, spot)
                    ks = sorted(set(cg.index).union(pg.index))
                    net = pd.DataFrame({"strike": ks})
                    net["net"] = [cg.get(k, 0) + sign * pg.get(k, 0) for k in ks]
                    if spot == spot:
                        net = net[(net.strike >= spot * 0.6) & (net.strike <= spot * 1.4)]
                    fig = go.Figure()
                    fig.add_bar(x=net.strike, y=net.net,
                                marker_color=np.where(net.net >= 0, "#2e86de", "#e74c3c"))
                    fig.add_vline(x=spot, line_dash="dash", line_color="#888")
                    fig.update_layout(height=380, xaxis_title="행사가",
                                      yaxis_title="Net GEX ($/1%)", showlegend=False,
                                      margin=dict(t=20))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("양(+)=안정화, 음(−)=변동성 증폭 구간. 딜러 가정에 따라 부호가 바뀜.")

# ---------------------------------------------------------- 넷 유동성
st.divider()
st.header("💧 넷 유동성 (Net Liquidity = Fed 총자산 − TGA − RRP)")

try:
    fred_key = st.secrets["FRED_API_KEY"]
except Exception:
    fred_key = ""

if not fred_key:
    st.warning(
        "FRED API 키가 필요합니다 (무료, 1분).\n\n"
        "1. https://fredaccount.stlouisfed.org/apikeys 에서 키 발급\n"
        "2. **Streamlit Cloud**: 앱 → Settings → Secrets 에 아래 한 줄 추가\n"
        "   ```\n   FRED_API_KEY = \"발급받은키\"\n   ```\n"
        "3. **로컬 실행**: 프로젝트에 `.streamlit/secrets.toml` 파일을 만들고 같은 줄 입력"
    )
else:
    try:
        with st.spinner("FRED 유동성 데이터 로딩..."):
            d = load_liquidity(fred_key)
        df_liq, comp = liquidity_table(d)

        with st.expander("판정 기준 조정 (z 컷오프)"):
            t_rich = st.slider("풍부함 ≥", 0.0, 1.5, 0.5, 0.05)
            t_norm = st.slider("보통 ≥", -1.0, 0.5, -0.5, 0.05)
            t_low = st.slider("적음 ≥ (미만은 위험)", -2.0, 0.0, -1.25, 0.05)

        label, color, emoji = classify_liquidity(comp, t_rich, t_norm, t_low)
        st.markdown(
            f"<div style='padding:16px 20px;border-radius:12px;background:{color};color:#fff;"
            f"display:flex;align-items:baseline;gap:16px;'>"
            f"<span style='font-size:30px;font-weight:800'>{emoji} {label}</span>"
            f"<span style='font-size:17px;opacity:.95'>종합점수 {comp:+.2f}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption("종합점수 = 각 지표를 추세 대비 z-score로 정규화하고 유동성 방향으로 부호 정렬한 뒤 가중 평균한 값.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("넷유동성", f"${d['NETLIQ'].iloc[-1] / 1e6:,.2f}T",
                  f"{(d['NETLIQ'].iloc[-1] - d['NETLIQ'].shift(91).iloc[-1]) / 1e6:+,.2f}T (13주)")
        m2.metric("Fed 총자산", f"${d['WALCL'].iloc[-1] / 1e6:,.2f}T")
        m3.metric("TGA", f"${d['WTREGEN'].iloc[-1] / 1e6:,.2f}T")
        m4.metric("RRP", f"${d['RRPONTSYD'].iloc[-1] / 1e3:,.2f}T")

        show = df_liq.copy()
        show["최신값"] = show.apply(lambda r: f"{r['_val']:,.2f} {r['단위']}", axis=1)
        show["z"] = show["z"].map(lambda v: f"{v:+.2f}" if v == v else "—")
        show["유동성기여(z)"] = show["유동성기여(z)"].map(lambda v: f"{v:+.2f}" if v == v else "—")
        st.dataframe(show[["지표", "최신값", "z", "유동성기여(z)", "가중치"]],
                     hide_index=True, use_container_width=True)

        nl = (d["NETLIQ"] / 1e6).tail(520)
        fig = go.Figure(go.Scatter(x=nl.index, y=nl.values, line=dict(color="#2e86de")))
        fig.update_layout(height=300, yaxis_title="넷유동성 ($T)",
                          margin=dict(t=20, b=10), title="넷유동성 추세 (최근 ~1.4년)")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "※ 넷유동성 구성요소(WALCL·TGA)는 **주간(H.4.1, 목요일 갱신)** 데이터라 일중 변화는 보조지표(RRP·SOFR·VIX·달러) 위주로 움직입니다. "
            "가중치·z 윈도우·컷오프는 모두 임의 설정값이며, 백테스트로 검증된 매매 신호가 아니라 **상태 요약용 대시보드**입니다."
        )
    except Exception as e:
        st.error(f"FRED 데이터 로드 실패: {e}")

# ---------------------------------------------------------- 진입 타이밍
st.divider()
st.header("🎯 진입 타이밍 — 지금 들어가도 되나? (EMA 기반)")

ec1, ec2, ec3 = st.columns([1.6, 1, 1])
et_ticker = ec1.text_input("티커 (미국: MU / 한국: 005930.KS, 000660.KS)",
                           value="MU", key="et_tkr").strip().upper()
et_cur = ec2.selectbox("통화", ["USD", "KRW"], key="et_cur")
horizon = ec3.selectbox("기간", ["스윙 (EMA 20·50·100)", "장기 (EMA 50·100·200)"], key="et_hz")

EMA_SET = [20, 50, 100] if horizon.startswith("스윙") else [50, 100, 200]
fast, mid, slow = EMA_SET

odf = None
if et_ticker:
    try:
        odf = get_ohlc(et_ticker, period="2y")
    except Exception:
        odf = None

if odf is None or odf.empty:
    st.warning("가격 데이터를 불러오지 못했습니다. 티커를 확인하세요. (한국주는 005930.KS 형식)")
else:
    close = odf["Close"]
    price = float(close.iloc[-1])
    atr = float(atr_wilder(odf).iloc[-1])
    emas = {n: float(close.ewm(span=n, adjust=False).mean().iloc[-1]) for n in EMA_SET}
    ef, em, es = emas[fast], emas[mid], emas[slow]
    ext_atr = (price - ef) / atr if atr > 0 else 0.0  # 빠른 EMA 대비 과열도(ATR 단위)

    # ---- 진입 적합도 판정
    if price < es:
        verdict, vc, ve = "진입 부적합", "#c62828", "🔴"
        reason = f"장기 추세선(EMA{slow}) 아래 — 신규 진입 부적합. 추세 전환 확인 후 검토하세요."
        include_now = False
    elif ext_atr > 2:
        verdict, vc, ve = "과열 — 대기", "#ef6c00", "🟠"
        reason = f"상승추세지만 EMA{fast} 대비 단기 과열(+{ext_atr:.1f} ATR). 눌림을 기다려 분할 진입하세요."
        include_now = False
    elif price < em:
        verdict, vc, ve = "분할 매수 구간", "#f9a825", "🟡"
        reason = f"장기추세 위·중기(EMA{mid}) 아래의 눌림 구간. 지금부터 분할 매수 적합."
        include_now = True
    else:
        verdict, vc, ve = "진입 양호", "#2e7d32", "🟢"
        reason = "상승추세 + 과열 아님 — 분할 진입 양호."
        include_now = True

    st.markdown(
        f"<div style='padding:16px 20px;border-radius:12px;background:{vc};color:#fff;'>"
        f"<span style='font-size:28px;font-weight:800'>{ve} {verdict}</span><br>"
        f"<span style='font-size:15px;opacity:.95'>{reason}</span></div>",
        unsafe_allow_html=True,
    )

    mcol1, mcol2 = st.columns(2)
    mcol1.metric("현재가", money(price, et_cur))
    mcol2.metric("ATR(14, 일봉)", money(atr, et_cur), f"{atr / price * 100:.1f}%")

    # ---- EMA 위치표
    rows = []
    for n in EMA_SET:
        v = emas[n]
        rows.append({"EMA": f"EMA{n}", "값": money(v, et_cur),
                     "현재가 대비": f"{(price / v - 1) * 100:+.1f}%",
                     "위치": "지지 (아래)" if v < price else "저항 (위)"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ---- 분할 진입 사다리 (현재가 아래 EMA = 눌림 매수 목표)
    below = sorted([n for n in EMA_SET if emas[n] < price], key=lambda n: emas[n], reverse=True)
    levels = []
    if include_now:
        levels.append(("지금 (현재가)", price, 1.0))
    w = 1.3
    for n in below:
        levels.append((f"EMA{n} 도달 시", emas[n], w))
        w += 0.6  # 더 깊은 눌림일수록 비중 ↑

    st.subheader("분할 진입 사다리")
    if not levels:
        st.info("현재가 아래에 지지 EMA가 없습니다 — 지지선이 없으니 관망을 권합니다.")
    else:
        tw = sum(l[2] for l in levels)
        lad = [{"도달 조건": nm, "가격": money(px, et_cur),
                "진입 비중": f"{wt / tw * 100:.0f}%"} for nm, px, wt in levels]
        st.dataframe(pd.DataFrame(lad), hide_index=True, use_container_width=True)
        st.caption("비중 = 계획한 전체 포지션 대비 %. 아래 EMA로 내려갈수록 더 담는 분할 매수 구조입니다.")
        if verdict.startswith("진입 부적합"):
            st.warning("추세가 하락이라 하단 EMA로 물타기는 위험할 수 있습니다. 참고용으로만 보세요.")

    st.caption("추세추종 진입 가이드일 뿐 수익을 보장하지 않습니다. ATR·EMA는 일봉 기준이며 변동성 급변 시 신호가 흔들릴 수 있습니다.")

with st.expander("계산 방식 / 데이터 한계"):
    st.markdown(
        """
- **Max Pain**: 각 가정 정산가에서 콜·풋 보유자 총 내재가치(=매도자 지급액)가 최소가 되는 행사가.
- **콜/풋 월(OI)**: 콜·풋 미결제약정이 가장 큰 행사가 → 저항/지지로 해석.
- **월물 = 셋째 금요일**, 그 외는 주물. 월물에 OI가 집중돼 벽이 뚜렷합니다.
- **한계**: Yahoo OI는 보통 전일 종가 기준 1일 지연, IV·체결량 누락 가능. 비공식 스크래핑이라 간헐적 실패 가능(잠시 후 재시도).
- 정밀 실시간 GEX는 유료 데이터(Polygon·ORATS·CBOE) 권장. 보조 지표일 뿐 매매 신호 아님.
        """
    )
