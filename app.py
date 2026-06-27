"""
미국 개별주 옵션 스캐너 — Max Pain / Call Wall / Put Wall (월물·주물 비교)
-----------------------------------------------------------------------
실행:  streamlit run app.py
데이터: yfinance (Yahoo Finance, OI는 보통 전일 종가 기준)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
