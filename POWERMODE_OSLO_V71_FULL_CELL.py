# ============================================================

# 🔥 POWERMODE OSLO v7.1 — DUAL ENGINE FULL ALPHA REGIME

# One-cell version

# Stores: V7_LIVE for MASTER

# ============================================================

import yfinance as yf

import pandas as pd

import numpy as np

import warnings

from datetime import date

warnings.filterwarnings("ignore")

# ============================================================

# SETTINGS

# ============================================================

LOOKBACK_DAYS = 120

TRADE_HOLD_DAYS = 10

STRUCTURE_HOLD_DAYS = 40

MIN_GAP_DAYS = 5

TODAY = date.today().isoformat()

# ============================================================

# OSLO / OSE UNIVERSE

# ============================================================

tickers = [

    "EQNR.OL","DNB.OL","KOG.OL","NHY.OL","YAR.OL","STB.OL",

    "AKRBP.OL","VAR.OL","DNO.OL","PEN.OL","SUBC.OL","TGS.OL",

    "BWO.OL","OET.OL","AKSO.OL","BWE.OL","DOFG.OL","NORAM.OL","BNOR.OL",

    "BWLPG.OL","HAFNI.OL","FRO.OL","MPCC.OL","KCC.OL",

    "HAUTO.OL","HSHP.OL","WAWI.OL","WWI.OL","WWIB.OL",

    "CADLR.OL","SOAG.OL","SMOP.OL",

    "SBNOR.OL","MING.OL","GJF.OL","PROT.OL","PARB.OL",

    "SPOL.OL","SKUE.OL","BONHR.OL",

    "KIT.OL","NOD.OL","KID.OL","TOM.OL","ELK.OL","AFG.OL",

    "VEI.OL","ATEA.OL","NORBT.OL","MULTI.OL",

    "MOWI.OL","SALM.OL","LSG.OL","AUSS.OL","AKH.OL",

    "ABL.OL","ENVIP.OL","ARCH.OL","AUTO.OL","ODL.OL",

    "SATS.OL","NAPA.OL","LINK.OL","DFENS.OL",

    "SCATC.OL","HEX.OL","MPCES.OL","TEL.OL"

]

tickers = sorted(list(set(tickers)))

bad_tickers = ["GOGL.OL","CRAYN.OL","XXL.OL","CMBTO.OL","HSHIP.OL"]

tickers = [t for t in tickers if t not in bad_tickers]

BENCH = "OSEBX.OL"

# ============================================================

# INDICATORS

# ============================================================

def ema(x, n):

    return x.ewm(span=n, adjust=False).mean()

def dema(x, n):

    e1 = ema(x, n)

    e2 = ema(e1, n)

    return 2 * e1 - e2

def kama(series, er_period=10, fast=2, slow=30):

    change = abs(series - series.shift(er_period))

    vol = series.diff().abs().rolling(er_period).sum()

    er = change / (vol + 1e-9)

    fast_sc = 2 / (fast + 1)

    slow_sc = 2 / (slow + 1)

    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    out = pd.Series(index=series.index, dtype=float)

    out.iloc[0] = series.iloc[0]

    for i in range(1, len(series)):

        if np.isnan(sc.iloc[i]):

            out.iloc[i] = out.iloc[i - 1]

        else:

            out.iloc[i] = (

                out.iloc[i - 1]

                + sc.iloc[i] * (series.iloc[i] - out.iloc[i - 1])

            )

    return out

def rsi(x, n=14):

    d = x.diff()

    up = d.clip(lower=0)

    down = -d.clip(upper=0)

    rs = (

        up.ewm(alpha=1/n, adjust=False).mean()

        /

        (down.ewm(alpha=1/n, adjust=False).mean() + 1e-9)

    )

    return 100 - (100 / (1 + rs))

def macd_hist(x):

    macd = ema(x, 12) - ema(x, 26)

    sig = ema(macd, 9)

    return macd - sig

def dmi(df, n=10):

    high = df["High"]

    low = df["Low"]

    close = df["Close"]

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = np.where(

        (up_move > down_move) & (up_move > 0),

        up_move,

        0.0

    )

    minus_dm = np.where(

        (down_move > up_move) & (down_move > 0),

        down_move,

        0.0

    )

    tr1 = high - low

    tr2 = abs(high - close.shift())

    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/n, adjust=False).mean()

    plus_di = (

        100

        * pd.Series(plus_dm, index=df.index)

        .ewm(alpha=1/n, adjust=False)

        .mean()

        / (atr + 1e-9)

    )

    minus_di = (

        100

        * pd.Series(minus_dm, index=df.index)

        .ewm(alpha=1/n, adjust=False)

        .mean()

        / (atr + 1e-9)

    )

    return plus_di, minus_di

# ============================================================

# CORE FEATURE ENGINE

# ============================================================

def add_core(df):

    df = df.copy()

    close = df["Close"]

    df["DEMA50"] = dema(close, 50)

    df["DEMA200"] = dema(close, 200)

    df["KAMA"] = kama(close)

    df["RSI"] = rsi(close)

    df["MACD_HIST"] = macd_hist(close)

    df["MACD_SLOPE"] = df["MACD_HIST"] - df["MACD_HIST"].shift(1)

    pdi, mdi = dmi(df)

    df["DI_PLUS"] = pdi

    df["DI_MINUS"] = mdi

    df["DMI_SPREAD"] = df["DI_PLUS"] - df["DI_MINUS"]

    df["DMI_ACCEL"] = df["DMI_SPREAD"] - df["DMI_SPREAD"].shift(1)

    df["RS20"] = ((close / close.shift(20)) - 1) * 100

    df["RS60"] = ((close / close.shift(60)) - 1) * 100

    df["Persistence"] = (

        df["Close"] > df["KAMA"]

    ).rolling(20).sum()

    return df.replace([np.inf, -np.inf], np.nan).dropna()

# ============================================================

# ENGINE 1 — 10D TRADE ENGINE

# ============================================================

def trade_signal(core):

    row = core.iloc[-1]

    prev = core.iloc[-2]

    rsi_now = row["RSI"]

    macd_improving = row["MACD_HIST"] > prev["MACD_HIST"]

    dmi_positive = row["DI_PLUS"] > row["DI_MINUS"]

    rs20 = row["RS20"]

    if (

        52 <= rsi_now <= 68

        and macd_improving

        and dmi_positive

        and -5 <= rs20 <= 15

    ):

        return "10D TRADE"

    return "NO SIGNAL"

# ============================================================

# ENGINE 2 — 40D STRUCTURE ENGINE

# ============================================================

def structure_signal(core):

    row = core.iloc[-1]

    prev = core.iloc[-2]

    persistence = row["Persistence"]

    rsi_now = row["RSI"]

    macd_improving = row["MACD_HIST"] > prev["MACD_HIST"]

    dmi_positive = row["DI_PLUS"] > row["DI_MINUS"]

    rs20 = row["RS20"]

    if (

        persistence >= 14

        and 50 <= rsi_now <= 68

        and macd_improving

        and dmi_positive

        and 0 <= rs20 <= 20

    ):

        return "40D STRUCTURE"

    return "NO SIGNAL"

# ============================================================

# DOWNLOAD DATA

# ============================================================

print("Downloading data...")

raw = yf.download(

    tickers + [BENCH],

    period="3y",

    interval="1d",

    group_by="ticker",

    auto_adjust=True,

    progress=False,

    threads=False

)

data = {}

for t in tickers + [BENCH]:

    try:

        df = raw[t].dropna()

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)

        df = (

            df[["Open","High","Low","Close","Volume"]]

            .replace([np.inf, -np.inf], np.nan)

            .dropna()

        )

        if len(df) > 250:

            data[t] = df

    except Exception:

        pass

print("Loaded:", len(data) - (1 if BENCH in data else 0))

missing = [t for t in tickers if t not in data]

print("\nMissing / not loaded:")

print(missing)

if BENCH not in data:

    raise ValueError(

        "Benchmark OSEBX.OL not loaded. Try BENCH='^OSEBX' or check Yahoo ticker."

    )

bench = data[BENCH].copy()

# ============================================================

# EDGE TRACKER

# ============================================================

dates = bench.index[

    -(LOOKBACK_DAYS + STRUCTURE_HOLD_DAYS):

    -STRUCTURE_HOLD_DAYS

]

trade_rows = []

structure_rows = []

for d in dates:

    for t in tickers:

        try:

            df = data.get(t)

            if df is None:

                continue

            hist = df[df.index <= d]

            future = df[df.index > d]

            if len(hist) < 250 or len(future) < STRUCTURE_HOLD_DAYS:

                continue

            core = add_core(hist.tail(250))

            if len(core) < 5:

                continue

            t_signal = trade_signal(core)

            if t_signal == "10D TRADE":

                entry = hist["Close"].iloc[-1]

                exit_trade = future["Close"].iloc[TRADE_HOLD_DAYS - 1]

                ret_trade = (exit_trade / entry - 1) * 100

                trade_rows.append([

                    d,

                    t,

                    round(core.iloc[-1]["RSI"], 1),

                    round(core.iloc[-1]["RS20"], 1),

                    round(core.iloc[-1]["RS60"], 1),

                    int(core.iloc[-1]["Persistence"]),

                    round(core.iloc[-1]["MACD_HIST"], 3),

                    round(core.iloc[-1]["MACD_SLOPE"], 3),

                    round(core.iloc[-1]["DI_PLUS"], 1),

                    round(core.iloc[-1]["DI_MINUS"], 1),

                    round(core.iloc[-1]["DMI_ACCEL"], 2),

                    round(ret_trade, 2)

                ])

            s_signal = structure_signal(core)

            if s_signal == "40D STRUCTURE":

                entry = hist["Close"].iloc[-1]

                exit_structure = future["Close"].iloc[

                    STRUCTURE_HOLD_DAYS - 1

                ]

                ret_structure = (exit_structure / entry - 1) * 100

                structure_rows.append([

                    d,

                    t,

                    round(core.iloc[-1]["RSI"], 1),

                    round(core.iloc[-1]["RS20"], 1),

                    round(core.iloc[-1]["RS60"], 1),

                    int(core.iloc[-1]["Persistence"]),

                    round(core.iloc[-1]["MACD_HIST"], 3),

                    round(core.iloc[-1]["MACD_SLOPE"], 3),

                    round(core.iloc[-1]["DI_PLUS"], 1),

                    round(core.iloc[-1]["DI_MINUS"], 1),

                    round(core.iloc[-1]["DMI_ACCEL"], 2),

                    round(ret_structure, 2)

                ])

        except Exception:

            pass

# ============================================================

# DATAFRAMES

# ============================================================

trade_bt = pd.DataFrame(

    trade_rows,

    columns=[

        "Date","Ticker","RSI","RS20","RS60","Persistence",

        "MACD_HIST","MACD_SLOPE","DI_PLUS","DI_MINUS",

        "DMI_ACCEL","Return_10D"

    ]

)

structure_bt = pd.DataFrame(

    structure_rows,

    columns=[

        "Date","Ticker","RSI","RS20","RS60","Persistence",

        "MACD_HIST","MACD_SLOPE","DI_PLUS","DI_MINUS",

        "DMI_ACCEL","Return_40D"

    ]

)

# ============================================================

# SIGNAL GAP FILTER

# ============================================================

def apply_signal_gap(signal_df, date_col="Date", ticker_col="Ticker", min_gap=5):

    if signal_df.empty:

        return signal_df

    signal_df = signal_df.sort_values([ticker_col, date_col]).copy()

    keep_rows = []

    last_signal_date = {}

    for idx, row in signal_df.iterrows():

        ticker = row[ticker_col]

        d = pd.to_datetime(row[date_col])

        if ticker not in last_signal_date:

            keep_rows.append(idx)

            last_signal_date[ticker] = d

        else:

            if (d - last_signal_date[ticker]).days >= min_gap:

                keep_rows.append(idx)

                last_signal_date[ticker] = d

    return signal_df.loc[keep_rows].copy()

trade_bt = apply_signal_gap(

    trade_bt,

    min_gap=MIN_GAP_DAYS

)

structure_bt = apply_signal_gap(

    structure_bt,

    min_gap=MIN_GAP_DAYS

)

# ============================================================

# BENCHMARK ALPHA

# ============================================================

bench["Future10"] = bench["Close"].shift(-TRADE_HOLD_DAYS)

bench["BenchReturn10D"] = (

    bench["Future10"] / bench["Close"] - 1

) * 100

bench["Future40"] = bench["Close"].shift(-STRUCTURE_HOLD_DAYS)

bench["BenchReturn40D"] = (

    bench["Future40"] / bench["Close"] - 1

) * 100

bench10 = bench["BenchReturn10D"]

bench40 = bench["BenchReturn40D"]

if not trade_bt.empty:

    trade_bt["Bench10D"] = trade_bt["Date"].map(bench10)

    trade_bt["Alpha10D"] = (

        trade_bt["Return_10D"] - trade_bt["Bench10D"]

    )

    trade_bt = trade_bt.dropna()

if not structure_bt.empty:

    structure_bt["Bench40D"] = structure_bt["Date"].map(bench40)

    structure_bt["Alpha40D"] = (

        structure_bt["Return_40D"] - structure_bt["Bench40D"]

    )

    structure_bt = structure_bt.dropna()

# ============================================================

# ELITE SIGNALS

# ============================================================

if not trade_bt.empty and not structure_bt.empty:

    elite = pd.merge(

        trade_bt,

        structure_bt,

        on=["Date", "Ticker"],

        suffixes=("_10D", "_40D")

    )

else:

    elite = pd.DataFrame()

# ============================================================

# SUMMARY FUNCTION

# ============================================================

def summarize(df, return_col, alpha_col, label):

    if df is None or len(df) == 0:

        return pd.DataFrame([{

            "Signal": label,

            "Trades": 0,

            "AvgReturn": np.nan,

            "MedianReturn": np.nan,

            "HitRate": np.nan,

            "AvgAlpha": np.nan,

            "MedianAlpha": np.nan,

            "Best": np.nan,

            "Worst": np.nan

        }])

    return pd.DataFrame([{

        "Signal": label,

        "Trades": len(df),

        "AvgReturn": round(df[return_col].mean(), 2),

        "MedianReturn": round(df[return_col].median(), 2),

        "HitRate": round(

            (df[return_col] > 0).mean() * 100,

            1

        ),

        "AvgAlpha": round(df[alpha_col].mean(), 2),

        "MedianAlpha": round(df[alpha_col].median(), 2),

        "Best": round(df[return_col].max(), 2),

        "Worst": round(df[return_col].min(), 2)

    }])

# ============================================================

# SUMMARY TABLES

# ============================================================

trade_summary = summarize(

    trade_bt,

    "Return_10D",

    "Alpha10D",

    "10D TRADE"

)

structure_summary = summarize(

    structure_bt,

    "Return_40D",

    "Alpha40D",

    "40D STRUCTURE"

)

if not elite.empty:

    elite10_summary = summarize(

        elite,

        "Return_10D",

        "Alpha10D",

        "ELITE 10D"

    )

    elite40_summary = summarize(

        elite,

        "Return_40D",

        "Alpha40D",

        "ELITE 40D"

    )

else:

    elite10_summary = summarize(

        pd.DataFrame(),

        "Return_10D",

        "Alpha10D",

        "ELITE 10D"

    )

    elite40_summary = summarize(

        pd.DataFrame(),

        "Return_40D",

        "Alpha40D",

        "ELITE 40D"

    )

summary = pd.concat([

    trade_summary,

    structure_summary,

    elite10_summary,

    elite40_summary

]).reset_index(drop=True)

# ============================================================

# LIVE MODEL

# ============================================================

live_rows = []

for t in tickers:

    try:

        df = data.get(t)

        if df is None:

            continue

        core = add_core(df.tail(250))

        if len(core) < 5:

            continue

        row = core.iloc[-1]

        trade_sig = trade_signal(core)

        struct_sig = structure_signal(core)

        if trade_sig == "NO SIGNAL" and struct_sig == "NO SIGNAL":

            continue

        if trade_sig == "10D TRADE" and struct_sig == "40D STRUCTURE":

            combined = "ELITE"

        elif trade_sig == "10D TRADE":

            combined = "TRADE"

        elif struct_sig == "40D STRUCTURE":

            combined = "STRUCTURE"

        else:

            combined = "WATCH"

        live_rows.append([

            t,

            round(row["Close"], 2),

            round(row["RSI"], 1),

            round(row["RS20"], 1),

            round(row["RS60"], 1),

            int(row["Persistence"]),

            round(row["MACD_HIST"], 3),

            round(row["MACD_SLOPE"], 3),

            round(row["DI_PLUS"], 1),

            round(row["DI_MINUS"], 1),

            round(row["DMI_ACCEL"], 2),

            combined

        ])

    except Exception:

        pass

live = pd.DataFrame(

    live_rows,

    columns=[

        "Ticker","Price","RSI","RS20","RS60","Persistence",

        "MACD_HIST","MACD_SLOPE","DI_PLUS","DI_MINUS",

        "DMI_ACCEL","Signal"

    ]

)

if not live.empty:

    rank = {

        "ELITE": 1,

        "TRADE": 2,

        "STRUCTURE": 3,

        "WATCH": 4

    }

    live["Rank"] = live["Signal"].map(rank)

    live = (

        live

        .sort_values(

            ["Rank", "Persistence", "RS20"],

            ascending=[True, False, False]

        )

        .drop(columns=["Rank"])

        .reset_index(drop=True)

    )

# ============================================================

# OUTPUT

# ============================================================

print("\n============================================================")

print("🔥 POWERMODE OSLO v7.1 — FULL ALPHA TEST REGIME")

print("============================================================")

print(summary.to_string(index=False))

print("\n============================================================")

print("\n🔥 OSLO TOP LIVE SIGNALS")

print(V7_LIVE.head(10).to_string(index=False))

# print("\n============================================================")

# print("🔥 ELITE SIGNALS — BACKTEST SAMPLE")

# print("============================================================")

#

# if elite.empty:

#     print("No elite historical signals.")

# else:

#     print(elite.head(30).to_string(index=False))

# ============================================================

# EXPORT

# ============================================================

filename = f"powermode_oslo_v71_dual_engine_alpha_{TODAY}.xlsx"

with pd.ExcelWriter(filename, engine="openpyxl") as writer:

    trade_bt.to_excel(

        writer,

        sheet_name="10D_Trades",

        index=False

    )

    structure_bt.to_excel(

        writer,

        sheet_name="40D_Structure",

        index=False

    )

    elite.to_excel(

        writer,

        sheet_name="Elite",

        index=False

    )

    summary.to_excel(

        writer,

        sheet_name="Summary",

        index=False

    )

    live.to_excel(

        writer,

        sheet_name="Live_Model",

        index=False

    )

# ============================================================

# SAVE FOR MASTER + TELEGRAM FRIENDLY OUTPUT

# ============================================================

V7_LIVE = live.copy()

print("\n🔥 OSLO TOP LIVE SIGNALS")

if V7_LIVE.empty:

    print("No Oslo live signals.")

else:

    print(V7_LIVE.head(15).to_string(index=False))

print("\n✅ Saved:", filename)

print("✅ V7_LIVE stored:", V7_LIVE.shape)
