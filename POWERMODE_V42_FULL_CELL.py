# =========================================================

# 🔥 POWERMODE v4.2 – FULL KELTNER CHANNEL BUY / SELL ENGINE

# =========================================================

# Upgrade from v4.1:

# ✅ Real Keltner Channel: EMA20 +/- 2*ATR20

# ✅ Volatility-adjusted extension

# ✅ Same signal names as v4.1 for direct comparison

# ✅ EdgeTracker backtest

# ✅ Current BUY / SELL rankings

# =========================================================



import yfinance as yf

import pandas as pd

import numpy as np

from datetime import date

import warnings

warnings.filterwarnings("ignore")
display = print
# =========================================================

# SETTINGS

# =========================================================

HOLD_DAYS = 10

PERIOD = "3y"

KELTNER_ATR_MULT = 2.0

# =========================================================

# FULL OSE UNIVERSE

# =========================================================

tickers = [

    "AKRBP.OL","EQNR.OL","SUBC.OL","OET.OL","PEN.OL",

    "DNO.OL","VAR.OL","BWO.OL","SBX.OL","NORAM.OL",

    "AKSO.OL","BWE.OL","BNOR.OL","TGS.OL","DOFG.OL",

    "SOFF.OL","SDRL.OL","PNOR.OL",

    "MPCC.OL","HAUTO.OL","BWLPG.OL",

    "2020.OL","SFL.OL","FRO.OL","HAFNI.OL","HSHIP.OL","DVD.OL",

    "NHY.OL","YAR.OL","ELK.OL","TOM.OL","BORR.OL",

    "AFG.OL","VEI.OL","KCC.OL","KIT.OL",

    "DNB.OL","KOG.OL","STB.OL","GJF.OL",

    "MING.OL","PROT.OL","SBNOR.OL","BONHR.OL",

    "MOWI.OL","SALM.OL","LSG.OL","AKH.OL","AUSS.OL",

    "OLT.OL","SNTIA.OL",

    "ATEA.OL","LINK.OL","NORBT.OL",

    "NOD.OL","KID.OL","NAPA.OL","NORCO.OL","MULTI.OL",

    "DFENS.OL",

    "AUTO.OL","ODL.OL","BELCO.OL","PARB.OL",

    "ARR.OL","EPR.OL","RECSI.OL","WAWI.OL",

    "SATS.OL","CADLR.OL","CMBTO.OL",

    "ENVIP.OL","ARCH.OL","SKUE.OL","ABL.OL",

    "SMOP.OL","SPOL.OL","HSHP.OL","SOAG.OL","WWI.OL","WWIB.OL",

    "SCATC.OL","HEX.OL","MPCES.OL",

    "TEL.OL"

]

# Remove duplicates and Yahoo-broken names

tickers = sorted(list(set(tickers)))

bad_tickers = [

    "GOGL.OL",

    "CRAYN.OL",

    "XXL.OL"

]

tickers = [t for t in tickers if t not in bad_tickers]

# =========================================================

# DOWNLOAD DATA

# =========================================================

print("Downloading data...")

raw = yf.download(

    tickers,

    period=PERIOD,

    auto_adjust=True,

    progress=False,

    group_by="ticker",

    threads=False

)

data = {}

for t in tickers:

    try:

        df = raw[t].dropna()

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)

        df = df[["Open","High","Low","Close","Volume"]].dropna()

        if len(df) > 220:

            data[t] = df

    except Exception:

        pass

print("Loaded:", len(data))

missing = [t for t in tickers if t not in data]

print("\nMissing / not loaded:")

print(missing)

# =========================================================

# INDICATORS

# =========================================================

def ema(s, span):

    return s.ewm(span=span, adjust=False).mean()

def true_range(df):

    return pd.concat([

        df["High"] - df["Low"],

        (df["High"] - df["Close"].shift()).abs(),

        (df["Low"] - df["Close"].shift()).abs()

    ], axis=1).max(axis=1)

def kama(series, er_period=10, fast=2, slow=30):

    change = abs(series - series.shift(er_period))

    volatility = abs(series.diff()).rolling(er_period).sum()

    er = change / volatility

    sc = (

        er * (2/(fast+1) - 2/(slow+1))

        + 2/(slow+1)

    ) ** 2

    out = series.copy()

    out.iloc[:er_period] = series.iloc[:er_period]

    for i in range(er_period, len(series)):

        if pd.isna(sc.iloc[i]):

            out.iloc[i] = out.iloc[i-1]

        else:

            out.iloc[i] = out.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - out.iloc[i-1])

    return out

def rsi(series, period=21):

    delta = series.diff()

    gain = delta.clip(lower=0).rolling(period).mean()

    loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / loss

    return 100 - (100 / (1 + rs))

def dmi(df, period=10):

    high = df["High"]

    low = df["Low"]

    close = df["Close"]

    plus_dm_raw = high.diff()

    minus_dm_raw = -low.diff()

    plus_dm = np.where(

        (plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0),

        plus_dm_raw,

        0

    )

    minus_dm = np.where(

        (minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0),

        minus_dm_raw,

        0

    )

    tr = true_range(df)

    atr = tr.rolling(period).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).sum() / atr

    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).sum() / atr

    return plus_di, minus_di

# =========================================================

# FEATURE ENGINE — v4.2 FULL KELTNER

# =========================================================

def build_features(df):

    df = df.copy()

    # -----------------------------------------------------

    # KELTNER CHANNEL

    # -----------------------------------------------------

    df["atr20"] = true_range(df).rolling(20).mean()

    df["keltner_mid"] = ema(df["Close"], 20)

    df["keltner_upper"] = df["keltner_mid"] + KELTNER_ATR_MULT * df["atr20"]

    df["keltner_lower"] = df["keltner_mid"] - KELTNER_ATR_MULT * df["atr20"]

    df["keltner_mid_up"] = df["keltner_mid"] > df["keltner_mid"].shift(5)

    df["price_above_keltner_mid"] = df["Close"] > df["keltner_mid"]

    df["price_above_keltner_upper"] = df["Close"] > df["keltner_upper"]

    df["price_below_keltner_lower"] = df["Close"] < df["keltner_lower"]

    df["keltner_position"] = (

        (df["Close"] - df["keltner_lower"]) /

        (df["keltner_upper"] - df["keltner_lower"])

    )

    df["keltner_extension_pct"] = (

        (df["Close"] / df["keltner_upper"] - 1) * 100

    )

    # -----------------------------------------------------

    # KAMA

    # -----------------------------------------------------

    df["kama"] = kama(df["Close"])

    df["kama_up"] = df["kama"] > df["kama"].shift(5)

    df["price_above_kama"] = df["Close"] > df["kama"]

    df["distance_kama"] = (

        (df["Close"] / df["kama"] - 1) * 100

    )

    # -----------------------------------------------------

    # MACD

    # -----------------------------------------------------

    df["macd"] = ema(df["Close"], 12) - ema(df["Close"], 26)

    df["macd_signal"] = ema(df["macd"], 9)

    df["macd_above_signal"] = df["macd"] > df["macd_signal"]

    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["macd_hist_slope"] = df["macd_hist"] - df["macd_hist"].shift(1)

    df["macd_hist_pivot_up"] = (

        (df["macd_hist"] > df["macd_hist"].shift(1)) &

        (df["macd_hist"].shift(1) < df["macd_hist"].shift(2))

    )

    # -----------------------------------------------------

    # DMI

    # -----------------------------------------------------

    df["dmi_plus"], df["dmi_minus"] = dmi(df)

    df["dmi_plus_gt_minus"] = df["dmi_plus"] > df["dmi_minus"]

    df["dmi_minus_gt_plus"] = df["dmi_minus"] > df["dmi_plus"]

    df["dmi_spread"] = df["dmi_plus"] - df["dmi_minus"]

    df["dmi_spread_acceleration"] = df["dmi_spread"] - df["dmi_spread"].shift(1)

    df["dmi_spread_accelerating"] = df["dmi_spread_acceleration"] > 0

    # -----------------------------------------------------

    # RSI

    # -----------------------------------------------------

    df["rsi21"] = rsi(df["Close"], 21)

    df["rsi21_pivot_up"] = (

        (df["rsi21"] > df["rsi21"].shift(1)) &

        (df["rsi21"].shift(1) < df["rsi21"].shift(2))

    )

    df["rsi21_pivot_down"] = (

        (df["rsi21"] < df["rsi21"].shift(1)) &

        (df["rsi21"].shift(1) > df["rsi21"].shift(2))

    )

    # -----------------------------------------------------

    # STRUCTURE

    # -----------------------------------------------------

    df["price_above_keltner"] = df["price_above_keltner_mid"]

    df["holds_structure"] = (

        df["price_above_kama"] |

        df["price_above_keltner_mid"]

    )

    df["strong_structure"] = (

        df["price_above_kama"] &

        df["price_above_keltner_mid"] &

        df["kama_up"] &

        df["keltner_mid_up"]

    )

    # -----------------------------------------------------

    # FULL KELTNER EXTENSION LOGIC

    # -----------------------------------------------------

    df["mild_extension"] = (

        df["price_above_keltner_upper"] |

        (df["keltner_position"] > 1.0)

    )

    df["extreme_extension"] = (

        df["price_above_keltner_upper"] &

        (df["distance_kama"] > 8) &

        (df["keltner_extension_pct"] > 1.5)

    )

    df["healthy_extension"] = (

        (df["keltner_position"] >= 0.55) &

        (df["keltner_position"] <= 1.05) &

        (df["distance_kama"] <= 8)

    )

    # -----------------------------------------------------

    # RESET / RELOAD

    # -----------------------------------------------------

    df["momentum_reset"] = (

        (df["macd_hist"].shift(1) < df["macd_hist"].shift(4)) |

        (df["rsi21"].shift(1) < df["rsi21"].shift(4)) |

        (df["dmi_spread"].shift(1) < df["dmi_spread"].shift(4))

    )

    # -----------------------------------------------------

    # PIVOT

    # -----------------------------------------------------

    df["pivot_trigger"] = (

        (

            df["macd_hist_pivot_up"] |

            (df["macd_hist_slope"] > 0)

        ) &

        (

            df["dmi_spread_accelerating"] |

            df["rsi21_pivot_up"]

        ) &

        df["holds_structure"]

    )

    # -----------------------------------------------------

    # RE-ACCELERATION

    # -----------------------------------------------------

    df["reacceleration"] = (

        (df["macd_hist_slope"] > 0) &

        (

            df["dmi_spread_accelerating"] |

            (df["dmi_spread"] > df["dmi_spread"].shift(2))

        )

    )

    # -----------------------------------------------------

    # BREAKDOWN / EXIT

    # -----------------------------------------------------

    df["breakdown"] = (

        (

            ~df["price_above_kama"] &

            ~df["price_above_keltner_mid"]

        )

        |

        (

            ~df["macd_above_signal"] &

            df["dmi_minus_gt_plus"]

        )

        |

        (

            df["price_below_keltner_lower"] &

            df["dmi_minus_gt_plus"]

        )

    )

    return df

# =========================================================

# PROCESS

# =========================================================

processed = {}

for ticker in data.keys():

    daily = build_features(data[ticker])

    weekly = data[ticker].resample("W-FRI").agg({

        "Open":"first",

        "High":"max",

        "Low":"min",

        "Close":"last",

        "Volume":"sum"

    }).dropna()

    weekly = build_features(weekly)

    weekly_cols = [

        "kama_up",

        "keltner_mid_up",

        "macd_above_signal",

        "dmi_plus_gt_minus",

        "strong_structure"

    ]

    weekly = weekly[weekly_cols].add_prefix("weekly_")

    weekly = weekly.reindex(

        daily.index,

        method="ffill"

    )

    daily = pd.concat([daily, weekly], axis=1)

    # -----------------------------------------------------

    # SCORES

    # -----------------------------------------------------

    daily["weekly_score"] = (

        daily[[

            "weekly_kama_up",

            "weekly_keltner_mid_up",

            "weekly_macd_above_signal",

            "weekly_dmi_plus_gt_minus"

        ]]

        .fillna(False)

        .astype(int)

        .sum(axis=1)

        / 4

    )

    daily["daily_score"] = (

        daily[[

            "kama_up",

            "keltner_mid_up",

            "macd_above_signal",

            "dmi_plus_gt_minus",

            "pivot_trigger",

            "reacceleration"

        ]]

        .fillna(False)

        .astype(int)

        .sum(axis=1)

        / 6

    )

    daily["hybrid_score"] = (

        daily["daily_score"] * 0.40 +

        daily["weekly_score"] * 0.60

    )

    # -----------------------------------------------------

    # EDGE SCORE v4.2

    # -----------------------------------------------------

    daily["edge_score"] = (

          daily["weekly_score"] * 35

        + daily["daily_score"] * 35

        + daily["pivot_trigger"].astype(int) * 15

        + daily["reacceleration"].astype(int) * 15

        + daily["healthy_extension"].astype(int) * 5

        - daily["extreme_extension"].astype(int) * 25

        - (

            daily["price_above_keltner_upper"] &

            (daily["macd_hist_slope"] < 0)

          ).astype(int) * 10

    )

    # -----------------------------------------------------

    # SIGNALS — same names as v4.1

    # -----------------------------------------------------

    def signal(row):

        # ---------------- BUY ----------------

        if (

            row["weekly_score"] == 1

            and row["pivot_trigger"]

            and row["reacceleration"]

            and not row["extreme_extension"]

            and row["healthy_extension"]

        ):

            return "FULL QUALITY BUY"

        if (

            row["weekly_score"] >= 0.75

            and row["pivot_trigger"]

            and not row["extreme_extension"]

            and row["holds_structure"]

        ):

            return "POWERMODE CORE"

        if (

            row["weekly_score"] >= 0.75

            and row["momentum_reset"]

            and row["reacceleration"]

            and not row["breakdown"]

            and not row["extreme_extension"]

        ):

            return "HEALTHY PULLBACK"

        # ---------------- SELL ----------------

        if (

            row["extreme_extension"]

            and row["macd_hist_slope"] < 0

        ):

            return "TRIM / TAKE PROFIT"

        if (

            row["rsi21_pivot_down"]

            and row["macd_hist_slope"] < 0

            and (

                row["dmi_minus_gt_plus"]

                or row["price_above_keltner_upper"]

            )

        ):

            return "EXIT WARNING"

        if row["breakdown"]:

            return "EXIT NOW"

        return "NEUTRAL"

    daily["signal"] = daily.apply(signal, axis=1)

    processed[ticker] = daily

# =========================================================

# EDGE TRACKER

# =========================================================

all_rows = []

for ticker in processed.keys():

    df = processed[ticker].copy()

    df["ticker"] = ticker

    df["future_close"] = df["Close"].shift(-HOLD_DAYS)

    df["return_10d"] = (

        (df["future_close"] / df["Close"] - 1) * 100

    )

    all_rows.append(df)

all_df = pd.concat(all_rows)

market_return = (

    all_df

    .groupby(all_df.index)["return_10d"]

    .mean()

)

all_df["market_return"] = all_df.index.map(market_return)

all_df["alpha_10d"] = (

    all_df["return_10d"] - all_df["market_return"]

)

signals = [

    "FULL QUALITY BUY",

    "POWERMODE CORE",

    "HEALTHY PULLBACK",

    "TRIM / TAKE PROFIT",

    "EXIT WARNING",

    "EXIT NOW",

    "NEUTRAL"

]

results = []

for s in signals:

    sample = all_df[

        all_df["signal"] == s

    ].dropna(subset=["return_10d"])

    if len(sample) < 20:

        continue

    results.append({

        "Signal": s,

        "Trades": len(sample),

        "AvgReturn10d": round(sample["return_10d"].mean(), 2),

        "HitRate": round((sample["return_10d"] > 0).mean() * 100, 1),

        "AvgAlpha10d": round(sample["alpha_10d"].mean(), 2),

        "MedianReturn10d": round(sample["return_10d"].median(), 2),

        "Best": round(sample["return_10d"].max(), 2),

        "Worst": round(sample["return_10d"].min(), 2)

    })

edge = pd.DataFrame(results)

edge = edge.sort_values(

    by="AvgAlpha10d",

    ascending=False

).reset_index(drop=True)

print("")

print("=================================================")

print("🔥 POWERMODE v4.2 FULL KELTNER — EDGE TRACKER")

print("=================================================")

display(edge)

# =========================================================

# LIVE BUY / SELL RANKINGS

# =========================================================

rows = []

for ticker in processed.keys():

    last = processed[ticker].dropna().iloc[-1]

    score = last["edge_score"]

    if score >= 85:

        grade = "A+"

    elif score >= 75:

        grade = "A"

    elif score >= 65:

        grade = "B"

    elif score >= 55:

        grade = "C"

    else:

        grade = "D"

    rows.append({

        "Ticker": ticker,

        "Price": round(last["Close"], 2),

        "Weekly": round(last["weekly_score"], 2),

        "Daily": round(last["daily_score"], 2),

        "Hybrid": round(last["hybrid_score"], 2),

        "EdgeScore": round(score, 1),

        "Grade": grade,

        "Pivot": bool(last["pivot_trigger"]),

        "ReAccel": bool(last["reacceleration"]),

        "Reset": bool(last["momentum_reset"]),

        "HealthyExt": bool(last["healthy_extension"]),

        "MildExt": bool(last["mild_extension"]),

        "Extended": bool(last["extreme_extension"]),

        "KeltnerPos": round(last["keltner_position"], 2),

        "KeltnerExt%": round(last["keltner_extension_pct"], 2),

        "DistKAMA%": round(last["distance_kama"], 2),

        "RSI21": round(last["rsi21"], 1),

        "Signal": last["signal"]

    })

live = pd.DataFrame(rows)

buy_list = live[

    live["Signal"].isin([

        "FULL QUALITY BUY",

        "POWERMODE CORE",

        "HEALTHY PULLBACK"

    ])

].sort_values(

    by="EdgeScore",

    ascending=False

).reset_index(drop=True)

sell_list = live[

    live["Signal"].isin([

        "TRIM / TAKE PROFIT",

        "EXIT WARNING",

        "EXIT NOW"

    ])

].sort_values(

    by="EdgeScore",

    ascending=True

).reset_index(drop=True)

print("")

print("=================================================")

print("🔥 CURRENT TOP BUYS — v4.2 FULL KELTNER")

print("=================================================")

display(buy_list.head(25))

print("")

print("=================================================")

print("🔥 CURRENT SELL / EXIT LIST — v4.2 FULL KELTNER")

print("=================================================")

display(sell_list.head(25))

# =========================================================

# EXPORT

# =========================================================

filename = f"powermode_v42_full_keltner_{date.today()}.xlsx"

with pd.ExcelWriter(filename, engine="openpyxl") as writer:

    edge.to_excel(writer, sheet_name="EdgeTracker", index=False)

    buy_list.to_excel(writer, sheet_name="TopBuys", index=False)

    sell_list.to_excel(writer, sheet_name="SellExit", index=False)

    live.to_excel(writer, sheet_name="LiveAll", index=False)

print("")

print("✅ Saved:", filename)

# =========================================================
# STORE FOR MASTER
# =========================================================

V42_BUY = buy_list.copy()
V42_SELL = sell_list.copy()
V42_LIVE = live.copy()
V42_EDGE = edge.copy()

print("")
print("✅ V42 stored for MASTER")
print("BUY:", V42_BUY.shape)
print("SELL:", V42_SELL.shape)
print("LIVE:", V42_LIVE.shape)
print("EDGE:", V42_EDGE.shape)
