# ============================================================
# POWERMODE SVERIGE V8.1 – EDGE TRACKER / KORRIGERT
# One-cell version
# Scan + Historical Edge + 10D/40D backtest
# Stores: SE_SCAN, SE_BT for MASTER
# ============================================================



import yfinance as yf
import pandas as pd
import numpy as np
# import pandas_ta as ta
from datetime import datetime
display = print
# -----------------------------
# TICKERS
# -----------------------------

TICKERS = [
    "VOLV-B.ST", "ATCO-A.ST", "ATCO-B.ST", "ABB.ST", "SAND.ST",
    "EPI-A.ST", "ALFA.ST", "SKF-B.ST", "ASSA-B.ST", "INDT.ST",
    "LATO-B.ST", "LIFCO-B.ST", "ADDT-B.ST", "BEIJ-B.ST", "OEM-B.ST",
    "SEB-A.ST", "SWED-A.ST", "SHB-A.ST", "INVE-B.ST", "KINV-B.ST",
    "EQT.ST", "AZA.ST",
    "GETI-B.ST", "ARJO-B.ST", "ALIV-SDB.ST", "BIOA-B.ST", "VITR.ST",
    "EVO.ST", "SINCH.ST", "TRUE-B.ST", "MIPS.ST",
    "VIT-B.ST", "MSAB-B.ST",
    "BALD-B.ST", "CAST.ST", "FABG.ST", "WIHL.ST", "HUFV-A.ST",
    "NP3.ST", "SBB-B.ST",
    "BOL.ST", "SSAB-A.ST", "SSAB-B.ST", "HOLM-B.ST", "SCA-B.ST",
    "NIBE-B.ST", "TREL-B.ST",
    "THULE.ST", "AAK.ST", "AXFO.ST", "DOM.ST", "BHG.ST",
    "BUFAB.ST", "INSTAL.ST", "CTT.ST", "WALL-B.ST"
]

PERIOD = "2y"

# -----------------------------
# HELPERS
# -----------------------------

def flatten(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def clean_ohlcv(df):
    df = flatten(df).copy()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame()
    df = df[needed].replace([np.inf, -np.inf], np.nan).dropna()
    return df

def add_indicators(df):
    df = clean_ohlcv(df)

    if df.empty or len(df) < 220:
        return pd.DataFrame()

    df["EMA10"] = ta.ema(df["Close"], length=10)
    df["EMA20"] = ta.ema(df["Close"], length=20)
    df["EMA50"] = ta.ema(df["Close"], length=50)
    df["EMA200"] = ta.ema(df["Close"], length=200)
    df["KAMA"] = ta.kama(df["Close"], length=10)

    macd = ta.macd(df["Close"])
    if macd is None or macd.empty or macd.shape[1] < 3:
        return pd.DataFrame()

    df["MACD"] = macd.iloc[:, 0]
    df["MACD_Hist"] = macd.iloc[:, 1]
    df["MACD_Signal"] = macd.iloc[:, 2]
    df["MACD_Accel"] = df["MACD_Hist"] - df["MACD_Hist"].shift(1)
    df["MACD_Accel_3D"] = df["MACD_Hist"] - df["MACD_Hist"].shift(3)

    df["RSI"] = ta.rsi(df["Close"], length=14)

    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx is None or adx.empty:
        return pd.DataFrame()

    if "ADX_14" not in adx.columns or "DMP_14" not in adx.columns or "DMN_14" not in adx.columns:
        return pd.DataFrame()

    df["ADX"] = adx["ADX_14"]
    df["+DI"] = adx["DMP_14"]
    df["-DI"] = adx["DMN_14"]
    df["DI_Spread"] = df["+DI"] - df["-DI"]
    df["DMI_Accel"] = df["DI_Spread"] - df["DI_Spread"].shift(1)

    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["ATR_%"] = df["ATR"] / df["Close"] * 100

    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Vol_MA20"]

    df["KAMA_Dist_%"] = (df["Close"] - df["KAMA"]) / df["KAMA"] * 100
    df["EMA20_Dist_%"] = (df["Close"] - df["EMA20"]) / df["EMA20"] * 100

    return df.replace([np.inf, -np.inf], np.nan).dropna()

def add_weekly(df):
    if df.empty:
        return pd.DataFrame()

    w = df.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    if w.empty or len(w) < 40:
        return pd.DataFrame()

    w["W_EMA10"] = ta.ema(w["Close"], length=10)
    w["W_EMA30"] = ta.ema(w["Close"], length=30)
    w["W_RSI"] = ta.rsi(w["Close"], length=14)

    wm = ta.macd(w["Close"])
    if wm is None or wm.empty or wm.shape[1] < 2:
        return pd.DataFrame()

    w["W_MACD_Hist"] = wm.iloc[:, 1]
    w["W_MACD_Accel"] = w["W_MACD_Hist"] - w["W_MACD_Hist"].shift(1)

    return w.replace([np.inf, -np.inf], np.nan).dropna()

def edge_score(last, wlast):
    score = 0

    # Trend permission
    if last["Close"] > last["EMA20"]:
        score += 10
    if last["EMA20"] > last["EMA50"]:
        score += 10
    if last["Close"] > last["EMA50"]:
        score += 10
    if last["EMA50"] > last["EMA200"]:
        score += 6

    # Main edge: acceleration
    if last["MACD_Hist"] > 0:
        score += 10
    if last["MACD_Accel"] > 0:
        score += 20
    if last["MACD_Accel_3D"] > 0:
        score += 16

    # RSI
    if 52 <= last["RSI"] <= 68:
        score += 20
    elif 45 <= last["RSI"] < 52:
        score += 6
    elif 68 < last["RSI"] <= 72 and last["MACD_Accel"] > 0:
        score += 4
    elif last["RSI"] > 72:
        score -= 15

    # DMI
    if last["+DI"] > last["-DI"]:
        score += 12
    if last["DMI_Accel"] > 0:
        score += 12
    if last["ADX"] > 20 and last["+DI"] > last["-DI"]:
        score += 6

    # Asymmetry / extension
    if 0 <= last["KAMA_Dist_%"] <= 8:
        score += 14
    elif 8 < last["KAMA_Dist_%"] <= 12:
        score += 2
    elif last["KAMA_Dist_%"] > 12:
        score -= 20
    elif last["KAMA_Dist_%"] < -3:
        score -= 10

    # Volume
    if last["Volume_Ratio"] > 1.1:
        score += 5
    if last["Volume_Ratio"] > 1.6:
        score += 5

    # Weekly overlay
    if wlast["Close"] > wlast["W_EMA10"]:
        score += 4
    if wlast["W_EMA10"] > wlast["W_EMA30"]:
        score += 6
    if wlast["W_MACD_Hist"] > 0:
        score += 4
    if wlast["W_MACD_Accel"] > 0:
        score += 6

    # Exhaustion penalties
    if last["RSI"] > 68 and last["MACD_Accel"] < 0:
        score -= 25
    if last["RSI"] > 70 and last["DMI_Accel"] < 0:
        score -= 18
    if last["KAMA_Dist_%"] > 10 and last["MACD_Accel"] < 0:
        score -= 20
    if wlast["W_MACD_Accel"] < 0 and last["MACD_Accel"] < 0:
        score -= 15
    if last["MACD_Accel_3D"] < 0 and last["DMI_Accel"] < 0:
        score -= 15

    return round(score, 1)

def strict_signal(row):
    # Hard downgrade rules
    if row["RSI"] > 68 and row["MACD_Accel"] < 0:
        return "EXHAUSTION WATCH"

    if row["KAMA_Dist_%"] > 12 and row["MACD_Accel"] < 0:
        return "EXHAUSTION WATCH"

    if row["MACD_Accel"] < 0 and row["MACD_Accel_3D"] < 0:
        return "WATCH"

    # True EDGE must have acceleration now + 3D
    if (
        row["EdgeScore"] >= 90
        and row["MACD_Accel"] > 0
        and row["MACD_Accel_3D"] > 0
        and row["DMI_Accel"] > 0
        and row["+DI"] > row["-DI"]
        and 52 <= row["RSI"] <= 68
        and 0 <= row["KAMA_Dist_%"] <= 10
    ):
        return "EDGE LEADER"

    if (
        row["EdgeScore"] >= 75
        and row["MACD_Accel"] > 0
        and row["+DI"] > row["-DI"]
        and row["RSI"] <= 68
        and row["KAMA_Dist_%"] <= 10
    ):
        return "POWER CORE"

    if (
        row["MACD_Accel"] > 0
        and row["DMI_Accel"] > 0
        and 45 <= row["RSI"] <= 60
        and row["Close"] > row["EMA50"]
    ):
        return "EARLY EDGE"

    if row["EdgeScore"] < 45:
        return "AVOID"

    return "WATCH"

def management(signal):
    if signal == "EDGE LEADER":
        return "BUY / HOLD"
    if signal == "POWER CORE":
        return "BUY SMALL / HOLD"
    if signal == "EARLY EDGE":
        return "WATCH / STARTER"
    if signal == "EXHAUSTION WATCH":
        return "DO NOT CHASE"
    if signal == "AVOID":
        return "AVOID"
    return "WAIT"

def comment(signal):
    comments = {
        "EDGE LEADER": "Ren acceleration-edge",
        "POWER CORE": "Continuation med kjøperstyrke",
        "EARLY EDGE": "Tidlig forbedring, vent på bekreftelse",
        "EXHAUSTION WATCH": "Momentum svekkes, ikke jag",
        "AVOID": "Svak eller negativ edge",
        "WATCH": "Ikke nok edge ennå"
    }
    return comments.get(signal, "")

def v81_signal_series(df):
    return (
        (df["Close"] > df["EMA20"])
        & (df["EMA20"] > df["EMA50"])
        & (df["Close"] > df["EMA50"])
        & (df["MACD_Accel"] > 0)
        & (df["MACD_Accel_3D"] > 0)
        & (df["RSI"].between(52, 68))
        & (df["+DI"] > df["-DI"])
        & (df["DMI_Accel"] > 0)
        & (df["KAMA_Dist_%"].between(0, 10))
        & ~((df["RSI"] > 68) & (df["MACD_Accel"] < 0))
        & ~((df["KAMA_Dist_%"] > 10) & (df["MACD_Accel"] < 0))
    )

# -----------------------------
# DOWNLOAD + COMPUTE
# -----------------------------

scan_rows = []
bt_rows = []
failed = []

for ticker in TICKERS:
    try:
        raw = yf.download(
            ticker,
            period=PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        raw = clean_ohlcv(raw)

        if raw.empty or len(raw) < 260:
            failed.append((ticker, "empty/insufficient raw data"))
            continue

        df = add_indicators(raw)
        w = add_weekly(df)

        if df.empty or w.empty:
            failed.append((ticker, "indicator/weekly empty"))
            continue

        last = df.iloc[-1]
        wlast = w.iloc[-1]

        score = edge_score(last, wlast)

        row = {
            "Ticker": ticker,
            "Close": round(last["Close"], 2),
            "EdgeScore": score,
            "RSI": round(last["RSI"], 1),
            "KAMA_Dist_%": round(last["KAMA_Dist_%"], 1),
            "MACD_Accel": round(last["MACD_Accel"], 3),
            "MACD_Accel_3D": round(last["MACD_Accel_3D"], 3),
            "+DI": round(last["+DI"], 1),
            "-DI": round(last["-DI"], 1),
            "DMI_Accel": round(last["DMI_Accel"], 2),
            "ADX": round(last["ADX"], 1),
            "Volume_Ratio": round(last["Volume_Ratio"], 2),
            "W_MACD_Accel": round(wlast["W_MACD_Accel"], 3),
            "EMA20": round(last["EMA20"], 2),
            "EMA50": round(last["EMA50"], 2),
            "EMA200": round(last["EMA200"], 2),
        }

        row["Signal"] = strict_signal(row)
        row["Management"] = management(row["Signal"])
        row["EDGE TRACKER"] = comment(row["Signal"])

        # Backtest
        df["Signal"] = v81_signal_series(df)
        df["Return_10D"] = (df["Close"].shift(-10) / df["Close"] - 1) * 100
        df["Return_40D"] = (df["Close"].shift(-40) / df["Close"] - 1) * 100

        sig = df[df["Signal"]].copy()

        if len(sig) > 0:
            avg10 = sig["Return_10D"].mean()
            avg40 = sig["Return_40D"].mean()
            hr10 = (sig["Return_10D"] > 0).mean() * 100
            hr40 = (sig["Return_40D"] > 0).mean() * 100

            hist_score = (
                max(min(avg40, 30), -20) * 1.5
                + (hr40 - 50) * 0.8
                + max(min(avg10, 12), -8) * 1.0
            )

            row["Signals"] = len(sig)
            row["Avg_10D"] = round(avg10, 2)
            row["HitRate_10D"] = round(hr10, 1)
            row["Avg_40D"] = round(avg40, 2)
            row["HitRate_40D"] = round(hr40, 1)
            row["HistoricalEdge"] = round(hist_score, 1)

            bt_rows.append({
                "Ticker": ticker,
                "Signals": len(sig),
                "Avg_10D": round(avg10, 2),
                "HitRate_10D": round(hr10, 1),
                "Avg_40D": round(avg40, 2),
                "HitRate_40D": round(hr40, 1),
                "HistoricalEdge": round(hist_score, 1)
            })

        else:
            row["Signals"] = 0
            row["Avg_10D"] = np.nan
            row["HitRate_10D"] = np.nan
            row["Avg_40D"] = np.nan
            row["HitRate_40D"] = np.nan
            row["HistoricalEdge"] = -20

        row["CompositeRank"] = round(row["EdgeScore"] + row["HistoricalEdge"], 1)
        scan_rows.append(row)

    except Exception as e:
        failed.append((ticker, str(e)))
        print(f"Feil på {ticker}: {e}")

# -----------------------------
# OUTPUT
# -----------------------------

scan = pd.DataFrame(scan_rows)

display_cols = [
    "Ticker", "Close", "CompositeRank", "EdgeScore", "HistoricalEdge",
    "Signal", "Management", "EDGE TRACKER",
    "Signals", "Avg_10D", "HitRate_10D", "Avg_40D", "HitRate_40D",
    "RSI", "KAMA_Dist_%", "MACD_Accel", "MACD_Accel_3D",
    "+DI", "-DI", "DMI_Accel", "ADX", "Volume_Ratio", "W_MACD_Accel"
]

if not scan.empty:
    scan = scan.sort_values(
        by=["CompositeRank", "EdgeScore"],
        ascending=False
    ).reset_index(drop=True)

    print("POWERMODE SVERIGE V8.1 – EDGE TRACKER / KORRIGERT")
    print("Dato:", datetime.now().strftime("%Y-%m-%d %H:%M"))
    display(scan[display_cols])

bt = pd.DataFrame(bt_rows)

if not bt.empty:
    print("\nBACKTEST – V8.1 EDGE SIGNAL")
    display(
        bt.sort_values(
            by=["HistoricalEdge", "Avg_40D"],
            ascending=False
        ).reset_index(drop=True)
    )
else:
    print("Ingen backtest-signaler.")

if failed:
    print("\nTickers skipped / failed:")
    print(pd.DataFrame(failed, columns=["Ticker", "Reason"]).head(30))

# -----------------------------
# EXPORT + STORE FOR MASTER
# -----------------------------

today = datetime.now().strftime("%Y-%m-%d")
filename = f"POWERMODE_SVERIGE_V81_{today}.xlsx"

with pd.ExcelWriter(filename, engine="openpyxl") as writer:
    scan.to_excel(writer, sheet_name="Scan", index=False)
    bt.to_excel(writer, sheet_name="Backtest", index=False)
    pd.DataFrame(failed, columns=["Ticker", "Reason"]).to_excel(
        writer,
        sheet_name="Failed",
        index=False
    )

SE_SCAN = scan.copy()
SE_BT = bt.copy()

print("\n✅ Saved:", filename)
print("✅ SE_SCAN stored:", SE_SCAN.shape)
print("✅ SE_BT stored:", SE_BT.shape)
