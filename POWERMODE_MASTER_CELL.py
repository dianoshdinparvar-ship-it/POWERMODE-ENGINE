# ============================================================
# 🔥 POWERMODE MASTER CELL — NORGE + SVERIGE + V4.2 EXIT OVERLAY
# Robust one-cell version
#
# Requires previous cells to have created:
#   V7_LIVE   from Oslo V7.1
#   V42_SELL  from V4.2 Keltner sell/exit engine
#   V42_LIVE  from V4.2 Keltner full live table
#   SE_SCAN   from Sverige V8.1
#
# Optional:
#   results / US_RESULTS from USA scanner
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime
display = print
# ============================================================
# HELPERS
# ============================================================

def is_df(x):
    return isinstance(x, pd.DataFrame) and not x.empty

def get_global_var(name):
    return globals().get(name, None)

def safe_col(df, col, default=np.nan):
    if isinstance(df, pd.DataFrame) and col in df.columns:
        return df[col]
    return default

def normalize_ticker(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def final_rank_value(signal):
    order = {
        "STRONG BUY": 1,
        "BUY": 2,
        "HOLD": 3,
        "WATCH": 4,
        "TRIM": 5,
        "EXIT": 6,
        "AVOID": 7
    }
    return order.get(str(signal), 99)

# ============================================================
# LOAD OBJECTS FROM PREVIOUS CELLS
# ============================================================

V7 = get_global_var("V7_LIVE")
V42_SELL_OBJ = get_global_var("V42_SELL")
V42_LIVE_OBJ = get_global_var("V42_LIVE")
SE = get_global_var("SE_SCAN")

# Optional USA, kept disabled/soft
USA = None
for possible_name in ["US_RESULTS", "USA_RESULTS", "results"]:
    obj = get_global_var(possible_name)
    if is_df(obj):
        USA = obj.copy()
        break

print("================================================")
print("DATA CHECK")
print("================================================")

print("V7_LIVE:", "DataFrame " + str(V7.shape) if is_df(V7) else "missing/empty")
print("V42_SELL:", "DataFrame " + str(V42_SELL_OBJ.shape) if is_df(V42_SELL_OBJ) else "missing/empty")
print("V42_LIVE:", "DataFrame " + str(V42_LIVE_OBJ.shape) if is_df(V42_LIVE_OBJ) else "missing/empty")
print("SE_SCAN:", "DataFrame " + str(SE.shape) if is_df(SE) else "missing/empty")
print("USA:", "DataFrame " + str(USA.shape) if is_df(USA) else "skipped / not active")

# ============================================================
# NORWAY MASTER — V7.1 + V4.2 EXIT OVERLAY
# ============================================================

norway_master = pd.DataFrame()

if is_df(V7):
    v7 = V7.copy()
    v7["Ticker"] = v7["Ticker"].apply(normalize_ticker)

    # V7 signal mapping
    def map_v7_signal(sig):
        sig = str(sig).upper().strip()
        if sig == "ELITE":
            return "STRONG BUY"
        if sig == "TRADE":
            return "BUY"
        if sig == "STRUCTURE":
            return "HOLD"
        if sig == "WATCH":
            return "WATCH"
        return "WATCH"

    v7["OriginalSignal"] = v7["Signal"].apply(map_v7_signal)

    # V4.2 exit overlay from sell_list
    exit_map = {}

    if is_df(V42_SELL_OBJ):
        v42_sell = V42_SELL_OBJ.copy()
        v42_sell["Ticker"] = v42_sell["Ticker"].apply(normalize_ticker)

        for _, r in v42_sell.iterrows():
            sig = str(r.get("Signal", "")).upper()

            if "EXIT NOW" in sig:
                exit_map[r["Ticker"]] = "EXIT"
            elif "EXIT WARNING" in sig:
                exit_map[r["Ticker"]] = "TRIM"
            elif "TRIM" in sig or "TAKE PROFIT" in sig:
                exit_map[r["Ticker"]] = "TRIM"

    v7["ExitOverlay"] = v7["Ticker"].map(exit_map).fillna("HOLD")

    def combine_norway(original, overlay):
        original = str(original)
        overlay = str(overlay)

        if overlay == "EXIT":
            return "EXIT"
        if overlay == "TRIM":
            if original in ["STRONG BUY", "BUY"]:
                return "TRIM"
            return "WATCH"
        return original

    v7["Final"] = [
        combine_norway(o, e)
        for o, e in zip(v7["OriginalSignal"], v7["ExitOverlay"])
    ]

    norway_master = pd.DataFrame({
        "Ticker": v7["Ticker"],
        "Country": "NO",
        "Model": "Oslo V7.1 + V4.2 Exit",
        "OriginalSignal": v7["OriginalSignal"],
        "ExitOverlay": v7["ExitOverlay"],
        "Final": v7["Final"],
        "Price": safe_col(v7, "Price"),
        "RSI": safe_col(v7, "RSI"),
        "RS20": safe_col(v7, "RS20"),
        "RS60": safe_col(v7, "RS60"),
        "Persistence": safe_col(v7, "Persistence"),
        "MACD_HIST": safe_col(v7, "MACD_HIST"),
        "MACD_SLOPE": safe_col(v7, "MACD_SLOPE"),
        "DI_PLUS": safe_col(v7, "DI_PLUS"),
        "DI_MINUS": safe_col(v7, "DI_MINUS"),
        "DMI_ACCEL": safe_col(v7, "DMI_ACCEL")
    })

    norway_master["FinalRank"] = norway_master["Final"].apply(final_rank_value)
    norway_master = norway_master.sort_values(
        by=["FinalRank", "Persistence", "RS20"],
        ascending=[True, False, False]
    ).drop(columns=["FinalRank"]).reset_index(drop=True)

else:
    print("WARNING: V7_LIVE missing. Norway master skipped.")

# ============================================================
# SWEDEN MASTER — V8.1 EDGE
# ============================================================

sweden_master = pd.DataFrame()

if is_df(SE):
    se = SE.copy()
    se["Ticker"] = se["Ticker"].apply(normalize_ticker)

    def map_sweden_final(sig):
        sig = str(sig).upper().strip()

        if sig == "EDGE LEADER":
            return "STRONG BUY"
        if sig == "POWER CORE":
            return "BUY"
        if sig == "EARLY EDGE":
            return "WATCH"
        if sig == "EXHAUSTION WATCH":
            return "TRIM"
        if sig == "AVOID":
            return "EXIT"
        return "WATCH"

    se["Final"] = se["Signal"].apply(map_sweden_final)

    edge_tracker_col = "EDGE TRACKER" if "EDGE TRACKER" in se.columns else None

    sweden_master = pd.DataFrame({
        "Ticker": se["Ticker"],
        "Country": "SE",
        "Model": "Sverige V8.1 Edge",
        "OriginalSignal": se["Signal"],
        "ExitOverlay": "",
        "Final": se["Final"],
        "Close": safe_col(se, "Close"),
        "CompositeRank": safe_col(se, "CompositeRank"),
        "EdgeScore": safe_col(se, "EdgeScore"),
        "HistoricalEdge": safe_col(se, "HistoricalEdge"),
        "Management": safe_col(se, "Management"),
        "EDGE_TRACKER": se[edge_tracker_col] if edge_tracker_col else "",
        "Signals": safe_col(se, "Signals"),
        "Avg_10D": safe_col(se, "Avg_10D"),
        "HitRate_10D": safe_col(se, "HitRate_10D"),
        "Avg_40D": safe_col(se, "Avg_40D"),
        "HitRate_40D": safe_col(se, "HitRate_40D"),
        "RSI": safe_col(se, "RSI"),
        "KAMA_Dist_%": safe_col(se, "KAMA_Dist_%"),
        "MACD_Accel": safe_col(se, "MACD_Accel"),
        "MACD_Accel_3D": safe_col(se, "MACD_Accel_3D"),
        "DMI_Accel": safe_col(se, "DMI_Accel"),
        "ADX": safe_col(se, "ADX"),
        "Volume_Ratio": safe_col(se, "Volume_Ratio")
    })

    sweden_master["FinalRank"] = sweden_master["Final"].apply(final_rank_value)
    sweden_master = sweden_master.sort_values(
        by=["FinalRank", "CompositeRank", "EdgeScore"],
        ascending=[True, False, False]
    ).drop(columns=["FinalRank"]).reset_index(drop=True)

else:
    print("WARNING: SE_SCAN missing. Sweden master skipped.")

# ============================================================
# OPTIONAL USA MASTER — SOFT / DISABLED IF NOT DATAFRAME
# ============================================================

usa_master = pd.DataFrame()

if is_df(USA):
    us = USA.copy()

    if "Ticker" in us.columns and "Signal" in us.columns:
        us["Ticker"] = us["Ticker"].apply(normalize_ticker)

        def map_us_final(sig):
            sig = str(sig).upper().strip()

            if sig == "FULL QUALITY":
                return "STRONG BUY"
            if sig in ["ACCELERATING LEADER", "POWERMODE CORE"]:
                return "BUY"
            if sig in ["WATCH / STRETCHED", "RECOVERY WATCH", "NEUTRAL WATCH"]:
                return "WATCH"
            if sig == "AVOID":
                return "EXIT"
            return "WATCH"

        us["Final"] = us["Signal"].apply(map_us_final)

        usa_master = pd.DataFrame({
            "Ticker": us["Ticker"],
            "Country": "US",
            "Model": "USA Fixed",
            "OriginalSignal": us["Signal"],
            "ExitOverlay": "",
            "Final": us["Final"],
            "Close": safe_col(us, "Close"),
            "HybridScore": safe_col(us, "HybridScore"),
            "DailyScore": safe_col(us, "DailyScore"),
            "WeeklyScore": safe_col(us, "WeeklyScore"),
            "EdgeScore": safe_col(us, "EdgeScore"),
            "EdgeSignal": safe_col(us, "EdgeSignal"),
            "RSI": safe_col(us, "RSI"),
            "ADX": safe_col(us, "ADX"),
            "Dist_KAMA_%": safe_col(us, "Dist_KAMA_%"),
            "Dist_EMA200_%": safe_col(us, "Dist_EMA200_%"),
            "VolumeExpansion": safe_col(us, "VolumeExpansion")
        })

        usa_master["FinalRank"] = usa_master["Final"].apply(final_rank_value)
        usa_master = usa_master.sort_values(
            by=["FinalRank", "HybridScore", "EdgeScore"],
            ascending=[True, False, False]
        ).drop(columns=["FinalRank"]).reset_index(drop=True)
    else:
        print("WARNING: USA object found but lacks Ticker/Signal. USA skipped.")

# ============================================================
# GLOBAL MASTER
# ============================================================

global_parts = []

if is_df(sweden_master):
    global_parts.append(sweden_master)

if is_df(usa_master):
    global_parts.append(usa_master)

if global_parts:
    global_master = pd.concat(global_parts, ignore_index=True, sort=False)
    global_master["FinalRank"] = global_master["Final"].apply(final_rank_value)

    sort_cols = ["FinalRank"]
    ascending = [True]

    if "CompositeRank" in global_master.columns:
        sort_cols.append("CompositeRank")
        ascending.append(False)
    elif "HybridScore" in global_master.columns:
        sort_cols.append("HybridScore")
        ascending.append(False)

    if "EdgeScore" in global_master.columns:
        sort_cols.append("EdgeScore")
        ascending.append(False)

    global_master = global_master.sort_values(
        by=sort_cols,
        ascending=ascending
    ).drop(columns=["FinalRank"]).reset_index(drop=True)
else:
    global_master = pd.DataFrame()

# ============================================================
# PRIORITY VIEWS
# ============================================================

norway_action = pd.DataFrame()
global_action = pd.DataFrame()
exit_list = pd.DataFrame()

if is_df(norway_master):
    norway_action = norway_master[
        norway_master["Final"].isin(["STRONG BUY", "BUY", "TRIM", "EXIT"])
    ].reset_index(drop=True)

if is_df(global_master):
    global_action = global_master[
        global_master["Final"].isin(["STRONG BUY", "BUY", "TRIM", "EXIT"])
    ].reset_index(drop=True)

exit_parts = []

if is_df(norway_master):
    exit_parts.append(
        norway_master[norway_master["Final"].isin(["TRIM", "EXIT"])].copy()
    )

if is_df(global_master):
    exit_parts.append(
        global_master[global_master["Final"].isin(["TRIM", "EXIT"])].copy()
    )

if exit_parts:
    exit_list = pd.concat(exit_parts, ignore_index=True, sort=False)
    exit_list["FinalRank"] = exit_list["Final"].apply(final_rank_value)
    exit_list = exit_list.sort_values(
        by=["FinalRank", "Country", "Ticker"],
        ascending=[True, True, True]
    ).drop(columns=["FinalRank"]).reset_index(drop=True)

# ============================================================
# DISPLAY
# ============================================================

print("\n================================================")
print("NORWAY MASTER — V7.1 + V4.2 EXIT OVERLAY")
print("================================================")

if is_df(norway_master):
    display(norway_master)
else:
    print("No Norway master data.")

print("\n================================================")
print("NORWAY ACTION LIST")
print("================================================")

if is_df(norway_action):
    display(norway_action)
else:
    print("No Norway action signals.")

print("\n================================================")
print("GLOBAL MASTER — SWEDEN V8.1 + OPTIONAL USA")
print("================================================")

if is_df(global_master):
    display(global_master)
else:
    print("No global master data.")

print("\n================================================")
print("GLOBAL ACTION LIST")
print("================================================")

if is_df(global_action):
    display(global_action)
else:
    print("No global action signals.")

print("\n================================================")
print("EXIT / TRIM LIST — ALL MARKETS")
print("================================================")

if is_df(exit_list):
    display(exit_list)
else:
    print("No exit/trim signals.")

# ============================================================
# EXPORT
# ============================================================

today = datetime.now().strftime("%Y-%m-%d")
filename = f"POWERMODE_MASTER_{today}.xlsx"

with pd.ExcelWriter(filename, engine="openpyxl") as writer:
    if is_df(norway_master):
        norway_master.to_excel(writer, sheet_name="Norway_Master", index=False)
    if is_df(norway_action):
        norway_action.to_excel(writer, sheet_name="Norway_Action", index=False)
    if is_df(sweden_master):
        sweden_master.to_excel(writer, sheet_name="Sweden_Master", index=False)
    if is_df(usa_master):
        usa_master.to_excel(writer, sheet_name="USA_Master", index=False)
    if is_df(global_master):
        global_master.to_excel(writer, sheet_name="Global_Master", index=False)
    if is_df(global_action):
        global_action.to_excel(writer, sheet_name="Global_Action", index=False)
    if is_df(exit_list):
        exit_list.to_excel(writer, sheet_name="Exit_Trim", index=False)

    # Summary sheets
    summary_rows = []

    if is_df(norway_master):
        for k, v in norway_master["Final"].value_counts().items():
            summary_rows.append({"Market": "Norway", "Final": k, "Count": int(v)})

    if is_df(global_master):
        for k, v in global_master["Final"].value_counts().items():
            summary_rows.append({"Market": "Global", "Final": k, "Count": int(v)})

    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

print("\n✅ Saved:", filename)

print("\nNORWAY SUMMARY")
if is_df(norway_master):
    print(norway_master["Final"].value_counts())

print("\nGLOBAL SUMMARY")
if is_df(global_master):
    print(global_master["Final"].value_counts())

# ============================================================
# STORE FOR LATER CELLS
# ============================================================

NORWAY_MASTER = norway_master.copy()
SWEDEN_MASTER = sweden_master.copy()
USA_MASTER = usa_master.copy()
GLOBAL_MASTER = global_master.copy()
EXIT_TRIM_LIST = exit_list.copy()

print("\n✅ Stored:")
print("NORWAY_MASTER:", NORWAY_MASTER.shape)
print("SWEDEN_MASTER:", SWEDEN_MASTER.shape)
print("USA_MASTER:", USA_MASTER.shape)
print("GLOBAL_MASTER:", GLOBAL_MASTER.shape)
print("EXIT_TRIM_LIST:", EXIT_TRIM_LIST.shape)
