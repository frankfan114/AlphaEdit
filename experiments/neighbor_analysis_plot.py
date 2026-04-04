import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "/rds/general/user/ff422/home/FYP/AlphaEdit/results/neighbor_prompt_analysis_maxspan/per_pair.csv"
OUT = "/rds/general/user/ff422/home/FYP/AlphaEdit/results/neighbor_prompt_analysis_maxspan/plots_A"
TAU = 0.10  # path existence threshold

os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(CSV)

# basic flags
df["has_path_base"] = df["peakR_base"] > TAU
df["has_path_edited"] = df["peakR_edited"] > TAU
df["leakage_strict"] = df["has_path_edited"] & (~df["has_path_base"])

def savefig(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, bbox_inches="tight", dpi=220)
    plt.close()
    print("wrote", path)

def ecdf(x):
    x = np.array([v for v in x if np.isfinite(v)], dtype=float)
    x.sort()
    if len(x) == 0:
        return x, x
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

# -------------------------
# A2-1: leakage rate by kind (neighbor-level)
# -------------------------
df_new = df[df["expect_field"] == "target_new"].copy()

rates = []
kinds = ["none", "mlp", "attn"]
for k in kinds:
    sub = df_new[df_new["kind"] == k]
    if len(sub) == 0:
        rates.append(np.nan)
    else:
        rates.append(sub["leakage_strict"].mean())

plt.figure()
plt.bar(kinds, rates)
plt.ylim(0, 1)
plt.ylabel(f"Leakage rate (peakR_edited>{TAU} & not base)")
plt.title("A2: target_new leakage rate on neighbor prompts")
savefig("A2_leakage_rate_by_kind.png")

# -------------------------
# A2-2: leakage intensity ECDF (delta_peakR) by kind
# -------------------------
plt.figure()
for k in kinds:
    sub = df_new[df_new["kind"] == k]
    x, y = ecdf(sub["delta_peakR"].values)
    plt.plot(x, y, label=k)
plt.xlabel("delta_peakR (edited - base)")
plt.ylabel("ECDF")
plt.title("A2: leakage intensity distribution (neighbor-level)")
plt.legend()
savefig("A2_ecdf_delta_peakR_by_kind.png")

# -------------------------
# A2-3: peak layer distribution for leaked neighbors (edited)
# -------------------------
df_new_leak = df_new[df_new["leakage_strict"]].copy()
plt.figure()
bins = np.arange(df["peak_layer_edited"].min(), df["peak_layer_edited"].max() + 2) - 0.5
plt.hist(df_new_leak["peak_layer_edited"].dropna().values, bins=bins)
plt.xlabel("peak_layer_edited")
plt.ylabel("count")
plt.title("A2: peak layer location for leaked neighbors (edited)")
savefig("A2_hist_peak_layer_edited_leak.png")

# -------------------------
# A1-1: delta_peakR boxplot by kind (target_true)
# -------------------------
df_true = df[df["expect_field"] == "target_true"].copy()
data = [df_true[df_true["kind"] == k]["delta_peakR"].dropna().values for k in kinds]

plt.figure()
plt.boxplot(data, labels=kinds, showfliers=False)
plt.ylabel("delta_peakR (edited - base)")
plt.title("A1: old knowledge change on neighbor prompts (target_true)")
savefig("A1_box_delta_peakR_by_kind.png")

# -------------------------
# A1-2: cosine_R boxplot by kind (target_true)
# -------------------------
data = [df_true[df_true["kind"] == k]["cosine_R"].dropna().values for k in kinds]
plt.figure()
plt.boxplot(data, labels=kinds, showfliers=False)
plt.ylabel("cosine similarity of R_layer (base vs edited)")
plt.title("A1: path similarity (target_true)")
savefig("A1_box_cosineR_by_kind.png")

# -------------------------
# A1-3: scatter cosine_R vs delta_peakR (target_true)
# -------------------------
for k in kinds:
    sub = df_true[df_true["kind"] == k].dropna(subset=["cosine_R", "delta_peakR"])
    plt.figure()
    plt.scatter(sub["cosine_R"].values, sub["delta_peakR"].values, s=8)
    plt.xlabel("cosine_R")
    plt.ylabel("delta_peakR")
    plt.title(f"A1: structure vs strength (kind={k})")
    savefig(f"A1_scatter_cosine_vs_delta_peakR_{k}.png")
