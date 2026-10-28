from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Data ──────────────────────────────────────────────────────────────────────
data_dir = Path(__file__).resolve().parent
df = pd.read_csv(data_dir / "anomaly_detection_dataset.csv")

# Keep only numeric sensor readings; timestamp and anomaly_type are text fields.
features = df.select_dtypes(include="number").drop(columns=["anomaly_label"], errors="ignore")
if features.shape[1] < 2:
    raise ValueError("The dataset must contain at least two numeric sensor columns.")

feature_names = features.columns.tolist()
X = features.to_numpy()

# ── Model ─────────────────────────────────────────────────────────────────────
clf = IsolationForest(contamination=0.2, random_state=42)
clf.fit(X)
predictions = clf.predict(X)          # +1 = normal, -1 = anomaly
scores = clf.decision_function(X)     # more negative → more anomalous

df["if_label"] = predictions
df["if_score"] = scores
anomaly_mask = predictions < 0

print(f"Total points : {len(df)}")
print(f"Anomalies    : {anomaly_mask.sum()}  ({anomaly_mask.mean()*100:.1f} %)")

# ── Plotting ──────────────────────────────────────────────────────────────────
NORMAL_C  = "#4C9BE8"   # blue
ANOMALY_C = "#E84C4C"   # red

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Isolation Forest — Anomaly Detection", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

# ── Panel 1: Anomaly score histogram ─────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.hist(scores[~anomaly_mask], bins=40, color=NORMAL_C,  alpha=0.75, label="Normal")
ax0.hist(scores[ anomaly_mask], bins=40, color=ANOMALY_C, alpha=0.75, label="Anomaly")
ax0.axvline(0, color="k", linestyle="--", linewidth=1, label="Decision boundary")
ax0.set_title("Anomaly Score Distribution")
ax0.set_xlabel("Decision Function Score")
ax0.set_ylabel("Count")
ax0.legend(fontsize=8)

# ── Panel 2: Temperature vs Vibration scatter ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
ax1.scatter(X[~anomaly_mask, 0], X[~anomaly_mask, 1],
            c=NORMAL_C,  s=20, alpha=0.5, label="Normal",  zorder=2)
ax1.scatter(X[ anomaly_mask, 0], X[ anomaly_mask, 1],
            c=ANOMALY_C, s=40, alpha=0.8, label="Anomaly",
            edgecolors="darkred", linewidths=0.5, zorder=3)
ax1.set_title(f"{feature_names[0]}  vs  {feature_names[1]}")
ax1.set_xlabel(feature_names[0])
ax1.set_ylabel(feature_names[1])
ax1.legend(fontsize=8)

# ── Panel 3: Pressure vs Power scatter ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
p_idx = feature_names.index("pressure_bar")    if "pressure_bar"    in feature_names else 2
pw_idx = feature_names.index("power_kw")       if "power_kw"        in feature_names else 4
ax2.scatter(X[~anomaly_mask, p_idx], X[~anomaly_mask, pw_idx],
            c=NORMAL_C,  s=20, alpha=0.5, label="Normal",  zorder=2)
ax2.scatter(X[ anomaly_mask, p_idx], X[ anomaly_mask, pw_idx],
            c=ANOMALY_C, s=40, alpha=0.8, label="Anomaly",
            edgecolors="darkred", linewidths=0.5, zorder=3)
ax2.set_title(f"{feature_names[p_idx]}  vs  {feature_names[pw_idx]}")
ax2.set_xlabel(feature_names[p_idx])
ax2.set_ylabel(feature_names[pw_idx])
ax2.legend(fontsize=8)

# ── Panel 4: Time-series anomaly score ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
idx = np.arange(len(df))
ax3.plot(idx, scores, color="steelblue", linewidth=0.8, alpha=0.7, label="Anomaly score")
ax3.scatter(idx[ anomaly_mask], scores[ anomaly_mask],
            c=ANOMALY_C, s=18, zorder=3, label="Anomaly points")
ax3.axhline(0, color="k", linestyle="--", linewidth=1)
ax3.fill_between(idx, scores, 0,
                 where=(scores < 0), color=ANOMALY_C, alpha=0.15)
ax3.set_title("Anomaly Score Over Time (sample index)")
ax3.set_xlabel("Sample Index")
ax3.set_ylabel("Decision Function Score")
ax3.legend(fontsize=8)

# ── Panel 5: Feature box-plots (normal vs anomaly) ───────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
scaled = StandardScaler().fit_transform(X)
bp_data_normal  = [scaled[~anomaly_mask, i] for i in range(len(feature_names))]
bp_data_anomaly = [scaled[ anomaly_mask, i] for i in range(len(feature_names))]

positions_n = np.arange(len(feature_names)) * 2.5 - 0.5
positions_a = np.arange(len(feature_names)) * 2.5 + 0.5

bp1 = ax4.boxplot(bp_data_normal,  positions=positions_n, widths=0.7,
                  patch_artist=True,
                  boxprops=dict(facecolor=NORMAL_C,  alpha=0.7),
                  medianprops=dict(color="navy"),
                  whiskerprops=dict(color="navy"),
                  capprops=dict(color="navy"),
                  flierprops=dict(marker=".", color=NORMAL_C, alpha=0.3, markersize=3))
bp2 = ax4.boxplot(bp_data_anomaly, positions=positions_a, widths=0.7,
                  patch_artist=True,
                  boxprops=dict(facecolor=ANOMALY_C, alpha=0.7),
                  medianprops=dict(color="darkred"),
                  whiskerprops=dict(color="darkred"),
                  capprops=dict(color="darkred"),
                  flierprops=dict(marker=".", color=ANOMALY_C, alpha=0.3, markersize=3))

short_names = [n.replace("_", "\n") for n in feature_names]
ax4.set_xticks(np.arange(len(feature_names)) * 2.5)
ax4.set_xticklabels(short_names, fontsize=7)
ax4.set_title("Feature Distribution\n(Normalised — Normal vs Anomaly)")
ax4.set_ylabel("Standardised Value")
ax4.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Normal", "Anomaly"], fontsize=8)

# ── Save & show ───────────────────────────────────────────────────────────────
out_path = data_dir / "isolation_forest_anomalies.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved -> {out_path}")
plt.show()
