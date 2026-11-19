from pathlib import Path

import matplotlib

# Headless backend: the pipeline also runs inside the web server's worker thread,
# where the default Tk backend crashes the process (Tcl_AsyncDelete).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"

def save_pm25_trend(df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    daily = df.groupby(df["timestamp"].dt.date)["PM2.5"].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    daily.plot(ax=ax)
    ax.set_title("Daily Average PM2.5")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (dataset units; verify instrument units)")
    fig.tight_layout()
    fig.savefig(FIGURES / "daily_pm25_trend.png", dpi=150)
    plt.close(fig)


def save_zone_chart(zones, naaqs=60):
    """Measured mean PM2.5 per SMC zone against the CPCB 24-h limit."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = zones.dropna(subset=["mean_pm25"]).sort_values("mean_pm25")
    colours = ["#94a3b8" if c == "Low" else "#dc2626" if v > naaqs else "#16a34a"
               for c, v in zip(data["coverage"], data["mean_pm25"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(data["zone"], data["mean_pm25"], color=colours)
    for bar, n, high in zip(bars, data["readings"], data["High_hotspots"]):
        label = f"  n={n:,}" + (f", {high} High hotspots" if high else "")
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)
    ax.axvline(naaqs, color="#111827", linestyle="--", linewidth=1)
    ax.text(naaqs, len(data) - 0.4, " CPCB 24-h limit (60)", fontsize=8)
    ax.set_xlabel("Mean measured PM2.5 (µg/m³)")
    ax.set_title("Measured PM2.5 by SMC zone (grey = low coverage, <500 readings)")
    ax.set_xlim(0, data["mean_pm25"].max() * 1.35)
    fig.tight_layout()
    fig.savefig(FIGURES / "zone_pm25.png", dpi=150)
    plt.close(fig)
