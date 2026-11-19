"""Planning context for hotspots: SMC zone, population density, nearest official
monitoring station, persistence and rule-based mitigation options.

Reference data (data/reference/):
  zones.csv    - SMC zone area and Census-2011 population; centroids are approximate.
  stations.csv - SMC CAAQMS / GPCB monitoring-station coordinates.
Zone assignment uses the nearest approximate centroid, so it is a screening label,
not an exact ward/zone boundary lookup.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference"
NAAQS_PM25 = 60          # CPCB 24-h standard, ug/m3
MIN_ZONE_READINGS = 500  # below this a zone mean is flagged as low coverage
# Weights from Harsh Vardhan's zone IPS model (severity 0.35, exposure 0.30, traffic 0.20,
# vulnerability 0.15); only the two factors with real data are used here.
ZONE_WEIGHTS = {"severity": 0.35, "exposure": 0.30}
ZONE_RULES = {
    "PM2.5 severity": ["Source audit (industry / construction / waste burning)",
                       "Construction-dust enforcement (barricades, covers)",
                       "Continuous monitoring + alert system"],
    "population exposure": ["Green buffers / tree belts along dense residential edges",
                            "Public-transport priority corridors",
                            "Install additional low-cost sensors to confirm hotspots"],
}
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 103.6   # at Surat's latitude (~21.2 N)


def _distance_km(lat, lon, ref_lat, ref_lon):
    """Equirectangular distance matrix (rows: points, columns: reference points)."""
    dlat = (np.asarray(lat)[:, None] - np.asarray(ref_lat)[None, :]) * KM_PER_DEG_LAT
    dlon = (np.asarray(lon)[:, None] - np.asarray(ref_lon)[None, :]) * KM_PER_DEG_LON
    return np.hypot(dlat, dlon)


def load_zones() -> pd.DataFrame:
    zones = pd.read_csv(REFERENCE / "zones.csv")
    zones["density_per_km2"] = (zones["pop_2011"] / zones["area_km2"]).round()
    return zones


def load_stations() -> pd.DataFrame:
    return pd.read_csv(REFERENCE / "stations.csv")


def nearest_zone(lat, lon, zones: pd.DataFrame) -> pd.Series:
    return pd.Series(zones["zone"].values[_distance_km(lat, lon, zones.lat, zones.lon).argmin(1)])


def mitigation(row) -> str:
    level = row["priority"]
    if level == "Low":
        return "Routine monitoring"
    options = []
    if row["x_naaqs"] >= 3:
        options.append("Source audit (traffic / construction / waste burning)")
    options.append("Road-dust control: mechanised sweeping + water sprinkling")
    if level == "High":
        options.append("Junction signal optimisation + anti-idling enforcement")
        options.append("Construction-dust enforcement (covers, barricades, wheel wash)")
    if row["days_observed"] >= 2:
        options.append("Recurring hotspot: green buffer / tree belt on corridor")
    if row["zone_density_per_km2"] >= 30000:
        options.append("Dense residential zone: public-transport priority")
    if row["station_km"] > 2:
        options.append("No official station within 2 km: add low-cost sensor")
    return "; ".join(options)


def add_context(priorities: pd.DataFrame) -> pd.DataFrame:
    if priorities.empty:
        return priorities
    zones, stations = load_zones(), load_stations()
    out = priorities.copy()
    lat, lon = out["latitude_group"], out["longitude_group"]

    out["x_naaqs"] = (out["average_pm25"] / NAAQS_PM25).round(2)
    out["zone"] = nearest_zone(lat, lon, zones).values
    out = out.merge(zones[["zone", "density_per_km2"]].rename(
        columns={"density_per_km2": "zone_density_per_km2"}), on="zone", how="left")

    dist = _distance_km(lat, lon, stations.lat, stations.lon)
    out["nearest_station"] = stations["station"].values[dist.argmin(1)]
    out["station_km"] = dist.min(1).round(2)

    out["mitigation"] = out.apply(mitigation, axis=1)
    return out


def zone_summary(df: pd.DataFrame, priorities: pd.DataFrame) -> pd.DataFrame:
    """Measured PM2.5 per SMC zone, joined with population density and hotspot counts."""
    zones = load_zones()
    geo = df.dropna(subset=["latitude", "longitude", "PM2.5"]).copy()
    geo["zone"] = nearest_zone(geo["latitude"], geo["longitude"], zones).values
    summary = geo.groupby("zone").agg(
        readings=("PM2.5", "size"),
        mean_pm25=("PM2.5", "mean"),
        median_pm25=("PM2.5", "median"),
        pct_above_naaqs=("PM2.5", lambda s: (s > NAAQS_PM25).mean() * 100),
    )
    counts = priorities.pivot_table(index="zone", columns="priority", values="ips",
                                    aggfunc="size", fill_value=0)
    summary = summary.join(counts.reindex(columns=["High", "Medium", "Low"], fill_value=0)
                           .add_suffix("_hotspots"), how="left").fillna(0)
    summary = zones.set_index("zone")[["lat", "lon", "area_km2", "pop_2011", "density_per_km2"]].join(summary, how="left")
    summary["readings"] = summary["readings"].fillna(0).astype(int)
    summary["coverage"] = np.where(summary["readings"] >= MIN_ZONE_READINGS, "Adequate", "Low")
    for col in ["High_hotspots", "Medium_hotspots", "Low_hotspots"]:
        summary[col] = summary[col].fillna(0).astype(int)
    summary = add_zone_ips(summary)
    return summary.round(2).sort_values(["zone_ips", "mean_pm25"], ascending=False).reset_index()


def add_zone_ips(summary: pd.DataFrame) -> pd.DataFrame:
    """Zone-level IPS from the civil-engineering zone model (severity 0.35, exposure 0.30),
    re-scaled to sum to 1. Traffic and receptor terms are dropped because no measured
    traffic counts or verified receptor data are available. Only zones with adequate
    measurement coverage are scored; the others are left blank rather than guessed."""
    s = summary.copy()
    ok = s["coverage"] == "Adequate"
    w_sev, w_exp = ZONE_WEIGHTS["severity"], ZONE_WEIGHTS["exposure"]
    total = w_sev + w_exp

    def minmax(col):
        v = s.loc[ok, col]
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng else v * 0.0

    s["f_severity"] = np.nan
    s["f_exposure"] = np.nan
    s.loc[ok, "f_severity"] = minmax("mean_pm25")
    s.loc[ok, "f_exposure"] = minmax("density_per_km2")
    s["zone_ips"] = (100 * (w_sev * s["f_severity"] + w_exp * s["f_exposure"]) / total).round(1)
    s["zone_tier"] = np.select(
        [~ok, s["zone_ips"] >= 60, s["zone_ips"] >= 25],
        ["Not scored (low coverage)", "Tier 1 - High", "Tier 2 - Medium"],
        default="Tier 3 - Low",
    )
    s["main_driver"] = np.where(
        ~ok, "", np.where(s["f_severity"] >= s["f_exposure"], "PM2.5 severity", "population exposure"))
    s["zone_actions"] = [
        "; ".join(ZONE_RULES.get(driver, [])) if tier.startswith(("Tier 1", "Tier 2"))
        else ("Continuous monitoring is sufficient" if tier.startswith("Tier 3")
              else "Collect more readings before ranking this zone")
        for driver, tier in zip(s["main_driver"], s["zone_tier"])
    ]
    return s
