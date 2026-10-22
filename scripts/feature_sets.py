"""Feature-source ablation on the Florida headline task (ring negatives; spatial5 + temporal), grouped by where a
feature can come from. Answers "what can we predict from remote sensing alone?" and "do fewer features help the FMs?".

  terrain          21 DEM derivatives (elevation, slope, curvature, TPI / relief / roughness at 150-600 m)
  sat_nolc         terrain + closed-depression metrics (DEM-derivable) + annual rainfall      -> remote-sensing, no land cover
  sat_all          sat_nolc + WorldCover land-cover fractions / class                          -> everything satellite-derived
  maps             only mapped layers: FGS geology, FAVA hydrogeology, DRASTIC, recharge, soils, distances to springs/swallets/mines
  subedi5          Subedi 2019 style 5 factors: depression distance, surficial geology, soil permeability, spring distance, mine distance
  no_landcover     committed headline set (41 num + 8 cat)
  all              committed full set (59 num + 9 cat)

Usage: python scripts/feature_sets.py [--sets ...] [--models ...]   -> results/fm_scenarios/feature_sets/
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import sys  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from fm_scenarios import OUT, all_runs, run_fit, write_summary  # noqa: E402
from run_florida_baselines import CAT, DATA, feature_cols, splits  # noqa: E402

TERRAIN = ["elev_m", "slope_deg", "laplacian_curv"] + [f"{s}_{r}m" for r in (150, 300, 600) for s in
           ("tpi", "relief", "elev_std", "depth_below_max", "height_above_min", "slope_mean")]
CTD = ["ctd_inside", "ctd_dist_m"] + [f"ctd_{k}_{r}m" for r in (500, 1000, 2000) for k in ("count", "area_frac")]
LC_NUM = [f"lc_{c}_frac_{r}m" for r in (250, 1000) for c in
          ("tree", "shrub", "grass", "crop", "built", "bare", "water", "wetland", "mangrove")]
MAPS_NUM = ["sas_depth_to_water_ft", "ias_thickness_ft", "head_diff_ft", "drastic_floridan_index", "dist_swallet_m",
            "dist_spring_m", "dist_phosphate_mine_m", "soil_ksat_0_5", "soil_ksat_30_60", "soil_clay_0_5", "soil_sand_0_5"]
MAPS_CAT = [c for c in CAT if c != "lc_class"]


def feature_sets(d):
    num_all, cat_all = feature_cols(d, feature_set="all")
    num_nl, cat_nl = feature_cols(d, feature_set="no_landcover")
    return {
        "terrain": (TERRAIN, []),
        "sat_nolc": (TERRAIN + CTD + ["ppt_annual_mm"], []),
        "sat_all": (TERRAIN + CTD + ["ppt_annual_mm"] + LC_NUM, ["lc_class"]),
        "maps": (MAPS_NUM, MAPS_CAT),
        "subedi5": (["ctd_dist_m", "soil_ksat_0_5", "dist_spring_m", "dist_phosphate_mine_m"], ["geo_unit"]),
        "no_landcover": (num_nl, cat_nl),
        "all": (num_all, cat_all),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=["logreg", "random_forest", "xgboost", "tabpfn", "kumo_tabular"])
    ap.add_argument("--protocols", nargs="*", default=["spatial5", "temporal"])
    a = ap.parse_args()
    out = f"{OUT}/feature_sets"
    os.makedirs(out, exist_ok=True)
    data = pd.read_parquet(DATA)
    sets = feature_sets(data)
    d = data[(data.label == 1) | (data.neg_strategy == "ring")].reset_index(drop=True)
    y = d.label.to_numpy()
    rows = []
    for name in a.sets or list(sets):
        num, cat = sets[name]
        assert all(c in d for c in num + cat), name
        for prot in a.protocols:
            for f, tr, te in splits(d, prot):
                for m in a.models:
                    run_fit(rows, {"task": "florida", "protocol": prot, "feature_set": name, "n_features": len(num) + len(cat),
                                   "fold": f"{prot}_f{f}", "rep": 0}, m, num, cat, 0,
                            d.loc[tr, num + cat], y[tr.to_numpy()], d.loc[te, num + cat], y[te.to_numpy()])
            pd.DataFrame(rows).to_csv(f"{out}/runs_{'_'.join(a.sets or ['all_sets'])}.csv", index=False)
    r = all_runs(out)
    write_summary(r, ["protocol", "feature_set", "n_features"], out, "Feature-source ablation (Florida, ring negatives)",
                  "terrain = DEM only; sat_nolc = DEM + depressions + rainfall; sat_all = + land cover; maps = geology / "
                  "hydrogeology / soils / distance layers only; subedi5 = Subedi 2019 style 5 factors. Mean fold AUC; "
                  "spatial5 over 5 folds, temporal is one split.")


if __name__ == "__main__":
    main()
