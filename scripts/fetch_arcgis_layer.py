"""Download every feature of an ArcGIS REST layer (MapServer/FeatureServer) to GeoPackage.

Pages by OBJECTID ranges, which works even on servers without resultOffset support.

Usage:
    python scripts/fetch_arcgis_layer.py <layer_url> <out.gpkg> [--chunk 1000] [--where "1=1"]
"""
import argparse
import json
import sys
import time

import geopandas as gpd
import requests

UA = {"User-Agent": "Mozilla/5.0 (research data download)"}


def get(url, params, retries=5):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=180)
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"  retry {i + 1}: {e}", file=sys.stderr)
            time.sleep(2 * (i + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--where", default="1=1")
    a = ap.parse_args()

    meta = get(a.url, {"f": "json"})
    oid = next(f["name"] for f in meta["fields"] if f["type"] == "esriFieldTypeOID")
    chunk = min(a.chunk, meta.get("maxRecordCount") or a.chunk)
    ids = get(f"{a.url}/query", {"where": a.where, "returnIdsOnly": "true", "f": "json"})["objectIds"] or []
    ids.sort()
    print(f"{meta.get('name')}: {len(ids)} features, oid={oid}, chunk={chunk}")

    frames = []
    for i in range(0, len(ids), chunk):
        lo, hi = ids[i], ids[min(i + chunk, len(ids)) - 1]
        d = get(f"{a.url}/query", {
            "where": f"({a.where}) AND {oid} >= {lo} AND {oid} <= {hi}",
            "outFields": "*", "outSR": 4326, "returnGeometry": "true", "f": "geojson",
        })
        if d.get("features"):
            frames.append(gpd.GeoDataFrame.from_features(d["features"], crs=4326))
        print(f"  {min(i + chunk, len(ids))}/{len(ids)}", end="\r")
    gdf = gpd.pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=4326)
    gdf.to_file(a.out, driver="GPKG")
    with open(a.out + ".meta.json", "w") as f:
        json.dump({"url": a.url, "where": a.where, "n": len(gdf), "downloaded": time.strftime("%Y-%m-%d"),
                   "description": meta.get("description"), "copyright": meta.get("copyrightText")}, f, indent=1)
    print(f"\nwrote {len(gdf)} features -> {a.out}")


if __name__ == "__main__":
    main()
