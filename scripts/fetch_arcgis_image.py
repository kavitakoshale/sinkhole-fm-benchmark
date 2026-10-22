"""Export an ArcGIS ImageServer raster at native resolution to a single GeoTIFF (tiled requests + mosaic).

Usage:
    python scripts/fetch_arcgis_image.py <imageserver_url> <out.tif> [--tile 4000]
"""
import argparse
import io
import math
import sys
import time

import numpy as np
import rasterio
import requests
from rasterio.merge import merge

UA = {"User-Agent": "Mozilla/5.0 (research data download)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--tile", type=int, default=4000)
    a = ap.parse_args()

    meta = requests.get(a.url, params={"f": "json"}, headers=UA, timeout=120).json()
    ext, px, py = meta["extent"], meta["pixelSizeX"], meta["pixelSizeY"]
    wkid = ext["spatialReference"].get("latestWkid") or ext["spatialReference"]["wkid"]
    ncol = round((ext["xmax"] - ext["xmin"]) / px)
    nrow = round((ext["ymax"] - ext["ymin"]) / py)
    t = min(a.tile, meta.get("maxImageWidth", a.tile), meta.get("maxImageHeight", a.tile))
    print(f"{meta.get('name')}: {ncol}x{nrow} px @ {px} m, EPSG:{wkid}, tile={t}")

    srcs = []
    for i in range(math.ceil(nrow / t)):
        for j in range(math.ceil(ncol / t)):
            w, h = min(t, ncol - j * t), min(t, nrow - i * t)
            x0 = ext["xmin"] + j * t * px
            y1 = ext["ymax"] - i * t * py
            bbox = f"{x0},{y1 - h * py},{x0 + w * px},{y1}"
            for k in range(5):
                try:
                    r = requests.get(f"{a.url}/exportImage", headers=UA, timeout=600, params={
                        "bbox": bbox, "bboxSR": wkid, "imageSR": wkid, "size": f"{w},{h}",
                        "format": "tiff", "interpolation": "RSP_NearestNeighbor", "f": "image"})
                    r.raise_for_status()
                    srcs.append(rasterio.open(io.BytesIO(r.content)))
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  retry {k + 1}: {e}", file=sys.stderr)
                    time.sleep(3 * (k + 1))
            print(f"  tile {i},{j} ok", end="\r")
    mosaic, transform = merge(srcs)
    prof = srcs[0].profile
    prof.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2], transform=transform,
                compress="deflate", tiled=True, crs=f"EPSG:{wkid}")
    with rasterio.open(a.out, "w", **prof) as dst:
        dst.write(mosaic)
    vals, cnt = np.unique(mosaic, return_counts=True)
    print(f"\nwrote {a.out} {mosaic.shape}; values: {dict(zip(vals.tolist()[:20], cnt.tolist()[:20]))}")


if __name__ == "__main__":
    main()
