"""Build a GDAL VRT mosaic from same-CRS, same-resolution GeoTIFF tiles without the GDAL CLI (rasterio only).

Usage: python scripts/build_vrt.py <out.vrt> <tile1.tif> [tile2.tif ...]
       python scripts/build_vrt.py <out.vrt> --glob 'data/raw/usgs_3dep_1arcsec/*.tif'
"""
import argparse
import glob
import os
from xml.sax.saxutils import escape

import numpy as np
import rasterio

GDAL_TYPE = {"uint8": "Byte", "int16": "Int16", "uint16": "UInt16", "int32": "Int32", "uint32": "UInt32",
             "float32": "Float32", "float64": "Float64"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("tiles", nargs="*")
    ap.add_argument("--glob", default=None)
    a = ap.parse_args()
    tiles = sorted(a.tiles + (glob.glob(a.glob) if a.glob else []))
    assert tiles, "no tiles"
    info = []
    for t in tiles:
        with rasterio.open(t) as r:
            info.append(dict(path=t, crs=r.crs, res=r.res, bounds=r.bounds, w=r.width, h=r.height,
                             dtype=r.dtypes[0], nodata=r.nodata, wkt=r.crs.to_wkt()))
    crs, (rx, ry), dtype = info[0]["crs"], info[0]["res"], info[0]["dtype"]
    assert all(i["crs"] == crs for i in info), "mixed CRS"
    assert all(np.allclose(i["res"], (rx, ry), rtol=1e-6) for i in info), "mixed resolution"
    minx = min(i["bounds"].left for i in info)
    maxx = max(i["bounds"].right for i in info)
    miny = min(i["bounds"].bottom for i in info)
    maxy = max(i["bounds"].top for i in info)
    W, H = int(round((maxx - minx) / rx)), int(round((maxy - miny) / ry))
    nod = info[0]["nodata"]
    out_dir = os.path.dirname(os.path.abspath(a.out))
    parts = [f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">',
             f"  <SRS>{escape(info[0]['wkt'])}</SRS>",
             f"  <GeoTransform>{minx!r}, {rx!r}, 0.0, {maxy!r}, 0.0, {-ry!r}</GeoTransform>",
             f'  <VRTRasterBand dataType="{GDAL_TYPE[dtype]}" band="1">']
    if nod is not None:
        parts.append(f"    <NoDataValue>{nod!r}</NoDataValue>")
    for i in info:
        xo = int(round((i["bounds"].left - minx) / rx))
        yo = int(round((maxy - i["bounds"].top) / ry))
        rel = os.path.relpath(os.path.abspath(i["path"]), out_dir)
        parts += [f'    <ComplexSource>',
                  f'      <SourceFilename relativeToVRT="1">{escape(rel)}</SourceFilename>',
                  f"      <SourceBand>1</SourceBand>",
                  f'      <SourceProperties RasterXSize="{i["w"]}" RasterYSize="{i["h"]}" DataType="{GDAL_TYPE[dtype]}"/>',
                  f'      <SrcRect xOff="0" yOff="0" xSize="{i["w"]}" ySize="{i["h"]}"/>',
                  f'      <DstRect xOff="{xo}" yOff="{yo}" xSize="{i["w"]}" ySize="{i["h"]}"/>']
        if nod is not None:
            parts.append(f"      <NODATA>{nod!r}</NODATA>")
        parts.append("    </ComplexSource>")
    parts += ["  </VRTRasterBand>", "</VRTDataset>"]
    with open(a.out, "w") as fh:
        fh.write("\n".join(parts))
    with rasterio.open(a.out) as r:  # validate
        print(f"wrote {a.out}: {len(tiles)} tiles, {r.width}x{r.height}, {r.crs}, res={r.res}")


if __name__ == "__main__":
    main()
