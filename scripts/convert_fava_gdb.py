"""Convert the FGS FAVA FileGDB rasters (Mendeley doi:10.17632/wybccj4wd7.1) to the GeoTIFFs the Florida pipeline reads.

  data/raw/florida_mendeley_youn2020/gdb/{SAS_DTW,IAS_THK,HEAD_DIFF}.gdb  ->  .../tif/<name>.tif

The rasterio wheel has no OpenFileGDB raster driver, so this uses the GDAL Python bindings from a separate uv venv
(binary wheels, no conda):
  uv venv .venv-gdal --python 3.12
  uv pip install --python .venv-gdal/bin/python --find-links https://girder.github.io/large_image_wheels GDAL
  .venv-gdal/bin/python scripts/convert_fava_gdb.py
"""
import os
import sys

from osgeo import gdal

gdal.UseExceptions()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = f"{ROOT}/data/raw/florida_mendeley_youn2020"
os.makedirs(f"{D}/tif", exist_ok=True)
for name in sys.argv[1:] or ["SAS_DTW", "IAS_THK", "HEAD_DIFF"]:
    src, dst = f"{D}/gdb/{name}.gdb", f"{D}/tif/{name}.tif"
    ds = gdal.Open(src)
    b = ds.GetRasterBand(1)
    print(f"{name}: {ds.RasterXSize}x{ds.RasterYSize}, res {ds.GetGeoTransform()[1]} m, dtype {gdal.GetDataTypeName(b.DataType)}, "
          f"nodata {b.GetNoDataValue()}, crs {ds.GetSpatialRef().GetName()}", flush=True)
    gdal.Translate(dst, ds, format="GTiff", creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"])
    out = gdal.Open(dst)
    print(f"  -> {dst} ({os.path.getsize(dst) / 1e6:.0f} MB), nodata {out.GetRasterBand(1).GetNoDataValue()}", flush=True)
