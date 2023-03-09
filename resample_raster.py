#Script to resample the raster to a specific cell size fom another raster
# Author: Chandler Ross

#librarys
try:
    from osgeo import gdal
    from osgeo import gdal_array
    from osgeo import gdalconst
except ImportError as e:
    print(e)
    import gdal
    import gdal_array
    import gdalconst


def resample(ref_file, in_raster, out_raster):
    reference = gdal.Open(ref_file, 0)  # this opens the file in only reading mode
    referenceTrans = reference.GetGeoTransform() #spatial information
    x_res = referenceTrans[1]
    y_res = -referenceTrans[5]  # make sure this value is positive

    # call gdal Warp
    kwargs = {"format": "GTiff", "xRes": x_res, "yRes": y_res, "targetAlignedPixels": True}
    ds = gdal.Warp(out_raster, in_raster, **kwargs)
    ds = None   #close the file

