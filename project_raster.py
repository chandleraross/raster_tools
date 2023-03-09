#Script to project the raster
# Author: Chandler Ross

#Import the library
try:
    from osgeo import gdal
    from osgeo import gdal_array
    from osgeo import gdalconst
except ImportError as e:
    print(e)
    import gdal
    import gdal_array
    import gdalconst


#project the raster
def project_raster(input_raster, output_raster, prj='EPSG:26943'):
    """
    projects the input and output rasters
    :param output_raster: filepath to output raster; string
    :param input_raster: filepath to input raster; string
    :param prj:string of the EPSG code in the format 'EPSG:#'. Defaults to CA state plane 3 meters; string
    :return:
    """
    raster_file = gdal.Open(input_raster)
    warp = gdal.Warp(destNameOrDestDS=output_raster,srcDSOrSrcDSTab=raster_file, dstSRS=prj)
    #try calling the prj straight to the terminal
    # os.system('gdalwarp ')
    warp = None  # Closes the files

if __name__ == '__main__':
    in_rast = 'C:/Users/caross2/Downloads/lidar_dem/test/hidden/DW_2020.tif'
    out_rast = 'C:/Users/caross2/Downloads/lidar_dem/test/hidden/DW_2020_test.tif'
    project_raster(in_rast, out_rast)


