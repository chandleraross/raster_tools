#clips a raster to a specific boundary
# Chandler Ross

#import the libraries
try:
    from osgeo import gdal
    from osgeo import gdal_array
    from osgeo import gdalconst
except ImportError as e:
    print(e)
    import gdal
    import gdal_array
    import gdalconst
import os, subprocess

# I will try using the command prompt for the clipping
#the command
def clip_raster_command(in_filepath, out_filepath, mask_filepath, mask_val):
    clp_call = 'gdal -clip'
    mask_call = '-mask'
    space = ' '

    # write the terminal command
    command = clp_call + space + mask_filepath + space + mask_call + space + mask_val + space + in_filepath + space + out_filepath
    # call the command in the terminal
    # os.system(command)
    subprocess.call(command)


def clip_raster(input_file, output_file, shapefile):
    outTile = gdal.Warp(output_file, input_file, cutlineDSName=shapefile, cropToCutline=True, dstNodata=0)
    outTile = None
#I can also use the warp function

if __name__ == '__main__':
    in_rast = 'C:/Users/caross2/Downloads/flammap/classified/fire_severity_flammap_1-5.tif'
    out_rast = 'C:/Users/caross2/Downloads/flammap/classified/clipped/fire_severity_flammap_1-5.tif'
    shp = 'C:/Users/caross2/Downloads/clip_data/helping_files/marin_polygon_clip.shp'
    clip_raster(in_rast, out_rast, shp)

