
#libraries
try:
    from osgeo import gdal
    from osgeo import gdalconst
except ImportError as e:
    print(e)
    import gdal
    import gdalconst
import numpy as np


#=======================================================================================================================

def class_counter(in_rast, driver_name='GTiff'):
    """
    Takes a raster and outputs a dictionary of the number of pixels in each class
    :param in_rast: filepath to the raster; string
    :param in_rast: name of the gdal driver; string
    :return: dictionary of the number of rasters in each class
    """
    # create the driver
    driver = gdal.GetDriverByName(driver_name)
    driver.Register()

    # open the dataset in read only mode
    dataset = gdal.Open(in_rast, gdalconst.GA_ReadOnly)

    # read some of the attributes of the data
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    # get the first band
    band = dataset.GetRasterBand(1)

    # read the data (all at the same time) this is the numpy array
    data = band.ReadAsArray(0, 0, cols, rows)

    # determine the classes in the raster
    unique_array = np.unique(data)

    #create the empty dictionary
    out_dict = {}

    #iterate through the unique array to determine how many pixels are in each class
    for pix_class in unique_array:

        #count the number of pixels
        class_num = np.count_nonzero(data == pix_class)

        #add the value to the dictionary
        out_dict[pix_class] = class_num

    #print the output dictionary
    print(out_dict)

    #close the raster
    data = None

    #output the dictionary
    return out_dict

#=======================================================================================================================

if __name__ == '__main__':
    in_rast = r'C:\Users\caross2\Downloads\flammap\clipped\marin_flammap_output_clp.tif'
    class_counter(in_rast)

