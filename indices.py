"""
Indice Library
This script is to seve as the library for computing indices for the 582 Final Project
2021-11-19
Chandler Ross
"""

"""
List of the indices that I will need to compute
BAI
1 / ((0.1 - red)**2 + (0.06 - nir)**2)
CSI
nir/swir2
EVI
2.5 * (nir - red) / (nir + (6.0 * red) - (7.5 * blue) + 1.0)
GEMI
n * (1.0 - 0.25 * n) - (red - 0.125) / (1 - red)
n = (2 * (nir**2 - red**2) + (1.5 * nir) + (0.5 * red))/ (nir + red + 0.5)
MIRBI
(10.0 * swir2) - 9.8 * swir1) + 2.0
NBR
(nir - swir2) / (nir + swir2)
NBR2
(swir1 - swir2) / (swir1 + swir2)
NBRT1
(nir - (swir2 * thermal)) / (nir + (swir2 * thermal))
NDMI
(nir - swir1) / (nir + swir1)
NDVI
(nir - red) / (nir + red)
NDWI
(green - nir) / (green + nir)
SAVI
1.5 * (nir - red) / (nir + red + 0.5)
VI6T
(nir - thermal) / (nir + thermal)
VI43
nir / red
VI45
nir / swir1
VI46
nir / thermal
VI57
swir1 / swir2
"""




import os, sys, rasterio, random
import numpy as np
from matplotlib import pyplot as plt
try:
    from osgeo import gdal
    from osgeo import gdalconst
except ImportError as e:
    print(e)
    import gdal
    import gdalconst


driver_name = 'GTiff'


#=======================================================================================================================
#                                Private/Public Functions to read in the data from Quiz 3
#=======================================================================================================================

#This following function was taken from Quiz 3
def read_data(fp_in_img, raster_driver_name='GTiff'):
    """
    register GDAL Driver and read input image file
    :param fp_in_img:
    :param raster_driver_name:
    :return: gdal.dataset
    """

    if raster_driver_name is None:
        gdal.AllRegister()
    else:
        driver = gdal.GetDriverByName(raster_driver_name)
        driver.Register()

    dataset = gdal.Open(fp_in_img, gdalconst.GA_ReadOnly)

    if dataset is None:
        print("Error: Could not read '{}'".format(fp_in_img))
        sys.exit()

    return dataset

#This following function was taken from Quiz 3
def get_info(fp_in_img, fp_out_info, flag_print=False, raster_driver_name='GTiff'):
    """
    get input raster data information
    :param fp_in_img: input raster image file path (string)
    :param fp_out_info: ouput file path (string)
    :param flag_print: flag to print raster information (boolean)
    :param raster_driver_name: input raster driver name (string)
    :return: a dictionary containing raster information (dictionary)
    """
    path, filename = os.path.split(fp_in_img)

    dataset = __read_data(fp_in_img, raster_driver_name)

    str_out = ""
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()

    raster_info = {
        'file_path': fp_in_img,
        'file_name': filename,
        'cols': cols,
        'rows': rows,
        'bands': bands,
        'projection': projection,
        'metadata': metadata
    }

    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols,rows,bands,projection)

    geotransform = dataset.GetGeoTransform()
    if geotransform:
        origin_x = geotransform[0]
        origin_y = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]

        str_out += "Geotransform:\n\torigin:({}, {})\n\tpixel width: {}\n\tpixel height: {}\n".format(origin_x,origin_y,geotransform[1],geotransform[5])

    else:
        str_out += "Geotransform: {}".format(geotransform)

    raster_info['geotransform'] = geotransform

    with open(fp_out_info, 'w') as out:
        out.write(str_out)

    # reading raster data for each band
    str_out += "\nRaster Band Information:\n"

    band_info = {}

    # iterate each band
    for i in range(bands):
        band = dataset.GetRasterBand(i+1)

        # cf) band data type & no data value
        band_type = gdal.GetDataTypeName(band.DataType)
        nodataval = band.GetNoDataValue()

        min = band.GetMinimum()
        max = band.GetMaximum()

        if min is None or max is None:
            min, max = band.ComputeRasterMinMax()

        str_out += "Band {}:\n\tBand DataType: {}\n\tMin Pixel Value: {}\n\tMax Pixel Value: {}\n".format(i+1, band_type, min, max)

        band_info[i+1] = {'band': i+1, 'band_type': band_type, 'nodata_val': nodataval, 'min': min, 'max':max}

        band = None
        data = None

    raster_info['band_info'] = band_info

    dataset = None

    if flag_print:
        print(str_out)

    with open(fp_out_info, 'w') as out:
        out.write(str_out)

    return raster_info

#This following function was taken from Quiz 3
def output_single_band_raster(data, out_fp, col_size, row_size, num_band=1, raster_driver_name='GTiff',
                              projection=None, geotransform=None, metadata=None, nodataval=None):
    """
    output single raster band to image file
    :param data: numpy array data (numpy array)
    :param out_fp: output file path (string)
    :param col_size: column size (integer)
    :param row_size: row size (integer)
    :param num_band: number of band (integer)
    :param raster_driver_name: raster driver name (string)
    :param nodataval: nodata value
    :param projection: projection
    :param geotransform: geotransform
    :param metadata: metadata
    :return: None
    """
    #set the data type
    data_type = gdal.GDT_Float32

    #Register the driver
    if raster_driver_name is None:
        gdal.AllRegister()
    else:
        driver = gdal.GetDriverByName(raster_driver_name)
        driver.Register()

    #output the dataset information
    dataset_output = driver.Create(out_fp, col_size, row_size, num_band, data_type)
    out_band = dataset_output.GetRasterBand(1)

    #write it
    out_band.WriteArray(data, 0, 0)

    #set information for the raster
    if nodataval:
        out_band.SetNoDataValue(nodataval)
    dataset_output.SetProjection(projection)
    dataset_output.SetGeoTransform(geotransform)
    dataset_output.SetMetadata(metadata)

    #for memory puropses
    data = None


#=======================================================================================================================
#                                Spectral Indices Functions
#=======================================================================================================================
'''
For my data for Landsat 5 & 7
b1 - QA_pixel
b2 - b1 blue
b3 - b2 green
b4 - b3 red
b5 - b4 nir
b6 - b5 swir1
b7 - b7 swir2
b8 - cloud_qa
b9 - b6 thermal
'''

def get_bai(fp_in_img, red_idx=4, nir_idx=5, raster_driver_name='GTiff'):
    """
       calcs the SVI and outputs an SVI raster in the temp folder
       :param fp_in_img: file path
       :param red_idx: index for red band (ex: 4)
       :param nir_idx: index for nir band (ex: 5)
       :param raster_driver_name: driver name
       :return:
       """
    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band4 = dataset.GetRasterBand(red_idx)
    band5 = dataset.GetRasterBand(nir_idx)

    # get the no data value
    nodataval_red = band4.GetNoDataValue()
    nodataval_nir = band5.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    red = band4.ReadAsArray(0, 0, cols, rows).astype('float64')
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    red[red == nodataval_red] = np.nan
    nir[nir == nodataval_nir] = np.nan

    #calc SVI
    bai = 1 / ((0.1 - red)**2 + (0.06 - nir)**2)

    #make ndvi band into a new image
    svi_temp_path = './temp/bai.tif'
    output_single_band_raster(data=bai, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_red)

def get_csi(fp_in_img,  nir_idx=5, swir2_idx=7, raster_driver_name='GTiff'):

    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band5 = dataset.GetRasterBand(nir_idx)
    band7 = dataset.GetRasterBand(swir2_idx)

    # get the no data value
    nodataval_swir2 = band7.GetNoDataValue()
    nodataval_nir = band5.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    swir2 = band7.ReadAsArray(0, 0, cols, rows).astype('float64')
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    swir2[swir2 == nodataval_swir2] = np.nan
    nir[nir == nodataval_nir] = np.nan

    #calc SVI
    csi = nir/swir2

    #make ndvi band into a new image
    svi_temp_path = './temp/csi.tif'
    output_single_band_raster(data=csi, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_nir)

def get_evi(fp_in_img, blue_idx=2,  nir_idx=5, red_idx=4, raster_driver_name='GTiff'):

    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band2 = dataset.GetRasterBand(blue_idx)
    band5 = dataset.GetRasterBand(nir_idx)
    band4 = dataset.GetRasterBand(red_idx)

    # get the no data value
    nodataval_red = band4.GetNoDataValue()
    nodataval_nir = band5.GetNoDataValue()
    nodataval_blue = band2.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    blue = band2.ReadAsArray(0, 0, cols, rows).astype('float64')
    red = band4.ReadAsArray(0, 0, cols, rows).astype('float64')
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    red[red == nodataval_red] = np.nan
    nir[nir == nodataval_nir] = np.nan
    blue[blue == nodataval_blue] = np.nan

    #calc SVI
    evi = 2.5 * (nir - red) / (nir + (6.0 * red) - (7.5 * blue) + 1.0)

    #make ndvi band into a new image
    svi_temp_path = './temp/evi.tif'
    output_single_band_raster(data=evi, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_blue)

def get_gemi(fp_in_img, nir_idx=5, red_idx=4, raster_driver_name='GTiff'):

    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band5 = dataset.GetRasterBand(nir_idx)
    band4 = dataset.GetRasterBand(red_idx)

    # get the no data value
    nodataval_red = band4.GetNoDataValue()
    nodataval_nir = band5.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    red = band4.ReadAsArray(0, 0, cols, rows).astype('float64')
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    red[red == nodataval_red] = np.nan
    nir[nir == nodataval_nir] = np.nan

    #calc SVI
    n = (2 * (nir ** 2 - red ** 2) + (1.5 * nir) + (0.5 * red)) / (nir + red + 0.5)
    gemi = n * (1.0 - 0.25 * n) - (red - 0.125) / (1 - red)


    #make ndvi band into a new image
    svi_temp_path = './temp/gemi.tif'
    output_single_band_raster(data=gemi, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_red)

def get_miribi(fp_in_img,  swir1_idx=6, swir2_idx=7, raster_driver_name='GTiff'):

    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band6 = dataset.GetRasterBand(swir1_idx)
    band7 = dataset.GetRasterBand(swir2_idx)

    # get the no data value
    nodataval_swir1 = band6.GetNoDataValue()
    nodataval_swir2 = band7.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    swir1 = band6.ReadAsArray(0, 0, cols, rows).astype('float64')
    swir2 = band7.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    swir1[swir1 == nodataval_swir1] = np.nan
    swir2[swir2 == nodataval_swir2] = np.nan

    #calc SVI
    miribi = (10.0 * swir2) - (9.8 * swir1) + 2.0

    #make ndvi band into a new image
    svi_temp_path = './temp/miribi.tif'
    output_single_band_raster(data=miribi, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_swir1)

def get_nbr(fp_in_img,  nir_idx=5, swir2_idx=7, raster_driver_name='GTiff'):

    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band5 = dataset.GetRasterBand(nir_idx)
    band7 = dataset.GetRasterBand(swir2_idx)

    # get the no data value
    nodataval_nir = band5.GetNoDataValue()
    nodataval_swir2 = band7.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')
    swir2 = band7.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    nir[nir == nodataval_nir] = np.nan
    swir2[swir2 == nodataval_swir2] = np.nan

    #calc SVI
    nbr = (nir - swir2) / (nir + swir2)

    #make ndvi band into a new image
    svi_temp_path = './temp/nbr.tif'
    output_single_band_raster(data=nbr, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_nir)

def get_nbr2(fp_in_img,  swir1_idx=6, swir2_idx=7, raster_driver_name='GTiff'):

    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band6 = dataset.GetRasterBand(swir1_idx)
    band7 = dataset.GetRasterBand(swir2_idx)

    # get the no data value
    nodataval_swir1 = band6.GetNoDataValue()
    nodataval_swir2 = band7.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    swir1 = band6.ReadAsArray(0, 0, cols, rows).astype('float64')
    swir2 = band7.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    swir1[swir1 == nodataval_swir1] = np.nan
    swir2[swir2 == nodataval_swir2] = np.nan

    #calc SVI
    nbr2 = (swir1 - swir2) / (swir1 + swir2)

    #make ndvi band into a new image
    svi_temp_path = './temp/nbr2.tif'
    output_single_band_raster(data=nbr2, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_swir1)

def get_nbrt1(fp_in_img,  nir_idx=5, swir2_idx=7, therm_idx=9, raster_driver_name='GTiff'):

    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band5 = dataset.GetRasterBand(nir_idx)
    band7 = dataset.GetRasterBand(swir2_idx)
    band9 = dataset.GetRasterBand(therm_idx)

    # get the no data value
    nodataval_nir = band5.GetNoDataValue()
    nodataval_swir2 = band7.GetNoDataValue()
    nodataval_therm = band9.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')
    swir2 = band7.ReadAsArray(0, 0, cols, rows).astype('float64')
    thermal = band9.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    nir[nir == nodataval_nir] = np.nan
    swir2[swir2 == nodataval_swir2] = np.nan
    thermal[thermal == nodataval_therm] = np.nan

    #calc SVI
    nbrt1 = (nir - (swir2 * thermal)) / (nir + (swir2 * thermal))

    #make ndvi band into a new image
    svi_temp_path = './temp/nbrt1.tif'
    output_single_band_raster(data=nbrt1, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_nir)

def get_ndmi(fp_in_img,  nir_idx=5, swir1_idx=6, raster_driver_name='GTiff'):

    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band5 = dataset.GetRasterBand(nir_idx)
    band6 = dataset.GetRasterBand(swir1_idx)

    # get the no data value
    nodataval_nir = band5.GetNoDataValue()
    nodataval_swir1 = band6.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')
    swir1 = band6.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    nir[nir == nodataval_nir] = np.nan
    swir1[swir1 == nodataval_swir1] = np.nan

    #calc SVI
    ndmi = (nir - swir1) / (nir + swir1)

    #make ndvi band into a new image
    svi_temp_path = './temp/ndmi.tif'
    output_single_band_raster(data=ndmi, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_nir)

#This following function was modified from Quiz 3 & my extra credit submission
def get_ndvi(fp_in_img, red_idx=4, nir_idx=5, raster_driver_name='GTiff'):
    """
    calcs NDVI and outputs an ndvi raster in the temp folder
    :param fp_in_img: file path
    :param red_idx: index for red band (ex: 4)
    :param nir_idx: index for nir band (ex: 5)
    :param raster_driver_name: driver name
    :return:
    """
    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    #get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    #info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols,rows,bands,projection)

    #get the band
    band4 = dataset.GetRasterBand(red_idx)
    band5 = dataset.GetRasterBand(nir_idx)

    # get the no data value
    nodataval_red = band4.GetNoDataValue()
    nodataval_nir = band5.GetNoDataValue()

    #turn the bands into numpy arrays
    data_band4 = band4.ReadAsArray(0, 0, cols, rows).astype('float64')
    data_band5 = band5.ReadAsArray(0, 0, cols, rows).astype('float64')

    #address errors
    np.seterr(divide='ignore', invalid='ignore')

    #set the non vales
    data_band4[data_band4 == nodataval_red] = np.nan
    data_band5[data_band5 == nodataval_nir] = np.nan

    #calc NDVI
    ndvi = (data_band5 - data_band4) / (data_band5 + data_band4)

    #make ndvi band into a new image
    svi_temp_path = './temp/ndvi.tif'
    output_single_band_raster(data=ndvi, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_red)

def get_ndwi(fp_in_img, green_idx=3, nir_idx=5, raster_driver_name='GTiff'):
    """
    calcs NDVI and outputs an ndvi raster in the temp folder
    :param fp_in_img: file path
    :param red_idx: index for red band (ex: 4)
    :param nir_idx: index for nir band (ex: 5)
    :param raster_driver_name: driver name
    :return:
    """
    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    #get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    #info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols,rows,bands,projection)

    #get the band
    band3 = dataset.GetRasterBand(green_idx)
    band5 = dataset.GetRasterBand(nir_idx)

    # get the no data value
    nodataval_green = band3.GetNoDataValue()
    nodataval_nir = band5.GetNoDataValue()

    #turn the bands into numpy arrays
    green = band3.ReadAsArray(0, 0, cols, rows).astype('float64')
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')

    #address errors
    np.seterr(divide='ignore', invalid='ignore')

    #set the non vales
    green[green == nodataval_green] = np.nan
    nir[nir == nodataval_nir] = np.nan

    #calc NDVI
    ndwi = (green - nir) / (green + nir)

    #make ndvi band into a new image
    svi_temp_path = './temp/ndwi.tif'
    output_single_band_raster(data=ndwi, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_nir)

def get_savi(fp_in_img, red_idx=4, nir_idx=5, raster_driver_name='GTiff'):
    """
       calcs the SVI and outputs an SVI raster in the temp folder
       :param fp_in_img: file path
       :param red_idx: index for red band (ex: 4)
       :param nir_idx: index for nir band (ex: 5)
       :param raster_driver_name: driver name
       :return:
       """
    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band4 = dataset.GetRasterBand(red_idx)
    band5 = dataset.GetRasterBand(nir_idx)

    # get the no data value
    nodataval_red = band4.GetNoDataValue()
    nodataval_nir = band5.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    red = band4.ReadAsArray(0, 0, cols, rows).astype('float64')
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    red[red == nodataval_red] = np.nan
    nir[nir == nodataval_nir] = np.nan

    #calc SVI
    savi = 1.5 * (nir - red) / (nir + red + 0.5)

    #make ndvi band into a new image
    svi_temp_path = './temp/savi.tif'
    output_single_band_raster(data=savi, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_red)

def get_vi6t(fp_in_img, thermal_idx=9, nir_idx=5, raster_driver_name='GTiff'):
    """
       calcs the SVI and outputs an SVI raster in the temp folder
       :param fp_in_img: file path
       :param red_idx: index for red band (ex: 4)
       :param nir_idx: index for nir band (ex: 5)
       :param raster_driver_name: driver name
       :return:
       """
    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band9 = dataset.GetRasterBand(thermal_idx)
    band5 = dataset.GetRasterBand(nir_idx)

    # get the no data value
    nodataval_thermal = band9.GetNoDataValue()
    nodataval_nir = band5.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    thermal = band9.ReadAsArray(0, 0, cols, rows).astype('float64')
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    thermal[thermal == nodataval_thermal] = np.nan
    nir[nir == nodataval_nir] = np.nan

    #calc SVI
    vi6t = (nir - thermal) / (nir + thermal)

    #make ndvi band into a new image
    svi_temp_path = './temp/vi6t.tif'
    output_single_band_raster(data=vi6t, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_nir)

def get_vi43(fp_in_img, red_idx=4, nir_idx=5, raster_driver_name='GTiff'):
    """
       calcs the SVI and outputs an SVI raster in the temp folder
       :param fp_in_img: file path
       :param red_idx: index for red band (ex: 4)
       :param nir_idx: index for nir band (ex: 5)
       :param raster_driver_name: driver name
       :return:
       """
    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band4 = dataset.GetRasterBand(red_idx)
    band5 = dataset.GetRasterBand(nir_idx)

    # get the no data value
    nodataval_red = band4.GetNoDataValue()
    nodataval_nir = band5.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    red = band4.ReadAsArray(0, 0, cols, rows).astype('float64')
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    red[red == nodataval_red] = np.nan
    nir[nir == nodataval_nir] = np.nan

    #calc SVI
    vi43 = nir / red

    #make ndvi band into a new image
    svi_temp_path = './temp/vi43.tif'
    output_single_band_raster(data=vi43, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_red)

def get_vi45(fp_in_img, swir1_idx=6, nir_idx=5, raster_driver_name='GTiff'):
    """
       calcs the SVI and outputs an SVI raster in the temp folder
       :param fp_in_img: file path
       :param red_idx: index for red band (ex: 4)
       :param nir_idx: index for nir band (ex: 5)
       :param raster_driver_name: driver name
       :return:
       """
    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band6 = dataset.GetRasterBand(swir1_idx)
    band5 = dataset.GetRasterBand(nir_idx)

    # get the no data value
    nodataval_swir1 = band6.GetNoDataValue()
    nodataval_nir = band5.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    swir1 = band6.ReadAsArray(0, 0, cols, rows).astype('float64')
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    swir1[swir1 == nodataval_swir1] = np.nan
    nir[nir == nodataval_nir] = np.nan

    #calc SVI
    vi45 = nir / swir1

    #make ndvi band into a new image
    svi_temp_path = './temp/vi45.tif'
    output_single_band_raster(data=vi45, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_swir1)

def get_vi46(fp_in_img, thermal_idx=9, nir_idx=5, raster_driver_name='GTiff'):
    """
       calcs the SVI and outputs an SVI raster in the temp folder
       :param fp_in_img: file path
       :param red_idx: index for red band (ex: 4)
       :param nir_idx: index for nir band (ex: 5)
       :param raster_driver_name: driver name
       :return:
       """
    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band9 = dataset.GetRasterBand(thermal_idx)
    band5 = dataset.GetRasterBand(nir_idx)

    # get the no data value
    nodataval_thermal = band9.GetNoDataValue()
    nodataval_nir = band5.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    thermal = band9.ReadAsArray(0, 0, cols, rows).astype('float64')
    nir = band5.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    thermal[thermal == nodataval_thermal] = np.nan
    nir[nir == nodataval_nir] = np.nan

    #calc SVI
    vi46 = nir / thermal

    #make ndvi band into a new image
    svi_temp_path = './temp/vi46.tif'
    output_single_band_raster(data=vi46, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_nir)

def get_vi57(fp_in_img, swir1_idx=6, swir2_idx=7, raster_driver_name='GTiff'):
    """
       calcs the SVI and outputs an SVI raster in the temp folder
       :param fp_in_img: file path
       :param red_idx: index for red band (ex: 4)
       :param nir_idx: index for nir band (ex: 5)
       :param raster_driver_name: driver name
       :return:
       """
    dataset = __read_data(fp_in_img, raster_driver_name)
    str_out = ""

    # get data about the raster
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    projection = dataset.GetProjection()
    metadata = dataset.GetMetadata()
    geotransform = dataset.GetGeoTransform()

    # info out
    str_out += "Input Image: '{}'\n".format(fp_in_img)
    str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols, rows, bands, projection)

    # get the band
    band6 = dataset.GetRasterBand(swir1_idx)
    band7 = dataset.GetRasterBand(swir2_idx)

    # get the no data value
    nodataval_swir1 = band6.GetNoDataValue()
    nodataval_swir2 = band7.GetNoDataValue()

    # turn the bands into numpy arrays & turn to float64 so it calculates correctly
    swir1 = band6.ReadAsArray(0, 0, cols, rows).astype('float64')
    swir2 = band7.ReadAsArray(0, 0, cols, rows).astype('float64')

    # address errors
    np.seterr(divide='ignore', invalid='ignore')

    # set the non vales
    swir1[swir1 == nodataval_swir1] = np.nan
    swir2[swir2 == nodataval_swir2] = np.nan

    #calc SVI
    vi57 = swir1 / swir2

    #make ndvi band into a new image
    svi_temp_path = './temp/vi57.tif'
    output_single_band_raster(data=vi57, out_fp=svi_temp_path, col_size=cols, row_size=rows, num_band=1,
                              raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                              metadata=metadata, nodataval=nodataval_swir1)

def create_raster(temp_folder, out_path, name):
    """
    :return: A raster that combines all of the added rasters
    """

    #looks like I will have to make a temp folder with the SVIs then add them all to a new raster then delete
    #   the temp SVIs

    #write the SVI as a band with rasterio
    #https://gis.stackexchange.com/questions/223910/using-rasterio-or-gdal-to-stack-multiple-bands-without-using-subprocess-commands
    #https://gis.stackexchange.com/questions/49706/adding-band-to-existing-geotiff-using-gdal

    #empyt list of the files
    file_list = []

    #add the SVIs to the list
    for i in os.listdir(temp_folder):
        if i.endswith('.tif'):
            file_list.append(temp_folder + i)

    # Read metadata of first file
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count = len(file_list))

    #out file
    file_out = out_path + name

    # Read each layer and write it to stack
    with rasterio.open(file_out, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))


def delete_svis(temp_folder):
    #delete the temporary SVI files in the temp folder
    for i in os.listdir(temp_folder):
        try:
            del_path = temp_folder + i
            os.remove(del_path)
        except OSError as e:
            print("Error: %s : %s" % (i, e.strerror))
