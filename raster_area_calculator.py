#This script will calculate number of pixels of a given value, then calculate the area given the pixel size



#Install the libraries
import libs.indices as ind
import time, os
try:
    from osgeo import gdal
    from osgeo import gdalconst
except ImportError as e:
    print(e)
    import gdal
    import gdalconst
import numpy as np

#=======================================================================================================================
#                                Main Function
#=======================================================================================================================

def burned_area_calculator(input_raster):

    #turn the raster to a gdal dataset
    ds = ind.read_data(input_raster)

    #get the pixel size information from the dataset
    geotransform = ds.GetGeoTransform()
    pix_width = abs(geotransform[1])
    pix_height = abs(geotransform[5])

    column_num = ds.RasterXSize
    row_num = ds.RasterYSize

    #calculate the are of the pixel
    pix_area = pix_height * pix_width

    #turn the dataset into a single band
    band = ds.GetRasterBand(1)

    #make a list to count the number burned pixels in each kernel
    burn_pix_list = []

    #set the block size of the kernel
    block_size_x = 10000
    block_size_y = 10000

    for j in range(0, row_num, block_size_y):
        if j + block_size_y < row_num:
            number_rows = block_size_y
        else:
            number_rows = row_num - j

        for k in range(0, column_num, block_size_x):
            if k + block_size_x < column_num:
                number_columns = block_size_x
            else:
                number_columns = column_num - k
            # this is the kernel
            kernel_array = band.ReadAsArray(k, j, number_columns, number_rows)

            #get the count of burned pixels (value == 1) in the kernel
            kernel_count = np.count_nonzero(kernel_array == 1)

            #append the value to the list
            burn_pix_list.append(kernel_count)


    #add all the values in the list
    pix_count = sum(burn_pix_list)

    #calculate the area in square meters
    sqm = float(pix_count) * float(pix_area)

    #convert to hectares
    hectares = sqm / 10000.0

    return hectares


def iterate_bac(folder):
    output_dict = {}
    for file in os.listdir(folder):
        if file.endswith('.tif'):
            full_path = os.path.join(folder, file)
            # print(full_path)
            out_hect = burned_area_calculator(full_path)
            fn = str(file)
            output_dict[fn]=str(out_hect)
    return output_dict


#=======================================================================================================================
#                                Run the Script
#=======================================================================================================================

if __name__ == '__main__':
    t1 = time.time()
    # input_raster = r"D:\other_agency_maps\other_agency_maps\MTBS_Data\burn_stack\stand_age_mtbs.tif"
    # hect = burned_area_calculator(input_raster)
    input_folder = r'D:\other_agency_maps\other_agency_maps\MTBS_Data\raster_severity\final_years'
    dict = iterate_bac(input_folder)
    t2 = time.time()
    sec = t2-t1
    minutes = sec / 60

    # print(hect, 'hectacres\nProcess took {} minutes to complete'.format(minutes))
    print(dict, '\nProcess took {} minutes to complete'.format(minutes))


