#this script moves converts the files to all be in the same format befiore further processing can be done
# Chandler Ross

import os
import libs.project_raster as prj
import libs.resample_raster as resamp
import libs.clip_raster as clp

#workflow: project raster, rescale raster, clip raster

def raster_process(raster_folder, scratch_folder, ref_file, shapefile_clip, out_folder):
    """
    Formats the rasters to all be the same shape and pixel size and aligned
    :param raster_folder: filepath to the input data; string
    :param scratch_folder: filepath to a scratch folder; string
    :param ref_file: tif file (filepath) used for spatial reference and the allign the cells; string
    :param shapefile_clip: filepath to a shapefile to clip to; string
    :param out_folder: filepath to an output folder where the data will end up; string
    :return:
    """
    #iterate the process and for each
    for filename in os.listdir(raster_folder):
        if filename.endswith('.tif'):
            #make the file path
            f =os.path.join(raster_folder, filename)
            #Make the file name
            prj_out_name = filename[:-4] + '_prj.tif'
            prj_out_path = os.path.join(scratch_folder, prj_out_name)
            #project the raster
            prj.project_raster(f, prj_out_path)

            #change the name of the new output
            resamp_out_name = prj_out_name[:-4] + '_resamp.tif'
            resamp_out_path = os.path.join(scratch_folder, resamp_out_name)

            #resample the raster
            resamp.resample(ref_file, prj_out_path, resamp_out_path)

            # Final output file
            clip_out_name = resamp_out_name[:-4] + '_clp.tif'
            clip_out_path = os.path.join(out_folder, clip_out_name)

            #clip the raster to the extent
            clp.clip_raster(input_file=resamp_out_path, output_file=clip_out_path, shapefile=shapefile_clip)

if __name__ == '__main__':
    folder = r'C:/Users/caross2/Downloads/flammap/projected/'
    temp_folder = 'C:/Users/caross2/Downloads/LC_fire_season/temp/'
    ref_file = 'C:/Users/caross2/Downloads/clip_data/helping_files/Marin_DEM_resamp/extract_resamp_dem.tif'
    out_folder = 'C:/Users/caross2/Downloads/flammap/clipped/'
    shapefile = 'C:/Users/caross2/Downloads/clip_data/helping_files/marin_polygon_clip.shp'
    raster_process(folder, temp_folder, ref_file, shapefile, out_folder)







