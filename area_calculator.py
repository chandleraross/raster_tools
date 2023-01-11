#calculates the area based on the number of pixels


def area_calculator(pix_count, area=900,x=0, y=0):
    '''
    calculates the area burned based on number of pixels
    :param pix_count:
    :param area:
    :param x:
    :param y:
    :return: area
    '''
    if(area==900):
        burned_m2 = pix_count * area
        burned_hect = burned_m2 / 10000
        return burned_hect
    else:
        new_area = x * y
        burned_m2 = pix_count * new_area
        burned_hect = burned_m2 / 10000
        return burned_hect

#run the script
if __name__ == '__main__':
    pc = 19518
    a = 900
    x = 0
    y = 0
    print(area_calculator(pc, a, x, y))