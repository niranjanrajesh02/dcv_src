import os


DATA_PATH = 'C:/Niranjan/Ashoka/Research/DCV/Datasets/Shapes2D/dataset/output'

shape_count = {}

for file in os.listdir(DATA_PATH):
    shape = file.split('_')[0]
    
    if shape not in shape_count.keys():
        shape_count[shape] = 1
    else:
        shape_count[shape] += 1
    
    new_name = shape + '_' + str(shape_count[shape]).zfill(4) + '.png'
    src = os.path.join(DATA_PATH, file)
    dest = os.path.join(DATA_PATH, new_name)
    os.rename(src, dest)
    