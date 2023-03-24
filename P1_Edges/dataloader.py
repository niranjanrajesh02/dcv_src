import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import cv2 as cv
import random




class DataLoader(tf.keras.utils.Sequence):
    
  def __init__(self, arg=None, is_val=True):

    self.is_training = True
    self.dim_w = 256
    self.dim_h = 256
    self.args = arg
    self.base_dir = '/storage/niranjan.rajesh_ug23/DCV_data/BIPEDv1/edges'
    self.is_val = is_val
    self.bs = 32
    self.shuffle=self.is_training
    self.data_list = self._build_index()
    self.on_epoch_end()
    self.scale = None

  def _build_index(self):

    # base_dir = os.path.join(self.base_dir, self.args.model_state.lower())

    file_path = os.path.join(self.base_dir, 'train_rgb.lst')
    with open(file_path,'r') as f:
        file_list = f.readlines()
    file_list = [line.strip() for line in file_list] # to clean the '\n'
    file_list = [line.split(' ') for line in file_list] # separate paths


    m_mode = 'train' if self.is_training else 'test'
    input_path = [os.path.join(
        self.base_dir,'imgs',m_mode,line[0]) for line in file_list]
    gt_path = [os.path.join(
        self.base_dir,'edge_maps',m_mode,line[1]) for line in file_list]
    

    # split training and validation, val=10%
    if self.is_training and self.is_val:
        input_path = input_path[int(0.9 * len(input_path)):]
        gt_path = gt_path[int(0.9 * len(gt_path)):]
    elif self.is_training:
        input_path = input_path[:int(0.9 * len(input_path))]
        gt_path = gt_path[:int(0.9 * len(gt_path))]

    if not self.is_training:
        self.imgs_name = [os.path.basename(k) for k in input_path]
        for tmp_path in input_path:
            tmp_i = cv.imread(tmp_path)
            tmp_shape = tmp_i.shape[:2]
            self.imgs_shape.append(tmp_shape)
    sample_indices= [input_path, gt_path]
    return sample_indices

  def on_epoch_end(self):
    self.indices = np.arange(len(self.data_list[0]))
    if self.shuffle:
        np.random.shuffle(self.indices)

  def __len__(self):
    return len(self.indices)//self.bs


  def __getitem__(self, index):

    indices = self.indices[index*self.bs:(index+1)*self.bs] 
    x_list,y_list = self.data_list
    tmp_x_path = [x_list[k] for k in indices]
    tmp_y_path = [y_list[k] for k in indices]

    x,y = self.__data_generation(tmp_x_path,tmp_y_path)
    
    return x,y

  def __data_generation(self,x_path,y_path):
    if self.scale is not None and not self.is_training:
        scl= self.scale
        scl_h = int(self.dim_h*scl) if (self.dim_h*scl)%16==0 else \
            int(((self.dim_h*scl) // 16 + 1) * 16)
        scl_w = int(self.dim_w * scl) if (self.dim_w * scl) % 16 == 0 else \
            int(((self.dim_h * scl) // 16 + 1) * 16)

        x = np.empty((self.bs, scl_h, scl_w, 3), dtype="float32")
    else:
        x = np.empty((self.bs, self.dim_h, self.dim_w, 3), dtype="float32")
    y = np.empty((self.bs, self.dim_h, self.dim_w, 1), dtype="float32")

    for i,tmp_data in enumerate(x_path):
        tmp_x_path = tmp_data
        tmp_y_path = y_path[i] 
        tmp_x,tmp_y = self.transformer(tmp_x_path,tmp_y_path)
        x[i,]=tmp_x
        y[i,]=tmp_y

    return x,y

  def transformer(self, x_path, y_path):
    tmp_x = cv.imread(x_path)
    if y_path is not None:
        tmp_y = cv.imread(y_path,cv.IMREAD_GRAYSCALE)
    else:
        tmp_y=None
    h,w,_ = tmp_x.shape
    
    if self.dim_w!=w and self.dim_h!=h:
        tmp_x = cv.resize(tmp_x, (self.dim_w, self.dim_h))
    if tmp_y is not None:
        tmp_y = cv.resize(tmp_y, (self.dim_w, self.dim_h))

    if tmp_y is not None:
        tmp_y = np.expand_dims(np.float32(tmp_y)/255.,axis=-1)
    tmp_x = np.float32(tmp_x)
    return tmp_x, tmp_y
