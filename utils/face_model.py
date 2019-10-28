from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from utils.mtcnn_detector import MtcnnDetector
import utils.face_image as face_image
import utils.face_preprocess as face_preprocess


def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, args):
    self.args = args
    ctx = mx.gpu(args.gpu)
#    ctx = mx.gpu()
    _vec = args.imagesize.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = '/home/lx/Documents/insightface/models/model-r50-am-lfw/model,0'
    self.ga_model = '/home/lx/Documents/insightface/gender-age/model/gamodel-r50/model,0'
    if len(self.model)>0:
      self.model = get_model(ctx, image_size, self.model, 'fc1')
    if len(self.ga_model)>0:
      self.ga_model = get_model(ctx, image_size, self.ga_model, 'fc1')

#    self.threshold = args.threshold
    self.det_minsize = 50
    det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    print(mtcnn_path)
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold= det_threshold)
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector


  def get_input(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    bbox_total  = []
    aliged_total = []
    for i in range(min(self.args.FaceMaxNum,len(bbox))):
        box = bbox[i,0:4]
        point = points[i,:].reshape((2,5)).T
        nimg = face_preprocess.preprocess(face_img, box, point, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        bbox_total.append(box)
        aliged_total.append(aligned)
    return aliged_total,bbox_total

  def get_align(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    box = bbox[0,0:4]
    point = points[0,:].reshape((2,5)).T
    nimg = face_preprocess.preprocess(face_img, box, point, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))

    return aligned

  def get_feature(self, aligned):
#    input_blob = np.expand_dims(aligned, axis=0)
#    input_blob = np.concatenate((input_blob,input_blob),axis=0)
    data = mx.nd.array(aligned)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

  def get_ga(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.ga_model.forward(db, is_train=False)
    ret = self.ga_model.get_outputs()[0].asnumpy()
    g = ret[:,0:2].flatten()
    gender = np.argmax(g)
    a = ret[:,2:202].reshape( (100,2) )
    a = np.argmax(a, axis=1)
    age = int(sum(a))

    return gender, age

