#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# coding:utf-8

from easydict import EasyDict as edict
import utils.face_model as face_model
import cv2
import requests
import random
import numpy as np 
import face_preprocess
import schedule
import cv2
import os
import datetime
from retinaface import RetinaFace
DetectMode = {
	"ASF_DETECT_MODE_IMAGE": 0,
	"ASF_DETECT_MODE_VIDEO": 1
}
gpuid = 0
thresh = 0.8   
detector = RetinaFace('./model/mnet.25', 0, gpuid, 'net3')
####################################################################################################
'''
error = {'0001':'the input image file error','0002':'No face detect'}

'''
def Jod_get_facelog():
    url = 'https://tdx-data.oss-cn-beijing.aliyuncs.com/store/1/api/config/facelog'
    face_log = requests.get(url)
    np.save('document/devices.npy',face_log)

def Jod_get_allow_device():
    url = 'https://tdx-data.oss-cn-beijing.aliyuncs.com/store/1/api/config/facelog'
    face_log = requests.get(url)
    np.save('document/devices.npy',face_log)
 
def job_update_face():
    os.system('./rsync.sh')
    
def run():
    schedule.every(10).seconds.do(Jod_get_facelog)
    schedule.every(10).seconds.do(job_update_face)
 
    while True:
        schedule.run_pending()

# struct data 
DetectOrient = {
	"ASF_OP_0_ONLY": 1,
	"ASF_OP_90_ONLY": 2,
	"ASF_OP_270_ONLY": 3,
	"ASF_OP_180_ONLY": 4,
	"ASF_OP_0_HIGHER_EXT": 5
}
engineConfigTemplate = {
	"detectFaceMaxNum": 1,
	"detectFaceOrientPriority": DetectOrient["ASF_OP_0_HIGHER_EXT"],
	"detectFaceScaleVal": 12,
	"detectMode": DetectMode["ASF_DETECT_MODE_IMAGE"],

	"functionConfig":{
		"age": True,
		"gender": True,
		"faceDetect": True,
		"faceRecognition": False,
		"liveness": True,
		"rgbLivenessThreshold": 0.8,
	}
}

engineConfig = engineConfigTemplate
def get_args(engineConfig):
    '''
    funtion:
        Setting model parameters
    parameters:
        filePath: engineConfig
    return:
        imageInfo: dict
    '''
    args = edict()
    args.imagesize=  "112,112"     ####  检测到人脸图片后的图像大小
    args.FaceMaxNum =  engineConfig["detectFaceMaxNum"]
    args.detectMode = engineConfig['detectMode']
    args.age = engineConfig['functionConfig']["age"]
    args.gender = engineConfig['functionConfig']["gender"]
    args.faceRecognition = engineConfig['functionConfig']["faceRecognition"]
    args.liveness = engineConfig['functionConfig']["liveness"]
    args.rgbLivenessThreshold = engineConfig['functionConfig']["rgbLivenessThreshold"]
    args.gpu = 0  ##########  指定GPU
    args.det = 0
    args.flip = 0
    args.filename =['jpg','bmp','png','jpeg','tif']
    return args
####################################################################################################
# image methods
def Face_align(queue_camera_img,queue_camera_face,queue_face_landmark,queue_aliged_face):
    while True:
            faces = queue_camera_face.get()
            landmarks = queue_face_landmark.get()
            nimg = face_preprocess.preprocess(queue_camera_img.get() , faces[0][:-1], landmarks.reshape((5,2)), image_size='112,112')
            queue_aliged_face.put(nimg)
def process(queue_camera_img_filename,queue_camera_img,queue_camera_face,queue_face_landmark):
    while True:
        img = queue_camera_img_filename.get()
        img = cv2.imread('document/'+img)
        queue_camera_img.put(img)
        im_shape = img.shape
        scales = [128, 128]
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [im_scale]
        faces, landmarks = detector.detect(img, thresh, scales=scales)
        queue_camera_face.put(faces)
        queue_face_landmark.put(landmarks)
        
def Get_face_compare_result(usr,queue_face_features,Features,Feature_index):
    while True:
        feature = queue_face_features.get()
        index = usr.compareFaceFeatures(feature,Features)
#        if index == -1:
#            r = requests.post(request_url,headers=headers ,data=json.dumps(datas),timeout=(2,4))
#        else:
#            if index in Dd_weight[i]:
#                if  datetime.now() - Dd_weight[i][index] > 10:
#                    r = requests.post(request_url,headers=headers ,data=json.dumps(datas),timeout=(2,4))
        print(index)
        Feature_index.put(index)
        
def load_facebank(path):
    facebank_path = path
    embeddings = np.load(os.path.join(facebank_path,'facebank_mxnet.npy'))
    names = np.load(os.path.join(facebank_path,'names_mxnet.npy'))
    return embeddings, names
    
class ImageInfo:

    
    def rgb(self,filepath):
        '''
    funtion:
        generate imageinfo from rgb file
    parameters:
        filePath: file path
    return:
        imageInfo: dict
        '''
        try :
            imageInfo_templete = {"format": "CP_PAF_BGR24","width": int,"height": int,"data": bytes()}
            imageInfo =  imageInfo_templete.copy()
            if filepath.split('.')[-1] in ['jpg','bmp','png','jpeg','tif'] and os.path.exists(filepath):
                 im = cv2.imread(filepath)
                 imageInfo['width'] = im.shape[1]
                 imageInfo['height'] = im.shape[0]
                 imageInfo['data'] = im
                 return imageInfo
            else:
                 return '0001'
        except:
            return None
####################################################################################################
# face methods
class  FaceEngine:
    def __init__(self):
        '''
        funtion:
            create and configure face engine
        parameters:
            engineConfig: dict, engine configurations
        return:
            None
        '''
        args = get_args(engineConfig)
        self.args = args
        self.threshold = 0.9
        self.Gender = {"female": 0,"male":1}
        self.imageInfoTemplate = {"format": "CP_PAF_BGR24","width": 600,"height": 800,"data": bytes()}
        self.model =  face_model.FaceModel(self.args)
        self.faceInfoTemplate = {"rect": {"left": int,"right": int,"bottom": int,"top": int},"face":bytes()}
        self.facePropertyTemplate = {"age": int,"gender": self.Gender["male"],"liveness": bool}

    def get_sorce(self,dist):
        face_define = ''
        if (dist>1.2):
            sorce = random.uniform(20, 40)
            face_define = '不相似'
        elif (dist >= 1.0):
            sorce = random.uniform(70, 75)
            face_define = '比较相似'
        elif (dist >= 0.9):
            sorce = random.uniform(75, 80)
            face_define = '比较相似'
        elif (dist >=0.8):
            sorce = random.uniform(80, 85)
            face_define = '很相似'
        elif (dist >=0.7):
            sorce = random.uniform(85, 90) 
            face_define = '很相似'
        else:
            sorce = random.uniform(90, 95) 
            face_define = '非常相似'
        return sorce,face_define


#    def __del__(self):
#        '''
#        funtion:
#            close face engine
#        parameters:
#            None
#        return:
#            None
#        '''
#        class_name = self.__class__.__name__
#        print(class_name, '销毁')
    def Get_feature(self,queue_aliged_face,queue_face_features):
        while True:
            face_info = {}
            img = queue_aliged_face.get()
            face_image = np.expand_dims(np.transpose(img,(2,0,1)), axis=0)
            for n in range(queue_aliged_face.qsize()):
                img = queue_aliged_face.get()
                face_image1 = np.expand_dims(np.transpose(img,(2,0,1)), axis=0)
    #            face_image  = F_input.get()
                face_image = np.concatenate((face_image,face_image1),axis=0)
            face_info['face'] = face_image[0:20,:,:,:]
            face_feature = self.extractFaceFeature(face_info)
            face_feature = self.extractFaceFeature(face_info)
            n = int(len(face_feature[0])/512)
            feature = face_feature[0].reshape( n,512)
            for i in range(n):
                queue_face_features.put(feature[i,:])

    def detectFaces(self, imagedata):
        '''
        funtion:
            detect faces from image data
        parameters:
            imageInfo: , image information
        return:
            faceInfos: [dict], face infomation array
        '''
        res = []
#        try:
        faces,rectes = self.model.get_input(imagedata['data'])
        for face,rect in zip(faces,rectes):
            face_T = self.faceInfoTemplate.copy()
            face_T['face']  =  face
            for i,key in enumerate( face_T['rect'].keys()):
                 face_T['rect'][key] = int(rect[i])
            res.append(face_T)
        return res
#        except:
#            return '0002'


    def extractFaceFeature(self,  faceInfors):
        '''
        funtion:
            extract face feature from specified image part
        parameters:
            faceInfo: dict, face information
        return:
            faceFeature: bytes, face feature
        '''
        faceFeature = []
#        try:
#        for faceInfor in faceInfors:
        feature = self.model.get_feature(faceInfors['face'])
        faceFeature.append(feature)
        return faceFeature
#        except:
#            return None


    def compareFaceFeature(self, faceFeature1, faceFeature2):
        '''
        funtion:
            compare two face features
        parameters:
            faceFeature1: bytes, face feature 1
            faceFeature2: bytes, face feature 2
        return:
            similarScore: float, similar score
        '''
        try :
            if (len(faceFeature1) == 1) and (len(faceFeature2) == 1):
                compare = faceFeature1[0] - faceFeature2[0]
                x_norm=np.linalg.norm(compare)
                similarScore = self.get_sorce(x_norm)
                return similarScore
            return 
        except:
            return None

    def compareFaceFeatures(self, faceFeature1, faceFeature2):
        '''
        funtion:
            compare two face features
        parameters:
            faceFeature1: bytes, face feature 1
            faceFeature2: bytes, face feature 2
        return:
            similarScore: float, similar score
        '''
        try :
            compare = faceFeature1 - faceFeature2
            x_norm = np.linalg.norm(compare , ord=None, axis=1, keepdims=False)  
            min_list = x_norm.min()# 返回最大值
#            print(min_list)
            if min_list < self.threshold:
                min_index =  np.argwhere(x_norm == min_list)# 最大值的索引
                min_index = np.squeeze( min_index)
            else:
                min_index = -1
            return min_index
        except:
            return None

    def processFaceProperty(self,  faceInfos):
        '''
        funtion:
            analyse face properties
        parameters:
            faceInfos: [dict], face information array
        return:
            faceProperties: [dict], face property array
        '''
        try:
            faceProperties = self.facePropertyTemplate
            gender,age = self.model.get_ga(faceInfos[0]['face'])
            faceProperties['gender'] = gender
            faceProperties['age'] = age
            return faceProperties
        except:
            return None
    
