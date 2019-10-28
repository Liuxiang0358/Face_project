# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
'''
import multiprocessing
import base64
import bottle
import uuid
from concurrent.futures import ThreadPoolExecutor
from retinaface import RetinaFace
thresh = 0.8
import face
count = 1
usr = face.FaceEngine()#global DeviceIds
@bottle.route('/main', method='POST')
def login():
    
#    DeviceId = bottle.request.POST.get('DeviceId')
#    if DeviceId in DeviceIds:
    data = bottle.request.json
    img_camera_base64 =data['img']
#    data   = bottle.request.forms.get('name')
#    for d in data:
    img_camera  = base64.b64decode(img_camera_base64)
    file_name = uuid.uuid4().hex + '.jpg'
    with open('document/'+file_name, 'wb') as jpg_file:
        jpg_file.write(img_camera)
    queue_camera_img_filename.put(file_name)
#    queue_camera_img_filename.put('7d34eca4358144b8a88be5af3ab96e32.jpg')
    
    return {"HTTP/1.1 200 OK\r\n""Connection: Close\r\n""Content-Type: application/json;charset=UTF-8\r\n""Content-Length: %d\r\n\r\n""%s"}
        
if __name__ == "__main__":
    executor = ThreadPoolExecutor(max_workers=4)
# 通过submit函数提交执行的函数到线程池中，submit函数立即返回，不阻塞
    Features, names = face.load_facebank('facebank1')
    queue_aliged_face = multiprocessing.Queue()
    queue_camera_img_filename = multiprocessing.Queue()
    queue_camera_img = multiprocessing.Queue()
    queue_camera_face = multiprocessing.Queue()
    queue_face_landmark = multiprocessing.Queue()
    queue_face_features = multiprocessing.Queue()
    queue_face_property = multiprocessing.Queue()
    Feature_index = multiprocessing.Queue()
#    Get_feature(queue_aliged_face,queue_face_features)
#    p2 = Process(target=Face_align, args=(queue_camera_face,queue_camera_face ,queue_face_landmark))
#    face.process(queue_camera_img_filename,queue_camera_img,queue_camera_face,queue_face_landmark)
#    face.Face_align(queue_camera_img,queue_camera_face,queue_face_landmark,queue_aliged_face) 
#    usr.Get_feature(queue_aliged_face,queue_face_features)
#    face.Get_face_compare_result(usr,queue_face_features,Features,Feature_index)
#    print(names[Feature_index.get()+1])
    
    P_face_align = multiprocessing.Process(target=face.Face_align, args=(queue_camera_img,queue_camera_face,queue_face_landmark,queue_aliged_face))
    P_Get_face_compare_result = multiprocessing.Process(target=face.Get_face_compare_result, args=(usr,queue_face_features,Features,Feature_index))
#    P_face_align.start()
#    P_Get_face_compare_result.start()
#    Get_feature(F_input,Feature)
    task1 = executor.submit(face.process, (queue_camera_img,queue_camera_face,queue_face_landmark))
    task2 = executor.submit(usr.Get_feature, args=(queue_aliged_face,queue_face_features))
#    task3 = executor.submit(usr.run(), args=())         ################  执行定时任务，从阿里云更新face库
#    task4 = executor.submit(usr.run(), args=())
    
    P_face_align.start()
    P_Get_face_compare_result.start()
    
    print(queue_camera_img_filename.qsize())
    print(queue_camera_face.qsize())
    print(queue_face_landmark.qsize())
    print(queue_face_features.qsize())
    print(queue_camera_img.qsize())
    print(queue_camera_img.qsize())
    print(queue_camera_img.qsize())
    
    bottle.run(host='0.0.0.0', port=8080,debug = True,reloader =True)
    
#    print(F.qsize())
#    p3.start()
#    print(Feature.qsize())
#    print(F_input.qsize())
#    print(F_input.qsize())
#    a = F_input.get()











