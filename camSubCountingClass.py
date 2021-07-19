# -*- coding: UTF-8 -*-
# @Time    : 2021/3/17 8:49
# @Author  : RichardoMu
# @File    : camSubCountingClass.py
# @Software: PyCharm
import cv2
import threading
import multiprocessing as mp
import datetime
import yacs.config
import os
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import non_max_suppression_face
from utils.datasets import letterbox

import numpy as np
from onvif import ONVIFCamera
from zeep.transports import Transport
import time
import requests
from requests.auth import HTTPDigestAuth
import io
from PIL import Image
import queue
from database import MysqlConnect
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# model paramters
conf_thres = 0.3
iou_thres = 0.5

class ReceiveThread(threading.Thread):
    def __init__(self,cam_url,thread_queues,failed_thread_queue=None,index=0,):
        super(ReceiveThread, self).__init__()
        self.start_time = time.time()
        self.cam_url = cam_url
        self.q1,self.q2 = thread_queues[0],thread_queues[1]
        self.index = index
       
        self.media = None
        self.ptz = None
        self.fail_connect = False
        self.faild_thread_queue = failed_thread_queue
      
        self.get_media_ptz()

    def get_media_ptz(self,port=80, admin_name='admin', passwd='abcd1234'):
            try:
                transport = Transport(timeout=3,operation_timeout=3)
                mycam = ONVIFCamera(self.cam_url, port, admin_name, passwd,transport=transport)
                self.media = mycam.create_media_service()
                self.ptz = mycam.create_ptz_service()
                self.faild_thread_queue[0].put(False)
            except Exception as e:
                self.fail_connect = True
                logger.info(f'error of get_media_ptz is -{e},the url is {self.cam_url}')
                self.faild_thread_queue[0].put(True)
                self.faild_thread_queue[1].put(self.cam_url)

    def get_image(self):
        media_profile = self.media.GetProfiles()[0]  
        res = self.media.GetSnapshotUri({'ProfileToken': media_profile.token})
        response = requests.get(res.Uri, auth=HTTPDigestAuth("admin", "abcd1234"))
        data = response.content
        return self.ByteStreamToImage(data)

    def ByteStreamToImage(self,byteData):
        bytes_stream = io.BytesIO(byteData)
        capture_img = Image.open(bytes_stream)
        capture_img = cv2.cvtColor(np.asarray(capture_img), cv2.COLOR_RGB2BGR)
        capture_img = cv2.resize(capture_img, (1280, 720))

        return capture_img

    def gotopreset(self,  i=1):
        params = self.ptz.create_type('GotoPreset')
        media_profile = self.media.GetProfiles()[0]  
        params.ProfileToken = media_profile.token
        params.PresetToken = i
        self.ptz.GotoPreset(params)


    def run(self):
        if not self.fail_connect:
            logger.info(
                f'{self.cam_url} start Receive')
            # sleep_time = np.random.randint(1, 10)
            # time.sleep(sleep_time)

            self.gotopreset(i=1)
            time.sleep(1)
     
            image = self.get_image()
            
            if (self.q1.qsize() == 1):
                self.q1.queue.clear()
                self.q1.put(image)
            else:
                self.q1.put(image)

            self.gotopreset(i=2)
            time.sleep(1)
            # 
            image = self.get_image()
            # 
            if (self.q2.qsize() == 1):
                self.q2.queue.clear()
                self.q2.put(image)
            else:
                self.q2.put(image)
            logger.info(
                f"{self.cam_url} get image, cost time :{time.time() - self.start_time}")
        else:
            pass

    def get_fail_connect_result(self):
        if self.fail_connect:
            return self.cam_url
        else:
            return False


class DisplayThread(threading.Thread):
    def __init__(self,thread_queues,cap_url=None,failed_thread_queues=None,index=0):
        super(DisplayThread, self).__init__()
        self.cap_url = cap_url
        self.index = index
        self.class_index = [i for i in range(len(self.cap_url))]  # [0,1,2,3,4]
        self.class_number_dict = {i: -1 for i in self.cap_url}
        self.class_failed_connect = []  
        self.thread_queues = thread_queues
        self.failed_thread_queues = failed_thread_queues

        self.device = torch.device('cuda')
        self.net = None
        # get model
        self.getmodel()

    def getmodel(self):
        torch.set_grad_enabled(False)
        # net and model
        self.net = RetinaStudent(cfg=cfg_resnext50, phase='test')
        self.net.load_state_dict(torch.load("./det.pth", map_location=torch.device('cpu')))
        self.net.to(self.device)
        self.net.eval()
        logger.info('Finished loading model!')
        cudnn.benchmark = True

    def run(self):
        while self.class_index:
            for i in self.class_index:
                # print(self.class_index)
                if not self.failed_thread_queues[i][0].empty():
                    res = self.failed_thread_queues[i][0].get()
                    if res:
                        self.class_failed_connect.append(self.failed_thread_queues[i][1].get())
                        self.class_index.remove(i)
                        del self.class_number_dict[self.cap_url[i]]
                    else:
                        q1 = self.thread_queues[i][0]
                        q2 = self.thread_queues[i][1]
                        print(f'i::{i}')
                        print(f'q2.empty() != True:{q2.empty() != True}')
                        print(f'q1.empty() != True:{q2.empty() != True}')
                        print(self.class_number_dict[self.cap_url[i]] == -1)
                        if q2.empty() != True and q1.empty() != True and self.class_number_dict[self.cap_url[i]] == -1:
                            # print("sadasdasdsa")
                            frame1 = q1.get()
                            frame2 = q2.get()
                            image1 = np.float32(frame1)
                            image2 = np.float32(frame2)
                            scale = torch.Tensor(
                                [image1.shape[1], image1.shape[0], image1.shape[1], image1.shape[0]]).to(self.device)
                            im_height1, im_width1, _ = image1.shape
                            im_height2, im_width2, _ = image1.shape
                            image1 -= (104, 117, 123)
                            image2 -= (104, 117, 123)
                            image1 = torch.from_numpy(image1).unsqueeze(0)
                            image2 = torch.from_numpy(image2).unsqueeze(0)
                            image1 = image1.to(self.device)
                            image2 = image2.to(self.device)
                            image1 = image1.permute(0, 3, 1, 2)
                            image2 = image2.permute(0, 3, 1, 2)
                            loc1, conf1 = self.net(image1)
                            loc2, conf2 = self.net(image2)
                            start_time = time.time()
                            num1 = getStudentsNum(loc1, conf1, im_height1, im_width1, scale, self.device)
                            end_time = time.time() - start_time
                            logger.info(f"after process cost:{end_time}")
                            num2 = getStudentsNum(loc2, conf2, im_height2, im_width2, scale, self.device)
                            num_total = num1 + num2
                            logger.info(f'num of student:{num_total},which index = {self.index},i = {i}')
                            self.class_number_dict[self.cap_url[i]] = num_total
                            self.class_index.remove(i)
                    self.failed_thread_queues[i][0].put(res)
                else:
                    pass

        connected_cap_url = list(self.class_number_dict.keys())
        db.upload_recording(cap_url=connected_cap_url, class_number=self.class_number_dict)


    def get_fail_connected_url(self):
        if self.class_failed_connect:
            return self.class_failed_connect
        else:
            return False


def get_model(device):
    torch.set_grad_enabled(False)
    # net and model
    net = attempt_load('./weights/yolov5n-0.5.pt', map_location=device) 
    net.eval()
    logger.info('Finished loading model!')
    cudnn.benchmark = True
    return net

def Display(thread_queues,cap_url=None,failed_thread_queues=None,index=0):
    class_index = [i for i in range(len(cap_url))]  # [0,1,2,3,4]
    class_number_dict = {i: -1 for i in cap_url}
    device = torch.device('cuda')
    net = get_model(device)
    while class_index:
        for i in class_index:
            if not failed_thread_queues[i][0].empty():
                res = failed_thread_queues[i][0].get()
                if res:
                    class_index.remove(i)
                    del class_number_dict[cap_url[i]]
                else:
                    q1 = thread_queues[i][0]
                    q2 = thread_queues[i][1]
                    if q2.empty() != True and q1.empty() != True and class_number_dict[cap_url[i]] == -1:
                    
                        torch.cuda.synchronize()
                        get_time = time.time()
                        frame1 = q1.get()
                        frame2 = q2.get()
                        start_time = time.time()
                        print(f"get images time:{start_time - get_time}")
        
                        image1 = np.float32(frame1)
                        image1 = letterbox(image1, new_shape=1920)[0]

                        image2 = np.float32(frame2)
                        image2 = letterbox(image2, new_shape=1920)[0]
                        torch.cuda.synchronize()
                        letter_time = time.time()
                        print(f"letterbox time :{letter_time - start_time}")
                        image1 = image1[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB
                        image1 = np.expand_dims(image1,axis=0)
                        image2 = image2[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB
                        image2 = np.expand_dims(image2,axis=0)
                        imgs = np.concatenate([image1,image2])
                        torch.cuda.synchronize()
                        prepocess_time = time.time() 
                        print(f"preprocess_time:{prepocess_time - letter_time}")
                        
                        imgs = torch.from_numpy(imgs).to(device)
                        imgs = imgs.float()
                        imgs/=255.0
                        
                        pred = net(imgs)[0]
                        torch.cuda.synchronize()
                        infer_time = time.time()
                        print(f"infer time :{infer_time - prepocess_time}")
                        # apply nms
                        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
                        torch.cuda.synchronize()
                        nms_time = time.time()
                        print(f"nms time :{nms_time - infer_time}")
                        num_total = 0
                        # print(pred.shape)
                        for det in pred:
                            num_total += len(det) if len(det) else 0
                        print(f"count time :{time.time() - nms_time}")
                        logger.info(f"{cap_url[i]}  process cost:{time.time() - start_time}")
                        logger.info(f'num of student:{num_total},which index = {index},i = {i}')
                        class_number_dict[cap_url[i]] = num_total
                        class_index.remove(i)
                        torch.cuda.empty_cache()
                        del imgs
                        del pred
                failed_thread_queues[i][0].put(res)
            else:
                pass

    connected_cap_url = list(class_number_dict.keys())
    db.upload_recording(cap_url=connected_cap_url, class_number=class_number_dict)


def image_process_thread(cap_url, index=0,fail_connected_url=None):
    # cap_url : xxx.xx.x.x
    # cap_url = ["172.21.0.69","172.21.0.99"]
    thread_num = len(cap_url)
    thread_queues = [[queue.Queue(maxsize=1), queue.Queue(maxsize=1)] for _ in cap_url]
    
    failed_thread_queues = [[queue.Queue(maxsize=1),queue.Queue(maxsize=1)] for _ in cap_url]
  
    threadings = []

    for i in range(thread_num):
        threadings.append(ReceiveThread(cam_url=cap_url[i], thread_queues=thread_queues[i], index=i,
                                        failed_thread_queue=failed_thread_queues[i]))
    threadings.append(threading.Thread(target=Display, args=(thread_queues,cap_url,failed_thread_queues,index)))
    for thread_ in threadings:
        thread_.setDaemon(True)
        thread_.start()
    for thread_ in threadings:
        thread_.join()

    temp_list =[]
    for i in failed_thread_queues:
        if i[0].get():
            temp_list.append(i[1].get())
    fail_connected_url.put(temp_list)



def run_multi_camera_by_multiprocess(config:yacs.config.CfgNode,epoch_time=300,loggers=None):
    start_time = time.time()
    start_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    global logger
    logger = loggers
    logger.debug(f'epoch time is {epoch_time}')



    global db
    db = MysqlConnect(params=config.database,loggers=logger)
  
    cam_url = list(set(config.model.cam_url))
    # cam_url = list(config.model.cam_url)*5
    CAMERA_COUNT = len(cam_url)
    process_gap = int(CAMERA_COUNT / config.system.process_num)
    logger.info(f"camera count :{CAMERA_COUNT}")
    logger.info(f"process_gap:{process_gap}")

    processes = []
    queues = []
    for i in range(config.system.process_num):
        if i == (config.system.process_num - 1):
            queue = mp.Queue()
            queues.append(queue)
            processes.append(mp.Process(target=image_process_thread, args=(cam_url[i * process_gap:],i,queue)))
            logger.info(f"len of cam_url : {len(cam_url[i * process_gap:])}")
            logger.info(f"cam url :{cam_url[i * process_gap:]}")
        else:
            queue = mp.Queue()
            queues.append(queue)
            processes.append(
                mp.Process(target=image_process_thread, args=(cam_url[i * process_gap:(i + 1) * process_gap],i,queue)))
            logger.info(f"len of cam_url : {len(cam_url[i * process_gap:(i + 1) * process_gap])}")
            logger.info(f"cam url :{cam_url[i * process_gap:(i + 1) * process_gap]}")

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()
    fail_connected_url = []
    for i in queues:
        while not i.empty():
            fail_connected_url += i.get()
    # print(time.time() - start_time)
    db.upload_statistics(start_time=start_date,epoch_time=time.time() - start_time, failed_url=fail_connected_url)
    sum = 1
    while time.time() - start_time < epoch_time :
            # or sum > 0 :
        pass
        sum = config.system.process_num
        for process in processes:
            if process.is_alive() != True:
                sum -= 1
    now_time = time.time() - start_time
    logger.info(f"run_multi_camera_by_multiprocess total cost time :{now_time}")

def run():
    while True:
        run_multi_camera_by_multiprocess()


if __name__ == '__main__':
    run()




