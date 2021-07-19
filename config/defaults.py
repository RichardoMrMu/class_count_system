# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 20:43
# @Author  : RichardoMu
# @File    : defaults.py
# @Software: PyCharm
from yacs.config import CfgNode

config = CfgNode()
config.database = CfgNode()
# database
config.database.user = ''
config.database.port = 3306
config.database.host =  ''
config.database.password = ''
config.database.charset = 'utf8'
config.database.db = 'class_face'

# system
config.system = CfgNode()
config.system.cpu = False # Use cpu inference
config.system.thread_num = 5 # number of threading
config.system.process_num = 4# number of processes
# time
config.time = CfgNode()
config.time.on = True
config.time.epoch_time = 300
config.time.time_scheme = 2 
config.time.frequent_quantum_summer = [[[8,0],[9,40]],[[10,10],[11,50]],[[14,30],[16,5]],[[16,25],[18,0]],[[19,0],[20,35]],[[20,45],[22,50]]] # 秋季作息时间
config.time.frequent_quantum_autumn = [[[8,0],[9,40]],[[10,10],[11,50]],[[14,00],[15,35]],[[15,55],[17,30]],[[18,30],[20,5]],[[20,15],[21,50]]] # 夏季作息时间
config.time.start_time = 750
config.time.end_time = 2220
config.time.frequent_time = 60
# image
config.image = CfgNode()
config.image.width = 720 # width of image
config.image.height = 1280 # height of image

# model
config.model = CfgNode()
config.model.network = 'resnext50' # Backbone network mobile0.25 or resnet50
config.model.cuda_visible_num = 1 # default cuda visible number
config.model.cam_path = '' # default cam url path
config.model.cam_url = [
                        ]


def get_default_config():
    return config.clone()


def main():
    config = get_default_config()
    print(type(config.time.end_time))
    config.merge_from_file("system.yaml")
    print(config)
    print(type(config.time.start_time))
if __name__ == '__main__':
    main()
