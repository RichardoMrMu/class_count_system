# coding=utf-8
# @Time    : 2021/3/9 11:10
# @Author  : RichardoMu
# @File    : mainControl.py
# @Software: PyCharm

from camSubCountingClass import run_multi_camera_by_multiprocess
import argparse

from config import get_default_config
from systemUtils import update_default_config,get_frequent_quantum
import datetime
import time
import os,pathlib,logger

log_dir = './logs/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
output_dir = pathlib.Path(log_dir)
time_now = datetime.datetime.now()
str_now = datetime.datetime.strftime(time_now,'%Y-%m-%d-%H-%M-%S')
logger = logger.create_logger(name=__name__,
    output_dir=output_dir,
    filename=str_now+'log.txt')


def main():
    parser = argparse.ArgumentParser(description='camparameters')
    parser.add_argument('--config',type=str,help='Config file ppath for YACS. When using a config file, all the other commandline arguments are ignored. '
             'See ./config/system.yaml'
    )
    parser.add_argument('--thread_num', type=int, help='number of threading') 
    parser.add_argument('--process_num',  type=int, help='number of processes')  
    parser.add_argument('--width', type=float, help='width of image')  
    parser.add_argument('--height', type=float, help='height of image')  
    parser.add_argument('--network',  help='Backbone network mobile0.25 or resnet50')
    # parser.add_argument('--cam_path',  help='default cam url path ')
    parser.add_argument('--cuda_visible_num',  type=int, help='default cam url path ')
    parser.add_argument('--epoch_time', type=float, help='default epoch time')
    parser.add_argument('--cam_path',type=str,help='default HK camera urls xlsx or csv file path')
    args = parser.parse_args()
    config = get_default_config()
    if args.config:
        config.merger_from(args)
    else:
        update_default_config(config,args)
    logger.info(config)


    quantum_list = get_frequent_quantum(config)
    start_time = datetime.time(hour=int((config.time.start_time)/100),minute=(config.time.start_time-int(config.time.start_time/100)*100))
    # end_time = datetime.time(hour=int(config.time.end_time/100),minute=(config.time.end_time-int(config.time.end_time/100)*100))
    end_time = quantum_list[-1][1]
    while True:
        epoch_time = 0
        weekday = datetime.datetime.now().isoweekday()
        current_hour = datetime.datetime.now().hour
        current_min = datetime.datetime.now().minute
        current_time = datetime.time(hour=current_hour, minute=current_min)

        if current_time >end_time or current_time < start_time:
            config.time.on = False
        elif 6<=weekday<=7:
            epoch_time = config.time.epoch_time
            config.time.on = True
        else:
            if quantum_list[0][0]<=current_time<=quantum_list[0][1] or quantum_list[1][0]<=current_time<=quantum_list[1][1] or \
                quantum_list[2][0]<=current_time<=quantum_list[2][1] or quantum_list[3][0]<=current_time<=quantum_list[3][1] or \
                quantum_list[4][0]<=current_time<=quantum_list[4][1] or quantum_list[5][0]<=current_time<=quantum_list[5][1] : \
                epoch_time = config.time.epoch_time
            elif (config.time.time_scheme==1 and quantum_list[1][1]<=current_time<=datetime.time(hour=13,minute=40)) or (config.time.time_scheme==2 and quantum_list[1][1]<=current_time<=datetime.time(hour=14,minute=10)) :
                epoch_time = config.time.epoch_time
            else:
                epoch_time = config.time.frequent_time
            config.time.on = True
        if not config.time.on:
            time.sleep(1000)
        else:
            logger.info(f"this epoch's epoch_time is {epoch_time}")
            run_multi_camera_by_multiprocess(config,epoch_time=epoch_time,loggers=logger)


if __name__ == '__main__':
    main()

