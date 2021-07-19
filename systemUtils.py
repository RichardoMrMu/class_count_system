# -*- coding: utf-8 -*-
# @Time    : 2021/3/10 10:22
# @Author  : RichardoMu
# @File    : systemUtils.py
# @Software: PyCharm

import yacs.config
import yaml
import datetime
import argparse
import logging
import os
import pandas as pd
import socket
logger = logging.getLogger(__name__)
def update_default_config(config: yacs.config.CfgNode,
                          args: argparse.Namespace) -> None:
    logger.debug('Called update_default_config()')
    if args.thread_num:
        logger.debug(f'--thread_num is {args.thread_num}')
        config.system.thread_num = args.thread_num
        logger.debug(f"Update config.system.thread_num to {config.system.thread_num}")
    if args.process_num:
        logger.debug(f'--process_num is {args.process_num}')
        config.system.process_num = args.process_num
        logger.debug(f"Update config.system.process_num to {config.system.process_num}")
    if args.width:
        logger.debug(f'--width is {args.width}')
        config.image.width = args.width
        logger.debug(f"Update config.image.width to {config.image.width}")
    if args.height:
        logger.debug(f'--height is {args.height}')
        config.image.height = args.height
        logger.debug(f"Update config.image.height to {config.image.height}")
    if args.network:
        logger.debug(f'--network is {args.network}')
        config.model.network = args.netword
        logger.debug(f"Update  config.model.network to { config.model.network}")
    if args.cam_path:
        logger.debug(f'--cam_path is {args.thread_num}')
        config.model.cam_path = args.cam_path
        logger.debug(f"Update config.model.cam_path  to {config.model.cam_path}")
    if args.cuda_visible_num:
        logger.debug(f'--cuda_visible_num is {args.thread_num}')
        config.model.cuda_visible_num = args.cuda_visible_num
        logger.debug(f"Update config.model.cuda_visible_num to {config.model.cuda_visible_num}")
    if args.epoch_time:
        logger.debug(f'--epoch_time is {args.thread_num}')
        config.time.epoch_time = args.epoch_time
        logger.debug(f"Update config.time.epoch_time to {config.time.epoch_time}")
    if args.cam_path:
        logger.debug(f'--cam_path is {args.cam_path}')
        config.model.cam_path = args.cam_path
    if config.model.cam_path:
        if os.path.exists(config.model.cam_path) and (config.model.cam_path.endswith('xlsx') or config.model.cam_path.endswith('csv')):
            if config.model.cam_path.endswith('csv') or config.model.cam_path.endswith('txt') :
                data = pd.read_csv(config.model.cam_path,index_col=False,header=0)
                config.model.cam_url = data.values.tolist()
            elif config.model.cam_path.endswith('xlsx'):
                data = pd.read_excel(config.model.cam_path,index_col=False,header=0)
                config.model.cam_url = data.values.tolist()
            else:
                raise ValueError('format of camera url file is not support, '
                                 'change the file like xlsx,txt or csc or modify the model.cam_url in ststem.yaml or defaults ')

    # config.time.start_time = datetime.time(hour=int((config.time.start_time)/100),minute=(config.time.start_time-int(config.time.start_time/100)*100))
    # config.time.end_time = datetime.time(hour=int(config.time.end_time/100),minute=(config.time.end_time-int(config.time.end_time/100)*100))


def get_frequent_quantum(config:yacs.config.CfgNode):
    quantum_list = []
    if config.time.time_scheme:
        if config.time.time_scheme == 1 and config.time.frequent_quantum_autumn:
            quantum = config.time.frequent_quantum_autumn
            for i in range(len(quantum)):
                quantum_list.append([datetime.time(hour=quantum[i][0][0], minute=quantum[i][0][1]),
                                     datetime.time(hour=quantum[i][1][0], minute=quantum[i][1][1])])
        elif config.time.time_scheme == 2 and config.time.frequent_quantum_summer:
            quantum = config.time.frequent_quantum_summer
            for i in range(len(quantum)):
                quantum_list.append([datetime.time(hour=quantum[i][0][0], minute=quantum[i][0][1]),
                                     datetime.time(hour=quantum[i][1][0], minute=quantum[i][1][1])])

    else:
        raise ValueError(
            'no specific time_scheme given'
        )
    print(quantum_list)
    return quantum_list


def isNetOK(testserver):
    s = socket.socket()
    s.settimeout(1)
    try:
        status = s.connect_ex(testserver)
        print(status)
        if status == 0:
            s.close()
            return True
        else:
            return False
    except Exception as e:
        return False
def isNetChinaOK(testserver=('www.baidu.com',443)):
    isOK = isNetOK(testserver)
    return isOK

def main():
    isok = isNetChinaOK()
    print(isok)


if __name__ == '__main__':
    main()






