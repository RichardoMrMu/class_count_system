# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 11:02
# @Author  : RichardoMu
# @File    : database.py
# @Software: PyCharm

import pymysql
import datetime
import yacs.config
import time
import logger
from config import get_default_config

logger = logger.create_logger(name=__name__)

class MysqlConnect(object):
    def __init__(self,params:yacs.config.CfgNode,loggers=None):
        super(MysqlConnect, self).__init__()
        self.__db_host = params.host
        self.__db_user = params.user
        self.__db_port = params.port
        self.__db_passwd = params.password
        self.__db_database = params.db
        self.__db_charset = params.charset
        self.logger = loggers
    # 连接数据库
    def isConnection(self):
        try:
            self.__conn = pymysql.connect(host=self.__db_host, user=self.__db_user, passwd=self.__db_passwd, db=self.__db_database, port=self.__db_port, charset=self.__db_charset)
            self.logger.info(f"isConnection is successfully connected")
            return True
        except Exception as e:
            self.logger.info(f"isConnetction's error :{e}")
            return False

    def get_cursor(self):
        return self.__conn.cursor()

    def upload_recording(self,cap_url,class_number):
        # 将cap_url 转为class_id
        # 根据cap_url在class表中查询class_id

        if len(cap_url)==0:
            self.logger.info(f'no cap_url left, upload recording finished')
            return 0
        elif len(cap_url)==1:
            sql = 'select class_ip,class_id from class where class_ip = %s'.format(tuple(cap_url[0]))
        else:
            sql = 'select class_ip,class_id from class where class_ip in {}'.format(tuple(cap_url))
        try:
            flag = self.isConnection() # 连接数据库
            global cursor
            cursor = self.__conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            results = {i:j for (i,j) in results}
            create_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = ((results[i], class_number[i],create_time) for i in cap_url)
            sql = 'insert into `recording` (class_id,class_number,created_at) values(%s,%s,%s)'
            cursor.executemany(sql,data)
        except Exception as e:
            self.logger.info(f"upload_recording's error:{e}")
        finally:
            if flag:
                cursor.close()
                self.__conn.commit()
                self.__conn.close()
                self.logger.info(f"upload_recording is successful")

    # 将cap_url存入class中
    def upload_url(self,cap_url):
        try:
            flag = self.isConnection()
            global cursor
            cursor = self.__conn.cursor()
            create_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = ((i,create_time) for i in cap_url)
            sql = 'insert into `class` (class_ip,created_at) values(%s,%s)'
            cursor.executemany(sql,data)
        except Exception as e:
            self.logger.info(f"upload_url's error:{e}")
        finally:
            if flag:
                cursor.close()
                self.__conn.commit()
                self.__conn.close()
                self.logger.info(f"upload_url is successful")

    def upload_statistics(self,start_time,epoch_time,failed_url):
        if not failed_url:
            sql = 'insert into `statistics` (start_time,epoch_time,created_time) values(%s,%s,%s)'
            data = (start_time,epoch_time,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            try:
                flag = self.isConnection()  # 连接数据库
                global cursor
                cursor = self.__conn.cursor()
                cursor.execute(sql,data)
            except Exception as e:
                self.logger.info(f"upload_statistics's error:{e}")
            finally:
                if flag:
                    cursor.close()
                    self.__conn.commit()
                    self.__conn.close()
                    self.logger.info(f"upload_statistics is successful")
        elif len(failed_url) == 1:
            sql = 'select class_ip,class_id from class where class_ip = %s'
            try:
                flag = self.isConnection()  # 连接数据库
                global cursor1
                cursor1 = self.__conn.cursor()
                cursor1.execute(sql,tuple(failed_url))
                results = cursor1.fetchall()
                failed_id = [str(j) for (i, j) in results]
                # 将failed_id转为以,分割的string
                failed_id = ','.join(failed_id)
                created_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                data = (start_time,epoch_time,failed_id,created_time)
                sql = 'insert into `statistics` (start_time,epoch_time,failed_id,created_time) values(%s,%s,%s,%s)'
                cursor1.execute(sql,data)
            except Exception as e:
                self.logger.info(f"upload_statistics's error:{e}")
            finally:
                if flag:
                    cursor1.close()
                    self.__conn.commit()
                    self.__conn.close()
                    self.logger.info(f"upload_statistics is successful")
        else:
            sql = 'select class_ip,class_id from class where class_ip in {}'.format(tuple(failed_url))
            try:
                flag = self.isConnection()  # 连接数据库
                global cursor2
                cursor2 = self.__conn.cursor()
                cursor2.execute(sql)
                results = cursor2.fetchall()
                failed_id = [str(j) for (i, j) in results]
                # 将failed_id转为以,分割的string
                failed_id = ','.join(failed_id)
                created_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                data = (start_time, epoch_time, failed_id, created_time)
                sql = 'insert into `statistics` (start_time,epoch_time,failed_id,created_time) values(%s,%s,%s,%s)'
                cursor2.execute(sql, data)
            except Exception as e:
                self.logger.info(f"upload_statistics's error:{e}")
            finally:
                if flag:
                    cursor2.close()
                    self.__conn.commit()
                    self.__conn.close()
                    self.logger.info(f"upload_statistics is successful")


def main3():
    config = get_default_config()
    cap_url = [""]
    db = MysqlConnect(params=config.database,loggers=logger)
    db.upload_url(cap_url=cap_url)
def main():
    config = get_default_config()
    cap_url = [
               ""]
    db = MysqlConnect(params=config.database,loggers=logger)
    # db.upload_statistics(start_time='2',epoch_time=1.,failed_url=cap_url)
    db.upload_recording(cap_url=cap_url,class_number={cap_url[0]:5})
def main2():
    # config = get_default_config()
    # cap_url = [
    #     ""]
    # db = MysqlConnect(params=config.database, loggers=logger)
    # db.test()
    pass
if __name__ == '__main__':
    # main3()
    pass