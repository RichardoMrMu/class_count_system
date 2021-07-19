# Class Count System
To use yolov5 to count class students which cameras are hikcameras.
## Introduction

Using yolov5 training by myself to get student heads and count the number of class. Using onvif to control hikvision cameras to goto preset position, get stream image data and put iimages into model to get head number. In the end, your can upload the student number with pymysql to mysql. All information can store as logger.

## operating
1. you should set your camera ip in `./config/defaults.py`, add your own ips.
2. set passwd and username in `camSubCountingClass.py `  [this line](https://github.com/RichardoMrMu/class_count_system/blob/c612584f0f7bbbd40f35dde65f3264842c60ea3e/camSubCountingClass.py#L49)
3. run mainControl.py
```python
python mainControl.py
```


## Setting  
1. you can change the time table in `./config/defaults.py` or `system.yaml` 
2. your can change epoch time, which is all ips run one epoch time. default is 300s.
