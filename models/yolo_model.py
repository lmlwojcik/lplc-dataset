from ultralytics import YOLO
import json

def create_yolo(cfg):
    yolo_name = cfg['yolo_path']
    yolo = YOLO(yolo_name)
    return yolo

def train_yolo(yolo, cfg, dataset):

    yolo.train(data=dataset['dir'], **cfg)

    pass
