import os
import sys
import cv2
import numpy as np
from read_datasets import *

def add_train_val(dir, train_ratio=0.8):
    annotations_dir = os.path.join(dir, 'Annotations')
    main_dir = os.path.join(dir, 'ImageSets', 'Main')
    files = os.listdir(annotations_dir)
    rand_idx = np.sort(np.random.randint(0, len(files), int(train_ratio * len(files))))
    train_file = open(os.path.join(main_dir, 'train.txt'), 'w')
    test_file = open(os.path.join(main_dir, 'test.txt'), 'w')
    train_val_file = open(os.path.join(main_dir, 'trainval.txt'), 'w')
    for i in range(len(files)):
        if i in rand_idx:
            train_file.write(files[i][:-4] + '\n')
        else:
            test_file.write(files[i][:-4] + '\n')
        train_val_file.write(files[i][:-4] + '\n')
    train_file.close()
    test_file.close()
    train_val_file.close()


CLASSES = ['lighthouse', 'sailboat', 'buoy', 'railbar', 'cargoship', 'navalvessels', 'passengership', 'dock',
           'submarine', 'fishingboat']


def add_dir(base_dir, dir_path):
    """
    在base_dir下添加dir_path
    :param base_dir: 基础路径
    :param dir_path: 文件夹路径
    :return: 拼接后的路径
    """
    res = os.path.join(base_dir, dir_path)
    if not os.path.exists(res):
        os.mkdir(res)
    return res


def convert_to_voc(data_dir, save_dir):
    """
    读取数据集，并将该格式转换为VOC格式
    :param save_dir: 保存的位置
    :param dir_path: 数据集所在的文件夹路径
    :return: 返回数据集的文件名列表
    """
    labels_path = os.path.join(data_dir, 'labels')
    images_path = os.path.join(data_dir, 'images')
    file_list = os.listdir(labels_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = add_dir(save_dir, 'VOC2007')
    image_save_dir = add_dir(save_dir, 'JPEGImages')
    annotation_save_dir = add_dir(save_dir, 'Annotations')
    i = 0
    file_num = len(file_list)
    classes_file = {}
    for c in CLASSES:
        classes_file[c] = []
    for file in file_list:
        boxes_str = open(os.path.join(labels_path, file), 'r', encoding='utf-8').readlines()
        boxes = []
        image = cv2.imread(os.path.join(images_path, file.replace('txt', 'jpg')))
        if image is None:
            continue
        height, width = image.shape[:2]
        for box_str in boxes_str:
            if len(box_str) > 0:
                box = box_str.split(' ')
                x_center = float(box[1]) * width
                y_center = float(box[2]) * height
                w = float(box[3]) * width / 2
                h = float(box[4]) * height / 2
                boxes.append([x_center - w, y_center - h, x_center + w, y_center + h, CLASSES[int(box[0])]])
                classes_file[CLASSES[int(box[0])]].append(file.replace('.txt', ''))
        xml_content = XmlContent(image, boxes)
        xml_content.save(os.path.join(annotation_save_dir, file.replace('txt', 'xml')),
                         os.path.join(image_save_dir, file.replace('txt', 'jpg')))
        del xml_content
        i += 1
        if i % 100 == 0:
            print(str(i) + '/' + str(file_num))
    # 训练集和验证集的文件名列表，采用k折交叉验证的方式，将数据集分为训练集和验证集
    # 创建ImageSets文件夹
    image_set_dir = add_dir(save_dir, 'ImageSets')
    main_dir = add_dir(image_set_dir, 'Main')
    # 统计类别数量
    for key in classes_file:
        print(key + ': ' + str(len(classes_file[key])))
        train_val_file = open(os.path.join(main_dir, key + '_trainval.txt'), 'w')
        for file in classes_file[key]:
            train_val_file.write(file + '\n')
        train_val_file.close()
    # 构建训练集和验证集的文件名列表
    train_file = open(os.path.join(main_dir, 'train.txt'), 'w')
    test_file = open(os.path.join(main_dir, 'test.txt'), 'w')
    idx = np.random.randint(0, len(file_list), size=int(len(file_list) * 0.8))
    for i in range(len(file_list)):
        if i in idx:
            train_file.write(file_list[i].replace('.txt', '') + '\n')
        else:
            test_file.write(file_list[i].replace('.txt', '') + '\n')
    train_file.close()
    test_file.close()


if __name__ == "__main__":
    #add_train_val(r'E:\datasets\ship_voc\VOC2007')
    #convert_to_voc(r'E:\datasets\train',r'E:\datasets\ship_voc')
    temp = read_xml_voc(r'E:\datasets\ship_voc\VOC2007\Annotations\5658.xml',r'E:\datasets\ship_voc\VOC2007\JPEGImages')
    img = temp.display()
    cv2.imshow('res',img)
    cv2.waitKey(0)
    sys.exit()
