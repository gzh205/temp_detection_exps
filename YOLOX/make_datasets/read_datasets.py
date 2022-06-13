import cv2
import numpy
import numpy as np
import os
import random
import shutil
import sys
from lxml import etree as ET


class XmlContent:
    """
    VOC数据集图片和标注框的信息
    """

    def __init__(self, image, boxes):
        """
        构造函数

        :param image opencv打开的图片
        :param boxes list，每个元素bbox都是一个list，bbox的第一个元素是xmin，第二个元素是ymin，第三个元素是xmax，第四个元素是ymax，第五个元素是label标签
        但是被dataloader加载后会将label从字符串改为整数，并且boxes的数据类型也会变成numpy数组
        """
        self.image = image
        self.boxes = boxes

    def save(self, path: str, img_filename: str):
        """
        保存所有成员函数为一个xml文件，该文件为VOC格式

        :param path xml文件的路径
        :param img_filename 图片文件夹的路径
        """
        # 将image写入文件中
        cv2.imwrite(img_filename, self.image)
        # 从path中截取文件名
        filename = os.path.basename(img_filename)
        # 保存xml文件
        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'Annoations'
        ET.SubElement(root, 'filename').text = filename
        ET.SubElement(root, 'path').text = 'we dont use it!'
        source = ET.SubElement(root, 'source')
        ET.SubElement(source, 'database').text = 'The VOC2007 Database'
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(self.image.shape[1])
        ET.SubElement(size, 'height').text = str(self.image.shape[0])
        ET.SubElement(size, 'depth').text = str(self.image.shape[2])
        ET.SubElement(root, 'segmented').text = '0'
        for box in self.boxes:
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = box[4]
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(round(box[0]))
            ET.SubElement(bndbox, 'ymin').text = str(round(box[1]))
            ET.SubElement(bndbox, 'xmax').text = str(round(box[2]))
            ET.SubElement(bndbox, 'ymax').text = str(round(box[3]))
        tree = ET.ElementTree(root)
        tree.write(path, pretty_print=True)

    def display(self) -> numpy.ndarray:
        """
        在原图中绘制检测框
        :return: 绘制完检测框的图片
        """
        img = self.image.copy()
        for box in self.boxes:
            text = box[4]
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            color = (0, 0, 255)
            txt_color = (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            cv2.putText(img, text, (x0, y0 - txt_size[1]), font, 0.4, txt_color, thickness=1)
        return img


def read_xml_voc(xml_path: str, img_dir: str) -> XmlContent:
    """
    用xpath读取VOC数据集

    :param xml_path xml文件的路径
    :param img_dir 图片文件夹的路径
    """
    tree = ET.parse(source=xml_path)
    root = tree.getroot()
    bbox = []
    # 读取标签和bbox
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        labels = obj.find('name').text
        bbox.append([xmin, ymin, xmax, ymax, labels])
    # 读取图片
    img_filename = root.find('filename').text
    image = cv2.imdecode(np.fromfile(os.path.join(img_dir, img_filename), dtype=np.uint8), -1)
    return XmlContent(image, bbox)


# 用xpath读取COCO数据集
def read_xml_coco(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    labels = []
    for obj in root.findall('object'):
        xmin.append(int(obj.find('bndbox').find('xmin').text))
        ymin.append(int(obj.find('bndbox').find('ymin').text))
        xmax.append(int(obj.find('bndbox').find('xmax').text))
        ymax.append(int(obj.find('bndbox').find('ymax').text))
        labels.append(obj.find('name').text)
    return xmin, ymin, xmax, ymax, labels


def read_xml_dir(xml_dir):
    """
    打开文件夹下的所有xml文档

    :param xml_dir xml文件夹的路径
    """
    xml_list = []
    for root, dirs, files in os.walk(xml_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.xml':
                xml_list.append(os.path.join(root, file))
    return xml_list
