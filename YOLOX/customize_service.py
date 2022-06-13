import numpy as np
from model_service.pytorch_model_service import PTServingBaseService
import os
import cv2
import torch
from torch.version import cuda
from yolox.data.datasets.voc_classes import VOC_CLASSES
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import time
import exps.example.yolox_voc.yolox_voc_s
from loguru import logger

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return None
        output = output.cpu()

        bboxes = output[:, 0:4]
        tmp = torch.Tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        bboxes = torch.mm(bboxes, tmp)
        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        return cls.tolist(), scores.tolist(), bboxes.tolist()


class DetectionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        # 调用父类构造方法
        super(PTServingBaseService, self).__init__(model_name, model_path)
        self.model_name = model_name
        self.model_path = model_path
        self.labels = VOC_CLASSES
        # 调用自定义函数加载模型
        self.predictor = self.get_model()
        logger.info("Loaded model from {}".format(model_path))
        self.model_inputs = {}
        self.model_outputs = None

    def get_model(self):
        # 加载saved_model格式的模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_file = self.model_path  # 训练好的参数文件的路径
        exp = exps.example.yolox_voc.yolox_voc_s.Exp()  # 创建实验
        model = exp.get_model()  # 创建模型
        model.to('cpu')
        model.eval()
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model"])
        predictor = Predictor(
            model, exp, self.labels, None, None,
            device, None, False,
        )
        predictor.device = 'cpu'
        return predictor

    def _preprocess(self, data):
        # https两种请求形式
        # 1. form-data文件格式的请求对应：data = {"请求key值":{"文件名":<文件io>}}
        # 2. json格式对应：data = json.loads("接口传入的json体")
        logger.info("data: {}".format(data))
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = cv2.imdecode(np.asarray(bytearray(file_content.read()), dtype=np.uint8), 1)
                self.model_inputs[k] = img
        if len(self.model_inputs) != 1:
            logger.error("data length: {}".format(len(self.model_inputs)))
        logger.info("preprocessed_data!")
        return self.model_inputs

    def _inference(self, data):
        self.model_outputs = {"detection_classes": [], "detection_scores": [], "detection_boxes": []}
        for key in self.model_inputs:
            outputs, img_info = self.predictor.inference(self.model_inputs[key])
            result = self.predictor.visual(output=outputs[0], img_info=img_info)
            logger.info("result: {}".format(result))
            for i in range(len(result[0])):
                if result[1][i] > 0.35:
                    self.model_outputs["detection_classes"].append(self.labels[int(result[0][i])])
                    self.model_outputs["detection_scores"].append(1.0)
                    self.model_outputs["detection_boxes"].append(result[2][i])
        logger.info("inferenced data:{}!".format(self.model_outputs))
        return self.model_outputs

    def _postprocess(self, data):
        return self.model_outputs
