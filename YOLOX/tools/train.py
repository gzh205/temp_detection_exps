#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

from loguru import logger

import torch
import torch.backends.cudnn as cudnn
import sys

from matplotlib import pyplot as plt

sys.path.append(r'/tmp/YOLOX')

from yolox.core import launch
from yolox.exp import Exp, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices

import smtplib
from email.mime.text import MIMEText
from email.header import Header


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def sendEmail(send_addr: str,aps: [...] = None, filename: str = './chart.html'):
    # draw pictures
    axis = []
    ap50 = []
    ap50_95 = []
    for i in range(len(aps)):
        axis.append(i + 1)
        ap50.append(aps[i][0])
        ap50_95.append(aps[i][1])
    plt.figure()
    plt.plot(axis, ap50, color='r', label='ap50')
    plt.plot(axis, ap50_95, color='b', label='ap50:95')
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='jpeg')
    figfile.seek(0)  # rewind to beginning of file
    smtpobj = smtplib.SMTP()
    smtpobj.connect('smtp.qq.com')
    smtpobj.login('1942592358', 'xaopzfujccbibbgg')
    msg = MIMEMultipart('mixed')
    msg['From'] = Header("YOLOX", 'utf-8')  # 发送者
    msg['To'] = Header("我自己", 'utf-8')  # 接收者
    msg['Subject'] = Header('YOLOX模型训练完毕', 'utf-8')
    image = MIMEImage(figfile.read())
    image.add_header('Content-ID', '<mean_ap>')
    msg.attach(image)
    text_html = MIMEText(open(filename,'r',encoding='utf-8').read(), 'html', 'utf-8')
    msg.attach(text_html)
    smtpobj.sendmail('1942592358@qq.com', send_addr, msg.as_string())
    smtpobj.quit()


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()
    sendEmail('1942592358@qq.com', aps=trainer.aps)

    # send email when train finished


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
