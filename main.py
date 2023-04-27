import os
import sys
from pathlib import Path

import torch
import cv2
import numpy as np
FILE = FILE = Path(os.path.abspath('')).resolve()

ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.augmentations import letterbox
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import torch
import torchvision

# print("PyTorch version:", torch.__version__)
# print("torchvision version:", torchvision.__version__)
weights = './runs/train/stitched_pretrained_rc2/weights/best.pt'
device = torch.device('cpu')
dnn = False
data = None
half = False
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
conf_thres=0.25
iou_thres=0.45
classes = None
agnostic_nms = False
max_det = 1000

class ImageViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("GUI")
        self.master.geometry('1000x800')

        # 创建左侧第一列
        self.title_label = tk.Label(master, text="Image")
        self.title_label.grid(row=0, column=0,  padx=10, pady=10)

        self.image_label_origin = tk.Label(master)
        self.image_label_origin.grid(row=1, column=0, padx=10, pady=10)

        self.image_label_detect = tk.Label(master)
        self.image_label_detect.grid(row=2, column=0, padx=10, pady=10)

        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.grid(row=3, column=0, padx=10, pady=10)

        # 创建右侧第二列
        self.text_label = tk.Label(master, text="Result")
        self.text_label.grid(row=0, column=1, padx=10, pady=10)

        self.result = tk.Label(master, wraplength=200)
        self.result.grid(row=1, column=1, padx=10, pady=10)

    def load_image(self):
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename()

        # 如果没有选择文件，则返回
        if not file_path:
            return
        img_show = Image.open(file_path).resize((450,300))
        photo = ImageTk.PhotoImage(img_show)
        self.image_label_origin.configure(image=photo)

        self.image_label_origin.image = photo


        # 加载图像并显示在标签上
        im0 = cv2.imread(file_path)
        imgsz = check_img_size((640,640), s=stride)
        im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        bs = 1  # batch_size
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for _, det in enumerate(pred):
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(im0, line_width=3, example=str(names))
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = names[c]
                annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()

        img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

        # 将OpenCV图像转换为PIL图像
        pil_img = Image.fromarray(img).resize((450,300))
        photo = ImageTk.PhotoImage(pil_img)
        self.image_label_detect.configure(image=photo)
        self.image_label_detect.image = photo


        # 更新文本标签的内容
        self.result.configure(text=f"已加载图像: {file_path}")



root = tk.Tk()
# 设置网格布局
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(1, weight=1)

# 创建ImageViewer对象并运行主循环
app = ImageViewer(root)
root.mainloop()
