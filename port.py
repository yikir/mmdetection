#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# n(net) o(oil) h(hang) r(rust) 检测模块
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from mmdet.models import build_detector
import mmcv
import torch
import cv2
import time
import json
from mmcv.runner import load_checkpoint
import PIL.Image as Image
import numpy as np
from torchvision.transforms import transforms
import pycocotools.mask as maskUtils

current_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(current_dir, 'configs','config_cascade_rcnn.py')
weight_file = '/home/kilox/weights/nohr_best.pth'
# weight_file = '/Weights/verified/oil_detection_v1/oil_best.pth'

class Object(object):
    def __init__(self):
        self.class_name = "Unknown"
        self.trust = 0.0
        self.rank = 0
    
    def to_json(self):
        return json.dumps(self.__dict__)


class Port:
    def __init__(self):
        self.cfg = mmcv.Config.fromfile(config_file)
        # 创建模型 , test_cfg 是rpn rcnn的nms等配置
        self.detector = build_detector(self.cfg.model, train_cfg=None, test_cfg=self.cfg.test_cfg)
        # 加载权重
        load_checkpoint(self.detector, weight_file, map_location='cpu')
        self.detector = self.detector.to('cuda')
        self.detector.eval()
        self.class_names = ('油污','鸟巢','锈蚀','飘挂物')
    
    def process(self, image,save=None):
        """
        :param image: PIL.Image 输入图像
        """
        np_image = np.asarray(image)
        img, img_meta = self.prepare_single(np_image)
        # forward
        with torch.no_grad():
            # 传入rescale则代表返回的mask是原图的
            result = self.detector.simple_test(img, [img_meta], proposals=None, rescale=True)
            # 将mask 以及bbox画在图上
            img = self.draw_image(np_image, img_meta, result)
            real_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
            output_file_name = os.path.join(real_time + '.jpg')
            cv2.imwrite(output_file_name, img)
        return False,None,output_file_name

    # 将图片添加meta的函数
    def prepare_single(self,img):
        img_info = {'height': img.shape[0], 'width': img.shape[1]}
        img_norm_cfg = self.cfg.img_norm_cfg
        size_divisor = self.cfg.data.test.size_divisor
        
        img, scale_factor = mmcv.imrescale(img, (4014,2400), return_scale=True)
        img_shape = img.shape
        
        img = mmcv.imnormalize(img, img_norm_cfg.mean, img_norm_cfg.std, img_norm_cfg.to_rgb)
        img = mmcv.impad_to_multiple(img, size_divisor)
        pad_shape = img.shape
        _img = transforms.ToTensor()(img).float()
        _img = _img.unsqueeze(0)
        _img_meta = dict(
            ori_shape=(img_info['height'], img_info['width'], 3),
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
        _img = _img.to('cuda')
        return _img, _img_meta,

    def draw_image(self,img, meta, result, score_thr=0.9):
        def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
            num_imgs = tensor.size(0)
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)
            imgs = []
            for img_id in range(num_imgs):
                img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
                img = mmcv.imdenormalize(
                    img, mean, std, to_bgr=to_rgb).astype(np.uint8)
                imgs.append(np.ascontiguousarray(img))
            return imgs
        
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        
        h, w, _ = meta['ori_shape']
        img_show = img[:h, :w, :].copy()
        
        bboxes = np.vstack(bbox_result)
        # 画mask
        # # draw segmentation masks
        # if segm_result is not None:
        #     segms = mmcv.concat_list(segm_result)
        #     inds = np.where(bboxes[:, -1] > score_thr)[0]
        #     for i in inds:
        #         color_mask = np.random.randint(
        #             0, 256, (1, 3), dtype=np.uint8)
        #         mask = maskUtils.decode(segms[i]).astype(np.bool)
        #         # todo fix dimension not equal
        #         img_check_shape = tuple(img_show.shape[0:2])
        #         if mask.shape != img_check_shape:
        #             width_diff = mask.shape[1] - img_check_shape[1]
        #             if mask.shape[1] < img_check_shape[1]:
        #                 mask = np.pad(mask, (0, width_diff), mode='constant', constant_values=False)
        #                 np.insert(mask, False, )
        #             else:
        #                 mask = mask[:, :-width_diff]
        #         img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
        # 画bbox
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img_show, left_top, right_bottom, (0, 255, 0), thickness=2)
            label_text = self.class_names[
                label] if self.class_names is not None else 'cls {}'.format(label)
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            cv2.putText(img_show, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0))
        
        return img_show

def test():
    pass


if __name__ == '__main__':
    im = Image.open('/home/kilox/3.jpg')
    port = Port()
    print(port.process(im,True))
