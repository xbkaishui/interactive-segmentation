import torch

from typing import List
from isegm.inference.clicker import Click
from isegm.utils.misc import get_bbox_iou
from .base import BaseTransform
import cv2


class ZoomIn(BaseTransform):
    def __init__(self,
                 target_size=480,
                 skip_clicks=1,
                 expansion_ratio=1.4,
                 min_crop_size=10,
                 recompute_thresh_iou=0.5,
                 prob_thresh=0.49,
                 prev_probs=None):
        super().__init__()
        self.target_size = target_size
        self.min_crop_size = min_crop_size
        self.skip_clicks = skip_clicks
        self.expansion_ratio = expansion_ratio
        self.recompute_thresh_iou = recompute_thresh_iou
        self.prob_thresh = prob_thresh
        self._prev_probs = prev_probs

        self._input_image_shape = None
        self._object_roi = None
        self._roi_image = None

    def transform(self, image_nd, clicks_lists: List[List[Click]]):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        self.image_changed = False
        clicks_list = clicks_lists[0]
        if len(clicks_list) <= self.skip_clicks:
            return image_nd, clicks_lists

        self._input_image_shape = image_nd.shape
        if self._prev_probs is None and self.skip_clicks >= 0:
            return image_nd, clicks_lists

        if self._prev_probs is not None:
            current_pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
            if current_pred_mask.sum() <= 0 and self.skip_clicks >= 0:
                return image_nd, clicks_lists

        click_y = clicks_list[-1].coords[0]
        click_x = clicks_list[-1].coords[1]
        self._object_roi = get_object_roi(image_nd, click_y, click_x, self.target_size)
        self._roi_image = get_roi_image_nd(image_nd, self._object_roi)

        # todo save image
        # img_to_save = (self._roi_image[:,0:3,:,:]).squeeze() * 255
        # cv2.imwrite("/tmp/zoom_img.jpg", img_to_save.permute(1,2,0).numpy())

        self.image_changed = True
        tclicks_lists = [self._transform_clicks(clicks_list)]

        return self._roi_image.to(image_nd.device), tclicks_lists

    def inv_transform(self, prob_map):
        if self._object_roi is None:
            self._prev_probs = prob_map.cpu().numpy().copy()
            return prob_map

        assert prob_map.shape[0] == 1
        rmin, rmax, cmin, cmax = self._object_roi
        prob_map = torch.nn.functional.interpolate(prob_map, size=(rmax - rmin, cmax - cmin),
                                                   mode='bilinear', align_corners=True)

        if self._prev_probs is not None:
            # Create new prob map by combining previous and current
            new_prob_map = torch.from_numpy(self._prev_probs).to(prob_map.device)
            new_prob_map[:, :, rmin:rmax, cmin:cmax] = prob_map
        else:
            new_prob_map = torch.zeros(1, 1, *self._input_image_shape[2:],
                                        device=prob_map.device, dtype=prob_map.dtype)
            # Create new prob map and set everything outside the roi to 0
            new_prob_map[:, :, rmin:rmax, cmin:cmax] = prob_map

        self._prev_probs = new_prob_map.cpu().numpy()

        return new_prob_map

    def check_possible_recalculation(self):
        if self._prev_probs is None or self._object_roi is not None or self.skip_clicks > 0:
            return False

        pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
        if pred_mask.sum() > 0:
            possible_object_roi = get_object_roi(pred_mask, [],
                                                 self.expansion_ratio, self.min_crop_size)
            image_roi = (0, self._input_image_shape[2] - 1, 0, self._input_image_shape[3] - 1)
            if get_bbox_iou(possible_object_roi, image_roi) < 0.50:
                return True
        return False

    def get_state(self):
        roi_image = self._roi_image.cpu() if self._roi_image is not None else None
        return self._input_image_shape, self._object_roi, self._prev_probs, roi_image, self.image_changed

    def set_state(self, state):
        self._input_image_shape, self._object_roi, self._prev_probs, self._roi_image, self.image_changed = state

    def set_prev_mask(self, prev_mask):
        self._prev_probs = prev_mask

    def reset(self):
        self._input_image_shape = None
        self._object_roi = None
        self._prev_probs = None
        self._roi_image = None
        self.image_changed = False

    def _transform_clicks(self, clicks_list):
        if self._object_roi is None:
            return clicks_list

        rmin, rmax, cmin, cmax = self._object_roi
        crop_height, crop_width = self._roi_image.shape[2:]

        transformed_clicks = []
        for click in clicks_list:
            new_r = crop_height * (click.coords[0] - rmin) / (rmax - rmin + 1)
            new_c = crop_width * (click.coords[1] - cmin) / (cmax - cmin + 1)
            transformed_clicks.append(click.copy(coords=(new_r, new_c)))
        return transformed_clicks


def get_roi_image_nd(image_nd, object_roi):
    rmin, rmax, cmin, cmax = object_roi

    with torch.no_grad():
        roi_image_nd = image_nd[:, :, rmin:rmax, cmin:cmax]

    return roi_image_nd


def check_object_roi(object_roi, clicks_list):
    for click in clicks_list:
        if click.is_positive:
            if click.coords[0] < object_roi[0] or click.coords[0] >= object_roi[1]:
                return False
            if click.coords[1] < object_roi[2] or click.coords[1] >= object_roi[3]:
                return False

    return True


def get_object_roi(image_nd, click_y, click_x, target_size):
    img_h = image_nd.shape[2]
    img_w = image_nd.shape[3]
    crop_h = min(img_h, target_size)
    crop_w = min(img_w, target_size)
    crop_start_x = max(0, click_x - crop_w // 2)
    crop_start_x = min(img_w - crop_w, crop_start_x)
    crop_start_y = max(0, click_y - crop_h // 2)
    crop_start_y = min(img_h - crop_h, crop_start_y)
    crop_end_x = crop_start_x + crop_w
    crop_end_y = crop_start_y + crop_h

    # Return roi coordinates
    return (crop_start_y, crop_end_y, crop_start_x, crop_end_x)
