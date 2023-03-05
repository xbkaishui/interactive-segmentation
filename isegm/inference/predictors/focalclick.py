import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide, ResizeTrans, ZoomIn
from isegm.utils.crop_local import map_point_in_bbox, get_focus_cropv1, draw_mask_on_image
from loguru import logger
import cv2
import copy
import numpy as np

debug = True


class FocalPredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 infer_size = 256,
                 focus_crop_r = 1.4):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model
        logger.info("click models {}", self.click_models)
        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.crop_l = infer_size
        self.focus_crop_r = focus_crop_r
        self.transforms.append(ResizeTrans(self.crop_l))
        self.transforms.append(SigmoidForPred())
        self.focus_roi = None
        self.global_roi = None

        if self.with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image):
        self.rgb_original_image = torch.from_numpy(image).numpy()
        logger.info("rgb_original_image shape {}", self.rgb_original_image.shape)
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def set_prev_mask(self, mask):
        self.prev_prediction = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device).float()

    def update_zoom_in(self, zoom_in_params):
        # Updates target size, skip clicks, and expansion ratio of last zoom-in transform
        if zoom_in_params is None:
            return
        for t in reversed(self.transforms):
            if isinstance(t, ZoomIn):
                t.target_size = zoom_in_params.get('target_size')
                t.skip_clicks = zoom_in_params.get('skip_clicks')
                t.expansion_ratio = zoom_in_params.get('expansion_ratio')
                return
    def get_prediction(self, clicker, prev_mask=None):
        clicks_list = clicker.get_clicks()
        click = clicks_list[-1]
        last_y, last_x = click.coords[0], click.coords[1]
        self.last_y = last_y
        self.last_x = last_x

        if self.click_models is not None:
            logger.info("update click model")
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
            if prev_mask is not None:
                logger.info("pre mask {}", prev_mask.shape)
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            logger.info("concat prev mask {}", prev_mask)
            input_image = torch.cat((input_image, prev_mask), dim=1)

        if debug:
            c_input_mask = draw_mask_on_image(self.rgb_original_image, prev_mask.squeeze(), clicks_list=clicks_list)
            cv2.imwrite("/tmp/c_input_mask.jpg", c_input_mask)

        image_nd, clicks_lists = self.apply_transforms(
            input_image, [clicks_list]
        )

        # for debug purpose todo save image
        if debug:
            img_to_save = image_nd[:, 0:3, :, :].squeeze() * 255
            # cv2.imwrite("/tmp/after_apply_transforms.jpg", img_to_save.permute(1, 2, 0).numpy())
        try:
            roi = self.transforms[0]._object_roi
            y1, y2, x1, x2 = roi
            global_roi = (y1, y2 + 1, x1, x2 + 1)
            if debug:
                input_crop_image = np.zeros_like(prev_mask.squeeze())
                input_crop_image[y1:y2 + 1, x1:x2 + 1] = 1
                c_input_crop = draw_mask_on_image(self.rgb_original_image, input_crop_image, clicks_list=clicks_list)
                cv2.imwrite("/tmp/c_input_crop.jpg", c_input_crop)
        except:
            h,w = prev_mask.shape[-2], prev_mask.shape[-1]
            global_roi = (0,h,0,w)            
        self.global_roi = global_roi
        logger.info("global roi {}", global_roi)

        pred_logits, feature = self._get_prediction(image_nd, clicks_lists)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        prev_mask_copied = copy.deepcopy(prev_mask)
        for t in reversed(self.transforms):
            logger.info("transform {}", t)
            prediction = t.inv_transform(prediction)
        prev_mask = prev_mask_copied
        prediction = torch.log(prediction/(1-prediction))
        coarse_mask = prediction
        logger.info("coarse_mask shape {}", coarse_mask.shape)
        clicks_list = clicker.get_clicks()

        coarse_mask_np = coarse_mask.cpu().numpy()[0, 0] 
        prev_mask_np = prev_mask.cpu().numpy()[0, 0] 
        logger.info("coarse_mask_np {}", coarse_mask_np[global_roi[0]:global_roi[1], global_roi[2]:global_roi[3]])
        logger.info("prev_mask_np {}", prev_mask_np[global_roi[0]:global_roi[1], global_roi[2]:global_roi[3]])
        y1, y2, x1, x2 = get_focus_cropv1(coarse_mask_np, prev_mask_np, global_roi, last_y, last_x, self.focus_crop_r, self.rgb_original_image, [click])

        focus_roi = (y1, y2, x1, x2)
        self.focus_roi = focus_roi
        logger.info("focus_roi {}", focus_roi)
        focus_roi_in_global_roi = self.mapp_roi(focus_roi, global_roi)
        focus_pred = self._get_refine(pred_logits, self.original_image, clicks_list, feature, focus_roi, focus_roi_in_global_roi)#.cpu().numpy()[0, 0]
        focus_pred = F.interpolate(focus_pred, (y2-y1, x2-x1), mode='bilinear',align_corners=True)#.cpu().numpy()[0, 0]

        if len(clicks_list) > 10:
            coarse_mask = self.prev_prediction
            coarse_mask = torch.log(coarse_mask/(1-coarse_mask))
        focus_pred_sigmoid = torch.sigmoid(focus_pred)
        # todo del
        coarse_mask[:,:, y1: y2, x1: x2] = focus_pred_sigmoid
        prev_mask[:,:, y1:y2, x1:x2] = focus_pred_sigmoid
        if debug:
            w, h, c = self.rgb_original_image.shape
            r_output_crop_img = np.zeros((w, h))
            focus_pred_res = torch.sigmoid(focus_pred)
            focus_pred_res[focus_pred_res > 0.5] = 1
            focus_pred_res[focus_pred_res < 0.5] = 0
            r_output_crop_img[y1:y2, x1:x2] = focus_pred_res
            r_output_predict_image = draw_mask_on_image(self.rgb_original_image, r_output_crop_img)
            cv2.imwrite("/tmp/r_output_predict.jpg", r_output_predict_image)
            coarse_mask_res = coarse_mask.squeeze()
            coarse_mask_res[y1:y2, x1:x2] = focus_pred_res
            coarse_mask_res[coarse_mask_res > 0.5] = 1
            coarse_mask_res[coarse_mask_res < 0.5] = 0
            r_output_full_image = draw_mask_on_image(self.rgb_original_image, coarse_mask_res)
            cv2.imwrite("/tmp/r_output_full.jpg", r_output_full_image)

            prev_mask_res = prev_mask.squeeze()
            prev_mask_res[y1:y2, x1:x2] = focus_pred_res
            prev_mask_res_img = draw_mask_on_image(self.rgb_original_image, prev_mask_res)
            cv2.imwrite("/tmp/r_output_pre_mask.jpg", prev_mask_res_img)
        # todo why all sigmoid
        # coarse_mask = torch.sigmoid(coarse_mask)
        self.prev_prediction = prev_mask
        self.transforms[0]._prev_probs = prev_mask.cpu().numpy()
        return prev_mask.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists):
        points_nd = self.get_points_nd(clicks_lists)
        output =  self.net(image_nd, points_nd)
        return output['instances'], output['feature']

    def _get_refine(self, coarse_mask, image, clicks, feature, focus_roi, focus_roi_in_global_roi):
        y1, y2, x1, x2 = focus_roi
        if debug:
            w, h, c = self.rgb_original_image.shape
            r_input_crop_image = np.zeros((w, h))
            r_input_crop_image[y1:y2, x1:x2] = 1
            r_input_crop = draw_mask_on_image(self.rgb_original_image, r_input_crop_image)
            cv2.imwrite("/tmp/r_input_crop.jpg", r_input_crop)
        image_focus = image[:, :, y1:y2, x1:x2]
        image_focus = F.interpolate(image_focus,(self.crop_l, self.crop_l), mode='bilinear', align_corners=True)
        mask_focus = coarse_mask
        points_nd = self.get_points_nd_inbbox(clicks,y1,y2,x1,x2)
        y1, y2, x1, x2 = focus_roi_in_global_roi
        logger.info("refine network focus_roi {}, focus_roi_in_global_roi {}", focus_roi, focus_roi_in_global_roi)
        roi = torch.tensor([0, x1, y1, x2, y2]).unsqueeze(0).float().to(image_focus.device)

        pred = self.net.refine(image_focus, points_nd, feature, mask_focus, roi) #['instances_refined']
        focus_coarse, focus_refined = pred['instances_coarse'], pred['instances_refined']
        return focus_refined


    def mapp_roi(self, focus_roi, global_roi):
        yg1,yg2,xg1,xg2 = global_roi
        hg,wg = yg2-yg1, xg2-xg1
        yf1,yf2,xf1,xf2 = focus_roi

        yf1_n = (yf1-yg1) * (self.crop_l/hg)
        yf2_n = (yf2-yg1) * (self.crop_l/hg)
        xf1_n = (xf1-xg1) * (self.crop_l/wg)
        xf2_n = (xf2-xg1) * (self.crop_l/wg)

        yf1_n = max(yf1_n,0)
        yf2_n = min(yf2_n,self.crop_l)
        xf1_n = max(xf1_n,0)
        xf2_n = min(xf2_n,self.crop_l)
        return (yf1_n,yf2_n,xf1_n,xf2_n)

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)

        return image_nd, clicks_lists

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_points_nd_inbbox(self, clicks_list, y1,y2,x1,x2):
        total_clicks = []
        num_pos = sum(x.is_positive for x in clicks_list)
        num_neg =len(clicks_list) - num_pos 
        num_max_points = max(num_pos, num_neg)
        num_max_points = max(1, num_max_points)
        pos_clicks, neg_clicks = [],[]
        for click in clicks_list:
            flag,y,x,index = click.is_positive, click.coords[0],click.coords[1], 0
            y,x = map_point_in_bbox(y,x,y1,y2,x1,x2,self.crop_l)
            if flag:
                pos_clicks.append( (y,x,index))
            else:
                neg_clicks.append( (y,x,index) )

        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)
        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
        print('set')
