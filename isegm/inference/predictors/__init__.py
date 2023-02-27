from .baseline import BaselinePredictor
from .focalclick import FocalPredictor
from isegm.inference.transforms import ZoomIn

def get_predictor(net, brs_mode, device,
                  infer_size = 256, focus_crop_r= 1.4, with_flip=False,
                  zoom_in_params=None, net_clicks_limit=None, max_size=None):

    zoom_in = None if zoom_in_params is None else ZoomIn(**zoom_in_params)
    # todo debug purpose
    brs_mode = 'FocalClick'

    if isinstance(net, (list, tuple)):
        assert brs_mode == 'NoBRS', "Multi-stage models support only NoBRS mode."

    if brs_mode in ('NoBRS', 'Baseline'):
        predictor = BaselinePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip,
                                      infer_size=infer_size, net_clicks_limit=net_clicks_limit,
                                      max_size=max_size)
    elif brs_mode == 'FocalClick':
        predictor = FocalPredictor(net, device, zoom_in=zoom_in, with_flip=with_flip,
                                   infer_size=infer_size, focus_crop_r=focus_crop_r,
                                   net_clicks_limit=net_clicks_limit, max_size=max_size)
    else:
        raise NotImplementedError

    return predictor
