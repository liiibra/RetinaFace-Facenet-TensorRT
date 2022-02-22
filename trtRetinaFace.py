import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from math import ceil
from itertools import product as product


class PriorBox(object):
    def __init__(self, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        anchors = np.array(anchors)
        output = anchors.reshape((-1, 4))
        return output


def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream


def pre_process(img):
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :]
    return img


def inference(context, bindings, inputs, outputs, stream):
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        stream.synchronize()
        return [out['host'] for out in outputs]


def reshape_outputs(res):
    loc = res[0].reshape([1, -1, 4])
    landms = res[1].reshape([1, -1, 10])
    conf = res[2].reshape([1, -1, 2])
    return loc, landms, conf


def decode(loc, priors, variances):
    """
    Decode locations from predictions using priors to undo the encoding we did for offset regression at train time
    :param loc:
    :param priors:
    :param variances:
    :return: decodes bounding box predictions
    """
    boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    landms = np.concatenate((
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
    ), 1)
    return landms


def py_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    score = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr < thresh)[0]
        order = order[inds + 1]

    return keep


def post_process(loc, conf, landms, img_width, img_height):
    scale = np.array([img_width, img_height, img_width, img_height])
    priorbox = PriorBox(image_size=(img_height, img_width))
    prior_data = priorbox.forward()
    variance = [0.1, 0.2]
    boxes = decode(loc.squeeze(0), prior_data, variance)
    boxes = boxes * scale
    scores = conf.squeeze(0)[:, 1]
    landms = decode_landm(landms.squeeze(0), prior_data, variance)
    scale1 = np.array(
        [img_width, img_height, img_width, img_height, img_width, img_height, img_width, img_height, img_width,
         img_height])
    landms = landms * scale1

    inds = np.where(scores > 0.5)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    order = scores.argsort()[::1][:1000]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_nms(dets, 0.5)

    dets = dets[keep, :]
    landms = landms[keep]

    dets = np.concatenate((dets, landms), axis=1)

    return dets


class TRTRetinaFace(object):
    def _load_engine(self):
        TRTbin = self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, model):
        self.model = model
        self.engine = self._load_engine()
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.inference_fn = inference

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def detect(self, img):
        img_raw = img
        WIDTH = img.shape[1]
        HEIGHT = img.shape[0]
        img_processed = pre_process(img)
        self.inputs[0]['host'] = np.ascontiguousarray(img_processed)
        trt_outputs = self.inference_fn(context=self.context, bindings=self.bindings, inputs=self.inputs,
                                        outputs=self.outputs, stream=self.stream)
        loc, landms, conf = reshape_outputs(trt_outputs)
        dets = post_process(loc, conf, landms, WIDTH, HEIGHT)

        image_list = []
        for b in dets:
            if b[4] < 0.5:
                continue
            b = list(map(int, b))
            img_crop = img_raw[b[1]: b[3], b[0]: b[2]]
            image_list.append(img_crop)
        return dets, image_list
