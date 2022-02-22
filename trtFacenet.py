import csv
import logging
import os
import time

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


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


class TRTFacenet(object):
    def _load_engine(self):
        TRTbin = self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, model):
        self.model = model
        self.engine = self._load_engine()
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        # self.inference_fn = inference

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def paddingImage(self, img):
        WIDTH = img.shape[1]
        HEIGHT = img.shape[0]

        maxData = max(WIDTH, HEIGHT)
        img = cv2.copyMakeBorder(img, int((maxData - HEIGHT) / 2), int((maxData - HEIGHT) / 2),
                                 int((maxData - WIDTH) / 2), int((maxData - WIDTH) / 2), borderType=cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255))
        img = cv2.resize(img, (160, 160))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, :]
        return img

    def facenetInference(self, context, bindings, inputs, outputs, stream):
        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            for out in outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
            stream.synchronize()
            return [out['host'] for out in outputs]

    def detect(self, img):
        img = self.paddingImage(img)
        img_pre_processed = img.astype(np.float32)
        self.inputs[0]['host'] = np.ascontiguousarray(img_pre_processed)
        facenet_output = self.facenetInference(context=self.context, bindings=self.bindings, inputs=self.inputs,
                                        outputs=self.outputs, stream=self.stream)
        return facenet_output