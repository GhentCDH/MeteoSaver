#!/usr/bin/env python
import atexit
import json
import os
import subprocess
import sys
import tempfile

import cv2

# One worker (and so one copy of the model in memory) per process and per set of settings
_clients = {}


class GlmOcrClient:
    '''
    Reads cell images with GLM-OCR through a worker process (glm_ocr_worker.py) that keeps the model in memory.

    Parameters
    --------------
    model : str
        Hugging Face model ID of the fine-tuned GLM-OCR model (e.g. GhentCDH/GLM-OCR-MeteoSaver).
        It is downloaded on first use and then read from the Hugging Face cache.
    python : str
        Python interpreter of an environment with torch>=2.10 and transformers>=5.3. Leave empty to use the current one.
    resize_height : int
        Height in pixels to which the cell images are rescaled before recognition (0 = keep as is).
    batch_size : int
        Number of cell images read at once.
    '''

    def __init__(self, model, python='', resize_height=128, batch_size=64):
        if not model:
            raise ValueError("No GLM-OCR model: set glm_model (a Hugging Face model ID) in the configuration file.")
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glm_ocr_worker.py')
        self.process = subprocess.Popen(
            [python or sys.executable, '-u', worker, '--model', model, '--resize-height', str(resize_height), '--batch-size', str(batch_size)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._receive()  # Wait until the model is loaded

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The GLM-OCR worker stopped unexpectedly (see its error message above). Check glm_python and glm_model in the configuration file: the model is downloaded from Hugging Face "
                               "on first use, which needs an internet connection (and 'hf auth login' for a private model).")
        return json.loads(line)

    def read(self, images):
        '''
        Returns the recognized text for each image (numpy arrays, as cut from the table image), in the same order.
        '''
        if not images:
            return []
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for i, image in enumerate(images):
                paths.append(os.path.join(folder, f'{i}.png'))
                cv2.imwrite(paths[-1], image)
            self.process.stdin.write(json.dumps(paths) + '\n')
            self.process.stdin.flush()
            return self._receive()['texts']

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait()


def get_glm_ocr_client(model, python='', resize_height=128, batch_size=64):
    '''
    Returns the GLM-OCR client for these settings, starting its worker on first use so that the model is loaded once and not per sheet.
    '''
    key = (os.getpid(), model, python, resize_height, batch_size)
    if key not in _clients:
        _clients[key] = GlmOcrClient(model, python, resize_height, batch_size)
        atexit.register(_clients[key].close)
    return _clients[key]
