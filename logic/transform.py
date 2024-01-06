import cv2 as cv
import numpy as np


class Step:
    def __init__(self): ...

    def apply(self): ...


class Translate(Step):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def apply(self):
        mat = np.array([
            [1, 0, self.x],
            [0, 1, self.y],
            [0, 0, 1]
        ], dtype=np.float32)
        return mat


class Rotation(Step):
    def __init__(self, r):
        self.r = r

    def apply(self):
        _rad = self.r * (np.pi / 180)
        mat = np.array([
            [np.cos(_rad), -np.sin(_rad), 0],
            [np.sin(_rad), np.cos(_rad), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        return mat


class Scale(Step):
    def __init__(self, s):
        self.s = s

    def apply(self):
        mat = np.array([
            [self.s, 0, 0],
            [0, self.s, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        return mat


class Shear(Step):
    def __init__(self, hx, hy):
        self.hx = hx
        self.hy = hy

    def apply(self):
        mat = np.array([
            [1, self.hx, 0],
            [self.hy, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        return mat


class Perspective(Step):
    def __init__(self, original, new):
        self.original = original
        self.new = new

    def apply(self):
        _A = []
        _b = []
        for i in range(len(self.original)):
            _ori = self.original[i]
            _new = self.new[i]
            print(_ori, _new)
            # Append A
            _A.append([_ori[0], _ori[1], 1, 0, 0, 0, -
                      _new[0]*_ori[0], -_new[0]*_ori[1]])
            _A.append([0, 0, 0, _ori[0], _ori[1], 1, -
                      _new[1]*_ori[0], -_new[1]*_ori[1]])
            # Append b
            _b.append(_new[0])
            _b.append(_new[1])
        _h = np.matmul(np.linalg.inv(np.array(_A)), np.array(_b))
        mat = np.array([
            [_h[0], _h[1], _h[2]],
            [_h[3], _h[4], _h[5]],
            [_h[6], _h[7], 1]
        ], dtype=np.float32)
        return mat


class Transform:
    def __init__(self):
        self.steps: list[Step] = []
        self.mat: np.ndarray = None
        self.pmat: np.ndarray = None

    def add_step(self, step: Step):
        self.steps.append(step)

    def get_transform(self):
        _mat = np.identity(3, dtype=np.float32)
        for step in self.steps:
            if not isinstance(step, Perspective):
                _mat = np.matmul(_mat, step.apply())
        self.mat = _mat
        return _mat

    def get_perspective(self):
        _mat = np.identity(3, dtype=np.float32)
        for step in self.steps:
            if isinstance(step, Perspective):
                _mat = np.matmul(_mat, step.apply())
        self.pmat = _mat
        return _mat

    def apply_transform(self, img: np.ndarray):
        _mat = self.mat[:2, :]
        return cv.warpAffine(img, _mat, dsize=(img.shape[1], img.shape[0]))

    def apply_perspective(self, img: np.ndarray):
        _mat = self.pmat
        return cv.warpPerspective(img, _mat, dsize=(img.shape[1], img.shape[0]))
