import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from metadata import get_image_ids


def get_bg_image_ids(df, bbox_col='boxes', id_col='image_id'):
    "Return array of image_ids with no bboxes"
    is_bg = df[bbox_col].isna() | df[bbox_col].apply(bool)
    return df.loc[is_bg, id_col].values


def get_bbox_xywh(df):
    """Returns list of lists [x_min, y_min, width, height]

    df: bbox xyxy labels in separate cols (x_min, y_min, x_max, y_max)"""
    xxyy = df[['x_min', 'x_max', 'y_min', 'y_max']].values
    x = xxyy[:, 0]
    y = xxyy[:, 2]
    w = xxyy[:, 1] - xxyy[:, 0]
    h = xxyy[:, 3] - xxyy[:, 2]
    return np.stack([x, y, w, h], axis=1).tolist()


def get_bbox_ccwh(df):
    """Returns list of lists [x_center, y_center, width, height]

    df: bbox xyxy labels in separate cols (x_min, y_min, x_max, y_max)"""
    xxyy = df[['x_min', 'x_max', 'y_min', 'y_max']].values
    x = xxyy[:, :2].mean(axis=1)
    y = xxyy[:, 2:].mean(axis=1)
    w = xxyy[:, 1] - xxyy[:, 0]
    h = xxyy[:, 3] - xxyy[:, 2]
    return np.stack([x, y, w, h], axis=1).tolist()


def get_bbox_xyxy(df):
    """Returns list of lists [x_min, y_min, x_max, y_max]

    df: bbox xyxy labels in separate cols (x_min, y_min, x_max, y_max)"""
    xyxy = df[['x_min', 'y_min', 'x_max', 'y_max']].values
    return xyxy.tolist()


def get_prediction_string(det_result, prec_b=4, prec_s=7, scale_x=1, scale_y=1):
    prediction_string = []
    for class_id, class_preds in enumerate(det_result):
        if len(class_preds) == 0: continue
        s = ' '.join([(f'{class_id} {s:.{prec_s}} '
                       f'{x1 * scale_x:.{prec_b}f} {y1 * scale_y:.{prec_b}f} '
                       f'{x2 * scale_x:.{prec_b}f} {y2 * scale_y:.{prec_b}f}')
                      for x1, y1, x2, y2, s in class_preds])
        prediction_string.append(s)
    return ' '.join(prediction_string) if prediction_string else '14 1 0 0 1 1'


def xywh2xyxy(x, y, w, h, W=1, H=1):
    "Floatify, scale and clip normalized xywh coordinates"
    x0 = max(0, min(1, (float(x) - 0.5 * float(w)))) * W
    x1 = max(0, min(1, (float(x) + 0.5 * float(w)))) * W
    y0 = max(0, min(1, (float(y) - 0.5 * float(h)))) * H
    y1 = max(0, min(1, (float(y) + 0.5 * float(h)))) * H
    return x0, y0, x1, y1


def txt2prediction_string(filename, original_size=None, prec=4):
    "Convert YOLOv5 txt file into a VinBigData prediction string"
    H, W = original_size or (1, 1)
    preds = []

    with open(filename, 'r') as fp:
        lines = fp.readlines()

    if len(lines) == 0:
        return '14 1 0 0 1 1'   # code for no finding

    for line in lines:
        class_id, x, y, w, h, score = line.split()
        x0, y0, x1, y1 = xywh2xyxy(x, y, w, h, W, H)
        preds.append(f'{class_id} {score} {x0:.{prec}f} {y0:.{prec}f} {x1:.{prec}f} {y1:.{prec}f}')
    return ' '.join(preds)


def txt2det_result(filename, n_classes, original_size=None, missing_ok=True):
    """Convert YOLOv5 txt file into iterable det_result[class_id] = class_preds

    class_preds = [[x0, y0, x1, y1, score], ...]
    """
    H, W = original_size or (1, 1)
    det_result = [[] for _ in range(n_classes)]

    if not os.path.exists(filename) and missing_ok:
        det_result[14].append([0, 0, 1, 1, 1.0])  # code for no finding
        return det_result

    with open(filename, 'r') as fp:
        lines = fp.readlines()

    if len(lines) == 0:
        det_result[14].append([0, 0, 1, 1, 1.0])  # code for no finding

    for line in lines:
        class_id, x, y, w, h, score = line.split()
        class_id = int(class_id)
        score = float(score)
        x0, y0, x1, y1 = xywh2xyxy(x, y, w, h, W, H)
        class_pred = [x0, y0, x1, y1, score]
        det_result[class_id].append(class_pred)

    return det_result


def hub_results2prediction_string(results, prec=4):
    "Generate prediction_string from a single-image result (list of torch.tensor)"
    # results.xyxyn = [img_0_results, img_1_results, ...], assume only one image here.
    t = results.xyxyn[0]
    return ' '.join([f'{c:.0f} {s:.6f} {x0:.{prec}f} {y0:.{prec}f} {x1:.{prec}f} {y1:.{prec}f}'
                     for x0, y0, x1, y1, s, c in t])


# --- My BBoxes class -----------------------------------------------------------------------
class BBox():
    def __init__(self, iterable=(0, 0, 1, 1), prec=0):
        if isinstance(iterable, self.__class__):
            self.xyxy = iterable.xyxy
            self.prec = prec or iterable.prec
        elif isinstance(iterable, str):
            self.xyxy = np.array([float(s) for s in iterable.split()])
        else:
            self.xyxy = np.array(iterable)

        if len(self.xyxy) != 4:
            msg = "BBox() takes iterable of len 4, but len is"
            raise ValueError(f"{msg} {len(self.xyxy)}")
        self.prec = prec or 0

    def __str__(self):
        return ' '.join(f'{x:.{self.prec}f}' for x in self.xyxy)

    def __repr__(self):
        prec_kwarg = '' if self.prec == 0 else f', prec={self.prec}'
        return f'BBox({list(self.xyxy)}{prec_kwarg})'

    def __add__(self, b):
        if isinstance(b, self.__class__):
            return BBox(self.xyxy + b.xyxy)
        return BBox(self.xyxy + b, prec=self.prec)

    def __sub__(self, b):
        if isinstance(b, self.__class__):
            return BBox(self.xyxy - b.xyxy)
        return BBox(self.xyxy - b, prec=self.prec)

    def __radd__(self, b):
        return BBox(self.xyxy + b, prec=self.prec)

    def __rsub__(self, b):
        return BBox(b - self.xyxy, prec=self.prec)

    def __iadd__(self, b):
        return BBox(self.xyxy + b, prec=self.prec)

    def __isub__(self, b):
        return BBox(self.xyxy - b, prec=self.prec)

    def __mul__(self, b):
        return BBox(self.xyxy * b, prec=self.prec)

    def __truediv__(self, b):
        return BBox(self.xyxy / b, prec=self.prec)

    def __itruediv__(self, b):
        return BBox(self.xyxy / b, prec=self.prec)

    def __rmul__(self, b):
        return BBox(self.xyxy * b, prec=self.prec)

    def __imul__(self, b):
        return BBox(self.xyxy * b, prec=self.prec)

    def __eq__(self, b):
        if isinstance(b, self.__class__): return (self.xyxy == b.xyxy).all()
        return self == BBox(b)

    def __neq__(self, b):
        return not self.__eq__(self, b)


class BBoxes(list):
    "List-like container for BBox instances with some array operators."
    def __init__(self, iterable=(), prec=None):
        if isinstance(iterable, BBox):
            iterable = [iterable]
        len_is_4 = hasattr(iterable, '__len__') and (len(iterable) == 4)
        if len_is_4 and not any(isinstance(x, BBox) or hasattr(x, '__iter__') for x in iterable):
            iterable = [BBox(iterable)]
        iterable = [BBox(x, prec) for x in iterable]
        super().__init__(iterable)
        self.prec = prec

    def __repr__(self):
        return f'BBoxes({[bbox for bbox in self]})'

    def __str__(self):
        return f'{[str(bbox) for bbox in self]}'

    def __add__(self, other):
        if isinstance(other, BBoxes):
            return BBoxes([a + b for a, b in zip(self, other)])
        return BBoxes([a + other for a in self])

    def __sub__(self, other):
        if isinstance(other, BBoxes):
            return BBoxes([a - b for a, b in zip(self, other)])
        return BBoxes([a - other for a in self])

    def __radd__(self, b):
        return BBoxes([a + b for a in self])

    def __rsub__(self, b):
        return BBoxes([b - a for a in self])

    def __iadd__(self, b):
        return BBoxes([a + b for a in self])

    def __isub__(self, b):
        return BBoxes([a - b for a in self])

    def __mul__(self, b):
        return BBoxes([a * b for a in self])

    def __truediv__(self, b):
        return BBoxes([a / b for a in self])

    def __rmul__(self, b):
        return BBoxes([a * b for a in self])

    def __imul__(self, b):
        return BBoxes([a * b for a in self])

    def __itruediv__(self, b):
        return BBoxes([a / b for a in self])

    @property
    def as_strings(self):
        "Return a list of str(BBox)"
        return [str(bbox) for bbox in self]

    def annotation_string(self, class_ids, scores, thresh=0, score_prec=5):
        "Combine BBoxes, class_ids and scores in one str."
        if isinstance(class_ids, int):
            class_ids = [class_ids] * len(self)
        msg = f"len differ: {len(self)}, {len(class_ids)}, {len(scores)}"
        assert len(self) == len(class_ids) == len(scores), msg
        if hasattr(thresh, '__iter__'):
            thresh = [t for t in thresh]
            msg = f"len differ: {len(self)}, {len(thresh)}"
            assert len(thresh) == len(self), msg
            thresholds = thresh
        else:
            thresholds = [thresh] * len(self)

        zipped = zip(class_ids, scores, self, thresholds)
        ann_strings = [f'{c} {s:.{score_prec}f} {b}' for c, s, b, t in zipped if s >= t]
        return ' '.join(ann_strings)

    @classmethod
    def from_annotation_string(cls, s: str, class_id=None):
        "Create BBoxes from `s`, optionally filtering for `class_id`"
        a = np.array(s.split()).reshape(-1, 6)
        class_ids, xyxys = a[:, 0], a[:, 2:]
        if class_id:
            return cls([BBox(xyxy.astype(float)) for i, xyxy in zip(class_ids, xyxys) if i == class_id])
        return cls([BBox(xyxy.astype(float)) for i, xyxy in zip(class_ids, xyxys)])


def split_siimcovid_boxes(df, cfg):
    "Split bboxes (dict, no class_ids), one row per bbox"

    labels = []
    for i, row in enumerate(df.itertuples()):
        for box in eval(row[cfg.bbox_col]):
            labels.append({
                'image_id': row.image_id,
                #'class_id': 0,
                'fold': row.fold,
                'image_path': row.image_path,
                'x_min': max(0, box['x'] / row.width),
                'y_min': box['y'] / row.height,
                'x_max': (box['x'] + box['width']) / row.width,
                'y_max': (box['y'] + box['height']) / row.height,
            })
    labels = pd.DataFrame(labels)

    # Check dims
    assert labels.x_min.min() >= 0
    assert labels.y_min.min() >= 0
    assert labels.x_max.max() <= 1
    assert labels.y_max.max() <= 1

    print("Annotations after dropping background class:", labels.shape[0])

    # Get normalized coords
    xxyy_cols = ['x_min', 'x_max', 'y_min', 'y_max']
    xxyy = labels[xxyy_cols].values
    labels['x'] = xxyy[:, :2].mean(axis=1)
    labels['y'] = xxyy[:, 2:].mean(axis=1)
    labels['w'] = xxyy[:, 1] - xxyy[:, 0]
    labels['h'] = xxyy[:, 3] - xxyy[:, 2]
    labels.drop(columns=xxyy_cols, inplace=True)

    return labels


def write_dataset_yaml(cfg, path):
    "Write YOLOv5 dataset yaml"
    path = Path(path)
    yaml_file = path / 'dataset.yaml'

    with open(yaml_file, 'w') as fp:
        yaml.dump(dict(
            train = str(path / 'images' / 'train'),
            val   = str(path / 'images' / 'valid'),
            test  = cfg.test_images_path,
            nc    = cfg.n_classes,
            names = cfg.classes,
        ), fp, default_flow_style=False)


def write_yolov5_labels(metadata, split, path):
    for image_id in get_image_ids(metadata, split):
        d = metadata.loc[metadata.image_id == image_id, 'class_id x y w h'.split()]
        d.to_csv(path / 'labels' / split / f'{image_id}.txt', sep=' ',
                 header=False, index=False, float_format='%.6f')
