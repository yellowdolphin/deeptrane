# mAP from refactored siim2021 pipeline1/utils/map_func.py (using gt_labels rather than gt_scores)
import json
import os
import shutil
import math
import numpy as np
import torchmetrics as tm
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn


def get_dist_sync_fn(xm):
    "Enables xla syncronization of torchmetrics classes"
    if xm.xrt_world_size() <= 1:
        return None

    def dist_sync_fn(data, group=None):
        assert group is None, f'group not None: {group}'
        local_data = data
        ordinal = xm.get_ordinal()
        assert isinstance(data, torch.Tensor), f'{type(data)} is not a Tensor'
        data = xm.mesh_reduce('metric_sync', data, list)
        assert type(data) is list
        assert len(data) == xm.xrt_world_size()
        assert isinstance(data[ordinal], torch.Tensor), f'{type(data[ordinal])} is not a Tensor'
        assert data[ordinal].dtype == local_data.dtype, f'dtype changed to {data[ordinal].dtype}'
        assert data[ordinal].shape == local_data.shape, f'shape changed to {data[ordinal].shape}'
        return data

    return dist_sync_fn


def reduce(values):
    if isinstance(values, torch.Tensor):
        return torch.mean(values)
    return sum(values) / len(values)


class AverageMeter(object):
    '''Computes and stores the average and current value'''

    def __init__(self, xm):
        self.xm = xm  # allow overload at runtime
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def average(self):
        eps = 1e-14
        reduced_sum = self.xm.mesh_reduce('meter_sum', self.sum, sum)
        reduced_count = self.xm.mesh_reduce('meter_count', self.count, sum)
        return reduced_sum / (reduced_count + eps)

    @property
    def current(self):
        # current value, averaged over devices (and minibatch)
        return self.xm.mesh_reduce('meter_val', self.val, reduce)


def log_average_miss_rate(prec, rec, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index as min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


# Calculate the AP given the recall and precision array
#    1st) We compute a version of the measured precision/recall curve with
#         precision monotonically decreasing
#    2nd) We compute the AP as the area under this curve by numerical integration.
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)   # insert 0.0 at begining of list
    rec.append(1.0)      # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)     # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def map_calc(det_data, gt_data, xm=None):
    """
     Create a ".temp_files/" and "output/" directory
    """
    TEMP_FILES_PATH = ".temp_files"
    if hasattr(xm, 'get_ordinal'):
        TEMP_FILES_PATH += str(xm.get_ordinal())
    os.makedirs(TEMP_FILES_PATH, exist_ok=True)

    MINOVERLAP = 0.5

    """
        Load each of the gt-results files into a temporary ".json" file.
    """
    gt_counter_per_class = {}
    counter_images_per_class = {}
    gt_files = []
    for frame, frame_data in gt_data.items():
        file_id = str(frame)
        # create ground-truth dictionary
        bounding_boxes = frame_data
        already_seen_classes = []
        for box in bounding_boxes:
            class_name = box['class_name']

            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1

            if class_name not in already_seen_classes:
                if class_name in counter_images_per_class:
                    counter_images_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)
        new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    """
     detection-results
        Load each of the detection-results files into a temporary ".json" file.
    """
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for frame, frame_data in det_data.items():
            file_id = str(frame)
            for box in frame_data:
                tmp_class_name, confidence, bbox = box['class_name'], box['confidence'], box['bbox']
                if tmp_class_name == class_name:
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id,
                                           "bbox": bbox})
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
     Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_list = []
    count_true_positives = {}
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        """
         Load detection-results of that class
        """
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))

        """
         Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [float(x) for x in detection["bbox"].split()]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
                          min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                             (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # assign detection as true positive/don't care/false positive
            # set minimum overlap
            min_overlap = MINOVERLAP
            # if specific_iou_flagged:
            #     if class_name in specific_iou_classes:
            #         index = specific_iou_classes.index(class_name)
            #         min_overlap = float(iou_list[index])
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap

        ap_list.append(ap)

    mAP = sum_AP / n_classes

    # remove the temp_files directory
    shutil.rmtree(TEMP_FILES_PATH)

    return mAP, ap_list


def val_map(gt_labels, pred_scores, score_thresh=1e-5, classwise=False, xm=None):
    gt_labels = gt_labels.numpy() if hasattr(gt_labels, 'numpy') else gt_labels
    multilabel = gt_labels.ndim > 1
    pred_scores = pred_scores.numpy() if hasattr(pred_scores, 'numpy') else pred_scores
    bbox = "0 0 1 1"
    det_data = {}
    gt_data = {}

    for cc, (gt_label, pred_score) in enumerate(zip(gt_labels, pred_scores)):
        gt_data[cc] = ([{"class_name": str(gt_label), "bbox": bbox, "used": False}] if not multilabel else
                       [{"class_name": str(i),        "bbox": bbox, "used": False}
                            for i, s in enumerate(gt_label) if s > 0])

        det_data[cc] = [{"class_name": str(i),        "bbox": bbox, "confidence": str(s)}
                            for i, s in enumerate(pred_score) if s > score_thresh]

    map, ap_list = map_calc(det_data, gt_data, xm)

    return ap_list if classwise else map


#def multiclass_average_precision_score(y_true, y_score):
#    """Return class-mean of single-class AP scores (obsolete)
#
#    Same as average_precision_score(y_true, y_score)"""
#    n_classes = y_true.shape[1]
#    class_aps = [average_precision_score(y_true[:, i], y_score[:, i]) for i in range(n_classes)]
#    return np.mean(class_aps)


# mAP metric implementations --------------------------------------------------

def vin_summarize(self):
    # From detectron2 notebook (https://www.kaggle.com/corochann/vinbigdata-detectron2-train)
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.5f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s

    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])  # orig
        #stats[2] = _summarize(1, iouThr=.4, maxDets=self.params.maxDets[2])  # mAP@0.40
        stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
        return stats

    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=.75)
        stats[3] = _summarize(1, maxDets=20, areaRng='medium')
        stats[4] = _summarize(1, maxDets=20, areaRng='large')
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=.75)
        stats[8] = _summarize(0, maxDets=20, areaRng='medium')
        stats[9] = _summarize(0, maxDets=20, areaRng='large')
        return stats

    if not self.eval:
        raise Exception('Please run accumulate() first')
    iouType = self.params.iouType
    if iouType == 'segm' or iouType == 'bbox':
        summarize = _summarizeDets
    elif iouType == 'keypoints':
        summarize = _summarizeKps
    self.stats = summarize()


class VinBigDataEval:
    """Helper class for calculating the competition metric.

    You should remove the duplicated annoatations from the `true_df` dataframe
    before using this script. Otherwise it may give incorrect results.

        >>> vineval = VinBigDataEval(valid_df)
        >>> cocoEvalResults = vineval.evaluate(pred_df)

    Arguments:
        true_df: pd.DataFrame Clean (no duplication) Training/Validating dataframe.

    Authors:
        Peter (https://kaggle.com/pestipeti)

    See:
        https://www.kaggle.com/pestipeti/competition-metric-map-0-4

    Returns: None
    """
    def __init__(self, true_df):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        
        # Monkey patch: print mAP's with more digits and custom IoU ranges
        print("HACKING: overriding COCOeval.summarize = vin_summarize...")
        COCOeval.summarize = vin_summarize

        self.true_df = true_df
        self.image_ids = true_df["image_id"].unique()
        self.annotations = {
            "type": "instances",
            "images": self.__gen_images(self.image_ids),
            "categories": self.__gen_categories(self.true_df),
            "annotations": self.__gen_annotations(self.true_df, self.image_ids)
        }

        self.predictions = {
            "images": self.annotations["images"].copy(),
            "categories": self.annotations["categories"].copy(),
            "annotations": None
        }


    def __gen_categories(self, df):
        print("Generating category data...")

        if "class_name" not in df.columns:
            df["class_name"] = df["class_id"]

        cats = df[["class_name", "class_id"]]
        cats = cats.drop_duplicates().sort_values(by='class_id').values

        results = []

        for cat in cats:
            results.append({
                "id": cat[1],
                "name": cat[0],
                "supercategory": "none",
            })

        return results

    def __gen_images(self, image_ids):
        print("Generating image data...")
        results = []

        for idx, image_id in enumerate(image_ids):

            # Add image identification.
            results.append({
                "id": idx,
            })

        return results

    def __gen_annotations(self, df, image_ids):
        print("Generating annotation data...")
        k = 0
        results = []

        for idx, image_id in enumerate(image_ids):

            # Add image annotations
            for i, row in df[df["image_id"] == image_id].iterrows():

                results.append({
                    "id": k,
                    "image_id": idx,
                    "category_id": row["class_id"],
                    # COCO bbox has xywh format
                    "bbox": np.array([
                        row["x_min"],
                        row["y_min"],
                        row["x_max"] - row["x_min"],
                        row["y_max"] - row["y_min"]]
                    ),
                    "segmentation": [],
                    "ignore": 0,
                    "area": (row["x_max"] - row["x_min"]) * (row["y_max"] - row["y_min"]),
                    "iscrowd": 0,
                })

                k += 1

        return results

    def __decode_prediction_string(self, pred_str):
        data = list(map(float, pred_str.split(" ")))
        data = np.array(data)

        return data.reshape(-1, 6)

    def __gen_predictions(self, df, image_ids):
        print("Generating prediction data...")
        k = 0
        results = []

        for i, row in df.iterrows():

            image_id = row["image_id"]
            preds = self.__decode_prediction_string(row["PredictionString"])

            for j, pred in enumerate(preds):

                results.append({
                    "id": k,
                    "image_id": int(np.where(image_ids == image_id)[0]),
                    "category_id": int(pred[0]),
                    # COCO bbox has xywh format
                    "bbox": np.array([
                        pred[2], pred[3], pred[4] - pred[2], pred[5] - pred[3]
                    ]),
                    "segmentation": [],
                    "ignore": 0,
                    "area": (pred[4] - pred[2]) * (pred[5] - pred[3]),
                    "iscrowd": 0,
                    "score": pred[1]
                })

                k += 1

        return results

    def evaluate(self, pred_df, n_imgs=-1):
        """Evaluating your results

        Arguments:
            pred_df: pd.DataFrame your predicted results in the
                     competition output format.

            n_imgs:  int Number of images use for calculating the
                     result.All of the images if `n_imgs` <= 0

        Returns:
            COCOEval object
        """

        if pred_df is not None:
            self.predictions["annotations"] = self.__gen_predictions(pred_df, self.image_ids)

        coco_ds = COCO()
        coco_ds.dataset = self.annotations
        coco_ds.createIndex()

        coco_dt = COCO()
        coco_dt.dataset = self.predictions
        coco_dt.createIndex()

        imgIds = sorted(coco_ds.getImgIds())

        if n_imgs > 0:
            imgIds = np.random.choice(imgIds, n_imgs)

        cocoEval = COCOeval(coco_ds, coco_dt, 'bbox')
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.useCats = True
        cocoEval.params.iouType = "bbox"
        cocoEval.params.iouThrs = np.arange(0.5, 1, 0.05)
        # VinChestXray competition metric
        #cocoEval.params.iouThrs = np.array([0.4])  # only saves 4 seconds
        #cocoEval.params.iouThrs = np.arange(0.4, 1, 0.05)

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        # Compute per-category AP (multiclass only)
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = cocoEval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        #                    (1,   101,    14,  4,          3)
        if precisions.shape[2] > 1:
            print(f"AP@0.40 for classes 0...{precisions.shape[2]}")
            for class_id in range(precisions.shape[2]):
                # is recall the AUC integration variable?
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                precision = precisions[0, :, class_id, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float("nan")
                print(f"{ap:.5f}")

        return cocoEval


# Metrics for image recognition -----------------------------------------------

class NegativeRate(nn.Module):
    def __init__(self, negative_class=0, name='neg_rate'):
        super().__init__()
        self.negative_class = negative_class
        self.needs_top5 = False
        self.needs_scores = False
        self.__name__ = name

    def forward(self, labels, preds):
        preds = preds[:, 0] if preds.ndim > 1 else preds
        return (preds == self.negative_class).sum() / len(preds)


class MAP(nn.Module):
    def __init__(self, xm, k=0, name='mAP'):
        super().__init__()
        self.xm = xm
        self.k = k
        self.needs_scores = True
        self.__name__ = name


    def forward(self, labels, features):
        m = np.matmul(features, np.transpose(features))  # similarity matrix
        for i in range(features.shape[0]):
            m[i,i] = -1000.0  # avoid self-reckognition
        predict_sorted = np.argsort(m, axis=-1)[:,::-1]  # most similar other examples

        #thresholds = np.arange(0.4, 0.3, -0.02)
        thresholds = np.arange(1, 0, -0.05)
        map5_list = []
        for threshold in thresholds:
            top5s = []
            for l, scores, indices in zip(labels, m, predict_sorted):  # (2799,) int64, (2799, 2799) float32, (2799, 2799) int64
                top5_labels, top5_scores = self.get_top5(scores, indices, labels, threshold)
                top5s.append(np.array(top5_labels))
            map5_list.append((threshold, self.mapk(labels, top5s)))
        map5_list = list(sorted(map5_list, key=lambda x: x[1], reverse=True))
        best_thres = map5_list[0][0]
        best_score = map5_list[0][1]
        self.xm.master_print(f"best_thres: {best_thres:.2f}")

        return best_score


    def get_top5(self, scores, indices, labels, threshold):
        used = set()
        ret_labels = []
        ret_scores = []

        for index in indices:
            l = labels[index]
            s = scores[index]
            if l in used:
                continue

            if 0 not in used and s < threshold:
                used.add(0)
                ret_labels.append(0)
                ret_scores.append(-2.0)
            if l in used:
                continue

            used.add(l)
            ret_labels.append(l)
            ret_scores.append(s)
            if len(ret_labels) >= self.k:
                break
        return ret_labels[:5], ret_scores[:5]


    def mapk(self, labels, preds):
        return np.mean([self.apk(l, p) for l, p in zip(labels, preds)])


    def apk(self, labels, preds):
        k = self.k
        if not labels:
            return 0.0
        if len(preds) > k:
                preds = preds[:k]

        if not hasattr(labels, '__len__') or len(labels) == 1:
            for i, p in enumerate(preds):
                if p == labels:
                    return 1.0 / (i + 1)
            return 0.0

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(preds):
            if p in labels and p not in preds[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(len(labels), k)


class EmbeddingAveragePrecision(tm.Metric):

    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def __init__(self, xm, k=0):
        super().__init__()
        self.xm = xm
        self.k = k
        self.add_state('embeddings', default=torch.FloatTensor(), dist_reduce_fx='cat')
        self.add_state('labels', default=torch.LongTensor(), dist_reduce_fx='cat')


    def update(self, labels: torch.LongTensor, embeddings: torch.FloatTensor):
        #labels, embeddings = self._input_format(labels, embeddings)  # outdated doc?
        assert embeddings.size(0) == labels.size(0), f'len mismatch between labels ' \
            f'({labels.size(0)}) and embeddings ({embeddings.size(0)})'
        self.embeddings = torch.cat([self.embeddings, embeddings])
        self.labels = torch.cat([self.labels, labels])


    def compute(self):
        labels, embeddings = self.labels, self.embeddings
        m = torch.matmul(embeddings, embeddings.T)  # similarity matrix
        m.diagonal = -1000.0  # penalize self-reckognition
        predict_sorted = torch.argsort(m, dim=-1)[:,::-1]  # most similar other examples

        map5_list = []
        for threshold in np.arange(1, 0, -0.05):
            top5s = []
            for l, scores, indices in zip(labels, m, predict_sorted):  # (2799,) int64, (2799, 2799) float32, (2799, 2799) int64
                top5_labels = self.get_top5(scores, indices, labels, threshold)
                top5s.append(top5_labels)
            map5_list.append((threshold, self.mapk(labels, top5s)))
        map5_list = list(sorted(map5_list, key=lambda x: x[1], reverse=True))
        best_thres = map5_list[0][0]
        best_score = map5_list[0][1]
        self.xm.master_print(f"best_thres: {best_thres:.2f}")

        return torch.tensor(best_score)


    def get_top5(self, scores, indices, labels, threshold):
        # TODO: vectorize, use torch functions
        used = set()
        ret_labels = []

        for index in indices:
            l = labels[index]
            s = scores[index]
            if l in used:
                continue

            if 0 not in used and s < threshold:
                used.add(0)
                ret_labels.append(0)
            if l in used:
                continue

            used.add(l)
            ret_labels.append(l)
            if len(ret_labels) >= self.k:
                break
        return torch.LongTensor(ret_labels[:5])


    def mapk(self, labels, preds):
        self.xm.master_print("mapk:", labels.shape, preds.shape)
        return torch.mean([self.apk(l, p) for l, p in zip(labels, preds)])


    def apk(self, labels, preds):
        self.xm.master_print("apk:", labels.shape, preds.shape)
        k = self.k
        if not labels:
            return 0.0
        if len(preds) > k:
                preds = preds[:k]

        if not hasattr(labels, '__len__') or len(labels) == 1:
            for i, p in enumerate(preds):
                if p == labels:
                    return 1.0 / (i + 1)
            return 0.0

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(preds):
            if p in labels and p not in preds[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(len(labels), k)