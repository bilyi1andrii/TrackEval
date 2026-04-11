import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment

from .mot_challenge_2d_box import MotChallenge2DBox
from ..utils import TrackEvalException


class VisDroneMOT(MotChallenge2DBox):
    """
    Class-agnostic VisDrone MOT evaluation over the 5 official MOT classes:
    pedestrian, car, van, truck, bus

    All valid GT are remapped to class 1 in gt.txt.
    Tracker outputs are also expected as class 1.
    Ignore regions are loaded from ignore_regions.txt and used to suppress
    unmatched tracker detections that overlap them.
    """

    @staticmethod
    def get_default_dataset_config():
        cfg = MotChallenge2DBox.get_default_dataset_config()
        cfg["CLASSES_TO_EVAL"] = ["pedestrian"]
        cfg["BENCHMARK"] = "VisDrone-val"
        cfg["DO_PREPROC"] = True
        cfg["IGNORE_IOA_THRESHOLD"] = 0.5
        return cfg

    def __init__(self, config=None):
        super().__init__(config)
        self.valid_classes = ["pedestrian"]
        self.class_list = ["pedestrian"]
        self.class_name_to_class_id = {"pedestrian": 1}
        self.valid_class_numbers = [1]
        self.ignore_ioa_threshold = float(self.config.get("IGNORE_IOA_THRESHOLD", 0.5))

    def _load_ignore_file(self, seq):
        ignore_file = self.config["GT_LOC_FORMAT"].format(
            gt_folder=self.gt_fol, seq=seq
        )
        ignore_file = os.path.join(os.path.dirname(ignore_file), "ignore_regions.txt")

        num_timesteps = self.seq_lengths[seq]
        ignore_regions = [
            np.empty((0, 4), dtype=np.float32) for _ in range(num_timesteps)
        ]

        if not os.path.isfile(ignore_file):
            return ignore_regions

        frame_map = {}
        with open(ignore_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5:
                    continue
                frame_idx = int(row[0])
                box = np.asarray(list(map(float, row[1:5])), dtype=np.float32)
                frame_map.setdefault(frame_idx, []).append(box)

        for t in range(num_timesteps):
            frame_idx = t + 1
            if frame_idx in frame_map:
                ignore_regions[t] = np.asarray(frame_map[frame_idx], dtype=np.float32)

        return ignore_regions

    def _load_raw_file(self, tracker, seq, is_gt):
        raw_data = super()._load_raw_file(tracker, seq, is_gt)

        if is_gt:
            raw_data["gt_crowd_ignore_regions"] = self._load_ignore_file(seq)

        return raw_data

    @staticmethod
    def _xywh_to_xyxy(boxes):
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)
        out = boxes.copy().astype(np.float32)
        out[:, 2] = out[:, 0] + out[:, 2]
        out[:, 3] = out[:, 1] + out[:, 3]
        return out

    @staticmethod
    def _calculate_box_ioa(det_boxes, ignore_boxes):
        """
        IoA = intersection(det, ignore) / area(det)
        det_boxes: N x 4, xywh
        ignore_boxes: M x 4, xywh
        """
        if len(det_boxes) == 0 or len(ignore_boxes) == 0:
            return np.empty((len(det_boxes), len(ignore_boxes)), dtype=np.float32)

        det = VisDroneMOT._xywh_to_xyxy(det_boxes)
        ign = VisDroneMOT._xywh_to_xyxy(ignore_boxes)

        # Broadcasting, (N, 1) x (1, M) -> (N, M)
        inter_x1 = np.maximum(det[:, None, 0], ign[None, :, 0])
        inter_y1 = np.maximum(det[:, None, 1], ign[None, :, 1])
        inter_x2 = np.minimum(det[:, None, 2], ign[None, :, 2])
        inter_y2 = np.minimum(det[:, None, 3], ign[None, :, 3])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        det_area = np.maximum((det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]), 1e-10)

        # Broadcasting makes the column from det_area stretch horizontally, (N, 1) -> (N, M)
        ioa = inter / det_area[:, None]
        return ioa.astype(np.float32)

    def get_preprocessed_seq_data(self, raw_data, cls):
        self._check_unique_ids(raw_data)

        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "tracker_confidences",
            "similarity_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}

        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data["num_timesteps"]):
            gt_ids = raw_data["gt_ids"][t]
            gt_dets = raw_data["gt_dets"][t]
            gt_zero_marked = raw_data["gt_extras"][t]["zero_marked"]

            tracker_ids = raw_data["tracker_ids"][t]
            tracker_dets = raw_data["tracker_dets"][t]
            tracker_confidences = raw_data["tracker_confidences"][t]
            similarity_scores = raw_data["similarity_scores"][t]

            ignore_regions = raw_data["gt_crowd_ignore_regions"][t]

            # keep valid GT only
            gt_keep_mask = np.not_equal(gt_zero_marked, 0)
            gt_ids_kept = gt_ids[gt_keep_mask]
            gt_dets_kept = gt_dets[gt_keep_mask]
            sim_kept = similarity_scores[gt_keep_mask]

            # match tracker to GT first
            to_remove_tracker = np.zeros(len(tracker_ids), dtype=bool)
            matched_cols = np.array([], dtype=int)

            if len(gt_ids_kept) > 0 and len(tracker_ids) > 0:
                matching_scores = sim_kept.copy()

                # a small epsilon to ensure correct math, e.g. 0.4999996 case
                matching_scores[matching_scores < 0.5 - np.finfo(float).eps] = 0

                # High IOU = good, HA minimizes, so pass the inverted matrix
                match_rows, match_cols = linear_sum_assignment(-matching_scores)

                # Filter only the correct pairs
                matched_mask = (
                    matching_scores[match_rows, match_cols] > 0 + np.finfo(float).eps
                )
                match_rows = match_rows[matched_mask]
                match_cols = match_cols[matched_mask]
                matched_cols = match_cols

            # remove unmatched tracker dets that overlap ignore regions
            if len(tracker_ids) > 0 and len(ignore_regions) > 0:
                unmatched_mask = np.ones(len(tracker_ids), dtype=bool)
                unmatched_mask[matched_cols] = False

                # Extract an array from tuple (array, )
                unmatched_idxs = np.where(unmatched_mask)[0]
                if len(unmatched_idxs) > 0:
                    unmatched_dets = tracker_dets[unmatched_idxs]
                    ioa = self._calculate_box_ioa(unmatched_dets, ignore_regions)
                    if ioa.size > 0:
                        # Take the biggest IoA of unmatched detection with any ignored region
                        # and compare it against a threshold
                        remove_unmatched = (
                            np.max(ioa, axis=1) > self.ignore_ioa_threshold
                        )
                        to_remove_tracker[unmatched_idxs[remove_unmatched]] = True

            # apply tracker filtering via bitwise NOT
            data["tracker_ids"][t] = tracker_ids[~to_remove_tracker]
            data["tracker_dets"][t] = tracker_dets[~to_remove_tracker]
            data["tracker_confidences"][t] = tracker_confidences[~to_remove_tracker]
            data["similarity_scores"][t] = np.delete(
                sim_kept, np.where(to_remove_tracker)[0], axis=1
            )

            # apply GT filtering
            data["gt_ids"][t] = gt_ids_kept
            data["gt_dets"][t] = gt_dets_kept

            unique_gt_ids += list(np.unique(data["gt_ids"][t]))
            unique_tracker_ids += list(np.unique(data["tracker_ids"][t]))
            num_gt_dets += len(data["gt_ids"][t])
            num_tracker_dets += len(data["tracker_ids"][t])

        # relabel GT ids contiguously
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["gt_ids"][t]) > 0:
                    data["gt_ids"][t] = gt_id_map[data["gt_ids"][t]].astype(int)

        # relabel tracker ids contiguously
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["tracker_ids"][t]) > 0:
                    data["tracker_ids"][t] = tracker_id_map[
                        data["tracker_ids"][t]
                    ].astype(int)

        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]

        self._check_unique_ids(data, after_preproc=True)
        return data
