#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI_Inference.py
YOLOv8 (ONNX) 推論モジュール

CompostAI は class_names と compute_cn を受け取り、用途別に2通り使える:
  1) C/Nモデル: class_names=[carrot_peels, mixed_veg, onion], compute_cn=True
     -> 枠を描画して保存 + C/N 推定
  2) 熟成段階モデル: class_names=[raw, middle, mature], compute_cn=False
     -> 結果(label)と信頼度(conf)だけ。描画も保存も不要なら infer_frame を使う。
"""

import os
import time
import cv2
import numpy as np
import onnxruntime as ort

from CN_Estimator import estimate_cn

CONF_THRES = 0.4
IOU_THRES = 0.5

# 既定（C/Nモデル）。stage モデルは __init__ で上書きする。
DEFAULT_CLASS_NAMES = ["carrot_peels", "mixed_veg", "onion"]

# クラス名 -> 表示色 (BGR)。無い場合はパレットから順番に割り当て。
CLASS_COLORS = {
    "carrot_peels": (0, 140, 255),
    "mixed_veg": (0, 200, 0),
    "onion": (200, 0, 200),
}
_PALETTE = [(0, 140, 255), (0, 200, 0), (200, 0, 200),
            (255, 0, 0), (0, 255, 255), (255, 255, 0)]


# ============================================================
# 前処理・補助関数
# ============================================================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(np.floor(dh)), int(np.ceil(dh))
    left, right = int(np.floor(dw)), int(np.ceil(dw))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    img = cv2.resize(img, new_shape)
    return img, r, dw, dh


def xywh2xyxy(x):
    y = np.copy(x)
    y[0] = x[0] - x[2] / 2
    y[1] = x[1] - x[3] / 2
    y[2] = x[0] + x[2] / 2
    y[3] = x[1] + x[3] / 2
    return y


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter)
    return inter / np.maximum(union, 1e-6)


def nms(boxes, scores, iou_threshold):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep


# ============================================================
# 推論クラス（モデルごとに1インスタンス）
# ============================================================
class CompostAI:
    def __init__(self, model_path, class_names=None, conf_thres=CONF_THRES,
                 iou_thres=IOU_THRES, compute_cn=True):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        _, _, self.INPUT_H, self.INPUT_W = self.session.get_inputs()[0].shape

        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.compute_cn = compute_cn
        self.portion_level = "medium"   # 全体の既定レベル
        self.portion_levels = {}        # クラス別レベル {class_name: level}
        self._last_inputs = None        # 再計算用に直近の検出を保持

        self.last_result = ("unknown", 0.0)   # (label, conf)
        self.last_detections = []
        self.cn_result = {"cn_ratio": None, "browns_grams": 0.0,
                          "total_mass": 0.0, "per_class_mass": {}, "n_detections": 0}

        print(f"AI model loaded ({self.INPUT_W}x{self.INPUT_H}) "
              f"classes={self.class_names} cn={self.compute_cn}")

    def _color_for(self, label, idx):
        return CLASS_COLORS.get(label, _PALETTE[idx % len(_PALETTE)])

    # --------------------------------------------------------
    # 推論コア：フレーム -> 元ピクセル座標のボックス
    # --------------------------------------------------------
    def _detect(self, img):
        img_resized, r, dw, dh = letterbox(img, (self.INPUT_H, self.INPUT_W))
        # OpenCV は BGR で読み込む。学習時(ultralytics)は RGB なので変換必須。
        # これを忘れると色依存クラス(オレンジのcarrot_peels等)が誤分類される。
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        blob = img_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        output = outputs[0][0]
        boxes = output[:4, :].T
        scores_all = output[4:, :].T

        scores = np.max(scores_all, axis=1)
        class_ids = np.argmax(scores_all, axis=1)

        mask = scores > self.conf_thres
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

        if len(boxes) > 0:
            boxes_xyxy = np.array([xywh2xyxy(b) for b in boxes])
            keep = nms(boxes_xyxy, scores, self.iou_thres)
            boxes_xyxy, scores, class_ids = boxes_xyxy[keep], scores[keep], class_ids[keep]
        else:
            boxes_xyxy = np.zeros((0, 4))

        orig_boxes = []
        for b in boxes_xyxy:
            x1 = int((b[0] - dw) / r)
            y1 = int((b[1] - dh) / r)
            x2 = int((b[2] - dw) / r)
            y2 = int((b[3] - dh) / r)
            orig_boxes.append((x1, y1, x2, y2))
        return orig_boxes, scores, class_ids

    def _build_result(self, orig_boxes, scores, class_ids, frame_w, frame_h):
        detections = [
            {"label": self.class_names[int(c)], "conf": float(s), "box": b}
            for b, s, c in zip(orig_boxes, scores, class_ids)
        ]
        if self.compute_cn:
            self._last_inputs = (list(orig_boxes), [int(c) for c in class_ids],
                                 frame_w, frame_h)
            cn = estimate_cn(orig_boxes, class_ids, frame_w=frame_w, frame_h=frame_h,
                             portion_level=self.portion_level,
                             portion_levels=self.portion_levels)
        else:
            cn = self.cn_result  # 空のまま

        if len(scores) > 0:
            idx = int(np.argmax(scores))
            top = (self.class_names[int(class_ids[idx])], float(scores[idx]))
        else:
            top = ("unknown", 0.0)

        self.last_detections = detections
        self.cn_result = cn
        self.last_result = top
        return {"detections": detections, "cn": cn, "top": top}

    def recompute_cn(self):
        """直近の検出ボックスから、現在の portion_levels で C/N を再計算（再撮影なし）"""
        if not self._last_inputs:
            return self.cn_result
        boxes, class_ids, fw, fh = self._last_inputs
        self.cn_result = estimate_cn(
            boxes, class_ids, frame_w=fw, frame_h=fh,
            portion_level=self.portion_level, portion_levels=self.portion_levels,
        )
        return self.cn_result

    # --------------------------------------------------------
    # ディスク書込なし：結果と信頼度だけ欲しい時（stageモデル等）
    # --------------------------------------------------------
    def infer_frame(self, frame):
        if frame is None:
            return None
        fh, fw = frame.shape[:2]
        orig_boxes, scores, class_ids = self._detect(frame)
        return self._build_result(orig_boxes, scores, class_ids, fw, fh)

    # --------------------------------------------------------
    # ファイル版：枠を描画して保存（C/Nモデルの撮影記録用）
    # --------------------------------------------------------
    def infer_and_save(self, input_path, output_path):
        if not os.path.exists(input_path):
            return False
        img = cv2.imread(input_path)
        if img is None:
            return False

        img0 = img.copy()
        fh, fw = img0.shape[:2]

        t0 = time.time()
        orig_boxes, scores, class_ids = self._detect(img)
        print(f"[AI] Inference time: {(time.time() - t0) * 1000:.1f} ms")

        for (x1, y1, x2, y2), score, cls in zip(orig_boxes, scores, class_ids):
            label = self.class_names[int(cls)]
            color = self._color_for(label, int(cls))
            cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img0, f"{label} {score:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img0)
        self._build_result(orig_boxes, scores, class_ids, fw, fh)
        return True
