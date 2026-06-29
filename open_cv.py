#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
open_cv.py
Compost Management System 用: カメラ撮影 + AI推論(2モデル) + ログ保存


2モデル運用:
- cn_model    : carrot_peels / mixed_veg / onion -> 枠を描画して保存 + C/N推定
- stage_model : raw / middle / mature           -> 結果(label)と信頼度だけ(保存なし)


手動撮影:
- capture_now() をボタンから呼ぶ。カメラを良い位置に動かしてから撮影できる。
- get_latest_frame() でGUIの照準プレビューに最新フレームを渡す。
"""


import os
import cv2
import time
import threading


from AI_Inference import CompostAI
from Database_Neon import DatabaseTool
from LocalLogger import log_ai_result


# モデルファイル（それぞれ別の .onnx を指す）
CN_MODEL_PATH = "/home/d5110/Desktop/Nicholas/AI Model/best_veg.onnx"
STAGE_MODEL_PATH = "/home/d5110/Desktop/Nicholas/AI Model/best_stage.onnx"


BIN_ID = 2




class OpenCVCamera:
    def __init__(
        self,
        save_dir="./Photos",
        interval_sec=1800,
        on_new_photo=None,
        before_capture=None,
        after_capture=None,
        overlay_interval=0.0,    # 0=ライブ時の自動推論なし（手動運用では0推奨）
        auto_capture=False,      # False=タイマー撮影なし、ボタンのみ
        debug=False,
    ):
        self.debug = debug
        self.save_dir = save_dir
        self.current_dir = os.path.join(save_dir, "Current_Image")


        self.on_new_photo = on_new_photo
        self.before_capture = before_capture
        self.after_capture = after_capture


        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.current_dir, exist_ok=True)


        # ---- カメラ ----
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):
            self.cap.read()
            time.sleep(0.05)


        # ---- 最新フレーム共有 ----
        self.latest_frame = None
        self.frame_lock = threading.Lock()


        # ---- AIモデル(2つ) + DB ----
        self.cn_model = CompostAI(
            CN_MODEL_PATH,
            class_names=["carrot_peels", "mixed_veg", "onion"],
            compute_cn=True,
        )
        self.stage_model = CompostAI(
            STAGE_MODEL_PATH,
            class_names=["raw", "middle", "mature"],
            compute_cn=False,
        )
        self.db = DatabaseTool()


        # 推論セッションは同時実行不可 -> 直列化
        self.inference_lock = threading.Lock()


        # GUI/ライブ用キャッシュ
        self.last_detections = []          # 野菜枠（照準プレビュー用）
        self.last_cn = None                # C/N結果
        self.last_stage = ("unknown", 0.0) # (stage, conf)


        # ---- スケジューラ状態 ----
        self.interval = max(5.0, float(interval_sec))
        self.next_capture_time = time.time() + self.interval
        self.scheduler_lock = threading.Lock()
        self.overlay_interval = float(overlay_interval)
        self.auto_capture = bool(auto_capture)


        self.running = False
        self._capturing = False  # 撮影中フラグ（多重実行防止）


        self.latest_photo = self._get_latest_photo()
        print(f"[CAM] Ready (interval={self.interval:.0f}s, "
              f"auto={self.auto_capture}, overlay={self.overlay_interval:.0f}s)")


    # ============================================================
    # 内部ユーティリティ
    # ============================================================
    def _d(self, msg):
        if self.debug:
            print(msg)


    def _get_latest_photo(self):
        ai_path = os.path.join(self.current_dir, "latest_ai.jpg")
        if os.path.exists(ai_path):
            return ai_path
        files = [f for f in os.listdir(self.save_dir)
                 if f.startswith("frame-") and f.endswith(".jpg")]
        if not files:
            return None
        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(self.save_dir, f)))
        return os.path.join(self.save_dir, latest_file)


    # ============================================================
    # 公開API
    # ============================================================
    def get_latest_frame(self):
        """照準プレビュー用：最新フレームのコピー"""
        with self.frame_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()


    def capture_now(self):
        """手動シャッター：カメラを良い位置に置いてから押す"""
        threading.Thread(target=self._do_capture, daemon=True).start()


    def set_auto_capture(self, enabled: bool):
        self.auto_capture = bool(enabled)
        print(f"[CAM] Auto capture -> {self.auto_capture}")


    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._frame_grabber, daemon=True).start()
        threading.Thread(target=self._capture_scheduler, daemon=True).start()
        if self.overlay_interval > 0:
            threading.Thread(target=self._overlay_scheduler, daemon=True).start()
        print("[CAM] Started")
        if self.latest_photo and callable(self.on_new_photo):
            try:
                self.on_new_photo(self.latest_photo)
            except Exception as e:
                self._d(f"[CAM] Startup image callback error: {e}")


    def stop(self):
        self.running = False
        time.sleep(0.2)
        if self.cap:
            for _ in range(3):
                self.cap.read()
            self.cap.release()
        cv2.destroyAllWindows()
        print("[CAM] Stopped")


    def set_interval(self, seconds):
        try:
            new_interval = max(5.0, float(seconds))
            with self.scheduler_lock:
                self.interval = new_interval
                self.next_capture_time = time.time() + new_interval
            print(f"[CAM] Interval updated -> {new_interval:.0f}s")
        except Exception as e:
            print(f"[CAM] Failed to update interval: {e}")


    # ============================================================
    # スレッド1: フレーム取得
    # ============================================================
    def _frame_grabber(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            time.sleep(0.01)


    # ============================================================
    # スレッド2: 撮影スケジューラ（auto_capture=Trueのときだけ撮る）
    # ============================================================
    def _capture_scheduler(self):
        while self.running:
            now = time.time()
            with self.scheduler_lock:
                due = now >= self.next_capture_time
                if due:
                    self.next_capture_time = now + self.interval
            if due and self.auto_capture:
                self._do_capture()
            time.sleep(1.0)


    # ============================================================
    # スレッド3: ライブ検出枠の更新（overlay_interval>0のときのみ）
    # ============================================================
    def _overlay_scheduler(self):
        while self.running:
            frame = self.get_latest_frame()
            if frame is not None:
                with self.inference_lock:
                    try:
                        res = self.cn_model.infer_frame(frame)
                        if res:
                            self.last_detections = res["detections"]
                            self.last_cn = res["cn"]
                    except Exception as e:
                        self._d(f"[CAM] overlay inference error: {e}")
            time.sleep(max(2.0, self.overlay_interval))


    # ============================================================
    # 撮影 + 2モデル推論 + DB/ローカルログ + GUI更新
    # ============================================================
    def _do_capture(self):
        if self._capturing:
            self._d("[CAM] capture already running, skip")
            return
        self._capturing = True
        try:
            if callable(self.before_capture):
                try:
                    self.before_capture()
                except Exception as e:
                    self._d(f"[CAM] before_capture error: {e}")


            time.sleep(0.15)


            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                print("[CAM] No frame available")
                return


            ts = time.strftime("%Y-%m-%d-%H-%M-%S")
            raw_path = os.path.join(self.save_dir, f"frame-{ts}.jpg")
            latest_path = os.path.join(self.current_dir, "latest.jpg")


            cv2.putText(frame, ts, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(raw_path, frame)
            cv2.imwrite(latest_path, frame)
            print(f"[CAM] Captured -> {raw_path}")


            if callable(self.after_capture):
                try:
                    self.after_capture()
                except Exception as e:
                    self._d(f"[CAM] after_capture error: {e}")


            # ---------- 2モデル推論 ----------
            ai_path = os.path.join(self.current_dir, "latest_ai.jpg")
            try:
                with self.inference_lock:
                    # 野菜モデル：枠を描画して保存 + C/N
                    cn_ok = self.cn_model.infer_and_save(latest_path, ai_path)
                    # 段階モデル：結果と信頼度だけ（保存なし）
                    stage_res = self.stage_model.infer_frame(frame)


                stage, conf = stage_res["top"] if stage_res else ("unknown", 0.0)
                cn = self.cn_model.cn_result


                # GUIキャッシュ更新
                self.last_detections = self.cn_model.last_detections
                self.last_cn = cn
                self.last_stage = (stage, conf)


                print(f"[AI] stage={stage} ({conf:.2f}) "
                      f"C/N={cn.get('cn_ratio')} browns={cn.get('browns_grams')}g")


                if not cn_ok:
                    ai_path = latest_path


                # ---------- DB + ローカルCSV ----------
                sensor_id = None
                try:
                    sensor_id = self.db.get_latest_sensor_id(bin_id=BIN_ID)
                except Exception as e:
                    print(f"[AI] Failed to fetch sensor_id: {e}")


                if sensor_id:
                    try:
                        self.db.insert_model_prediction(
                            bin_id=BIN_ID,
                            sensor_data_id=sensor_id,
                            stage=stage,
                            confidence=conf,
                        )
                        log_ai_result(
                            bin_id=BIN_ID,
                            sensor_id=sensor_id,
                            stage=stage,
                            confidence=conf,
                            cn_ratio=cn.get("cn_ratio"),
                            browns_grams=cn.get("browns_grams"),
                        )
                        self._d("[AI] Logged to DB + local CSV")
                    except Exception as e:
                        print(f"[AI] DB/local log error: {e}")
                else:
                    self._d("[AI] No sensor_id -> not linked")


            except Exception as e:
                print(f"[CAM] AI error: {e}")
                ai_path = latest_path


            # ---------- GUI更新 ----------
            if callable(self.on_new_photo):
                try:
                    self.on_new_photo(ai_path)
                except Exception as e:
                    self._d(f"[CAM] GUI callback error: {e}")
        finally:
            self._capturing = False



