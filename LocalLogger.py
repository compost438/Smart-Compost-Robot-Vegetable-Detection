#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalLogger.py
センサーデータ／AI推論結果をローカルCSVへ保存するモジュール
"""


import os
import csv
from datetime import datetime


LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)


SENSOR_LOG_PATH = os.path.join(LOG_DIR, "sensor_data.csv")
AI_LOG_PATH = os.path.join(LOG_DIR, "ai_results.csv")




def _append_csv(path: str, fieldnames: list, row: dict) -> None:
    """指定CSVに1行追記（無ければヘッダーも書く）"""
    file_exists = os.path.exists(path)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)




def log_sensor_data(
    bin_id: int,
    air_temp: float,
    air_hum: float,
    air_pres: float,
    soil_temp: float,
    soil_hum: float,
    lux: float,
    air_ratio: float,
    co2_ppm: float,
) -> None:
    """センサーデータを sensor_data.csv に追記（timestampはUTC）"""
    ts = datetime.utcnow().isoformat()
    fieldnames = [
        "timestamp", "bin_id", "air_temp", "air_hum", "air_pres",
        "soil_temp", "soil_hum", "lux", "air_ratio", "co2_ppm",
    ]
    row = {
        "timestamp": ts, "bin_id": bin_id,
        "air_temp": air_temp, "air_hum": air_hum, "air_pres": air_pres,
        "soil_temp": soil_temp, "soil_hum": soil_hum, "lux": lux,
        "air_ratio": air_ratio, "co2_ppm": co2_ppm,
    }
    _append_csv(SENSOR_LOG_PATH, fieldnames, row)




def log_ai_result(
    bin_id: int,
    sensor_id: int,
    stage: str,
    confidence: float,
    cn_ratio=None,
    browns_grams=None,
) -> None:
    """
    AI推論結果を ai_results.csv に追記。
    - stage / confidence : 熟成段階モデルの結果
    - cn_ratio / browns_grams : C/Nモデルの推定値（任意）


    ※ 既存の ai_results.csv が古い5列ヘッダーで残っていると新しい列が
       書かれない。初回はファイルを削除するか、列ありで作り直すこと。
    """
    ts = datetime.utcnow().isoformat()
    fieldnames = [
        "timestamp", "bin_id", "sensor_data_id",
        "stage", "confidence", "cn_ratio", "browns_grams",
    ]
    row = {
        "timestamp": ts, "bin_id": bin_id, "sensor_data_id": sensor_id,
        "stage": stage, "confidence": confidence,
        "cn_ratio": cn_ratio, "browns_grams": browns_grams,
    }
    _append_csv(AI_LOG_PATH, fieldnames, row)
