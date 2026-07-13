# -*- coding: utf-8 -*-
"""
Database_Neon.py
Neon(PostgreSQL) にセンサーデータ / AI推論結果を保存するDBライブラリ。


想定テーブル:
  - sensor_data(id, bin_id, air_temp, air_humidity, air_pressure,
                soil_temp, soil_humidity, lux, air_quality_ratio, co2_ppm, created_at)
  - model_predictions(id, bin_id, sensor_data_id, stage, confidence,
                      cn_ratio, browns_grams, created_at)


C/N を保存する場合は先に列を追加すること:
  ALTER TABLE model_predictions
    ADD COLUMN IF NOT EXISTS cn_ratio     DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS browns_grams DOUBLE PRECISION;
※ 列が無い場合は自動で旧4列INSERTにフォールバックするので、
  移行前でもクラッシュしない。
"""


import os
from datetime import datetime


import psycopg2
from dotenv import load_dotenv




class DatabaseTool:
    def __init__(self, debug: bool = False):
        self.debug = debug
        load_dotenv()
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in .env file.")
        if self.debug:
            print("DatabaseTool initialized (Neon PostgreSQL)")


    def _get_connection(self):
        return psycopg2.connect(self.database_url, connect_timeout=5)


    # ------------------------------------------------------------
    # Insert compost sensor data
    # ------------------------------------------------------------
    def insert_compost_data(
        self, bin_id, air_temp, air_humidity, air_pressure,
        soil_temp, soil_humidity, lux, air_quality_ratio, co2_ppm,
    ):
        query = """
        INSERT INTO sensor_data (
            bin_id, air_temp, air_humidity, air_pressure,
            soil_temp, soil_humidity, lux, air_quality_ratio, co2_ppm
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id;
        """
        params = (bin_id, air_temp, air_humidity, air_pressure,
                  soil_temp, soil_humidity, lux, air_quality_ratio, co2_ppm)
        conn = None
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    new_id = cur.fetchone()[0]
                    if self.debug:
                        print(f"[DB] sensor_data inserted id={new_id}")
                    return new_id
        finally:
            if conn:
                conn.close()


    # ------------------------------------------------------------
    # Get latest sensor_data.id for a bin
    # ------------------------------------------------------------
    def get_latest_sensor_id(self, bin_id):
        query = """
        SELECT id FROM sensor_data
        WHERE bin_id = %s ORDER BY id DESC LIMIT 1;
        """
        conn = None
        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(query, (bin_id,))
                    row = cur.fetchone()
                    latest_id = row[0] if row else None
                    if self.debug:
                        print(f"[DB] latest sensor_id bin={bin_id}: {latest_id}")
                    return latest_id
        finally:
            if conn:
                conn.close()


    # ------------------------------------------------------------
    # Insert AI model prediction (stage + confidence + optional C/N)
    # ------------------------------------------------------------
    def insert_model_prediction(
        self, bin_id, sensor_data_id, stage, confidence,
        cn_ratio=None, browns_grams=None,
    ):
        """
        model_predictions に推論結果を INSERT。
        cn_ratio / browns_grams 列があればそれも保存。無ければ旧4列で保存。
        """
        full_query = """
        INSERT INTO model_predictions (
            bin_id, sensor_data_id, stage, confidence, cn_ratio, browns_grams
        )
        VALUES (%s,%s,%s,%s,%s,%s);
        """
        full_params = (bin_id, sensor_data_id, stage, confidence, cn_ratio, browns_grams)


        legacy_query = """
        INSERT INTO model_predictions (bin_id, sensor_data_id, stage, confidence)
        VALUES (%s,%s,%s,%s);
        """
        legacy_params = (bin_id, sensor_data_id, stage, confidence)


        conn = None
        try:
            conn = self._get_connection()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(full_query, full_params)
                if self.debug:
                    print(f"[DB] model_prediction inserted (+C/N) bin={bin_id}")
            except psycopg2.errors.UndefinedColumn:
                # cn_ratio/browns_grams 列がまだ無い -> 旧4列で再試行
                # (直前の `with conn:` が例外時に自動でrollback済み)
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(legacy_query, legacy_params)
                if self.debug:
                    print(f"[DB] model_prediction inserted (legacy) bin={bin_id}")
        finally:
            if conn:
                conn.close()
