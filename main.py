#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, sys, time, datetime, json
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageOps
import pigpio

from Device_Library import DeviceClass
from Database_Neon import DatabaseTool
from open_cv import OpenCVCamera
from LocalLogger import log_sensor_data

IMAGE_SIZE = 360  # smaller so the whole page fits a 1080p screen

# ---- Theme palette (dark + compost green) ----
ACCENT = "#34c759"        # bright green accent
ACCENT_HOVER = "#248a3d"  # pressed/hover
TEXT_MUTED = "#9aa0a6"    # secondary text on dark
CARD_BG = "#1c1c1e"       # card / header background
BTN_OFF = "#3a3a3a"       # neutral (relay off)

# ============================================================
# デバッグ出力スイッチ（必要なときだけ True にする）
# ============================================================
DEBUG = False
def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# ============================================================
# GPIO CONTROLLER
# （リレーは Active-Low: 0=ON, 1=OFF）
# ============================================================
class GPIOController:
    def __init__(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpiod not running. Run 'sudo pigpiod' first.")

        # GPIOピン割り当て
        self.pins = {"LED": 25, "Fan": 24, "Spray": 23, "Mixer": 22}

        # 現在の状態（ON/OFF）
        self.state = {n: False for n in self.pins}

        # 手動操作（MANUAL）中かどうか
        self.manual = {n: False for n in self.pins}  # manual override flags

        for p in self.pins.values():
            self.pi.set_mode(p, pigpio.OUTPUT)
            self.pi.write(p, 1)  # relays OFF (active-low)

        print("GPIOController initialized")

    def set(self, name, on: bool):
        """AUTO制御用の set（manual フラグは変更しない）"""
        if name in self.pins:
            self.pi.write(self.pins[name], 0 if on else 1)
            self.state[name] = on

    def toggle(self, name):
        """互換性のため残す（現在GUIボタンでは未使用）"""
        if name in self.pins:
            self.set(name, not self.state[name])
            dprint(f"{name}: {'ON' if self.state[name] else 'OFF'}")

    # ----------------------------
    # Manual override API
    # ----------------------------
    def set_manual(self, name, on: bool):
        """
        ユーザーが強制的にON/OFFする（AUTOロジックは上書きしない）
        """
        if name in self.pins:
            self.manual[name] = True
            self.set(name, on)
            print(f"{name}: {'ON' if on else 'OFF'} (MANUAL)")

    def clear_manual(self, name):
        """手動制御を解除し、AUTO制御に戻す"""
        if name in self.pins:
            self.manual[name] = False
            print(f"{name}: MANUAL CLEARED → AUTO")

    def is_manual(self, name):
        return self.manual.get(name, False)

    def get(self, name):
        return self.state.get(name, False)

    def cleanup(self):
        """終了時に全リレーを安全OFFへ戻す"""
        for p in self.pins.values():
            self.pi.write(p, 1)
        self.pi.stop()
        print("GPIO cleaned up (all pins set HIGH)")


# ============================================================
# MAIN GUI
# ============================================================
def F_MakeScreen_Demo():
    # センサー / DB / GPIO 初期化
    dev, db, gpio = DeviceClass(), DatabaseTool(), GPIOController()

    camera = None
    gpio_buttons = {}      # GPIOボタン参照
    capture_led_on = False # 撮影時にLEDを自動点灯したか
    LIGHT_MIN_LUX = 50.0   # 暗い場合のみ撮影前LED点灯（Enable Light がONのとき）

    # DB送信用：直近N回の平均を送る
    DB_AVG_WINDOW = 20
    reading_buffer = []

    # ============================================================
    # MQ135 CONFIG（Rs/R0 の計算）
    # ============================================================
    MQ135_R0_PATH = "/home/d5110/Desktop/Gordon/mq135_r0.json"
    VCC = 5.0
    RL = 10000.0  # 10kΩ

    try:
        with open(MQ135_R0_PATH, "r") as f:
            MQ135_R0 = json.load(f)["R0"]
        print(f"Loaded MQ135 calibration R0 = {MQ135_R0/1000:.2f} kΩ")
    except Exception as e:
        MQ135_R0 = 10000.0
        print(f"Could not load MQ135 calibration ({e}) — using default R0 = 10kΩ")

    def compute_rs(vout, vcc=VCC, rl=RL):
        # 0除算防止・不正値防止
        vout = max(0.001, min(vout, vcc - 0.001))
        return rl * (vcc / vout - 1.0)

    # ============================================================
    # GUI WINDOW
    # ============================================================
    control = ctk.CTk()
    control.title("Compost Management System - Live Monitor")
    _sw = control.winfo_screenwidth()
    _sh = control.winfo_screenheight()
    _w = min(1200, _sw)
    _h = min(980, _sh - 100)   # タスクバー/枠の分を残して画面に収める
    control.geometry(f"{_w}x{_h}+0+0")
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("green")
    control.grid_rowconfigure(0, weight=0)
    control.grid_rowconfigure(1, weight=1)
    control.grid_columnconfigure((0, 1), weight=1)

    # ============================================================
    # HEADER BANNER（両カラムにまたがるタイトル + 状態）
    # ============================================================
    header = ctk.CTkFrame(control, corner_radius=14, fg_color=CARD_BG)
    header.grid(row=0, column=0, columnspan=2, padx=20, pady=(16, 4), sticky="ew")
    header.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(header, text="Compost Management System",
                 font=("Arial", 26, "bold")).grid(row=0, column=0, padx=22, pady=14, sticky="w")
    control.lbl_header_status = ctk.CTkLabel(header, text="● Live",
                                             text_color=ACCENT, font=("Arial", 16, "bold"))
    control.lbl_header_status.grid(row=0, column=1, padx=22, pady=14, sticky="e")

    # ============================================================
    # LEFT PANEL（センサー表示・閾値・インターバル・Enable）
    # ============================================================
    left = ctk.CTkScrollableFrame(control, corner_radius=12)
    left.grid(row=1, column=0, padx=(20, 10), pady=(8, 20), sticky="nsew")
    ctk.CTkLabel(left, text="Sensor Data", font=("Arial", 20, "bold")).pack(pady=(12, 8))

    sensors_grid = ctk.CTkFrame(left, fg_color="transparent")
    sensors_grid.pack(fill="x", padx=8, pady=(0, 4))
    sensors_grid.grid_columnconfigure((0, 1), weight=1)

    def add_stat_card(row, col, label, attr):
        card = ctk.CTkFrame(sensors_grid, corner_radius=10, fg_color=CARD_BG)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="ew")
        ctk.CTkLabel(card, text=label, text_color=TEXT_MUTED,
                     font=("Arial", 11)).pack(anchor="w", padx=12, pady=(7, 0))
        v = ctk.CTkLabel(card, text="--", font=("Arial", 22, "bold"))
        v.pack(anchor="w", padx=12, pady=(0, 7))
        setattr(control, attr, v)
        return card

    add_stat_card(0, 0, "Air Temp (°C)", "text_Temp")
    add_stat_card(0, 1, "Air Hum (%)", "text_Humi")
    add_stat_card(1, 0, "Pressure (hPa)", "text_Pres")
    add_stat_card(1, 1, "Light (Lux)", "text_Lux")
    add_stat_card(2, 0, "Soil Temp (°C)", "text_Temp_SHT")
    add_stat_card(2, 1, "Soil Hum (%)", "text_Humi_SHT")
    add_stat_card(3, 0, "CO2 (ppm)", "text_Co2")

    # Air Quality card（インジケータ + ステータス付き）
    aq_card = add_stat_card(3, 1, "Air Quality (Rs/R0)", "text_MQ135")
    aq_row = ctk.CTkFrame(aq_card, fg_color="transparent")
    aq_row.pack(anchor="w", padx=12, pady=(0, 7))
    control.air_indicator = ctk.CTkFrame(aq_row, width=14, height=14, corner_radius=7, fg_color="grey")
    control.air_indicator.pack(side="left")
    control.air_status_label = ctk.CTkLabel(aq_row, text="---", text_color=TEXT_MUTED, font=("Arial", 11))
    control.air_status_label.pack(side="left", padx=(6, 0))

    # --- Control ranges ---
    ctk.CTkLabel(left, text="Control Ranges (Target Environment)", font=("Arial", 16, "bold")).pack(pady=(15, 6))

    def add_range_row(parent, label, min_default, max_default, width=60):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row, text=label, width=180, anchor="w").pack(side="left")
        e1 = ctk.CTkEntry(row, width=width); e1.insert(0, min_default); e1.pack(side="left", padx=2)
        ctk.CTkLabel(row, text="–").pack(side="left")
        e2 = ctk.CTkEntry(row, width=width); e2.insert(0, max_default); e2.pack(side="left", padx=2)
        return e1, e2

    def add_single_row(parent, label, default, width=60):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row, text=label, width=180, anchor="w").pack(side="left")
        e = ctk.CTkEntry(row, width=width); e.insert(0, default); e.pack(side="left", padx=2)
        return e

    control.text_soilhum_min, control.text_soilhum_max = add_range_row(left, "Soil Humidity Range (%)", "35", "55")
    control.text_soiltemp_min, control.text_soiltemp_max = add_range_row(left, "Soil Temperature Range (°C)", "50", "60")
    control.text_airtemp_max = add_single_row(left, "Air Temp Max (°C)", "40")
    control.text_airhum_min, control.text_airhum_max = add_range_row(left, "Air Humidity Range (%)", "50", "70")
    control.text_co2_max = add_single_row(left, "CO₂ Max (ppm)", "2000")

    # --- Intervals ---
    ctk.CTkLabel(left, text="Intervals (seconds)", font=("Arial", 16, "bold")).pack(pady=(15, 6))

    def add_interval_row(parent, label, default):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row, text=label, width=160, anchor="w").pack(side="left")
        e = ctk.CTkEntry(row, width=80); e.insert(0, default); e.pack(side="left")
        return e

    control.text_sensorinterval   = add_interval_row(left, "Sensor Interval", "3")
    control.text_dbinterval       = add_interval_row(left, "Database Interval", "60")
    control.text_mixer_interval   = add_interval_row(left, "Mixer Interval", "1800")
    control.text_mixer_duration   = add_interval_row(left, "Mixer Duration", "60")
    control.text_camera_interval  = add_interval_row(left, "Camera Interval", "3600")
    control.text_spray_duration   = add_interval_row(left, "Spray Duration", "60")

    # --- Enable checkboxes ---
    ctk.CTkLabel(left, text="Enable Controls", font=("Arial", 16, "bold")).pack(pady=(15, 5))
    grid = ctk.CTkFrame(left, fg_color="transparent")
    grid.pack(padx=10, pady=3)
    grid.grid_columnconfigure((0, 1), weight=1)

    control.chk_fan_var   = ctk.BooleanVar(value=True)
    control.chk_spray_var = ctk.BooleanVar(value=True)
    control.chk_light_var = ctk.BooleanVar(value=True)
    control.chk_mixer_var = ctk.BooleanVar(value=True)

    ctk.CTkCheckBox(grid, text="Enable Fan",   variable=control.chk_fan_var).grid(row=0, column=0, sticky="w", padx=5, pady=2)
    ctk.CTkCheckBox(grid, text="Enable Spray", variable=control.chk_spray_var).grid(row=0, column=1, sticky="w", padx=5, pady=2)
    ctk.CTkCheckBox(grid, text="Enable Light", variable=control.chk_light_var).grid(row=1, column=0, sticky="w", padx=5, pady=2)
    ctk.CTkCheckBox(grid, text="Enable Mixer", variable=control.chk_mixer_var).grid(row=1, column=1, sticky="w", padx=5, pady=2)

    # ============================================================
    # RIGHT PANEL（画像表示 + GPIO手動ボタン + 状態表示）
    # ============================================================
    right = ctk.CTkFrame(control, corner_radius=12)
    right.grid(row=1, column=1, padx=(10, 20), pady=(8, 20), sticky="nsew")
    right.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(right, text="Live Compost View", font=("Arial", 20, "bold")).grid(row=0, column=0, pady=10)

    control.canvas1 = ctk.CTkCanvas(right, width=IMAGE_SIZE, height=IMAGE_SIZE, bg="black", highlightthickness=0)
    control.canvas1.grid(row=1, column=0, padx=10, pady=(10, 4))

    gpio_frame = ctk.CTkFrame(right, corner_radius=12)
    gpio_frame.grid(row=2, column=0, padx=10, pady=(8, 8))

    ctk.CTkLabel(gpio_frame, text="GPIO Controls", font=("Arial", 18, "bold")).pack(pady=8)
    btn_row = ctk.CTkFrame(gpio_frame, fg_color="transparent")
    btn_row.pack()

    for i, n in enumerate(["LED", "Fan", "Spray", "Mixer"]):
        b = ctk.CTkButton(
            btn_row,
            text=f"{n} OFF",
            width=120,
            fg_color=BTN_OFF,
            # 手動切替：AUTOでは上書きしない
            command=lambda x=n: (gpio.set_manual(x, not gpio.get(x)), update_gpio_button(x)),
        )
        b.grid(row=0, column=i, padx=10, pady=6)
        gpio_buttons[n] = b

    control.lbl_photo_timestamp = ctk.CTkLabel(right, text="No image loaded", text_color=TEXT_MUTED)
    control.lbl_photo_timestamp.grid(row=3, column=0, pady=(6, 2))
    control.lbl_db_status = ctk.CTkLabel(right, text="Database: Idle", text_color=TEXT_MUTED)
    control.lbl_db_status.grid(row=4, column=0, pady=(0, 8))

    # --- AI / C/N card ---
    ai_card = ctk.CTkFrame(right, corner_radius=14, fg_color=CARD_BG,
                           border_width=2, border_color=ACCENT)
    ai_card.grid(row=5, column=0, padx=12, pady=(0, 10), sticky="ew")
    ai_card.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(ai_card, text="CARBON / NITROGEN", text_color=TEXT_MUTED,
                 font=("Arial", 12, "bold")).grid(row=0, column=0, pady=(12, 0))
    control.lbl_cn = ctk.CTkLabel(ai_card, text="C/N --   •   add -- g browns",
                                  font=("Arial", 21, "bold"), text_color=ACCENT)
    control.lbl_cn.grid(row=1, column=0, pady=(2, 6))
    control.lbl_stage = ctk.CTkLabel(ai_card, text="Stage: --", text_color=TEXT_MUTED,
                                     font=("Arial", 14))
    control.lbl_stage.grid(row=2, column=0, pady=(0, 6))

    # --- Per-class fill-level selectors (populated after each capture) ---
    _LEVELS = {"Light": "light", "Medium": "medium",
               "Heavy": "heavy", "Very Heavy": "very_heavy"}
    _KEY_TO_LABEL = {v: k for k, v in _LEVELS.items()}
    control.portion_level = "medium"   # default for any class not explicitly set

    ctk.CTkLabel(ai_card, text="Fill level per class", text_color=TEXT_MUTED,
                 font=("Arial", 12, "bold")).grid(row=3, column=0, pady=(0, 2))
    control.levels_container = ctk.CTkFrame(ai_card, fg_color="transparent")
    control.levels_container.grid(row=4, column=0, padx=10, pady=(0, 12), sticky="ew")
    control.levels_container.grid_columnconfigure(0, weight=1)
    control.level_hint = ctk.CTkLabel(control.levels_container,
                                      text="(capture to set levels)",
                                      text_color=TEXT_MUTED, font=("Arial", 12))
    control.level_hint.pack()

    control.live_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(right, text="Live View (aim the camera)",
                    variable=control.live_var).grid(row=6, column=0, pady=(0, 8))

    control.btn_capture = ctk.CTkButton(
        right, text="Capture Now", width=210, height=50, corner_radius=12,
        fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#0b3d1a",
        font=("Arial", 18, "bold"),
        command=lambda: manual_capture(),
    )
    control.btn_capture.grid(row=7, column=0, pady=(0, 12))

    # ============================================================
    # HELPERS
    # ============================================================
    def update_gpio_button(name):
        """GPIOボタン表示更新（ON/OFFの色と文字）"""
        if name in gpio_buttons:
            state = gpio.get(name)
            btn = gpio_buttons[name]
            btn.configure(text=f"{name} {'ON' if state else 'OFF'}",
                          fg_color=ACCENT if state else BTN_OFF)

    def update_air_indicator(ratio_value: float):
        """MQ135（Rs/R0）に応じて色とステータス表示を更新"""
        if ratio_value >= 2.0:
            color, text = "green", "Good"
        elif 1.5 <= ratio_value < 2.0:
            color, text = "yellow", "Moderate"
        else:
            color, text = "red", "Poor"
        control.air_indicator.configure(fg_color=color)
        control.air_status_label.configure(text=f"Air: {text}")

    def F_update_image(latest_path):
        """
        画像表示更新：
        - latest_ai.jpg があれば推論後画像を優先
        - 画像は中央フィット（伸縮せず黒背景でパディング）
        """
        try:
            # ライブ表示中はキャンバスは live_view_loop が描画するので触らない
            if getattr(control, "live_var", None) is not None and control.live_var.get():
                try:
                    ts = time.ctime(os.path.getmtime(latest_path))
                    control.lbl_photo_timestamp.configure(text=f"Last capture: {ts}", text_color=TEXT_MUTED)
                except Exception:
                    pass
                return

            ai_path = os.path.join(os.path.dirname(latest_path), "latest_ai.jpg")
            image_to_show = ai_path if os.path.exists(ai_path) else latest_path

            if os.path.exists(image_to_show):
                img = Image.open(image_to_show)
                img = ImageOps.pad(img, (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
            else:
                img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(20, 20, 20))
                draw = ImageDraw.Draw(img)
                draw.text((IMAGE_SIZE // 2 - 80, IMAGE_SIZE // 2), "No Image Found", fill=(200, 200, 200))

            control.photo_img = ImageTk.PhotoImage(img)
            control.canvas1.delete("all")
            control.canvas1.create_image(0, 0, image=control.photo_img, anchor=ctk.NW)

            try:
                ts = time.ctime(os.path.getmtime(image_to_show))
            except Exception:
                ts = "Unknown"

            control.lbl_photo_timestamp.configure(text=f"Captured at: {ts}", text_color=TEXT_MUTED)

        except Exception as e:
            print(f"Image repaint error: {e}")
            control.lbl_photo_timestamp.configure(text="Image load error", text_color="red")

    # ============================================================
    # 設定値の即時反映（FocusOut / Enter）
    # ============================================================
    def bind_auto_update(entry_widget, callback):
        entry_widget.bind("<Return>", lambda e: callback())
        entry_widget.bind("<FocusOut>", lambda e: callback())

    def update_all_settings():
        """インターバルを即時反映（カメラなど）"""
        try:
            if camera:
                new_cam_int = float(control.text_camera_interval.get())
                camera.set_interval(new_cam_int)
                dprint(f"Camera interval updated to {new_cam_int}s")

            dprint(
                f"Updated intervals: Sensor={control.text_sensorinterval.get()}s, "
                f"DB={control.text_dbinterval.get()}s, "
                f"Mixer I={control.text_mixer_interval.get()}s, "
                f"Mixer D={control.text_mixer_duration.get()}s"
            )
        except Exception as e:
            print(f"Auto-update error: {e}")

    for entry in [
        control.text_sensorinterval,
        control.text_dbinterval,
        control.text_mixer_interval,
        control.text_mixer_duration,
        control.text_camera_interval,
        control.text_spray_duration,
        control.text_soilhum_min,
        control.text_soilhum_max,
        control.text_airtemp_max,
        control.text_airhum_min,
        control.text_airhum_max,
        control.text_co2_max,
    ]:
        bind_auto_update(entry, update_all_settings)

    # ============================================================
    # MIXER AUTO TIMER（定期撹拌）
    # ============================================================
    mixer_running = False
    mixer_job = None

    def run_mixer_cycle():
        """一定周期で Mixer を duration 秒だけON"""
        nonlocal mixer_running, mixer_job
        if not mixer_running:
            return
        try:
            interval = float(control.text_mixer_interval.get())
            duration = float(control.text_mixer_duration.get())
            dprint(f"Timed Aeration: Mixer ON for {duration}s (interval: {interval}s)")

            if control.chk_mixer_var.get() and (not gpio.is_manual("Mixer")):
                gpio.set("Mixer", True)
                update_gpio_button("Mixer")

            control.after(
                int(duration * 1000),
                lambda: (
                    (gpio.set("Mixer", False), update_gpio_button("Mixer"))
                    if not gpio.is_manual("Mixer") else None
                ),
            )
            mixer_job = control.after(int(interval * 1000), run_mixer_cycle)

        except Exception as e:
            print(f"Mixer cycle error: {e}")

    def start_auto_mixer():
        nonlocal mixer_running
        if mixer_running:
            return
        mixer_running = True
        control.btn_mixer_auto.configure(text="Stop Auto Mixer", fg_color="#dc3545")
        run_mixer_cycle()

    def stop_auto_mixer():
        nonlocal mixer_running, mixer_job
        mixer_running = False
        if mixer_job:
            control.after_cancel(mixer_job)
            mixer_job = None

        if not gpio.is_manual("Mixer"):
            gpio.set("Mixer", False)
            update_gpio_button("Mixer")

        control.btn_mixer_auto.configure(text="Start Auto Mixer", fg_color="#007bff")

    bottom = ctk.CTkFrame(gpio_frame, fg_color="transparent")
    bottom.pack(pady=6)

    ctk.CTkButton(
        bottom,
        text="Refresh Data",
        width=140,
        command=lambda: F_button_getdata_function(force_now=True),
    ).pack(side="left", padx=5)

    control.btn_mixer_auto = ctk.CTkButton(
        bottom,
        text="Start Auto Mixer",
        width=160,
        fg_color="#007bff",
        command=lambda: (stop_auto_mixer() if mixer_running else start_auto_mixer()),
    )
    control.btn_mixer_auto.pack(side="left", padx=5)

    # ============================================================
    # SENSOR LOOP（センサー取得→UI更新→制御→ログ→DB送信）
    # ============================================================
    next_db_time = time.time() + float(control.text_dbinterval.get())

    def F_button_getdata_function(force_now: bool = False):
        """
        センサーループ本体：
        - センサー読み取り
        - UI更新
        - 自動制御（優先順位 + ヒステリシス + クールダウン）
        - ローカルログ（毎回）
        - DB送信（一定周期、直近N回平均）
        """
        nonlocal next_db_time, reading_buffer

        # --- Read sensors ---
        try:
            _status_sht, soil_temp, soil_hum = dev.def_sht35()
        except Exception:
            soil_temp, soil_hum = 0.0, 0.0

        try:
            air_temp, air_hum, air_pres = dev.F_bme280()
        except Exception:
            air_temp, air_hum, air_pres = 0.0, 0.0, 0.0

        try:
            lux, _vis, _ir = dev.F_tsl2561()
        except Exception:
            lux = 0.0

        try:
            ads_status, ads_voltage, _ads_raw = dev.def_ads1115(channel=0)
        except Exception:
            ads_status, ads_voltage = 1, 0.0

        try:
            co2_ppm = dev.read_mh_z19c()
        except Exception:
            co2_ppm = 0.0

        # --- MQ135 ratio + EMA ---
        if ads_status == 0 and ads_voltage > 0:
            Rs = compute_rs(ads_voltage)
            ratio = Rs / MQ135_R0
        else:
            ratio = 0.0

        EMA_ALPHA = 0.3
        if not hasattr(F_button_getdata_function, "_ratio_ema"):
            F_button_getdata_function._ratio_ema = ratio
        else:
            F_button_getdata_function._ratio_ema = (
                EMA_ALPHA * ratio + (1 - EMA_ALPHA) * F_button_getdata_function._ratio_ema
            )
        ratio_smooth = F_button_getdata_function._ratio_ema

        # センサー詳細ログはDEBUGのときだけ
        dprint(
            f"[SENSORS] air={air_temp:.2f}C {air_hum:.2f}% pres={air_pres:.2f} "
            f"soil={soil_temp:.2f}C {soil_hum:.2f}% lux={lux:.2f} "
            f"mq135={ratio_smooth:.2f} co2={co2_ppm:.0f}"
        )

        # Update GUI indicator
        update_air_indicator(ratio_smooth)

        # --- Fill UI entries ---
        def set_entry(name, value):
            w = getattr(control, name)
            try:
                w.configure(text=f"{value:.2f}")   # stat-card labels
            except Exception:
                w.delete(0, ctk.END)               # fallback for entries
                w.insert(ctk.END, f"{value:.2f}")

        set_entry("text_Temp", air_temp)
        set_entry("text_Humi", air_hum)
        set_entry("text_Pres", air_pres)
        set_entry("text_Lux", lux)
        set_entry("text_Temp_SHT", soil_temp)
        set_entry("text_Humi_SHT", soil_hum)
        set_entry("text_MQ135", ratio_smooth)
        set_entry("text_Co2", co2_ppm)

        # ============================================================
        # CONTROL LOGIC（ヒステリシス + クールダウン + 優先順位）
        # ============================================================
        try:
            soil_hum_min  = float(control.text_soilhum_min.get())
            soil_hum_max  = float(control.text_soilhum_max.get())
            soil_temp_max = float(control.text_soiltemp_max.get())

            air_temp_max  = float(control.text_airtemp_max.get())
            air_hum_min   = float(control.text_airhum_min.get())
            air_hum_max   = float(control.text_airhum_max.get())
            co2_max       = float(control.text_co2_max.get())

            mixer_duration = float(control.text_mixer_duration.get())
            mixer_interval = float(control.text_mixer_interval.get())

            now = time.time()

            if not hasattr(F_button_getdata_function, "timers"):
                F_button_getdata_function.timers = {
                    "mixer_on_until": 0.0,
                    "mixer_cooldown_until": 0.0,
                    "spray_on_until": 0.0,
                    "spray_cooldown_until": 0.0,
                }
            timers = F_button_getdata_function.timers

            def request_mixer_start(_reason: str):
                if now < timers["mixer_on_until"]:
                    return True
                if now >= timers["mixer_cooldown_until"]:
                    timers["mixer_on_until"] = now + mixer_duration
                    timers["mixer_cooldown_until"] = now + mixer_duration + mixer_interval
                    dprint(f"[MIXER] start({_reason}) ON={mixer_duration:.0f}s cooldown={mixer_interval:.0f}s")
                    return True
                return False

            SPRAY_COOLDOWN_SECONDS = 120.0
            spray_duration = float(control.text_spray_duration.get())

            def request_spray_start(_reason: str):
                if now < timers["spray_on_until"]:
                    return True
                if now >= timers["spray_cooldown_until"]:
                    timers["spray_on_until"] = now + spray_duration
                    timers["spray_cooldown_until"] = now + spray_duration + SPRAY_COOLDOWN_SECONDS
                    dprint(f"[SPRAY] start({_reason}) ON={spray_duration:.0f}s cooldown={SPRAY_COOLDOWN_SECONDS:.0f}s")
                    return True
                return False

            # 現在の状態から開始
            fan   = gpio.get("Fan")
            spray = gpio.get("Spray")
            mixer = gpio.get("Mixer")

            # ---- Soil temperature ----
            soil_too_hot = soil_temp > soil_temp_max

            # ---- Soil humidity ----
            if soil_hum >= soil_hum_max:
                soil_status = "WET"
            elif soil_hum <= soil_hum_min:
                soil_status = "DRY"
            else:
                soil_status = "OK"

            # Soil hysteresis
            if soil_status == "WET" and soil_hum <= (soil_hum_max - 5):
                soil_status = "OK"
            if soil_status == "DRY" and soil_hum >= (soil_hum_min + 5):
                soil_status = "OK"

            # ---- Air humidity ----
            if air_hum < air_hum_min:
                air_hum_status = "DRY"
            elif air_hum > air_hum_max:
                air_hum_status = "HIGH"
            else:
                air_hum_status = "OK"

            # Air humidity hysteresis
            if air_hum_status == "HIGH" and air_hum <= (air_hum_max - 5):
                air_hum_status = "OK"
            if air_hum_status == "DRY" and air_hum >= (air_hum_min + 5):
                air_hum_status = "OK"

            # ---- Other conditions ----
            air_too_hot      = air_temp > air_temp_max
            co2_high         = co2_ppm > co2_max
            air_quality_bad  = ratio_smooth < 1.6

            # PRIORITY 1 — Soil temperature safety
            if soil_too_hot:
                mixer = True
                fan = True
                spray = False
                timers["mixer_on_until"] = max(timers["mixer_on_until"], now + mixer_duration)
                timers["mixer_cooldown_until"] = max(timers["mixer_cooldown_until"], now + mixer_duration + mixer_interval)

            # PRIORITY 2 — Soil humidity control
            if soil_status == "WET":
                fan = True
                spray = False
                if control.chk_mixer_var.get():
                    mixer = request_mixer_start("soil_wet")

            elif soil_status == "DRY":
                if control.chk_mixer_var.get():
                    mixer = request_mixer_start("soil_dry")
                if control.chk_spray_var.get():
                    spray = request_spray_start("soil_dry")

            # PRIORITY 3 — Air humidity + temp (only when soil OK)
            soil_ok = (soil_status == "OK")
            if soil_ok:
                if air_too_hot and control.chk_fan_var.get():
                    fan = True
                if air_hum_status == "HIGH" and control.chk_fan_var.get():
                    fan = True
                    spray = False
                if air_hum_status == "DRY" and control.chk_spray_var.get():
                    spray = request_spray_start("air_dry")

            # PRIORITY 4 — Air quality / CO2
            if (air_quality_bad or co2_high) and control.chk_fan_var.get():
                fan = True

            # 最終的なFan判定（OR）
            fan_required = (
                soil_too_hot or
                soil_status == "WET" or
                air_too_hot or
                air_hum_status == "HIGH" or
                air_quality_bad or
                co2_high
            )
            fan = fan_required

            # APPLY FINAL STATES（手動中はAUTOで上書きしない）
            if not gpio.is_manual("Mixer"):
                gpio.set("Mixer", mixer if control.chk_mixer_var.get() else False)
            if not gpio.is_manual("Fan"):
                gpio.set("Fan", fan if control.chk_fan_var.get() else False)
            if not gpio.is_manual("Spray"):
                gpio.set("Spray", spray if control.chk_spray_var.get() else False)

            # ボタン表示更新
            for n in ["Mixer", "Fan", "Spray", "LED"]:
                update_gpio_button(n)

        except Exception as e:
            print("Control logic error:", e)

        # ============================================================
        # ローカルログ（毎回）
        # ============================================================
        try:
            log_sensor_data(
                bin_id=2,
                air_temp=air_temp,
                air_hum=air_hum,
                air_pres=air_pres,
                soil_temp=soil_temp,
                soil_hum=soil_hum,
                lux=lux,
                air_ratio=ratio_smooth,
                co2_ppm=co2_ppm,
            )
        except Exception as e:
            print("Local log error:", e)

        # ============================================================
        # DB送信用バッファ更新（直近N回）
        # ============================================================
        reading_buffer.append(
            {
                "air_temp": air_temp,
                "air_humidity": air_hum,
                "air_pressure": air_pres,
                "soil_temp": soil_temp,
                "soil_humidity": soil_hum,
                "lux": lux,
                "air_quality_ratio": ratio_smooth,
                "co2_ppm": co2_ppm,
            }
        )
        if len(reading_buffer) > DB_AVG_WINDOW:
            reading_buffer.pop(0)

        # ============================================================
        # DB upload（一定周期で直近N回平均）
        # force_now=True は「Refresh Data」ボタンのみ
        # ============================================================
        if force_now or time.time() >= next_db_time:
            if reading_buffer:
                try:
                    n = len(reading_buffer)
                    avg = {key: sum(p[key] for p in reading_buffer) / n for key in reading_buffer[0].keys()}

                    db.insert_compost_data(
                        bin_id=2,
                        air_temp=avg["air_temp"],
                        air_humidity=avg["air_humidity"],
                        air_pressure=avg["air_pressure"],
                        soil_temp=avg["soil_temp"],
                        soil_humidity=avg["soil_humidity"],
                        lux=avg["lux"],
                        air_quality_ratio=avg["air_quality_ratio"],
                        co2_ppm=avg["co2_ppm"],
                    )
                    control.lbl_db_status.configure(text=f"Database: Uploaded (avg {n} readings)", text_color=ACCENT)
                    dprint(f"[DB] inserted avg({n})")

                except Exception as e:
                    control.lbl_db_status.configure(text="Database: Failed", text_color="red")
                    print("Neon database upload error:", e)

            next_db_time = time.time() + float(control.text_dbinterval.get())

        # 次回ループ予約
        try:
            sensor_interval = max(0.5, float(control.text_sensorinterval.get()))
        except Exception:
            sensor_interval = 3.0
        control.after(int(sensor_interval * 1000), F_button_getdata_function)

    # ============================================================
    # LIVE VIEW（照準プレビュー） + 手動撮影
    # ============================================================
    LIVE_FPS = 15

    def live_view_loop():
        """最新フレーム + キャッシュ済み検出枠をキャンバスに描画（照準用）"""
        try:
            if camera is not None and control.live_var.get():
                frame = camera.get_latest_frame()
                if frame is not None:
                    for det in list(camera.last_detections):
                        x1, y1, x2, y2 = det["box"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
                        cv2.putText(frame, f"{det['label']} {det['conf']:.2f}",
                                    (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 180, 0), 2)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = ImageOps.pad(Image.fromarray(rgb), (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
                    control.live_imgtk = ImageTk.PhotoImage(img)
                    control.canvas1.delete("all")
                    control.canvas1.create_image(0, 0, image=control.live_imgtk, anchor=ctk.NW)
        except Exception as e:
            print(f"[LIVE] loop error: {e}")
        control.after(int(1000 / LIVE_FPS), live_view_loop)

    def update_cn_label(cn):
        if cn and cn.get("cn_ratio") is not None:
            control.lbl_cn.configure(
                text=f"C/N {cn['cn_ratio']} : 1  •  add ~{cn['browns_grams']} g browns")
        else:
            control.lbl_cn.configure(text="C/N: -- (no veg detected)")

    def on_class_level_change(class_name, label):
        """あるクラスのレベル変更 → 再撮影せず C/N を再計算"""
        if camera is None:
            return
        camera.cn_model.portion_levels[class_name] = _LEVELS.get(label, "medium")
        cn = camera.cn_model.recompute_cn()
        camera.last_cn = cn
        update_cn_label(cn)

    def build_level_selectors():
        """撮影で検出されたクラスごとに、レベル選択ドロップダウンを作る"""
        for w in control.levels_container.winfo_children():
            w.destroy()
        control.level_menus = {}

        dets = camera.last_detections if camera else []
        classes = []
        for d in dets:
            if d["label"] not in classes:
                classes.append(d["label"])

        if not classes:
            ctk.CTkLabel(control.levels_container, text="(no veg detected)",
                         text_color=TEXT_MUTED, font=("Arial", 12)).pack()
            return

        for name in classes:
            rowf = ctk.CTkFrame(control.levels_container, fg_color="transparent")
            rowf.pack(fill="x", pady=2)
            ctk.CTkLabel(rowf, text=name, text_color=TEXT_MUTED, font=("Arial", 12),
                         width=110, anchor="w").pack(side="left", padx=(0, 8))
            cur_key = camera.cn_model.portion_levels.get(name, control.portion_level)
            menu = ctk.CTkOptionMenu(
                rowf, values=list(_LEVELS.keys()), width=120,
                command=lambda label, n=name: on_class_level_change(n, label),
            )
            menu.set(_KEY_TO_LABEL.get(cur_key, "Medium"))
            menu.pack(side="left")
            control.level_menus[name] = menu

    def refresh_result_labels():
        """撮影後に C/N・Stage ラベルとクラス別レベル選択を更新"""
        try:
            if camera is None:
                return
            update_cn_label(camera.last_cn)
            stage, conf = camera.last_stage
            control.lbl_stage.configure(text=f"Stage: {stage} ({conf:.2f})")
            build_level_selectors()
        except Exception as e:
            print(f"[LIVE] label refresh error: {e}")

    def on_new_photo_cb(path):
        """カメラスレッドから呼ばれる。GUI更新はメインスレッドへ委譲（Tkinter安全）"""
        control.after(0, lambda: (F_update_image(path), refresh_result_labels()))

    def manual_capture():
        """Capture Now ボタン：今のフレームで撮影 → 2モデル推論 → 結果表示"""
        if camera is None:
            return
        control.lbl_db_status.configure(text="Capturing...", text_color="#b8860b")
        # プレビューを止め、枠付き結果が live ループで上書きされないようにする
        control.live_var.set(False)
        camera.capture_now()

    # ============================================================
    # CAMERA STARTUP（撮影時のみLED点灯、手動LEDなら触らない）
    # ============================================================
    def led_on_before_capture():
        nonlocal capture_led_on

        # 手動LEDの場合、撮影フックで制御しない
        if gpio.is_manual("LED"):
            return

        try:
            lux, _, _ = dev.F_tsl2561()
        except Exception as e:
            lux = 0.0
            print(f"TSL2561 read error before capture: {e}")

        if control.chk_light_var.get() and lux < LIGHT_MIN_LUX:
            gpio.set("LED", True)
            capture_led_on = True
            dprint(f"[LED] ON(capture) lux={lux:.2f}")
        else:
            capture_led_on = False

    def led_off_after_capture():
        nonlocal capture_led_on

        if gpio.is_manual("LED"):
            return

        if capture_led_on:
            try:
                gpio.set("LED", False)
                dprint("[LED] OFF(post-capture)")
            except Exception as e:
                print("LED OFF error:", e)
            finally:
                capture_led_on = False

    try:
        camera = OpenCVCamera(
            save_dir="./Photos",
            interval_sec=float(control.text_camera_interval.get()),
            on_new_photo=on_new_photo_cb,
            before_capture=led_on_before_capture,
            after_capture=led_off_after_capture,
            overlay_interval=0.0,   # 照準中の自動推論はしない（手動運用）
            auto_capture=False,     # タイマー撮影オフ：Capture Now ボタンのみ
        )
        camera.start()
        camera.cn_model.portion_level = getattr(control, "portion_level", "medium")
        control.after(500, live_view_loop)
    except Exception as e:
        print(f"Camera failed to start: {e}")

    # 起動時に最新画像があれば表示
    latest_ai = "./Photos/Current_Image/latest_ai.jpg"
    if os.path.exists(latest_ai):
        F_update_image(latest_ai)
    else:
        control.canvas1.create_text(
            IMAGE_SIZE // 2,
            IMAGE_SIZE // 2,
            text="Camera sleeping...\nWaiting for first capture...",
            fill="white",
            font=("Arial", 14),
            anchor="center",
        )

    # ============================================================
    # 終了処理
    # ============================================================
    def on_close():
        try:
            if camera:
                camera.stop()
            gpio.cleanup()
        except Exception as e:
            print(f"Cleanup error: {e}")
        finally:
            control.destroy()
            sys.exit(0)

    # ============================================================
    # MAIN LOOP
    # ============================================================
    control.after(1000, F_button_getdata_function)  # 起動後1秒でセンサーループ開始（force_now=False）
    control.protocol("WM_DELETE_WINDOW", on_close)
    control.mainloop()

if __name__ == "__main__":
    F_MakeScreen_Demo()
