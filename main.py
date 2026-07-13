#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, sys, time, datetime, json, math, threading
from collections import deque
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageOps
import pigpio

from Device_Library import DeviceClass
from Database_Neon import DatabaseTool
from open_cv import OpenCVCamera
from LocalLogger import log_sensor_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = 360  # smaller so the whole page fits a 1080p screen

# ---- Theme palette (dark + compost green) ----
BG = "#0e1512"            # app background
CARD_BG = "#141b17"       # card background
CARD_BORDER = "#26302b"   # subtle card border
TILE_BG = "#1a221d"       # inner tile / pill background
ACCENT = "#34c759"        # bright green accent
ACCENT_HOVER = "#248a3d"  # pressed/hover
TEXT = "#e7ece8"          # primary text
TEXT_MUTED = "#8b948d"    # secondary text
RED = "#e5484d"           # alert
YELLOW = "#f2a53a"        # caution
GREY = "#6b7280"          # neutral
BTN_OFF = "#232b26"       # relay off

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

        # main.py の1秒センサーループとカメラの撮影スレッド(LED制御)が
        # 同時にpigpioへ書き込みうるため、排他制御が必要
        self._lock = threading.RLock()

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
        with self._lock:
            if name in self.pins:
                self.pi.write(self.pins[name], 0 if on else 1)
                self.state[name] = on

    def toggle(self, name):
        """互換性のため残す（現在GUIボタンでは未使用）"""
        with self._lock:
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
        with self._lock:
            if name in self.pins:
                self.manual[name] = True
                self.set(name, on)
                print(f"{name}: {'ON' if on else 'OFF'} (MANUAL)")

    def clear_manual(self, name):
        """手動制御を解除し、AUTO制御に戻す"""
        with self._lock:
            if name in self.pins:
                self.manual[name] = False
                print(f"{name}: MANUAL CLEARED → AUTO")

    def is_manual(self, name):
        with self._lock:
            return self.manual.get(name, False)

    def get(self, name):
        with self._lock:
            return self.state.get(name, False)

    def cleanup(self):
        """終了時に全リレーを安全OFFへ戻す"""
        with self._lock:
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
    MQ135_R0_PATH = os.environ.get("MQ135_R0_PATH", os.path.join(BASE_DIR, "mq135_r0.json"))
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
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("green")
    control.configure(fg_color=BG)
    _sw = control.winfo_screenwidth()
    _sh = control.winfo_screenheight()
    _w = min(1360, _sw)
    _h = min(1000, _sh - 80)
    control.geometry(f"{_w}x{_h}+0+0")
    control.grid_rowconfigure(0, weight=0)
    control.grid_rowconfigure(1, weight=1)
    control.grid_columnconfigure(0, weight=5, uniform="cols")
    control.grid_columnconfigure(1, weight=4, uniform="cols")

    # ---- reusable builders ----
    def make_card(parent):
        return ctk.CTkFrame(parent, corner_radius=16, fg_color=CARD_BG,
                            border_width=1, border_color=CARD_BORDER)

    def section_header(parent, text, icon=None):
        hf = ctk.CTkFrame(parent, fg_color="transparent")
        hf.pack(fill="x", padx=16, pady=(14, 8))
        bar = ctk.CTkFrame(hf, width=4, height=16, corner_radius=2, fg_color=ACCENT)
        bar.pack(side="left", pady=1); bar.pack_propagate(False)
        ctk.CTkLabel(hf, text=text.upper(), text_color=TEXT_MUTED,
                     font=("Arial", 12, "bold")).pack(side="left", padx=(10, 0))
        return hf

    def make_pill(parent, text, color, bg=None):
        return ctk.CTkLabel(parent, text=text, text_color=color, font=("Arial", 12, "bold"),
                            fg_color=(bg or TILE_BG), corner_radius=13)

    # ============================================================
    # HEADER
    # ============================================================
    header = make_card(control)
    header.grid(row=0, column=0, columnspan=2, padx=16, pady=(14, 6), sticky="ew")
    header.grid_columnconfigure(1, weight=1)

    logo = ctk.CTkFrame(header, width=42, height=42, corner_radius=12, fg_color="#173a27")
    logo.grid(row=0, column=0, padx=(16, 0), pady=14); logo.grid_propagate(False)
    ctk.CTkLabel(logo, text="C", text_color="#eafff0", font=("Arial", 18, "bold")).pack(expand=True)

    title_box = ctk.CTkFrame(header, fg_color="transparent")
    title_box.grid(row=0, column=1, sticky="w", padx=(12, 0))
    ctk.CTkLabel(title_box, text="Compost Management System",
                 font=("Arial", 22, "bold"), text_color=TEXT).pack(anchor="w")
    control.lbl_subtitle = ctk.CTkLabel(title_box, text="Live Monitor",
                                        font=("Arial", 12), text_color=TEXT_MUTED)
    control.lbl_subtitle.pack(anchor="w")

    status_box = ctk.CTkFrame(header, fg_color="transparent")
    status_box.grid(row=0, column=2, sticky="e", padx=16)
    control.lbl_clock = ctk.CTkLabel(status_box, text="--:--:--", font=("Consolas", 15, "bold"),
                                     text_color=TEXT, fg_color=TILE_BG, corner_radius=13,
                                     width=96, height=30)
    control.lbl_clock.pack(side="left", padx=(0, 8))
    control.pill_db = make_pill(status_box, "\u25CF Database: Idle", TEXT_MUTED)
    control.pill_db.pack(side="left", padx=(0, 8), ipadx=10, ipady=5)
    control.pill_live = make_pill(status_box, "\u25CF LIVE", ACCENT)
    control.pill_live.pack(side="left", ipadx=10, ipady=5)

    # ============================================================
    # LEFT COLUMN
    # ============================================================
    left = ctk.CTkScrollableFrame(control, fg_color="transparent")
    left.grid(row=1, column=0, padx=(16, 8), pady=(6, 16), sticky="nsew")
    left.grid_columnconfigure(0, weight=1)

    # ---- Sensor Data ----
    sensor_card = make_card(left)
    sensor_card.pack(fill="x", pady=(0, 12))
    section_header(sensor_card, "Sensor Data", "\u2248")
    sensors_grid = ctk.CTkFrame(sensor_card, fg_color="transparent")
    sensors_grid.pack(fill="x", padx=10, pady=(0, 10))
    sensors_grid.grid_columnconfigure((0, 1), weight=1, uniform="stat")

    control._stat_icons = []

    def make_icon(kind, size=16, c=(150, 158, 152, 255)):
        F = 6
        S = size * F
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        lw = max(2, int(S * 0.07))
        if kind == "thermometer":
            # stem drawn from primitives (rounded_rectangle needs Pillow>=8.2)
            x = S * 0.5
            w2 = S * 0.11
            d.ellipse([x - w2, S*0.10 - w2, x + w2, S*0.10 + w2], outline=c, width=lw)  # rounded top
            d.line([(x - w2, S*0.10), (x - w2, S*0.60)], fill=c, width=lw)              # left side
            d.line([(x + w2, S*0.10), (x + w2, S*0.60)], fill=c, width=lw)              # right side
            r = S*0.17
            d.ellipse([x - r, S*0.58, x + r, S*0.58 + 2*r], fill=c)                     # bulb
            d.line([(x, S*0.34), (x, S*0.64)], fill=c, width=int(lw*1.6))               # mercury
        elif kind == "droplet":
            cx = S*0.5; r = S*0.24; cy = S*0.60
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
            d.polygon([(cx, S*0.14), (cx - r*0.92, cy), (cx + r*0.92, cy)], fill=c)
        elif kind == "gauge":
            m = S*0.18
            d.arc([m, m, S - m, S - m], start=180, end=360, fill=c, width=lw)
            cx = S*0.5; cy = S*0.62
            d.line([(cx, cy), (S*0.66, S*0.36)], fill=c, width=lw)
            d.ellipse([cx - S*0.05, cy - S*0.05, cx + S*0.05, cy + S*0.05], fill=c)
        elif kind == "sun":
            cx = cy = S*0.5; r = S*0.15
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
            for a in range(0, 360, 45):
                rad = math.radians(a)
                d.line([(cx + math.cos(rad)*r*1.5, cy + math.sin(rad)*r*1.5),
                        (cx + math.cos(rad)*r*2.3, cy + math.sin(rad)*r*2.3)], fill=c, width=lw)
        elif kind == "cloud":
            d.ellipse([S*0.14, S*0.40, S*0.52, S*0.72], fill=c)
            d.ellipse([S*0.38, S*0.28, S*0.74, S*0.64], fill=c)
            d.ellipse([S*0.56, S*0.44, S*0.86, S*0.72], fill=c)
            d.rectangle([S*0.28, S*0.58, S*0.76, S*0.72], fill=c)
        elif kind == "wave":
            pts = [(S*0.12 + S*0.76*(i/100.0),
                    S*0.5 - math.sin((i/100.0)*math.pi*2)*S*0.20) for i in range(101)]
            d.line(pts, fill=c, width=lw, joint="curve")
        else:
            d.ellipse([S*0.3, S*0.3, S*0.7, S*0.7], outline=c, width=lw)
        try:
            _RESAMPLE = Image.Resampling.LANCZOS   # Pillow >= 9.1
        except AttributeError:
            _RESAMPLE = Image.LANCZOS              # older Pillow
        img = img.resize((size, size), _RESAMPLE)
        ic = ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
        control._stat_icons.append(ic)
        return ic

    def add_stat_card(row, col, label, attr, icon="", spark=False):
        card = ctk.CTkFrame(sensors_grid, corner_radius=12, fg_color=TILE_BG,
                            border_width=1, border_color=CARD_BORDER)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="ew")
        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=(6, 0))
        ctk.CTkLabel(top, text=label, text_color=TEXT_MUTED, font=("Arial", 10)).pack(side="left")
        if icon:
            ctk.CTkLabel(top, text="", image=make_icon(icon)).pack(side="right")
        v = ctk.CTkLabel(card, text="--", font=("Arial", 19, "bold"), text_color=TEXT)
        v.pack(anchor="w", padx=10, pady=(0, 0))
        setattr(control, attr, v)
        if spark:
            sc = ctk.CTkCanvas(card, height=26, bg=TILE_BG, highlightthickness=0)
            sc.pack(fill="x", padx=10, pady=(2, 0))
            setattr(control, attr + "_spark", sc)
        srow = ctk.CTkFrame(card, fg_color="transparent")
        srow.pack(fill="x", padx=10, pady=(0, 6))
        dot = ctk.CTkFrame(srow, width=7, height=7, corner_radius=4, fg_color=GREY)
        dot.pack(side="left", pady=2)
        stt = ctk.CTkLabel(srow, text="--", text_color=TEXT_MUTED, font=("Arial", 10))
        stt.pack(side="left", padx=(6, 0))
        setattr(control, attr + "_dot", dot)
        setattr(control, attr + "_status", stt)
        return card, dot, stt

    add_stat_card(0, 0, "Air Temp (\u00B0C)", "text_Temp", "thermometer")
    add_stat_card(0, 1, "Air Hum (%)", "text_Humi", "droplet")
    add_stat_card(1, 0, "Pressure (hPa)", "text_Pres", "gauge")
    add_stat_card(1, 1, "Light (Lux)", "text_Lux", "sun")
    add_stat_card(2, 0, "Soil Temp (\u00B0C)", "text_Temp_SHT", "thermometer", spark=True)
    add_stat_card(2, 1, "Soil Hum (%)", "text_Humi_SHT", "droplet")
    add_stat_card(3, 0, "CO2 (ppm)", "text_Co2", "cloud", spark=True)
    _, control.air_indicator, control.air_status_label = add_stat_card(
        3, 1, "Air Quality (Rs/R0)", "text_MQ135", "wave")

    # ---- GPIO Controls ----
    gpio_frame = make_card(left)
    gpio_frame.pack(fill="x", pady=(0, 12))
    gpio_hdr = section_header(gpio_frame, "GPIO Controls")
    btn_row = ctk.CTkFrame(gpio_frame, fg_color="transparent")
    btn_row.pack(fill="x", padx=14, pady=(0, 4))
    btn_row.grid_columnconfigure((0, 1), weight=1)
    for i, n in enumerate(["LED", "Fan", "Spray", "Mixer"]):
        b = ctk.CTkButton(
            btn_row, text=f"{n} OFF", height=42, fg_color=BTN_OFF, hover_color="#2b332e",
            text_color=TEXT, corner_radius=10, border_width=0, border_color=YELLOW,
            command=lambda x=n: (gpio.set_manual(x, not gpio.get(x)), update_gpio_button(x)),
        )
        b.grid(row=i // 2, column=i % 2, padx=6, pady=6, sticky="ew")
        b.bind("<Button-3>", lambda e, x=n: (gpio.clear_manual(x), update_gpio_button(x)))
        gpio_buttons[n] = b

    ctk.CTkLabel(gpio_frame, text="Click to toggle manually · right-click to return to AUTO",
                 text_color=TEXT_MUTED, font=("Arial", 10)).pack(anchor="w", padx=16, pady=(0, 8))

    db_pill = ctk.CTkFrame(gpio_hdr, fg_color=TILE_BG, corner_radius=13,
                           border_width=1, border_color=CARD_BORDER)
    db_pill.pack(side="right", padx=(0, 2))
    control.lbl_db_status = ctk.CTkLabel(db_pill, text="\u25CF DB Idle", text_color=TEXT_MUTED,
                                         font=("Arial", 11, "bold"))
    control.lbl_db_status.pack(padx=10, pady=3)

    # ---- Control Ranges + Intervals (side by side) ----
    def add_range_row(parent, label, mn, mx, width=54):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(row, text=label, text_color=TEXT_MUTED, anchor="w",
                     font=("Arial", 12)).pack(fill="x")
        er = ctk.CTkFrame(row, fg_color="transparent")
        er.pack(fill="x", pady=(3, 0))
        e1 = ctk.CTkEntry(er, width=width); e1.insert(0, mn); e1.pack(side="left")
        ctk.CTkLabel(er, text="\u2013", text_color=TEXT_MUTED).pack(side="left", padx=6)
        e2 = ctk.CTkEntry(er, width=width); e2.insert(0, mx); e2.pack(side="left")
        return e1, e2

    def add_single_row(parent, label, default, width=54):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(row, text=label, text_color=TEXT_MUTED, anchor="w",
                     font=("Arial", 12)).pack(fill="x")
        e = ctk.CTkEntry(row, width=width); e.insert(0, default); e.pack(anchor="w", pady=(3, 0))
        return e

    def add_interval_row(parent, label, default):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(row, text=label, text_color=TEXT_MUTED, anchor="w",
                     font=("Arial", 12)).pack(fill="x")
        e = ctk.CTkEntry(row, width=90); e.insert(0, default); e.pack(anchor="w", pady=(3, 0))
        return e

    ri = ctk.CTkFrame(left, fg_color="transparent")
    ri.pack(fill="x", pady=(0, 12))
    ri.grid_columnconfigure((0, 1), weight=1, uniform="ri")

    ranges_card = make_card(ri)
    ranges_card.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
    section_header(ranges_card, "Control Ranges", "\u2317")
    control.text_soilhum_min, control.text_soilhum_max = add_range_row(ranges_card, "Soil Humidity (%)", "35", "55")
    control.text_soiltemp_min, control.text_soiltemp_max = add_range_row(ranges_card, "Soil Temp (\u00B0C)", "50", "60")
    control.text_airtemp_max = add_single_row(ranges_card, "Air Temp Max (\u00B0C)", "40")
    control.text_airhum_min, control.text_airhum_max = add_range_row(ranges_card, "Air Humidity (%)", "50", "70")
    control.text_co2_max = add_single_row(ranges_card, "CO2 Max (ppm)", "2000")
    ctk.CTkLabel(ranges_card, text="", height=2).pack()

    intervals_card = make_card(ri)
    intervals_card.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
    section_header(intervals_card, "Intervals", "\u23F1")
    control.text_sensorinterval  = add_interval_row(intervals_card, "Sensor", "3")
    control.text_dbinterval      = add_interval_row(intervals_card, "Database", "60")
    control.text_mixer_interval  = add_interval_row(intervals_card, "Mixer", "1800")
    control.text_mixer_duration  = add_interval_row(intervals_card, "Mixer Duration", "60")
    control.text_camera_interval = add_interval_row(intervals_card, "Camera", "3600")
    control.text_spray_duration  = add_interval_row(intervals_card, "Spray Duration", "60")
    ctk.CTkLabel(intervals_card, text="", height=2).pack()

    # ---- Enable Controls ----
    enable_card = make_card(left)
    enable_card.pack(fill="x", pady=(0, 4))
    section_header(enable_card, "Enable Controls", "\u23FB")
    eg = ctk.CTkFrame(enable_card, fg_color="transparent")
    eg.pack(fill="x", padx=16, pady=(0, 14))
    eg.grid_columnconfigure((0, 1), weight=1)
    control.chk_fan_var   = ctk.BooleanVar(value=True)
    control.chk_spray_var = ctk.BooleanVar(value=True)
    control.chk_light_var = ctk.BooleanVar(value=True)
    control.chk_mixer_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(eg, text="Enable Fan",   variable=control.chk_fan_var).grid(row=0, column=0, sticky="w", padx=5, pady=6)
    ctk.CTkCheckBox(eg, text="Enable Spray", variable=control.chk_spray_var).grid(row=0, column=1, sticky="w", padx=5, pady=6)
    ctk.CTkCheckBox(eg, text="Enable Light", variable=control.chk_light_var).grid(row=1, column=0, sticky="w", padx=5, pady=6)
    ctk.CTkCheckBox(eg, text="Enable Mixer", variable=control.chk_mixer_var).grid(row=1, column=1, sticky="w", padx=5, pady=6)

    # ============================================================
    # RIGHT COLUMN
    # ============================================================
    right = ctk.CTkScrollableFrame(control, fg_color="transparent")
    right.grid(row=1, column=1, padx=(8, 16), pady=(6, 16), sticky="nsew")
    right.grid_columnconfigure(0, weight=1)

    _LEVELS = {"Light": "light", "Medium": "medium", "Heavy": "heavy", "Very Heavy": "very_heavy"}
    _KEY_TO_LABEL = {v: k for k, v in _LEVELS.items()}
    control.portion_level = "medium"

    # ---- Live Compost View ----
    live_card = make_card(right)
    live_card.pack(fill="x", pady=(0, 12))
    lv_head = ctk.CTkFrame(live_card, fg_color="transparent")
    lv_head.pack(fill="x", padx=16, pady=(14, 8))
    lv_badge = ctk.CTkFrame(lv_head, width=4, height=16, corner_radius=2, fg_color=ACCENT)
    lv_badge.pack(side="left", pady=1); lv_badge.pack_propagate(False)
    ctk.CTkLabel(lv_head, text="LIVE COMPOST VIEW", text_color=TEXT_MUTED,
                 font=("Arial", 12, "bold")).pack(side="left", padx=(10, 0))
    rec = ctk.CTkFrame(lv_head, fg_color="transparent")
    ctk.CTkFrame(rec, width=8, height=8, corner_radius=4, fg_color=RED).pack(side="left", pady=4)
    ctk.CTkLabel(rec, text="REC", text_color=RED, font=("Arial", 11, "bold")).pack(side="left", padx=(5, 0))

    control.canvas1 = ctk.CTkCanvas(live_card, width=IMAGE_SIZE, height=IMAGE_SIZE,
                                    bg="#0a0f0c", highlightthickness=0)
    control.canvas1.pack(padx=16, pady=(0, 6))
    control.lbl_photo_timestamp = ctk.CTkLabel(live_card, text="No image loaded",
                                               text_color=TEXT_MUTED, font=("Arial", 11))
    control.lbl_photo_timestamp.pack(anchor="w", padx=16, pady=(0, 6))

    lvbar = ctk.CTkFrame(live_card, fg_color="transparent")
    lvbar.pack(fill="x", padx=16, pady=(0, 14))
    control.live_var = ctk.BooleanVar(value=False)

    def update_rec_indicator():
        if control.live_var.get():
            rec.pack(side="right")
        else:
            rec.pack_forget()

    update_rec_indicator()

    ctk.CTkCheckBox(lvbar, text="Live View (aim the camera)",
                    variable=control.live_var,
                    command=update_rec_indicator).pack(anchor="w", pady=(0, 8))
    control.btn_capture = ctk.CTkButton(
        lvbar, text="Capture Now", height=48, corner_radius=12,
        fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#0b3d1a",
        font=("Arial", 16, "bold"), command=lambda: manual_capture())
    control.btn_capture.pack(fill="x")

    # ---- Estimated Weight ----
    weight_card = make_card(right)
    weight_card.pack(fill="x", pady=(0, 12))
    wh = ctk.CTkFrame(weight_card, fg_color="transparent")
    wh.pack(fill="x", padx=16, pady=(14, 8))
    w_badge = ctk.CTkFrame(wh, width=4, height=16, corner_radius=2, fg_color=ACCENT)
    w_badge.pack(side="left", pady=1); w_badge.pack_propagate(False)
    ctk.CTkLabel(wh, text="ESTIMATED WEIGHT", text_color=TEXT_MUTED,
                 font=("Arial", 12, "bold")).pack(side="left", padx=(10, 0))
    ctk.CTkLabel(wh, text="vision estimate", text_color=TEXT_MUTED, font=("Arial", 11)).pack(side="right")
    control.weight_container = ctk.CTkFrame(weight_card, fg_color="transparent")
    control.weight_container.pack(fill="x", padx=18, pady=(0, 8))
    ctk.CTkLabel(control.weight_container, text="(capture to estimate)",
                 text_color=TEXT_MUTED, font=("Arial", 12)).pack(anchor="w")
    total_row = ctk.CTkFrame(weight_card, fg_color=TILE_BG, corner_radius=12)
    total_row.pack(fill="x", padx=14, pady=(0, 14))
    ctk.CTkLabel(total_row, text="TOTAL", text_color=TEXT, font=("Arial", 14, "bold")).pack(side="left", padx=16, pady=12)
    control.lbl_weight_total = ctk.CTkLabel(total_row, text="-- g", text_color=ACCENT, font=("Arial", 18, "bold"))
    control.lbl_weight_total.pack(side="right", padx=16, pady=12)

    # ---- Carbon / Nitrogen ----
    cn_card = make_card(right)
    cn_card.pack(fill="x", pady=(0, 4))
    ch = ctk.CTkFrame(cn_card, fg_color="transparent")
    ch.pack(fill="x", padx=16, pady=(14, 6))
    c_badge = ctk.CTkFrame(ch, width=4, height=16, corner_radius=2, fg_color=ACCENT)
    c_badge.pack(side="left", pady=1); c_badge.pack_propagate(False)
    ctk.CTkLabel(ch, text="CARBON / NITROGEN", text_color=TEXT_MUTED,
                 font=("Arial", 12, "bold")).pack(side="left", padx=(10, 0))
    ctk.CTkLabel(cn_card, text="C / N RATIO", text_color=TEXT_MUTED, font=("Arial", 11, "bold")).pack(pady=(4, 0))
    control.lbl_cn = ctk.CTkLabel(cn_card, text="\u2014", font=("Arial", 30, "bold"), text_color=TEXT_MUTED)
    control.lbl_cn.pack()
    control.lbl_cn_sub = ctk.CTkLabel(cn_card, text="no veg detected", text_color=TEXT_MUTED, font=("Arial", 13))
    control.lbl_cn_sub.pack(pady=(0, 10))

    cn_bottom = ctk.CTkFrame(cn_card, fg_color="transparent")
    cn_bottom.pack(fill="x", padx=16, pady=(0, 14))
    cn_bottom.grid_columnconfigure((0, 1), weight=1, uniform="cnb")
    stage_box = ctk.CTkFrame(cn_bottom, fg_color=TILE_BG, corner_radius=12)
    stage_box.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
    ctk.CTkLabel(stage_box, text="Stage", text_color=TEXT_MUTED, font=("Arial", 11)).pack(anchor="w", padx=12, pady=(10, 0))
    control.lbl_stage = ctk.CTkLabel(stage_box, text="unknown (0.00)", text_color=TEXT, font=("Arial", 14, "bold"))
    control.lbl_stage.pack(anchor="w", padx=12, pady=(0, 10))
    lvl_box = ctk.CTkFrame(cn_bottom, fg_color=TILE_BG, corner_radius=12)
    lvl_box.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
    ctk.CTkLabel(lvl_box, text="Fill level / class", text_color=TEXT_MUTED, font=("Arial", 11)).pack(anchor="w", padx=12, pady=(10, 0))
    control.levels_container = ctk.CTkFrame(lvl_box, fg_color="transparent")
    control.levels_container.pack(fill="x", padx=8, pady=(2, 10))
    control.levels_container.grid_columnconfigure(0, weight=1)
    control.level_hint = ctk.CTkLabel(control.levels_container, text="\u2014",
                                      text_color=TEXT_MUTED, font=("Arial", 12))
    control.level_hint.pack(anchor="w")

    # ============================================================
    # HELPERS
    # ============================================================
    def update_gpio_button(name):
        """GPIOボタン表示更新（ON/OFFの色と文字、MANUAL中は黄枠+ラベル）"""
        if name in gpio_buttons:
            state = gpio.get(name)
            manual = gpio.is_manual(name)
            btn = gpio_buttons[name]
            label = f"{name} {'ON' if state else 'OFF'}" + ("  (M)" if manual else "")
            btn.configure(text=label,
                          fg_color=ACCENT if state else BTN_OFF,
                          hover_color=ACCENT_HOVER if state else "#2b332e",
                          border_width=2 if manual else 0)

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

    CLASS_COLORS = {"carrot_peels": "#e0894b", "mixed_veg": ACCENT, "onion": "#c9b06a"}

    def set_stat(attr, text, dot_color, text_color=None):
        try:
            getattr(control, attr + "_dot").configure(fg_color=dot_color)
            getattr(control, attr + "_status").configure(text=text, text_color=(text_color or TEXT_MUTED))
        except Exception:
            pass

    # ---- Sparkline history (last ~10 min at a 3 s sensor interval) ----
    APP_START = time.time()
    CO2_WARMUP_SECONDS = 180        # MH-Z19C returns a placeholder while heating (~1-3 min)
    CO2_CEILING = 5000.0            # default range max: readings clip here
    spark_history = {
        "text_Temp_SHT": deque(maxlen=200),
        "text_Co2": deque(maxlen=200),
    }

    def draw_sparkline(attr, values, color=ACCENT):
        sc = getattr(control, attr + "_spark", None)
        if sc is None:
            return
        try:
            sc.delete("all")
            vals = [v for v in values if v is not None]
            if len(vals) < 2:
                return
            w = sc.winfo_width()
            if w <= 1:
                w = 150
            h = 26
            pad = 2
            vmin, vmax = min(vals), max(vals)
            rng = (vmax - vmin) or 1.0
            n = len(vals)
            pts = []
            for i, v in enumerate(vals):
                x = pad + (w - 2 * pad) * (i / (n - 1))
                y = (h - pad) - (h - 2 * pad) * ((v - vmin) / rng)
                pts.extend((x, y))
            sc.create_line(*pts, fill=color, width=2, smooth=True)
            # min/max hint in the corner, very muted
            sc.create_text(w - pad, 2, text=f"{vmax:.0f}", anchor="ne",
                           fill=TEXT_MUTED, font=("Arial", 7))
            sc.create_text(w - pad, h - 2, text=f"{vmin:.0f}", anchor="se",
                           fill=TEXT_MUTED, font=("Arial", 7))
        except Exception:
            pass

    def update_sparklines():
        draw_sparkline("text_Temp_SHT", spark_history["text_Temp_SHT"])
        co2_vals = spark_history["text_Co2"]
        # red trend when the latest reading is pegged at the ceiling
        col = RED if (co2_vals and co2_vals[-1] >= CO2_CEILING - 10) else ACCENT
        draw_sparkline("text_Co2", co2_vals, color=col)

    def update_stat_statuses(air_temp, air_hum, air_pres, lux, soil_temp, soil_hum, co2_ppm):
        def fget(name, d):
            try:
                return float(getattr(control, name).get())
            except Exception:
                return d
        at_max = fget("text_airtemp_max", 40.0)
        ah_min = fget("text_airhum_min", 50.0); ah_max = fget("text_airhum_max", 70.0)
        sh_min = fget("text_soilhum_min", 35.0); sh_max = fget("text_soilhum_max", 55.0)
        st_min = fget("text_soiltemp_min", 50.0); st_max = fget("text_soiltemp_max", 60.0)
        co2_max = fget("text_co2_max", 2000.0)

        hot = air_temp > at_max
        set_stat("text_Temp", "High" if hot else "Normal", RED if hot else ACCENT, RED if hot else TEXT_MUTED)

        if air_hum < ah_min:
            set_stat("text_Humi", "Low", YELLOW, YELLOW)
        elif air_hum > ah_max:
            set_stat("text_Humi", "High", RED, RED)
        else:
            set_stat("text_Humi", "In range", ACCENT, TEXT_MUTED)

        set_stat("text_Pres", "Nominal", ACCENT, TEXT_MUTED)

        if lux < LIGHT_MIN_LUX:
            set_stat("text_Lux", "Dark", GREY, TEXT_MUTED)
        else:
            set_stat("text_Lux", "Lit", ACCENT, TEXT_MUTED)

        if soil_temp < st_min:
            set_stat("text_Temp_SHT", "Below target", RED, RED)
        elif soil_temp > st_max:
            set_stat("text_Temp_SHT", "Above target", RED, RED)
        else:
            set_stat("text_Temp_SHT", "In range", ACCENT, TEXT_MUTED)

        if soil_hum < sh_min:
            set_stat("text_Humi_SHT", "Below target", RED, RED)
        elif soil_hum > sh_max:
            set_stat("text_Humi_SHT", "Above target", RED, RED)
        else:
            set_stat("text_Humi_SHT", "In range", ACCENT, TEXT_MUTED)

        # CO2 with sensor-state awareness (warm-up / ceiling / read error)
        if co2_ppm <= 0:
            set_stat("text_Co2", "Read error", RED, RED)
        elif (time.time() - APP_START) < CO2_WARMUP_SECONDS:
            set_stat("text_Co2", "Warming up", YELLOW, YELLOW)
        elif co2_ppm >= CO2_CEILING - 10:
            set_stat("text_Co2", "\u22655000 (max)", RED, RED)
        elif co2_ppm > co2_max:
            set_stat("text_Co2", "High", RED, RED)
        else:
            set_stat("text_Co2", "Normal", ACCENT, TEXT_MUTED)

    def tick_clock():
        now = datetime.datetime.now()
        try:
            control.lbl_clock.configure(text=now.strftime("%H:%M:%S"))
            control.lbl_subtitle.configure(text="Live Monitor \u00B7 " + now.strftime("%a, %b %d %Y"))
            raw = str(control.lbl_db_status.cget("text"))
            col = control.lbl_db_status.cget("text_color")
            if "Uploaded" in raw:
                short = "Database: OK"
            elif "Failed" in raw:
                short = "Database: Failed"
            elif "Capturing" in raw:
                short = "Database: Busy"
            else:
                short = "Database: Idle"
            control.pill_db.configure(text="\u25CF " + short, text_color=col)
        except Exception:
            pass
        control.after(1000, tick_clock)

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

        control.btn_mixer_auto.configure(text="Start Auto Mixer", fg_color="#2563c9")

    bottom = ctk.CTkFrame(gpio_frame, fg_color="transparent")
    bottom.pack(fill="x", padx=14, pady=(2, 12))
    bottom.grid_columnconfigure((0, 1), weight=1)

    ctk.CTkButton(
        bottom,
        text="Refresh Data",
        fg_color=BTN_OFF, hover_color="#2b332e", text_color=TEXT, corner_radius=10,
        command=lambda: F_button_getdata_function(force_now=True),
    ).grid(row=0, column=0, padx=6, sticky="ew")

    control.btn_mixer_auto = ctk.CTkButton(
        bottom,
        text="Start Auto Mixer",
        fg_color="#2563c9", hover_color="#1d4ea0", corner_radius=10,
        command=lambda: (stop_auto_mixer() if mixer_running else start_auto_mixer()),
    )
    control.btn_mixer_auto.grid(row=0, column=1, padx=6, sticky="ew")

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

        update_stat_statuses(air_temp, air_hum, air_pres, lux, soil_temp, soil_hum, co2_ppm)

        # sparkline history: skip CO2 warm-up placeholder and failed reads
        spark_history["text_Temp_SHT"].append(soil_temp)
        if co2_ppm > 0 and (time.time() - APP_START) >= CO2_WARMUP_SECONDS:
            spark_history["text_Co2"].append(co2_ppm)
        update_sparklines()

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
                spray = False
                timers["mixer_on_until"] = max(timers["mixer_on_until"], now + mixer_duration)
                timers["mixer_cooldown_until"] = max(timers["mixer_cooldown_until"], now + mixer_duration + mixer_interval)

            # PRIORITY 2 — Soil humidity control
            if soil_status == "WET":
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
                if air_hum_status == "HIGH" and control.chk_fan_var.get():
                    spray = False
                if air_hum_status == "DRY" and control.chk_spray_var.get():
                    spray = request_spray_start("air_dry")

            # 最終的なFan判定（OR）。air_too_hot / air_hum HIGH は soil_ok の
            # ときだけトリガーにする（土壌DRY中にファンで追加乾燥させないため）
            air_fan_trigger = soil_ok and (air_too_hot or air_hum_status == "HIGH")
            fan_required = (
                soil_too_hot or
                soil_status == "WET" or
                air_fan_trigger or
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
        for w in control.weight_container.winfo_children():
            w.destroy()
        if cn and cn.get("cn_ratio") is not None:
            control.lbl_cn.configure(text=f"{cn['cn_ratio']} : 1", text_color=ACCENT)
            control.lbl_cn_sub.configure(text=f"add ~{cn['browns_grams']} g browns", text_color=TEXT_MUTED)
            pcm = cn.get("per_class_mass", {}) or {}
            total = cn.get("total_mass", 0.0) or 0.0
            if pcm:
                for name, grams in pcm.items():
                    rowf = ctk.CTkFrame(control.weight_container, fg_color="transparent")
                    rowf.pack(fill="x", pady=4)
                    ctk.CTkFrame(rowf, width=11, height=11, corner_radius=3,
                                 fg_color=CLASS_COLORS.get(name, GREY)).pack(side="left", pady=3)
                    ctk.CTkLabel(rowf, text=name.replace("_", " ").capitalize(),
                                 text_color=TEXT, font=("Arial", 13)).pack(side="left", padx=(10, 0))
                    ctk.CTkLabel(rowf, text=f"{grams:.1f} g", text_color=TEXT,
                                 font=("Arial", 13, "bold")).pack(side="right")
                control.lbl_weight_total.configure(text=f"{total:.1f} g", text_color=ACCENT)
            else:
                ctk.CTkLabel(control.weight_container, text="\u2014", text_color=TEXT_MUTED,
                             font=("Arial", 12)).pack(anchor="w")
                control.lbl_weight_total.configure(text="-- g", text_color=TEXT_MUTED)
        else:
            control.lbl_cn.configure(text="\u2014", text_color=TEXT_MUTED)
            control.lbl_cn_sub.configure(text="no veg detected", text_color=TEXT_MUTED)
            ctk.CTkLabel(control.weight_container, text="(capture to estimate)",
                         text_color=TEXT_MUTED, font=("Arial", 12)).pack(anchor="w")
            control.lbl_weight_total.configure(text="-- g", text_color=TEXT_MUTED)

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
            control.lbl_stage.configure(text=f"{stage} ({conf:.2f})")
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
    tick_clock()
    control.after(1000, F_button_getdata_function)  # 起動後1秒でセンサーループ開始（force_now=False）
    # ============================================================
    # MOUSE-WHEEL SCROLLING
    #   X11 (Raspberry Pi) sends wheel as Button-4 / Button-5; Windows/Mac
    #   send <MouseWheel>. Child widgets often swallow these before the
    #   scroll frame sees them, so we bind globally and forward the scroll
    #   to whichever column (left / right) the pointer is currently over.
    # ============================================================
    def _pointer_over(widget):
        try:
            x, y = control.winfo_pointerxy()
            w = control.winfo_containing(x, y)
            while w is not None:
                if w == widget:
                    return True
                w = w.master
        except Exception:
            pass
        return False

    def _scroll_target():
        if _pointer_over(right):
            return right
        if _pointer_over(left):
            return left
        return None

    def _wheel(event):
        target = _scroll_target()
        if target is None:
            return
        canvas = getattr(target, "_parent_canvas", None)
        if canvas is None:
            return
        if getattr(event, "num", None) == 4:      # X11 wheel up
            delta = -1
        elif getattr(event, "num", None) == 5:    # X11 wheel down
            delta = 1
        else:                                      # Windows/Mac
            delta = -1 if event.delta > 0 else 1
        try:
            canvas.yview_scroll(delta, "units")
        except Exception:
            pass

    control.bind_all("<Button-4>", _wheel, add="+")
    control.bind_all("<Button-5>", _wheel, add="+")
    control.bind_all("<MouseWheel>", _wheel, add="+")

    control.protocol("WM_DELETE_WINDOW", on_close)
    control.mainloop()

if __name__ == "__main__":
    F_MakeScreen_Demo()

