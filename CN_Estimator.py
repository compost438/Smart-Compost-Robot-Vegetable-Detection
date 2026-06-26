#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CN_Estimator.py
検出結果（バウンディングボックス）からコンポストの C/N 比を推定するモジュール

役割:
- YOLO の検出ボックス面積 → 推定質量（相対値）
- 質量加重 C/N 比を計算  C/N = Σmass / Σ(mass_i / cn_i)
- 目標 C/N (既定 30:1) に到達するために必要な「browns（炭素源）」の量を提案

重要な前提（必ず読むこと）:
- 出力されるグラムは「相対値」。Pi のカメラ・設置高さ・解像度で
  既知量を秤量・撮影して MASS_PER_FRAME を較正するまで実グラムにならない。
- 面積はフレーム全体に対する比率で計算（解像度非依存）。
- 1枚の2D画像から深さは測れないため DEPTH_FACTORS は固定値を使う。
  全体深さは式の中で相殺されるので、効くのはクラス間の相対差のみ。
"""

# ============================================================
# 設定値 (CONFIG) — すべて編集可能
# ============================================================

# クラスID → クラス名（学習時の順番と一致させること）
CLASS_NAMES = ["carrot_peels", "mixed_veg", "onion"]

# 各クラスの C/N 比（文献からの推定値。実測ではない）
CN_RATIO = {
    "carrot_peels": 27.0,
    "mixed_veg": 15.0,
    "onion": 15.0,
}

# クラスごとの深さ補正係数（light=0.5 / medium=1.0 / heavy=2.0 相当）
# 単一画像では深さ不明 → 既定は medium 相当(1.0)。クラス間で差をつけたい時のみ変更。
DEPTH_FACTORS = {
    "carrot_peels": 1.0,
    "mixed_veg": 1.0,
    "onion": 1.0,
}

# 面積比(フレーム全体に対する割合) → 相対質量 への変換係数。
# ★ Pi のカメラで既知量を撮影して較正するまでは相対値。
MASS_PER_FRAME = 1000.0

# 実測の充填レベル別グラム数（各レベルは「箱の1/4」を基準に計量した値）。
# 検出ボックスが箱の何クォーター分を覆うか × この値 = 推定グラム数。
# 例: 箱いっぱい(=4クォーター)の light mixed_veg = 25 * 4 = 100 g
PORTION_GRAMS = {
    "carrot_peels": {"light": 15,  "medium": 40,  "heavy": 100, "very_heavy": 250},
    "mixed_veg":    {"light": 25,  "medium": 75,  "heavy": 200, "very_heavy": 500},
    "onion":        {"light": 50,  "medium": 125, "heavy": 250, "very_heavy": 500},
}
# 1クォーター = フレーム面積の何割か（カメラ画角 ≒ 箱の上面を想定）
QUARTER_FRACTION = 0.25

# 目標 C/N 比（堆肥化の一般的目安）
TARGET_CN = 30.0

# browns（炭素源）の C/N 比。例: 乾いた落ち葉≈60, おがくず≈325, ダンボール≈350。
BROWNS_CN = 60.0


# ============================================================
# 推定本体
# ============================================================

def estimate_cn(
    boxes_xyxy,
    class_ids,
    frame_w,
    frame_h,
    cn_ratio=None,
    depth_factors=None,
    mass_per_frame=MASS_PER_FRAME,
    target_cn=TARGET_CN,
    browns_cn=BROWNS_CN,
    portion_level="medium",
    portion_levels=None,
    portion_grams=None,
):
    """
    検出ボックスから C/N 比と必要 browns 量を推定する。

    Parameters
    ----------
    boxes_xyxy : iterable of (x1, y1, x2, y2)
        元画像ピクセル座標のボックス（letterbox を戻した後の座標）
    class_ids : iterable of int
        各ボックスのクラスID（CLASS_NAMES のインデックス）
    frame_w, frame_h : int
        元画像の幅・高さ（面積比の正規化に使用）
    その他: 上の CONFIG を一時的に上書きしたい場合のみ指定

    Returns
    -------
    dict:
        cn_ratio        : 推定 C/N 比（検出なしなら None）
        browns_grams    : 目標に到達するための browns 推定量（相対グラム）
        total_mass      : 推定総質量（相対値）
        per_class_mass  : {class_name: mass} の内訳
        n_detections    : 採用した検出数
    """
    cn_ratio = cn_ratio or CN_RATIO
    depth_factors = depth_factors or DEPTH_FACTORS
    portion_grams = portion_grams or PORTION_GRAMS

    frame_area = float(frame_w * frame_h)
    if frame_area <= 0:
        return _empty_result()

    per_class_mass = {}
    total_mass = 0.0

    for box, cid in zip(boxes_xyxy, class_ids):
        cid = int(cid)
        if cid < 0 or cid >= len(CLASS_NAMES):
            continue
        name = CLASS_NAMES[cid]

        x1, y1, x2, y2 = box
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        area_frac = (w * h) / frame_area  # 0..1 フレーム比率

        if portion_level or portion_levels:
            # クラスごとのレベルがあれば優先、無ければ全体レベル
            lvl = portion_level
            if portion_levels:
                lvl = portion_levels.get(name, portion_level)
            # 実測グラム方式：このクラス・レベルの「1クォーター当たりグラム」×
            # ボックスが覆うクォーター数（= area_frac / 0.25）
            quarters = area_frac / QUARTER_FRACTION
            grams_q = portion_grams.get(name, {}).get(lvl)
            if grams_q is None:
                # レベル未登録なら面積方式にフォールバック
                mass = area_frac * mass_per_frame * depth_factors.get(name, 1.0)
            else:
                mass = grams_q * quarters
        else:
            # 旧・面積方式（較正前の相対値）
            mass = area_frac * mass_per_frame * depth_factors.get(name, 1.0)

        per_class_mass[name] = per_class_mass.get(name, 0.0) + mass
        total_mass += mass

    if total_mass <= 0:
        return _empty_result()

    # 質量加重 C/N（比率の単純平均ではない。炭素割合一定の仮定で相殺）
    denom = sum(
        mass / cn_ratio.get(name, 1e-6)
        for name, mass in per_class_mass.items()
    )
    blended_cn = total_mass / denom if denom > 0 else None

    browns_grams = _browns_to_target(
        greens_mass=total_mass,
        greens_cn=blended_cn,
        target_cn=target_cn,
        browns_cn=browns_cn,
    )

    return {
        "cn_ratio": round(blended_cn, 2) if blended_cn else None,
        "browns_grams": round(browns_grams, 1),
        "total_mass": round(total_mass, 2),
        "per_class_mass": {k: round(v, 2) for k, v in per_class_mass.items()},
        "n_detections": sum(1 for _ in per_class_mass),
    }


def _browns_to_target(greens_mass, greens_cn, target_cn, browns_cn):
    """
    現在の greens (検出物) に browns を足して目標 C/N に近づける必要量を解く。
    target = (M_g + M_b) / (M_g/cn_g + M_b/cn_b) を M_b について解く。
    既に目標以上なら 0 を返す。
    """
    if not greens_cn or greens_cn <= 0:
        return 0.0
    if greens_cn >= target_cn:
        return 0.0  # 既に十分に炭素寄り → 追加不要

    numerator = greens_mass * (1.0 - target_cn / greens_cn)
    denominator = (target_cn / browns_cn) - 1.0
    if denominator == 0:
        return 0.0

    m_b = numerator / denominator
    return max(0.0, m_b)


def _empty_result():
    return {
        "cn_ratio": None,
        "browns_grams": 0.0,
        "total_mass": 0.0,
        "per_class_mass": {},
        "n_detections": 0,
    }


# ============================================================
# 単体テスト
# ============================================================
if __name__ == "__main__":
    # 例: 640x640 の画像に carrot と onion の領域ボックス
    demo_boxes = [
        (50, 50, 350, 300),   # carrot_peels
        (380, 320, 600, 560),  # onion
    ]
    demo_ids = [0, 2]
    result = estimate_cn(demo_boxes, demo_ids, frame_w=640, frame_h=640)
    print("C/N estimate:", result)
