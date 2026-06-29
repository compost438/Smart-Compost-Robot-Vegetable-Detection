# -*- coding: utf-8 -*-
"""
Device_Library.py
I2C / UART センサをまとめて扱うためのデバイスライブラリ

対応センサ:
  - SHT35   : 温度/湿度（I2C）
  - BME280  : 温度/湿度/気圧（I2C）
  - TSL2561 : 照度（I2C）
  - ADS1115 : アナログ入力（MQ135 等）（I2C）
  - MH-Z19C : CO₂（UART）
"""

import time
import smbus2
import serial


class DeviceClass:
    """
    複数センサを一つのクラスで統一的に読み取るためのクラス
    main.py 側はこのクラスのメソッドだけを呼べばOK
    """

    def __init__(self, i2c_bus_number: int = 1, debug: bool = False):
        # デバッグ出力のON/OFF（通常は False 推奨）
        self.debug = debug

        # I2Cバスの初期化
        self.i2c = smbus2.SMBus(i2c_bus_number)

        # I2Cアドレス
        self.sht35_addr = 0x44
        self.bme280_addr = 0x76
        self.tsl2561_addr = 0x39
        self.ads1115_addr = 0x48

        # TSL2561 レジスタ定義
        self.COMMAND_BIT = 0x80
        self.CONTROL_REG = 0x00
        self.TIMING_REG = 0x01
        self.POWER_ON = 0x03
        self.POWER_OFF = 0x00

        # BME280 校正値（初期化後に読み込み）
        self.digT, self.digP, self.digH = [], [], []
        self.t_fine = 0.0
        self._bme280_initialized = False
        self._tsl2561_initialized = False

        # UART（CO2 センサ）
        self.co2_serial = None
        try:
            self.co2_serial = serial.Serial("/dev/serial0", baudrate=9600, timeout=0.05)
            self._debug("MH-Z19C connected on /dev/serial0")
        except Exception as e:
            self.co2_serial = None
            self._debug(f"MH-Z19C not available: {e}")

    def _debug(self, msg: str) -> None:
        """デバッグ出力（debug=Trueのときのみ表示）"""
        if self.debug:
            print(f"[DeviceClass] {msg}")

    # ------------------------------------------------------------
    # SHT35（温度/湿度）
    # ------------------------------------------------------------
    def def_sht35(self):
        """
        SHT35 を読み取る
        return: (status, temperature, humidity)
          status: 0=OK, 1=短いデータ, 2=例外
        """
        try:
            # 高精度計測コマンド（0x2400）
            self.i2c.write_i2c_block_data(self.sht35_addr, 0x24, [0x00])
            time.sleep(0.015)

            data = self.i2c.read_i2c_block_data(self.sht35_addr, 0x00, 6)
            if len(data) != 6:
                return (1, 0.0, 0.0)

            temp_raw = (data[0] << 8) | data[1]
            hum_raw = (data[3] << 8) | data[4]

            # データシートの変換式
            temperature = -45 + (175 * temp_raw / 65535.0)
            humidity = 100 * hum_raw / 65535.0

            return (0, round(temperature, 2), round(humidity, 2))
        except Exception as e:
            self._debug(f"SHT35 I2C error: {e}")
            return (2, 0.0, 0.0)

    # ------------------------------------------------------------
    # ADS1115（アナログ入力）
    # ------------------------------------------------------------
    def def_ads1115(self, channel: int = 0):
        """
        ADS1115 を読み取る（シングルエンド 0〜3ch）
        return: (status, voltage, raw)
          status: 0=OK, 1=チャンネル不正, 2=例外
        """
        if channel not in (0, 1, 2, 3):
            return (1, 0.0, 0)

        mux_map = {0: 0x4000, 1: 0x5000, 2: 0x6000, 3: 0x7000}

        try:
            CONFIG_REG = 0x01
            CONV_REG = 0x00

            # 設定:
            # - OS=1 (start single conversion)
            # - MUX=single-ended
            # - PGA=±4.096V
            # - MODE=single-shot
            # - DR=1600SPS
            # - COMP=disabled
            config = (
                0x8000
                | mux_map[channel]
                | 0x0200
                | 0x0100
                | 0x0080
                | 0x0003
            )

            self.i2c.write_i2c_block_data(
                self.ads1115_addr,
                CONFIG_REG,
                [(config >> 8) & 0xFF, config & 0xFF],
            )

            time.sleep(0.003)

            data = self.i2c.read_i2c_block_data(self.ads1115_addr, CONV_REG, 2)
            raw = (data[0] << 8) | data[1]
            if raw > 32767:
                raw -= 65536

            voltage = (raw * 4.096) / 32768.0
            return (0, round(voltage, 4), raw)
        except Exception as e:
            self._debug(f"ADS1115 error: {e}")
            return (2, 0.0, 0)

    # ------------------------------------------------------------
    # BME280（温度/湿度/気圧）
    # ------------------------------------------------------------
    def F_bme280(self):
        """
        BME280 の値を読み取る（初回のみ初期化）
        return: (temp, hum, pres)
        """
        if not self._bme280_initialized:
            try:
                self.setup_bme280()
                self.get_calib_param()
                self._bme280_initialized = True
                time.sleep(0.01)
            except Exception as e:
                self._debug(f"BME280 init error: {e}")
                return 0.0, 0.0, 0.0

        temp, hum, pres = self.readData()
        return round(temp, 2), round(hum, 2), round(pres, 2)

    def writeReg(self, reg_address, data):
        """BME280 レジスタ書き込み"""
        self.i2c.write_byte_data(self.bme280_addr, reg_address, data)

    def get_calib_param(self):
        """BME280 の校正パラメータ読み込み（温度/気圧/湿度）"""
        self.digT.clear()
        self.digP.clear()
        self.digH.clear()

        calib = []
        for i in range(0x88, 0x88 + 24):
            calib.append(self.i2c.read_byte_data(self.bme280_addr, i))

        calib.append(self.i2c.read_byte_data(self.bme280_addr, 0xA1))

        for i in range(0xE1, 0xE1 + 7):
            calib.append(self.i2c.read_byte_data(self.bme280_addr, i))

        self.digT.extend(
            [
                (calib[1] << 8) | calib[0],
                (calib[3] << 8) | calib[2],
                (calib[5] << 8) | calib[4],
            ]
        )
        self.digP.extend([(calib[i + 1] << 8) | calib[i] for i in range(6, 24, 2)])
        self.digH.extend(
            [
                calib[24],
                (calib[26] << 8) | calib[25],
                calib[27],
                (calib[28] << 4) | (calib[29] & 0x0F),
                (calib[30] << 4) | (calib[29] >> 4),
                calib[31],
            ]
        )

        # 2の補数（signed）変換
        for i in range(1, 3):
            if self.digT[i] & 0x8000:
                self.digT[i] = -((~self.digT[i] & 0xFFFF) + 1)
        for i in range(1, 8):
            if self.digP[i] & 0x8000:
                self.digP[i] = -((~self.digP[i] & 0xFFFF) + 1)
        for i in range(0, 6):
            if self.digH[i] & 0x8000:
                self.digH[i] = -((~self.digH[i] & 0xFFFF) + 1)

    def readData(self):
        """BME280 生データ読み取り → 補正計算して返す"""
        data = self.i2c.read_i2c_block_data(self.bme280_addr, 0xF7, 8)
        pres_raw = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
        temp_raw = (data[3] << 12) | (data[4] << 4) | (data[5] >> 4)
        hum_raw = (data[6] << 8) | data[7]

        temperature = self.compensate_T(temp_raw)
        pressure = self.compensate_P(pres_raw)
        humidity = self.compensate_H(hum_raw)
        return temperature, humidity, pressure

    def compensate_T(self, adc_T):
        """温度補正"""
        v1 = (adc_T / 16384.0 - self.digT[0] / 1024.0) * self.digT[1]
        v2 = ((adc_T / 131072.0 - self.digT[0] / 8192.0) ** 2) * self.digT[2]
        self.t_fine = v1 + v2
        return self.t_fine / 5120.0

    def compensate_P(self, adc_P):
        """気圧補正"""
        if self.t_fine == 0:
            return 0.0
        v1 = (self.t_fine / 2.0) - 64000.0
        v2 = (((v1 / 4.0) * (v1 / 4.0)) / 2048.0) * self.digP[5]
        v2 = v2 + ((v1 * self.digP[4]) * 2.0)
        v2 = (v2 / 4.0) + (self.digP[3] * 65536.0)
        v1 = (
            (
                (self.digP[2] * ((v1 / 4.0) * (v1 / 4.0) / 8192.0)) / 8.0
                + ((self.digP[1] * v1) / 2.0)
            )
            / 262144.0
        )
        v1 = ((32768.0 + v1) * self.digP[0]) / 32768.0
        if v1 == 0:
            return 0.0
        pressure = ((1048576.0 - adc_P) - (v2 / 4096.0)) * 3125.0
        if pressure < 0x80000000:
            pressure = (pressure * 2.0) / v1
        else:
            pressure = (pressure / v1) * 2.0
        v1 = (self.digP[8] * ((pressure / 8.0) * (pressure / 8.0)) / 8192.0) / 4096.0
        v2 = ((pressure / 4.0) * self.digP[7]) / 8192.0
        pressure = pressure + ((v1 + v2 + self.digP[6]) / 16.0)
        return pressure / 100.0

    def compensate_H(self, adc_H):
        """湿度補正"""
        var_h = self.t_fine - 76800.0
        if var_h == 0:
            return 0.0
        var_h = (
            adc_H
            - (self.digH[3] * 64.0 + self.digH[4] / 16384.0 * var_h)
        ) * (
            self.digH[1]
            / 65536.0
            * (
                1.0
                + self.digH[5] / 67108864.0 * var_h *
                (1.0 + self.digH[2] / 67108864.0 * var_h)
            )
        )
        var_h = var_h * (1.0 - self.digH[0] * var_h / 524288.0)
        if var_h > 100.0:
            var_h = 100.0
        elif var_h < 0.0:
            var_h = 0.0
        return var_h

    def setup_bme280(self):
        """BME280 動作設定（オーバーサンプリング等）"""
        osrs_t = 1
        osrs_p = 1
        osrs_h = 1
        mode = 3
        t_sb = 5
        filter_c = 0
        spi3w_en = 0
        ctrl_meas_reg = (osrs_t << 5) | (osrs_p << 2) | mode
        config_reg = (t_sb << 5) | (filter_c << 2) | spi3w_en
        ctrl_hum_reg = osrs_h
        self.writeReg(0xF2, ctrl_hum_reg)
        self.writeReg(0xF4, ctrl_meas_reg)
        self.writeReg(0xF5, config_reg)

    # ------------------------------------------------------------
    # TSL2561（照度）
    # ------------------------------------------------------------
    def F_tsl2561(self):
        """
        TSL2561 を読み取る（初回のみ初期化）
        return: (lux, visible_raw, ir_raw)
        """
        if not self._tsl2561_initialized:
            try:
                self.tsl2561_init()
                self._tsl2561_initialized = True
                time.sleep(0.05)
            except Exception as e:
                self._debug(f"TSL2561 init error: {e}")
                return 0.0, 0, 0

        try:
            visible = self.getVisibleLightRawData()
            infrared = self.getInfraredRawData()
            lux = self.getLux(visible, infrared)
            return round(lux, 2), visible, infrared
        except Exception as e:
            self._debug(f"TSL2561 read error: {e}")
            return 0.0, 0, 0

    def tsl2561_power(self, on: bool = True):
        """TSL2561 電源ON/OFF"""
        self.i2c.write_byte_data(
            self.tsl2561_addr,
            self.COMMAND_BIT | self.CONTROL_REG,
            self.POWER_ON if on else self.POWER_OFF,
        )

    def tsl2561_init(self):
        """TSL2561 初期設定（ゲイン/積分時間等）"""
        self.tsl2561_power(True)
        time.sleep(0.1)
        self.i2c.write_byte_data(
            self.tsl2561_addr,
            self.COMMAND_BIT | self.TIMING_REG,
            0x02,
        )

    def getVisibleLightRawData(self):
        """可視光チャンネル（CH0）生データ"""
        data = self.i2c.read_i2c_block_data(self.tsl2561_addr, 0xAC, 2)
        return (data[1] << 8) | data[0]

    def getInfraredRawData(self):
        """赤外チャンネル（CH1）生データ"""
        data = self.i2c.read_i2c_block_data(self.tsl2561_addr, 0xAE, 2)
        return (data[1] << 8) | data[0]

    def getLux(self, VLRD, IRRD):
        """CH0/CH1 比率から lux を推定（データシート近似式）"""
        if float(VLRD) == 0:
            ratio = 9999
        else:
            ratio = IRRD / float(VLRD)

        if 0 <= ratio <= 0.52:
            lux = (0.0315 * VLRD) - (0.0593 * VLRD * (ratio ** 1.4))
        elif ratio <= 0.65:
            lux = (0.0229 * VLRD) - (0.0291 * IRRD)
        elif ratio <= 0.80:
            lux = (0.0157 * VLRD) - (0.018 * IRRD)
        elif ratio <= 1.3:
            lux = (0.00338 * VLRD) - (0.0026 * IRRD)
        else:
            lux = 0.0

        return max(lux, 0.0)

    # ------------------------------------------------------------
    # MH-Z19C（CO2）
    # ------------------------------------------------------------
    def read_mh_z19c(self):
        """
        MH-Z19C の CO2(ppm) を読む
        - UART にコマンドを送って 9byte 応答を待つ
        """
        if not self.co2_serial:
            return 0.0

        try:
            cmd = bytearray([0xFF, 0x01, 0x86, 0, 0, 0, 0, 0, 0x79])
            self.co2_serial.reset_input_buffer()
            self.co2_serial.write(cmd)

            start = time.time()
            resp = b""
            while len(resp) < 9 and (time.time() - start) < 0.08:
                chunk = self.co2_serial.read(9 - len(resp))
                if not chunk:
                    break
                resp += chunk

            if len(resp) == 9 and resp[0] == 0xFF and resp[1] == 0x86:
                return float((resp[2] << 8) | resp[3])

        except Exception as e:
            self._debug(f"MH-Z19C read error: {e}")

        return 0.0

    def mh_z19c_zero_calibration(self):
        """ゼロ校正（400ppm基準）コマンド送信"""
        if self.co2_serial:
            self.co2_serial.write(bytearray([0xFF, 0x01, 0x87, 0, 0, 0, 0, 0, 0x78]))
            self._debug("MH-Z19C zero calibration command sent (baseline 400 ppm).")

    def mh_z19c_disable_auto_calibration(self):
        """自動校正（ABC）を無効化"""
        if self.co2_serial:
            self.co2_serial.write(
                bytearray([0xFF, 0x01, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x86])
            )
            self._debug("MH-Z19C automatic calibration disabled.")
