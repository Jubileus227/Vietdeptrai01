# ============================================================
# XAU/USD SIGNAL BOT — Bot tín hiệu vàng đa khung thời gian
# ============================================================
# Chạy được ở 2 nơi:
#  - Google Colab (thủ công): điền trực tiếp 3 dòng CONFIG bên dưới
#  - GitHub Actions (tự động, định kỳ): để nguyên CONFIG, khai báo
#    3 giá trị qua Secrets (xem hướng dẫn kèm theo)
# ============================================================

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ============================================================
# CONFIG — ĐIỀN THÔNG TIN CỦA BẠN VÀO ĐÂY (nếu chạy trên Colab)
# ============================================================
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "DÁN_API_KEY_TWELVEDATA_VÀO_ĐÂY")
TELEGRAM_BOT_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "DÁN_TOKEN_BOT_TELEGRAM_VÀO_ĐÂY")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID", "DÁN_CHAT_ID_CỦA_BẠN_VÀO_ĐÂY")

SYMBOL = "XAU/USD"
# SL của Trend giờ ĐỘNG trong khoảng [SL_MIN_POINTS, SL_MAX_POINTS] theo ATR(M5) hiện tại -
# ATR càng cao (biến động mạnh) thì SL càng gần mức tối đa, tránh bị "stop-hunt" quét SL
# trong lúc biến động rồi mới đi đúng hướng (phát hiện qua dữ liệu thật: tỷ lệ hòa vốn cao
# + nhiều lệnh thua sát entry). ATR_SL_LOW/HIGH là vùng tham chiếu ATR M5 "thấp"/"cao" điển
# hình của XAU/USD - có thể cần tinh chỉnh lại sau khi có thêm dữ liệu thực tế.
SL_MIN_POINTS = 10
SL_MAX_POINTS = 20
ATR_SL_LOW = 0.5
ATR_SL_HIGH = 2.5
SIGNAL_THRESHOLD = 5        # chỉ gửi Telegram khi |điểm tổng hợp| >= giá trị này (đã tối ưu qua backtest sau khi thêm OB+Inside Bar: ngưỡng=5, TP:SL=2.0 cho kỳ vọng dương tốt trên mẫu 48 lệnh)

ADX_MIN = 20                # ADX dưới mức này coi là thị trường đi ngang -> không khuyến nghị vào lệnh
SESSION_FILTER_ENABLED = True   # bật/tắt bộ lọc phiên thanh khoản cao
SESSION_START_UTC = 7       # 07:00 UTC ~ 14:00 giờ VN (mở phiên London)
SESSION_END_UTC = 21        # 21:00 UTC ~ 04:00 giờ VN hôm sau (đóng phiên New York)

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")  # để trống nếu không dùng cảnh báo tin tức
NEWS_WARNING_MINUTES = 45   # cảnh báo nếu có tin quan trọng trong vòng X phút tới

SIGNAL_LOG_PATH = "signal_log.json"   # file lưu lịch sử tín hiệu để tự tính tỷ lệ thắng/thua
SIGNAL_LOG_MAX = 300                  # số bản ghi tối đa giữ lại trong file log
SIGNAL_TIMEOUT_HOURS = 4              # sau X giờ chưa chạm TP/SL thì coi là hết hạn, không tính thắng/thua

LOT_SIZE = 0.05                       # lot cố định mỗi lệnh (số thật, không phải ví dụ)
USD_PER_POINT = 5.0                   # 0.05 lot XAU/USD -> mỗi $1 giá thay đổi = $5 lãi/lỗ
MAX_STACK_PER_DIRECTION = 3           # tối đa bao nhiêu lệnh cùng chiều/cùng loại được chạy song song
MIN_SCORE_IMPROVEMENT_TO_STACK = 2    # điểm mới phải mạnh hơn lệnh đang chạy ít nhất bấy nhiêu mới được "nhồi"

# ============================================================
# 1. LẤY DỮ LIỆU GIÁ TỪ TWELVE DATA
# ============================================================
def get_ohlc(interval, outputsize=100):
    """
    interval: '5min', '15min', '30min', '1h'
    Trả về DataFrame với cột: datetime, open, high, low, close
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "order": "ASC",
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    if "values" not in data:
        raise Exception(f"Lỗi lấy dữ liệu ({interval}): {data.get('message', data)}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def resample_ohlc(df_m5, rule):
    """
    Gộp nến M5 thành khung lớn hơn (15min/30min/1h) NGAY TRONG MÁY,
    không cần gọi thêm API -> tiết kiệm request, cho phép chạy nhanh hơn.
    rule: '15min', '30min', '1h'
    """
    df = df_m5.set_index("datetime")
    out = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()
    return out.reset_index()


# ============================================================
# 2. CHỈ BÁO KỸ THUẬT
# ============================================================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def detect_trend(df, fast=9, slow=21):
    """Trả về 'up', 'down' dựa trên EMA nhanh vs EMA chậm"""
    df = df.copy()
    df["ema_fast"] = ema(df["close"], fast)
    df["ema_slow"] = ema(df["close"], slow)
    last = df.iloc[-1]
    return "up" if last["ema_fast"] > last["ema_slow"] else "down"


def detect_candle_pattern(df):
    """Phát hiện Bullish/Bearish Engulfing và Doji trên 2 nến gần nhất"""
    if len(df) < 2:
        return "none"
    prev, curr = df.iloc[-2], df.iloc[-1]

    body_curr = abs(curr["close"] - curr["open"])
    range_curr = curr["high"] - curr["low"]

    # Doji: thân nến rất nhỏ so với biên độ
    if range_curr > 0 and body_curr / range_curr < 0.1:
        return "doji"

    # Bullish Engulfing: nến hiện tại xanh, "nuốt" thân nến đỏ trước đó
    if (prev["close"] < prev["open"] and curr["close"] > curr["open"]
            and curr["close"] >= prev["open"] and curr["open"] <= prev["close"]):
        return "bullish_engulfing"

    # Bearish Engulfing
    if (prev["close"] > prev["open"] and curr["close"] < curr["open"]
            and curr["open"] >= prev["close"] and curr["close"] <= prev["open"]):
        return "bearish_engulfing"

    return "none"


def detect_bos(df, lookback=20):
    """
    Break of Structure đơn giản: giá hiện tại có phá đỉnh/đáy gần nhất không.
    Trả về 'up' (phá đỉnh), 'down' (phá đáy), hoặc None.
    """
    recent = df.iloc[-lookback:-1]
    curr_close = df.iloc[-1]["close"]
    if curr_close > recent["high"].max():
        return "up"
    if curr_close < recent["low"].min():
        return "down"
    return None


def detect_bos_level(df, lookback=20):
    """
    Giống detect_bos nhưng trả về CẢ mức giá đã bị phá vỡ (không chỉ hướng) - dùng làm
    điểm "chờ giá quay lại test" (retest) trước khi vào lệnh, thay vì vào ngay lúc breakout
    (tránh FOMO đúng đỉnh/đáy sóng - giá thường quay lại test vùng vừa phá vỡ trước khi
    tiếp tục đi, khung càng lớn thời gian quay lại test càng lâu).
    """
    if len(df) <= lookback:
        return None
    recent = df.iloc[-lookback:-1]
    curr_close = df.iloc[-1]["close"]
    if curr_close > recent["high"].max():
        return {"direction": "up", "level": recent["high"].max()}
    if curr_close < recent["low"].min():
        return {"direction": "down", "level": recent["low"].min()}
    return None


# ============================================================
# 2d. BOX DETECTOR - công cụ hỗ trợ quyết định thay cho hệ thống tín hiệu chấm điểm cũ
# ============================================================
# Ý tưởng (theo phương pháp price action thực chiến người dùng cung cấp):
# 1. Tìm nến "tập trung thanh khoản" - biên độ (high-low) lớn hơn hẳn ATR trung bình gần đó
#    (proxy thay cho khối lượng giao dịch thật - XAU/USD OTC không có volume đáng tin cậy)
# 2. Lấy High/Low của nến đó làm biên trên/dưới của "box"
# 3. Chờ nến sau đó ĐÓNG CỬA hẳn ngoài box (xác nhận breakout lên hoặc xuống)
# 4. Sau xác nhận, chờ giá QUAY LẠI TEST đúng biên trước khi coi là "sẵn sàng vào lệnh"
# 5. Màu nến thanh khoản quyết định hướng "thuận xu hướng" hay "ngược xu hướng":
#    - Nến XANH bị phá XUỐNG -> SELL thuận xu hướng (3 điểm entry: dưới/giữa/trên)
#    - Nến XANH bị phá LÊN   -> BUY ngược xu hướng (2 điểm entry: giữa/dưới)
#    - Nến ĐỎ bị phá LÊN     -> BUY thuận xu hướng (3 điểm entry: trên/giữa/dưới)
#    - Nến ĐỎ bị phá XUỐNG   -> SELL ngược xu hướng (2 điểm entry: giữa/trên)
# Nếu 2 box gần nhất chồng lấn nhau: dùng phần GIAO NHAU cho điểm entry "giữa biên",
# dùng CẠNH XA NHẤT cho điểm entry "cạnh trên"/"cạnh dưới" (an toàn hơn, ít bị quét sớm).

BOX_LOOKBACK = 100          # số nến quét ngược để tìm nến thanh khoản
BOX_RANGE_ATR_MULT = 2.0    # nến thanh khoản phải có biên độ >= 2x ATR trung bình gần đó
BOX_RETEST_TOLERANCE_ATR = 0.3  # giá được coi là "đã quay về test" nếu cách điểm entry <= 0.3x ATR
BOX_SL_CONFIRMED = 20       # SL cho entry đã xác nhận (giá) - theo đúng bảng đã thống nhất
BOX_SL_RISK = 10            # SL cho entry rủi ro (box chưa xác nhận) - giá
BOX_MAX_DISTANCE_ATR = 8.0  # box cách giá hiện tại quá xa (theo ATR HIỆN TẠI) thì coi là hết liên quan
BOX_MIDDLE_ZONE_RATIO = 0.5  # giá nằm trong 50% khoảng giữa box (chưa xác nhận) -> không khuyến khích vào lệnh


def find_recent_liquidity_boxes(df, atr_series, lookback=BOX_LOOKBACK,
                                 range_mult=BOX_RANGE_ATR_MULT, max_boxes=2, skip_last=1,
                                 max_distance_atr=BOX_MAX_DISTANCE_ATR, bound_range=None):
    """
    Quét TOÀN BỘ phạm vi lookback tìm mọi nến "tập trung thanh khoản" hợp lệ (biên độ
    >= range_mult x ATR), rồi ƯU TIÊN CHỌN NẾN GẦN GIÁ HIỆN TẠI NHẤT (không phải nến gần
    đây nhất theo thời gian) - tránh trường hợp có box khác gần giá hơn nhưng bị bỏ qua chỉ
    vì có 1 box xa hơn nhưng mới hình thành gần đây hơn.
    Có 2 lớp lọc "còn liên quan tới giá hiện tại", tránh chọn nhầm 1 box cũ/nhỏ đã hết ý nghĩa:
    1. Biên độ nến phải đủ lớn so với ATR HIỆN TẠI (không chỉ so với ATR lúc nó hình thành).
    2. Khoảng cách từ giá hiện tại đến biên gần nhất của box không được quá max_distance_atr
       lần ATR hiện tại - nếu giá đã trôi quá xa mà chưa từng quay lại test, coi như hết liên quan.
    Nếu truyền bound_range=(low, high): CHỈ chấp nhận nến nằm LỌT HẲN trong khoảng đó - dùng để
    tìm box khung nhỏ (M15) NẰM TRONG box khung lớn (H1) đã chọn trước.
    """
    n = len(df)
    start = max(0, n - lookback)
    current_price = df.iloc[-1]["close"]
    atr_now = atr_series.iloc[-1]

    candidates = []
    i = n - 1 - skip_last
    while i > start:
        candle = df.iloc[i]
        candle_range = candle["high"] - candle["low"]
        local_atr = atr_series.iloc[i]
        qualifies = not pd.isna(local_atr) and local_atr > 0 and candle_range >= range_mult * local_atr

        if qualifies and atr_now > 0:
            qualifies = candle_range >= range_mult * atr_now * 0.5

        if qualifies and bound_range:
            lo, hi = bound_range
            if not (candle["low"] >= lo - 1e-9 and candle["high"] <= hi + 1e-9):
                qualifies = False

        dist_atr = None
        if qualifies and atr_now > 0:
            dist_to_box = 0.0
            if current_price > candle["high"]:
                dist_to_box = current_price - candle["high"]
            elif current_price < candle["low"]:
                dist_to_box = candle["low"] - current_price
            dist_atr = dist_to_box / atr_now
            if dist_atr > max_distance_atr:
                qualifies = False

        if qualifies:
            candidates.append({
                "idx": i, "high": float(candle["high"]), "low": float(candle["low"]),
                "color": "green" if candle["close"] > candle["open"] else "red",
                "dist_atr": dist_atr if dist_atr is not None else 0.0,
            })
        i -= 1

    # Ưu tiên GẦN giá hiện tại nhất trước (thay vì gần đây nhất theo thời gian)
    candidates.sort(key=lambda c: c["dist_atr"])

    boxes = []
    for c in candidates:
        # Tránh chọn 2 nến quá sát nhau (cùng 1 đợt biến động) làm 2 box riêng biệt
        if all(abs(c["idx"] - b["idx"]) >= 5 for b in boxes):
            boxes.append(c)
        if len(boxes) >= max_boxes:
            break
    return boxes


def merge_boxes_if_overlap(boxes):
    """
    Nếu 2 box gần nhất CHỒNG LẤN nhau: vùng giao nhau dùng cho entry "giữa biên",
    cạnh XA NHẤT trong số 2 box dùng cho entry "cạnh trên"/"cạnh dưới" (an toàn hơn).
    Nếu chỉ có 1 box hoặc không chồng lấn: dùng nguyên box gần nhất như bình thường.
    """
    b1 = boxes[0]
    if len(boxes) < 2:
        return {"mid_high": b1["high"], "mid_low": b1["low"],
                "outer_high": b1["high"], "outer_low": b1["low"],
                "color": b1["color"], "idx": b1["idx"]}

    b2 = boxes[1]
    overlap = not (b1["high"] < b2["low"] or b2["high"] < b1["low"])
    if not overlap:
        return {"mid_high": b1["high"], "mid_low": b1["low"],
                "outer_high": b1["high"], "outer_low": b1["low"],
                "color": b1["color"], "idx": b1["idx"]}

    return {
        "mid_high": min(b1["high"], b2["high"]), "mid_low": max(b1["low"], b2["low"]),
        "outer_high": max(b1["high"], b2["high"]), "outer_low": min(b1["low"], b2["low"]),
        "color": b1["color"], "idx": b1["idx"],
    }


def find_box_state(df, atr_series, lookback=BOX_LOOKBACK, range_mult=BOX_RANGE_ATR_MULT,
                    retest_tolerance_atr=BOX_RETEST_TOLERANCE_ATR, bound_range=None):
    """
    Trả về trạng thái box gần nhất:
    - None: không tìm thấy nến thanh khoản nào phù hợp trong phạm vi quét
    - state="unconfirmed": box vừa hình thành, CHƯA có nến đóng cửa phá vỡ hẳn ra ngoài
      -> chỉ có "entry rủi ro" (SL 10 giá) tại 2 cạnh, chưa rõ hướng
    - state="waiting_retest": đã xác nhận breakout nhưng giá CHƯA quay lại test biên
    - state="ready": đã xác nhận VÀ giá đã quay lại test - sẵn sàng entry (SL 20 giá)
    Truyền bound_range=(low, high) để CHỈ tìm box nằm LỌT trong 1 box khung lớn hơn đã chọn
    trước - dùng cho cấu trúc "box M15 nằm trong box H1".
    """
    boxes = find_recent_liquidity_boxes(df, atr_series, lookback=lookback, range_mult=range_mult,
                                         bound_range=bound_range)
    if not boxes:
        return None

    merged = merge_boxes_if_overlap(boxes)
    box_high, box_low, color = merged["mid_high"], merged["mid_low"], merged["color"]
    box_mid = (box_high + box_low) / 2
    current_price = df.iloc[-1]["close"]
    atr_now = atr_series.iloc[-1]

    # Giá đang nằm trong "vùng giữa" box (chưa xác nhận) hay không - nếu có, KHÔNG khuyến khích
    # vào lệnh vì chưa rõ hướng và không gần biên nào để có điểm tham chiếu hợp lý.
    box_height = box_high - box_low
    margin = box_height * (1 - BOX_MIDDLE_ZONE_RATIO) / 2
    in_middle = (box_low + margin) < current_price < (box_high - margin)

    after = df.iloc[merged["idx"] + 1:]
    if len(after) == 0:
        return {"box_high": box_high, "box_low": box_low, "box_mid": box_mid,
                "color": color, "state": "unconfirmed", "in_middle": in_middle}

    confirm_dir = None
    for j in range(len(after)):
        c = after.iloc[j]
        if c["close"] > merged["outer_high"]:
            confirm_dir = "up"; break
        if c["close"] < merged["outer_low"]:
            confirm_dir = "down"; break

    if confirm_dir is None:
        return {"box_high": box_high, "box_low": box_low, "box_mid": box_mid,
                "color": color, "state": "unconfirmed", "in_middle": in_middle}

    # VÔ HIỆU HÓA: nếu giá không chỉ "quay lại test" mà đã XUYÊN QUA TOÀN BỘ box sang phía
    # đối diện (vd: phá đáy, chờ tăng lại test, nhưng giá đã tăng vượt LUÔN cả cạnh trên) -
    # giả thuyết ban đầu không còn ý nghĩa, box này hết hiệu lực, không nên tiếp tục tham khảo.
    if confirm_dir == "down" and current_price > box_high:
        return None
    if confirm_dir == "up" and current_price < box_low:
        return None

    # Bảng ma trận: màu nến thanh khoản x hướng xác nhận -> hướng lệnh + độ thuận/ngược xu hướng
    if color == "green":
        if confirm_dir == "down":
            direction, alignment, entry_labels = "SELL", "thuận", ["low", "mid", "high"]
        else:
            direction, alignment, entry_labels = "BUY", "ngược", ["mid", "low"]
    else:
        if confirm_dir == "up":
            direction, alignment, entry_labels = "BUY", "thuận", ["high", "mid", "low"]
        else:
            direction, alignment, entry_labels = "SELL", "ngược", ["mid", "high"]

    # Dùng NHẤT QUÁN biên hiển thị (box_high/box_low) cho mọi điểm entry - tránh lệch số liệu
    # giữa box hiển thị và giá entry thật (trước đây "cạnh trên/dưới" dùng biên xa hơn khi có
    # box chồng lấn, gây chênh lệch khó hiểu so với box hiển thị trong tin nhắn).
    label_price = {"low": box_low, "high": box_high, "mid": box_mid}
    entries = [{"label": lbl, "price": label_price[lbl]} for lbl in entry_labels]

    tol = retest_tolerance_atr * atr_now if atr_now > 0 else 0
    retested = tol > 0 and any(abs(current_price - e["price"]) <= tol for e in entries)

    return {
        "box_high": box_high, "box_low": box_low, "box_mid": box_mid, "color": color,
        "state": "ready" if retested else "waiting_retest",
        "direction": direction, "alignment": alignment, "entries": entries,
        "confirm_dir": confirm_dir,
    }


BOX_TP1_MIN = 5.0           # TP1 tối thiểu (giá) - box nhỏ
BOX_TP1_MAX = 10.0          # TP1 tối đa (giá) - box lớn
BOX_TP1_HEIGHT_LOW = 15.0   # chiều cao box được coi là "nhỏ" (TP1 = mức tối thiểu)
BOX_TP1_HEIGHT_HIGH = 50.0  # chiều cao box được coi là "lớn" (TP1 = mức tối đa)


def compute_box_tp(direction, entry, box_height, multiples=(1.0, 2.0, 3.0)):
    """
    TP1 CHỈ 5-10 giá (không phải cả chiều cao box như trước - quá xa, hiếm khi chạm tới) -
    nội suy theo box nhỏ hay lớn: box nhỏ -> TP1 gần mức tối thiểu (5 giá), box lớn -> TP1
    gần mức tối đa (10 giá). TP2/TP3 tăng dần theo bội số của khoảng TP1 này (không phải
    theo chiều cao box nữa) - giữ tỷ lệ chốt lời từng phần hợp lý mà vẫn khả thi hơn nhiều.
    """
    if box_height <= BOX_TP1_HEIGHT_LOW:
        tp1_distance = BOX_TP1_MIN
    elif box_height >= BOX_TP1_HEIGHT_HIGH:
        tp1_distance = BOX_TP1_MAX
    else:
        ratio = (box_height - BOX_TP1_HEIGHT_LOW) / (BOX_TP1_HEIGHT_HIGH - BOX_TP1_HEIGHT_LOW)
        tp1_distance = BOX_TP1_MIN + ratio * (BOX_TP1_MAX - BOX_TP1_MIN)

    if direction == "BUY":
        return tuple(round(entry + tp1_distance * m, 2) for m in multiples)
    return tuple(round(entry - tp1_distance * m, 2) for m in multiples)


def build_box_signal(box_m15, box_h1, atr_m5):
    """
    Chọn box CHÍNH để giao dịch - ưu tiên box đã 'ready' (sẵn sàng entry), khung H1 trước
    (cấu trúc lớn hơn, đáng tin hơn), rồi tới M15. Nếu không box nào 'ready', vẫn chọn 1 box
    để hiển thị bối cảnh (ready > waiting_retest > unconfirmed), ưu tiên H1.
    Trả về dict mô tả đầy đủ box đã chọn + danh sách entry cụ thể (giá, SL, TP, nhãn) để hiển thị.
    """
    def _priority(b):
        if not b:
            return -1
        return {"ready": 3, "waiting_retest": 2, "unconfirmed": 1}[b["state"]]

    candidates = [("H1", box_h1), ("M15", box_m15)]
    candidates.sort(key=lambda x: _priority(x[1]), reverse=True)
    tf_name, box = candidates[0]
    if not box:
        return None

    box_height = box["box_high"] - box["box_low"]
    result = {"tf": tf_name, "box_high": box["box_high"], "box_low": box["box_low"],
              "box_mid": box["box_mid"], "color": box["color"], "state": box["state"],
              "entries": []}

    if box["state"] == "unconfirmed":
        result["in_middle"] = box.get("in_middle", False)
        if result["in_middle"]:
            # Giá đang ở GIỮA box (không gần biên nào) - KHÔNG khuyến khích vào lệnh vì chưa
            # rõ hướng và không có điểm tham chiếu hợp lý để đặt SL/TP lúc này.
            return result
        # Entry RỦI RO: fade 2 cạnh (chưa rõ hướng breakout), SL nhỏ hơn (10 giá)
        for label, price, direction in (("cạnh trên", box["box_high"], "SELL"),
                                         ("cạnh dưới", box["box_low"], "BUY")):
            sl = price + BOX_SL_RISK if direction == "SELL" else price - BOX_SL_RISK
            tp1, tp2, tp3 = compute_box_tp(direction, price, box_height)
            result["entries"].append({
                "label": label, "direction": direction, "entry": price, "sl": sl,
                "tp1": tp1, "tp2": tp2, "tp3": tp3, "risk": True,
            })
        return result

    result["direction"] = box["direction"]
    result["alignment"] = box["alignment"]
    result["confirm_dir"] = box["confirm_dir"]
    for e in box["entries"]:
        direction = box["direction"]
        sl = e["price"] - BOX_SL_CONFIRMED if direction == "BUY" else e["price"] + BOX_SL_CONFIRMED
        tp1, tp2, tp3 = compute_box_tp(direction, e["price"], box_height)
        label_map = {"low": "cạnh dưới", "mid": "giữa biên", "high": "cạnh trên"}
        result["entries"].append({
            "label": label_map[e["label"]], "direction": direction, "entry": e["price"], "sl": sl,
            "tp1": tp1, "tp2": tp2, "tp3": tp3, "risk": False,
        })
    return result


def detect_order_block(df, lookback=20):
    """
    Order Block đơn giản (không phải chuẩn SMC chính thức):
    tìm nến cuối cùng đi ngược hướng trước một đợt di chuyển mạnh.
    - Nến giảm cuối cùng trước đợt tăng mạnh -> Order Block "bullish"
    - Nến tăng cuối cùng trước đợt giảm mạnh -> Order Block "bearish"
    """
    recent = df.iloc[-lookback:].reset_index(drop=True)
    if len(recent) < 6:
        return None

    avg_body = (recent["close"] - recent["open"]).abs().mean()
    if avg_body == 0:
        return None

    for i in range(len(recent) - 4, 0, -1):
        candle = recent.iloc[i]
        next3 = recent.iloc[i + 1:i + 4]
        if len(next3) < 3:
            continue
        body = abs(candle["close"] - candle["open"])
        is_down = candle["close"] < candle["open"]
        is_up = candle["close"] > candle["open"]
        move_up = next3["close"].iloc[-1] - candle["close"]
        move_down = candle["close"] - next3["close"].iloc[-1]

        if is_down and move_up > avg_body * 2 and body > avg_body * 0.5:
            return {"type": "bullish", "zone": (candle["low"], candle["high"])}
        if is_up and move_down > avg_body * 2 and body > avg_body * 0.5:
            return {"type": "bearish", "zone": (candle["low"], candle["high"])}
    return None


def detect_inside_bar_setup(df, atr_series, mother_min_atr_mult=1.5, inside_max_ratio=0.6, max_inside_bars=6):
    """
    Mẫu hình "nến mẹ - nến con" (mother bar / inside bar) theo price action:
    - Nến MẸ: biên độ (high-low) >= 1.5x ATR -> nến biến động mạnh, "quyết định" rõ ràng
    - Nến CON: 1 hoặc nhiều nến liên tiếp nằm GỌN bên trong biên độ nến mẹ -> vùng tích lũy/do dự
    - Cụm nến con phải co lại đủ nhỏ (<=60% biên độ nến mẹ) mới coi là setup "đẹp"
    - Breakout CHỈ được xác nhận khi có nến ĐÓNG CỬA vượt hẳn qua đỉnh/đáy nến mẹ
      (chỉ chạm/chọc râu qua không tính -> lọc bớt false breakout)

    Quét lùi từ nến gần nhất để tìm setup đang hoạt động. Trả về None nếu không tìm thấy.
    """
    n = len(df)
    if n < max_inside_bars + 2:
        return None

    current = df.iloc[-1]  # nến gần nhất - ứng viên breakout hoặc vẫn đang là nến con

    for mother_offset in range(2, max_inside_bars + 2):
        mother_idx = n - 1 - mother_offset
        if mother_idx < 0:
            break
        mother = df.iloc[mother_idx]
        mother_range = mother["high"] - mother["low"]
        atr_at_mother = atr_series.iloc[mother_idx]
        if pd.isna(atr_at_mother) or atr_at_mother <= 0:
            continue
        if mother_range < mother_min_atr_mult * atr_at_mother:
            continue  # nến này không đủ "dài" để làm nến mẹ

        inside_bars = df.iloc[mother_idx + 1: n - 1]
        if len(inside_bars) == 0:
            continue

        all_inside = (inside_bars["high"] <= mother["high"]).all() and (inside_bars["low"] >= mother["low"]).all()
        if not all_inside:
            continue

        cluster_range = inside_bars["high"].max() - inside_bars["low"].min()
        if cluster_range > mother_range * inside_max_ratio:
            continue  # nến con chưa co lại đủ chặt, setup chưa "đẹp"

        touched_high = bool((inside_bars["high"] >= mother["high"] - 0.05 * mother_range).any())
        touched_low = bool((inside_bars["low"] <= mother["low"] + 0.05 * mother_range).any())

        breakout = None
        if current["close"] > mother["high"]:
            breakout = "up"
        elif current["close"] < mother["low"]:
            breakout = "down"

        return {
            "mother_high": mother["high"],
            "mother_low": mother["low"],
            "num_inside_bars": len(inside_bars),
            "touched_high": touched_high,
            "touched_low": touched_low,
            "breakout": breakout,
        }

    return None


def adx(df, period=14):
    """
    ADX (Average Directional Index) — đo ĐỘ MẠNH của xu hướng, không quan tâm hướng.
    ADX < 20: xu hướng yếu / thị trường đi ngang -> tín hiệu trend dễ sai.
    ADX > 25: xu hướng đang rõ ràng, tín hiệu trend đáng tin hơn.
    """
    high, low, close = df["high"], df["low"], df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_.fillna(0)


def is_active_session(now_utc=None):
    """
    Kiểm tra hiện tại có đang trong phiên thanh khoản cao (London + New York) không.
    Ngoài khung này, giá dễ đi ngang/nhiễu, tín hiệu kém tin cậy hơn.
    """
    if not SESSION_FILTER_ENABLED:
        return True
    now_utc = now_utc or datetime.now(timezone.utc)
    hour = now_utc.hour
    if SESSION_START_UTC <= SESSION_END_UTC:
        return SESSION_START_UTC <= hour < SESSION_END_UTC
    return hour >= SESSION_START_UTC or hour < SESSION_END_UTC


def is_market_closed(now_utc=None):
    """
    XAU/USD (vàng, giao dịch theo giờ forex) đóng cửa cuối tuần: khoảng từ
    21:00 UTC thứ Sáu đến 21:00 UTC Chủ Nhật (giờ đóng/mở chính xác có thể lệch
    ~1 tiếng tùy sàn, dùng mốc an toàn hơi rộng ra 1 chút để tránh chạy nhầm lúc
    thị trường vừa đóng/mở, dữ liệu chưa ổn định).
    Weekday: Monday=0 ... Friday=4, Saturday=5, Sunday=6.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    weekday, hour = now_utc.weekday(), now_utc.hour
    if weekday == 5:  # Thứ Bảy - luôn đóng cửa
        return True
    if weekday == 4 and hour >= 21:  # Thứ Sáu từ 21:00 UTC trở đi
        return True
    if weekday == 6 and hour < 21:  # Chủ Nhật trước 21:00 UTC
        return True
    return False


def is_market_flat(df, lookback=12, min_range_ratio=0.0003):
    """
    Phát hiện thị trường ĐANG ĐỨNG YÊN (nghỉ lễ, đóng cửa ngoài lịch cuối tuần thông
    thường, hoặc feed dữ liệu bị "đứng") - dựa THẲNG vào dữ liệu giá thật, không chỉ
    đoán theo lịch cố định (is_market_closed không bắt được ngày nghỉ lễ vì nó không
    nằm trong lịch cuối tuần cứng).

    Coi là "đứng yên" khi trong 'lookback' nến gần nhất (mặc định 12 nến M5 = 1 tiếng):
    - Biên độ dao động (high-low) quá nhỏ so với giá (< 0.03% giá), HOẶC
    - Giá đóng cửa gần như giống hệt nhau suốt cả khung đó (dấu hiệu feed bị "đứng"
      vì sàn không cập nhật giá mới - đúng những gì xảy ra khi API vẫn trả về được
      nhưng chỉ lặp lại giá cuối cùng trước khi nghỉ lễ).
    """
    if len(df) < lookback:
        return False
    recent = df.iloc[-lookback:]
    price = recent["close"].iloc[-1]
    if price <= 0:
        return False

    price_range = recent["high"].max() - recent["low"].min()
    range_ratio = price_range / price
    identical_closes = recent["close"].nunique() <= 2  # gần như không đổi giá suốt cả khung

    return range_ratio < min_range_ratio or identical_closes


def trading_hours_elapsed(start, end):
    """
    Tính số GIỜ GIAO DỊCH THỰC TẾ trôi qua giữa start và end - KHÔNG tính giờ cuối
    tuần thị trường đóng cửa. Dùng để hạn mức hết hạn (SIGNAL_TIMEOUT_HOURS) công
    bằng, tránh 1 lệnh mở chiều Thứ Sáu bị tính "hết hạn oan" chỉ vì cộng dồn luôn
    48 tiếng cuối tuần thị trường còn chưa mở cửa để giá có cơ hội chạm TP/SL.
    """
    if end <= start:
        return 0.0
    total_seconds = 0.0
    cursor = start
    step = timedelta(minutes=30)  # bước nhỏ để đủ chính xác quanh mốc đóng/mở cửa
    while cursor < end:
        next_cursor = min(cursor + step, end)
        midpoint = cursor + (next_cursor - cursor) / 2
        if not is_market_closed(midpoint):
            total_seconds += (next_cursor - cursor).total_seconds()
        cursor = next_cursor
    return total_seconds / 3600


def check_upcoming_news():
    """
    Kiểm tra tin kinh tế quan trọng (USD, high impact) sắp ra trong NEWS_WARNING_MINUTES phút tới.
    Dùng Finnhub (cần FINNHUB_API_KEY, để trống thì bỏ qua tính năng này).
    Trả về tên sự kiện gần nhất nếu có, hoặc None. Lỗi mạng/API sẽ bị bỏ qua êm (không làm chết bot).
    """
    if not FINNHUB_API_KEY:
        return None
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = "https://finnhub.io/api/v1/calendar/economic"
        params = {"from": today, "to": today, "token": FINNHUB_API_KEY}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        events = data.get("economicCalendar", data.get("data", []))
        now_utc = datetime.now(timezone.utc)

        for ev in events:
            impact = str(ev.get("impact", "")).lower()
            country = str(ev.get("country", ev.get("economy", ""))).upper()
            if impact not in ("3", "high") or country not in ("US", "USD"):
                continue
            ev_time_str = ev.get("time") or ev.get("data")
            if not ev_time_str:
                continue
            try:
                ev_time = pd.to_datetime(ev_time_str, utc=True)
            except Exception:
                continue
            minutes_away = (ev_time - now_utc).total_seconds() / 60
            if 0 <= minutes_away <= NEWS_WARNING_MINUTES:
                return f"{ev.get('event', ev.get('name', 'Tin quan trọng'))} lúc {ev_time.strftime('%H:%M UTC')}"
        return None
    except Exception:
        return None  # không để lỗi API tin tức làm chết cả bot


def rsi(series, period=14):
    """RSI chuẩn — đo quá mua (>70) / quá bán (<30)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50)


def atr(df, period=14):
    """Average True Range — đo mức độ biến động hiện tại"""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def macd(series, fast=12, slow=26, signal=9):
    """
    MACD — đo MOMENTUM (tốc độ/gia tốc chuyển động giá), khác với trend (chỉ đo hướng)
    và khác RSI (đo vùng quá mua/quá bán). Trả về (đường MACD, đường Signal, Histogram).
    Histogram dương và đang phình to = momentum tăng đang mạnh lên.
    Histogram âm và đang phình to (về độ lớn) = momentum giảm đang mạnh lên.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def detect_divergence(df, indicator_series, lookback=40, left=2, right=2):
    """
    Phát hiện PHÂN KỲ (divergence) giữa giá và 1 chỉ báo momentum (thường dùng MACD histogram):
    - "bearish": giá tạo ĐỈNH CAO HƠN nhưng chỉ báo tạo đỉnh THẤP HƠN -> đà tăng có dấu hiệu
      cạn kiệt, cảnh báo sớm khả năng đảo chiều xuống dù giá vẫn đang "thắng".
    - "bullish": giá tạo ĐÁY THẤP HƠN nhưng chỉ báo tạo đáy CAO HƠN -> đà giảm cạn kiệt,
      cảnh báo sớm khả năng đảo chiều lên.
    Đây là 1 trong những tín hiệu cảnh báo sớm đáng tin cậy nhất trong phân tích kỹ thuật.
    """
    recent = df.iloc[-lookback:].reset_index(drop=True)
    ind = indicator_series.iloc[-lookback:].reset_index(drop=True)
    n = len(recent)
    if n < left + right + 10:
        return None

    swing_highs, swing_lows = [], []
    for i in range(left, n - right):
        window_high = recent["high"].iloc[i - left:i + right + 1]
        window_low = recent["low"].iloc[i - left:i + right + 1]
        if recent["high"].iloc[i] == window_high.max():
            swing_highs.append((recent["high"].iloc[i], ind.iloc[i]))
        if recent["low"].iloc[i] == window_low.min():
            swing_lows.append((recent["low"].iloc[i], ind.iloc[i]))

    if len(swing_highs) >= 2:
        (p1, i1), (p2, i2) = swing_highs[-2], swing_highs[-1]
        if p2 > p1 and i2 < i1:
            return "bearish"
    if len(swing_lows) >= 2:
        (p1, i1), (p2, i2) = swing_lows[-2], swing_lows[-1]
        if p2 < p1 and i2 > i1:
            return "bullish"
    return None


def support_resistance(df, lookback=30):
    """Vùng hỗ trợ/kháng cự gần nhất = đáy/đỉnh gần nhất trong lookback nến"""
    recent = df.iloc[-lookback:]
    return {"support": recent["low"].min(), "resistance": recent["high"].max()}


def fibonacci_levels(df, lookback=50):
    """
    Tính các mức Fibonacci retracement/extension từ đợt sóng (đỉnh-đáy) gần nhất
    trong 'lookback' nến. Dùng để tham khảo vùng SL/TP hợp lý (không thay thế
    cách tính ATR đang dùng, chỉ là lớp xác nhận thêm - confluence check).
    """
    recent = df.iloc[-lookback:].reset_index(drop=True)
    if len(recent) < 10:
        return None

    high_idx = recent["high"].idxmax()
    low_idx = recent["low"].idxmin()
    swing_high = recent.loc[high_idx, "high"]
    swing_low = recent.loc[low_idx, "low"]
    diff = swing_high - swing_low
    if diff <= 0:
        return None

    # Nếu đáy hình thành SAU đỉnh -> sóng đang giảm gần nhất -> retracement tính từ trên xuống
    # Nếu đỉnh hình thành SAU đáy -> sóng đang tăng gần nhất -> retracement tính từ dưới lên
    uptrend_leg = low_idx < high_idx

    if uptrend_leg:
        levels = {
            "0.0": swing_high,
            "23.6": swing_high - diff * 0.236,
            "38.2": swing_high - diff * 0.382,
            "50.0": swing_high - diff * 0.5,
            "61.8": swing_high - diff * 0.618,
            "78.6": swing_high - diff * 0.786,
            "100.0": swing_low,
            "ext_127.2": swing_high + diff * 0.272,
            "ext_161.8": swing_high + diff * 0.618,
        }
    else:
        levels = {
            "0.0": swing_low,
            "23.6": swing_low + diff * 0.236,
            "38.2": swing_low + diff * 0.382,
            "50.0": swing_low + diff * 0.5,
            "61.8": swing_low + diff * 0.618,
            "78.6": swing_low + diff * 0.786,
            "100.0": swing_high,
            "ext_127.2": swing_low - diff * 0.272,
            "ext_161.8": swing_low - diff * 0.618,
        }

    return {"swing_high": swing_high, "swing_low": swing_low, "uptrend_leg": uptrend_leg, "levels": levels}


def fib_confluence_note(fib, entry, sl, tp1, atr_value, tolerance_mult=0.5):
    """
    Kiểm tra SL/TP hiện tại có 'trùng' (nằm gần) 1 mức Fib quan trọng không.
    Nếu trùng -> tăng độ tin cậy, trả về ghi chú mô tả. Ngưỡng 'trùng' = tolerance_mult * ATR.
    """
    if not fib:
        return None
    tolerance = atr_value * tolerance_mult
    notes = []
    key_levels = ["38.2", "50.0", "61.8", "ext_127.2", "ext_161.8"]
    for name in key_levels:
        price = fib["levels"][name]
        if abs(sl - price) <= tolerance:
            notes.append(f"SL gần trùng Fib {name}%")
        if abs(tp1 - price) <= tolerance:
            notes.append(f"TP1 gần trùng Fib {name}%")
    return "; ".join(notes) if notes else None


# ============================================================
# 2c. MỨC GIÁ H1/H4 LỊCH SỬ (đỉnh/đáy cũ vẫn còn "phản ứng")
# ============================================================
# Ý tưởng: đỉnh/đáy nổi bật (swing high/low) trên khung H1/H4 thường vẫn là vùng
# giá thị trường "nhớ" và phản ứng lại nhiều ngày/tuần sau, kể cả khi không còn
# hỗ trợ/kháng cự nào khác gần đó. H1 thường còn hiệu lực ~1 tuần, H4 ~1 tháng.
# Chỉ tính lại mỗi giờ (cache) để không tốn thêm request mỗi lần chạy (5 phút/lần).

HTF_CACHE_PATH = "htf_levels_cache.json"
HTF_CACHE_MAX_AGE_MINUTES = 60  # chỉ tải lại dữ liệu H1/H4 mỗi 60 phút


def find_swing_levels(df, left=2, right=2):
    """Tìm đỉnh/đáy 'swing' (fractal) - cao/thấp hơn hẳn các nến lân cận 2 bên."""
    levels = []
    n = len(df)
    for i in range(left, n - right):
        window_high = df["high"].iloc[i - left:i + right + 1]
        window_low = df["low"].iloc[i - left:i + right + 1]
        candle = df.iloc[i]
        if candle["high"] == window_high.max():
            levels.append({"price": float(candle["high"]), "type": "high", "idx": i})
        if candle["low"] == window_low.min():
            levels.append({"price": float(candle["low"]), "type": "low", "idx": i})
    return levels


def detect_dow_trend(df, left=2, right=2, lookback=100):
    """
    Xác định xu hướng CHÍNH theo Lý thuyết Dow - dựa vào chuỗi đỉnh/đáy (swing) gần nhất:
    - 2 đỉnh gần nhất TĂNG DẦN + 2 đáy gần nhất TĂNG DẦN (Higher High + Higher Low liên tiếp)
      -> xu hướng TĂNG
    - 2 đỉnh gần nhất GIẢM DẦN + 2 đáy gần nhất GIẢM DẦN (Lower High + Lower Low liên tiếp)
      -> xu hướng GIẢM
    - Còn lại (đỉnh/đáy không đồng thuận) -> chưa rõ ràng/đi ngang, trả về None
    Dùng để đối chiếu với hướng lệnh Box Detector - lệnh THUẬN với xu hướng Dow đáng tin hơn
    lệnh đi ngược lại.
    """
    recent = df.iloc[-lookback:].reset_index(drop=True)
    swings = find_swing_levels(recent, left=left, right=right)
    highs = [s["price"] for s in swings if s["type"] == "high"]
    lows = [s["price"] for s in swings if s["type"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return "up"
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return "down"
    return None


def detect_ichimoku_signal(df, tenkan_period=9, kijun_period=26, senkou_b_period=52):
    """
    Xác định tín hiệu Ichimoku (Kumo/mây) tại thời điểm HIỆN TẠI - đúng chuẩn: mây hiển thị
    tại nến hiện tại thực ra được TÍNH TỪ dữ liệu 26 kỳ TRƯỚC đó (Senkou Span dịch chuyển
    tới trước 26 kỳ khi vẽ), nên phải lùi lại đúng 26 kỳ để lấy dữ liệu tính mây cho đúng.

    Phân loại 5 trường hợp theo độ tin cậy (đã thống nhất qua trao đổi trước):
    - "strong_bull": giá TRÊN mây XANH (Span A > Span B) - tăng, đáng tin nhất
    - "new_bull": giá TRÊN mây ĐỎ (Span A < Span B) - tăng nhưng mới, chưa chắc
    - "strong_bear": giá DƯỚI mây ĐỎ - giảm, đáng tin nhất
    - "new_bear": giá DƯỚI mây XANH - giảm nhưng mới, chưa chắc
    - "unclear": giá đang NẰM TRONG mây - chưa rõ xu hướng
    """
    n = len(df)
    idx = n - 1 - kijun_period  # điểm dữ liệu dùng để tính mây tại thời điểm hiện tại
    if idx < senkou_b_period:
        return None

    window = df.iloc[:idx + 1]
    tenkan = (window["high"].iloc[-tenkan_period:].max() + window["low"].iloc[-tenkan_period:].min()) / 2
    kijun = (window["high"].iloc[-kijun_period:].max() + window["low"].iloc[-kijun_period:].min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (window["high"].iloc[-senkou_b_period:].max() + window["low"].iloc[-senkou_b_period:].min()) / 2

    current_price = df.iloc[-1]["close"]
    cloud_color = "green" if span_a > span_b else "red"
    cloud_top, cloud_bottom = max(span_a, span_b), min(span_a, span_b)

    if current_price > cloud_top:
        position = "above"
    elif current_price < cloud_bottom:
        position = "below"
    else:
        position = "inside"

    if position == "above" and cloud_color == "green":
        strength = "strong_bull"
    elif position == "above" and cloud_color == "red":
        strength = "new_bull"
    elif position == "below" and cloud_color == "red":
        strength = "strong_bear"
    elif position == "below" and cloud_color == "green":
        strength = "new_bear"
    else:
        strength = "unclear"

    return {"position": position, "cloud_color": cloud_color, "strength": strength,
            "span_a": span_a, "span_b": span_b}


def filter_unbroken_levels(df, levels):
    """Chỉ giữ lại mức CHƯA bị giá đóng cửa phá vỡ sau khi hình thành (còn 'nguyên vẹn')."""
    unbroken = []
    for lv in levels:
        after = df.iloc[lv["idx"] + 1:]
        if len(after) == 0:
            unbroken.append(lv)  # vừa hình thành, chưa có nến nào sau để kiểm tra phá vỡ
            continue
        if lv["type"] == "high":
            broken = (after["close"] > lv["price"]).any()
        else:
            broken = (after["close"] < lv["price"]).any()
        if not broken:
            unbroken.append(lv)
    return unbroken


def load_htf_cache():
    if not os.path.exists(HTF_CACHE_PATH):
        return None
    try:
        with open(HTF_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_htf_cache(cache):
    with open(HTF_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def refresh_htf_cache_if_needed():
    """
    Chỉ tải lại dữ liệu H1 (~1 tuần) + H4 (~1 tháng) nếu cache đã cũ hơn 60 phút.
    Tốn thêm 2 request MỖI GIỜ (không phải mỗi lần chạy) - vẫn an toàn trong hạn mức free.
    Nếu lỗi mạng, dùng lại cache cũ (nếu có) thay vì làm crash cả bot.
    """
    cache = load_htf_cache()
    now = datetime.now(timezone.utc)

    if cache:
        try:
            updated_at = datetime.fromisoformat(cache["updated_at"])
            if (now - updated_at).total_seconds() / 60 < HTF_CACHE_MAX_AGE_MINUTES:
                return cache  # cache còn mới, dùng lại luôn
        except Exception:
            pass

    try:
        df_h1 = get_ohlc("1h", outputsize=170)   # ~1 tuần
        df_h4 = get_ohlc("4h", outputsize=190)   # ~1 tháng
    except Exception:
        return cache  # lỗi mạng -> dùng cache cũ nếu có, không crash

    h1_levels = filter_unbroken_levels(df_h1, find_swing_levels(df_h1))
    h4_levels = filter_unbroken_levels(df_h4, find_swing_levels(df_h4))

    new_cache = {
        "updated_at": now.isoformat(),
        "h1_levels": [{"price": lv["price"], "type": lv["type"]} for lv in h1_levels],
        "h4_levels": [{"price": lv["price"], "type": lv["type"]} for lv in h4_levels],
        "atr_h1": round(float(atr(df_h1).iloc[-1]), 3),   # dùng làm SL cho Zone Setup có nguồn H1
        "atr_h4": round(float(atr(df_h4).iloc[-1]), 3),   # dùng làm SL cho Zone Setup có nguồn H4
    }
    save_htf_cache(new_cache)
    return new_cache


def nearest_htf_levels(cache, current_price, atr_value, max_count=2):
    """Lấy các mức H1/H4 gần giá hiện tại nhất, đủ xa để có ý nghĩa (>=0.3x ATR)."""
    if not cache or atr_value <= 0:
        return []
    all_levels = (
        [{"price": lv["price"], "type": lv["type"], "tf": "H1"} for lv in cache.get("h1_levels", [])] +
        [{"price": lv["price"], "type": lv["type"], "tf": "H4"} for lv in cache.get("h4_levels", [])]
    )
    result = []
    for lv in all_levels:
        dist_atr = abs(current_price - lv["price"]) / atr_value
        if dist_atr < 0.3:
            continue
        result.append({**lv, "distance_atr": round(dist_atr, 1)})
    result.sort(key=lambda x: x["distance_atr"])
    return result[:max_count]


def detect_fvg(df):
    """
    Fair Value Gap đơn giản: khoảng trống giữa nến[-3] và nến[-1]
    (không giao nhau giữa high nến 1 và low nến 3, hoặc ngược lại)
    """
    if len(df) < 3:
        return None
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    if c1["high"] < c3["low"]:
        return {"type": "bullish", "zone": (c1["high"], c3["low"])}
    if c1["low"] > c3["high"]:
        return {"type": "bearish", "zone": (c3["high"], c1["low"])}
    return None


# ============================================================
# 2b. TỰ THEO DÕI KẾT QUẢ TÍN HIỆU (WIN-RATE TRACKING)
# ============================================================
def load_signal_log():
    if not os.path.exists(SIGNAL_LOG_PATH):
        return []
    try:
        with open(SIGNAL_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_signal_log(log):
    log = log[-SIGNAL_LOG_MAX:]
    with open(SIGNAL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def update_signal_outcomes(log, current_price):
    """
    Kiểm tra các tín hiệu cũ:
    - 'waiting_fill' (lệnh limit "chờ giá về"): kiểm tra giá đã pullback đủ để khớp lệnh chưa.
      CHƯA khớp thì CHƯA tính thắng/thua - tránh đếm nhầm lệnh chưa từng vào.
    - 'pending' (đã khớp - lệnh market, hoặc limit đã khớp): kiểm tra chạm TP1/SL/hết hạn như cũ.
    """
    now = datetime.now(timezone.utc)
    for rec in log:
        status = rec.get("status")
        if status not in ("pending", "waiting_fill"):
            continue
        try:
            rec_time = datetime.fromisoformat(rec["time_iso"])
        except Exception:
            rec["status"] = "expired"
            continue

        timeout = rec.get("timeout_hours", SIGNAL_TIMEOUT_HOURS)  # Zone Setup có thời hạn riêng (ngắn/trung/dài)

        if status == "waiting_fill":
            direction = rec["direction"]
            entry = rec["entry"]
            # BUY limit: chờ giá GIẢM về entry. SELL limit: chờ giá TĂNG về entry.
            filled = (direction == "BUY" and current_price <= entry) or \
                     (direction == "SELL" and current_price >= entry)
            if filled:
                rec["status"] = "pending"  # đã khớp -> từ giờ mới bắt đầu tính thắng/thua
            elif trading_hours_elapsed(rec_time, now) > timeout:
                rec["status"] = "expired"  # giá không bao giờ pullback về -> lệnh chưa từng vào, bỏ qua
            continue  # dù khớp hay chưa, vòng lặp này chưa xét thắng/thua

        # status == "pending": đã khớp lệnh thật sự, xét thắng/thua như bình thường
        if rec["direction"] == "BUY":
            if current_price >= rec["tp1"]:
                rec["status"] = "win"
                rec["usd"] = round((rec["tp1"] - rec["entry"]) * USD_PER_POINT, 2)
            elif current_price <= rec["sl"]:
                rec["status"] = "loss"
                rec["usd"] = round((rec["sl"] - rec["entry"]) * USD_PER_POINT, 2)  # số âm
        else:  # SELL
            if current_price <= rec["tp1"]:
                rec["status"] = "win"
                rec["usd"] = round((rec["entry"] - rec["tp1"]) * USD_PER_POINT, 2)
            elif current_price >= rec["sl"]:
                rec["status"] = "loss"
                rec["usd"] = round((rec["entry"] - rec["sl"]) * USD_PER_POINT, 2)  # số âm

        if rec["status"] == "pending" and trading_hours_elapsed(rec_time, now) > timeout:
            rec["status"] = "expired"
    return log


def manage_active_trades_before_append(log, sig, current_price):
    """
    Quản lý việc "nhồi lệnh" khi có tín hiệu MỚI cùng chiều + cùng loại (trend/mean_reversion)
    với 1 hoặc nhiều lệnh CŨ đang chạy (waiting_fill/pending). Áp dụng 3 quy tắc đã thống nhất:

    1. HÒA VỐN: lệnh cũ nào chưa chạm SL và giá hiện tại đang ở phía CÓ LỢI so với entry của nó
       (đã "an toàn") -> đóng lại luôn với kết quả "hòa vốn" (breakeven), không tính thắng/thua,
       giải phóng chỗ thay vì để nó "trôi" song song vô thời hạn.
    2. GIỚI HẠN NHỒI: sau khi hòa vốn xong, nếu số lệnh CÙNG chiều/loại còn đang chạy đã đạt
       MAX_STACK_PER_DIRECTION -> KHÔNG cho nhồi thêm, dù điểm số có mạnh hơn bao nhiêu.
    3. NHỒI CÓ LÝ DO: nếu còn dưới giới hạn, chỉ cho nhồi thêm khi điểm tín hiệu mới mạnh hơn
       lệnh đang chạy gần nhất ít nhất MIN_SCORE_IMPROVEMENT_TO_STACK điểm - kèm lý do rõ ràng.
       Nếu không đủ mạnh hơn, KHÔNG tạo bản ghi mới - chỉ trả về ghi chú để hiển thị tham khảo,
       tránh vừa nhồi lệnh vô tội vạ vừa làm loãng thống kê thắng/thua.

    Trả về: (log đã cập nhật, should_append: bool, stack_note: str hoặc None)
    """
    direction = sig.get("direction")
    mode = sig.get("signal_mode")
    if not direction or mode not in ("trend", "mean_reversion", "box"):
        return log, True, None

    active = [r for r in log if r.get("mode") == mode and r.get("direction") == direction
              and r.get("status") in ("waiting_fill", "pending")]

    # --- Bước 1: hòa vốn các lệnh cũ đã "an toàn" (chưa chạm SL, đang có lợi) ---
    for rec in active:
        if rec["status"] != "pending":
            continue  # lệnh limit chưa khớp thì chưa có gì để tính hòa vốn
        favorable = (direction == "BUY" and current_price > rec["entry"]) or \
                    (direction == "SELL" and current_price < rec["entry"])
        if favorable:
            rec["status"] = "breakeven"
            rec["usd"] = 0.0

    active = [r for r in active if r["status"] in ("waiting_fill", "pending")]  # cập nhật lại sau hòa vốn

    # --- Bước 2 + 3: quyết định có cho nhồi thêm không ---
    if not active:
        return log, True, None  # không có lệnh nào đang chạy -> tạo bình thường

    if len(active) >= MAX_STACK_PER_DIRECTION:
        note = (f"Đã đạt giới hạn {MAX_STACK_PER_DIRECTION} lệnh {direction} ({mode}) chạy song song "
                f"-> KHÔNG nhồi thêm, chỉ tham khảo phân tích lần này.")
        return log, False, note

    latest_active = active[-1]
    old_score = abs(latest_active.get("score", 0))
    new_score = abs(sig.get("score", 0))
    if new_score - old_score >= MIN_SCORE_IMPROVEMENT_TO_STACK:
        note = (f"📦 NHỒI LỆNH: điểm mới ({sig['score']}) mạnh hơn lệnh {direction} đang chạy "
                f"(điểm {latest_active.get('score', 0)}) từ {MIN_SCORE_IMPROVEMENT_TO_STACK}+ điểm.")
        return log, True, note

    entry_old = latest_active["entry"]
    usd_now = round((current_price - entry_old) * USD_PER_POINT, 2) if direction == "BUY" \
        else round((entry_old - current_price) * USD_PER_POINT, 2)
    note = (f"Đã có {len(active)} lệnh {direction} ({mode}) đang chạy (gần nhất entry {entry_old:.2f}, "
            f"hiện {'+' if usd_now >= 0 else ''}${usd_now}) - điểm mới ({sig['score']}) chưa đủ mạnh hơn "
            f"rõ rệt để nhồi thêm, chỉ tham khảo.")
    return log, False, note


def append_signal(log, sig):
    """Thêm tín hiệu vừa tạo (nếu có hướng BUY/SELL) vào log để theo dõi sau này."""
    if not sig.get("direction"):
        return log
    entry_type = sig.get("entry_type", "market")
    # Lệnh limit ("chờ giá về") bắt đầu ở trạng thái CHỜ KHỚP, không tính thắng/thua ngay.
    # Lệnh market (vào giá hiện tại) coi như khớp ngay lập tức.
    initial_status = "waiting_fill" if entry_type == "limit" else "pending"
    log.append({
        "time_iso": datetime.now(timezone.utc).isoformat(),
        "direction": sig["direction"],
        "entry": sig["entry"],
        "sl": sig["sl"],
        "tp1": sig["tp1"],
        "score": sig["score"],
        "mode": sig.get("signal_mode", "trend"),  # "trend"/"mean_reversion"/"zone_setup"/"experimental"
        "confidence": sig.get("confidence", "normal"),  # "normal" hoặc "low" - để so sánh 2 mức tin cậy
        "entry_type": entry_type,
        "status": initial_status,
        "timeout_hours": sig.get("timeout_hours", SIGNAL_TIMEOUT_HOURS),
    })
    return log


def append_zone_setup_secondary(log, sig):
    """
    Ghi thêm bản ghi cho Zone Setup THỨ 2 (chiều đối diện), nếu cả BUY lẫn SELL cùng
    xác nhận đồng thời ở 2 vùng khác nhau (vd: BUY tại hỗ trợ dưới + SELL tại kháng cự trên).
    """
    sec = sig.get("zone_setup_secondary")
    if not sec:
        return log
    log.append({
        "time_iso": datetime.now(timezone.utc).isoformat(),
        "direction": sec["direction"],
        "entry": sec["entry"],
        "sl": sec["sl"],
        "tp1": sec["tp1"],
        "score": sig["score"],
        "mode": "zone_setup",
        "confidence": "normal",
        "entry_type": "market",
        "status": "pending",
        "timeout_hours": sec["timeout_hours"],
    })
    return log


def active_zone_setup_directions(log):
    """
    Trả về tập các hướng (BUY/SELL) đang có Zone Setup CÒN HIỆU LỰC (chưa thắng/thua/hết hạn) -
    dùng để tránh tạo thêm setup trùng chiều mỗi 5 phút (không "chồng lệnh" cùng 1 bên).
    """
    return {r["direction"] for r in log
            if r.get("mode") == "zone_setup" and r.get("status") in ("waiting_fill", "pending")}


def compute_win_rate(log, mode=None, confidence=None):
    """
    Tính tỷ lệ thắng/thua + tổng $ lãi/lỗ + số lệnh hòa vốn. Nếu truyền mode
    ("trend" hoặc "mean_reversion"...), chỉ tính riêng loại đó. Nếu truyền confidence
    ("normal" hoặc "low"), lọc thêm theo độ tin cậy -> cho phép so sánh tín hiệu
    🟡 THẤP có thực sự kém hơn 🟢 bình thường không.
    Bản ghi log cũ (trước khi có trường 'mode'/'confidence') coi là 'trend'/'normal' để không mất dữ liệu.
    """
    if mode:
        log = [r for r in log if r.get("mode", "trend") == mode]
    if confidence:
        log = [r for r in log if r.get("confidence", "normal") == confidence]
    closed = [r for r in log if r.get("status") in ("win", "loss")]
    wins = [r for r in closed if r["status"] == "win"]
    breakevens = [r for r in log if r.get("status") == "breakeven"]
    if not closed and not breakevens:
        return None
    total_usd = round(sum(r.get("usd", 0) for r in closed), 2)
    return {
        "wins": len(wins),
        "losses": len(closed) - len(wins),
        "total": len(closed),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
        "breakevens": len(breakevens),
        "total_usd": total_usd,
    }


def active_trades_summary(log, current_price=None, max_count=3):
    """
    Liệt kê các lệnh CÒN HIỆU LỰC (chưa thắng/thua/hòa vốn/hết hạn), tách RIÊNG 2 loại để
    tránh nhầm lẫn "đã vào lệnh thật" với "mới chỉ là lệnh chờ chưa khớp":
    - "waiting": lệnh limit CHƯA khớp (giá chưa chạm tới) - chỉ là kế hoạch, chưa phải giao dịch thật
    - "running": lệnh ĐÃ khớp, đang chạy chờ chạm TP/SL - giao dịch thật đang mở
    Nếu truyền current_price, hiện thêm số $ lời/lỗ tạm tính cho lệnh đã khớp.
    Trả về dict {"waiting": [...], "running": [...]}.
    """
    now = datetime.now(timezone.utc)
    active = [r for r in log if r.get("status") in ("waiting_fill", "pending")]
    active = active[-max_count:]  # chỉ lấy các lệnh gần nhất, tránh tin nhắn quá dài

    waiting, running = [], []
    for rec in active:
        timeout = rec.get("timeout_hours", SIGNAL_TIMEOUT_HOURS)
        try:
            rec_time = datetime.fromisoformat(rec["time_iso"])
            hours_left = max(0, timeout - trading_hours_elapsed(rec_time, now))
        except Exception:
            hours_left = None

        icon = "🔁" if rec.get("mode") == "mean_reversion" else "📈"
        if rec["status"] == "waiting_fill":
            time_txt = f", còn {hours_left:.1f}h" if hours_left is not None else ""
            waiting.append(f"{icon} {rec['direction']} Limit @ {rec['entry']:.2f} (còn hiệu lực{time_txt})")
        else:  # pending - đã khớp, đang chạy chờ TP/SL
            usd_txt = ""
            if current_price is not None:
                usd_now = (current_price - rec["entry"]) * USD_PER_POINT if rec["direction"] == "BUY" \
                    else (rec["entry"] - current_price) * USD_PER_POINT
                usd_txt = f", {'+' if usd_now >= 0 else ''}${usd_now:.2f}"
            running.append(f"{icon} {rec['direction']} @ {rec['entry']:.2f} → TP {rec['tp1']:.2f}{usd_txt}")

    return {"waiting": waiting, "running": running}


def mean_reversion_signal(current_price, sr, rsi_value, atr_value, near_threshold=0.25):
    """
    Chiến lược RIÊNG cho lúc thị trường sideway (ADX thấp) — khác hẳn logic trend-following.
    Ý tưởng: trong sideway, giá dao động qua lại giữa hỗ trợ/kháng cự thay vì đi theo xu hướng.
    - Giá gần ĐÁY range + RSI thấp (chưa quá bán hẳn nhưng nghiêng yếu) -> kỳ vọng bật lên -> BUY
    - Giá gần ĐỈNH range + RSI cao -> kỳ vọng giảm trở lại -> SELL
    R:R thấp hơn logic trend (range hẹp thì target cũng phải gần, không kỳ vọng xa như lúc có trend).
    """
    support, resistance = sr["support"], sr["resistance"]
    range_size = resistance - support
    if range_size <= 0:
        return None

    position = (current_price - support) / range_size  # 0 = tại đáy, 1 = tại đỉnh range
    mid = (support + resistance) / 2

    direction = None
    if position <= near_threshold and rsi_value <= 45:
        direction = "BUY"
    elif position >= (1 - near_threshold) and rsi_value >= 55:
        direction = "SELL"

    if not direction:
        return None

    buffer = atr_value * 0.5
    if direction == "BUY":
        sl = support - buffer
        tp1 = mid
        tp2 = resistance
        tp3 = resistance + range_size * 0.3
    else:
        sl = resistance + buffer
        tp1 = mid
        tp2 = support
        tp3 = support - range_size * 0.3

    # Bỏ qua nếu target quá gần entry (range quá hẹp, không đáng vào lệnh sau khi trừ phí)
    if abs(tp1 - current_price) < atr_value * 0.5:
        return None

    return {
        "direction": direction, "entry": current_price, "sl": sl,
        "tp1": tp1, "tp2": tp2, "tp3": tp3, "position_in_range": round(position, 2),
    }


def has_reaction_candle(df, direction, atr_value):
    """
    Nến M5 xác nhận phản ứng ĐỦ MẠNH tại 1 vùng giá - điều kiện bắt buộc để Zone Setup
    được coi là "đã xác nhận" (không phải chờ mù trong tương lai). Chấp nhận 2 kiểu:
    1. Engulfing đúng chiều (đảo chiều rõ ràng)
    2. Nến có RÂU DÀI từ chối rõ (>=40% biên độ nến) + đóng cửa đúng hướng mong đợi
    Bắt buộc biên độ nến >= 1x ATR(M5) - lọc bớt phản ứng yếu ớt, không đáng tin (theo đề xuất
    đã thống nhất: 1 nến Engulfing/reject nhỏ xíu không đáng tin bằng nến biên độ lớn).
    """
    if len(df) < 2 or atr_value <= 0:
        return False
    c = df.iloc[-1]
    candle_range = c["high"] - c["low"]
    if candle_range < atr_value:
        return False  # nến quá nhỏ so với biến động trung bình, phản ứng yếu, bỏ qua

    pattern = detect_candle_pattern(df)
    if direction == "BUY" and pattern == "bullish_engulfing":
        return True
    if direction == "SELL" and pattern == "bearish_engulfing":
        return True

    body_low, body_high = min(c["open"], c["close"]), max(c["open"], c["close"])
    if direction == "BUY":
        lower_wick = body_low - c["low"]
        if lower_wick >= candle_range * 0.4 and c["close"] > c["open"]:
            return True
    else:
        upper_wick = c["high"] - body_high
        if upper_wick >= candle_range * 0.4 and c["close"] < c["open"]:
            return True
    return False


def dynamic_sl_distance(atr_value, min_sl=SL_MIN_POINTS, max_sl=SL_MAX_POINTS,
                         atr_low=ATR_SL_LOW, atr_high=ATR_SL_HIGH):
    """
    SL động trong khoảng [min_sl, max_sl] theo ATR(M5) hiện tại - ATR càng cao (biến động
    mạnh) thì SL càng gần max_sl (tránh bị stop-hunt), ATR càng thấp thì càng gần min_sl
    (không cần SL quá rộng khi thị trường đang yên). Nội suy tuyến tính giữa 2 mốc tham chiếu.
    """
    if atr_value <= atr_low:
        return min_sl
    if atr_value >= atr_high:
        return max_sl
    ratio = (atr_value - atr_low) / (atr_high - atr_low)
    return min_sl + ratio * (max_sl - min_sl)


def rr_profile_for_score(score, threshold, max_score=9):
    """
    Xác định hồ sơ R:R (tỷ lệ TP1/TP2/TP3 tính theo R = khoảng cách SL) dựa trên độ mạnh
    của điểm số Trend - điểm càng gần mức tối đa, thị trường càng mạnh, TP càng đặt xa hơn.
    Thay thế cho việc dùng ADX (đã bỏ theo yêu cầu) làm thước đo độ mạnh.
    - Điểm vừa đủ ngưỡng (yếu) -> 0.5R / 0.8R / 1.2R
    - Điểm ở giữa (trung bình) -> 0.7R / 1.3R / 2.0R
    - Điểm gần tối đa (mạnh)   -> 1.0R / 2.0R / 3.0R
    """
    span = max_score - threshold
    if span <= 0:
        return (1.0, 2.0, 3.0)
    ratio = (abs(score) - threshold) / span  # 0.0 (vừa đủ ngưỡng) -> 1.0 (tối đa)
    if ratio < 0.34:
        return (0.5, 0.8, 1.2)
    elif ratio < 0.67:
        return (0.7, 1.3, 2.0)
    return (1.0, 2.0, 3.0)


def rr_profile_for_tier(tier):
    """Zone Setup: dùng luôn 'tuổi thọ' nguồn (short/medium/long) làm hồ sơ R:R, nhất quán
    với ý tưởng nguồn càng lớn (H4) thì biên độ kỳ vọng càng xa."""
    return {
        "short": (0.5, 0.8, 1.2),
        "medium": (0.7, 1.3, 2.0),
        "long": (1.0, 2.0, 3.0),
    }.get(tier, (1.0, 2.0, 3.0))


def compute_walled_tps(direction, entry, sl_distance, rr_tuple, zones_same_side):
    """
    "Bức tường cản": tính 3 mức TP theo rr_tuple (R-multiple), nhưng mỗi mức tự động bị
    CHẶN LẠI nếu có 1 vùng cộng hưởng (>=1 sao, tức >=2 nguồn đồng thuận) nằm gần hơn mục
    tiêu lý thuyết - dùng biên gần của vùng đó làm TP thực tế thay vì phóng xuyên qua.
    Xử lý TUẦN TỰ: TP sau tìm bức tường TIẾP THEO (không lặp lại đúng bức tường TP trước
    đã dùng), đảm bảo TP1 <= TP2 <= TP3 (BUY) hoặc TP1 >= TP2 >= TP3 (SELL).
    zones_same_side: zones_above (nếu BUY) hoặc zones_below (nếu SELL), đã sắp gần->xa.
    """
    starred_walls = [z for z in zones_same_side if z.get("stars")]
    used_wall_idx = 0
    tps = []
    prev_price = entry
    for r in rr_tuple:
        raw = entry + r * sl_distance if direction == "BUY" else entry - r * sl_distance
        clamped = raw
        for i in range(used_wall_idx, len(starred_walls)):
            w = starred_walls[i]
            near_edge = w["price_low"] if direction == "BUY" else w["price_high"]
            in_between = (direction == "BUY" and prev_price < near_edge < raw) or \
                         (direction == "SELL" and raw < near_edge < prev_price)
            if in_between:
                clamped = near_edge
                used_wall_idx = i + 1  # bức tường này đã "dùng" - TP sau tìm bức tường kế tiếp
                break
        tps.append(clamped)
        prev_price = clamped

    # An toàn: đảm bảo thứ tự TP1/TP2/TP3 không bị đảo lộn sau khi chặn
    if direction == "BUY":
        tps[1] = max(tps[1], tps[0])
        tps[2] = max(tps[2], tps[1])
    else:
        tps[1] = min(tps[1], tps[0])
        tps[2] = min(tps[2], tps[1])
    return tuple(tps)


def zone_tier_info(sources, htf_cache, atr_m5):
    """
    Xác định 'tuổi thọ' của Zone Setup dựa theo nguồn XA NHẤT cấu thành vùng giá -
    đúng nguyên lý đã thống nhất: khung nguồn càng lớn, SL càng rộng, chờ càng lâu.
    - Có H4 trong nguồn -> DÀI HẠN: SL theo ATR(H4), chờ tối đa ~18 ngày (giao dịch)
    - Có H1 (không H4)  -> TRUNG HẠN: SL theo ATR(H1), chờ tối đa ~3 ngày (giao dịch)
    - Chỉ nguồn M5 (OB/Fib/HT/KC) -> NGẮN HẠN: SL theo ATR(M5), chờ tối đa 4 tiếng (mặc định)
    """
    atr_h1 = htf_cache.get("atr_h1") if htf_cache else None
    atr_h4 = htf_cache.get("atr_h4") if htf_cache else None

    if "H4" in sources and atr_h4:
        return {"tier": "long", "label": "DÀI HẠN (nguồn H4)", "atr": atr_h4, "timeout_hours": 24 * 18}
    if "H1" in sources and atr_h1:
        return {"tier": "medium", "label": "TRUNG HẠN (nguồn H1)", "atr": atr_h1, "timeout_hours": 24 * 3}
    return {"tier": "short", "label": "NGẮN HẠN (M5)", "atr": atr_m5, "timeout_hours": SIGNAL_TIMEOUT_HOURS}


def build_zone_setup_candidate(direction, zone, current_price, df_m5, atr_m5, htf_cache, opposite_zones):
    """
    Tạo 1 Zone Setup THẬT từ 1 vùng theo dõi - chỉ kích hoạt khi ĐỦ CẢ 3 điều kiện:
    1. Vùng có độ tin cậy (>=1 sao, tức >=2 nguồn đồng thuận) - không dùng vùng đơn lẻ 1 nguồn
    2. Giá đã chạy TỚI hoặc rất gần vùng đó (<=0.6x ATR) - không phải setup cho tương lai xa
    3. Có nến M5 xác nhận phản ứng đủ mạnh tại đó (has_reaction_candle)
    Entry = khoảng giá của vùng (không phải 1 điểm) + vào theo giá thị trường ngay khi xác nhận.
    SL/thời hạn theo "tuổi thọ" của nguồn (zone_tier_info). TP ưu tiên lấy từ vùng đối diện gần
    nhất (mức cấu trúc thật), không phải nhân hệ số tùy ý.
    """
    if not zone.get("stars"):
        return None
    if zone["distance_atr"] > 0.6:
        return None
    if not has_reaction_candle(df_m5, direction, atr_m5):
        return None

    tier = zone_tier_info(zone["sources"], htf_cache, atr_m5)
    sl_buffer = tier["atr"]

    if direction == "BUY":
        sl = zone["price_low"] - sl_buffer
    else:
        sl = zone["price_high"] + sl_buffer

    entry = current_price
    risk = abs(entry - sl)
    if risk <= 0:
        return None

    # Chỉ dùng vùng đối diện CÓ SAO (đáng tin cậy) làm mục tiêu TP - nhất quán với nguyên tắc
    # "bức tường cản" chỉ tính vùng cộng hưởng, không dùng vùng đơn lẻ 1 nguồn làm target.
    starred_opposite = [z for z in opposite_zones if z.get("stars")]
    r1, r2, r3 = rr_profile_for_tier(tier["tier"])  # fallback theo tuổi thọ nguồn nếu không có vùng đối diện phù hợp

    def _opp_price(z):
        return z["price_low"] if direction == "BUY" else z["price_high"]

    tp1 = _opp_price(starred_opposite[0]) if starred_opposite else None
    if tp1 is None or abs(tp1 - entry) < risk:  # vùng đối diện quá gần/không có -> fallback theo hồ sơ R của tier
        tp1 = entry + risk * r1 if direction == "BUY" else entry - risk * r1
    tp2 = _opp_price(starred_opposite[1]) if len(starred_opposite) > 1 else \
        (entry + risk * r2 if direction == "BUY" else entry - risk * r2)
    tp3 = _opp_price(starred_opposite[2]) if len(starred_opposite) > 2 else \
        (entry + risk * r3 if direction == "BUY" else entry - risk * r3)

    return {
        "direction": direction, "entry": entry, "entry_zone": (zone["price_low"], zone["price_high"]),
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "rr": round(abs(tp1 - entry) / risk, 2),
        "tier": tier["tier"], "tier_label": tier["label"], "timeout_hours": tier["timeout_hours"],
        "stars": zone["stars"], "sources": zone["sources"],
    }


def experimental_range_signal(current_price, sr, rsi_value, atr_value, near_threshold=0.4):
    """
    Hạng mục THỬ NGHIỆM - RỦI RO CAO HƠN mean-reversion chuẩn.
    Chỉ kích hoạt khi thị trường THỰC SỰ đứng yên: không đủ điều kiện trend, cũng không đủ
    điều kiện mean-reversion chuẩn (giá chưa ở sát biên range đủ rõ, RSI chưa đủ cực đoan).

    Điều kiện LỎNG HƠN mean-reversion (near_threshold rộng hơn: 0.4 thay vì 0.25, RSI chỉ
    cần lệch nhẹ khỏi 50 thay vì phải <=45/>=55) -> bắt được nhiều setup hơn, nhưng vì vậy
    ĐỘ TIN CẬY THẤP HƠN, CHƯA ĐƯỢC BACKTEST kỹ. Luôn có SL rõ ràng (không phải "cược mù"),
    nhưng khuyến nghị khối lượng vào lệnh NHỎ HƠN NHIỀU so với các hạng mục khác.
    Được theo dõi thắng/thua HOÀN TOÀN TÁCH RIÊNG để đánh giá bằng dữ liệu thật, không cảm tính.
    """
    support, resistance = sr["support"], sr["resistance"]
    range_size = resistance - support
    if range_size <= 0:
        return None

    position = (current_price - support) / range_size
    mid = (support + resistance) / 2

    direction = None
    if position <= near_threshold and rsi_value <= 50:
        direction = "BUY"
    elif position >= (1 - near_threshold) and rsi_value >= 50:
        direction = "SELL"

    if not direction:
        return None

    # SL chặt hơn mean-reversion chuẩn (rủi ro cao hơn thì phải kiểm soát chặt hơn, không phải lỏng hơn)
    sl_distance = max(atr_value * 1.0, range_size * 0.15)
    if direction == "BUY":
        sl = current_price - sl_distance
        tp1 = mid
    else:
        sl = current_price + sl_distance
        tp1 = mid

    if abs(tp1 - current_price) < atr_value * 0.3:
        return None  # target quá gần, không đáng vào lệnh sau khi trừ phí

    tp1_dist = abs(tp1 - current_price)
    if direction == "BUY":
        tp2 = current_price + tp1_dist * 1.3
        tp3 = current_price + tp1_dist * 1.6
    else:
        tp2 = current_price - tp1_dist * 1.3
        tp3 = current_price - tp1_dist * 1.6

    return {
        "direction": direction, "entry": current_price, "sl": sl,
        "tp1": tp1, "tp2": tp2, "tp3": tp3, "position_in_range": round(position, 2),
    }


def check_entry_chase(direction, current_price, ob, atr_value, max_atr_distance=2.5):
    """
    Kiểm tra giá hiện tại đã chạy quá xa vùng Order Block chưa (nguy cơ "mua đuổi/bán đuổi").
    Nếu quá xa (>= max_atr_distance x ATR), trả về gợi ý entry CHỜ (limit) tại biên gần
    của OB thay vì entry thị trường ngay - giống nguyên tắc "không mua đuổi, chờ giá về".
    """
    if not ob:
        return None
    zone_low, zone_high = ob["zone"]

    if direction == "BUY":
        ref_edge = zone_high  # biên gần nhất để chờ giá hồi về khi đang ở trên vùng OB
        distance = current_price - ref_edge
    else:
        ref_edge = zone_low
        distance = ref_edge - current_price

    if distance <= 0 or atr_value <= 0:
        return None  # giá còn trong/chưa vượt vùng OB, chưa cần cảnh báo

    distance_atr = distance / atr_value
    if distance_atr >= max_atr_distance:
        return {"distance_atr": round(distance_atr, 1), "suggested_entry": ref_edge}
    return None


def _point_zone(price, tag, buffer_atr, atr_value):
    """Biến 1 điểm giá thành 1 KHOẢNG phản ứng (range) bằng cách nới ra 2 bên theo ATR."""
    b = buffer_atr * atr_value
    return {"tag": tag, "price_low": price - b, "price_high": price + b}


def build_raw_zones(ob, sr, fib, htf_levels, atr_value):
    """
    Thu thập vùng giá từ mọi nguồn, biểu diễn dưới dạng KHOẢNG (price_low, price_high) -
    đúng bản chất: thị trường phản ứng trong 1 vùng, không phải 1 điểm tuyệt đối.
    - Order Block: dùng thẳng khoảng thật của nến OB (đã là 1 range sẵn)
    - Hỗ trợ/Kháng cự/Fib/H1/H4: là các mức 1 điểm -> nới ra thành khoảng nhỏ theo ATR
    """
    zones = []
    if ob:
        zone_low, zone_high = ob["zone"]
        zones.append({"tag": "OB", "price_low": zone_low, "price_high": zone_high})
    if sr:
        zones.append(_point_zone(sr["support"], "HT", 0.3, atr_value))
        zones.append(_point_zone(sr["resistance"], "KC", 0.3, atr_value))
    if fib:
        zones.append(_point_zone(fib["levels"]["61.8"], "Fib", 0.25, atr_value))
    for lv in htf_levels:
        zones.append(_point_zone(lv["price"], lv["tf"], 0.2, atr_value))
    return zones


def merge_zones_into_ranges(zones, atr_value, merge_gap_mult=0.4):
    """
    Gộp các khoảng CHỒNG LẤN hoặc đủ GẦN NHAU (trong phạm vi merge_gap_mult x ATR) thành
    1 vùng phản ứng duy nhất - đây chính là cách tính "confluence" đúng bản chất: nhiều
    nguồn độc lập cùng rơi vào 1 khu vực thì gộp lại, càng nhiều nguồn gộp -> càng đáng tin.
    """
    if not zones:
        return []
    gap = atr_value * merge_gap_mult
    sorted_zones = sorted(zones, key=lambda z: z["price_low"])

    clusters = []
    current = dict(sorted_zones[0])
    current["sources"] = {sorted_zones[0]["tag"]}
    for z in sorted_zones[1:]:
        if z["price_low"] <= current["price_high"] + gap:
            current["price_low"] = min(current["price_low"], z["price_low"])
            current["price_high"] = max(current["price_high"], z["price_high"])
            current["sources"].add(z["tag"])
        else:
            clusters.append(current)
            current = dict(z)
            current["sources"] = {z["tag"]}
    clusters.append(current)

    for c in clusters:
        c["sources"] = sorted(c["sources"])
        count = len(c["sources"])
        c["stars"] = "⭐⭐" if count >= 3 else ("⭐" if count == 2 else "")
    return clusters


def finalize_watch_zones(clusters, current_price, atr_value, max_per_side=3):
    """
    Tính khoảng cách từ giá hiện tại tới từng vùng (0 nếu giá đang NẰM TRONG vùng),
    xác định loại lệnh chờ phù hợp, tách theo Trên/Dưới, lọc bớt vùng quá gần (<0.3 ATR).
    """
    result = []
    for c in clusters:
        if current_price < c["price_low"]:
            # vùng nằm TRÊN giá hiện tại -> giá phải tăng mới chạm -> đóng vai trò kháng cự
            distance_atr = round((c["price_low"] - current_price) / atr_value, 1)
            order_type = "Sell Limit"
        elif current_price > c["price_high"]:
            # vùng nằm DƯỚI giá hiện tại -> giá phải giảm mới chạm -> đóng vai trò hỗ trợ
            distance_atr = round((current_price - c["price_high"]) / atr_value, 1)
            order_type = "Buy Limit"
        else:
            continue  # giá đang nằm ngay trong vùng - không phải vùng "chờ tới" nữa
        if distance_atr < 0.3:
            continue
        result.append({**c, "distance_atr": distance_atr, "order_type": order_type})

    above = sorted([z for z in result if z["price_low"] > current_price], key=lambda z: z["distance_atr"])
    below = sorted([z for z in result if z["price_high"] < current_price], key=lambda z: z["distance_atr"])
    return above[:max_per_side], below[:max_per_side]


# ============================================================
# 3. LOGIC TẠO TÍN HIỆU
# ============================================================
def generate_signal(active_zone_directions=None):
    active_zone_directions = active_zone_directions or set()
    # Chỉ gọi API 1 lần (lấy nhiều nến M5), sau đó tự gộp thành M15/M30/H1
    # -> tiết kiệm request, cho phép chạy mỗi 5 phút mà vẫn trong hạn mức free
    df_m5 = get_ohlc("5min", outputsize=1000)  # ~3.5 ngày dữ liệu M5
    df_m15 = resample_ohlc(df_m5, "15min")
    df_m30 = resample_ohlc(df_m5, "30min")
    df_h1 = resample_ohlc(df_m5, "1h")

    # Phát hiện thị trường ĐANG ĐỨNG YÊN (nghỉ lễ, ngoài lịch cuối tuần cố định) dựa
    # thẳng vào dữ liệu giá thật - is_market_closed() chỉ bắt được cuối tuần, không
    # bắt được ngày nghỉ lễ vì nó không nằm trong lịch cứng.
    market_flat = is_market_flat(df_m5)

    trend_m5 = detect_trend(df_m5)
    trend_m15 = detect_trend(df_m15)
    trend_m30 = detect_trend(df_m30)

    pattern = detect_candle_pattern(df_m5)
    bos = detect_bos(df_m5)
    fvg = detect_fvg(df_m5)
    ob = detect_order_block(df_m5)

    # BOS đa khung kèm mức giá đã phá vỡ - dùng để chờ giá quay lại test (retest) thay vì
    # vào lệnh ngay lúc breakout. Khung càng lớn thì thời gian chờ quay lại test càng lâu.
    bos_m5_info = detect_bos_level(df_m5)
    bos_m15_info = detect_bos_level(df_m15)
    bos_h1_info = detect_bos_level(df_h1)

    rsi_m5 = rsi(df_m5["close"]).iloc[-1]
    atr_m5 = atr(df_m5).iloc[-1]
    atr_m15_series = atr(df_m15)
    adx_m15 = adx(df_m15).iloc[-1]
    sr = support_resistance(df_m5)
    fib = fibonacci_levels(df_m30, lookback=50)
    current_price = df_m5.iloc[-1]["close"]

    # Cache mức giá H1/H4 lịch sử - chỉ tải lại mỗi giờ, không tốn request mỗi lần chạy
    htf_cache = refresh_htf_cache_if_needed()
    htf_levels = nearest_htf_levels(htf_cache, current_price, atr_m5, max_count=6)

    # Vùng theo dõi: mỗi nguồn là 1 KHOẢNG giá (không phải điểm tuyệt đối), các khoảng
    # chồng lấn/gần nhau tự động gộp thành 1 vùng phản ứng duy nhất (confluence tự nhiên).
    # Tính SỚM ở đây (trước khi quyết định hướng) vì Zone Setup bên dưới cần dùng đến.
    raw_zones = build_raw_zones(ob, sr, fib, htf_levels, atr_m5)
    clusters = merge_zones_into_ranges(raw_zones, atr_m5)
    zones_above, zones_below = finalize_watch_zones(clusters, current_price, atr_m5)

    # Mẫu hình nến mẹ - nến con (Inside Bar), quét trên M15 theo đề xuất
    # (M5 quá nhiễu cho pattern này, M15 phản ánh cấu trúc rõ hơn)
    inside_bar = detect_inside_bar_setup(df_m15, atr_m15_series)

    # MACD - đo momentum (tốc độ/gia tốc), khác trend (chỉ đo hướng)
    macd_line, signal_line, hist = macd(df_m5["close"])
    hist_now = hist.iloc[-1]
    macd_bias = "up" if hist_now > 0 else ("down" if hist_now < 0 else None)
    divergence = detect_divergence(df_m5, hist)

    session_ok = is_active_session()
    news_warning = check_upcoming_news()

    # --- BOX DETECTOR: thay thế hoàn toàn hệ thống chấm điểm cũ (Trend/Mean-Reversion/
    # Zone Setup/Thử nghiệm). Tìm box H1 TRƯỚC (cấu trúc lớn), sau đó ràng buộc box M15 phải
    # nằm LỌT trong phạm vi box H1 đã chọn - đúng cấu trúc "range nhỏ lồng trong range to".
    # Cả 2 đều tự loại bỏ box đã quá xa giá hiện tại (không còn liên quan để giao dịch).
    atr_h1_series = atr(df_h1)
    box_h1 = find_box_state(df_h1, atr_h1_series)
    box_m15 = None
    if box_h1:
        box_m15 = find_box_state(df_m15, atr_m15_series,
                                  bound_range=(box_h1["box_low"], box_h1["box_high"]))
    else:
        box_m15 = find_box_state(df_m15, atr_m15_series)  # không có box H1 thì M15 tìm tự do
    box_signal = build_box_signal(box_m15, box_h1, atr_m5)

    # Xu hướng CHÍNH theo Lý thuyết Dow (chuỗi đỉnh/đáy H1) - dùng để đối chiếu với hướng lệnh
    # box, không phải điều kiện chặn cứng mà chỉ là 1 lớp cảnh báo thêm khi 2 bên mâu thuẫn.
    dow_trend = detect_dow_trend(df_h1)
    if box_signal:
        box_signal["dow_trend"] = dow_trend

    # Ichimoku Kumo (mây) H1 - tương tự Dow, chỉ cảnh báo khi mâu thuẫn ở mức MẠNH NHẤT
    # (giá trên mây xanh / dưới mây đỏ) - các trường hợp "mới hình thành"/"trong mây" chưa đủ
    # rõ ràng để coi là mâu thuẫn thật sự, không hạ độ tin cậy vì lý do đó.
    ichimoku_h1 = detect_ichimoku_signal(df_h1)
    if box_signal:
        box_signal["ichimoku"] = ichimoku_h1

    # Mũi tên xu hướng đa khung (H4/H1/M30/M15/M5) - chỉ để theo dõi trực quan, không ảnh
    # hưởng logic ra lệnh.
    df_h4_arrow = resample_ohlc(df_m5, "4h")
    trend_arrows = {
        "H4": detect_trend(df_h4_arrow) if len(df_h4_arrow) >= 21 else None,
        "H1": detect_trend(df_h1) if len(df_h1) >= 21 else None,
        "M30": detect_trend(df_m30) if len(df_m30) >= 21 else None,
        "M15": detect_trend(df_m15) if len(df_m15) >= 21 else None,
        "M5": detect_trend(df_m5) if len(df_m5) >= 21 else None,
    }

    direction = None
    block_reason = None
    signal_mode = "box"
    confidence = "normal"
    confidence_notes = []

    if market_flat:
        block_reason = "Thị trường đang đứng yên (nghỉ lễ/ngoài giờ giao dịch thực) - dữ liệu gần như không đổi, tạm dừng phân tích"
    elif not box_signal:
        block_reason = "Chưa tìm thấy nến tập trung thanh khoản nào đủ điều kiện trong phạm vi quét"
    elif box_signal["state"] == "ready":
        direction = box_signal["direction"]
        signal_mode = "box"
        if box_signal["alignment"] == "ngược":
            confidence = "low"
            confidence_notes.append("Lệnh NGƯỢC xu hướng của nến thanh khoản (chỉ 2 điểm entry, thận trọng hơn)")
        dow_conflict = (dow_trend == "up" and direction == "SELL") or (dow_trend == "down" and direction == "BUY")
        if dow_conflict:
            confidence = "low"
            dow_txt = "TĂNG" if dow_trend == "up" else "GIẢM"
            confidence_notes.append(f"Ngược xu hướng chính theo Dow (H1 đang {dow_txt} theo chuỗi đỉnh/đáy) - rủi ro cao hơn")
        if ichimoku_h1:
            ichimoku_conflict = (ichimoku_h1["strength"] == "strong_bull" and direction == "SELL") or \
                                 (ichimoku_h1["strength"] == "strong_bear" and direction == "BUY")
            if ichimoku_conflict:
                confidence = "low"
                pos_txt = "TRÊN mây XANH" if ichimoku_h1["strength"] == "strong_bull" else "DƯỚI mây ĐỎ"
                confidence_notes.append(f"Ngược Ichimoku H1 (giá đang {pos_txt} - xác nhận mạnh) - rủi ro cao hơn")
    else:
        state_txt = "chưa xác nhận breakout" if box_signal["state"] == "unconfirmed" else "đã xác nhận, đang chờ giá quay về test biên"
        block_reason = f"Có box {box_signal['tf']} ({state_txt}) - xem chi tiết box bên dưới"

    if direction and news_warning:
        confidence = "low"
        confidence_notes.append(f"Sắp có tin quan trọng: {news_warning} (rủi ro SL bị gap qua khi tin ra)")

    liquidity_note = None
    if not session_ok:
        liquidity_note = "Đang ngoài phiên thanh khoản cao (London/New York) — giá dễ nhiễu/đi ngang hơn bình thường, cân nhắc khối lượng nhỏ hơn nếu vào lệnh."

    # % thay đổi so với ~24 giờ trước (ước lượng thô từ khung H1)
    try:
        ref_price = df_h1.iloc[max(0, len(df_h1) - 24)]["close"]
        pct_change = (current_price - ref_price) / ref_price * 100
    except Exception:
        pct_change = None

    # Mức độ mạnh hiển thị: box thuận xu hướng đáng tin hơn box ngược, box chưa xác nhận thấp nhất
    strength_10 = {"thuận": 8.0, "ngược": 5.0}.get(box_signal.get("alignment") if box_signal else None, 3.0)
    score = 0  # không còn khái niệm điểm số - giữ field để tương thích ngược với log/thống kê cũ

    # --- Nhận định tổng quan (ghép các yếu tố thành 1-2 câu dễ hiểu) ---
    notes = []
    if rsi_m5 >= 70:
        notes.append("RSI cho thấy vùng quá mua, cẩn trọng nếu mua đuổi")
    elif rsi_m5 <= 30:
        notes.append("RSI cho thấy vùng quá bán, cẩn trọng nếu bán đuổi")
    else:
        notes.append("RSI trung tính, chưa quá mua/quá bán")

    trend_count_up = sum(1 for t in [trend_m5, trend_m15, trend_m30] if t == "up")
    if trend_count_up == 3:
        notes.append("cả 3 khung đều đồng thuận tăng")
    elif trend_count_up == 0:
        notes.append("cả 3 khung đều đồng thuận giảm")
    else:
        notes.append("các khung thời gian đang lệch hướng nhau")

    dist_to_res = sr["resistance"] - current_price
    dist_to_sup = current_price - sr["support"]
    if dist_to_res < dist_to_sup:
        notes.append(f"giá đang gần kháng cự {sr['resistance']:.2f} hơn, khả năng bị cản")
    else:
        notes.append(f"giá đang gần hỗ trợ {sr['support']:.2f} hơn, khả năng được nâng đỡ")

    overview = "; ".join(notes) + "."

    result = {
        "time": datetime.now().strftime("%H:%M:%S %d/%m"),
        "price": current_price,
        "pct_change": pct_change,
        "score": score,
        "strength_10": strength_10,
        "direction": direction,
        "trend_m5": trend_m5,
        "trend_m15": trend_m15,
        "trend_m30": trend_m30,
        "pattern": pattern,
        "bos": bos,
        "fvg": fvg,
        "ob": ob,
        "inside_bar": inside_bar,
        "rsi": rsi_m5,
        "atr": atr_m5,
        "adx": adx_m15,
        "support": sr["support"],
        "resistance": sr["resistance"],
        "overview": overview,
        "session_ok": session_ok,
        "liquidity_note": liquidity_note,
        "news_warning": news_warning,
        "block_reason": block_reason,
        "fib": fib,
        "fib_note": None,
        "signal_mode": signal_mode,
        "entry_type": "market",
        "chase_warning": None,
        "zones_above": zones_above,
        "zones_below": zones_below,
        "market_flat": market_flat,
        "confidence": confidence,
        "confidence_notes": confidence_notes,
        "macd_hist": hist_now,
        "macd_bias": macd_bias,
        "divergence": divergence,
        "momentum_note": None,
        "zone_setup_secondary": None,
        "timeout_hours": SIGNAL_TIMEOUT_HOURS,
        "bos_retest_note": None,
        "box_signal": box_signal,
        "box_chart_df": (df_h1 if box_signal["tf"] == "H1" else df_m15) if box_signal else None,
        "trend_arrows": trend_arrows,
    }

    if box_signal and box_signal["state"] == "ready":
        # Entry ĐẦU TIÊN trong danh sách làm đại diện theo dõi thắng/thua (các entry còn lại
        # hiển thị đầy đủ trong tin nhắn để bạn tự chọn/vào thang, nhưng chỉ 1 được log lại).
        primary = box_signal["entries"][0]
        result.update({
            "entry": primary["entry"], "sl": primary["sl"],
            "tp1": primary["tp1"], "tp2": primary["tp2"], "tp3": primary["tp3"],
            "entry_type": "limit",
        })
        result["fib_note"] = fib_confluence_note(fib, primary["entry"], primary["sl"], primary["tp1"], atr_m5)

    return result


# ============================================================
# 4. FORMAT TIN NHẮN & GỬI TELEGRAM
# ============================================================
def format_message(sig, win_stats=None, active_trades=None):
    """
    2 kiểu tin nhắn:
    - CÓ tín hiệu (BUY/SELL): đầy đủ chi tiết kỹ thuật, đặt trong khối "🆕 TÍN HIỆU MỚI"
      có tiêu đề + đường kẻ riêng - tách biệt rõ với "lệnh đang chạy" (khối khác, dễ nhầm
      nếu đọc lướt).
    - KHÔNG có tín hiệu (đa số các lần chạy): RÚT GỌN mạnh, không cần khối tiêu đề to.
    Các khối "Lệnh đang chạy"/"Vùng theo dõi"/"Thống kê" đều có tiêu đề + đường kẻ riêng,
    và tự ẩn khi không có gì để hiển thị (không hiện khối rỗng).
    """
    SEP = "━━━━━━━"

    def section_header(emoji, label):
        return f"{SEP} {emoji} {label} {SEP}"

    trend_icon = lambda t: "⬆️" if t == "up" else "⬇️"
    if sig.get("confidence") == "low" and sig["direction"]:
        icon = "🟡"  # vàng = tín hiệu có nhưng độ tin cậy thấp, khác với xanh/đỏ bình thường
    else:
        icon = "🟢" if sig["direction"] == "BUY" else ("🔴" if sig["direction"] == "SELL" else "⚪")

    lines = []
    price_line = f"⚡ XAU/USD {sig['price']:.2f}"
    if sig["pct_change"] is not None:
        price_line += f" ({sig['pct_change']:+.2f}%)"
    price_line += f"   {sig['time']}"
    lines.append(price_line)

    # Mũi tên xu hướng đa khung (H4/H1/M30/M15/M5) - luôn hiển thị để theo dõi trực quan
    arrows = sig.get("trend_arrows")
    if arrows:
        arrow_txt = "  ".join(
            f"{tf}:{'🟢⬆️' if t == 'up' else ('🔴⬇️' if t == 'down' else '➖')}"
            for tf, t in arrows.items()
        )
        lines.append(arrow_txt)
    lines.append("")

    # ---------- Khối Box: LUÔN hiển thị nếu tìm thấy box (kể cả chưa xác nhận/chưa retest) ----------
    box = sig.get("box_signal")
    if box:
        state_label = {"ready": "SẴN SÀNG ENTRY", "waiting_retest": "CHỜ RETEST",
                       "unconfirmed": "CHƯA XÁC NHẬN"}[box["state"]]
        header_emoji = "🆕" if box["state"] == "ready" else "📦"
        lines.append(section_header(header_emoji, f"BOX {box['tf']} - {state_label}"))
        color_txt = "🟢 XANH" if box["color"] == "green" else "🔴 ĐỎ"
        lines.append(f"Nến thanh khoản {color_txt}   |   Biên: {box['box_low']:.2f} – {box['box_high']:.2f}")
        dow_trend = box.get("dow_trend")
        if dow_trend:
            lines.append(f"📐 Xu hướng chính (Dow, H1): {'TĂNG (HH+HL)' if dow_trend == 'up' else 'GIẢM (LH+LL)'}")

        ichimoku = box.get("ichimoku")
        if ichimoku:
            ichimoku_txt = {
                "strong_bull": "TRÊN mây XANH (tăng, mạnh nhất)",
                "new_bull": "TRÊN mây ĐỎ (tăng, mới hình thành)",
                "strong_bear": "DƯỚI mây ĐỎ (giảm, mạnh nhất)",
                "new_bear": "DƯỚI mây XANH (giảm, mới hình thành)",
                "unclear": "TRONG mây (chưa rõ xu hướng)",
            }[ichimoku["strength"]]
            lines.append(f"☁️ Ichimoku (H1): Giá {ichimoku_txt}")

        if box["state"] == "unconfirmed":
            if box.get("in_middle"):
                lines.append("🚫 Giá đang Ở GIỮA range (không gần biên nào) - KHÔNG khuyến khích vào lệnh lúc này.")
                lines.append("   Chờ giá tiến gần biên trên/dưới hơn để có điểm tham chiếu SL/TP hợp lý.")
            else:
                lines.append("⚠️ Box mới hình thành, CHƯA có nến xác nhận breakout - chỉ có entry RỦI RO:")
                for e in box["entries"]:
                    e_icon = "🟢" if e["direction"] == "BUY" else "🔴"
                    lines.append(f"   {e_icon} {e['direction']} tại {e['label']} ({e['entry']:.2f})  "
                                  f"SL {e['sl']:.2f}  TP {e['tp1']:.2f}/{e['tp2']:.2f}/{e['tp3']:.2f}")
        else:
            align_txt = "THUẬN xu hướng" if box["alignment"] == "thuận" else "NGƯỢC xu hướng (thận trọng hơn)"
            lines.append(f"Xác nhận phá {'đỉnh' if box['confirm_dir']=='up' else 'đáy'} -> "
                          f"{box['direction']} {align_txt}")
            for e in box["entries"]:
                lines.append(f"   📍 {e['label']} ({e['entry']:.2f})  SL {e['sl']:.2f}  "
                              f"TP {e['tp1']:.2f}/{e['tp2']:.2f}/{e['tp3']:.2f}")
            if box["state"] == "waiting_retest":
                lines.append("⏳ Đã xác nhận nhưng giá CHƯA quay về test biên - chưa nên vào lệnh")
            else:
                lines.append("✅ Giá ĐÃ quay về test - đủ điều kiện xem xét entry")

        if sig.get("confidence") == "low":
            for note in sig.get("confidence_notes", []):
                lines.append(f"🚨 {note}")
        if sig["liquidity_note"]:
            lines.append("⚠️ Thanh khoản thấp (ngoài phiên chính)")

        if sig.get("stack_note"):
            prefix = "✅" if sig.get("stack_appended") else "🚫"
            lines.append(f"{prefix} {sig['stack_note']}")
        lines.append("")
    else:
        lines.append(f"⚪ {sig['block_reason'] if sig['block_reason'] else 'Chưa tìm thấy box nào để theo dõi'}")
        lines.append("")

    # ---------- Khối lệnh: TÁCH RIÊNG "chờ khớp" (chưa phải giao dịch thật) và "đang chạy"
    # (đã khớp, giao dịch thật đang mở) - tránh nhầm lẫn giữa kế hoạch và lệnh đã vào thật ----------
    if active_trades and (active_trades.get("waiting") or active_trades.get("running")):
        if active_trades.get("waiting"):
            lines.append(section_header("⏳", "ĐANG CHỜ KHỚP (chưa vào lệnh)"))
            for t in active_trades["waiting"]:
                lines.append(t)
            lines.append("")
        if active_trades.get("running"):
            lines.append(section_header("✅", "ĐANG CHẠY (đã khớp)"))
            for t in active_trades["running"]:
                lines.append(t)
            lines.append("")

    # ---------- Khối "📋 VÙNG THEO DÕI": LUÔN hiển thị (cả khi có tín hiệu lẫn không) ----------
    def _fmt_zone(z):
        tags = "+".join(z["sources"])
        return f"{z['price_low']:.2f}–{z['price_high']:.2f}({tags}{z.get('stars', '')})"

    if sig.get("zones_above") or sig.get("zones_below"):
        lines.append(section_header("📋", "VÙNG THEO DÕI"))
        if sig.get("zones_above"):
            lines.append("🔼 Trên: " + "  ".join(_fmt_zone(z) for z in sig["zones_above"]))
        if sig.get("zones_below"):
            lines.append("🔽 Dưới: " + "  ".join(_fmt_zone(z) for z in sig["zones_below"]))
        lines.append("")

    # ---------- Khối "📊 THỐNG KÊ": thắng/thua 3 nhóm, mỗi nhóm 1 dòng gọn (kèm $ + hòa vốn) ----------
    if win_stats:
        def _stat_txt(s):
            if not s:
                return "chưa đủ dữ liệu"
            be_txt = f"/{s['breakevens']}BE" if s.get("breakevens") else ""
            usd_txt = f" {'+' if s['total_usd'] >= 0 else ''}${s['total_usd']}"
            return f"{s['wins']}W/{s['losses']}L{be_txt} ({s['win_rate']}%){usd_txt}"

        lines.append(section_header("📊", "THỐNG KÊ"))
        lines.append(f"📦 Box đã xác nhận: {_stat_txt(win_stats.get('box'))}")
        lines.append("")

    lines.append("⚠️ Chỉ tham khảo | Quản lý vốn 1-2%")

    return "\n".join(lines)




def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    r = requests.post(url, data=payload, timeout=15)
    if r.status_code != 200:
        raise Exception(f"Lỗi gửi Telegram: {r.text}")
    return r.json()


def generate_box_chart(df, box, filename="box_chart.png", lookback_candles=40):
    """
    Vẽ biểu đồ nến kèm box (biên trên/dưới tô màu, đường giữa biên) để gửi kèm Telegram -
    giúp nhìn trực quan hơn thay vì chỉ đọc số. Dùng backend 'Agg' (không cần màn hình đồ
    họa) để chạy được trên GitHub Actions. Giảm số nến + tăng độ phân giải + thêm nhãn thời
    gian để đỡ bị "dính" nến khi nén ảnh nhỏ gửi qua Telegram xem trên điện thoại.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    recent = df.iloc[-lookback_candles:].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    # Sàn tối thiểu để nến doji (thân gần như = 0) vẫn hiện được 1 vạch mỏng nhìn thấy -
    # tính theo % biên độ HIỂN THỊ trên biểu đồ (không phải % giá tuyệt đối, tránh phóng to
    # sai thân nến thật khi giá vàng ở mức cao như ~4100 USD).
    price_range = recent["high"].max() - recent["low"].min()
    min_body_height = max(price_range * 0.003, 0.01)

    for i, row in recent.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=1)
        body_low = min(row["open"], row["close"])
        body_high = max(row["open"], row["close"])
        ax.add_patch(plt.Rectangle((i - 0.35, body_low), 0.7, max(body_high - body_low, min_body_height),
                                    facecolor=color, edgecolor=color))

    if box:
        box_color = "#2196f3" if box["color"] == "green" else "#ff9800"
        ax.axhspan(box["box_low"], box["box_high"], alpha=0.15, color=box_color)
        ax.axhline(box["box_high"], color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(box["box_low"], color="gray", linestyle="--", linewidth=0.8)
        if box["state"] != "unconfirmed":
            ax.axhline(box["box_mid"], color="gray", linestyle=":", linewidth=0.6)
        state_txt = {"ready": "SẴN SÀNG", "waiting_retest": "CHỜ RETEST", "unconfirmed": "CHƯA XÁC NHẬN"}[box["state"]]
        ax.set_title(f"XAU/USD - Box {box['tf']} ({state_txt})")
    else:
        ax.set_title("XAU/USD")

    # Nhãn thời gian trục ngang - chỉ hiện ~8 mốc để không rối, định dạng ngắn gọn dd/mm HH:MM
    if "datetime" in recent.columns:
        n_ticks = min(8, len(recent))
        tick_idx = np.linspace(0, len(recent) - 1, n_ticks).astype(int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([recent["datetime"].iloc[j].strftime("%d/%m %H:%M") for j in tick_idx],
                            rotation=30, ha="right", fontsize=8)

    ax.set_xlim(-1, len(recent))
    ax.set_ylabel("Giá (USD)")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close(fig)
    return filename


def send_telegram_photo(photo_path, caption=""):
    """Gửi ảnh qua Telegram (sendPhoto) - caption giới hạn 1024 ký tự nên chỉ dùng caption
    ngắn, nội dung đầy đủ đã gửi riêng bằng send_telegram()."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1024]}
        r = requests.post(url, data=data, files=files, timeout=30)
    if r.status_code != 200:
        raise Exception(f"Lỗi gửi ảnh Telegram: {r.text}")
    return r.json()


# ============================================================
# 5. CHẠY BOT
# ============================================================
if __name__ == "__main__":
    if is_market_closed():
        print("Thị trường XAU/USD đang đóng cửa cuối tuần -> bỏ qua lần chạy này (không gọi API, không gửi Telegram).")
        exit(0)

    log = load_signal_log()
    active_zone_dirs = active_zone_setup_directions(log)  # vestigial, không còn dùng (Zone Setup đã bị thay thế)

    print("Đang lấy dữ liệu và phân tích...")
    signal = generate_signal(active_zone_directions=active_zone_dirs)

    if signal.get("market_flat"):
        print("Thị trường đang đứng yên (nghỉ lễ/dữ liệu không đổi) -> bỏ qua lần chạy này, không gửi Telegram.")
        exit(0)

    # --- Cập nhật kết quả các tín hiệu cũ TRƯỚC khi xét nhồi lệnh/hòa vốn cho tín hiệu mới ---
    log = update_signal_outcomes(log, signal["price"])

    # --- Quản lý nhồi lệnh: hòa vốn lệnh cũ đã an toàn, chỉ cho nhồi thêm khi điểm mạnh hơn rõ rệt ---
    log, should_append, stack_note = manage_active_trades_before_append(log, signal, signal["price"])
    signal["stack_note"] = stack_note
    signal["stack_appended"] = should_append

    active_trades = active_trades_summary(log, current_price=signal["price"])

    if should_append:
        log = append_signal(log, signal)
    save_signal_log(log)
    win_stats = {
        "box": compute_win_rate(log, mode="box"),
    }

    message = format_message(signal, win_stats=win_stats, active_trades=active_trades)
    print(message)

    print("\nĐang gửi vào Telegram...")
    send_telegram(message)

    if signal.get("box_signal") and signal.get("box_chart_df") is not None:
        try:
            box = signal["box_signal"]
            chart_path = generate_box_chart(signal["box_chart_df"], box)
            state_txt = {"ready": "Sẵn sàng entry", "waiting_retest": "Chờ retest",
                         "unconfirmed": "Chưa xác nhận"}[box["state"]]
            send_telegram_photo(chart_path, caption=f"📊 Box {box['tf']} - {state_txt}")
            print("Đã gửi ảnh biểu đồ.")
        except Exception as e:
            print(f"Lỗi khi vẽ/gửi ảnh biểu đồ (bỏ qua, không ảnh hưởng tin nhắn chính): {e}")

    print("Đã gửi xong! Kiểm tra Telegram của bạn.")
