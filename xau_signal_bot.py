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
        # BẮT BUỘC: không có tham số này, Twelve Data trả timestamp theo múi giờ "Exchange"
        # (lệch UTC nhiều tiếng) -> tracking path-aware so timestamp nến với time_iso (UTC)
        # của lệnh bị lệch múi -> nến QUÁ KHỨ bị coi là "sau khi tạo lệnh" -> lệnh khớp ảo
        # và "thắng" bằng chuyển động giá đã xảy ra TRƯỚC khi lệnh tồn tại (lỗi thực tế
        # đã gặp: fade SELL 4051.33 thắng +$47.63 dù giá chưa hề chạm entry sau khi tạo).
        "timezone": "UTC",
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


H1_NATIVE_CACHE_PATH = "h1_native_cache.json"


def fetch_h1_native_cached(outputsize=300):
    """
    Lấy nến H1 GỐC từ API, cache theo GIỜ - chỉ gọi API khi đã sang giờ UTC mới (tức có
    nến H1 mới đóng), giữa các giờ dùng lại cache -> chi phí chỉ ~24 credits/ngày.

    Vì sao cần H1 gốc khi đã có resample từ M5? KHÔNG phải vì râu nến (về toán học,
    high H1 = max các high M5 trong giờ - gộp và gốc trùng nhau nếu feed không thủng),
    mà vì ĐỘ DÀI LỊCH SỬ: 1000 nến M5 chỉ gộp được ~83 nến H1, trong khi Ichimoku H1
    cần ~78 nến (Senkou B 52 + displacement 26) - đang chạy sát nút; Dow/Fib/FVG H1
    cũng chỉ nhìn được ~3.5 ngày. H1 gốc outputsize=300 -> nhìn 12+ ngày.

    Trả về DataFrame H1 (đã loại nến đang hình thành) hoặc None nếu lỗi - caller tự
    rơi về resample từ M5 như cũ, không bao giờ mất dữ liệu.
    """
    now = datetime.now(timezone.utc)
    hour_key = now.strftime("%Y-%m-%dT%H")
    # Cache còn trong cùng giờ UTC -> dùng lại, không tốn credit
    if os.path.exists(H1_NATIVE_CACHE_PATH):
        try:
            with open(H1_NATIVE_CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("hour_key") == hour_key and cache.get("rows"):
                df = pd.DataFrame(cache["rows"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                return df
        except Exception:
            pass

    try:
        df = get_ohlc("1h", outputsize=outputsize)
        # Loại nến H1 đang hình thành (chưa đóng) - cùng lý do với resample_ohlc
        if len(df) > 0:
            last_end = df["datetime"].iloc[-1] + pd.Timedelta(hours=1)
            now_naive = now.replace(tzinfo=None)
            if last_end > pd.Timestamp(now_naive):
                df = df.iloc[:-1].reset_index(drop=True)
        rows = [{"datetime": d.isoformat(), "open": float(o), "high": float(h),
                 "low": float(l), "close": float(c)}
                for d, o, h, l, c in zip(df["datetime"], df["open"], df["high"], df["low"], df["close"])]
        with open(H1_NATIVE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"hour_key": hour_key, "rows": rows}, f, ensure_ascii=False)
        return df
    except Exception as e:
        print(f"Lỗi lấy H1 gốc (rơi về resample M5): {e}")
        return None


def resample_h1_to_h4(df_h1, anchor_hours=(0, 4, 8, 12, 16, 20)):
    """
    Gộp nến H1 GỐC thành nến H4 - KHÔNG gọi API riêng (0 credit), tái dùng df_h1 sẵn có.
    Vì sao không dùng interval "4h" của Twelve Data: mốc neo "4h" của họ theo múi giờ
    riêng, dễ lệch với H4 quen nhìn trên TradingView. Ở đây tự neo mốc 00/04/08/12/16/20
    UTC -> nhất quán tuyệt đối với H1.

    XỬ LÝ NẾN CHƯA ĐÓNG (điểm mấu chốt): nến H4 chỉ HỢP LỆ khi có ĐỦ 4 nến H1 con đã đóng
    trong khối 4 giờ đó. Nến H4 đang hình thành (mới 1-3 nến H1) bị LOẠI hoàn toàn - biến
    câu hỏi khó "nến H4 đã đóng chưa" thành phép đếm đơn giản "đủ 4 nến H1 chưa", không
    phụ thuộc múi giờ API. Nến H4 dở dang mà lọt vào sẽ làm box H4 nhấp nháy và xác nhận
    breakout giả (close còn đổi tới 3.5h nữa).
    """
    if df_h1 is None or len(df_h1) < 4:
        return None
    df = df_h1.copy()
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    # Gán mỗi nến H1 vào khối H4 theo giờ UTC (floor về mốc neo gần nhất phía dưới)
    hours = df["datetime"].dt.hour
    block_hour = (hours // 4) * 4  # 0-3->0, 4-7->4, ... khớp anchor_hours
    block_start = df["datetime"].dt.normalize() + pd.to_timedelta(block_hour, unit="h")
    df = df.assign(_block=block_start)

    rows = []
    for block, g in df.groupby("_block"):
        g = g.sort_values("datetime")
        # Chỉ nhận khối ĐỦ 4 nến H1 con - loại nến H4 đang hình thành
        if len(g) < 4:
            continue
        rows.append({"datetime": block, "open": float(g["open"].iloc[0]),
                     "high": float(g["high"].max()), "low": float(g["low"].min()),
                     "close": float(g["close"].iloc[-1])})
    if len(rows) < 10:
        return None
    out = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    return out


def resample_ohlc(df_m5, rule, drop_incomplete=True):
    """
    Gộp nến M5 thành khung lớn hơn (15min/30min/1h) NGAY TRONG MÁY,
    không cần gọi thêm API -> tiết kiệm request, cho phép chạy nhanh hơn.
    rule: '15min', '30min', '1h'

    drop_incomplete=True: LOẠI BỎ nến cuối nếu nó CHƯA ĐÓNG (khung thời gian của nó chưa
    trôi qua hết so với nến M5 mới nhất). Trước đây nến H1 mới chạy được 10 phút vẫn được
    dùng để "xác nhận nến đóng cửa ngoài box" -> xác nhận GIẢ (nến đó có thể rút chân quay
    về trong box sau đó), tín hiệu nhấp nháy giữa các lần chạy 5 phút.
    """
    df = df_m5.set_index("datetime")
    out = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()
    out = out.reset_index()
    if drop_incomplete and len(out) > 0:
        # Nến resample cuối được coi là ĐÃ ĐÓNG khi: thời điểm bắt đầu + độ dài khung
        # <= thời điểm kết thúc của nến M5 mới nhất (= datetime nến M5 cuối + 5 phút)
        last_m5_end = df_m5["datetime"].iloc[-1] + pd.Timedelta(minutes=5)
        bar_end = out["datetime"].iloc[-1] + pd.Timedelta(rule)
        if bar_end > last_m5_end:
            out = out.iloc[:-1].reset_index(drop=True)
    return out


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
BOX_RANGE_ATR_MULT_H4 = 1.8 # H4 dùng ngưỡng thấp hơn: ít nến hơn nhiều (75 nến H4 vs 300 H1)
                            # nên nến thanh khoản H4 hiếm hơn - 1.8x để có đủ mẫu mà đo, sau
                            # vài tuần nhìn phân bố range/ATR thật rồi tinh chỉnh bằng dữ liệu
BOX_RETEST_TOLERANCE_ATR = 0.3  # giá được coi là "đã quay về test" nếu cách điểm entry <= 0.3x ATR
# (SL cố định 20/10 giá cũ đã bỏ - SL giờ tính theo CẤU TRÚC box, xem structural_sl bên dưới)
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
                    retest_tolerance_atr=BOX_RETEST_TOLERANCE_ATR, bound_range=None,
                    live_price=None):
    """
    Trả về trạng thái box gần nhất:
    - None: không tìm thấy nến thanh khoản nào phù hợp trong phạm vi quét
    - state="unconfirmed": box vừa hình thành, CHƯA có nến đóng cửa phá vỡ hẳn ra ngoài
      -> chỉ có "entry rủi ro" (SL 10 giá) tại 2 cạnh, chưa rõ hướng
    - state="waiting_retest": đã xác nhận breakout nhưng giá CHƯA quay lại test biên
    - state="ready": đã xác nhận VÀ giá đã quay lại test - sẵn sàng entry (SL 20 giá)
    Truyền bound_range=(low, high) để CHỈ tìm box nằm LỌT trong 1 box khung lớn hơn đã chọn
    trước - dùng cho cấu trúc "box M15 nằm trong box H1".
    live_price: giá M5 MỚI NHẤT - dùng cho kiểm tra vô hiệu hóa/spring/retest. Nến khung
    lớn (H1) đã đóng gần nhất có thể cũ tới 1 giờ; đã có trường hợp giá thật vượt hẳn cạnh
    trên box SELL mà box vẫn hiện "sẵn sàng entry" vì bước vô hiệu hóa nhìn giá đóng H1 cũ.
    """
    boxes = find_recent_liquidity_boxes(df, atr_series, lookback=lookback, range_mult=range_mult,
                                         bound_range=bound_range)
    if not boxes:
        return None

    merged = merge_boxes_if_overlap(boxes)
    box_high, box_low, color = merged["mid_high"], merged["mid_low"], merged["color"]
    box_mid = (box_high + box_low) / 2
    current_price = live_price if live_price is not None else df.iloc[-1]["close"]
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
    confirm_idx = None
    for j in range(len(after)):
        c = after.iloc[j]
        if c["close"] > merged["outer_high"]:
            confirm_dir = "up"; confirm_idx = j; break
        if c["close"] < merged["outer_low"]:
            confirm_dir = "down"; confirm_idx = j; break

    if confirm_dir is None:
        return {"box_high": box_high, "box_low": box_low, "box_mid": box_mid,
                "color": color, "state": "unconfirmed", "in_middle": in_middle}

    # ====== PHÁ VỠ GIẢ (SPRING / STOP-HUNT REVERSAL) ======
    # Rất nhiều cú phá biên range trên vàng thực chất là QUÉT STOP LOSS rồi đảo chiều
    # (Wyckoff: Spring). Nhận diện: sau nến xác nhận breakout, giá ĐÓNG CỬA quay lại
    # TRONG box trong vòng SPRING_WINDOW nến -> cú phá là GIẢ, tín hiệu đúng là lệnh
    # NGƯỢC hướng phá: BUY tại biên dưới vừa bị quét (phá xuống giả) / SELL tại biên
    # trên (phá lên giả). Khác "bắt dao rơi": vào khi phe phá vỡ vừa bị chứng minh là
    # bẫy, vùng đã cạn stop loss - SL đặt dưới/trên ĐÁY/ĐỈNH CÚ QUÉT (điểm vô hiệu rõ).
    post = after.iloc[confirm_idx + 1:]
    spring = None
    if len(post) > 0:
        reentry_idx = None
        for k in range(min(SPRING_WINDOW, len(post))):
            c = post.iloc[k]
            if confirm_dir == "down" and c["close"] > box_low:
                reentry_idx = k; break
            if confirm_dir == "up" and c["close"] < box_high:
                reentry_idx = k; break
        if reentry_idx is not None:
            after_reentry = post.iloc[reentry_idx + 1:]
            sweep_slice = after.iloc[confirm_idx: confirm_idx + 1 + reentry_idx + 1]
            if confirm_dir == "down":
                # Spring còn sống khi giá KHÔNG đóng cửa thủng đáy quét lần nữa
                dead = (after_reentry["close"] < merged["outer_low"]).any() if len(after_reentry) else False
                if not dead and current_price > box_low:
                    spring = {"direction": "BUY", "boundary": box_low,
                              "sweep_extreme": float(sweep_slice["low"].min())}
            else:
                dead = (after_reentry["close"] > merged["outer_high"]).any() if len(after_reentry) else False
                if not dead and current_price < box_high:
                    spring = {"direction": "SELL", "boundary": box_high,
                              "sweep_extreme": float(sweep_slice["high"].max())}

    if spring:
        tol = retest_tolerance_atr * atr_now if atr_now > 0 else 0
        # Entry tại biên vừa bị quét; sẵn sàng khi giá đã/đang quay về chạm biên đó
        near = abs(current_price - spring["boundary"]) <= tol if tol > 0 else False
        touched = False
        if tol > 0 and reentry_idx is not None and len(post) > reentry_idx + 1:
            pr = post.iloc[reentry_idx + 1:]
            touched = ((pr["low"] - tol <= spring["boundary"]) & (spring["boundary"] <= pr["high"] + tol)).any()
        return {
            "box_high": box_high, "box_low": box_low, "box_mid": box_mid, "color": color,
            "state": "spring_ready" if (near or touched) else "spring_waiting",
            "direction": spring["direction"], "alignment": "spring",
            "confirm_dir": confirm_dir, "sweep_extreme": spring["sweep_extreme"],
            "entries": [{"label": "spring", "price": spring["boundary"]}],
        }
    # ====== HẾT PHẦN SPRING - dưới đây là luồng breakout thật như cũ ======

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

    # RETEST theo ĐƯỜNG ĐI của giá: xét high/low của TẤT CẢ nến SAU nến xác nhận breakout -
    # nếu bất kỳ nến nào đã chạm về vùng entry (trong dung sai) thì coi là ĐÃ retest, kể cả
    # khi cú chạm xảy ra giữa 2 lần bot chạy (trước đây chỉ so giá tại đúng thời điểm chạy
    # -> bỏ lỡ hầu hết cú retest diễn ra trong chu kỳ 5 phút).
    tol = retest_tolerance_atr * atr_now if atr_now > 0 else 0
    retested = False
    if tol > 0 and confirm_idx is not None:
        post_confirm = after.iloc[confirm_idx + 1:]
        for e in entries:
            level = e["price"]
            touched = ((post_confirm["low"] - tol <= level) & (level <= post_confirm["high"] + tol)).any()
            if touched or abs(current_price - level) <= tol:
                retested = True
                break

    return {
        "box_high": box_high, "box_low": box_low, "box_mid": box_mid, "color": color,
        "state": "ready" if retested else "waiting_retest",
        "direction": direction, "alignment": alignment, "entries": entries,
        "confirm_dir": confirm_dir,
    }


# ---- HỆ TP/SL MỚI: TP tính theo bội số R (R = khoảng cách SL), SL neo vào CẤU TRÚC box ----
# Thay cho TP cố định 5-10 giá + SL cố định 10/20 giá cũ (R:R nghịch 2:1 đến 4:1, kỳ vọng
# âm cài sẵn). Chuẩn mới: TP1 = 1R tối thiểu, TP2 = 1.8R, TP3 = 2.8R - thắng 1 lệnh đủ bù
# 1 lệnh thua ngay từ TP1.
BOX_RR_MULTIPLES = (1.0, 1.8, 2.8)   # TP1/TP2/TP3 theo bội số R
BOX_SL_BUFFER_ATR = 0.3              # đệm SL ngoài mức cấu trúc = 0.3x ATR khung của box
BOX_SL_MIN_POINTS = 5.0              # SL cấu trúc quá sát (nhiễu quét dễ) -> nới ra tối thiểu 5 giá
BOX_SL_MAX_POINTS = 25.0             # SL cấu trúc quá rộng -> BỎ QUA entry đó (không ép SL, không vào lệnh xấu)
BOX_SL_RISK_ATR = 0.75               # entry rủi ro (box chưa xác nhận): SL = 0.75x ATR, kẹp [5, 12]
BOX_SL_RISK_MIN = 5.0
BOX_SL_RISK_MAX = 12.0
BOX_SL_MIN_ATR = 1.0                 # sàn SL động = max(5 giá, 1.0x ATR khung của box) - SL 5 giá
                                     # cứng trên box H1 quá sát, 1 cú quét râu H1 bình thường đủ đá văng
CHASE_WARNING_ATR = 2.5              # giá cách entry >= 2.5x ATR -> cảnh báo "chờ giá về, không đuổi"
FAR_ENTRY_ATR = 4.0                  # entry gần nhất cách giá > 4x ATR -> KHÔNG phát lệnh (gần như
                                     # chắc chắn hết hạn vô nghĩa, chỉ gây nhiễu) - chỉ hiển thị box
SPRING_WINDOW = 5                    # phá vỡ giả: giá phải đóng cửa QUAY LẠI trong box trong vòng
                                     # N nến (khung của box) sau nến xác nhận breakout
REQUIRE_REJECTION_CANDLE = False     # True = chỉ phát lệnh khi có NẾN TỪ CHỐI M15 tại vùng entry.
                                     # Mặc định TẮT: thay đổi tính chất entry, chỉ nên bật sau khi
                                     # đã có vài tuần dữ liệu tracking sạch để so sánh trước/sau.


def compute_box_tp(direction, entry, sl, multiples=BOX_RR_MULTIPLES):
    """
    TP theo bội số R: R = |entry - SL|. TP1 = entry + 1R, TP2 = +1.8R, TP3 = +2.8R (BUY;
    SELL ngược lại). SL đã được tính theo cấu trúc box trước đó nên TP tự thích ứng theo
    độ rộng cấu trúc - box lớn SL rộng thì TP cũng xa tương ứng, giữ R:R luôn >= 1:1.
    """
    r = abs(entry - sl)
    if direction == "BUY":
        return tuple(round(entry + r * m, 2) for m in multiples)
    return tuple(round(entry - r * m, 2) for m in multiples)


def structural_sl(direction, entry_label, box_low, box_mid, box_high, atr_tf):
    """
    SL neo vào mức CẤU TRÚC gần nhất phía dưới entry (BUY) / phía trên entry (SELL),
    cộng đệm 0.3x ATR - thay cho số cố định 20 giá cũ:
    - BUY tại cạnh trên  -> SL dưới ĐƯỜNG GIỮA box (mid)
    - BUY tại giữa biên  -> SL dưới CẠNH DƯỚI box
    - BUY tại cạnh dưới  -> SL dưới CẠNH DƯỚI box
    (SELL đối xứng ngược lại)
    Trả về (giá SL, khoảng cách SL) - hoặc (None, None) nếu khoảng cách vượt trần
    BOX_SL_MAX_POINTS: cấu trúc quá rộng, entry này không đáng vào, bỏ qua thay vì ép SL.
    """
    buffer = BOX_SL_BUFFER_ATR * atr_tf if atr_tf and atr_tf > 0 else BOX_SL_MIN_POINTS * 0.2
    # Sàn SL ĐỘNG theo khung: max(5 giá, 1.0x ATR khung của box). Box H1 (ATR ~5-8 giá)
    # sẽ có SL tối thiểu đủ "thở"; box M15 vẫn giữ SL nhỏ gọn. SL 5 giá cứng trên cấu trúc
    # H1 quá sát - 1 cú quét râu H1 bình thường đủ đá văng trước khi giá đi đúng hướng.
    sl_floor = max(BOX_SL_MIN_POINTS, BOX_SL_MIN_ATR * atr_tf) if atr_tf and atr_tf > 0 \
        else BOX_SL_MIN_POINTS
    if direction == "BUY":
        anchor = box_mid if entry_label == "high" else box_low
        sl = anchor - buffer
        entry_price = {"low": box_low, "mid": box_mid, "high": box_high}[entry_label]
        dist = entry_price - sl
    else:
        anchor = box_mid if entry_label == "low" else box_high
        sl = anchor + buffer
        entry_price = {"low": box_low, "mid": box_mid, "high": box_high}[entry_label]
        dist = sl - entry_price

    if dist < sl_floor:
        dist = sl_floor
        sl = entry_price - dist if direction == "BUY" else entry_price + dist
    if dist > BOX_SL_MAX_POINTS:
        return None, None
    return round(sl, 2), round(dist, 2)


def build_box_signal(box_m15, box_h1, atr_m5, current_price=None, atr_by_tf=None, box_h4=None):
    """
    Chọn box CHÍNH để giao dịch - ưu tiên box đã 'ready' (sẵn sàng entry), khung H1 trước
    (cấu trúc lớn hơn, đáng tin hơn), rồi tới M15. Nếu không box nào 'ready', vẫn chọn 1 box
    để hiển thị bối cảnh (ready > waiting_retest > unconfirmed), ưu tiên H1.
    SL từng entry tính theo CẤU TRÚC box (structural_sl) - entry nào SL cấu trúc quá rộng
    (> BOX_SL_MAX_POINTS) bị LOẠI thay vì ép SL. TP theo bội số R. Mỗi entry kèm loại lệnh
    chờ (limit/stop) so với giá hiện tại - dùng cho logic khớp lệnh path-aware sau này.
    """
    def _priority(b):
        if not b:
            return (-1, 0)
        # Ưu tiên 2 tầng: (độ CHÍN của setup, độ LỚN của khung).
        # Độ chín thắng trước - H4 mới waiting_retest KHÔNG được nuốt H1 đã ready (bỏ lỡ
        # cơ hội thật). Cùng độ chín thì khung lớn hơn thắng (H4 > H1 > M15: cấu trúc lớn
        # đáng tin hơn khi xung đột hướng).
        maturity = {"spring_ready": 4, "ready": 3, "spring_waiting": 2, "waiting_retest": 2,
                    "unconfirmed": 1}.get(b["state"], 0)
        tf_rank = {"H4": 3, "H1": 2, "M15": 1}.get(b.get("_tf_tag"), 0)
        return (maturity, tf_rank)

    atr_by_tf = atr_by_tf or {}
    # Gắn thẻ khung vào từng box để _priority so được độ lớn khung
    for tf_tag, b in (("H4", box_h4), ("H1", box_h1), ("M15", box_m15)):
        if b:
            b["_tf_tag"] = tf_tag
    candidates = [("H4", box_h4), ("H1", box_h1), ("M15", box_m15)]
    candidates.sort(key=lambda x: _priority(x[1]), reverse=True)
    tf_name, box = candidates[0]
    if not box:
        return None

    atr_tf = atr_by_tf.get(tf_name, atr_m5)
    result = {"tf": tf_name, "box_high": box["box_high"], "box_low": box["box_low"],
              "box_mid": box["box_mid"], "color": box["color"], "state": box["state"],
              "entries": []}

    def _order_kind(direction, entry_price):
        """Limit = chờ giá QUAY VỀ entry; Stop = chờ giá VƯỢT QUA entry theo hướng lệnh."""
        if current_price is None:
            return "limit"
        if direction == "BUY":
            return "limit" if entry_price <= current_price else "stop"
        return "limit" if entry_price >= current_price else "stop"

    if box["state"] == "unconfirmed":
        result["in_middle"] = box.get("in_middle", False)
        # THAY ĐỔI: giá ở GIỮA box giờ KHÔNG trả về rỗng nữa - đây chính là lúc tốt nhất
        # để lập KẾ HOẠCH 2 KỊCH BẢN (fade 2 cạnh, đặt lệnh chờ trước khi giá chạm vùng
        # tranh chấp). Entry vẫn được tạo như unconfirmed thường; khác biệt chỉ ở cách
        # hiển thị (khối kế hoạch) và việc log mode="fade" để thống kê riêng.
        # Entry RỦI RO: fade 2 cạnh (chưa rõ hướng breakout) - SL theo ATR khung của box,
        # kẹp trong [BOX_SL_RISK_MIN, BOX_SL_RISK_MAX] thay cho số cố định 10 giá cũ.
        risk_dist = min(max(BOX_SL_RISK_ATR * atr_tf, BOX_SL_RISK_MIN), BOX_SL_RISK_MAX) \
            if atr_tf and atr_tf > 0 else BOX_SL_RISK_MIN
        for label, price, direction in (("cạnh trên", box["box_high"], "SELL"),
                                         ("cạnh dưới", box["box_low"], "BUY")):
            sl = round(price + risk_dist, 2) if direction == "SELL" else round(price - risk_dist, 2)
            tp1, tp2, tp3 = compute_box_tp(direction, price, sl)
            result["entries"].append({
                "label": label, "direction": direction, "entry": price, "sl": sl,
                "sl_points": round(risk_dist, 2), "tp1": tp1, "tp2": tp2, "tp3": tp3,
                "risk": True, "order_kind": _order_kind(direction, price),
            })
        return result

    result["direction"] = box["direction"]
    result["alignment"] = box["alignment"]
    result["confirm_dir"] = box["confirm_dir"]
    result["rejected_entries"] = []  # minh bạch: entry bị loại + lý do, hiển thị trong tin nhắn

    # ---- SETUP SPRING (phá vỡ giả): 1 entry duy nhất tại biên vừa bị quét ----
    if box["state"] in ("spring_ready", "spring_waiting"):
        direction = box["direction"]
        boundary = box["entries"][0]["price"]
        sweep = box["sweep_extreme"]
        result["sweep_extreme"] = sweep
        buffer = BOX_SL_BUFFER_ATR * atr_tf if atr_tf and atr_tf > 0 else 1.0
        sl_floor = max(BOX_SL_MIN_POINTS, BOX_SL_MIN_ATR * atr_tf) if atr_tf and atr_tf > 0 \
            else BOX_SL_MIN_POINTS
        # SL dưới đáy cú quét (BUY) / trên đỉnh cú quét (SELL) + đệm - điểm vô hiệu hóa
        # rõ ràng nhất có thể: đáy/đỉnh quét thủng nghĩa là "phá giả" hóa ra phá thật.
        sl = round(sweep - buffer, 2) if direction == "BUY" else round(sweep + buffer, 2)
        dist = (boundary - sl) if direction == "BUY" else (sl - boundary)
        if dist < sl_floor:
            dist = sl_floor
            sl = round(boundary - dist, 2) if direction == "BUY" else round(boundary + dist, 2)
        if dist > BOX_SL_MAX_POINTS:
            result["rejected_entries"].append(
                {"label": "spring", "reason": f"SL cấu trúc {dist:.1f}p vượt trần {BOX_SL_MAX_POINTS:.0f}p"})
            return None
        tp1, tp2, tp3 = compute_box_tp(direction, boundary, sl)
        result["entries"].append({
            "label": "biên phá giả", "direction": direction, "entry": boundary, "sl": sl,
            "sl_points": round(dist, 2), "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "risk": False, "order_kind": _order_kind(direction, boundary), "spring": True,
        })
        return result

    label_map = {"low": "cạnh dưới", "mid": "giữa biên", "high": "cạnh trên"}
    for e in box["entries"]:
        direction = box["direction"]
        sl, sl_dist = structural_sl(direction, e["label"], box["box_low"], box["box_mid"],
                                     box["box_high"], atr_tf)
        if sl is None:
            # Minh bạch hóa: ghi lý do loại thay vì biến mất trong im lặng - trước đây
            # entry "giữa biên" của box cao > 25 giá biến mất không dấu vết, gây khó hiểu
            result["rejected_entries"].append(
                {"label": label_map[e["label"]],
                 "reason": f"SL cấu trúc vượt trần {BOX_SL_MAX_POINTS:.0f}p"})
            continue
        tp1, tp2, tp3 = compute_box_tp(direction, e["price"], sl)
        result["entries"].append({
            "label": label_map[e["label"]], "direction": direction, "entry": e["price"],
            "sl": sl, "sl_points": sl_dist, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "risk": False, "order_kind": _order_kind(direction, e["price"]),
        })
    if not result["entries"]:
        return None  # mọi entry đều bị loại vì SL cấu trúc quá rộng -> không có gì để giao dịch
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
        # Tái dùng cache H1 GỐC theo giờ (fetch_h1_native_cached) thay vì gọi API riêng -
        # tiết kiệm 1 credit/lần cập nhật; chỉ còn H4 phải gọi thật (~24 credits/ngày)
        df_h1 = fetch_h1_native_cached()
        if df_h1 is None or len(df_h1) < 100:
            df_h1 = get_ohlc("1h", outputsize=170)   # ~1 tuần (fallback khi cache H1 lỗi)
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


# ============================================================
# 2c. SỔ ĐĂNG KÝ MỨC ĐỔI VAI (POLARITY FLIP REGISTRY)
# ============================================================
# Nguyên lý đổi vai: cạnh box bị phá vỡ KHÔNG mất giá trị - kháng cự vỡ thành hỗ trợ,
# hỗ trợ vỡ thành kháng cự. Bot trước đây chỉ nhớ 1 box gần nhất, box cũ bị quên sạch
# trong khi thị trường vẫn nhớ các mức đó nhiều ngày sau. Sổ này lưu các mức đổi vai
# qua các lần chạy (file json, như signal_log) và dùng vào 3 việc:
#   1. TƯỜNG CHẮN TP: TP2/TP3 nằm sau mức đổi vai ngược hướng -> co về trước mức đó
#   2. CỘNG HƯỞNG ENTRY: entry box mới trùng mức đổi vai cũ -> ghi chú xác nhận kép
#   3. VÙNG THEO DÕI: mức đổi vai hiện trong khối zones với nhãn riêng
# QUAN TRỌNG: mức đổi vai KHÔNG tự phát lệnh - Box Detector vẫn là nơi duy nhất ra lệnh.
FLIP_LEVELS_PATH = "flipped_levels.json"
FLIP_MAX_AGE_DAYS = 14        # tuổi thọ tối đa nếu giá chưa chạm (theo yêu cầu: 1-2 tuần)
FLIP_MAX_TOUCHES = 3          # bị test đến lần thứ 3 -> mức đã yếu, loại (lần test đầu mạnh nhất)
FLIP_TOUCH_TOL_ATR = 0.3      # dung sai vùng chạm quanh mức = 0.3x ATR M5
FLIP_TP_CLEARANCE_ATR = 0.3   # co TP về TRƯỚC mức đổi vai 0.3x ATR (không đặt TP ngay trên tường)
FLIP_CONFLUENCE_ATR = 0.5     # entry cách mức đổi vai <= 0.5x ATR -> ghi nhận cộng hưởng


def load_flipped_levels():
    if not os.path.exists(FLIP_LEVELS_PATH):
        return []
    try:
        with open(FLIP_LEVELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_flipped_levels(levels):
    with open(FLIP_LEVELS_PATH, "w", encoding="utf-8") as f:
        json.dump(levels, f, ensure_ascii=False, indent=2)


def register_flipped_level(levels, box_signal):
    """
    Box xác nhận breakout (không phải spring) -> ghi cạnh vừa vỡ vào sổ:
    - Phá LÊN qua cạnh trên  -> cạnh trên thành SUPPORT (kháng cự -> hỗ trợ)
    - Phá XUỐNG qua cạnh dưới -> cạnh dưới thành RESISTANCE (hỗ trợ -> kháng cự)
    Spring (phá giả) -> XÓA mức tương ứng nếu đã lỡ ghi ở lần chạy trước: cú phá
    hóa ra thất bại, cạnh đó KHÔNG đổi vai, vẫn giữ vai trò cũ.
    Chống trùng bằng khóa (mức làm tròn 1 số lẻ, vai trò).
    """
    if not box_signal or not box_signal.get("confirm_dir"):
        return levels
    confirm = box_signal["confirm_dir"]
    if confirm == "up":
        level, role = box_signal["box_high"], "support"
    else:
        level, role = box_signal["box_low"], "resistance"

    key = (round(level, 1), role)
    if box_signal.get("alignment") == "spring" or str(box_signal.get("state", "")).startswith("spring"):
        # Phá giả -> cạnh không đổi vai; gỡ bản ghi nếu lần chạy trước (lúc còn tưởng
        # là breakout thật) đã lỡ đăng ký
        return [lv for lv in levels
                if (round(lv["level"], 1), lv["role"]) != key]

    if any((round(lv["level"], 1), lv["role"]) == key for lv in levels):
        return levels
    levels.append({
        "level": round(level, 2), "role": role, "tf": box_signal["tf"],
        "created_iso": datetime.now(timezone.utc).isoformat(),
        "last_checked_iso": datetime.now(timezone.utc).isoformat(),
        "touches": 0,
    })
    return levels


def maintain_flipped_levels(levels, df_m5, atr_m5):
    """
    Bảo trì sổ mỗi lần chạy - 3 luật loại bỏ:
    1. HẾT HẠN: quá FLIP_MAX_AGE_DAYS (14 ngày) kể từ khi tạo
    2. BỊ XUYÊN THỦNG: giá ĐÓNG CỬA vượt qua mức ngược vai trò (support mà đóng cửa
       dưới hẳn / resistance mà đóng cửa trên hẳn) -> mức không còn ý nghĩa
    3. TEST QUÁ NHIỀU: chạm đến lần thứ FLIP_MAX_TOUCHES -> mức đã cạn lệnh chờ, yếu dần
    Đếm chạm theo "sự kiện": nến đầu chu kỳ phải NGOÀI vùng rồi có nến chạm vùng mới
    tính 1 lần chạm (tránh đếm trùng khi giá lượn quanh mức qua nhiều lần chạy).
    """
    now = datetime.now(timezone.utc)
    tol = FLIP_TOUCH_TOL_ATR * atr_m5 if atr_m5 and atr_m5 > 0 else 1.0
    kept = []
    for lv in levels:
        try:
            created = datetime.fromisoformat(lv["created_iso"])
        except Exception:
            continue
        if (now - created).days >= FLIP_MAX_AGE_DAYS:
            continue  # luật 1: hết hạn

        try:
            last_checked = datetime.fromisoformat(lv.get("last_checked_iso", lv["created_iso"]))
        except Exception:
            last_checked = created
        candles = _candles_since(df_m5, last_checked)
        level = lv["level"]

        if len(candles) > 0:
            # luật 2: đóng cửa xuyên thủng (vượt qua mức + dung sai)
            if lv["role"] == "support":
                broken = (candles["close"] < level - tol).any()
            else:
                broken = (candles["close"] > level + tol).any()
            if broken:
                continue

            # luật 3: đếm sự kiện chạm
            first = candles.iloc[0]
            first_outside = not ((first["low"] - tol) <= level <= (first["high"] + tol))
            touched = ((candles["low"] - tol <= level) & (level <= candles["high"] + tol)).any()
            if first_outside and touched:
                lv["touches"] = lv.get("touches", 0) + 1
                lv["last_touch_iso"] = now.isoformat()
            if lv.get("touches", 0) >= FLIP_MAX_TOUCHES:
                continue

        lv["last_checked_iso"] = now.isoformat()
        kept.append(lv)
    return kept


def apply_flip_walls(entry_dict, levels, atr_m5):
    """
    TƯỜNG CHẮN TP: TP2/TP3 nằm SAU một mức đổi vai ngược hướng lệnh -> co về trước mức
    đó (mức - đệm). TP1 GIỮ NGUYÊN = 1R để bảo toàn hạch toán R:R và thống kê thắng/thua;
    nếu có tường trước cả TP1 thì chỉ CẢNH BÁO (bạn tự quyết), không tự đổi.
    Sửa entry_dict tại chỗ, trả về danh sách ghi chú.
    """
    if not levels or not atr_m5 or atr_m5 <= 0:
        return []
    direction = entry_dict["direction"]
    entry = entry_dict["entry"]
    clearance = FLIP_TP_CLEARANCE_ATR * atr_m5
    # Tường liên quan: SELL bị chặn bởi SUPPORT nằm dưới entry; BUY bởi RESISTANCE trên entry
    if direction == "SELL":
        walls = sorted([lv["level"] for lv in levels
                        if lv["role"] == "support" and lv["level"] < entry], reverse=True)
    else:
        walls = sorted([lv["level"] for lv in levels
                        if lv["role"] == "resistance" and lv["level"] > entry])
    if not walls:
        return []

    notes = []
    tp1 = entry_dict["tp1"]
    # Cảnh báo nếu tường chắn trước TP1 (không tự đổi TP1)
    first_wall = walls[0]
    tp1_blocked = (direction == "SELL" and tp1 < first_wall) or                   (direction == "BUY" and tp1 > first_wall)
    if tp1_blocked:
        notes.append(f"⚠️ Mức đổi vai {first_wall:.2f} chắn TRƯỚC TP1 - cản mạnh, cân nhắc chốt sớm")

    for tp_key in ("tp2", "tp3"):
        tp = entry_dict[tp_key]
        for wall in walls:
            beyond = (direction == "SELL" and tp < wall) or (direction == "BUY" and tp > wall)
            if beyond:
                new_tp = round(wall + clearance, 2) if direction == "SELL" else round(wall - clearance, 2)
                # TP sau khi co vẫn phải xa hơn TP1 (giữ thứ tự TP1<TP2<TP3 theo hướng lệnh)
                still_valid = (direction == "SELL" and new_tp < tp1) or                               (direction == "BUY" and new_tp > tp1)
                if still_valid:
                    entry_dict[tp_key] = new_tp
                    notes.append(f"🧱 {tp_key.upper()} co về {new_tp:.2f} - trước mức đổi vai {wall:.2f}")
                break  # chỉ xét tường GẦN NHẤT chắn TP này
    return notes


def flip_confluence_note(entry_dict, levels, atr_m5):
    """
    CỘNG HƯỞNG: entry trùng vùng (0.5x ATR) với mức đổi vai CÙNG vai trò với hướng lệnh
    (BUY tại support đổi vai / SELL tại resistance đổi vai) -> 2 cấu trúc độc lập cùng
    chỉ 1 mức, bằng chứng mạnh hơn hẳn. Chỉ ghi chú, không cộng điểm (giữ thang điểm
    bối cảnh ổn định cho việc so sánh thống kê đang chạy).
    """
    if not levels or not atr_m5 or atr_m5 <= 0:
        return None
    want_role = "support" if entry_dict["direction"] == "BUY" else "resistance"
    tol = FLIP_CONFLUENCE_ATR * atr_m5
    for lv in levels:
        if lv["role"] == want_role and abs(lv["level"] - entry_dict["entry"]) <= tol:
            role_txt = "kháng cự→hỗ trợ" if want_role == "support" else "hỗ trợ→kháng cự"
            return (f"⭐ Entry trùng mức đổi vai {lv['level']:.2f} ({role_txt}, "
                    f"box {lv['tf']} cũ) - xác nhận kép")
    return None


def _candles_since(df_m5, start_time_utc, exclusive=False):
    """
    Lấy các nến có thời điểm >= start_time_utc (hoặc > nếu exclusive=True).

    exclusive=True dùng cho lệnh ĐÃ KHỚP ở lượt chạy TRƯỚC: loại chính nến khớp ra khỏi
    cửa sổ đánh giá. Nến khớp đã được chấm đúng (chỉ phần giá SAU điểm khớp) ngay tại
    lượt nó khớp; nếu lượt sau đưa nó vào lại thì nó được tính với TOÀN BỘ biên độ ->
    phần giá TRƯỚC điểm khớp bị tính thành chạm TP/SL -> thắng/thua ảo.

    LƯỚI AN TOÀN MÚI GIỜ (đã siết): chỉ neo lại timestamp khi độ lệch giống LỆCH MÚI GIỜ
    THẬT - tức xấp xỉ bội số GIỜ TRÒN (múi giờ luôn lệch nguyên giờ). Trước đây mọi độ
    lệch > 30 phút đều bị dịch, nên dữ liệu chỉ CŨ (gap thanh khoản đêm, thị trường mỏng)
    cũng bị đẩy timestamp -> nến quá khứ biến thành "sau khi tạo lệnh" -> khớp ảo.
    Ba nhánh xử lý:
      - Lệch xấp xỉ giờ tròn  -> neo lại (lệch múi giờ thật)
      - Lệch dương, không tròn -> dữ liệu chỉ CŨ, timestamp vốn đúng -> GIỮ NGUYÊN
      - Lệch âm, không tròn    -> timestamp ở TƯƠNG LAI mà không rõ nguyên nhân -> trả
        RỖNG (bỏ qua theo dõi lượt này) thay vì đoán; lượt sau dữ liệu tốt sẽ bù lại
    """
    dt = df_m5["datetime"]
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC")
    now = datetime.now(timezone.utc)
    skew = now - dt.iloc[-1]
    skew_sec = skew.total_seconds()

    if abs(skew_sec) >= 1800:
        hours_off = round(skew_sec / 3600.0)
        residual = abs(skew_sec - hours_off * 3600.0)
        if abs(hours_off) >= 1 and residual <= 360:
            print(f"⚠️ Dữ liệu lệch múi giờ ~{hours_off}h - tự neo lại theo giờ UTC thật")
            dt = dt + skew
        elif skew_sec < 0:
            print(f"⚠️ Timestamp nến ở TƯƠNG LAI {abs(skew_sec)/60:.0f} phút không rõ nguyên "
                  f"nhân -> bỏ qua theo dõi lượt này (an toàn hơn đoán)")
            empty = df_m5.iloc[0:0].copy()
            empty["datetime_utc"] = dt.iloc[0:0]
            return empty
        # else: lệch dương không tròn giờ = dữ liệu chỉ CŨ -> giữ nguyên timestamp (đúng)

    mask = (dt > start_time_utc) if exclusive else (dt >= start_time_utc)
    out = df_m5.loc[mask].copy()
    out["datetime_utc"] = dt[mask]
    return out.reset_index(drop=True)


def update_signal_outcomes(log, df_m5, current_price):
    """
    Kiểm tra các tín hiệu cũ theo ĐƯỜNG ĐI GIÁ THẬT (high/low từng nến M5 kể từ lúc tạo
    tín hiệu) thay vì chỉ 1 điểm giá tại thời điểm bot chạy như trước - sửa 2 lỗi lớn:
    1. TP/SL bị chạm GIỮA 2 lần chạy (chu kỳ 5 phút) trước đây bị bỏ lỡ hoàn toàn.
    2. Không có path-dependency: nếu giá chạm SL trước rồi mới lên TP, bản cũ có thể ghi
       nhầm thành thắng. Bản này duyệt nến theo thứ tự thời gian - cái nào chạm TRƯỚC tính trước.

    Quy tắc thận trọng: nếu CÙNG 1 nến M5 quét cả SL lẫn TP (không phân định được cái nào
    trước) -> tính là THUA. Thà thống kê khắt khe hơn thực tế còn hơn tự lừa mình.

    Khớp lệnh chờ ('waiting_fill') cũng theo path và ĐÚNG loại lệnh:
    - BUY limit (entry DƯỚI giá lúc tạo): khớp khi low nến <= entry (giá giảm về)
    - BUY stop  (entry TRÊN giá lúc tạo): khớp khi high nến >= entry (giá vượt lên)
    - SELL đối xứng. Trước đây mọi lệnh đều coi là limit -> lệnh bản chất là stop bị
      "khớp ảo" ngay lập tức tại mức giá không có thật, làm méo toàn bộ thống kê.
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

        timeout = rec.get("timeout_hours", SIGNAL_TIMEOUT_HOURS)
        direction = rec["direction"]
        entry, sl, tp1 = rec["entry"], rec["sl"], rec["tp1"]

        # Bản ghi cũ (trước nâng cấp) không có order_kind/signal_price -> suy ra hợp lý nhất
        order_kind = rec.get("order_kind")
        if order_kind is None:
            sig_price = rec.get("signal_price")
            if sig_price is not None:
                if direction == "BUY":
                    order_kind = "limit" if entry <= sig_price else "stop"
                else:
                    order_kind = "limit" if entry >= sig_price else "stop"
            else:
                order_kind = "limit"

        # Mốc bắt đầu duyệt nến: lệnh đã khớp thì từ lúc khớp, chưa khớp thì từ lúc tạo.
        # Lệnh đã khớp ở lượt TRƯỚC (có fill_time_iso) -> dùng exclusive để LOẠI chính nến
        # khớp: nó đã được chấm đúng (chỉ phần giá sau điểm khớp) ngay tại lượt nó khớp.
        # Nếu đưa lại vào, nó được tính với TOÀN BỘ biên độ -> phần giá TRƯỚC điểm khớp
        # bị tính thành chạm TP/SL -> thắng/thua ảo ở lượt chạy kế tiếp.
        exclusive = False
        if status == "pending" and rec.get("fill_time_iso"):
            start_iso = rec["fill_time_iso"]
            exclusive = True
        else:
            start_iso = rec["time_iso"]
        try:
            start_time = datetime.fromisoformat(start_iso)
        except Exception:
            start_time = rec_time

        candles = _candles_since(df_m5, start_time, exclusive=exclusive)

        filled = (status == "pending")
        fill_this_candle = False
        for _, c in candles.iterrows():
            lo, hi = c["low"], c["high"]
            fill_this_candle = False

            if not filled:
                if direction == "BUY":
                    hit = (order_kind == "limit" and lo <= entry) or \
                          (order_kind == "stop" and hi >= entry)
                else:
                    hit = (order_kind == "limit" and hi >= entry) or \
                          (order_kind == "stop" and lo <= entry)
                if not hit:
                    continue
                filled = True
                fill_this_candle = True
                rec["status"] = "pending"
                rec["fill_time_iso"] = c["datetime_utc"].isoformat()

            # QUAN TRỌNG - NẾN KHỚP LỆNH: chỉ phần giá SAU ĐIỂM KHỚP mới được tính.
            # - Lệnh LIMIT (giá đi NGƯỢC hướng lệnh tới entry): đoạn hậu khớp chắc chắn là
            #   phần vượt entry theo chiều di chuyển -> [entry, high] (Sell Limit) hoặc
            #   [low, entry] (Buy Limit). Phía TP nằm ở phần TRƯỚC khớp -> không được tính
            #   (lỗi cũ: Sell Limit "thắng" bằng cái low xảy ra trước khi giá chạm entry).
            # - Lệnh STOP (giá đi CÙNG hướng lệnh xuyên entry): mọi mức giá vượt entry theo
            #   chiều di chuyển đều hậu khớp -> phía TP hợp lệ. NHƯNG phía SL vẫn phải xét
            #   ĐỦ BIÊN ĐỘ NẾN: nến dài có thể khớp -> BẬT NGƯỢC quét SL -> rồi mới chạy
            #   tới TP (thua thật). Không xác minh được thứ tự trong 1 nến -> để quy tắc
            #   thận trọng "SL ưu tiên trước" bên dưới xử lý: chạm cả 2 phía = THUA.
            if fill_this_candle and order_kind == "limit":
                travel_up = (direction == "SELL")
                eff_lo = entry if travel_up else lo
                eff_hi = hi if travel_up else entry
            else:
                eff_lo, eff_hi = lo, hi

            # Đã khớp -> kiểm tra SL/TP (cùng nến chạm cả 2 -> tính SL, thận trọng)
            if direction == "BUY":
                hit_sl, hit_tp = eff_lo <= sl, eff_hi >= tp1
            else:
                hit_sl, hit_tp = eff_hi >= sl, eff_lo <= tp1

            if hit_sl:
                # Vết kiểm toán: nến nào chốt kết quả - đối chiếu chart xác minh thắng/thua thật
                rec["outcome_candle_iso"] = c["datetime_utc"].isoformat()
                rec["outcome_time_iso"] = now.isoformat()
                usd = round((sl - entry) * USD_PER_POINT, 2) if direction == "BUY" \
                    else round((entry - sl) * USD_PER_POINT, 2)
                # SL đã được dời về entry (cơ chế hòa vốn mới) -> chạm SL = hòa, không phải thua
                if rec.get("be_moved") or abs(usd) < 0.01:
                    rec["status"] = "breakeven"
                    rec["usd"] = 0.0
                else:
                    rec["status"] = "loss"
                    rec["usd"] = usd
                break
            if hit_tp:
                rec["outcome_candle_iso"] = c["datetime_utc"].isoformat()
                rec["outcome_time_iso"] = now.isoformat()
                # Cờ kiểm toán: thắng quyết định NGAY TRONG nến khớp (chỉ xảy ra với lệnh
                # stop, nến dài xuyên entry chạy thẳng tới TP, phía SL sạch) - hợp lệ về
                # logic nhưng đáng soi lại vì nến dài thực tế thường kèm trượt giá
                if fill_this_candle:
                    rec["same_candle_fill_tp"] = True
                rec["status"] = "win"
                rec["usd"] = round((tp1 - entry) * USD_PER_POINT, 2) if direction == "BUY" \
                    else round((entry - tp1) * USD_PER_POINT, 2)
                break

        if rec["status"] == "waiting_fill" and trading_hours_elapsed(rec_time, now) > timeout:
            rec["status"] = "expired"  # giá không bao giờ quay về entry -> lệnh chưa từng vào
        elif rec["status"] == "pending" and trading_hours_elapsed(rec_time, now) > timeout:
            rec["status"] = "expired"
    return log


def detect_rejection_at_level(df_tf, level, direction, tol):
    """
    Nến TỪ CHỐI tại vùng entry (dùng nến ĐÃ ĐÓNG gần nhất của khung truyền vào, thường M15):
    nến có chạm vùng entry (level +/- tol) VÀ đóng cửa quay theo hướng lệnh:
    - BUY: low chạm vùng, đóng cửa > mở cửa và đóng phía trên level (từ chối giảm)
    - SELL: high chạm vùng, đóng cửa < mở cửa và đóng phía dưới level (từ chối tăng)
    Trả về True/False. Mặc định chỉ dùng để HIỂN THỊ (REQUIRE_REJECTION_CANDLE=False);
    khi bật cờ, tín hiệu không có nến từ chối sẽ bị giữ lại chờ xác nhận.
    """
    if df_tf is None or len(df_tf) < 1 or tol <= 0:
        return False
    c = df_tf.iloc[-1]  # nến đã đóng gần nhất (resample_ohlc đã loại nến chưa đóng)
    touched = (c["low"] - tol) <= level <= (c["high"] + tol)
    if not touched:
        return False
    if direction == "BUY":
        return c["close"] > c["open"] and c["close"] > level
    return c["close"] < c["open"] and c["close"] < level


def cancel_dead_premise_orders(log, df_m5):
    """
    Hủy lệnh CHỜ KHỚP khi TIỀN ĐỀ của nó đã chết - sửa lỗi lộ ra từ thực tế: bot treo
    SELL Limit dựa trên cú phá đáy đã THẤT BẠI (giá quay hẳn vào trong box) mà lệnh chờ
    vẫn sống, cách giá cả chục ATR.

    Mỗi bản ghi mang sẵn cancel_level + cancel_side (đặt lúc tạo lệnh):
    - Box thường: giá đóng cửa M5 vượt qua ĐƯỜNG GIỮA box ngược hướng lệnh -> tiền đề
      breakout đã bị phủ nhận -> hủy.
    - Spring: giá đóng cửa thủng đáy/đỉnh CÚ QUÉT -> "phá giả" hóa ra phá thật -> hủy.
    Lệnh hủy ghi status='cancelled', KHÔNG tính vào thắng/thua (lệnh chưa từng khớp).
    Chỉ áp dụng cho waiting_fill - lệnh ĐÃ khớp thì SL là cơ chế thoát, không hủy giữa chừng.
    """
    for rec in log:
        if rec.get("status") != "waiting_fill":
            continue
        level, side = rec.get("cancel_level"), rec.get("cancel_side")
        if level is None or side not in ("above", "below"):
            continue
        try:
            start_time = datetime.fromisoformat(rec["time_iso"])
        except Exception:
            continue
        candles = _candles_since(df_m5, start_time)
        if len(candles) == 0:
            continue
        dead = (candles["close"] > level).any() if side == "above" else (candles["close"] < level).any()
        if dead:
            rec["status"] = "cancelled"
            rec["cancel_reason"] = ("Giá đóng cửa vượt qua mức vô hiệu hóa "
                                    f"{level:.2f} - tiền đề setup không còn, hủy lệnh chờ")
    return log


def manage_active_trades_before_append(log, sig, current_price):
    """
    Quản lý lệnh đang chạy khi có tín hiệu MỚI cùng chiều + cùng loại. 3 quy tắc ĐÃ SỬA:

    1. HÒA VỐN = DỜI SL VỀ ENTRY (không đóng lệnh!): lệnh cũ đang có lãi >= 50% khoảng SL
       của nó -> dời SL về đúng entry. Lệnh VẪN CHẠY tiếp tới TP, chỉ là hết rủi ro.
       (Quy tắc cũ ĐÓNG NGAY lệnh đang lãi thành $0 mỗi khi có tín hiệu mới cùng chiều -
       mà box "ready" tồn tại qua nhiều lần chạy 5 phút nên tín hiệu cùng chiều xuất hiện
       liên tục -> mọi lệnh thắng đều bị flush về $0, lệnh thua thì chạy đủ SL. Đây là
       lỗi kỳ vọng âm nghiêm trọng nhất của bản trước.)
    2. GIỚI HẠN NHỒI: số lệnh cùng chiều/loại đang chạy đạt MAX_STACK_PER_DIRECTION
       -> KHÔNG nhồi thêm.
    3. CHỐNG TRÙNG BOX: mỗi box chỉ giao dịch 1 LẦN. Tín hiệu mới có cùng "vân tay"
       (fingerprint = khung + biên box + hướng) với bất kỳ bản ghi nào đã có trong log
       (kể cả đã đóng) -> không tạo bản ghi mới. Trước đây cùng 1 box "ready" có thể
       sinh lệnh lặp lại sau khi lệnh trước đóng, làm loãng thống kê.

    Trả về: (log đã cập nhật, should_append: bool, stack_note: str hoặc None)
    """
    direction = sig.get("direction")
    mode = sig.get("signal_mode")
    if not direction or mode not in ("trend", "mean_reversion", "box", "box_h4"):
        return log, True, None

    # --- Quy tắc 3: chống trùng box (kiểm tra TRƯỚC mọi thứ khác) ---
    fp = sig.get("fingerprint")
    if fp and any(r.get("fingerprint") == fp for r in log):
        return log, False, "Box đã có lệnh trước đó - không tạo lệnh trùng"

    active = [r for r in log if r.get("mode") == mode and r.get("direction") == direction
              and r.get("status") in ("waiting_fill", "pending")]

    # --- Quy tắc 1: dời SL về entry cho lệnh đã lãi >= 50% khoảng rủi ro (KHÔNG đóng lệnh) ---
    be_moved_count = 0
    for rec in active:
        if rec["status"] != "pending" or rec.get("be_moved"):
            continue
        risk_dist = abs(rec["entry"] - rec["sl"])
        if risk_dist <= 0:
            continue
        profit = (current_price - rec["entry"]) if rec["direction"] == "BUY" \
            else (rec["entry"] - current_price)
        if profit >= risk_dist * 0.5:
            rec["sl"] = rec["entry"]  # chỉ dời theo hướng CÓ LỢI, không bao giờ nới rộng
            rec["be_moved"] = True
            be_moved_count += 1

    be_note = f"🛡️ Đã dời SL về entry cho {be_moved_count} lệnh (lãi >=0.5R, hết rủi ro). " \
        if be_moved_count else ""

    # --- Quy tắc 2: giới hạn số lệnh chạy song song ---
    if not active:
        return log, True, (be_note or None)

    if len(active) >= MAX_STACK_PER_DIRECTION:
        note = (f"{be_note}Đã đạt giới hạn {MAX_STACK_PER_DIRECTION} lệnh {direction} ({mode}) "
                f"chạy song song -> KHÔNG nhồi thêm, chỉ tham khảo phân tích lần này.")
        return log, False, note

    # Còn chỗ trống nhưng đã có lệnh cùng chiều đang chạy từ box khác -> vẫn cho phép
    # (fingerprint đã chặn trùng box; box KHÁC cùng chiều là setup độc lập hợp lệ)
    latest_active = active[-1]
    entry_old = latest_active["entry"]
    usd_now = round((current_price - entry_old) * USD_PER_POINT, 2) if direction == "BUY" \
        else round((entry_old - current_price) * USD_PER_POINT, 2)
    note = (f"{be_note}Đang có {len(active)} lệnh {direction} ({mode}) chạy song song "
            f"(gần nhất entry {entry_old:.2f}, hiện {'+' if usd_now >= 0 else ''}${usd_now}) "
            f"- lệnh mới từ box khác, được phép chạy thêm.")
    return log, True, note


def append_signal(log, sig):
    """Thêm tín hiệu vừa tạo (nếu có hướng BUY/SELL) vào log để theo dõi sau này.
    Lưu thêm: signal_price (giá lúc tạo - để phân biệt limit/stop), order_kind (loại lệnh
    chờ), fingerprint (vân tay box - chống giao dịch trùng 1 box nhiều lần)."""
    if not sig.get("direction"):
        return log
    entry_type = sig.get("entry_type", "market")
    # Lệnh chờ (limit/stop) bắt đầu ở trạng thái CHỜ KHỚP, không tính thắng/thua ngay.
    # Lệnh market (vào giá hiện tại) coi như khớp ngay lập tức.
    initial_status = "waiting_fill" if entry_type == "limit" else "pending"
    now_iso = datetime.now(timezone.utc).isoformat()
    rec = {
        "time_iso": now_iso,
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
        "signal_price": sig.get("price"),
        "order_kind": sig.get("order_kind", "limit"),
        "fingerprint": sig.get("fingerprint"),
        # Điểm bối cảnh + breakdown từng phiếu - nguyên liệu cho so sánh win rate
        # nhóm điểm cao/thấp sau 30-50 lệnh (mục đích tồn tại của hệ điểm này)
        "ctx_score": sig["ctx"]["score"] if sig.get("ctx") else None,
        "ctx_votes": sig["ctx"]["votes"] if sig.get("ctx") else None,
        # Điểm VỊ TRÍ entry (Premium/Discount + FVG) - so sánh ĐẸP vs XẤU sau 30-50 lệnh
        "loc_score": sig["loc"]["score"] if sig.get("loc") else None,
        "loc_max": sig["loc"]["max"] if sig.get("loc") else None,
        # Mức vô hiệu hóa tiền đề: lệnh chờ tự HỦY khi giá đóng cửa vượt qua mức này
        # (box thường = đường giữa box; spring = đáy/đỉnh cú quét)
        "cancel_level": sig.get("cancel_level"),
        "cancel_side": sig.get("cancel_side"),
    }
    if initial_status == "pending":
        rec["fill_time_iso"] = now_iso
    log.append(rec)
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


def append_fade_plans(log, box_signal, signal_price):
    """
    Log CẢ 2 kịch bản fade (SELL cạnh trên + BUY cạnh dưới) của box CHƯA XÁC NHẬN với
    mode="fade" - thống kê RIÊNG, không trộn với lệnh box chuẩn. Mục đích: đánh giá
    KHÁCH QUAN bằng dữ liệu xem setup "vùng tranh chấp phản ứng" có edge thật không,
    thay vì tin vào cảm giác. Cả 2 lệnh cùng được theo dõi path-aware như lệnh thường:
    - Chưa khớp mà giá đóng cửa vượt QUA SL (tức cạnh đã vỡ hẳn) -> tự hủy (cancel_level=SL)
    - Khớp rồi thì TP1/SL cái nào chạm trước tính trước
    Chống trùng bằng fingerprint FADE riêng từng hướng - mỗi box mỗi hướng chỉ log 1 lần.
    """
    if not box_signal or box_signal.get("state") != "unconfirmed":
        return log, []
    appended = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for e in box_signal.get("entries", []):
        fp = (f"{box_signal['tf']}:{box_signal['box_low']:.1f}-"
              f"{box_signal['box_high']:.1f}:FADE:{e['direction']}")
        if any(r.get("fingerprint") == fp for r in log):
            continue
        log.append({
            "time_iso": now_iso,
            "direction": e["direction"],
            "entry": e["entry"], "sl": e["sl"], "tp1": e["tp1"],
            "score": 0, "mode": "fade", "confidence": "normal",
            "entry_type": "limit", "status": "waiting_fill",
            "timeout_hours": SIGNAL_TIMEOUT_HOURS,
            "signal_price": signal_price,
            "order_kind": e.get("order_kind", "limit"),
            "fingerprint": fp,
            # Tiền đề của lệnh fade = "cạnh box giữ được". Giá đóng cửa vượt QUA SL khi
            # chưa khớp nghĩa là cạnh đã vỡ hẳn -> hủy kế hoạch, không đu theo.
            "cancel_level": e["sl"],
            "cancel_side": "above" if e["direction"] == "SELL" else "below",
        })
        appended.append(e["direction"])
    return log, appended


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
# 2c2. GHIM BOX ĐÃ XÁC NHẬN (BOX PINNING) - chống "box mới cướp sóng"
# ============================================================
# Lỗi thực tế: bot chỉ nhớ box "gần giá nhất" MỖI LẦN CHẠY. Khi box vừa xác nhận
# breakout, chính cú breakout đẩy giá vào vùng một nến lớn khác -> lần chạy sau bot
# chọn box MỚI, box cũ (đang chờ retest - đúng lúc giá trị nhất) bị quên sạch ->
# không bao giờ có tín hiệu khi giá quay về test cạnh vừa phá.
# Giải pháp: box đạt trạng thái xác nhận (waiting_retest/ready/spring_*) được GHIM
# vào file box_state.json (sống qua các lần chạy như signal_log). Các lần chạy sau,
# box ghim được ưu tiên đánh giá lại (retest? vô hiệu?) và THAY THẾ box mới phát hiện
# trên cùng khung - cho đến khi nó: đã ra lệnh (fingerprint có trong log) / bị vô hiệu
# / hết hạn theo dõi. Chỉ khi đó box mới mới được lên sóng.
BOX_STATE_PATH = "box_state.json"
BOX_TRACK_MAX_HOURS = 24   # theo dõi retest tối đa 24h sau xác nhận (mức đổi vai dài hạn
                           # hơn đã có sổ flipped_levels lo, ghim chỉ phục vụ setup retest tươi)
PIN_ABANDON_ATR = 6.0      # VAN XẢ: entry cách giá > 6x ATR theo hướng THUẬN -> giá đã bỏ đi,
                           # setup lỡ, nhả ghim để box mới ở vùng giá hiện tại lên sóng
PIN_SPRING_MAX_HOURS = 6   # VAN XẢ: box spring chờ retest quá 6h mà giá không quay về biên
                           # bị quét -> phá vỡ giả đã thất bại thực tế (giá đi luôn), nhả ghim


def load_pinned_box():
    if not os.path.exists(BOX_STATE_PATH):
        return None
    try:
        with open(BOX_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_pinned_box(pin):
    with open(BOX_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(pin, f, ensure_ascii=False, indent=2)


def clear_pinned_box():
    try:
        if os.path.exists(BOX_STATE_PATH):
            os.remove(BOX_STATE_PATH)
    except Exception:
        pass


def evaluate_pinned_box(pin, df_m5, live_price, atr_m5=None):
    """
    Đánh giá lại box đã ghim bằng dữ liệu hiện tại. Trả về dict box (đúng cấu trúc
    find_box_state trả ra) với state cập nhật, hoặc None nếu box đã chết:
    - HẾT HẠN: quá BOX_TRACK_MAX_HOURS kể từ lúc xác nhận
    - VÔ HIỆU (xuyên ngược): giá đi XUYÊN NGƯỢC hết box (phá lên mà giá về dưới cạnh dưới
      / phá xuống mà giá vượt cạnh trên); spring: giá thủng đáy/đỉnh cú quét
    - GIÁ BỎ ĐI QUÁ XA (van xả): entry gần nhất cách giá > PIN_ABANDON_ATR lần ATR theo
      HƯỚNG THUẬN của lệnh -> setup đã lỡ, giá không quay lại, NHẢ GHIM để box mới ở vùng
      giá hiện tại lên sóng. Sửa bệnh "bot ôm 1 box cả ngày": box 4001-4026 ghim từ sáng
      trong khi giá chạy lên 4071, entry cách 22x ATR mà vẫn treo.
    - SPRING HẾT KIÊN NHẪN (van xả): box spring chờ retest quá PIN_SPRING_MAX_HOURS mà giá
      không quay về biên bị quét -> phá giả đã thất bại thực tế, nhả.
    - RETEST: kiểm tra bằng đường đi high/low nến M5 KỂ TỪ LÚC XÁC NHẬN -> ready khi chạm entry
    """
    try:
        confirmed_at = datetime.fromisoformat(pin["confirmed_at_iso"])
    except Exception:
        return None
    now = datetime.now(timezone.utc)
    if (now - confirmed_at).total_seconds() > BOX_TRACK_MAX_HOURS * 3600:
        return None

    is_spring = str(pin.get("state", "")).startswith("spring")

    # --- Van xả 1: SPRING hết kiên nhẫn ---
    if is_spring and (now - confirmed_at).total_seconds() > PIN_SPRING_MAX_HOURS * 3600:
        return None

    # --- Van xả 2: GIÁ BỎ ĐI QUÁ XA theo hướng thuận ---
    if atr_m5 and atr_m5 > 0:
        entries = pin.get("entries", [])
        if entries:
            nearest = min(entries, key=lambda e: abs(e["price"] - live_price))
            dist_atr = abs(live_price - nearest["price"]) / atr_m5
            direction = pin["direction"]
            # "thuận" = giá đã đi đúng hướng lệnh và bỏ xa entry (BUY: giá vượt LÊN trên
            # entry; SELL: giá tụt XUỐNG dưới entry) - cơ hội vào đã lỡ, không chờ nữa
            gone_favorable = (direction == "BUY" and live_price > nearest["price"]) or \
                             (direction == "SELL" and live_price < nearest["price"])
            if gone_favorable and dist_atr > PIN_ABANDON_ATR:
                return None

    # --- Vô hiệu hóa (xuyên ngược) ---
    if is_spring:
        sweep = pin.get("sweep_extreme")
        if sweep is not None:
            if pin["direction"] == "BUY" and live_price < sweep:
                return None
            if pin["direction"] == "SELL" and live_price > sweep:
                return None
    else:
        if pin["confirm_dir"] == "up" and live_price < pin["box_low"]:
            return None
        if pin["confirm_dir"] == "down" and live_price > pin["box_high"]:
            return None

    # --- Retest theo đường đi nến M5 kể từ lúc xác nhận ---
    tol = pin.get("retest_tol", 1.0)
    candles = _candles_since(df_m5, confirmed_at)
    retested = False
    for e in pin.get("entries", []):
        level = e["price"]
        if len(candles) > 0:
            touched = ((candles["low"] - tol <= level) & (level <= candles["high"] + tol)).any()
        else:
            touched = False
        if touched or abs(live_price - level) <= tol:
            retested = True
            break

    box = {k: pin[k] for k in ("box_high", "box_low", "box_mid", "color",
                                 "direction", "alignment", "confirm_dir", "entries")}
    if is_spring:
        box["sweep_extreme"] = pin.get("sweep_extreme")
        box["state"] = "spring_ready" if retested else "spring_waiting"
    else:
        box["state"] = "ready" if retested else "waiting_retest"
    box["pinned"] = True  # đánh dấu để hiển thị/debug biết box này đến từ ghim
    return box


def make_box_fingerprint(tf, box_low, box_high, direction, is_spring):
    """Cùng công thức fingerprint dùng cho log lệnh - đảm bảo pin và lệnh khớp nhau."""
    suffix = ":SPRING" if is_spring else ""
    return f"{tf}:{box_low:.1f}-{box_high:.1f}:{direction}{suffix}"


# ============================================================
# 2d. VỊ TRÍ ENTRY (LOCATION GRADE) - Premium/Discount + FVG (ICT/SMC)
# ============================================================
# Khác với ĐIỂM BỐI CẢNH 💪 (đo HƯỚNG lệnh có thuận dòng chảy không), khối 🎯 này đo
# VỊ TRÍ: cùng một hướng SELL, bán ở giá nào thì có lợi thế?
# - Premium/Discount: chia đôi con sóng gần nhất tại Fib 0.5. SELL "đẹp" ở nửa TRÊN
#   (premium - bán giá đắt), BUY "đẹp" ở nửa DƯỚI (discount - mua giá rẻ). Bán sát đáy
#   sóng = đúng chỗ dễ bật hồi nhất, hướng đúng vẫn thua vì SL bị quét trước.
# - FVG (Fair Value Gap): vùng giá "bỏ trống" khi nến chạy quá nhanh (high nến 1 không
#   chạm low nến 3) - giá có xu hướng quay lại LẤP rồi mới đi tiếp. Entry ngay tại FVG
#   cùng hướng = đứng đúng chỗ "giá có hẹn quay lại".
# Chỉ CHẤM ĐIỂM + LOG, không chặn lệnh - sau 30-50 lệnh, thống kê ĐẸP vs XẤU tự trả lời
# phương pháp ICT có tạo khác biệt thật trên XAU không.

FVG_SCAN_CANDLES = 120       # quét FVG trong N nến gần nhất của mỗi khung
FVG_ENTRY_TOL_ATR = 0.2      # entry cách mép FVG <= 0.2x ATR vẫn tính là "trong vùng"


def detect_fvgs(df, max_scan=FVG_SCAN_CANDLES):
    """
    Tìm các FVG CHƯA LẤP trong max_scan nến cuối. FVG 3 nến:
    - Bullish FVG: high nến[i] < low nến[i+2] -> vùng trống (high[i], low[i+2]), đóng vai
      trò HỖ TRỢ khi giá quay về (BUY zone)
    - Bearish FVG: low nến[i] > high nến[i+2] -> vùng trống (high[i+2], low[i]), KHÁNG CỰ
    Đã lấp = có nến sau đó xuyên hết vùng (low <= đáy vùng với bullish / high >= đỉnh vùng
    với bearish) -> loại. Trả về list {"low","high","direction"} sắp theo mới nhất trước.
    """
    if df is None or len(df) < 3:
        return []
    recent = df.iloc[-max_scan:].reset_index(drop=True)
    fvgs = []
    for i in range(len(recent) - 2):
        c1, c3 = recent.iloc[i], recent.iloc[i + 2]
        after = recent.iloc[i + 3:]
        if c1["high"] < c3["low"]:  # bullish gap
            gap_lo, gap_hi = c1["high"], c3["low"]
            filled = (after["low"] <= gap_lo).any() if len(after) else False
            if not filled:
                fvgs.append({"low": float(gap_lo), "high": float(gap_hi), "direction": "bull"})
        elif c1["low"] > c3["high"]:  # bearish gap
            gap_lo, gap_hi = c3["high"], c1["low"]
            filled = (after["high"] >= gap_hi).any() if len(after) else False
            if not filled:
                fvgs.append({"low": float(gap_lo), "high": float(gap_hi), "direction": "bear"})
    return list(reversed(fvgs))


def entry_location_grade(entry, direction, fib, fvgs_by_tf, atr_m5):
    """
    Chấm VỊ TRÍ của entry lệnh chính. Trả về:
    {"score": int, "max": int, "label": str, "checks": [chuỗi hiển thị]} hoặc None nếu
    không có dữ liệu nào để chấm. Mỗi tiêu chí 1 điểm:
    1. Premium/Discount: SELL ở nửa trên sóng / BUY ở nửa dưới sóng (mốc = Fib 0.5)
    2. FVG: entry nằm trong (hoặc sát <= 0.2 ATR) một FVG CÙNG HƯỚNG chưa lấp (ưu tiên
       khung lớn: H1 trước, M15 sau)
    """
    checks, score, mx = [], 0, 0

    # --- Tiêu chí 1: Premium/Discount ---
    if fib and fib.get("swing_high") and fib.get("swing_low"):
        mx += 1
        mid = (fib["swing_high"] + fib["swing_low"]) / 2
        in_premium = entry > mid
        ok = (direction == "SELL" and in_premium) or (direction == "BUY" and not in_premium)
        zone_txt = "Premium" if in_premium else "Discount"
        if ok:
            score += 1
            side_txt = "nửa TRÊN con sóng (bán giá đắt)" if direction == "SELL"                 else "nửa DƯỚI con sóng (mua giá rẻ)"
            checks.append(f"· {zone_txt} ✓ - {direction} ở {side_txt}")
        else:
            side_txt = "nửa DƯỚI sóng, dễ bán đúng đáy hồi" if direction == "SELL"                 else "nửa TRÊN sóng, dễ mua đúng đỉnh hồi"
            checks.append(f"· {zone_txt} ✗ - {direction} ở {side_txt}")

    # --- Tiêu chí 2: FVG cùng hướng ---
    want = "bull" if direction == "BUY" else "bear"
    tol = FVG_ENTRY_TOL_ATR * atr_m5 if atr_m5 and atr_m5 > 0 else 0.5
    has_any_fvg_data = any(v is not None for v in fvgs_by_tf.values())
    if has_any_fvg_data:
        mx += 1
        hit_tf = None
        for tf_name in ("H1", "M15"):  # ưu tiên khung lớn
            for g in (fvgs_by_tf.get(tf_name) or []):
                if g["direction"] == want and (g["low"] - tol) <= entry <= (g["high"] + tol):
                    hit_tf = tf_name
                    break
            if hit_tf:
                break
        if hit_tf:
            score += 1
            checks.append(f"· FVG {hit_tf} ✓ - entry trong vùng giá bỏ trống chưa lấp")
        else:
            checks.append("· FVG ✗ - entry không trùng vùng giá bỏ trống nào")

    if mx == 0:
        return None
    if score == mx:
        label = "ĐẸP"
    elif score == 0:
        label = "XẤU"
    else:
        label = "TRUNG BÌNH"
    return {"score": score, "max": mx, "label": label, "checks": checks}


def compare_loc_win_rate(log):
    """
    So sánh win rate nhóm vị trí ĐẸP (đủ điểm) vs XẤU (0 điểm) trên các lệnh đã đóng -
    câu trả lời bằng dữ liệu cho câu hỏi "Premium/Discount + FVG có tạo khác biệt thật
    trên XAU không". Trả về chuỗi hiển thị hoặc None nếu chưa có bản ghi mang điểm vị trí.
    """
    closed = [r for r in log if r.get("status") in ("win", "loss")
              and r.get("loc_score") is not None and r.get("loc_max")]
    if not closed:
        return None

    def _grp(records):
        if not records:
            return "0 lệnh"
        w = sum(1 for r in records if r["status"] == "win")
        return f"{w}W/{len(records) - w}L ({round(w / len(records) * 100)}%)"

    good = [r for r in closed if r["loc_score"] == r["loc_max"]]
    bad = [r for r in closed if r["loc_score"] == 0]
    return f"Vị trí ĐẸP: {_grp(good)} | XẤU: {_grp(bad)}"


# ============================================================
# 2e. ĐIỂM BỐI CẢNH (CONTEXT SCORE) - lớp tin cậy, KHÔNG phải cò súng
# ============================================================
# Khác hẳn hệ chấm điểm cũ (đã bỏ vì cộng nhiều chỉ báo TRÙNG thông tin - EMA↓ + MA↓ +
# BOS↓ là 1 thông tin đếm 3 lần): mỗi phiếu ở đây đo 1 khía cạnh ĐỘC LẬP, và điểm này
# KHÔNG tự sinh lệnh, không chặn lệnh, không đổi SL/TP - chỉ trả lời câu hỏi "bối cảnh
# đang thuận hay nghịch với lệnh box?" + được LƯU VÀO LOG để sau 30-50 lệnh so sánh
# win rate nhóm điểm cao vs thấp bằng dữ liệu thật, rồi mới quyết có trao quyền lọc lệnh.
#
# 5 phiếu (mỗi phiếu +1/0/-1 so với HƯỚNG LỆNH box), thang tổng -5 -> +4:
# 1. Dow (H1)          - cấu trúc đỉnh/đáy
# 2. Ichimoku (H1)     - vị thế giá so với mây (chỉ tính mức "mạnh nhất")
# 3. EMA đa khung      - CHỈ H4/H1/M30 (cố ý loại M15/M5: entry box là lệnh chờ pullback,
#                        khung nhỏ gần như luôn ngược hướng lệnh ngay trước lúc khớp)
# 4. MACD hist (M15)   - động lượng (M15 thay vì M5 để phiếu không nhấp nháy giữa các lần chạy)
# 5. RSI (M5)          - CHỈ PHẠT: BUY khi quá mua / SELL khi quá bán -> -1, không có phiếu +
# (Phiên giao dịch KHÔNG tham gia chấm điểm - chỉ giữ dòng cảnh báo riêng như cũ.)

CTX_HIGH_GROUP_MIN = 2   # ngưỡng chia nhóm "điểm cao" khi so sánh thống kê (>= +2)
CTX_LOW_CONFIDENCE = -3  # điểm <= mức này -> gắn confidence "low" (thay 2 quy tắc Dow/Ichimoku cũ)


def compute_context_score(direction, dow_trend, ichimoku, trend_arrows, macd_hist_m15, rsi_m5):
    """
    Trả về {"score": int, "votes": {tên: +1/0/-1}, "label": str, "icon_line": str}.
    direction: "BUY"/"SELL" (hướng lệnh box). Mỗi phiếu so với hướng này.
    """
    want = "up" if direction == "BUY" else "down"
    against = "down" if direction == "BUY" else "up"
    votes = {}

    # 1. Dow (H1)
    if dow_trend == want:
        votes["Dow"] = 1
    elif dow_trend == against:
        votes["Dow"] = -1
    else:
        votes["Dow"] = 0

    # 2. Ichimoku (H1) - chỉ mức "mạnh nhất" mới được bỏ phiếu, mới hình thành/trong mây = 0
    votes["Ichimoku"] = 0
    if ichimoku:
        strong_same = (ichimoku["strength"] == "strong_bull" and direction == "BUY") or \
                      (ichimoku["strength"] == "strong_bear" and direction == "SELL")
        strong_opposite = (ichimoku["strength"] == "strong_bull" and direction == "SELL") or \
                          (ichimoku["strength"] == "strong_bear" and direction == "BUY")
        if strong_same:
            votes["Ichimoku"] = 1
        elif strong_opposite:
            votes["Ichimoku"] = -1

    # 3. EMA đa khung - chỉ H4/H1/M30
    big_tfs = [trend_arrows.get(tf) for tf in ("H4", "H1", "M30")]
    big_tfs = [t for t in big_tfs if t is not None]
    votes["EMA"] = 0
    if len(big_tfs) >= 2:
        same = sum(1 for t in big_tfs if t == want)
        opposite = sum(1 for t in big_tfs if t == against)
        if same >= 2:
            votes["EMA"] = 1
        elif opposite >= 2:
            votes["EMA"] = -1

    # 4. MACD histogram (M15) - sát 0 (|hist| < 0.05) coi là trung tính
    votes["MACD"] = 0
    if macd_hist_m15 is not None and abs(macd_hist_m15) >= 0.05:
        hist_dir = "up" if macd_hist_m15 > 0 else "down"
        votes["MACD"] = 1 if hist_dir == want else -1

    # 5. RSI (M5) - chỉ phạt đu vào vùng cực đoan ngược
    votes["RSI"] = 0
    if rsi_m5 is not None:
        if (direction == "BUY" and rsi_m5 >= 70) or (direction == "SELL" and rsi_m5 <= 30):
            votes["RSI"] = -1

    score = sum(votes.values())
    if score >= 3:
        label = "THUẬN MẠNH 💪"
    elif score >= 1:
        label = "thuận nhẹ"
    elif score == 0:
        label = "trung tính"
    elif score >= -2:
        label = "nghịch nhẹ ⚠️"
    else:
        label = "NGHỊCH MẠNH 🚨"

    sym = {1: "✓", -1: "✗", 0: "–"}
    icon_line = "  ".join(f"{name}{sym[v]}" for name, v in votes.items())
    return {"score": score, "votes": votes, "label": label, "icon_line": icon_line}


def compare_ctx_win_rate(log, high_min=CTX_HIGH_GROUP_MIN):
    """
    So sánh win rate nhóm điểm bối cảnh CAO (>= high_min) vs THẤP (<= 0) trên các lệnh đã
    đóng thắng/thua - đây là mục đích tồn tại của cả hệ điểm: sau 30-50 lệnh, dòng này cho
    biết bối cảnh có thực sự dự báo được kết quả không, quyết định có nâng thành bộ lọc.
    Trả về chuỗi hiển thị, hoặc None nếu chưa có bản ghi nào mang điểm bối cảnh.
    """
    closed = [r for r in log if r.get("status") in ("win", "loss") and r.get("ctx_score") is not None]
    if not closed:
        return None

    def _grp(records):
        if not records:
            return "0 lệnh"
        w = sum(1 for r in records if r["status"] == "win")
        return f"{w}W/{len(records) - w}L ({round(w / len(records) * 100)}%)"

    high = [r for r in closed if r["ctx_score"] >= high_min]
    low = [r for r in closed if r["ctx_score"] <= 0]
    return f"Điểm ≥+{high_min}: {_grp(high)} | Điểm ≤0: {_grp(low)}"



# ============================================================
# 3. LOGIC TẠO TÍN HIỆU
# ============================================================
def generate_signal(active_zone_directions=None):
    active_zone_directions = active_zone_directions or set()
    # 3 lời gọi API mỗi lần chạy (~600 credits/ngày, trần free 800):
    # 1. M5 x1000 - nguồn chính cho box/chỉ báo, tự gộp thành M15/M30
    # 2. M1 x1000 (~16.7h) - RIÊNG cho theo dõi kết quả: độ phân giải đường đi giá gấp 5,
    #    nến khớp "mơ hồ" (quét cả SL lẫn TP trong 1 nến M5) phần lớn được M1 phân xử rõ
    # 3. H1 gốc x300 - cache theo giờ (~24 credits/ngày), lịch sử dài cho Ichimoku/Dow/FVG
    df_m5 = get_ohlc("5min", outputsize=1000)  # ~3.5 ngày dữ liệu M5
    try:
        df_m1 = get_ohlc("1min", outputsize=1000)
        if df_m1 is None or len(df_m1) < 50:
            df_m1 = None
    except Exception as e:
        print(f"Lỗi lấy M1 (tracking rơi về M5): {e}")
        df_m1 = None  # LƯỚI AN TOÀN: M1 lỗi -> tracking dùng M5 như cũ, không mất lượt theo dõi
    df_m15 = resample_ohlc(df_m5, "15min")
    df_m30 = resample_ohlc(df_m5, "30min")
    df_h1_native = fetch_h1_native_cached()
    df_h1 = df_h1_native if (df_h1_native is not None and len(df_h1_native) >= 60) \
        else resample_ohlc(df_m5, "1h")

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
    # ---- SỔ MỨC ĐỔI VAI: nạp + bảo trì trước (dùng cho zones), đăng ký sau khi có box ----
    flip_levels = load_flipped_levels()
    flip_levels = maintain_flipped_levels(flip_levels, df_m5, atr_m5)

    zones_above, zones_below = finalize_watch_zones(clusters, current_price, atr_m5)

    # Mức đổi vai vào VÙNG THEO DÕI (ưu tiên đầu danh sách - loại mức chất lượng cao nhất:
    # cấu trúc đã được thị trường xác nhận bằng breakout thật)
    for lv in flip_levels:
        zone = {"price_low": round(lv["level"] - 0.15, 2), "price_high": round(lv["level"] + 0.15, 2),
                "sources": ["ĐổiVai"], "stars": "⭐",
                "distance_atr": round(abs(current_price - lv["level"]) / atr_m5, 1) if atr_m5 > 0 else 0,
                "order_type": "Buy Limit" if lv["role"] == "support" else "Sell Limit"}
        if lv["level"] > current_price:
            zones_above.insert(0, zone)
        elif lv["level"] < current_price:
            zones_below.insert(0, zone)
    zones_above, zones_below = zones_above[:3], zones_below[:3]

    # Mẫu hình nến mẹ - nến con (Inside Bar), quét trên M15 theo đề xuất
    # (M5 quá nhiễu cho pattern này, M15 phản ánh cấu trúc rõ hơn)
    inside_bar = detect_inside_bar_setup(df_m15, atr_m15_series)

    # MACD - đo momentum (tốc độ/gia tốc), khác trend (chỉ đo hướng)
    macd_line, signal_line, hist = macd(df_m5["close"])
    hist_now = hist.iloc[-1]
    macd_bias = "up" if hist_now > 0 else ("down" if hist_now < 0 else None)
    divergence = detect_divergence(df_m5, hist)

    # MACD histogram M15 - dùng RIÊNG cho phiếu điểm bối cảnh (M5 đổi dấu quá thường
    # xuyên làm phiếu nhấp nháy giữa các lần chạy 5 phút; M15 ổn định hơn)
    _, _, hist_m15 = macd(df_m15["close"])
    macd_hist_m15 = float(hist_m15.iloc[-1]) if len(hist_m15) else None

    session_ok = is_active_session()
    news_warning = check_upcoming_news()

    # --- BOX DETECTOR: thay thế hoàn toàn hệ thống chấm điểm cũ (Trend/Mean-Reversion/
    # Zone Setup/Thử nghiệm). Tìm box H1 TRƯỚC (cấu trúc lớn), sau đó ràng buộc box M15 phải
    # nằm LỌT trong phạm vi box H1 đã chọn - đúng cấu trúc "range nhỏ lồng trong range to".
    # Cả 2 đều tự loại bỏ box đã quá xa giá hiện tại (không còn liên quan để giao dịch).
    atr_h1_series = atr(df_h1)
    box_h1 = find_box_state(df_h1, atr_h1_series, live_price=current_price)
    box_m15 = None
    if box_h1:
        box_m15 = find_box_state(df_m15, atr_m15_series,
                                  bound_range=(box_h1["box_low"], box_h1["box_high"]),
                                  live_price=current_price)
    else:
        # không có box H1 thì M15 tìm tự do
        box_m15 = find_box_state(df_m15, atr_m15_series, live_price=current_price)

    # ---- BOX H4 (setup swing): gộp từ nến H1 gốc, 0 credit; chỉ dùng nến H4 đủ 4 con
    # đã đóng (resample_h1_to_h4 tự loại nến đang hình thành). Ngưỡng thanh khoản 1.8x. ----
    df_h4 = resample_h1_to_h4(df_h1)
    box_h4 = None
    atr_h4_series = None
    if df_h4 is not None and len(df_h4) >= 30:
        atr_h4_series = atr(df_h4)
        box_h4 = find_box_state(df_h4, atr_h4_series, range_mult=BOX_RANGE_ATR_MULT_H4,
                                 live_price=current_price)

    # ---- BOX GHIM: box đã xác nhận từ lần chạy trước được ƯU TIÊN, chống box mới cướp sóng ----
    pinned = load_pinned_box()
    if pinned:
        # Đã ra lệnh cho box này rồi (fingerprint có trong log) -> nhiệm vụ hoàn thành, gỡ ghim
        log_ro = load_signal_log()
        if pinned.get("fingerprint") and any(r.get("fingerprint") == pinned["fingerprint"] for r in log_ro):
            clear_pinned_box()
            pinned = None
    if pinned:
        pinned_box = evaluate_pinned_box(pinned, df_m5, current_price, atr_m5=atr_m5)
        if pinned_box is None:
            clear_pinned_box()  # hết hạn/vô hiệu -> box mới được lên sóng từ lần này
        elif pinned.get("tf") == "H4":
            box_h4 = pinned_box
        elif pinned.get("tf") == "H1":
            box_h1 = pinned_box
        else:
            box_m15 = pinned_box
    atr_by_tf = {"H1": float(atr_h1_series.iloc[-1]) if len(atr_h1_series) else atr_m5,
                 "M15": float(atr_m15_series.iloc[-1]) if len(atr_m15_series) else atr_m5,
                 "H4": float(atr_h4_series.iloc[-1]) if atr_h4_series is not None and len(atr_h4_series) else atr_m5}
    box_signal = build_box_signal(box_m15, box_h1, atr_m5,
                                   current_price=current_price, atr_by_tf=atr_by_tf, box_h4=box_h4)

    # ---- SỔ MỨC ĐỔI VAI (tiếp): đăng ký cạnh vừa vỡ -> lưu -> áp tường TP + cộng hưởng ----
    flip_levels = register_flipped_level(flip_levels, box_signal)
    save_flipped_levels(flip_levels)

    # ---- LƯU GHIM: box đạt trạng thái xác nhận -> ghim lại để các lần chạy sau theo dõi
    # retest, kể cả khi box mới xuất hiện. Bảo toàn confirmed_at gốc nếu là cùng box. ----
    if box_signal and box_signal.get("confirm_dir") and \
            box_signal["state"] in ("waiting_retest", "ready", "spring_waiting", "spring_ready"):
        raw_src = {"H4": box_h4, "H1": box_h1, "M15": box_m15}.get(box_signal["tf"])
        if raw_src:
            fp = make_box_fingerprint(box_signal["tf"], box_signal["box_low"],
                                       box_signal["box_high"], box_signal["direction"],
                                       box_signal["alignment"] == "spring")
            old_pin = load_pinned_box()
            confirmed_at = old_pin["confirmed_at_iso"] if old_pin and old_pin.get("fingerprint") == fp \
                else datetime.now(timezone.utc).isoformat()
            atr_tf_pin = atr_by_tf.get(box_signal["tf"], atr_m5)
            pin = {k: raw_src[k] for k in ("box_high", "box_low", "box_mid", "color",
                                             "direction", "alignment", "confirm_dir", "entries")}
            pin.update({"tf": box_signal["tf"], "state": raw_src["state"],
                        "fingerprint": fp, "confirmed_at_iso": confirmed_at,
                        "retest_tol": round(BOX_RETEST_TOLERANCE_ATR * atr_tf_pin, 3)})
            if raw_src.get("sweep_extreme") is not None:
                pin["sweep_extreme"] = raw_src["sweep_extreme"]
            save_pinned_box(pin)

    if box_signal and box_signal.get("entries"):
        for e in box_signal["entries"]:
            e["wall_notes"] = apply_flip_walls(e, flip_levels, atr_m5)
            e["flip_note"] = flip_confluence_note(e, flip_levels, atr_m5)

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

    # Điểm bối cảnh: tính cho MỌI box đã có hướng (kể cả waiting_retest - để bạn thấy trước
    # bối cảnh trong lúc chờ retest), không chỉ lúc ready.
    ctx = None
    if box_signal and box_signal.get("direction"):
        ctx = compute_context_score(box_signal["direction"], dow_trend, ichimoku_h1,
                                     trend_arrows, macd_hist_m15, rsi_m5)
        box_signal["ctx"] = ctx

    if market_flat:
        block_reason = "Thị trường đang đứng yên (nghỉ lễ/ngoài giờ giao dịch thực) - dữ liệu gần như không đổi, tạm dừng phân tích"
    elif not box_signal:
        block_reason = "Chưa tìm thấy nến tập trung thanh khoản nào đủ điều kiện trong phạm vi quét"
    elif box_signal["state"] in ("ready", "spring_ready"):
        direction = box_signal["direction"]
        # Box H4 = setup swing, log mode RIÊNG "box_h4" để thống kê tách biệt với box H1
        signal_mode = "box_h4" if box_signal["tf"] == "H4" else "box"
        if box_signal["alignment"] == "spring":
            pass  # setup Spring có khối hiển thị riêng đầy đủ trong tin nhắn, không cần note
        elif box_signal["alignment"] == "ngược":
            confidence = "low"
            confidence_notes.append("Lệnh NGƯỢC xu hướng của nến thanh khoản (chỉ 2 điểm entry, thận trọng hơn)")
        # Bối cảnh NGHỊCH MẠNH (điểm <= -3) -> hạ độ tin cậy. Thay thế 2 quy tắc ad-hoc cũ
        # (xung đột Dow, xung đột Ichimoku) bằng 1 cơ chế thống nhất - mỗi xung đột đơn lẻ
        # giờ chỉ là 1 phiếu trừ, phải nhiều phiếu cùng chống mới đủ hạ tin cậy.
        if ctx and ctx["score"] <= CTX_LOW_CONFIDENCE:
            confidence = "low"
            confidence_notes.append(f"Bối cảnh NGHỊCH MẠNH ({ctx['score']:+d}: {ctx['icon_line']}) - rủi ro cao hơn hẳn")
    else:
        state_txt = {"unconfirmed": "chưa xác nhận breakout",
                     "waiting_retest": "đã xác nhận, đang chờ giá quay về test biên",
                     "spring_waiting": "PHÁ VỠ GIẢ phát hiện, chờ giá quay về biên bị quét",
                     }.get(box_signal["state"], box_signal["state"])
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
        "ctx": ctx,
        "box_chart_df": (df_h1 if box_signal["tf"] == "H1" else df_m15) if box_signal else None,
        "trend_arrows": trend_arrows,
    }

    result["df_m5"] = df_m5
    # Nguồn nến cho theo dõi kết quả path-aware: ưu tiên M1 (độ phân giải gấp 5), rơi về M5 nếu M1 lỗi
    result["df_track"] = df_m1 if df_m1 is not None else df_m5

    if box_signal and box_signal["state"] in ("ready", "spring_ready") and box_signal.get("entries"):
        # Entry GẦN GIÁ NHẤT làm đại diện theo dõi thắng/thua - hợp lý hơn "entry đầu tiên"
        # cũ: entry gần nhất là lệnh có xác suất khớp cao nhất, thống kê phản ánh sát thực tế.
        primary = min(box_signal["entries"], key=lambda e: abs(current_price - e["entry"]))
        dist_atr = abs(current_price - primary["entry"]) / atr_m5 if atr_m5 > 0 else 0

        # CHẶN ENTRY QUÁ XA: entry gần nhất cách giá > 4x ATR -> gần như chắc chắn hết hạn
        # vô nghĩa (thực tế: lệnh cách 10.9x ATR, còn 3.9h - không bao giờ khớp). Không phát
        # lệnh, chỉ hiển thị box tham khảo.
        if dist_atr > FAR_ENTRY_ATR:
            result["direction"] = None
            result["block_reason"] = (f"Entry gần nhất ({primary['entry']:.2f}) cách giá "
                                       f"{dist_atr:.1f}x ATR - ngoài tầm với, không phát lệnh "
                                       f"(box vẫn hiển thị để theo dõi)")
            return result

        # NẾN TỪ CHỐI tại vùng entry (M15): chạm vùng entry và đóng cửa quay theo hướng lệnh.
        # Mặc định chỉ HIỂN THỊ thông tin; bật REQUIRE_REJECTION_CANDLE=True để bắt buộc.
        rejection_ok = detect_rejection_at_level(df_m15, primary["entry"], primary["direction"],
                                                  tol=0.3 * atr_m5)
        result["rejection_candle"] = rejection_ok
        if REQUIRE_REJECTION_CANDLE and not rejection_ok:
            result["direction"] = None
            result["block_reason"] = ("Chưa có nến từ chối M15 tại vùng entry "
                                       "(REQUIRE_REJECTION_CANDLE đang bật) - chờ xác nhận")
            return result

        # Mức HỦY LỆNH CHỜ (tiền đề chết): box thường = đường giữa box (ngược hướng lệnh);
        # spring = đáy/đỉnh cú quét (thủng = phá giả hóa ra phá thật).
        if box_signal["alignment"] == "spring":
            cancel_level = box_signal.get("sweep_extreme")
            cancel_side = "below" if primary["direction"] == "BUY" else "above"
        else:
            cancel_level = box_signal["box_mid"]
            cancel_side = "above" if primary["direction"] == "SELL" else "below"

        result.update({
            "entry": primary["entry"], "sl": primary["sl"],
            "tp1": primary["tp1"], "tp2": primary["tp2"], "tp3": primary["tp3"],
            "entry_type": "limit",
            "order_kind": primary.get("order_kind", "limit"),
            "cancel_level": round(cancel_level, 2) if cancel_level is not None else None,
            "cancel_side": cancel_side,
            # Vân tay box: cùng công thức với box ghim (make_box_fingerprint) - lệnh ra xong
            # thì fingerprint khớp pin -> pin tự gỡ ở lần chạy sau
            "fingerprint": make_box_fingerprint(box_signal["tf"], box_signal["box_low"],
                                                 box_signal["box_high"], box_signal["direction"],
                                                 box_signal["alignment"] == "spring"),
        })
        result["fib_note"] = fib_confluence_note(fib, primary["entry"], primary["sl"], primary["tp1"], atr_m5)

        # 🎯 VỊ TRÍ ENTRY (Premium/Discount + FVG): chấm cho LỆNH CHÍNH - đo "chỗ đứng"
        # của giá entry trong con sóng, tách bạch với điểm bối cảnh 💪 (đo hướng)
        fvgs_by_tf = {"H1": detect_fvgs(df_h1), "M15": detect_fvgs(df_m15)}
        result["loc"] = entry_location_grade(primary["entry"], primary["direction"],
                                              fib, fvgs_by_tf, atr_m5)

        # Cảnh báo "không mua đuổi": giá hiện tại cách entry khá xa (2.5-4x ATR M5)
        # -> nhấn mạnh đây là lệnh CHỜ, tuyệt đối không vào market đuổi theo.
        if dist_atr >= CHASE_WARNING_ATR:
            result["chase_warning"] = (f"Giá đang cách entry {dist_atr:.1f}x ATR - "
                                        f"CHỜ GIÁ VỀ {primary['entry']:.2f}, không đuổi lệnh!")

    return result


# ============================================================
# 4. FORMAT TIN NHẮN & GỬI TELEGRAM
# ============================================================
def _esc(s):
    """Escape ký tự đặc biệt HTML cho Telegram parse_mode=HTML (nội dung động có thể chứa >=, <...)."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_message(sig, win_stats=None, active_trades=None):
    """
    Bố cục MỚI tối ưu cho màn hình điện thoại dọc (mỗi dòng <= ~34 ký tự, không bị Telegram
    bẻ dòng vô duyên), dùng HTML bold để tạo PHÂN CẤP thị giác:
    - LỆNH CHÍNH (entry gần giá nhất - đúng lệnh được log) nổi bật, in đậm, đủ E/SL/TP1-3
    - Các entry còn lại nén thành "Thang phụ" 1 dòng/entry (E · SL · TP1)
    - Tên lệnh chuẩn: Sell Limit / Buy Stop... thay cho "CHỜ GIÁ VƯỢT QUA" dài dòng
    - Ghi chú phụ (phiên, chống trùng...) nén thành footer 1 dòng
    """
    lines = []

    # ---------- Dòng 1-2: giá + mũi tên đa khung (nén) + điểm bối cảnh ----------
    price_line = f"⚡ XAU {sig['price']:.2f}"
    if sig["pct_change"] is not None:
        price_line += f" ({sig['pct_change']:+.2f}%)"
    price_line += f" · {sig['time']}"
    lines.append(price_line)

    arrows = sig.get("trend_arrows")
    arrow_line = ""
    if arrows:
        sym = lambda t: "↑" if t == "up" else ("↓" if t == "down" else "–")
        arrow_line = " ".join(f"{tf}{sym(t)}" for tf, t in arrows.items())
    box = sig.get("box_signal")
    ctx = box.get("ctx") if box else None
    if ctx:
        arrow_line += f" · 💪 {ctx['score']:+d} {_esc(ctx['label'])}"
    if arrow_line:
        lines.append(arrow_line)
    lines.append("")

    # ---------- Khối box ----------
    if box:
        d_icon = {"BUY": "🟢", "SELL": "🔴"}.get(box.get("direction"), "📦")
        if sig.get("confidence") == "low" and sig.get("direction"):
            d_icon = "🟡"
        broke_txt = "phá đỉnh" if box.get("confirm_dir") == "up" else "phá đáy"
        state = box["state"]

        # Tiêu đề khối (in đậm)
        if state in ("ready", "spring_ready"):
            if box["alignment"] == "spring":
                title = f"{d_icon} {box['direction']} · BOX {box['tf']} · PHÁ VỠ GIẢ ⚡"
            else:
                align_txt = "THUẬN" if box["alignment"] == "thuận" else "NGƯỢC"
                title = f"{d_icon} {box['direction']} · BOX {box['tf']} · {broke_txt} ({align_txt})"
        elif state == "spring_waiting":
            title = f"📦 BOX {box['tf']} · PHÁ VỠ GIẢ · chờ giá về biên"
        elif state == "waiting_retest":
            title = f"📦 BOX {box['tf']} · {broke_txt} · chờ retest"
        else:
            title = f"📦 BOX {box['tf']} · chưa xác nhận"
        lines.append(f"<b>{_esc(title)}</b>")

        # Dòng thông tin box nén: biên + màu nến + trạng thái retest
        color_txt = "nến 🟢" if box["color"] == "green" else "nến 🔴"
        info = f"Box {box['box_low']:.2f}–{box['box_high']:.2f} · {color_txt}"
        if state == "ready":
            info += " · retest ✅"
        lines.append(info)

        # Bối cảnh phụ nén 1 dòng: Dow + Ichimoku
        ctx_bits = []
        if box.get("dow_trend"):
            ctx_bits.append(f"📐 Dow{'↑' if box['dow_trend'] == 'up' else '↓'}")
        ichi = box.get("ichimoku")
        if ichi:
            ichi_txt = {"strong_bull": "☁️ trên mây (mạnh)", "new_bull": "☁️ trên mây (mới)",
                        "strong_bear": "☁️ dưới mây (mạnh)", "new_bear": "☁️ dưới mây (mới)",
                        "unclear": "☁️ trong mây"}[ichi["strength"]]
            ctx_bits.append(ichi_txt)
        if ctx:
            ctx_bits.append(_esc(ctx["icon_line"]))
        if ctx_bits:
            lines.append(" · ".join(ctx_bits))

        if box.get("sweep_extreme") is not None and box["alignment"] == "spring":
            lines.append(f"⚡ Quét tới {box['sweep_extreme']:.2f} - thủng = vô hiệu")

        entries = box.get("entries", [])

        def _order_name(e):
            side = "Buy" if e["direction"] == "BUY" else "Sell"
            kind = "Limit" if e.get("order_kind", "limit") == "limit" else "Stop"
            return f"{side} {kind}"

        def _entry_extras(e):
            """Ghi chú tường/cộng hưởng của 1 entry - mỗi cái 1 dòng ngắn."""
            out = []
            for wn in e.get("wall_notes", []) or []:
                out.append(_esc(wn))
            if e.get("flip_note"):
                out.append(_esc(e["flip_note"]))
            return out

        # ----- Trường hợp CÓ LỆNH (ready/spring_ready + đã chọn được entry chính) -----
        if state in ("ready", "spring_ready") and sig.get("entry") is not None and entries:
            primary = min(entries, key=lambda e: abs(e["entry"] - sig["entry"]))
            secondary = [e for e in entries if e is not primary]
            lines.append("")
            lines.append(f"<b>▶️ LỆNH CHÍNH · {_order_name(primary)}</b>")
            lines.append(f"📍 E   <b>{primary['entry']:.2f}</b>  ({_esc(primary['label'])})")
            lines.append(f"🔴 SL  {primary['sl']:.2f}  ({primary.get('sl_points', abs(primary['entry']-primary['sl'])):.1f}p)")
            lines.append(f"✅ TP  {primary['tp1']:.2f} → {primary['tp2']:.2f} → {primary['tp3']:.2f}")
            lines.append("        (1R → 1.8R → 2.8R)")
            for x in _entry_extras(primary):
                lines.append(x)
            if sig.get("rejection_candle"):
                lines.append("🕯️ Có nến từ chối M15 tại entry ✓")

            # 🎯 Vị trí entry (Premium/Discount + FVG) - chỉ hiện cho lệnh chính
            loc = sig.get("loc")
            if loc:
                lines.append("")
                lines.append(f"<b>🎯 Vị trí entry: {_esc(loc['label'])} ({loc['score']}/{loc['max']})</b>")
                for c in loc["checks"]:
                    lines.append(_esc(c))

            if secondary:
                lines.append("")
                lines.append("Thang phụ (tham khảo):")
                for e in secondary:
                    lines.append(f"· {e['entry']:.2f} {_esc(e['label'])} · SL {e['sl']:.2f} · TP1 {e['tp1']:.2f}")
                    for x in _entry_extras(e):
                        lines.append("  " + x)

        # ----- Ready nhưng KHÔNG phát lệnh (entry quá xa / chờ nến từ chối...) -----
        elif state in ("ready", "spring_ready") and entries:
            for e in entries:
                lines.append(f"· {_order_name(e)} {e['entry']:.2f} · SL {e['sl']:.2f} · TP1 {e['tp1']:.2f}")
            if sig.get("block_reason"):
                lines.append(f"⛔ {_esc(sig['block_reason'])}")

        # ----- Chờ retest / spring chờ: entries nén 1 dòng -----
        elif state in ("waiting_retest", "spring_waiting") and entries:
            for e in entries:
                lines.append(f"· {_order_name(e)} {e['entry']:.2f} · SL {e['sl']:.2f} · TP1 {e['tp1']:.2f}")
            lines.append("⏳ Chưa vào lệnh - chờ giá quay về test")

        # ----- Chưa xác nhận: giá GIỮA box = khối KẾ HOẠCH 2 KỊCH BẢN, gần biên = entry rủi ro -----
        elif state == "unconfirmed":
            if box.get("in_middle"):
                lines.append("🧭 Giá giữa box - kế hoạch 2 đầu:")
                for e in entries:
                    arrow = "⬆️ Nếu TĂNG chạm" if e["direction"] == "SELL" else "⬇️ Nếu GIẢM chạm"
                    star = " ⭐" if e.get("flip_note") else ""
                    lines.append(f"{arrow} {e['entry']:.2f}{star}:")
                    lines.append(f"   {_order_name(e)} {e['entry']:.2f} · SL {e['sl']:.2f} ({e.get('sl_points', 0):.1f}p)")
                    lines.append(f"   TP {e['tp1']:.2f} → {e['tp2']:.2f} → {e['tp3']:.2f}")
                    if e.get("flip_note"):
                        lines.append(f"   {_esc(e['flip_note'])}")
                    for wn in e.get("wall_notes", []) or []:
                        lines.append(f"   {_esc(wn)}")
                lines.append("🛡️ BE khi lãi ≥0.5R, không dời sớm hơn")
            else:
                lines.append("⚠️ Chưa xác nhận breakout - entry RỦI RO:")
                for e in entries:
                    e_icon = "🟢" if e["direction"] == "BUY" else "🔴"
                    lines.append(f"{e_icon} {_order_name(e)} {e['entry']:.2f} · SL {e['sl']:.2f} ({e.get('sl_points', 0):.1f}p)")
                    lines.append(f"   TP {e['tp1']:.2f} → {e['tp2']:.2f} → {e['tp3']:.2f}")
                    for x in _entry_extras(e):
                        lines.append("   " + x)

        # Entry bị bộ lọc SL loại - minh bạch lý do (mọi trạng thái)
        for rej in box.get("rejected_entries", []) or []:
            lines.append(f"🚫 {_esc(rej['label'])} bị loại: {_esc(rej['reason'])}")

        if sig.get("chase_warning"):
            lines.append(f"⚖️ {_esc(sig['chase_warning'])}")
        if sig.get("confidence") == "low":
            for note in sig.get("confidence_notes", []):
                lines.append(f"🚨 {_esc(note)}")
        lines.append("")
    else:
        lines.append(f"⚪ {_esc(sig['block_reason'] if sig['block_reason'] else 'Chưa tìm thấy box nào để theo dõi')}")
        lines.append("")

    # ---------- Lệnh đang chờ khớp / đang chạy ----------
    if active_trades and active_trades.get("waiting"):
        lines.append("<b>⏳ CHỜ KHỚP</b>")
        for t in active_trades["waiting"]:
            lines.append(_esc(t))
    if active_trades and active_trades.get("running"):
        lines.append("<b>✅ ĐANG CHẠY</b>")
        for t in active_trades["running"]:
            lines.append(_esc(t))
    if active_trades and (active_trades.get("waiting") or active_trades.get("running")):
        lines.append("")

    # ---------- Thống kê + vùng theo dõi (nén) ----------
    if win_stats:
        s = win_stats.get("box")
        if s:
            be_txt = f"/{s['breakevens']}BE" if s.get("breakevens") else ""
            usd_txt = f" {'+' if s['total_usd'] >= 0 else ''}${s['total_usd']}"
            lines.append(f"📊 Box: {s['wins']}W/{s['losses']}L{be_txt} ({s['win_rate']}%){usd_txt}")
        else:
            lines.append("📊 Box: chưa đủ dữ liệu")
        h4s = win_stats.get("box_h4")
        if h4s and h4s.get("total"):
            be_txt = f"/{h4s['breakevens']}BE" if h4s.get("breakevens") else ""
            usd_txt = f" {'+' if h4s['total_usd'] >= 0 else ''}${h4s['total_usd']}"
            lines.append(f"📊 Box H4: {h4s['wins']}W/{h4s['losses']}L{be_txt} ({h4s['win_rate']}%){usd_txt}")
        f = win_stats.get("fade")
        if f:
            be_txt = f"/{f['breakevens']}BE" if f.get("breakevens") else ""
            usd_txt = f" {'+' if f['total_usd'] >= 0 else ''}${f['total_usd']}"
            lines.append(f"🧭 Fade 2 đầu: {f['wins']}W/{f['losses']}L{be_txt} ({f['win_rate']}%){usd_txt}")
        if win_stats.get("ctx_compare"):
            lines.append(f"💪 {_esc(win_stats['ctx_compare'])}")
        if win_stats.get("loc_compare"):
            lines.append(f"🎯 {_esc(win_stats['loc_compare'])}")

    def _fmt_zone(z):
        tags = "+".join(z["sources"])
        return f"{z['price_low']:.2f}–{z['price_high']:.2f}({tags}{z.get('stars', '')})"

    if sig.get("zones_above"):
        lines.append("📋 ▲ " + " · ".join(_fmt_zone(z) for z in sig["zones_above"]))
    if sig.get("zones_below"):
        lines.append("📋 ▼ " + " · ".join(_fmt_zone(z) for z in sig["zones_below"]))

    # Footer: ghi chú phụ - ghi chú NGẮN gộp 1 dòng, ghi chú dài (chống trùng, nhồi lệnh...)
    # xuống dòng riêng để không bị Telegram bẻ dòng vô duyên
    if sig.get("stack_note"):
        prefix = "✅" if sig.get("stack_appended") else "🚫"
        note = _esc(sig["stack_note"])
        if len(note) <= 42:
            lines.append(f"{prefix} {note}")
        else:
            lines.append(f"{prefix} {note[:120]}")  # vẫn 1 ý, chấp nhận 2-3 dòng hiển thị nhưng là dòng riêng
    if sig.get("liquidity_note"):
        lines.append("⚠️ Ngoài phiên chính - thanh khoản thấp")

    lines.append("")
    lines.append("⚠️ Chỉ tham khảo | Quản lý vốn 1-2%")
    return "\n".join(lines)


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    r = requests.post(url, data=payload, timeout=15)
    if r.status_code != 200:
        # HTML lỗi (thẻ hở/ký tự lạ) -> gửi lại bản thuần, bỏ thẻ bold, đảm bảo tin luôn tới
        plain = message.replace("<b>", "").replace("</b>", "") \
                       .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": plain}, timeout=15)
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
        state_txt = {"ready": "SẴN SÀNG", "waiting_retest": "CHỜ RETEST", "unconfirmed": "CHƯA XÁC NHẬN",
                     "spring_ready": "PHÁ VỠ GIẢ - SẴN SÀNG", "spring_waiting": "PHÁ VỠ GIẢ - CHỜ"}.get(box["state"], box["state"])
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
    # LƯỚI ĐỠ TOÀN CỤC: bất kỳ lỗi nào (điển hình: hết quota API giữa ngày -> get_ohlc
    # raise) trước đây làm workflow chết đỏ với "exit code 1" mà không ai biết vì sao.
    # Giờ: in traceback đầy đủ vào log Actions + CỐ GẮNG báo qua Telegram (kênh bạn
    # thực sự theo dõi) + thoát mã 0 để không spam đỏ - lỗi API tạm thời sẽ tự hết ở
    # lượt chạy sau hoặc khi quota reset 00:00 UTC.
    try:
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
        # (path-aware: duyệt high/low từng nến M5 kể từ lúc tạo/khớp lệnh, không chỉ giá hiện tại)
        log = update_signal_outcomes(log, signal["df_track"], signal["price"])
        # Hủy các lệnh chờ có tiền đề đã chết (breakout bị phủ nhận / phá giả hóa phá thật)
        log = cancel_dead_premise_orders(log, signal["df_track"])

        # --- Quản lý nhồi lệnh: hòa vốn lệnh cũ đã an toàn, chỉ cho nhồi thêm khi điểm mạnh hơn rõ rệt ---
        log, should_append, stack_note = manage_active_trades_before_append(log, signal, signal["price"])
        signal["stack_note"] = stack_note
        signal["stack_appended"] = should_append

        active_trades = active_trades_summary(log, current_price=signal["price"])

        if should_append:
            log = append_signal(log, signal)
        # Log kế hoạch fade 2 đầu của box CHƯA XÁC NHẬN (mode="fade", thống kê riêng) -
        # để đánh giá khách quan setup "vùng tranh chấp phản ứng" bằng dữ liệu thật
        log, fade_appended = append_fade_plans(log, signal.get("box_signal"), signal["price"])
        if fade_appended:
            print(f"Đã log {len(fade_appended)} kế hoạch fade: {', '.join(fade_appended)}")
        save_signal_log(log)
        win_stats = {
            "box": compute_win_rate(log, mode="box"),
            "box_h4": compute_win_rate(log, mode="box_h4"),
            "fade": compute_win_rate(log, mode="fade"),
            "ctx_compare": compare_ctx_win_rate(log),
            "loc_compare": compare_loc_win_rate(log),
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
                             "unconfirmed": "Chưa xác nhận", "spring_ready": "Phá vỡ giả - sẵn sàng",
                             "spring_waiting": "Phá vỡ giả - chờ giá về biên"}.get(box["state"], box["state"])
                send_telegram_photo(chart_path, caption=f"📊 Box {box['tf']} - {state_txt}")
                print("Đã gửi ảnh biểu đồ.")
            except Exception as e:
                print(f"Lỗi khi vẽ/gửi ảnh biểu đồ (bỏ qua, không ảnh hưởng tin nhắn chính): {e}")

        print("Đã gửi xong! Kiểm tra Telegram của bạn.")
    except Exception as _fatal:
        import traceback as _tb
        print("===== BOT GẶP LỖI - TRACEBACK ĐẦY ĐỦ =====")
        _tb.print_exc()
        try:
            send_telegram(f"⚠️ Bot lỗi, lượt chạy này bị bỏ qua:\n{type(_fatal).__name__}: {str(_fatal)[:300]}")
            print("Đã báo lỗi qua Telegram.")
        except Exception:
            print("Không gửi được thông báo lỗi qua Telegram.")
        raise SystemExit(0)
