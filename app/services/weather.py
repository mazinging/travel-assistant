"""天气服务：使用免费的 wttr.in。

注意：本环境实测 wttr.in 的 HTTPS 版会 SSL 失败，HTTP 版稳定可用；天气为公开
非敏感数据，因此这里使用 http 协议。wttr.in 免费版只返回**从今天起**的 3 天预报。

⚠️ 这里的 `days` 是「向数据源索取几天」，起点永远是**今天**，不是用户的出发日。
调用方（planner._fetch_trip_weather）必须自己按行程日期做对齐，否则会把今天的
天气当成出发日的天气展示出去。
"""
from collections import Counter

import httpx

# wttr.in 免费版实际能给出的最大预报天数
MAX_FORECAST_DAYS = 3


def get_weather(city: str, days: int = 3) -> list[dict]:
    """取某城市从今天起的未来 `days` 天天气（不超过 MAX_FORECAST_DAYS）。

    返回的每一项都带 `date`（YYYY-MM-DD）字段，供调用方按行程日期筛选。
    """
    url = f"http://wttr.in/{city}"
    try:
        resp = httpx.get(
            url,
            params={"format": "j1", "lang": "zh"},
            headers={"User-Agent": "curl/8.0"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        raise RuntimeError(f"天气查询失败: {e}") from e

    weather = data.get("weather", [])
    result = []
    for item in weather[: max(min(days, MAX_FORECAST_DAYS), 1)]:
        result.append(_format_day(item))
    return result


def _format_day(item: dict) -> dict:
    """把 wttr.in 单日数据压缩成对 LLM 友好的字段。"""
    hourly = item.get("hourly", [])
    # 只关注白天（6:00 ~ 20:00）的天气，用于判断是否适合户外
    day_hours = [h for h in hourly if 600 <= int(h.get("time", "0") or 0) <= 2000]

    descs = []
    rain_chances = []
    for h in day_hours or hourly:
        wd = h.get("weatherDesc", [{}])
        if isinstance(wd, list) and wd:
            descs.append(wd[0].get("value", ""))
        rain_chances.append(int(h.get("chanceofrain", "0") or 0))

    dominant = _dominant_desc(descs)
    max_rain = max(rain_chances) if rain_chances else 0

    return {
        "date": item.get("date", ""),
        "max_temp": item.get("maxtempC", ""),
        "min_temp": item.get("mintempC", ""),
        "day_condition": dominant,
        "rain_chance": max_rain,
    }


def _dominant_desc(descs: list[str]) -> str:
    """把白天的逐小时天气描述归类，并取「出现次数最多」的类别作为当天主导天气。

    之所以统计主导，而不是「只要某个小时有雨就判整天雨」：
    因为一天里可能「上午零星小雨、下午放晴」，整天下雨和局部小雨体验完全不同。
    """
    if not descs:
        return "未知"
    cats = [c for c in (_classify(d) for d in descs) if c != "未知"]
    if not cats:
        return max(set(descs), key=descs.count)
    return Counter(cats).most_common(1)[0][0]


def _classify(desc: str) -> str:
    """把单个天气描述归到「雨雪/雨/阴天/多云/晴」之一。"""
    d = desc.lower()
    # 雪：只有明确提到 snow/雪 才算「雨雪」
    if any(k in d for k in ("雪", "snow")):
        return "雨雪"
    # 雨：rain/shower/drizzle 等都是「雨」，不是「雨雪」
    if any(k in d for k in ("雷", "雨", "rain", "shower", "drizzle")):
        return "雨"
    # 阴/雾霾：overcast 阴、haze 霾、fog/mist 雾
    if any(k in d for k in ("阴", "雾", "霾", "overcast", "haze", "fog", "mist")):
        return "阴天"
    # 多云
    if any(k in d for k in ("多云", "云", "cloudy", "partly")):
        return "多云"
    # 晴
    if any(k in d for k in ("晴", "sunny", "clear")):
        return "晴"
    return "未知"
