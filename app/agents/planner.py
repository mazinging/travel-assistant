"""旅行规划 Agent（ReAct / reasoning + action 范式）。

与之前「固定流水线」不同，这里把各个数据源包装成「工具」交给 DeepSeek，
让模型通过 function calling 自主决定查什么、按什么顺序查，跑一个
「思考(Thought) → 行动(Action) → 观察(Observation) → 再思考」的循环，
直到信息足够后输出最终方案。

整体流程：
  1. 计算日期，构造 system + user 消息（只给基本信息和个人需求，不预查数据）
  2. ReAct 循环：模型输出 tool_calls → 执行工具 → 结果作为 Observation 回填 → 继续
  3. 模型认为信息足够、停止调用工具时，输出最终 JSON 方案
  4. Python 补上天气与元信息后返回
"""
import asyncio
import json
import logging
import re
import time
from datetime import date, timedelta

from app import config
from app.agents import trace as trace_mod
from app.services import city, deepseek, tavily, weather
from app.services.tuniu import TuniuClient

logger = logging.getLogger(__name__)


# 途牛车次价格字段 -> 中文席别
SEAT_NAME_MAP = {
    "edzPrice": "二等座",
    "ydzPrice": "一等座",
    "swzPrice": "商务座",
    "tdzPrice": "特等座",
    "yzPrice": "硬座",
    "ywPrice": "硬卧",
    "rwPrice": "软卧",
    "gjrwPrice": "高级软卧",
    "wzPrice": "无座",
    "dwPrice": "动卧",
    "ydwPrice": "一等卧",
    "edwPrice": "二等卧",
}


# 门票查询缓存：避免同一景点在 ReAct 循环里被重复查询而浪费限流额度
_ticket_cache: dict[str, str] = {}


# ---- 工具定义：把数据源包装成 LLM 可调用的 function ----
# 这是 OpenAI function calling 的标准格式（DeepSeek 完全兼容）：
# 每个工具用 JSON Schema 声明「名字、用途、参数」，模型据此生成结构化的调用请求。
# 模型不会直接操作数据源，它只负责「决定调哪个工具、传什么参数」，
# 真正的数据查询由 _execute_tool 完成——这就是「模型负责思考、代码负责执行」的分工。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_trains",
            "description": "查询两个城市之间某日期的火车票车次，返回车次号、出发/到达站、发到时间、历时、各席别价格。",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure_city": {"type": "string", "description": "出发城市名，如 合肥"},
                    "arrival_city": {"type": "string", "description": "到达城市名，如 苏州"},
                    "date": {"type": "string", "description": "出发日期 YYYY-MM-DD"},
                },
                "required": ["departure_city", "arrival_city", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "查询某城市的酒店，返回酒店名、价格、评分、地址、商圈。若用户指定价格范围，务必用 prices 参数（如 0-200）过滤。注意：返回的 price_per_night_from 是该酒店最低房型的单晚起步价，不是成交价，方案中必须注明「起价，实际以预订页为准」。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "check_in": {"type": "string", "description": "入住日期 YYYY-MM-DD"},
                    "check_out": {"type": "string", "description": "离店日期 YYYY-MM-DD"},
                    "prices": {"type": "string", "description": "价格区间，如 0-200，可选"},
                },
                "required": ["city", "check_in", "check_out"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tickets",
            "description": "查询某个景点的门票价格，返回门票名称与价格。",
            "parameters": {
                "type": "object",
                "properties": {
                    "scenic_name": {"type": "string", "description": "景点名称，如 拙政园"},
                },
                "required": ["scenic_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某城市未来几天的天气（温度、天气状况、降水概率）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "days": {"type": "integer", "description": "未来天数，默认3"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description": "搜索某城市的必去景点和旅游攻略，返回景点介绍文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，如 苏州 必去景点"},
                },
                "required": ["query"],
            },
        },
    },
]


# system prompt 是 ReAct 范式的「指挥棒」：
# 它告诉模型三件事——能调什么工具、按什么规则推理、最终输出什么格式。
# 所有「硬约束」都写在这里（而不是写死在代码里），这是 ReAct 与流水线的本质区别：
# 改需求只需改这段文字，不用改执行流程。
SYSTEM_PROMPT = """你是一名专业的中国国内旅行规划助手。你可以调用工具查询实时数据（车次、酒店、门票、天气、景点），然后为用户规划一次完整的旅行。

你的工作方式是「推理 + 行动」的循环：
1. 先分析用户需求，确定需要查询哪些数据
2. 按需调用工具（可以多次、每次可同时调用多个工具），根据返回结果继续分析
3. 信息足够后，停止调用工具，直接输出最终方案 JSON

硬约束（必须严格遵守，不得违背）：
- 用户明确提出的需求（出发时间、上车站点、酒店价格上限等）是硬约束，任何方案都不得违背，绝不能为了省钱或省时而牺牲用户要求
- 若用户提到酒店价格，调用 search_hotels 时必须用 prices 参数过滤
- 车次、酒店、门票的价格必须来自工具返回的真实数据，不得虚构
- 酒店价格字段是 price_per_night_from（单晚**起价**，通常是最低房型），不是成交价：计算 hotel.total 时照常用它 × nights，但必须在 reason 里注明「酒店为起价，实际以预订页为准」
- 车次的终点/起点站必须是目的地城市的主要车站，注意区分下辖县市（例如目的地是「苏州」时，张家港、常熟、昆山等站不可作为苏州的终点）
- 中转组合必须保证时间衔接：第二段发车时间要晚于第一段到达时间并留出换乘时间，绝不能出现时间倒挂
- 如果用户指定了出发时间（如「9点左右」），所有方案的去程发车时间都必须落在该时段，凌晨（0:00-6:00）、深夜（22:00-24:00）的车次绝对禁止
- 门票查询接口有调用频率限制：只查询最终方案真正会用到的 3-4 个关键景点即可，不要一次查询过多；若门票查询返回限流错误，可用 search_attractions 返回文本里提到的价格作为参考，并在方案的 reason 中注明「门票价格为参考价」
- 若「基本信息」里写明酒店住 0 晚（1 天当天往返），则 hotel 字段填 {"name": "无需住宿", "price_per_night": 0, "nights": 0, "total": 0}，不要虚构酒店，total_cost 里也就不会有酒店费用
- **若「基本信息」里出现【市内游】标记**，说明出发地和目的地是同一座城市：不要调用 search_trains 查城际车次（查不到且浪费时间），outbound 和 return 都填 {"mode": "市内游", "detail": "市内游，无需城际交通", "price": 0}，规划重点放在市内景点串联与路线安排上

最终输出要求：
当信息足够时，停止调用工具，输出一个 JSON 对象（不要输出 结Markdown 或解释），构如下：
{"plans": [{"name": "方案名", "outbound": {"mode": "直达或中转", "detail": "车次+站点+发到时间+席别", "price": 数字}, "return": {"mode": "...", "detail": "...", "price": 数字}, "hotel": {"name": "酒店名", "price_per_night": 数字, "nights": 数字, "total": 数字}, "tickets": [{"spot": "景点名", "price": 数字}], "itinerary": [{"day": 1, "theme": "主题", "spots": ["景点"], "plan": "当天安排"}], "total_cost": 数字, "reason": "推荐理由"}], "tips": "整体出行建议"}

要求：
- 生成 2-3 份方案，每份自洽（车次、酒店、门票、路线、总价互不矛盾）
- total_cost = 去程 price + 回程 price + hotel.total + 所有门票 price 之和
- 路线按天规划，结合天气（雨天优先室内、晴天优先户外）
- 若用户指定了中转城市，可提供含中转的方案并说明利弊
"""


CHAT_SYSTEM_PROMPT = """你是一名旅行咨询助手。你可以调用工具查询实时数据（车次、酒店、门票、天气、景点），然后回答用户的问题。

工作方式：
1. 理解用户的问题，判断需要查询哪些数据
2. 按需调用工具获取真实数据（可以多次调用，也可以一次调用多个）
3. 基于查到的真实数据回答用户

要求：
- 回答必须基于工具返回的真实数据，不得编造价格、车次、天气等信息
- 用简体中文简洁、直接地回答
- 如果用户的问题不需要查数据，可以直接回答
- 回答时尽量标注关键信息（如价格、日期、车次号）
"""


class CityResolutionNeeded(Exception):
    """输入的是省份而非具体城市，需要用户先选一个城市。

    这不是「错误」——用户很可能只是没说清楚（"我想去江苏玩"）。所以它单独成一个
    异常类型，由 API 层转成 **409 Conflict + 候选城市列表**，而不是 400 报错。
    前端据此弹出城市选择，用户点一下即可继续，不需要重新填整个表单。
    """

    def __init__(self, field: str, province: str, cities: list[str]):
        self.field = field            # 出问题的是哪个字段：departure_city / arrival_city / transfer_city
        self.province = province
        self.cities = cities
        super().__init__(f"{field} 是省份「{province}」，请选择具体城市")


class InvalidCityError(ValueError):
    """城市名无法识别（乱码、错别字、空值）。

    继承 ValueError，这样 main.py 里已有的「ValueError → 400」映射会自动生效，
    前端能拿到明确的中文提示，而不是一个 HTTP 500。
    """

    def __init__(self, field: str, value: str, reason: str):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(reason)


# 字段名 -> 中文标签，用于拼错误提示
_FIELD_LABELS = {
    "departure_city": "所在城市",
    "arrival_city": "目的地城市",
    "transfer_city": "中转城市",
}


def _detect_same_city(departure_city: str, arrival_city: str) -> bool:
    """判断是否是「市内游」（出发地与目的地相同）。

    以前这种情况**完全没被处理**，会一路走到 ReAct 里，产生一串连锁问题：

      1. 模型按指令去查「南京→南京」的车次，途牛返回空列表
      2. 空列表不是异常，所以 `_execute_tool` 正常返回 `[]`（没有任何错误信号）
      3. 模型拿到空数据，但**必须交差**——它开始硬凑方案
      4. `_fix_total_cost` 因为「车票数据缺失」按设计跳过
      5. `_check_itinerary` 因为「天数对得上」通过
      6. 用户拿到一份没有任何真实交通依据、但看起来正常的方案

    **每一环单独看都对，合起来产出了坏结果。** 这就是典型的「沉默错误链」。

    现在的处理方式：不报错（市内游是合理需求），而是**改变告知模型的内容**——
    见 _plan_async 里对 user 消息的分支。模型知道这是市内游后，
    就不会去查城际车次，而是直接规划市内行程。
    """
    return departure_city == arrival_city


class InvalidTripError(ValueError):
    """输入组合起来不成立（如中转城市等于出发地、日期已过去）。

    继承 ValueError，这样 main.py 里「ValueError → 400」的映射自动生效。
    与 InvalidCityError 分开定义，是因为它们的语义不同：
    前者是"这个字段填错了"，后者是"这些字段组合起来没意义"。

    定义在 _validate_trip_inputs 之前——虽然运行时查找能容忍后置定义，
    但把异常类放在使用它的函数之前更易读，也避免后续编辑时踩坑。
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _validate_trip_inputs(
    departure_city: str,
    arrival_city: str,
    transfer_city: str | None,
    departure_date: str,
) -> None:
    """校验**字段之间的关系**是否自洽（单字段校验抓不到的那一类）。

    为什么需要单独一层：`_validate_cities` 只回答「南京合法吗」，
    它回答不了「南京→南京有意义吗」。这类「格式全对、语义矛盾」的输入
    是 agent 项目里最隐蔽的问题来源——因为每一道单字段校验都会放行。

    注意：`出发地 == 目的地` 在这里**不报错**（市内游是合理需求），
    真正被拦下的是另外两种**逻辑上无解**的组合。
    """
    # 中转城市不能等于出发地或目的地。
    # 「从南京出发，经南京中转」在逻辑上无解——不是用户想要的结果，
    # 而是填错了。这种情况没有合理的降级方式，只能拦下。
    if transfer_city and transfer_city in (departure_city, arrival_city):
        raise InvalidTripError(
            f"中转城市不能是出发地或目的地（当前填的是「{transfer_city}」）。"
            f"请换一个中转城市，或清空该栏。"
        )

    # 出发日期不能在过去：查不到任何车次，只会让模型硬凑方案。
    try:
        dep = date.fromisoformat(departure_date)
    except ValueError:
        return  # 格式错误交给 _plan_async 里的原有逻辑处理，避免重复报错
    if dep < date.today():
        raise InvalidTripError(
            f"出发日期（{departure_date}）已经过去了，请选择今天或之后的日期。"
        )


def _validate_cities(
    departure_city: str, arrival_city: str, transfer_city: str | None
) -> tuple[str, str, str | None]:
    """在任何网络请求之前，校验三个城市字段，并返回**归一化后**的城市名。

    为什么必须前置：以前乱码城市会一路穿透到 ReAct 循环——途牛返回空结果、
    模型在「没有任何真实数据」的情况下硬凑方案，或者烧完 24 步才失败报 500。
    用户看到的是「途牛接口返回 HTTP 500」这类和他输入毫无关系的报错。

    现在挡在最前面：无效立刻给出「城市名无法识别」；省份则抛出
    CityResolutionNeeded 让前端引导选择。**成本接近零，且不消耗任何 API 配额。**

    返回归一化后的三元组（例如「苏州市」→「苏州」），调用方必须使用返回值，
    否则下游仍会拿到带后缀的原始输入。
    """
    normalized: dict[str, str | None] = {}
    for field, value in (
        ("departure_city", departure_city),
        ("arrival_city", arrival_city),
        ("transfer_city", transfer_city),
    ):
        if field == "transfer_city" and not value:
            normalized[field] = None
            continue  # 中转城市是选填，留空合法
        result = city.resolve(value or "")
        if result["status"] == "invalid":
            raise InvalidCityError(
                field,
                value or "",
                f"{_FIELD_LABELS[field]}「{value}」无法识别：{result['reason']}",
            )
        if result["status"] == "province":
            raise CityResolutionNeeded(field, result["province"], result["cities"])
        normalized[field] = result["city"]

    return (
        normalized["departure_city"] or "",
        normalized["arrival_city"] or "",
        normalized["transfer_city"],
    )


def plan_trip(
    departure_city: str,
    arrival_city: str,
    departure_date: str,
    days: int,
    transfer_city: str | None = None,
    requirements: str = "",
) -> dict:
    """同步入口：内部用 asyncio.run 驱动 ReAct 循环。"""
    return asyncio.run(
        _plan_async(
            departure_city,
            arrival_city,
            departure_date,
            days,
            transfer_city,
            requirements,
        )
    )


async def _plan_async(
    departure_city: str,
    arrival_city: str,
    departure_date: str,
    days: int,
    transfer_city: str | None,
    requirements: str,
) -> dict:
    """旅行规划的异步主流程：准备上下文 → 跑 ReAct 循环 → 组装返回。

    注意这里不再像旧版那样「代码主动去查数据」，而是：
      1. 算好日期、把基本信息 + 个人需求拼成 user 消息
      2. 交给 _react_loop，让模型自己决定查什么、怎么查
      3. 循环结束后，只补上「前端展示需要」的天气和元信息
    """
    # ---- 前置校验：三个密钥必须齐全 ----
    missing = config.ensure_keys()
    if missing:
        raise RuntimeError(f"缺少配置: {', '.join(missing)}，请填写 .env 文件")

    # ---- 前置校验：城市名（必须早于任何网络请求）----
    # 这一步把「乱码城市」「输入省份」这两类输入挡在 ReAct 循环之外。
    # 归一化后的城市名必须用于后续所有查询，否则「苏州市」这类输入会带着后缀
    # 传给数据源，导致查不到结果。
    departure_city, arrival_city, transfer_city = _validate_cities(
        departure_city, arrival_city, transfer_city
    )
    # 关系校验：单字段校验抓不到的那一类（中转城市冲突、日期已过去）
    _validate_trip_inputs(departure_city, arrival_city, transfer_city, departure_date)
    same_city = _detect_same_city(departure_city, arrival_city)

    tuniu = TuniuClient(config.TUNIU_API_KEY)

    # ---- 日期计算 ----
    # 约定：玩 N 天 → 住 N-1 晚，回程日 = 出发日 + (N-1)，即最后一天返程。
    # 例：9-22 出发玩 2 天 → 9-22 入住、9-23 退房并回程。
    #
    # 注意 check_out 与 hotel_nights 必须「同源」：两者都由 hotel_nights 推出。
    # 早期版本写成 `check_out = 出发日 + max(days-1, 1)`，在 days=1 时
    # hotel_nights=0 但 check_out=出发日+1，等于告诉用户「住 0 晚」却拿着
    # 「住 1 晚」的日期去查房价，模型的输入自相矛盾，总价也就不可信。
    try:
        dep = date.fromisoformat(departure_date)
    except ValueError as e:
        raise ValueError(f"出发日期格式错误，应为 YYYY-MM-DD: {departure_date}") from e

    # 「玩几天」与「住几晚」是两个独立概念，用常量显式表达，避免再出现下界不一致
    nights_of_stay = max(days - 1, 0)   # 住宿晚数：1 天行程 = 当天往返，不住
    hotel_nights = nights_of_stay
    check_in = dep.isoformat()
    # 住 0 晚时 check_out 必须等于 check_in，否则数据源会按 1 晚计价
    check_out = (dep + timedelta(days=nights_of_stay)).isoformat()
    return_date = (dep + timedelta(days=max(days - 1, 0))).isoformat()

    # ---- 构造 user 消息 ----
    # 只给「结构化信息 + 个人需求」，不预查任何数据——查什么由模型自主决定。
    req_text = requirements.strip() if requirements and requirements.strip() else "无特殊要求"
    transfer_hint = (
        f"用户指定了中转城市：{transfer_city.strip()}，可考虑提供含中转的方案。"
        if transfer_city
        else ""
    )

    # 市内游是**另一种场景**，不能只靠"放行"来处理——必须明确告诉模型
    # "这次不用查城际交通"。否则模型仍会去查"南京→南京"的车次，拿到空结果后硬凑。
    # 这里把差异写进 user 消息，是成本最低、最可靠的做法（不需要改任何工具）。
    if same_city:
        trip_hint = (
            f"⚠️ 本次是【市内游】：出发地和目的地都是 {arrival_city}，"
            f"不需要查询城际火车票，outbound 和 return 的 price 都填 0，"
            f"detail 写明「市内游，无需城际交通」。请把重点放在市内景点、"
            f"路线安排和（如需要的）当地酒店上。"
        )
    else:
        trip_hint = ""

    user = f"""
【基本信息】
出发城市：{departure_city}
目的地城市：{arrival_city}
出发日期：{departure_date}
回程日期：{return_date}
旅行天数：{days} 天（酒店住 {hotel_nights} 晚）

【个人需求（硬约束，必须遵守）】
{req_text}
{transfer_hint}
{trip_hint}

请先按需调用工具查询数据（车次、酒店、天气、景点、门票），信息足够后输出最终方案 JSON。
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    # ---- 跑 ReAct 循环（json_mode=True：要求模型最终输出 JSON 方案） ----
    # 传入轨迹收集器：记录每轮 token、耗时、工具调用与失败情况。
    # 循环结束后放进 _meta.trace，前端可折叠查看。
    trace = trace_mod.Trace()
    result = await _react_loop(messages, tuniu, json_mode=True, trace=trace)
    trace.finished = True

    # ---- 后处理 ----
    # warnings 是本函数贯穿始终的「已知问题」收集器：任何「方案还能用、但有
    # 已知缺陷」的情况都往这里写一条，最后通过 _meta.warnings 透传到前端展示。
    # 这样用户看到的不再是一个静默缺失的区块，而是明确的说明。
    warnings: list[str] = []

    # ---- 关键：校验并修正模型输出 ----
    # 模型负责「编排」，但算数和时序是确定性任务，不该交给概率模型。
    # 这里用代码重算总价、校验时间线，把错误拦在返回给用户之前。
    plans = result.get("plans")
    if isinstance(plans, list):
        for idx, plan in enumerate(plans):
            if not isinstance(plan, dict):
                continue
            warnings.extend(_fix_total_cost(plan, idx))
            warnings.extend(_check_itinerary(plan, idx, days, dep, return_date))

    # 1 天行程没有住宿，但方案结构里硬性要求 hotel 字段，这里提前说明，
    # 避免模型被 `hotel.nights = 0` 和「总价里要不要算酒店」搞糊涂。
    if hotel_nights == 0:
        warnings.append("行程为 1 天（当天往返），未安排住宿，总价不含酒店费用")

    # 市内游：必须让用户知道系统理解成了什么。
    # 这很关键——如果用户其实是填错了，这条提示就是唯一的线索；
    # 如果他确实想市内游，这条提示让他知道"不用等车票了"。
    if same_city:
        warnings.append(
            f"出发地与目的地相同（{arrival_city}），已按【市内游】规划："
            f"不包含城际交通费用。若这不是你的本意，请修改出发地或目的地。"
        )

    # 天气这里单独再查一次：模型循环里查的天气是给它「规划路线」用的，
    # 但前端要展示一份准确、结构化的天气，所以由 Python 确定性补一份。
    weather_list, weather_warnings = await _fetch_trip_weather(arrival_city, dep, days)
    warnings.extend(weather_warnings)

    result["weather"] = weather_list
    # 轨迹汇总：把最终上下文规模补进去（这是外部的信息，Trace 自己不知道）
    trace_data = trace.to_dict()
    trace_data["summary"]["final_message_count"] = len(messages)
    result["_meta"] = {
        "departure_city": departure_city,
        "arrival_city": arrival_city,
        "departure_date": departure_date,
        "return_date": return_date,
        "days": days,
        "hotel_nights": hotel_nights,
        "check_in": check_in,
        "check_out": check_out,
        "transfer_city": transfer_city,
        "warnings": warnings,
        "trace": trace_data,
    }
    return result


def _fix_total_cost(plan: dict, idx: int) -> list[str]:
    """用代码重算 total_cost，而不是相信模型算的加法。

    为什么要重算：模型做多步算术很不可靠，尤其是「往返 + 酒店 × 晚数 + 多张门票」
    这种带乘法的组合。用户反馈的「总花销计算错误」正是这个原因——**模型不是不会
    加，而是它每轮都在重新"估算"这些数字。** 加法本身没有歧义，就该用代码做。

    策略：能算出来就**直接覆盖**并记一条 warning（让用户知道修正过），
    数据缺失严重（连车票钱都没有）则不动它，避免算出一个更误导的数字。
    """
    parts: dict[str, float] = {}
    for key, label in (("outbound", "去程"), ("return", "回程")):
        seg = plan.get(key)
        if isinstance(seg, dict):
            val = _to_float(seg.get("price"))
            if val is not None:
                parts[label] = val

    hotel = plan.get("hotel")
    if isinstance(hotel, dict):
        val = _to_float(hotel.get("total"))
        if val is not None:
            parts["酒店"] = val

    ticket_sum = 0.0
    tickets = plan.get("tickets")
    if isinstance(tickets, list):
        for t in tickets:
            if isinstance(t, dict):
                val = _to_float(t.get("price"))
                if val is not None:
                    ticket_sum += val
    parts["门票"] = ticket_sum

    # 至少要拿到车票钱才重算，否则数据缺失严重，算出来反而误导
    if "去程" not in parts and "回程" not in parts:
        return []

    expected = round(sum(parts.values()), 2)
    actual = _to_float(plan.get("total_cost"))
    name = plan.get("name") or f"方案{idx + 1}"

    if actual is None:
        plan["total_cost"] = expected
        return [f"「{name}」总价缺失，已按各部分金额重算为 ¥{expected:.0f}"]

    if abs(actual - expected) > 1:
        plan["total_cost"] = expected
        detail = " + ".join(f"{k}¥{v:.0f}" for k, v in parts.items() if v)
        return [
            f"「{name}」总价原为 ¥{actual:.0f}，与明细不符（{detail} = ¥{expected:.0f}），已修正"
        ]
    return []


def _check_itinerary(
    plan: dict, idx: int, days: int, dep: date, return_date: str
) -> list[str]:
    """校验行程天数与时间线是否自洽。

    用户反馈的「双城游玩攻略时间线对不上」，根因是**没有任何代码检查过 itinerary**：
    模型可能排出 5 天的行程（而用户只玩 3 天）、把日期写成不存在的第 0 天、
    或者让回程时间早于去程。这些错误在页面上看起来只是「有点怪」，
    用户却要自己对照才能发现。

    这里做三项确定性检查（都能用代码判定，不需要模型判断）：
      1. 行程天数是否与用户要求一致
      2. day 序号是否连续（1..days）
      3. 回程/去程时间是否倒挂，以及 itinerary 里的日期是否越界
    """
    issues: list[str] = []
    name = plan.get("name") or f"方案{idx + 1}"

    itinerary = plan.get("itinerary")
    if isinstance(itinerary, list) and itinerary:
        if len(itinerary) != days:
            issues.append(
                f"「{name}」行程安排了 {len(itinerary)} 天，与要求的 {days} 天不一致"
            )
        # day 序号连续性：模型偶尔会输出 0 或跳号
        day_nums = [
            d.get("day") for d in itinerary
            if isinstance(d, dict) and isinstance(d.get("day"), int)
        ]
        if day_nums and sorted(day_nums) != list(range(1, len(day_nums) + 1)):
            issues.append(f"「{name}」行程的第几天序号不连续：{sorted(day_nums)}")

    # 回程不得早于去程（时间倒挂）
    out_t = _extract_first_time(plan.get("outbound"))
    ret_t = _extract_first_time(plan.get("return"))
    if out_t and ret_t and ret_t < out_t:
        issues.append(f"「{name}」回程发车时间（{ret_t}）早于去程（{out_t}），存在时间倒挂")

    # 若 itinerary 里带了日期，必须落在 [出发日, 回程日] 内
    for d in itinerary if isinstance(itinerary, list) else []:
        if not isinstance(d, dict):
            continue
        ddate = str(d.get("date") or "")
        if len(ddate) == 10 and ddate[:4].isdigit():
            if not (dep.isoformat() <= ddate <= return_date):
                issues.append(
                    f"「{name}」第 {d.get('day')} 天的日期 {ddate} 超出行程范围"
                    f"（{dep.isoformat()} ~ {return_date}）"
                )
                break  # 只报一次，避免刷屏
    return issues


_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def _extract_first_time(segment) -> str | None:
    """从车次描述文本里抽出第一个 HH:MM 时间，抽不到返回 None。

    模型的 `detail` 是自由文本（如 "G1234 苏州北 09:15 → 合肥南 12:30"），
    没有结构化字段可用，所以只能正则提取。这是「模型输出不可信」的典型场景：
    我们**不解析它的语义，只提取可验证的片段**。
    """
    if not isinstance(segment, dict):
        return None
    detail = str(segment.get("detail") or "")
    m = _TIME_RE.search(detail)
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None


async def _fetch_trip_weather(
    arrival_city: str, dep: date, days: int
) -> tuple[list[dict], list[str]]:
    """查目的地天气，并**按行程日期对齐**后返回。

    为什么必须对齐：wttr.in 的 `j1` 格式返回的是「从今天起」的若干天，它并不知道
    用户的出发日期。用户通常提前几天到几周做规划，若直接把返回值贴到页面上，
    展示的是**今天的天气**而不是出发日的天气——数据没错，但和行程完全无关。

    因此这里按 days 参数取回原始预报后，用 `_format_day` 里已经解析好的 `date`
    字段与行程日期求交集；对不上的日期既不展示也不编造，改为产出一条 warning。

    返回 (对齐后的天气列表, 警告列表)；任何失败都不抛异常，而是降级为 warning。
    """
    try:
        raw = await asyncio.to_thread(weather.get_weather, arrival_city, days)
    except Exception as e:  # noqa: BLE001
        # 这里以前是 `except Exception: pass`，异常被彻底吞掉：
        # 服务端没有任何痕迹，前端也只是整块天气消失。现在改为降级 + 告警。
        logger.warning(
            "weather fetch failed city=%s days=%s err=%s: %s",
            arrival_city, days, type(e).__name__, e,
        )
        return [], [f"天气查询失败（{type(e).__name__}），方案中未含天气预报"]

    # 行程覆盖的日期集合，例：9-22 出发 3 天 → {9-22, 9-23, 9-24}
    trip_dates = {(dep + timedelta(days=i)).isoformat() for i in range(days)}
    by_date = {w.get("date"): w for w in raw if w.get("date")}
    matched = [by_date[d] for d in sorted(trip_dates) if d in by_date]

    warnings: list[str] = []
    missing = len(trip_dates) - len(matched)
    if missing > 0:
        # 区分两种成因，否则措辞会把用户引向错误的理解：
        #   a) 出发日就在近期，但行程比预报窗口长 → 是「行程太长」
        #   b) 出发日远在预报窗口之外 → 是「出发日太远」，和行程长短无关
        today = date.today()
        offset = (dep - today).days
        if offset >= len(raw):
            reason = f"出发日（{dep.isoformat()}）超出天气预报范围，届时请重新查询"
        else:
            reason = f"行程中另有 {missing} 天超出预报范围"
        warnings.append(f"暂无完整天气数据：{reason}（免费源 wttr.in 仅提供近期短期预报）")
    return matched, warnings


def chat(message: str) -> dict:
    """自由问答入口：用户问任意问题，模型自主决定查什么工具来回答。

    与 plan_trip 共用同一个 ReAct 引擎（_react_loop + TOOLS），
    只是换了 system prompt（咨询助手角色）和输出方式（自然语言而非 JSON）。
    """
    return asyncio.run(_chat_async(message))


async def _chat_async(message: str) -> dict:
    # 和 _plan_async 结构一致：构造消息 → 跑 ReAct 循环，但 json_mode=False。
    missing = config.ensure_keys()
    if missing:
        raise RuntimeError(f"缺少配置: {', '.join(missing)}，请填写 .env 文件")
    tuniu = TuniuClient(config.TUNIU_API_KEY)
    # 注入当前日期，让模型能正确换算「今天/明天/后天」等相对日期
    today = date.today().isoformat()
    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": f"今天是 {today}。\n\n{message}"},
    ]
    return await _react_loop(messages, tuniu, json_mode=False)


async def _react_loop(
    messages: list[dict],
    tuniu: TuniuClient,
    max_steps: int = 24,
    json_mode: bool = False,
    total_budget: float = 180.0,
    max_empty_retries: int = 3,
    trace: "trace_mod.Trace | None" = None,
) -> dict:
    """ReAct 主循环：整个 Agent 的心脏。

    核心机制是「反复调模型，直到它不再要调工具」：
        调模型 → 模型说"我要调这些工具" → 执行工具 → 把结果喂回模型 → 再调模型
    每一轮模型都基于上一轮的工具结果继续推理，直到它认为信息够了、直接给答案。

    这里体现的是 function calling 的对话约定（OpenAI 兼容格式）：
      - 当模型想调用工具时，返回的 message 里带 tool_calls 数组，
        此时必须把它作为一条 assistant 消息追加回 messages
      - 每个 tool_call 都要对应一条 role="tool" 的消息，
        用 tool_call_id 把「调用」和「结果」关联起来
      - 模型看到 tool 结果后，下一轮才会基于它继续推理
    这样 messages 越滚越长，形成「思考→行动→观察→再思考」的轨迹。

    参数：
      json_mode=True  → 最终 content 是 JSON（旅行规划），解析成 dict 返回
      json_mode=False → 最终 content 是自然语言（自由问答），包成 {"answer":...} 返回
      max_steps       → 步数安全阀：防止模型无限调用工具导致死循环
      total_budget    → **总时长预算（秒）**。只有步数上限是不够的：24 步 × 90s
                        的 LLM 超时 + 工具耗时，最坏情况用户要等半小时以上。这里
                        用 asyncio.timeout 兜住整轮循环的墙上时间，超时抛
                        TimeoutError，由上层转成 504 友好提示。
      max_empty_retries → 空响应重试的**独立**上限。空响应重试必须和工具调用分开
                        计数：否则模型若陷入「反复返回空 content」，会把 max_steps
                        全部烧在空转上（24 次 LLM 调用，只花钱不产出）。
      trace           → 轨迹收集器（app/agents/trace.py）。传入后会记录每轮的
                        token 用量、耗时、工具调用与失败情况。
                        **这是观察 agent 内部行为的唯一窗口** —— 读代码只能知道
                        "可能的路径"，只有轨迹能告诉你"实际走了哪条"。
                        为 None 时不记录，行为与加轨迹前完全一致。
    """
    empty_retries = 0
    try:
        async with asyncio.timeout(total_budget):
            for step in range(max_steps):
                # ---- 第 1 步：调模型（带上完整对话历史 + 工具定义） ----
                # 用 with_usage=True 拿到 token 用量 —— 这是轨迹里最有价值的观测项
                t0 = time.monotonic()
                message, usage = await asyncio.to_thread(
                    deepseek.chat_with_tools,
                    messages, config.DEEPSEEK_API_KEY, TOOLS, True,
                )
                llm_ms = int((time.monotonic() - t0) * 1000)
                tool_calls = message.get("tool_calls")

                # 记录本轮 LLM 调用。注意 prompt_tokens 是"本次请求的输入大小"，
                # 因为每轮都重发全部历史，这个数会持续增长 —— 连续看它就是
                # 在直接观察 ReAct 的成本曲线。
                if trace is not None:
                    trace.add_step(trace_mod.Step(
                        step=step,
                        prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        completion_tokens=int(usage.get("completion_tokens") or 0),
                        tool_call_count=len(tool_calls or []),
                        elapsed_ms=llm_ms,
                    ))

                # ---- 第 2 步：判断是否结束 ----
                # 模型没有再返回 tool_calls，说明它认为信息已经够了，
                # 此时 content 里就是最终答案（JSON 方案 或 自然语言回答）。
                if not tool_calls:
                    content = message.get("content") or ""
                    # 模型偶尔会返回「空内容且不调工具」的异常状态（比如还在思考却被截断），
                    # 这时不能当作最终答案，而是追加一句提示让模型继续，回到循环重试。
                    if not content.strip():
                        empty_retries += 1
                        if trace is not None and trace.steps:
                            trace.steps[-1].empty_retry = True
                        if empty_retries > max_empty_retries:
                            raise RuntimeError(
                                f"DeepSeek 连续 {max_empty_retries} 次返回空响应，已中止"
                            )
                        logger.warning(
                            "empty response from LLM, retry %s/%s",
                            empty_retries, max_empty_retries,
                        )
                        # 统一写 content or ""：模型可能返回 content=null，
                        # 直接塞 None 进历史在严格校验的网关上会 400。
                        messages.append(
                            {"role": "assistant", "content": content}
                        )
                        messages.append(
                            {"role": "user", "content": "请继续，直接输出你的最终回答。"}
                        )
                        continue
                    if json_mode:
                        return _parse_json(content)
                    return {"answer": content}

                # ---- 第 3 步：把模型「要调工具」的决策记入对话历史 ----
                # 这条 assistant 消息（带 tool_calls）是 function calling 协议要求的，
                # 缺了它，模型就无法和后面的 tool 结果对应起来。
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": tool_calls,
                    }
                )

                # ---- 第 4 步：逐个执行工具，结果作为 Observation 回填 ----
                # 模型一次可能请求多个工具（tool_calls 是数组），这里串行执行：
                # 解析参数 → 调真正的数据源 → 把精简结果作为 tool 消息追加。
                for tc in tool_calls:
                    name = tc.get("function", {}).get("name", "")
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    t_tool = time.monotonic()
                    observation = await _execute_tool(tuniu, name, args)
                    tool_ms = int((time.monotonic() - t_tool) * 1000)

                    # 记录工具调用。is_error 的判断依据是 _execute_tool 的约定：
                    # 失败时它返回 {"error": ...} 而不是抛异常。所以轨迹里的
                    # "错误"是从返回内容识别的，这也顺带验证了这个约定是否被遵守。
                    if trace is not None:
                        is_err = '"error"' in observation
                        trace.add_tool_call(trace_mod.ToolCall(
                            step=step,
                            name=name,
                            args=args,
                            elapsed_ms=tool_ms,
                            observation_chars=len(observation),
                            is_error=is_err,
                            error_msg=observation[:200] if is_err else "",
                        ))

                    # tool_call_id 必须和上面的 tool_calls 一一对应，
                    # 这样模型才知道这条结果是对应它发起的哪个调用。
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": observation,
                        }
                    )
                # ---- 回到第 1 步，带着新拿到的工具结果继续循环 ----
    except TimeoutError as e:
        # 转成带中文说明的异常，上层 main.py 会把它映射成 504
        raise TimeoutError(
            f"规划超时：已超过 {total_budget:.0f} 秒仍未得出方案，请减少需求复杂度后重试"
        ) from e

    raise RuntimeError(f"ReAct 循环超过 {max_steps} 步仍未得到最终答案")


async def _execute_tool(tuniu: TuniuClient, name: str, args: dict) -> str:
    """执行工具：把模型的工具名/参数路由到真正的数据源，返回精简结果。

    返回的是「Observation」——一段给模型看的文本（JSON 字符串）。
    注意这里做了两层关键处理：
      1. 精简数据：车次只取前 20 条、酒店前 15 条、景点前 3 条，
         避免一次性塞给模型太多内容导致上下文爆炸
      2. 异常兜底：任何工具失败都不抛异常，而是返回 {"error": ...}，
         让模型看到错误后自行决定下一步（重试、换工具、或说明失败）
    """
    try:
        if name == "search_trains":
            # 查车次：先取原始数据，规整字段，再过滤掉下辖县市站点
            raw = tuniu.search_trains(
                args.get("departure_city", ""),
                args.get("arrival_city", ""),
                args.get("date", ""),
            )
            trains = _format_trains(raw)
            trains = _filter_trains_by_station(trains, args.get("arrival_city", ""), "arriveStation")
            return json.dumps(trains[:20], ensure_ascii=False)

        if name == "search_hotels":
            # 查酒店：prices 参数用于按用户预算过滤
            raw = tuniu.search_hotels(
                args.get("city", ""),
                args.get("check_in", ""),
                args.get("check_out", ""),
                args.get("prices"),
            )
            hotels = _format_hotels(raw)
            return json.dumps(hotels[:15], ensure_ascii=False)

        if name == "search_tickets":
            # 查门票：先查缓存，避免 ReAct 循环里重复查同一景点浪费限流额度
            spot = args.get("scenic_name", "")
            if spot in _ticket_cache:
                return _ticket_cache[spot]
            raw = tuniu.search_tickets(spot)
            result = json.dumps(_extract_ticket(spot, raw), ensure_ascii=False)
            # 只有成功拿到价格的才缓存；限流错误不缓存，下次可重试
            if '"price"' in result and '"error"' not in result:
                _ticket_cache[spot] = result
            return result

        if name == "get_weather":
            # 查天气：返回未来几天（含温度、天气状况、降水概率）
            w = weather.get_weather(args.get("city", ""), int(args.get("days") or 3))
            return json.dumps(w, ensure_ascii=False)

        if name == "search_attractions":
            # 搜景点：用 Tavily 返回景点介绍文本，供模型提取景点和门票价格参考
            results = tavily.search(
                args.get("query", ""),
                config.TAVILY_API_KEY,
                max_results=3,
            )
            return _format_attractions(results)

        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)

    except Exception as e:  # noqa: BLE001
        # 兜底：任何异常都包装成 error 返回，而不是让整个循环崩溃
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _format_trains(raw: dict) -> list[dict]:
    """把途牛车次原始字段规整成精简结构。

    两个关键处理：
      - 席别价格字段（edzPrice/ydzPrice/...）映射成中文（二等座/一等座/...），
        且只保留非空价格，避免把一堆空字段塞给模型
      - 限制前 30 个车次，控制信息量
    """
    items = raw.get("data", []) if isinstance(raw, dict) else []
    out = []
    for t in items[:30]:
        prices = {}
        for field, label in SEAT_NAME_MAP.items():
            val = (t.get("price") or {}).get(field, "")
            if val not in ("", None, 0, "0"):
                prices[label] = val
        out.append(
            {
                "trainNum": t.get("trainNum", ""),
                "departStation": t.get("departStationName", ""),
                "arriveStation": t.get("destStationName", ""),
                "departureTime": t.get("departureTime", ""),
                "arrivalTime": t.get("arrivalTime", ""),
                "duration": t.get("duration", ""),
                "seatPrices": prices,
            }
        )
    return out


def _filter_trains_by_station(trains: list[dict], city_name: str, field: str) -> list[dict]:
    """只保留站点以城市名开头的车次，过滤下辖县市。

    背景：途牛按「苏州」查车次时，会返回终点是「张家港」「常熟」等
    苏州下辖县级市的车次，它们离苏州市区很远，不是用户要去的地方。
    这里用「站点名是否以城市名开头」做启发式过滤：
      - 苏州 / 苏州北 / 苏州园区  → 以「苏州」开头，保留
      - 张家港 / 常熟 / 昆山      → 不以「苏州」开头，过滤
    若过滤后为空，则退回原列表（避免误伤导致无车次可用）。

    ⚠️ 参数叫 city_name 而不是 city，是为了**避免遮蔽模块级的 `city`**
    （app/services/city.py）——否则这个函数内后续任何 `city.resolve(...)`
    调用都会静默地在字符串上找方法而报 AttributeError。
    """
    filtered = [t for t in trains if str(t.get(field, "")).startswith(city_name)]
    return filtered or trains


def _format_hotels(raw: dict) -> list[dict]:
    """把途牛酒店原始字段规整成精简结构，限制前 20 家。

    字段命名上有一个刻意设计：价格叫 `price_per_night_from` 而不是 `price`。
    途牛返回的 `lowestPrice` 是**该酒店最低房型的单晚起步价**，通常是无窗/无早/
    不可取消的引流房型，既不是成交价，也不代表用户会订到的那间。若把它命名为
    含糊的 `price`，模型很容易当成"就是这个价"直接乘晚数写进总价，用户按方案
    去下单就会发现对不上。字段名自带「起价」语义，是成本最低的一道防呆。
    """
    items = raw.get("hotels", []) if isinstance(raw, dict) else []
    out = []
    for h in items[:20]:
        out.append(
            {
                "name": h.get("hotelName", ""),
                "star": h.get("starName", ""),
                "address": h.get("address", ""),
                "business": h.get("business", ""),
                "score": h.get("commentScore", ""),
                # 单晚「起价」，非成交价 —— 见 docstring 说明
                "price_per_night_from": h.get("lowestPrice", ""),
            }
        )
    return out


def _format_attractions(results: list[dict]) -> str:
    """把 Tavily 的搜索结果拼成一段文本，供模型提取景点信息。"""
    lines = []
    for r in results:
        lines.append(f"- {r.get('title', '')}: {r.get('content', '')}")
    return "\n".join(lines)


def _extract_ticket(spot: str, raw: dict) -> dict:
    """从途牛门票结果里挑出「真正的门票」，过滤掉非门票票种。

    途牛查景点时可能返回「VIP 讲解服务」「夜游演出票」「游船票」等，
    它们不是标准门票，价格会虚高。这里：
      1. 过滤名称里含「讲解/服务/演出/游船/导游/VIP」的票种
      2. 优先取成人票，取其中最低价
      3. 若过滤后没有真正门票，返回「免费或暂无标准门票」
    """
    tickets = raw.get("tickets", []) if isinstance(raw, dict) else []
    if not tickets:
        return {"spot": spot, "price": None, "note": "免费或暂无门票信息"}
    exclude_words = ("讲解", "服务", "演出", "游船", "导游", "vip", "VIP")
    real = [
        t for t in tickets
        if not any(k in (t.get("resName") or "") for k in exclude_words)
    ]
    if not real:
        return {"spot": spot, "price": None, "note": "免费或暂无标准门票"}
    adult = [t for t in real if t.get("personTypeName") == "成人票"]
    candidates = adult or real
    best = min(candidates, key=lambda t: _to_float(t.get("startPrice")) or float("inf"))
    price = _to_float(best.get("startPrice"))
    return {
        "spot": spot,
        "price": price,
        "ticket_name": best.get("resName", ""),
        "person_type": best.get("personTypeName", ""),
    }


def _to_float(v) -> float | None:
    """安全转 float：转换失败返回 None，避免脏数据导致崩溃。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_json(text: str) -> dict:
    """把模型输出解析成 JSON dict，兼容模型「在 JSON 前夹杂推理文本」的情况。

    模型在 ReAct 的最后一步，有时会先输出一段推理（英文思考过程），
    再输出真正的 JSON。这里先尝试直接解析，失败则截取
    「第一个 { 到最后一个 }」之间的子串再解析。
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # 模型有时会在 JSON 前输出推理文本，这里从「第一个 {」到「最后一个 }」截取 JSON 子串
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"DeepSeek 输出不是合法 JSON: {e}") from e
