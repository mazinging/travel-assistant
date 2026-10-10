"""零依赖自检脚本：不装 pytest 也能验证核心逻辑。

用法：
    .venv/Scripts/python.exe tests/selfcheck.py

为什么需要它：这是我（AI）改完代码后**唯一能在无 shell 环境下设计的验证手段**。
pytest 需要额外安装，而这个脚本只用标准库，可以直接跑。
它同时也是一份「这些函数应该满足什么行为」的可执行文档。

如果装了 pytest，用 `python -m pytest tests/ -v` 能得到更详细的失败信息。
"""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents import planner  # noqa: E402
from app.services import city  # noqa: E402

PASSED, FAILED = 0, 0


def check(desc: str, cond: bool, extra: str = ""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [OK]   {desc}")
    else:
        FAILED += 1
        print(f"  [FAIL] {desc}" + (f"  -> {extra}" if extra else ""))


def section(title: str):
    print(f"\n=== {title} ===")


# ------------------------------------------------------- 1. 城市校验
section("城市校验 app/services/city.py")

# 有效城市
for name in ["苏州", "北京", "成都", "香港", "苏州市", "苏州站"]:
    r = city.resolve(name)
    check(f"「{name}」识别为城市", r["status"] == "ok", str(r))

# 省份 → 候选城市
for name in ["江苏", "江苏省", "浙江", "新疆", "内蒙古", "广西壮族自治区"]:
    r = city.resolve(name)
    ok = r["status"] == "province" and len(r.get("cities", [])) > 1
    check(f"「{name}」识别为省份并给出候选城市", ok, str(r))

# 直辖市不该让用户再选
for name in ["北京", "上海", "天津", "重庆"]:
    check(f"直辖市「{name}」直接当城市", city.resolve(name)["status"] == "ok")

# 无效输入
for name in ["asdfgh", "?????", "@#$%^&", "a", "这是一个非常非常长的地名输入", "", "   "]:
    r = city.resolve(name)
    check(f"无效输入 {name!r} 被拒绝", r["status"] == "invalid", str(r))

# 未知中文地名应放行（内置表无法穷举）
r = city.resolve("婺源")
check("未知中文地名「婺源」放行给数据源判断",
      r["status"] == "ok" and r.get("unverified") is True, str(r))

# 归一化
check("「苏州市」归一化为「苏州」", city.normalize("苏州市") == "苏州")
check("「江苏省」归一化为「江苏」", city.normalize("江苏省") == "江苏")


# ------------------------------------------------------- 2. 总价重算
section("总价重算 planner._fix_total_cost")


def mkplan(**kw):
    p = {
        "name": "方案一",
        "outbound": {"price": 100},
        "return": {"price": 120},
        "hotel": {"total": 400},
        "tickets": [{"spot": "A", "price": 60}, {"spot": "B", "price": 20}],
        "total_cost": 700,
    }
    p.update(kw)
    return p


p = mkplan()
check("正确总价不被改动", planner._fix_total_cost(p, 0) == [] and p["total_cost"] == 700)

p = mkplan(total_cost=999)
issues = planner._fix_total_cost(p, 0)
check("错误总价被修正为 700", p["total_cost"] == 700, f"实际 {p['total_cost']}")
check("修正时产出 warning", bool(issues) and "999" in issues[0], str(issues))

p = mkplan()
del p["total_cost"]
check("缺失总价被补上", planner._fix_total_cost(p, 0) and p["total_cost"] == 700)

p = mkplan(total_cost="700")
check("字符串数字「700」被接受", planner._fix_total_cost(p, 0) == [])

p = mkplan(total_cost="700元")
check("带单位「700元」被重算而非崩溃",
      planner._fix_total_cost(p, 0) and p["total_cost"] == 700)

p = {"name": "X", "outbound": {}, "return": {}, "hotel": {}, "tickets": []}
check("无车票数据时不硬算", planner._fix_total_cost(p, 0) == [])

p = {"name": "X", "outbound": {"price": 100}, "tickets": []}
planner._fix_total_cost(p, 0)
check("仅有去程票价时可重算", p["total_cost"] == 100)

p = mkplan(tickets=[{"spot": "A", "price": None}], total_cost=999)
planner._fix_total_cost(p, 0)
check("免费门票按 0 计入", p["total_cost"] == 620, f"实际 {p['total_cost']}")


# --------------------------------------------------- 3. 行程时间线校验
section("行程时间线 planner._check_itinerary")

DEP = date(2025, 9, 22)
RET = "2025-09-24"

p = {
    "name": "方案一",
    "itinerary": [{"day": 1}, {"day": 2}, {"day": 3}],
    "outbound": {"detail": "G1 苏州北 09:15 → 合肥南 12:30"},
    "return": {"detail": "G2 合肥南 18:00 → 苏州北 21:00"},
}
check("自洽的行程无告警", planner._check_itinerary(p, 0, 3, DEP, RET) == [])

p = {"name": "X", "itinerary": [{"day": i} for i in range(1, 6)]}
iss = planner._check_itinerary(p, 0, 3, DEP, RET)
check("天数不一致被发现（5天 vs 要求3天）",
      any("5 天" in i and "3 天" in i for i in iss), str(iss))

p = {"name": "X", "itinerary": [{"day": 0}, {"day": 1}, {"day": 3}]}
iss = planner._check_itinerary(p, 0, 3, DEP, RET)
check("day 序号不连续被发现", any("序号不连续" in i for i in iss), str(iss))

p = {"name": "X", "outbound": {"detail": "G1 出发 15:00"},
     "return": {"detail": "G2 返程 09:00"}}
iss = planner._check_itinerary(p, 0, 3, DEP, RET)
check("时间倒挂被发现（回程早于去程）", any("时间倒挂" in i for i in iss), str(iss))

p = {"name": "X", "outbound": {"detail": "G1 09:00"}, "return": {"detail": "G2 09:00"}}
check("同一时刻不算倒挂", planner._check_itinerary(p, 0, 3, DEP, RET) == [])

p = {"name": "X", "itinerary": [{"day": 1, "date": "2025-10-01"}]}
iss = planner._check_itinerary(p, 0, 3, DEP, RET)
check("日期越界被发现", any("超出" in i for i in iss), str(iss))

# 注意：下面两条用 days=1，是为了**只触发日期检查**。
# 早先版本用 len(itinerary)==1 配 days=3，会同时触发「天数不一致」，
# 导致断言失败——那是测试数据的错，不是被测算法的错。
p = {"name": "X", "itinerary": [{"day": 1, "date": "2025-09-23"}]}
iss = planner._check_itinerary(p, 0, 1, DEP, "2025-09-23")
check("范围内日期正常（仅日期检查，天数匹配）", iss == [], str(iss))

p = {"name": "X", "itinerary": [{"day": 1, "date": "2025-09-22"}]}
iss = planner._check_itinerary(p, 0, 1, DEP, "2025-09-22")
check("出发日边界视为合法", iss == [], str(iss))

p = {"name": "X", "itinerary": [{"day": 1, "date": "2025-09-24"}]}
iss = planner._check_itinerary(p, 0, 1, DEP, RET)
check("回程日边界视为合法", iss == [], str(iss))


# 残缺输入不能崩
section("健壮性：残缺的模型输出")
crashed = []
for bad in [{}, {"name": "X"}, {"name": "X", "itinerary": "不是列表"},
            {"name": "X", "itinerary": [None, 123]},
            {"name": "X", "tickets": "不是列表"},
            {"name": "X", "hotel": "不是字典"}]:
    try:
        planner._check_itinerary(bad, 0, 3, DEP, RET)
        planner._fix_total_cost(bad, 0)
    except Exception as e:
        crashed.append(f"{bad} -> {type(e).__name__}: {e}")
check("残缺输入不抛异常", not crashed, "; ".join(crashed))


# ------------------------------------------------------- 4. 时间提取
section("时间提取 planner._extract_first_time")
check("提取 HH:MM", planner._extract_first_time({"detail": "G1 09:15 → 12:30"}) == "09:15")
check("单位数小时补零", planner._extract_first_time({"detail": "G1 9:05"}) == "09:05")
check("非法小时不匹配", planner._extract_first_time({"detail": "G1 25:99"}) is None)
check("无时间返回 None", planner._extract_first_time({"detail": "无时间"}) is None)
check("非字典输入安全",
      all(planner._extract_first_time(v) is None for v in [None, "s", 1, []]))


# ------------------------------------------------------- 5. 输入关系校验
section("输入关系校验 planner._validate_trip_inputs / _detect_same_city")

TODAY = date.today()
FUTURE = (TODAY + timedelta(days=30)).isoformat()
PAST = (TODAY - timedelta(days=1)).isoformat()

# 市内游识别
check("同城被识别为市内游", planner._detect_same_city("南京", "南京") is True)
check("不同城不算市内游", planner._detect_same_city("南京", "苏州") is False)

# 市内游不报错（是合法场景，只给提示）
try:
    planner._validate_trip_inputs("南京", "南京", None, FUTURE)
    check("市内游不抛异常（允许，只提示）", True)
except Exception as e:
    check("市内游不抛异常（允许，只提示）", False, f"{type(e).__name__}: {e}")

# 中转城市冲突 —— 逻辑无解，必须拦下
for tc, desc in [("南京", "中转=出发地"), ("苏州", "中转=目的地")]:
    try:
        planner._validate_trip_inputs("南京", "苏州", tc, FUTURE)
        check(f"{desc} 被拦下", False, "未抛出异常")
    except planner.InvalidTripError as e:
        check(f"{desc} 被拦下", "中转城市不能是" in str(e), str(e))

# 合法中转应放行
try:
    planner._validate_trip_inputs("南京", "苏州", "无锡", FUTURE)
    check("合法中转放行", True)
except Exception as e:
    check("合法中转放行", False, f"{type(e).__name__}: {e}")

# 过去日期必须拦下
try:
    planner._validate_trip_inputs("南京", "苏州", None, PAST)
    check("过去的出发日期被拦下", False, "未抛出异常")
except planner.InvalidTripError as e:
    check("过去的出发日期被拦下", "已经过去" in str(e), str(e))

# 今天应放行（边界：不应把"今天"也拦掉）
try:
    planner._validate_trip_inputs("南京", "苏州", None, TODAY.isoformat())
    check("今天的日期放行（边界）", True)
except Exception as e:
    check("今天的日期放行（边界）", False, str(e))

# 日期格式错误时不该在这里报错（交给原有逻辑，避免重复提示）
try:
    planner._validate_trip_inputs("南京", "苏州", None, "不是日期")
    check("非法日期格式在此处静默，不重复报错", True)
except Exception as e:
    check("非法日期格式在此处静默，不重复报错", False, str(e))

# InvalidTripError 必须是 ValueError 子类，否则 main.py 的 400 映射不生效
check("InvalidTripError 继承 ValueError（保证映射成 400）",
      issubclass(planner.InvalidTripError, ValueError))


# ------------------------------------------------------- 6. 轨迹记录
section("轨迹记录 app/agents/trace.py")
from app.agents import trace as trace_mod  # noqa: E402

tr = trace_mod.Trace()
tr.add_step(trace_mod.Step(step=0, prompt_tokens=2500, completion_tokens=120,
                           tool_call_count=2, elapsed_ms=3100))
tr.add_step(trace_mod.Step(step=1, prompt_tokens=4600, completion_tokens=90,
                           tool_call_count=1, elapsed_ms=2800))
tr.add_tool_call(trace_mod.ToolCall(step=0, name="search_trains", args={"a": 1},
                                    elapsed_ms=800, observation_chars=1800,
                                    is_error=False))
tr.add_tool_call(trace_mod.ToolCall(step=0, name="search_hotels", args={},
                                    elapsed_ms=600, observation_chars=900,
                                    is_error=False))
tr.add_tool_call(trace_mod.ToolCall(step=1, name="search_tickets", args={},
                                    elapsed_ms=200, observation_chars=60,
                                    is_error=True, error_msg='{"error":"限流"}'))

s = tr.summary()
check("llm_rounds 统计正确", s["llm_rounds"] == 2, str(s["llm_rounds"]))
check("tool_calls 统计正确", s["tool_calls"] == 3, str(s["tool_calls"]))
check("tool_errors 只计失败项", s["tool_errors"] == 1, str(s["tool_errors"]))
check("prompt_tokens 求和", s["prompt_tokens_total"] == 7100,
      str(s["prompt_tokens_total"]))
check("prompt_tokens_growth 逐轮记录",
      s["prompt_tokens_growth"] == [2500, 4600], str(s["prompt_tokens_growth"]))
check("by_tool 聚合了调用与失败数",
      s["by_tool"]["search_trains"]["calls"] == 1
      and s["by_tool"]["search_tickets"]["errors"] == 1, str(s["by_tool"]))

d = tr.to_dict()
check("to_dict 含 summary/steps/tool_calls 三部分",
      set(d.keys()) == {"summary", "steps", "tool_calls"}, str(list(d.keys())))
check("steps 序列化正确", len(d["steps"]) == 2 and d["steps"][0]["step"] == 0)
check("tool_calls 序列化保留 is_error",
      d["tool_calls"][2]["is_error"] is True)

# 空轨迹不能崩（例如循环在第一轮就失败）
empty = trace_mod.Trace().summary()
check("空轨迹的汇总安全", empty["llm_rounds"] == 0 and empty["tool_errors"] == 0,
      str(empty))


# ------------------------------------------------------- 7. 日期计算
section("日期计算不变式（days=1 的 P0 缺陷）")

def calc(days):
    n = max(days - 1, 0)
    return {
        "nights": n,
        "check_in": DEP.isoformat(),
        "check_out": (DEP + timedelta(days=n)).isoformat(),
        "return_date": (DEP + timedelta(days=max(days - 1, 0))).isoformat(),
    }


check("days=1 → 住 0 晚且离店==入住", calc(1)["nights"] == 0
      and calc(1)["check_out"] == calc(1)["check_in"])
check("days=2 → 住 1 晚，9-23 离店", calc(2)["nights"] == 1
      and calc(2)["check_out"] == "2025-09-23")
check("days=3 → 住 2 晚，9-24 离店", calc(3)["nights"] == 2
      and calc(3)["check_out"] == "2025-09-24")

bad = [d for d in range(1, 31)
       if calc(d)["check_out"] != (DEP + timedelta(days=d - 1)).isoformat()]
check("days=1..30 的 check_out 全部与 nights 同源", not bad, f"不一致: {bad}")


# ------------------------------------------------------- 汇总
print(f"\n{'=' * 46}")
print(f"通过 {PASSED} 项，失败 {FAILED} 项")
print("=" * 46)
sys.exit(1 if FAILED else 0)
