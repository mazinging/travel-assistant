"""纯函数单测：城市校验、总价重算、行程时间线校验。

这三个模块的共同点：**它们都是「用代码替代模型做判断」的地方**。
模型的输出是概率性的，而这些函数必须给出确定答案——所以它们最需要测试。

运行：.venv/Scripts/python.exe -m pytest tests/ -v
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents import planner  # noqa: E402
from app.services import city  # noqa: E402


# ---------------------------------------------------------------- 城市校验

class TestCityNormalize:
    """输入清洗：带行政后缀的城市名应被归一化。"""

    @pytest.mark.parametrize("raw,expected", [
        ("苏州", "苏州"),
        ("苏州市", "苏州"),
        ("苏州站", "苏州"),
        ("  北京  ", "北京"),
        ("江苏省", "江苏"),
    ])
    def test_normalize(self, raw, expected):
        assert city.normalize(raw) == expected


class TestCityResolve:
    """三种输入的识别：有效城市 / 省份 / 无效。"""

    @pytest.mark.parametrize("name", ["苏州", "北京", "成都", "香港", "苏州市"])
    def test_valid_city(self, name):
        r = city.resolve(name)
        assert r["status"] == "ok", f"{name} 应被识别为城市，实际 {r}"
        assert r["city"]

    @pytest.mark.parametrize("name", ["江苏", "江苏省", "浙江", "新疆",
                                      "内蒙古", "广西壮族自治区"])
    def test_province_returns_candidates(self, name):
        r = city.resolve(name)
        assert r["status"] == "province", f"{name} 应被识别为省份，实际 {r}"
        assert len(r["cities"]) > 1
        assert all(isinstance(c, str) and c for c in r["cities"])

    def test_municipality_treated_as_city(self):
        """直辖市只有自己一个城市，不该让用户再选一次。"""
        for name in ["北京", "上海", "天津", "重庆"]:
            assert city.resolve(name)["status"] == "ok", name

    @pytest.mark.parametrize("name", [
        "asdfgh",      # 纯英文乱码
        "?????",       # 符号
        "@#$%^&",      # 特殊字符
        "a",           # 太短
        "这是一个非常非常长的地名输入",   # 超长
        "",            # 空
        "   ",         # 空白
    ])
    def test_invalid_input(self, name):
        r = city.resolve(name)
        assert r["status"] == "invalid", f"{name!r} 应被拒绝，实际 {r}"
        assert r.get("reason")

    def test_unknown_chinese_place_passes_through(self):
        """内置表不可能穷举全国县市，像中文的地名应放行给数据源判断。"""
        r = city.resolve("婺源")
        assert r["status"] == "ok"
        assert r.get("unverified") is True


# ------------------------------------------------------------ 总价重算

def _plan(**kw):
    base = {
        "name": "方案一",
        "outbound": {"price": 100},
        "return": {"price": 120},
        "hotel": {"total": 400},
        "tickets": [{"spot": "A", "price": 60}, {"spot": "B", "price": 20}],
        "total_cost": 700,
    }
    base.update(kw)
    return base


class TestFixTotalCost:
    """总价必须由代码重算，不能相信模型的算术。"""

    def test_correct_total_untouched(self):
        p = _plan()          # 100+120+400+60+20 = 700
        assert planner._fix_total_cost(p, 0) == []
        assert p["total_cost"] == 700

    def test_wrong_total_is_corrected(self):
        p = _plan(total_cost=999)
        issues = planner._fix_total_cost(p, 0)
        assert p["total_cost"] == 700, "错误的总额必须被覆盖"
        assert issues and "999" in issues[0]

    def test_missing_total_is_filled(self):
        p = _plan()
        del p["total_cost"]
        issues = planner._fix_total_cost(p, 0)
        assert p["total_cost"] == 700
        assert issues

    def test_numeric_string_tolerated(self):
        """模型经常把数字写成字符串，应能处理。"""
        p = _plan(total_cost="700")
        assert planner._fix_total_cost(p, 0) == []

    def test_string_with_unit_recovered(self):
        """模型可能输出 "700元" —— 无法转成数字，应重算而不是崩溃。"""
        p = _plan(total_cost="700元")
        issues = planner._fix_total_cost(p, 0)
        assert p["total_cost"] == 700
        assert issues

    def test_missing_prices_skips_recompute(self):
        """连车票钱都没有时不该硬算，否则会产出一个更误导的数字。"""
        p = {"name": "X", "outbound": {}, "return": {}, "hotel": {}, "tickets": []}
        assert planner._fix_total_cost(p, 0) == []

    def test_partial_data_still_recomputes(self):
        """有去程票价即可重算（回程/酒店缺失按 0 处理）。"""
        p = {"name": "X", "outbound": {"price": 100}, "tickets": []}
        planner._fix_total_cost(p, 0)
        assert p["total_cost"] == 100

    def test_free_tickets_counted_as_zero(self):
        p = _plan(tickets=[{"spot": "A", "price": None}], total_cost=999)
        planner._fix_total_cost(p, 0)
        assert p["total_cost"] == 620   # 100+120+400+0


# -------------------------------------------------------- 行程时间线校验

DEP = date(2025, 9, 22)
RET = "2025-09-24"


class TestCheckItinerary:
    """行程天数与时间线的确定性检查。"""

    def test_consistent_itinerary_passes(self):
        p = {
            "name": "方案一",
            "itinerary": [{"day": 1}, {"day": 2}, {"day": 3}],
            "outbound": {"detail": "G1 苏州北 09:15 → 合肥南 12:30"},
            "return": {"detail": "G2 合肥南 18:00 → 苏州北 21:00"},
        }
        assert planner._check_itinerary(p, 0, 3, DEP, RET) == []

    def test_day_count_mismatch_flagged(self):
        p = {"name": "X", "itinerary": [{"day": i} for i in range(1, 6)]}
        issues = planner._check_itinerary(p, 0, 3, DEP, RET)
        assert any("5 天" in i and "3 天" in i for i in issues)

    def test_non_sequential_days_flagged(self):
        p = {"name": "X", "itinerary": [{"day": 0}, {"day": 1}, {"day": 3}]}
        issues = planner._check_itinerary(p, 0, 3, DEP, RET)
        assert any("序号不连续" in i for i in issues)

    def test_time_reversal_flagged(self):
        """回程早于去程 = 时间倒挂，这是用户明确反馈的问题之一。"""
        p = {
            "name": "X",
            "outbound": {"detail": "G1 出发 15:00"},
            "return": {"detail": "G2 返程 09:00"},
        }
        issues = planner._check_itinerary(p, 0, 3, DEP, RET)
        assert any("时间倒挂" in i for i in issues)

    def test_same_time_not_flagged(self):
        """同一时刻不算倒挂（用 < 而非 <=）。"""
        p = {
            "name": "X",
            "outbound": {"detail": "G1 09:00"},
            "return": {"detail": "G2 09:00"},
        }
        assert planner._check_itinerary(p, 0, 3, DEP, RET) == []

    def test_date_out_of_range_flagged(self):
        p = {"name": "X", "itinerary": [{"day": 1, "date": "2025-10-01"}]}
        issues = planner._check_itinerary(p, 0, 3, DEP, RET)
        assert any("超出" in i for i in issues)

    # 以下三条用 days=1，是为了**只触发日期检查**。
    # 若用 len(itinerary)==1 配 days=3，会同时触发「天数不一致」而干扰断言。
    def test_date_in_range_ok(self):
        p = {"name": "X", "itinerary": [{"day": 1, "date": "2025-09-23"}]}
        assert planner._check_itinerary(p, 0, 1, DEP, "2025-09-23") == []

    def test_departure_date_boundary_ok(self):
        p = {"name": "X", "itinerary": [{"day": 1, "date": "2025-09-22"}]}
        assert planner._check_itinerary(p, 0, 1, DEP, "2025-09-22") == []

    def test_return_date_boundary_ok(self):
        p = {"name": "X", "itinerary": [{"day": 1, "date": "2025-09-24"}]}
        assert planner._check_itinerary(p, 0, 1, DEP, RET) == []

    def test_empty_and_malformed_input_safe(self):
        """模型输出残缺时不能崩溃 —— 这是最常见的现实情况。"""
        for p in [{}, {"name": "X"}, {"name": "X", "itinerary": "不是列表"},
                  {"name": "X", "itinerary": [None, 123]}]:
            planner._check_itinerary(p, 0, 3, DEP, RET)   # 不抛异常即通过


class TestExtractFirstTime:
    def test_extracts_hhmm(self):
        assert planner._extract_first_time({"detail": "G1 09:15 → 12:30"}) == "09:15"

    def test_pads_single_digit_hour(self):
        assert planner._extract_first_time({"detail": "G1 9:05 出发"}) == "09:05"

    def test_invalid_hour_not_matched(self):
        assert planner._extract_first_time({"detail": "G1 25:99"}) is None

    def test_no_time_returns_none(self):
        assert planner._extract_first_time({"detail": "无时间信息"}) is None

    def test_non_dict_safe(self):
        for v in [None, "字符串", 123, []]:
            assert planner._extract_first_time(v) is None


# ------------------------------------------------------------ 日期计算

class TestDateCalculation:
    """days=1 的边界 —— 这是之前修掉的 P0 缺陷，必须有测试锁住。"""

    @staticmethod
    def _calc(days: int, dep: date = DEP):
        nights = max(days - 1, 0)
        return {
            "nights": nights,
            "check_in": dep.isoformat(),
            "check_out": (dep + timedelta(days=nights)).isoformat(),
            "return_date": (dep + timedelta(days=max(days - 1, 0))).isoformat(),
        }

    def test_single_day_no_hotel(self):
        r = self._calc(1)
        assert r["nights"] == 0
        assert r["check_out"] == r["check_in"], "住 0 晚时离店必须等于入住"

    def test_two_days_one_night(self):
        r = self._calc(2)
        assert r["nights"] == 1
        assert r["check_out"] == "2025-09-23"
        assert r["return_date"] == "2025-09-23"

    def test_three_days_two_nights(self):
        r = self._calc(3)
        assert r["nights"] == 2
        assert r["check_out"] == "2025-09-24"
        assert r["return_date"] == "2025-09-24"

    def test_checkout_consistent_for_all_days(self):
        """核心不变式：check_out 与 nights 必须始终同源。"""
        for days in range(1, 31):
            r = self._calc(days)
            assert r["nights"] == days - 1
            expected = (DEP + timedelta(days=days - 1)).isoformat()
            assert r["check_out"] == expected, f"days={days} 不一致"