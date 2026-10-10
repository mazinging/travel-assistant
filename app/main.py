"""FastAPI 应用入口：定义 HTTP 接口，连接前端页面与后端规划逻辑。

路由：
  POST /api/plan          生成旅行方案（核心接口，前端表单/反馈都走这里）
  POST /api/chat          自由问答
  GET  /api/health        健康检查（返回缺失的密钥，便于排查配置问题）
  GET  /api/resolve-city  识别城市名（城市 / 省份 / 无效），供前端即时校验
  GET  /                  返回前端页面（static/index.html）

前端静态资源（index.html / app.js / style.css）通过 /static 路径挂载。
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config
from app.agents import planner
from app.services import city

logger = logging.getLogger(__name__)

app = FastAPI(title="智能旅行助手", description="查询车次、推荐酒店、天气与游玩路线规划")


def _to_http_error(e: Exception) -> HTTPException:
    """把内部异常映射成语义正确的 HTTP 状态码。

    以前是一律 500 + `detail=str(e)`，有两个问题：
      1. 超时（用户等太久）和参数错误（用户填错）都被报成「服务器内部错误」，
         前端无法区分「该重试」还是「该改输入」
      2. `str(e)` 直接回显给客户端，而上游异常里可能夹带 API Key 或响应体片段
         这里只对**我们自己构造的、确定安全**的异常回显原文，其余给通用文案。

    城市相关异常单独处理，因为它们需要携带**结构化信息**给前端：
      - InvalidCityError     → 400，告诉用户哪个字段填错了
      - CityResolutionNeeded → 409，附带候选城市列表，让前端弹出选择
    """
    if isinstance(e, planner.InvalidCityError):
        return HTTPException(
            status_code=400,
            detail={"code": "invalid_city", "field": e.field, "message": str(e)},
        )
    if isinstance(e, planner.InvalidTripError):
        # 输入组合不成立（中转城市冲突、日期已过去）。
        # 单独一个 code 是为了前端将来能针对不同情况给不同引导。
        return HTTPException(
            status_code=400,
            detail={"code": "invalid_trip", "message": str(e)},
        )
    if isinstance(e, planner.CityResolutionNeeded):
        # 409 Conflict：请求本身没问题，但需要用户补充一个选择才能继续
        return HTTPException(
            status_code=409,
            detail={
                "code": "province_needs_city",
                "field": e.field,
                "province": e.province,
                "cities": e.cities,
                "message": str(e),
            },
        )
    if isinstance(e, TimeoutError):
        return HTTPException(status_code=504, detail=str(e) or "规划超时，请重试")
    if isinstance(e, ValueError):
        # 例如出发日期格式错误 —— 属于用户输入问题，用 400
        return HTTPException(status_code=400, detail=str(e))
    return HTTPException(status_code=500, detail="服务内部错误，请稍后重试或查看服务端日志")


class PlanRequest(BaseModel):
    """POST /api/plan 的请求体，用 Pydantic 做参数校验。"""
    departure_city: str = Field(..., min_length=1, description="所在城市")
    arrival_city: str = Field(..., min_length=1, description="目的地城市")
    departure_date: str = Field(..., description="出发日期 YYYY-MM-DD")
    days: int = Field(..., ge=1, le=30, description="旅行天数")
    transfer_city: str | None = Field(None, description="中转城市（可选，用于对比中转票价）")
    requirements: str = Field("", description="个人需求（自然语言描述，可选）")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="自由问答的问题")


@app.post("/api/plan")
def api_plan(req: PlanRequest):
    """核心接口：接收结构化输入 + 个人需求，返回多份旅行方案。

    这里把异常映射成合适的状态码（超时 504 / 输入错误 400 / 其余 500），
    让前端能拿到可读的中文提示，同时不把内部细节泄露给客户端。
    """
    try:
        return planner.plan_trip(
            req.departure_city,
            req.arrival_city,
            req.departure_date,
            req.days,
            req.transfer_city,
            req.requirements,
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("plan failed: %s", e)
        raise _to_http_error(e) from e


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    """自由问答接口：用户问任意问题，模型自主决定查什么工具来回答。"""
    try:
        return planner.chat(req.message)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("chat failed: %s", e)
        raise _to_http_error(e) from e


@app.get("/api/health")
def health():
    """健康检查：前端可在启动时调用，确认后端和密钥配置是否就绪。"""
    return {
        "status": "ok",
        "missing_keys": config.ensure_keys(),
    }


@app.get("/api/resolve-city")
def resolve_city(name: str):
    """识别一个城市名：是有效城市、省份、还是无效输入。

    这个接口让前端能在**用户提交表单之前**就给出反馈（例如输入「江苏」时立刻
    展示候选城市），而不必等到跑完整个规划流程才报错。它不做任何网络请求，
    纯本地判断，所以可以放心地在输入时频繁调用。
    """
    return city.resolve(name)


_static_dir = config.BASE_DIR / "static"
# 把 static 目录挂载到 /static，前端页面引用的 css/js 都从这里加载
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
def index():
    """返回首页 HTML。"""
    return FileResponse(_static_dir / "index.html")
