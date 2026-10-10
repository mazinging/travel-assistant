"""智能旅行助手后端包。

在这里统一配置日志：以前整个项目没有任何日志输出，出问题时只能看到 HTTP 500
的 detail 字符串，服务端毫无痕迹（例如天气查询失败被 `except Exception: pass`
吞掉，谁也不知道）。这里给出一个开箱可用的基础配置，`run.py` 启动即可看到
带时间戳的日志。

用 `basicConfig` 而非 `dictConfig`：本项目是本地单进程工具，不需要多 handler
分级落盘；只保证「至少能看到 INFO 及以上」这一件事。
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
