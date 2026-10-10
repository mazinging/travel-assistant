"""应用配置：从项目根目录的 .env 读取密钥。"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

TUNIU_API_KEY = os.getenv("TUNIU_API_KEY", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()


def ensure_keys() -> list[str]:
    """检查必需的密钥是否已配置，返回缺失项列表。"""
    missing = []
    if not TUNIU_API_KEY:
        missing.append("TUNIU_API_KEY")
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    return missing
