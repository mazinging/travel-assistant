"""Tavily 搜索服务。"""
import httpx


def search(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    if not api_key:
        raise RuntimeError("未配置 Tavily API Key")
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        raise RuntimeError(f"Tavily 搜索失败: {e}") from e

    results = data.get("results", [])
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": (r.get("content") or "")[:500],
        }
        for r in results
    ]
