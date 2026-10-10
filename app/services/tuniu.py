"""途牛 MCP 客户端：火车票 / 酒店 / 门票三个数据源的统一封装。

途牛开放平台的 MCP 采用「无状态 + JSON-RPC 2.0 over HTTP」设计，与标准 MCP 不同，
它**不需要 initialize 握手**，直接向固定端点 POST `tools/call` 就能调工具。

请求体（JSON-RPC 2.0）：
    {"jsonrpc":"2.0","id":1,"method":"tools/call",
     "params":{"name":"工具名","arguments":{...}}}

认证：请求头 `apiKey`（不是常见的 Authorization）。

两个容易踩坑的细节（都已在本文件处理）：
  1. 响应是 SSE 格式（`event: message` + `data: {...}`），不能直接 resp.json()，
     需要先提取 `data:` 行再解析 —— 见 _parse_body
  2. tools/call 的返回体是嵌套的：result.content[0].text 是一个「JSON 字符串」，
     需要再 json.loads 一次才是真实数据 —— 见 _unwrap
"""
import json

import httpx


class TuniuError(Exception):
    """途牛接口调用异常。"""


SERVERS = {
    "train": "https://openapi.tuniu.cn/mcp/train",
    "hotel": "https://openapi.tuniu.cn/mcp/hotel",
    "ticket": "https://openapi.tuniu.cn/mcp/ticket",
}


class TuniuClient:
    def __init__(self, api_key: str, timeout: float = 30.0):
        if not api_key:
            raise TuniuError("未配置途牛 API Key")
        self.api_key = api_key
        self.timeout = timeout

    def call_tool(self, server: str, tool_name: str, arguments: dict) -> dict:
        """通用调用入口：向指定服务端点发起一次 tools/call 并解析出真实数据。

        参数：
          server    → 服务名（train/hotel/ticket），映射到对应端点 URL
          tool_name → 途牛的工具名（如 searchLowestPriceTrain、tuniuHotelSearch）
          arguments → 工具参数（如 {"departureCityName": "合肥", ...}）

        返回值：解析后的业务数据（dict），可能是车次列表 / 酒店列表 / 门票等。
        """
        url = SERVERS.get(server)
        if not url:
            raise TuniuError(f"未知的途牛服务: {server}")

        # 标准 JSON-RPC 2.0 请求体：
        #   method = "tools/call" 表示「调用某个工具」
        #   params.name / params.arguments 指定调哪个工具、传什么参数
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "apiKey": self.api_key,
        }

        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        except httpx.HTTPError as e:
            raise TuniuError(f"途牛接口连接失败: {e}") from e

        if resp.status_code == 401:
            raise TuniuError("途牛认证失败：请检查 TUNIU_API_KEY 是否正确")
        if resp.status_code >= 400:
            raise TuniuError(f"途牛接口返回 HTTP {resp.status_code}: {resp.text[:200]}")

        # 先解析外层（可能是 SSE，也可能是纯 JSON），再解包内层嵌套
        try:
            body = self._parse_body(resp.text)
        except (json.JSONDecodeError, ValueError) as e:
            raise TuniuError(f"途牛响应解析失败: {e}\n原始响应: {resp.text[:300]}") from e

        if isinstance(body, dict) and body.get("error"):
            raise TuniuError(f"途牛返回错误: {body['error']}")

        return self._unwrap(body.get("result"))

    @staticmethod
    def _parse_body(text: str) -> dict:
        """途牛接口返回纯 JSON 或 SSE（Server-Sent Events）格式，统一解析。"""
        text = (text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 解析 SSE：提取所有 data: 行
        data_lines = []
        for line in text.splitlines():
            stripped = line.rstrip()
            if stripped.startswith("data:"):
                content = stripped[5:].strip()
                if content:
                    data_lines.append(content)
        if data_lines:
            return json.loads("\n".join(data_lines))
        raise ValueError("既不是合法 JSON 也不是 SSE 格式")

    @staticmethod
    def _unwrap(result):
        """解包 MCP 的返回结构。

        tools/call 的 result 通常是：
            {"content": [{"type": "text", "text": "{\"data\": [...]}"}]}
        其中 content[0].text 才是真正的业务数据，但它是「字符串形式」的 JSON，
        所以要再 json.loads 一次。若已经是普通 dict（某些接口直接返回），则原样返回。
        """
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list) and content:
                text = content[0].get("text")
                if isinstance(text, str):
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
            return result
        return result

    def search_trains(self, departure_city: str, arrival_city: str, date: str) -> dict:
        """查询车次列表（工具名 searchLowestPriceTrain，按城市名 + 日期查询）。"""
        return self.call_tool(
            "train",
            "searchLowestPriceTrain",
            {
                "departureCityName": departure_city,
                "arrivalCityName": arrival_city,
                "departureDate": date,
            },
        )

    def search_hotels(self, city: str, check_in: str, check_out: str, prices: str | None = None) -> dict:
        """搜索酒店（工具名 tuniuHotelSearch）。

        prices 为可选的价格区间参数，格式如 "0-200"，由调用方从用户需求里提取，
        用于在数据源层面就过滤掉超预算的酒店。
        """
        args = {"cityName": city, "checkIn": check_in, "checkOut": check_out}
        if prices:
            args["prices"] = prices
        return self.call_tool("hotel", "tuniuHotelSearch", args)

    def search_tickets(self, scenic_name: str) -> dict:
        """查询景点门票（工具名 query_cheapest_tickets，按景点名查询）。"""
        return self.call_tool(
            "ticket",
            "query_cheapest_tickets",
            {"scenic_name": scenic_name},
        )
