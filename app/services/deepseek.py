"""DeepSeek 大模型服务（OpenAI 兼容接口）。

提供两种调用方式：
  - chat()：普通对话，可强制 JSON 输出
  - chat_with_tools()：function calling（ReAct 范式下让模型自主决定调用工具）
"""
import httpx


def chat(messages: list[dict], api_key: str, json_mode: bool = True) -> str:
    """普通对话，返回模型回复文本。"""
    if not api_key:
        raise RuntimeError("未配置 DeepSeek API Key")

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = httpx.post(
            "https://api.deepseek.com/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=90,
        )
    except httpx.HTTPError as e:
        raise RuntimeError(f"DeepSeek 调用失败: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek 返回 HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"DeepSeek 响应格式异常: {data}") from e


def chat_with_tools(
    messages: list[dict], api_key: str, tools: list[dict], with_usage: bool = False
):
    """function calling 调用：返回完整的 message（可能含 tool_calls）。

    这是 ReAct 循环的核心接口。模型返回的 message 有两种可能：
      - 含 tool_calls：说明模型想调用工具，调用方需执行工具并把结果作为
        tool 角色消息追加回 messages，继续下一轮
      - 不含 tool_calls：说明模型已推理完毕，content 即最终答案

    请求体与 OpenAI 兼容，关键字段：
      - tools：工具定义数组（JSON Schema），告诉模型「你能调什么」
      - tool_choice="auto"：让模型自己决定是否调用、调哪个（而非强制）

    返回的 message 里 tool_calls 结构：
      [{"id": "call_xxx", "type": "function",
        "function": {"name": "工具名", "arguments": "JSON 字符串形式的参数"}}]
    注意 arguments 是字符串，需要调用方自行 json.loads 解析。

    参数 with_usage：
      **这是为轨迹记录加的参数。** DeepSeek 的响应里本来就有 `usage`
      （prompt_tokens / completion_tokens），但旧代码只取了 message 就把它丢了
      —— 导致我们完全不知道 agent 的成本结构。
      开启后返回 (message, usage) 元组。
      默认 False 是为了不破坏既有调用方的返回值约定。
    """
    if not api_key:
        raise RuntimeError("未配置 DeepSeek API Key")

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "stream": False,
        "tools": tools,
        "tool_choice": "auto",
    }
    try:
        resp = httpx.post(
            "https://api.deepseek.com/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=90,
        )
    except httpx.HTTPError as e:
        raise RuntimeError(f"DeepSeek 调用失败: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek 返回 HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"DeepSeek 响应格式异常: {data}") from e

    if with_usage:
        # usage 缺失时不报错——它只是观测数据，不该让规划失败
        usage = data.get("usage") or {}
        return message, usage
    return message
