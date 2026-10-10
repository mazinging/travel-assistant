"""Agent 执行轨迹记录。

## 为什么需要这个模块

传统的调试方式是读代码 + 看输出。但 **agent 不行**——因为 agent 的行为不是
代码决定的，是模型在运行时决定的。读 `_react_loop` 你只能知道"可能的路径"，
不可能知道"实际走了哪条"。

具体来说，下面这些问题**读代码一个都答不出来**：

  - 模型实际按什么顺序查数据？每次都一样吗？
  - 一轮规划平均跑几轮循环？最多几轮？
  - prompt_tokens 涨到多少？（ReAct 的成本是 O(N²)，但实际 N 是几？）
  - 某个工具是不是经常失败？
  - 我加的那些校验器（总价重算、时间线检查）触发了几次？
  - 模型一次并行调几个工具？

**这些都是 agent 工程最核心的问题，而它们只能通过观察得到答案。**

所以这个模块的目的不是"打日志"，而是**给 agent 装一个观察窗口**。

## 设计原则

1. **零侵入**：记录不影响主流程。任何记录失败都不该让规划失败。
2. **面向问题**：每个字段都对应一个"我想知道的 agent 行为问题"，见下方注释。
3. **可对比**：结构固定，便于跑多次后横向比较（做实验的基础）。
"""
import time
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """一次工具调用的记录。

    字段与「想回答的问题」的对应关系：

      step              → 这次调用发生在第几轮？模型是"先查车次"还是"先查天气"？
      name              → 模型最常调哪个工具？有没有调过某个工具？
      args              → 模型传的参数对不对？有没有编造日期、传错城市？
      elapsed_ms        → 哪个工具最慢？并行执行能省多少时间？
      observation_chars → 返回给模型的数据有多大？是不是 context 膨胀的主因？
      is_error          → 失败率多高？失败后模型怎么办？
      error_msg         → 失败原因是什么（限流？超时？参数错？）
    """

    step: int
    name: str
    args: dict
    elapsed_ms: int
    observation_chars: int
    is_error: bool
    error_msg: str = ""

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "name": self.name,
            "args": self.args,
            "elapsed_ms": self.elapsed_ms,
            "observation_chars": self.observation_chars,
            "is_error": self.is_error,
            "error_msg": self.error_msg,
        }


@dataclass
class Step:
    """一次 LLM 调用的记录（ReAct 的每一"轮"）。

    字段对应的问题：

      step           → 第几轮
      prompt_tokens  → **最重要**：本轮发给模型的输入有多大。
                       连续看这个数就能看到历史累积的速度。
      completion_tokens → 模型本轮"思考"了多久（输出多少）
      tool_call_count   → 模型这轮想同时调几个工具？（>1 说明可以并行优化）
      elapsed_ms        → 这轮 LLM 调用耗时
      empty_retry       → 是不是触发了空响应重试？（模型健康度指标）
    """

    step: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_call_count: int = 0
    elapsed_ms: int = 0
    empty_retry: bool = False

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_call_count": self.tool_call_count,
            "elapsed_ms": self.elapsed_ms,
            "empty_retry": self.empty_retry,
        }


@dataclass
class Trace:
    """一次完整规划的执行轨迹。

    这是最终暴露给外部的对象（放进 `_meta.trace`）。它同时提供**原始记录**
    和**汇总统计**：原始记录用于逐行排查，汇总用于快速判断"这次正常吗"。
    """

    steps: list[Step] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished: bool = False

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def add_tool_call(self, call: ToolCall) -> None:
        self.tool_calls.append(call)

    def to_dict(self) -> dict:
        """返回给前端的结构：明细 + 汇总。"""
        return {
            "summary": self.summary(),
            "steps": [s.to_dict() for s in self.steps],
            "tool_calls": [t.to_dict() for t in self.tool_calls],
        }

    def summary(self) -> dict:
        """汇总统计 —— 这是"跑完一次快速判断"的入口。

        设计上刻意包含 `prompt_tokens_growth`：它直接展示 ReAct 的成本特征。
        如果这个数组增长很快（比如 2500 → 4500 → 6500），说明每一轮都在重发
        全部历史，总成本是平方级的 —— 这正是需要做 context 管理的信号。
        """
        total_ms = int((time.time() - self.started_at) * 1000)
        by_tool: dict[str, dict] = {}
        for t in self.tool_calls:
            entry = by_tool.setdefault(t.name, {"calls": 0, "errors": 0, "total_ms": 0})
            entry["calls"] += 1
            entry["errors"] += int(t.is_error)
            entry["total_ms"] += t.elapsed_ms

        return {
            "llm_rounds": len(self.steps),          # 实际跑了几轮（对比 max_steps 看安全阀是否过大）
            "tool_calls": len(self.tool_calls),
            "tool_errors": sum(t.is_error for t in self.tool_calls),
            "empty_retries": sum(s.empty_retry for s in self.steps),
            "prompt_tokens_total": sum(s.prompt_tokens for s in self.steps),
            "completion_tokens_total": sum(s.completion_tokens for s in self.steps),
            # 每轮的输入大小：直接暴露"历史累积"的速度
            "prompt_tokens_growth": [s.prompt_tokens for s in self.steps],
            "by_tool": by_tool,
            "total_ms": total_ms,
            # 最终 messages 的长度 = 喂给模型的上下文规模
            "final_message_count": 0,   # 由调用方填充
        }
