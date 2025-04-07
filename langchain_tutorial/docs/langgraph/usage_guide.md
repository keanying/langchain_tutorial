# LangGraph 使用指南

## 安装配置

### 安装依赖

LangGraph 可以通过 pip 安装：

```bash
pip install langgraph
```

如果您想使用最新的功能和改进，可以从源代码安装：

```bash
pip install git+https://github.com/langchain-ai/langgraph.git
```

### 环境配置

LangGraph 与 LangChain 共享许多配置设置。确保您设置了必要的环境变量，例如：

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 基本概念

### 状态

在 LangGraph 中，状态是图执行过程中传递的关键数据结构。状态可以是任何可序列化的 Python 对象，但通常是一个包含以下类型信息的字典或 Pydantic 模型：

- 对话历史
- 中间结果
- 工作内存
- 元数据

### 节点

节点是图中的处理单元，接受状态作为输入并返回修改后的状态：

- 函数节点：包装普通 Python 函数
- LLM 节点：封装与语言模型的交互
- 工具节点：提供与外部系统的集成

### 边

边定义了节点之间的连接和执行流：

- 直接边：从一个节点到另一个节点的简单流程
- 条件边：基于条件表达式的分支逻辑

## 创建简单图

以下是创建和使用简单 LangGraph 的示例：

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END

# 定义状态类型
class ConversationState(TypedDict):
    messages: list[dict]
    intermediate_steps: list

# 创建节点函数
def call_model(state: ConversationState) -> ConversationState:
    messages = state["messages"]  
    # 这里使用模型处理消息
    response = {"role": "assistant", "content": "这是一个示例回复。"}
    return {"messages": messages + [response]}

def route_based_on_intent(state: ConversationState) -> Annotated[str, ("direct_answer", "use_tool")]:
    last_message = state["messages"][-1]["content"]
    # 简单的路由逻辑
    if "工具" in last_message or "搜索" in last_message:
        return "use_tool"
    else:
        return "direct_answer"

def use_tool(state: ConversationState) -> ConversationState:
    # 这里实现工具调用逻辑
    result = "这是工具调用的结果"
    return {"intermediate_steps": state.get("intermediate_steps", []) + [result]}

# 创建图
builder = StateGraph(ConversationState)

# 添加节点
builder.add_node("call_model", call_model)
builder.add_node("route_intent", route_based_on_intent)
builder.add_node("use_tool", use_tool)

# 添加边
builder.add_edge("route_intent", "call_model", condition="direct_answer")
builder.add_edge("route_intent", "use_tool", condition="use_tool")
builder.add_edge("use_tool", "call_model")
builder.add_edge("call_model", END)

# 设置入口节点
builder.set_entry_point("route_intent")

# 编译图
graph = builder.compile()

# 使用图
initial_state = {"messages": [{"role": "user", "content": "你好！"}], "intermediate_steps": []}
result = graph.invoke(initial_state)
print(result["messages"][-1]["content"])
```

## 高级功能

### 循环和迭代

LangGraph 支持循环和迭代模式，用于实现复杂的推理过程：

```python
from langgraph.graph import StateGraph, END

def should_continue(state):
    # 检查是否需要继续迭代
    if len(state["steps"]) < 5 and not state.get("final_answer"):
        return "continue"
    else:
        return "complete"

# 在图中实现循环
builder.add_node("process_step", process_step)
builder.add_node("check_completion", should_continue)
builder.add_edge("process_step", "check_completion")
builder.add_edge("check_completion", "process_step", condition="continue")
builder.add_edge("check_completion", END, condition="complete")
```

### 并行执行

LangGraph 支持节点的并行执行：

```python
from langgraph.graph import StateGraph, END

# 定义可以并行执行的节点
def search_web(state):
    # 搜索网络的实现
    return {"web_results": "网络搜索结果"}

def query_database(state):
    # 查询数据库的实现
    return {"db_results": "数据库查询结果"}

# 合并结果
def combine_results(state):
    combined = f"Web: {state['web_results']} | DB: {state['db_results']}"
    return {"combined_results": combined}

# 在图中实现并行
builder.add_node("search_web", search_web)
builder.add_node("query_database", query_database)
builder.add_node("combine_results", combine_results)

# 从入口节点并行执行
builder.add_edge("entry", ["search_web", "query_database"])

# 当所有并行节点完成后执行合并
builder.add_edge(["search_web", "query_database"], "combine_results")
builder.add_edge("combine_results", END)
```

## 与 LangChain 集成

LangGraph 设计为与 LangChain 无缝集成：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# 创建 LangChain 组件
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("回答以下问题：{question}")
chain = prompt | model

# 在 LangGraph 节点中使用 LangChain
def process_with_langchain(state):
    question = state["question"]
    response = chain.invoke({"question": question})
    return {"answer": response.content}

# 创建图
builder = StateGraph(dict)
builder.add_node("process", process_with_langchain)
builder.add_edge("process", END)
builder.set_entry_point("process")
graph = builder.compile()

# 使用图
result = graph.invoke({"question": "北京的首都是什么？"})
print(result["answer"])
```

## 多智能体系统

LangGraph 特别适合构建多智能体系统：

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# 定义不同的智能体
def researcher_agent(state):
    # 研究员智能体实现
    context = state.get("context", "") 
    question = state["question"]
    # 使用 LLM 生成研究结果
    model = ChatOpenAI(model="gpt-4")
    research_result = model.invoke(f"作为研究员，请查找关于 {question} 的信息。已知背景：{context}")
    return {"research": research_result.content}

def writer_agent(state):
    # 撰写员智能体实现
    research = state["research"]
    question = state["question"]
    # 使用 LLM 生成文章
    model = ChatOpenAI(model="gpt-4")
    article = model.invoke(f"作为撰写员，根据以下研究结果撰写一篇关于 {question} 的文章：\n{research}")
    return {"article": article.content}

def editor_agent(state):
    # 编辑智能体实现
    article = state["article"]
    # 使用 LLM 编辑文章
    model = ChatOpenAI(model="gpt-4")
    edited_article = model.invoke(f"作为编辑，请修改和完善以下文章：\n{article}")
    return {"final_article": edited_article.content}

# 创建多智能体图
builder = StateGraph(dict)
builder.add_node("researcher", researcher_agent)
builder.add_node("writer", writer_agent)
builder.add_node("editor", editor_agent)

# 定义工作流
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "editor")
builder.add_edge("editor", END)
builder.set_entry_point("researcher")

multi_agent_system = builder.compile()

# 使用多智能体系统
result = multi_agent_system.invoke({"question": "量子计算的基本原理是什么？"})
print(result["final_article"])
```

## 调试与可视化

LangGraph 提供了调试和可视化图执行的工具：

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint import jsonable_checkpoint

# 创建带检查点的图
graph_with_checkpoints = builder.compile(checkpointer=jsonable_checkpoint)

# 执行并检查中间状态
for event, state in graph_with_checkpoints.stream({"initial": "state"}):
    print(f"Event: {event}, State: {state}")

# 可视化图结构
dot_graph = builder.to_dot()
with open("graph_visualization.dot", "w") as f:
    f.write(dot_graph)
```

## 与 LangSmith 集成

LangGraph 可以与 LangSmith 集成，提供高级监控和调试功能：

```python
import os

# 设置 LangSmith API 密钥
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-langgraph-project"

# LangGraph 将自动与 LangSmith 集成进行跟踪
```

## 最佳实践

### 状态管理

- 使用类型提示确保状态一致性
- 为复杂状态使用 Pydantic 模型
- 避免状态中的大型或不可序列化对象

### 节点设计

- 保持节点功能单一，遵循单一责任原则
- 确保节点是幂等的，同样的输入总是产生同样的输出
- 适当处理异常，避免图执行中断

### 图结构

- 从简单图开始，逐步增加复杂性
- 使用子图组织复杂的工作流
- 测试各个节点，然后再测试整个图

### 性能优化

- 使用异步节点处理 I/O 密集型操作
- 适当使用并行执行提高吞吐量
- 缓存昂贵的计算结果

## 常见问题解决

### 执行无限循环

如果您的图陷入无限循环，请检查循环条件并添加适当的终止条件：

```python
def should_continue(state):
    # 添加计数器或明确的终止条件
    if state.get("iterations", 0) < MAX_ITERATIONS and not state.get("final_answer"):
        return {"iterations": state.get("iterations", 0) + 1, "continue": True}
    else:
        return {"continue": False}
```

### 类型错误

使用类型注解和验证避免类型相关错误：

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class AgentState(BaseModel):
    messages: List[dict] = Field(default_factory=list)
    tools_results: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = None
    iterations: int = 0
```

## 进阶学习资源

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub 仓库](https://github.com/langchain-ai/langgraph)
- [示例项目和教程](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [LangChain 集成示例](https://python.langchain.com/docs/langgraph)