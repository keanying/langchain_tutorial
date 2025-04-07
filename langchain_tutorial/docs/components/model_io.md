# 模型输入输出 (Model I/O)

模型输入输出是LangChain的核心组件，负责处理与各种语言模型的交互。本文档详细介绍了这些组件的功能和使用方法。

## 概述

模型输入输出组件负责：

1. **连接各种语言模型**：统一不同提供商的模型接口
2. **格式化输入**：将原始输入转换为模型可理解的格式
3. **处理输出**：解析和格式化模型的输出内容

这些组件构成了LangChain应用程序的基础，是构建各种AI应用的起点。

## 主要组件

### 1. 语言模型 (LLMs)

语言模型组件处理文本到文本的生成任务：

```python
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.7)
result = llm.invoke("介绍一下中国的四大发明")
print(result)
```

主要特点：
- 接收文本输入，返回文本输出
- 支持参数调整（温度、最大生成长度等）
- 提供多种模型类型和供应商选择

支持的主要模型：
- **OpenAI** - GPT-3.5, GPT-4等
- **Anthropic** - Claude系列
- **百度文心** - ERNIE-Bot系列
- **阿里通义** - 通义千问系列
- **本地模型** - Hugging Face模型、LLaMA等

### 2. 聊天模型 (Chat Models)

聊天模型组件专门处理多轮对话交互：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOpenAI()
messages = [
    SystemMessage(content="你是一位友好的AI助手"),
    HumanMessage(content="你好！请介绍一下自己")
]
response = chat.invoke(messages)
print(response.content)
```

主要特点：
- 使用消息列表作为输入（系统消息、用户消息、助手消息）
- 返回结构化消息对象
- 支持多轮对话上下文

消息类型：
- **SystemMessage**: 设置模型行为和角色的系统指令
- **HumanMessage**: 用户输入信息
- **AIMessage**: AI回复内容
- **FunctionMessage**: 函数调用结果
- **ToolMessage**: 工具使用的消息

### 3. 提示模板 (Prompt Templates)

提示模板用于动态构建发送给模型的提示：

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# 文本提示模板
template = PromptTemplate.from_template("请介绍{country}的{topic}")
prompt = template.format(country="中国", topic="传统节日")

# 聊天提示模板
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}专家，擅长解答{field}问题"),
    ("human", "请回答：{question}")
])
messages = chat_template.format_messages(
    role="历史",
    field="中国古代文化",
    question="唐朝的科举制度是怎样的？"
)
```

主要特点：
- 支持变量插入
- 支持条件逻辑
- 允许组合多个模板

### 4. 输出解析器 (Output Parsers)

输出解析器将模型输出转换为结构化数据：

```python
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# 简单字符串解析
parser = StrOutputParser()

# 结构化数据解析
class Movie(BaseModel):
    title: str = Field(description="电影标题")
    director: str = Field(description="导演姓名")
    year: int = Field(description="上映年份")
    rating: float = Field(description="评分（1-10）")

structured_parser = PydanticOutputParser(pydantic_object=Movie)

# 在格式化指令中使用
format_instructions = structured_parser.get_format_instructions()
template = """生成一部经典电影的信息。
{format_instructions}
电影类型: {genre}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["genre"],
    partial_variables={"format_instructions": format_instructions}
)
```

主要解析器类型：
- **StrOutputParser**: 简单字符串提取
- **PydanticOutputParser**: 将输出解析为Pydantic模型
- **JsonOutputParser**: 将输出解析为JSON结构
- **XMLOutputParser**: 将输出解析为XML结构
- **CommaSeparatedListOutputParser**: 将输出解析为逗号分隔的列表

## 高级用法

### 流式处理

流式处理允许逐步接收模型生成的内容：

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(streaming=True)

for chunk in chat.stream("写一首关于人工智能的短诗"):
    print(chunk.content, end="")
```

### 模型参数调整

调整模型的生成行为：

```python
from langchain_openai import ChatOpenAI

# 创造性参数调整
creative_model = ChatOpenAI(
    model="gpt-4",
    temperature=0.9,  # 提高随机性
    max_tokens=2000,  # 设置最大生成长度
    top_p=0.95,  # 控制词汇多样性
    frequency_penalty=0.5  # 减少重复
)

# 精确模式
precise_model = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,  # 降低随机性，更确定性的输出
    max_tokens=500
)
```

### 使用多模型

在不同场景使用不同模型：

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_qianfan import ChatQianfan

models = {
    "general": ChatOpenAI(model="gpt-3.5-turbo"),  # 通用对话
    "complex": ChatOpenAI(model="gpt-4"),  # 复杂推理
    "creative": ChatAnthropic(model="claude-3-opus-20240229"),  # 创意写作
    "chinese": ChatQianfan()  # 中文处理优化
}

def get_appropriate_model(task):
    if task == "creative_writing":
        return models["creative"]
    elif task == "complex_reasoning":
        return models["complex"]
    elif task == "chinese_content":
        return models["chinese"]
    else:
        return models["general"]
```

### 函数调用

让模型选择并调用函数：

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_weather(location: str, unit: str = "celsius"):
    """获取指定位置的天气信息"""
    # 这里应该有实际的API调用，这里用模拟数据
    weather_data = {"北京": "晴朗，25°C", "上海": "多云，28°C"}
    return weather_data.get(location, "未知地区")

@tool
def calculate(expression: str):
    """计算数学表达式"""
    return eval(expression)

# 设置模型和工具
model = ChatOpenAI()
tools = [get_weather, calculate]

# 使用函数调用
response = model.invoke(
    "北京今天的天气怎么样？另外，计算一下25乘以4等于多少？",
    tools=tools
)

print(response)
```

### 处理上下文窗口限制

处理长文本输入：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# 长文本处理
def process_long_text(text):
    # 分割文本为小块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    
    # 处理各个文本块
    model = ChatOpenAI()
    results = []
    
    for chunk in chunks:
        response = model.invoke(f"请总结以下内容：{chunk}")
        results.append(response.content)
    
    # 合并结果
    combined = "\n\n".join(results)
    final_summary = model.invoke(f"请将以下多个摘要整合为一个连贯的总结：{combined}")
    
    return final_summary.content
```

## 集成实例

### 基本链式处理

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 创建基本组件
prompt = ChatPromptTemplate.from_template("请用中文回答：{question}")
model = ChatOpenAI()
output_parser = StrOutputParser()

# LCEL组合
chain = prompt | model | output_parser

# 执行链
result = chain.invoke({"question": "人工智能的发展历史是什么？"})
print(result)
```

### 使用记忆组件

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 创建带有历史的提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位友好的AI助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 创建链
chain = prompt | model | output_parser

# 添加记忆组件
store = {}
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history"
)

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="history"
)

# 使用带历史的链
response1 = chain_with_history.invoke(
    {"input": "你好！"},
    config={"configurable": {"session_id": "user1"}}
)

response2 = chain_with_history.invoke(
    {"input": "我们刚才聊了什么？"},
    config={"configurable": {"session_id": "user1"}}
)
```

## 总结

模型输入输出组件是LangChain的基础构建块，它们提供与语言模型交互的标准接口。通过这些组件，您可以：

1. 连接各种大语言模型
2. 动态构建提示
3. 解析和结构化输出
4. 处理多轮对话
5. 实现高级功能如流式处理、函数调用等

正确使用这些组件是构建高效且可靠的LLM应用的关键。根据不同的应用场景，组合使用这些组件，可以满足各种复杂需求。

## 后续学习

- [提示模板](./prompt_templates.md) - 进一步了解提示工程
- [输出解析器](./output_parsers.md) - 深入学习输出处理
- [记忆系统](./memory.md) - 学习管理对话历史
- [链](./chains.md) - 组合多个组件构建复杂应用