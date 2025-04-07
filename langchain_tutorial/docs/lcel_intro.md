# LangChain 表达式语言 (LCEL) 介绍

LangChain 表达式语言（LCEL）是一种声明式的编程接口，用于构建基于LLM的应用。它提供了一种简洁、直观的方式来组合LangChain的各种组件，构建复杂的AI应用流程。

## 核心理念

LCEL的设计理念是让开发者能够:

1. **声明式定义流程**：清晰表达数据如何流经不同组件
2. **轻松组合组件**：通过管道操作符链接不同功能
3. **简化复杂性**：减少样板代码，聚焦业务逻辑
4. **提升可维护性**：使代码更加模块化和可测试

## 基本语法

LCEL的核心语法是使用管道操作符 `|` 将不同的组件连接起来，形成数据处理流。

### 基础示例

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 定义提示模板
prompt = ChatPromptTemplate.from_template("告诉我关于{topic}的信息")

# 定义模型
model = ChatOpenAI()

# 使用LCEL组合组件
chain = prompt | model

# 执行链
response = chain.invoke({"topic": "人工智能"})
print(response.content)
```

在这个简单的例子中，数据流如下:
- `prompt`将输入参数转换为格式化提示
- 通过管道操作符将提示传递给`model`
- `model`生成回复并返回结果

## LCEL的主要优势

### 1. 简洁的语法

传统链式编程和LCEL的对比:

**传统方式:**
```python
prompt = ChatPromptTemplate.from_template("告诉我关于{topic}的信息")
model = ChatOpenAI()

# 手动连接各组件
formatted_prompt = prompt.format(topic="人工智能")
response = model.predict(formatted_prompt)
```

**使用LCEL:**
```python
chain = prompt | model
response = chain.invoke({"topic": "人工智能"})
```

### 2. 强大的组合性

LCEL允许轻松组合多个处理步骤:

```python
from langchain_core.output_parsers import StrOutputParser

# 添加输出解析器
chain = prompt | model | StrOutputParser()

# 结果现在是字符串而非消息对象
result = chain.invoke({"topic": "人工智能"})
```

### 3. 流程可视化

LCEL链可以轻松可视化，帮助理解数据流:

```python
from langchain_core.tracers import ConsoleTracer

tracer = ConsoleTracer()
result = chain.invoke({"topic": "人工智能"}, config={"callbacks": [tracer]})
```

### 4. 并行和条件处理

LCEL支持高级流程控制:

```python
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

# 条件处理示例
def is_technical(input):
    return "技术" in input["topic"] or "编程" in input["topic"]

chain = RunnableBranch(
    (is_technical, technical_prompt | model),  # 技术主题使用专业提示
    general_prompt | model  # 默认使用通用提示
)

# 并行处理示例
chain = {
    "简介": intro_prompt | model | StrOutputParser(),
    "应用": application_prompt | model | StrOutputParser(),
    "历史": history_prompt | model | StrOutputParser()
}
```

## 常见使用模式

### 1. 基本链接模式

```python
chain = prompt | model | output_parser
```

### 2. 检索增强生成 (RAG) 模式

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = {
    "context": lambda x: retriever.get_relevant_documents(x["query"]),
    "query": lambda x: x["query"]
} | prompt | model | output_parser
```

### 3. 智能体模式

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools
)

chain = {"input": RunnablePassthrough()} | agent_executor
```

### 4. 流式处理模式

```python
for chunk in chain.stream({"topic": "人工智能"}):
    print(chunk, end="", flush=True)
```

## LCEL与传统链的对比

LCEL优势:
- 更简洁的语法
- 更强的组合性
- 更好的类型提示
- 统一的接口 (invoke, batch, stream)
- 更容易调试和可视化

传统链 (LangChain 0.1.x):
- 熟悉的面向对象方式
- 预定义的方法名

## 最佳实践

1. **从简单开始**：先构建基础链，再逐步添加复杂功能
2. **使用类型注释**：利用Python类型提示增强代码可读性
3. **模块化设计**：将复杂链拆分为可重用的子链
4. **使用配置而非硬编码**：通过配置对象传递参数和回调
5. **添加错误处理**：使用.with_retry()添加重试逻辑

## 后续学习

- [LCEL接口](./lcel_interface.md) - 深入了解LCEL的接口和类型
- [LCEL食谱](./lcel_cookbook.md) - 实用LCEL模式和示例
- [组件概述](./components.md) - 了解可用于LCEL的各种组件

LCEL是LangChain最推荐的编程接口，掌握它将大大提高您构建复杂AI应用的效率。