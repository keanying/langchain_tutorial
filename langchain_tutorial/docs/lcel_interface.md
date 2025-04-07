# LangChain 表达式语言 (LCEL) 接口

本文档详细介绍了LangChain表达式语言(LCEL)的核心接口，帮助开发者更深入地理解和使用这一强大工具。

## Runnable接口

LCEL的核心是`Runnable`接口，它为所有可组合的组件提供了统一的接口。几乎所有LangChain组件都实现了这一接口。

### 主要方法

`Runnable`接口定义了以下核心方法：

1. **invoke**：同步执行组件
   ```python
   result = runnable.invoke(input)
   ```

2. **ainvoke**：异步执行组件
   ```python
   result = await runnable.ainvoke(input)
   ```

3. **batch**：批量处理多个输入
   ```python
   results = runnable.batch([input1, input2, input3])
   ```

4. **abatch**：异步批量处理
   ```python 
   results = await runnable.abatch([input1, input2, input3])
   ```

5. **stream**：流式处理输出
   ```python
   for chunk in runnable.stream(input):
       print(chunk)
   ```

6. **astream**：异步流式处理
   ```python
   async for chunk in runnable.astream(input):
       print(chunk)
   ```

### 配置选项

所有方法都接受一个可选的`config`参数，用于自定义执行行为：

```python
result = runnable.invoke(input, config={
    "callbacks": [my_callback],
    "run_name": "My Custom Run",
    "tags": ["production", "important"],
    "metadata": {"user_id": "123"}
})
```

主要配置选项包括：

- **callbacks**：执行期间调用的回调函数列表
- **run_name**：为此次运行命名（用于跟踪与调试）
- **tags**：为此次运行添加标签
- **metadata**：附加到此次运行的元数据
- **configurable**：特定组件的配置参数

## 组件修饰器

LCEL提供了多种修饰器方法，可以在不改变基础组件的情况下修改其行为：

### 1. 配置修改

```python
# 改变模型参数
model = ChatOpenAI()
creative_model = model.with_config({"temperature": 0.9})
conservative_model = model.with_config({"temperature": 0.1})
```

### 2. 添加重试逻辑

```python
# 添加自动重试
robust_model = model.with_retry(
    retry_if_exception_type=ServerUnavailableError,
    max_retries=3,
    exponential_backoff=True
)
```

### 3. 添加回调

```python
# 添加日志记录
from langchain.callbacks import StdOutCallbackHandler
logging_model = model.with_listeners([StdOutCallbackHandler()])
```

### 4. 添加后处理

```python
# 添加结果后处理
def postprocess(result):
    return result.upper()

uppercase_chain = chain.with_handlers(on_end=postprocess)
```

## 核心组合器

LCEL提供了几种关键的组合器，用于构建复杂的执行流程：

### 1. 管道（|）

最基本的组合方式，将一个组件的输出传递给下一个组件：

```python
chain = prompt | model | output_parser
```

等同于：

```python
chain = prompt | (model | output_parser)
chain = (prompt | model) | output_parser
```

### 2. RunnableMap（字典组合）

并行执行多个路径，并将结果合并为字典：

```python
from langchain_core.runnables import RunnableMap, RunnablePassthrough

chain = {
    "summary": document | summarize_chain,
    "entities": document | extract_entities_chain,
    "original": RunnablePassthrough()
}
```

### 3. RunnableSequence（序列组合）

显式定义执行序列：

```python
from langchain_core.runnables import RunnableSequence

chain = RunnableSequence([
    prompt,
    model,
    output_parser
])
```

### 4. RunnableParallel（并行执行）

并行执行多个组件：

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({
    "summary": summarize_chain,
    "translation": translate_chain
})
```

### 5. RunnableBranch（条件分支）

基于条件选择执行路径：

```python
from langchain_core.runnables import RunnableBranch

chain = RunnableBranch(
    (lambda x: len(x) > 1000, long_text_chain),
    (lambda x: "query" in x, question_chain),
    short_text_chain  # 默认分支
)
```

### 6. RunnableLambda（自定义函数）

集成自定义Python函数：

```python
from langchain_core.runnables import RunnableLambda

def my_function(input):
    # 处理输入
    return processed_result

processor = RunnableLambda(my_function)
chain = prompt | model | processor
```

## 输入输出类型系统

LCEL具有灵活的类型系统，支持以下主要输入输出类型：

1. **字典（Dict）**：最常用的输入格式，适用于多参数场景
   ```python
   chain.invoke({"query": "人工智能", "max_tokens": 100})
   ```

2. **字符串（String）**：简单输入的常见格式
   ```python
   model.invoke("人工智能的历史是什么？")
   ```

3. **消息列表（List[BaseMessage]）**：聊天模型的原生格式
   ```python
   model.invoke([HumanMessage(content="你好"), AIMessage(content="你好！有什么可以帮助你的吗？")])
   ```

4. **文档列表（List[Document]）**：检索和文档处理的标准格式
   ```python
   summarizer.invoke(retrieved_docs)
   ```

### 类型转换和适配

LCEL会自动处理许多类型转换，但有时需要显式转换：

```python
from langchain_core.runnables import RunnablePassthrough

# 将字符串输入转换为dict
chain = RunnablePassthrough.assign(query=lambda x: x) | retrieval_chain
result = chain.invoke("人工智能")
```

## 高级功能

### 1. 运行时配置

```python
# 创建配置项
from langchain_core.runnables import ConfigurableField

configurable_chain = chain.configurable_fields(
    temperature=ConfigurableField(id="temperature", name="Temperature", description="模型创造性")
)

# 配置使用
configured = configurable_chain.with_config({"configurable": {"temperature": 0.9}})
result = configured.invoke(input)
```

### 2. 工具集成

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

# 集成到链中  
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[calculator]
)
```

### 3. 自定义Runnable

创建自定义Runnable类：

```python
from langchain_core.runnables import Runnable

class MyCustomRunnable(Runnable):
    def __init__(self, some_param):
        self.some_param = some_param
        
    def invoke(self, input, config=None):
        # 处理逻辑
        return processed_result
```

## 常见模式与最佳实践

### 1. 输入预处理

```python
def preprocess(input_dict):
    # 处理输入
    return processed_input

chain = RunnableLambda(preprocess) | prompt | model
```

### 2. 结果后处理

```python
def extract_key_points(model_output):
    # 提取关键点
    return key_points

chain = prompt | model | RunnableLambda(extract_key_points)
```

### 3. 缓存结果

```python
from langchain_core.cache import InMemoryCache

model.cache = InMemoryCache()
# 第二次调用相同输入会使用缓存
model.invoke("中国的首都是哪里？")
```

### 4. 跟踪与调试

```python
from langchain.callbacks import ConsoleCallbackHandler

# 跟踪执行过程  
result = chain.invoke(input, config={"callbacks": [ConsoleCallbackHandler()]})
```

## 常见问题与解决方案

1. **类型不匹配**：使用RunnableLambda或RunnablePassthrough进行转换
2. **复杂输入处理**：使用字典解构和RunnableMap组合多个输入源
3. **条件逻辑**：使用RunnableBranch实现基于输入的不同处理路径
4. **错误处理**：使用with_retry和错误回调处理异常情况
5. **性能优化**：使用batch处理和缓存机制提升处理效率

## 总结

LCEL接口提供了一种强大而灵活的方式来组合LangChain组件。通过掌握Runnable接口和各种组合器，开发者可以构建复杂而优雅的AI应用程序。

更多实际应用示例，请参考[LCEL食谱](./lcel_cookbook.md)。