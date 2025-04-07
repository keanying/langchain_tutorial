# 链 (Chains)

链是LangChain中的核心组件，用于将多个组件（如提示模板、语言模型、解析器等）连接成序列，构建复杂的工作流。本文档详细介绍链的概念、类型和使用方法。

## 概述

在LangChain中，链提供了将各个组件组合成端到端应用的方法。通过链，您可以：

1. **顺序执行多个组件**：按特定顺序处理数据
2. **组合不同功能**：整合模型调用、数据检索、输出处理等
3. **创建复杂工作流**：实现条件逻辑、循环和分支
4. **复用常见模式**：使用预定义链快速实现常见功能

LangChain提供了两种构建链的方式：
- **LangChain表达式语言(LCEL)**：新的声明式API（推荐）
- **传统链类**：基于类的旧API

## LCEL构建链

### 基本链构建

使用管道操作符(`|`)连接组件：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 创建组件
prompt = ChatPromptTemplate.from_template("请解释{concept}是什么？")
model = ChatOpenAI()
output_parser = StrOutputParser()

# 构建链
chain = prompt | model | output_parser

# 执行链
result = chain.invoke({"concept": "量子计算"})
```

### 数据转换和分支

使用字典和函数进行数据处理：

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 定义处理函数
def process_concept(concept):
    return concept.strip().lower()

# 构建复杂链
chain = {
    "concept": RunnableLambda(process_concept),
    "original_input": RunnablePassthrough()
} | prompt | model | output_parser

# 执行链
result = chain.invoke("量子计算")
```

### 条件逻辑

使用RunnableBranch实现分支逻辑：

```python
from langchain_core.runnables import RunnableBranch

# 定义不同提示
simple_prompt = ChatPromptTemplate.from_template("简单解释{concept}")
detailed_prompt = ChatPromptTemplate.from_template("详细解释{concept}，包括历史和应用")

# 条件函数
def is_advanced_mode(inputs):
    return inputs.get("mode") == "advanced"

# 构建条件分支
prompt_branch = RunnableBranch(
    (is_advanced_mode, detailed_prompt),
    simple_prompt  # 默认分支
)

# 完整链
chain = prompt_branch | model | output_parser

# 基本模式
basic_result = chain.invoke({"concept": "量子计算", "mode": "basic"})

# 高级模式
advanced_result = chain.invoke({"concept": "量子计算", "mode": "advanced"})
```

### 并行执行

同时执行多个操作并合并结果：

```python
# 定义不同处理链
summary_chain = ChatPromptTemplate.from_template(
    "总结{text}的主要观点"
) | model | StrOutputParser()

analysis_chain = ChatPromptTemplate.from_template(
    "分析{text}中的情感和态度"
) | model | StrOutputParser()

action_chain = ChatPromptTemplate.from_template(
    "基于{text}提出三个可行的行动建议"
) | model | StrOutputParser()

# 并行执行多个链
full_chain = {
    "summary": summary_chain,
    "analysis": analysis_chain,
    "actions": action_chain
}

# 执行
result = full_chain.invoke({"text": "我们需要提高产品质量，客户反馈显示最近的返修率上升了15%。团队应该集中精力解决这个问题。"})
```

## 预定义链

LangChain提供了多种预定义链用于常见任务。

### 1. LLM链

最基本的链，将提示模板与语言模型结合：

```python
from langchain.chains import LLMChain

llm_chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=ChatPromptTemplate.from_template("请解释{concept}是什么？"),
    verbose=True  # 显示执行详情
)

result = llm_chain.run(concept="机器学习")
```

### 2. 顺序链

连接多个链，前一个链的输出作为后一个链的输入：

```python
from langchain.chains import SequentialChain, LLMChain

# 第一个链：生成故事大纲
outline_prompt = ChatPromptTemplate.from_template("为一个关于{topic}的短篇故事创建大纲。")
outline_chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=outline_prompt,
    output_key="outline"  # 定义输出键名
)

# 第二个链：基于大纲写故事
story_prompt = ChatPromptTemplate.from_template("基于以下大纲创作一个短篇故事:\n{outline}")
story_chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=story_prompt,
    output_key="story"  # 定义输出键名
)

# 组合为顺序链
sequential_chain = SequentialChain(
    chains=[outline_chain, story_chain],
    input_variables=["topic"],  # 链的输入变量
    output_variables=["outline", "story"],  # 想要返回的变量
    verbose=True
)

result = sequential_chain({"topic": "太空探险"})
```

### 3. 路由链

根据输入内容选择不同的处理链：

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

# 定义不同领域的提示
prompt_infos = [
    {
        "name": "physics",
        "description": "处理物理学问题",
        "prompt_template": "你是物理学专家。回答以下问题：{input}"
    },
    {
        "name": "math",
        "description": "处理数学问题",
        "prompt_template": "你是数学专家。解答以下问题：{input}"
    },
    {
        "name": "history",
        "description": "处理历史问题",
        "prompt_template": "你是历史学家。回答以下历史问题：{input}"
    }
]

# 创建路由模板
router_template = """基于用户问题，选择最合适的领域。

用户问题: {input}

{format_instructions}"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    partial_variables={
        "format_instructions": RouterOutputParser().get_format_instructions()
    }
)

router_chain = LLMRouterChain.from_llm(
    llm=ChatOpenAI(),
    prompt=router_prompt
)

# 创建多提示链
destination_chains = {}
for p_info in prompt_infos:
    chain = LLMChain(
        llm=ChatOpenAI(),
        prompt=ChatPromptTemplate.from_template(p_info["prompt_template"])
    )
    destination_chains[p_info["name"]] = chain
    
# 默认链
default_chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=ChatPromptTemplate.from_template("请回答以下一般问题：{input}")
)

# 创建多提示链
multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

# 测试不同领域的问题
physics_result = multi_prompt_chain.run("光速是多少？")
history_result = multi_prompt_chain.run("谁是中国的第一位皇帝？")
```

### 4. 检索增强生成链 (RAG)

结合文档检索和生成：

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# 假设已经有一个向量数据库
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# LCEL方式构建RAG
from langchain_core.prompts import ChatPromptTemplate

template = """使用以下内容回答问题：

{context}

问题: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

lcel_rag_chain = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough()
} | prompt | ChatOpenAI() | StrOutputParser()

# 传统方式构建RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",  # 将所有文档合并为一个上下文
    retriever=retriever,
    verbose=True
)

# 执行查询
lcel_result = lcel_rag_chain.invoke("什么是量子计算？")
traditional_result = qa_chain.run("什么是量子计算？")
```

### 5. 对话链

管理对话历史的链：

```python
from langchain.chains import ConversationChain
from langchain_community.memory import ConversationBufferMemory

# 创建对话链
conversation = ConversationChain(
    llm=ChatOpenAI(),
    memory=ConversationBufferMemory(),
    verbose=True
)

# 进行对话
response1 = conversation.predict(input="你好！")
response2 = conversation.predict(input="告诉我关于人工智能的信息。")
response3 = conversation.predict(input="我们之前讨论了什么？")
```

## 高级链模式

### 1. 链中的内存管理

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 创建带记忆的提示
memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位有帮助的AI助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 创建基本链
base_chain = memory_prompt | ChatOpenAI() | StrOutputParser()

# 存储会话历史
store = {}

# 添加记忆功能
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    lambda session_id: store.get(session_id, ChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="history"
)

# 使用带记忆的链
response1 = chain_with_history.invoke(
    {"input": "你好！"},
    config={"configurable": {"session_id": "user_123"}}
)

response2 = chain_with_history.invoke(
    {"input": "我们之前说了什么？"},
    config={"configurable": {"session_id": "user_123"}}
)
```

### 2. 流式处理与异步执行

```python
# 流式处理
for chunk in chain.stream({"concept": "深度学习"}):
    print(chunk, end="", flush=True)

# 异步执行
import asyncio

async def process_queries(queries):
    tasks = [chain.ainvoke({"concept": query}) for query in queries]
    results = await asyncio.gather(*tasks)
    return results

queries = ["深度学习", "神经网络", "强化学习"]
results = asyncio.run(process_queries(queries))
```

### 3. 错误处理与重试

```python
from langchain_core.runnables import RunnableRetry
from langchain_core.pydantic_v1 import BaseModel, Field
import time

# 定义可能出错的函数
def unstable_function(x):
    # 模拟随机失败
    import random
    if random.random() < 0.5:
        raise ValueError("随机错误")
    return f"成功处理: {x}"

# 添加重试逻辑
retryable_func = RunnableLambda(unstable_function).with_retry(
    retry_policy=RunnableRetry(
        max_retries=3,
        wait_exponential_jitter=True,  # 指数退避策略
        on_retry=lambda state: print(f"重试中... 尝试 #{state['attempt']} 失败原因: {state['error']}"),
    )
)

# 构建链
robust_chain = {
    "input": RunnablePassthrough(),
    "processed": retryable_func
}

try:
    result = robust_chain.invoke("测试输入")
ex```

### 4. 调试与可视化

```python
from langchain_core.tracers import ConsoleTracer

# 使用控制台跟踪器
tracer = ConsoleTracer()

# 在调用时启用跟踪
result = chain.invoke(
    {"concept": "神经网络"},
    config={"callbacks": [tracer]}
)
```

## 链的组合模式

### 1. 自评估链

让模型评估自己的输出质量：

```python
from langchain_community.evaluation import load_evaluator

# 创建基本链
base_chain = prompt | model | output_parser

# 创建评估器
evaluator = load_evaluator("criteria", criteria="accuracy")

def evaluate_response(inputs_and_outputs):
    input_text = inputs_and_outputs["input"]
    output_text = inputs_and_outputs["output"]
    eval_result = evaluator.evaluate_strings(
        prediction=output_text,
        input=input_text
    )
    return {**inputs_and_outputs, "evaluation": eval_result}

# 构建带评估的链
chain_with_eval = {
    "input": RunnablePassthrough(),
    "output": base_chain
} | evaluate_response

# 运行并获取评估结果
result = chain_with_eval.invoke("解释量子纠缠")
```

### 2. 工具使用链

结合工具使用能力：

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 定义工具
@tool
def search(query: str) -> str:
    """搜索网络上的信息"""
    # 简化实现
    return f"关于'{query}'的搜索结果..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

# 使用工具的提示
tool_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位助手，可以使用工具帮助回答问题。"),
    ("human", "{input}")
])

# 创建能使用工具的模型
model_with_tools = ChatOpenAI().bind_tools([search, calculator])

# 构建工具链
tool_chain = tool_prompt | model_with_tools

# 测试工具使用
result = tool_chain.invoke({"input": "谁是当前的中国总理？另外，计算25乘以16等于多少？"})
```

### 3. 多阶段处理链

将问题分解为多个阶段处理：

```python
# 第一阶段：问题分解
decompose_prompt = ChatPromptTemplate.from_template(
    "将以下复杂问题分解为2-3个更简单的子问题:\n{question}"
)
decompose_chain = decompose_prompt | ChatOpenAI() | StrOutputParser()

# 第二阶段：回答子问题
answer_prompt = ChatPromptTemplate.from_template(
    "回答以下具体问题:\n{subquestion}"
)
answer_chain = answer_prompt | ChatOpenAI() | StrOutputParser()

# 第三阶段：综合答案
synthesize_prompt = ChatPromptTemplate.from_template(
    """基于以下子问题的答案，回答原始问题：
    
    原始问题：{original_question}
    
    子问题与答案：
    {subquestion_answers}
    """
)
synthesize_chain = synthesize_prompt | ChatOpenAI() | StrOutputParser()

# 将所有阶段组合起来
def process_complex_question(question):
    # 分解问题
    subquestions_text = decompose_chain.invoke({"question": question})
    subquestions = [q.strip() for q in subquestions_text.split("\n") if q.strip()]
    
    # 回答各子问题
    answers = []
    for i, subq in enumerate(subquestions):
        if not subq:
            continue
        answer = answer_chain.invoke({"subquestion": subq})
        answers.append(f"子问题 {i+1}: {subq}\n回答: {answer}")
    
    subquestion_answers = "\n\n".join(answers)
    
    # 综合答案
    final_answer = synthesize_chain.invoke({
        "original_question": question,
        "subquestion_answers": subquestion_answers
    })
    
    return {
        "original_question": question,
        "subquestions": subquestions,
        "subquestion_answers": answers,
        "final_answer": final_answer
    }

# 测试多阶段链
result = process_complex_question(
    "人工智能如何影响未来的就业市场，以及人们应该如何适应这种变化？"
)
```

## 最佳实践

### 1. 选择正确的链构建方法

- 对于新项目，优先使用LCEL（更灵活、更易于调试）
- 使用预定义链类可以快速实现常见用例
- 复杂流程可以组合使用两种方法

### 2. 链的模块化设计

- 将复杂链分解为可管理的子链
- 为常用功能创建可重用组件
- 使用描述性变量名提高可读性

### 3. 错误处理与调试

- 添加适当的错误处理逻辑
- 使用`verbose=True`或跟踪器查看链执行过程
- 实施重试策略处理临时失败

### 4. 性能优化

- 使用`batch`方法处理多个输入
- 考虑异步执行提高并发性
- 实现缓存减少重复调用

### 5. 输入验证

```python
from pydantic import BaseModel, Field, validator

class QuestionInput(BaseModel):
    question: str = Field(..., description="用户问题")
    
    @validator("question")
    def validate_question(cls, v):
        if len(v) < 3:
            raise ValueError("问题太短")
        return v

def validate_input(input_dict):
    try:
        validated = QuestionInput(**input_dict)
        return validated.dict()
    except Exception as e:
        return {"error": str(e)}

# 在链中使用输入验证
validated_chain = RunnableLambda(validate_input) | chain
```

## 常见问题与解决方案

1. **输入输出键不匹配**：确保各组件之间的输入输出变量名一致
2. **链执行太慢**：考虑并行执行、缓存或使用更高效的模型
3. **内存溢出**：限制历史长度，使用更高效的存储方法
4. **输出格式不一致**：使用输出解析器和格式指令
5. **错误传播**：实现全面的错误处理策略

## 总结

链是LangChain的核心概念，提供了组合各种组件构建复杂应用的能力。通过LCEL的声明式语法或传统的链类，您可以创建从简单问答到复杂多阶段推理的各种应用。

掌握链的构建和组合技术，可以帮助您创建更强大、更灵活和更易于维护的LLM应用。从基本的顺序执行到复杂的多阶段处理，链提供了应对各种需求的工具和模式。

选择合适的链类型和构建方法对应用性能和可维护性至关重要。随着对LangChain的深入了解，您可以创建越来越复杂和强大的应用程序。

## 后续学习

- [LCEL介绍](../lcel_intro.md) - 深入学习LangChain表达式语言
- [输出解析器](./output_parsers.md) - 了解如何结构化链的输出
- [智能体](../agents.md) - 探索如何使用链构建自主智能体
- [检索系统](./retrieval.md) - 学习结合外部知识的高级模式