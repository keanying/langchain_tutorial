# LangChain 表达式语言 (LCEL) 食谱

本文档提供了LangChain表达式语言(LCEL)的实用示例和模式，帮助您解决常见问题和构建复杂应用。

## 基础模式

### 1. 简单问答链

最基础的LCEL链，将提示模板、模型和输出解析器连接起来：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("回答这个问题: {question}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

result = chain.invoke({"question": "什么是机器学习？"})
print(result)
```

### 2. 多重提示组合

将多个提示模板组合为一个，用于复杂指令：

```python
system_prompt = "你是一位有{years}年经验的{profession}专家。"
question_prompt = "请回答关于{topic}的问题: {question}"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", question_prompt)
])

chain = prompt | model | output_parser

result = chain.invoke({
    "years": "10",
    "profession": "人工智能",
    "topic": "深度学习",
    "question": "神经网络和传统机器学习有什么区别？"
})
```

### 3. 自定义函数集成

将Python函数集成到LCEL链中：

```python
from langchain_core.runnables import RunnableLambda

def get_current_weather(location):
    # 模拟天气API调用
    weather_data = {"北京": "晴朗，25°C", "上海": "多云，28°C", "深圳": "雨，30°C"}
    return weather_data.get(location, "未知")

weather_chain = RunnableLambda(get_current_weather)

# 在更大的链中使用
prompt = ChatPromptTemplate.from_template(
    "地点: {location}\n当前天气: {weather}\n根据天气情况，推荐适合的活动。"
)

chain = {
    "location": lambda x: x["location"],
    "weather": lambda x: weather_chain.invoke(x["location"])
} | prompt | model | output_parser

result = chain.invoke({"location": "北京"})
```

## 中级模式

### 1. 检索增强生成 (RAG)

基础RAG模式，从向量数据库检索相关文档并生成回答：

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 假设已有文档集合和向量数据库
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 创建RAG提示模板
template = """根据以下上下文回答问题。如果上下文中没有答案，就说不知道。

上下文:
{context}

问题: {question}

回答:"""

prompt = ChatPromptTemplate.from_template(template)

# 定义检索格式化函数
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 构建RAG链
rag_chain = {
    "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"])),
    "question": lambda x: x["question"]
} | prompt | model | output_parser

result = rag_chain.invoke({"question": "什么是量子计算？"})
```

### 2. 对话记忆整合

为LCEL链添加对话历史管理：

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

# 创建包含历史的提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位友好的AI助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model | output_parser

# 初始化对话历史
history = []

def chat(user_input):
    global history
    # 调用链并包含历史
    response = chain.invoke({"history": history, "input": user_input})
    # 更新历史
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response))
    return response

# 使用示例
print(chat("你好！"))
print(chat("我们刚才聊了什么？"))
```

### 3. 工具使用与函数调用

创建集成工具的LCEL链：

```python
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent

# 定义工具
@tool
def search(query: str) -> str:
    """搜索互联网以获取信息"""
    # 模拟搜索结果
    return f"关于{query}的搜索结果..."

@tool
def calculator(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

# 创建工具列表
tools = [search, calculator]

# 创建提示
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的AI助手，可以使用工具来帮助回答问题。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 创建代理
agent = create_openai_tools_agent(model, tools, prompt)

# 创建代理执行器
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 在LCEL链中使用
chain = {
    "input": lambda x: x["question"],
    "history": lambda x: x.get("history", [])
} | agent_executor

result = chain.invoke({"question": "谁是中国的总理？另外，计算25 * 16是多少？"})
```

## 高级模式

### 1. 多模型路由

根据输入内容选择不同的模型：

```python
from langchain_core.runnables import RunnableBranch

# 创建不同的模型
fast_model = ChatOpenAI(model="gpt-3.5-turbo")
smarter_model = ChatOpenAI(model="gpt-4")

# 定义选择逻辑
def is_complex_question(input_dict):
    question = input_dict["question"]
    complex_indicators = ["比较", "分析", "为什么", "如何", "评估"]
    word_count = len(question)
    return any(indicator in question for indicator in complex_indicators) or word_count > 50

# 创建分支路由
model_router = RunnableBranch(
    (is_complex_question, smarter_model),
    fast_model  # 默认模型
)

# 集成到链中
chain = prompt | model_router | output_parser

result = chain.invoke({"question": "请深入分析人工智能对就业市场的长期影响。"})
```

### 2. 流水线处理与重试机制

创建包含多阶段处理和错误重试的复杂链：

```python
import time
from langchain_core.runnables import RunnablePassthrough

# 自定义处理函数
def extract_keywords(text):
    # 假设实现关键词提取
    return ["关键词1", "关键词2", "关键词3"]

def search_documents(keywords):
    # 假设实现基于关键词的文档搜索
    time.sleep(1)  # 模拟搜索延迟
    return ["文档1", "文档2"]

# 添加重试逻辑的函数
retryable_search = RunnableLambda(search_documents).with_retry(
    max_retries=3,
    on_retry=lambda x: print(f"重试搜索: {x}"),
    retry_sleep=2.0
)

# 构建多阶段处理流水线
pipeline = {
    "original_query": RunnablePassthrough(),
    "keywords": lambda x: extract_keywords(x["query"]),
    "documents": lambda x: retryable_search(x["keywords"]),
    "query": lambda x: x["query"]
} | prompt | model | output_parser

result = pipeline.invoke({"query": "量子计算的商业应用"})
```

### 3. 并行处理与结果聚合

并行调用多个模型并聚合结果：

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 定义输出结构
class Analysis(BaseModel):
    summary: str = Field(description="内容摘要")
    sentiment: str = Field(description="情感分析")
    key_points: list = Field(description="关键点列表")

# 创建不同的分析链
summary_chain = ChatPromptTemplate.from_template(
    "请总结以下内容: {text}"
) | ChatOpenAI(temperature=0.1) | StrOutputParser()

sentiment_chain = ChatPromptTemplate.from_template(
    "请分析以下内容的情感(积极、消极或中性): {text}"
) | ChatOpenAI(temperature=0.1) | StrOutputParser()

key_points_chain = ChatPromptTemplate.from_template(
    "请从以下内容中提取5个关键点: {text}"
) | ChatOpenAI(temperature=0.1) | JsonOutputParser()

# 并行执行链
parallel_chain = {
    "summary": summary_chain,
    "sentiment": sentiment_chain,
    "key_points": key_points_chain
}

# 构建完整的分析链
analysis_chain = lambda x: parallel_chain.invoke({"text": x["document"]})

# 使用示例
document = """人工智能的进步速度令人震惊。近年来，大型语言模型展现了前所未有的能力，
从编写代码到创作内容，再到辅助科学研究。这些进步带来了巨大的社会效益，
但同时也引发了关于就业、隐私和安全等方面的担忧。研究人员和政策制定者
正在努力制定框架，确保AI发展以负责任和公平的方式进行。"""

result = analysis_chain({"document": document})
print(result)
```

### 4. 流式处理与增量输出

实现流式响应的LCEL链：

```python
from langchain_core.output_parsers import StrOutputParser

# 创建基本链
stream_chain = prompt | ChatOpenAI(streaming=True) | StrOutputParser()

# 流式处理示例
def process_streaming():
    question = "请写一首关于人工智能的短诗。"
    print("开始生成...")
    
    for chunk in stream_chain.stream({"question": question}):
        print(chunk, end="", flush=True)
    print("\n完成!")

# 异步流式处理
async def process_async_streaming():
    question = "请写一首关于人工智能的短诗。"
    print("开始异步生成...")
    
    async for chunk in stream_chain.astream({"question": question}):
        print(chunk, end="", flush=True)
    print("\n完成!")
```

## 实用技巧

### 1. 输入验证和预处理

```python
from pydantic import BaseModel, Field, validator

# 定义输入模型
class QuestionInput(BaseModel):
    question: str = Field(..., description="用户问题")
    language: str = Field("中文", description="回答语言")
    
    @validator("question")
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("问题不能为空")
        return v

# 输入验证函数
def validate_input(input_dict):
    try:
        validated = QuestionInput(**input_dict)
        return validated.dict()
    except Exception as e:
        return {"error": str(e)}

# 在链中使用输入验证
validated_chain = RunnableLambda(validate_input) | prompt | model | output_parser
```

### 2. 缓存和记忆模式

```python
from langchain_core.cache import InMemoryCache, SQLiteCache

# 内存缓存
memory_cache = InMemoryCache()
model_with_memory_cache = model.with_config({"cache": memory_cache})

# 持久化缓存
sqlite_cache = SQLiteCache(database_path="./cache.db")
model_with_persistent_cache = model.with_config({"cache": sqlite_cache})

# 在链中使用缓存
cached_chain = prompt | model_with_memory_cache | output_parser
```

### 3. 调试和跟踪

```python
from langchain_core.tracers import ConsoleTracer
from langchain_core.callbacks import StdOutCallbackHandler

# 创建跟踪器
console_tracer = ConsoleTracer()
stdout_handler = StdOutCallbackHandler()

# 在调用时启用跟踪
result = chain.invoke(
    {"question": "什么是深度学习？"}, 
    config={"callbacks": [console_tracer, stdout_handler]}
)

# 持久化跟踪
from langchain.callbacks import FileCallbackHandler

with open("trace.log", "w") as f:
    file_handler = FileCallbackHandler(f)
    result = chain.invoke(
        {"question": "什么是深度学习？"},
        config={"callbacks": [file_handler]}
    )
```

### 4. 错误处理模式

```python
from langchain_core.runnables import RunnableConfig

# 创建自定义错误处理器
def handle_error(error, **kwargs):
    print(f"发生错误: {type(error).__name__}: {str(error)}")
    return "抱歉，处理您的请求时出现了错误。"

# 在配置中使用错误处理器
config = RunnableConfig(
    callbacks=[],
    error_handling={
        "handler": handle_error
    }
)

# 使用配置
result = chain.invoke({"question": "复杂问题"}, config=config)
```

## 实际应用示例

### 1. 智能客服机器人

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 初始化组件
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的客服代表，帮助解决产品问题。使用友好专业的语气。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

model = ChatOpenAI(temperature=0.3)

# 创建基本链
customer_service_chain = prompt | model | StrOutputParser()

# 模拟对话系统
class CustomerServiceBot:
    def __init__(self):
        self.history = []
    
    def respond(self, user_question):
        response = customer_service_chain.invoke({
            "history": self.history,
            "question": user_question
        })
        
        # 更新对话历史
        self.history.append(HumanMessage(content=user_question))
        self.history.append(AIMessage(content=response))
        
        return response
    
    def clear_history(self):
        self.history = []
        return "对话历史已清除"
```

### 2. 文档分析系统

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 文档加载和处理
def process_document(file_path):
    # 加载文档
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("不支持的文件类型")
        
    documents = loader.load()
    
    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    return splits

# 向量存储和检索
def create_retrieval_chain(splits):
    # 创建向量存储
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    # 创建RAG链
    template = """根据以下上下文回答问题：
    {context}
    
    问题: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = {
        "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"])),
        "question": lambda x: x["question"]
    } | prompt | model | StrOutputParser()
    
    return rag_chain
```

## 结论

LCEL提供了强大而灵活的组件组合方式，使您能够构建从简单到复杂的各种AI应用。随着您对这些模式的掌握，可以创建更高效、可维护和功能丰富的应用程序。

本食谱中的模式可以混合搭配，根据您的具体需求进行调整。随着LangChain的不断发展，更多强大的模式将会出现，进一步扩展LCEL的能力。

如需了解更多关于LCEL的详细信息，请参考[LCEL接口](./lcel_interface.md)文档。