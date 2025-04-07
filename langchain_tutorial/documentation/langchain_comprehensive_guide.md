# LangChain 框架全面指南

## 1. LangChain 框架概述

LangChain 是一个用于开发由大型语言模型（LLMs）驱动的应用程序的框架，它提供了一系列组件和工具，使开发者能够创建复杂的、交互式的、基于语言模型的应用。

### 1.1 框架核心理念

LangChain 的设计理念围绕以下几个核心原则：

- **组件化设计**：提供模块化的组件，可以独立使用或组合成复杂的系统
- **与语言模型的无缝集成**：优化与各种语言模型的交互方式
- **链式处理**：允许将多个组件组合成处理管道
- **状态管理**：提供记忆组件以维护对话历史和状态
- **工具集成**：允许语言模型与外部工具和系统交互

### 1.2 LangChain 表达式语言 (LCEL)

LangChain 表达式语言是一种声明式语言，用于组合 LangChain 的各种组件，具有以下特点：

- 使用管道操作符 (`|`) 连接组件
- 支持同步和异步操作
- 内置错误处理和重试机制
- 支持流式传输和批处理
- 简化复杂链的构建过程

示例：
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

result = chain.invoke({"topic": "人工智能"})
```

## 2. LangChain 核心组件

LangChain 框架由多个核心组件构成，每个组件负责特定的功能：

### 2.1 模型 (Models)

模型组件是 LangChain 的核心，包括语言模型和嵌入模型：

#### 2.1.1 语言模型 (LLMs/Chat Models)

- **LLM**：文本输入，文本输出的模型（如 GPT-3.5, Llama 2）
- **ChatModel**：结构化输入（消息），结构化输出的模型（如ChatGPT, Claude）

示例：
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# 传统LLM
llm = OpenAI(temperature=0)
result = llm.predict("一个 AI 走进了酒吧...")

# 聊天模型
chat_model = ChatOpenAI(temperature=0)
messages = [
    SystemMessage(content="你是一个有幽默感的助手"),
    HumanMessage(content="讲一个 AI 笑话")
]
response = chat_model.predict_messages(messages)
```

#### 2.1.2 嵌入模型 (Embeddings)

嵌入模型将文本转换为数值向量，用于语义搜索和其他相似性比较：

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("Hello world")
```

### 2.2 提示模板 (Prompts)

提示模板用于构建结构化的提示：

- **PromptTemplate**：构建简单的文本提示
- **ChatPromptTemplate**：构建聊天消息格式的提示
- **支持变量和条件逻辑**：动态构建提示

示例：
```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# 基本提示模板
template = "给我提供关于{topic}的摘要，长度约{length}个字"
prompter = PromptTemplate.from_template(template)
prompt = prompter.format(topic="量子计算", length="100")

# 聊天提示模板
template = "你是{role}，请回答关于{topic}的问题"
chat_prompter = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{question}")
])
```

### 2.3 记忆 (Memory)

记忆组件用于管理对话历史或持久状态：

- **ConversationBufferMemory**：存储完整对话历史
- **ConversationSummaryMemory**：存储对话摘要以节省空间
- **VectorStoreRetrieverMemory**：使用向量存储实现的语义记忆

示例：
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=ChatOpenAI(),
    memory=memory,
    verbose=True
)

conversation.predict(input="你好，我叫王小明")
conversation.predict(input="你记得我的名字吗？")
```

### 2.4 检索 (Retrievers)

检索组件用于从各种数据源获取相关信息：

- **向量存储检索**：基于语义相似度检索文档
- **多查询检索**：使用多个不同查询增强检索结果
- **上下文压缩检索**：删减不相关内容以优化上下文窗口

示例：
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(["内容1", "内容2", "内容3"], embeddings)

# 基本检索器
retriever = vectorstore.as_retriever()

# 上下文压缩检索器
compressor = LLMChainExtractor.from_llm(ChatOpenAI())
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 2.5 输出解析器 (Output Parsers)

输出解析器将语言模型的输出转换为结构化的格式：

- **PydanticOutputParser**：解析为 Pydantic 模型
- **StrOutputParser**：提取纯文本输出
- **JsonOutputParser**：解析 JSON 格式输出

示例：
```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class Movie(BaseModel):
    title: str = Field(description="电影标题")
    director: str = Field(description="导演姓名")
    year: int = Field(description="上映年份")

class MovieList(BaseModel):
    movies: List[Movie] = Field(description="电影列表")

parser = PydanticOutputParser(pydantic_object=MovieList)
```

## 3. 单智能体系统

### 3.1 智能体架构

智能体系统是 LangChain 中的高级组件，允许语言模型使用工具并进行推理。智能体架构包括：

- **智能体 (Agent)**：决定下一步行动的语言模型
- **工具 (Tools)**：智能体可以使用的函数
- **执行器 (AgentExecutor)**：协调智能体与工具之间的交互

### 3.2 主要智能体类型

#### 3.2.1 ReAct 智能体

ReAct（Reasoning + Acting）智能体结合了推理和行动的能力，是最常用的智能体类型之一。

**特点：**
- 结合推理（思考）和行动的能力
- 提供中间推理步骤
- 支持"链式思考"过程

```python
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper

# 创建工具
wikipedia = WikipediaAPIWrapper()
tools = [Tool(name="维基百科", func=wikipedia.run, description="用于查询信息的工具")]

# 创建LLM
llm = ChatOpenAI(temperature=0)

# 创建记忆组件
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 初始化ReAct智能体
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)
```

#### 3.2.2 OpenAI函数智能体

OpenAI函数智能体利用OpenAI模型的函数调用能力，提供更结构化的工具使用方式。

**特点：**
- 基于OpenAI的函数调用API
- 工具调用更加可靠
- 减少解析错误和幻觉

```python
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI

# 创建工具
tools = [Tool(name="计算器", func=lambda x: eval(x), description="用于数学计算")]

# 创建OpenAI函数智能体
llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)
```