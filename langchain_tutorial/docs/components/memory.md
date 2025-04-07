# 记忆系统 (Memory)

记忆系统是LangChain的核心组件之一，允许应用程序记住和使用过去的交互信息。本文档详细介绍了LangChain中的记忆组件类型、工作原理和使用场景。

## 概述

在构建对话式AI应用时，能够记住上下文和之前的交互至关重要。LangChain的记忆组件负责：

1. **存储对话历史**：保存之前的消息交换
2. **管理上下文窗口**：控制传递给模型的上下文量
3. **提取相关信息**：从历史中选择重要信息
4. **处理长期记忆**：管理超出上下文窗口的信息

## 基础记忆类型

### 1. 对话缓冲记忆 (ConversationBufferMemory)

最简单的记忆类型，存储所有之前的对话：

```python
from langchain_community.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 创建记忆组件
memory = ConversationBufferMemory(return_messages=True, memory_key="history")

# 创建带记忆的提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位友好的AI助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 创建模型和链
model = ChatOpenAI()
chain = prompt | model

# 首次交互
input_1 = {"input": "你好！我叫李明。"}
memory.chat_memory.add_user_message(input_1["input"])
response_1 = chain.invoke({"history": memory.load_memory_variables({})["history"], **input_1})
memory.chat_memory.add_ai_message(response_1.content)

# 第二次交互，记忆中已经有之前的对话
input_2 = {"input": "你还记得我的名字吗？"}
response_2 = chain.invoke({"history": memory.load_memory_variables({})["history"], **input_2})
```

**特点**：
- 简单易用
- 存储完整对话历史
- 随着对话增长可能超出上下文窗口

### 2. 对话摘要记忆 (ConversationSummaryMemory)

通过总结之前的对话来节省上下文空间：

```python
from langchain_community.memory import ConversationSummaryMemory

# 创建摘要记忆
summary_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(),  # 用于生成摘要的模型
    return_messages=True,
    memory_key="history"
)

# 使用摘要记忆
summary_memory.chat_memory.add_user_message("你好！我叫李明，我是一名软件工程师。")
summary_memory.chat_memory.add_ai_message("你好李明！很高兴认识你。作为软件工程师，你主要负责什么技术领域？")
summary_memory.chat_memory.add_user_message("我主要做后端开发，使用Python和Go语言，专注于微服务架构。")

# 加载记忆变量（内部会生成摘要）
summarized_history = summary_memory.load_memory_variables({})["history"]
```

**特点**：
- 通过总结节省上下文空间
- 适合长对话
- 可能丢失细节信息

### 3. 对话摘要缓冲记忆 (ConversationSummaryBufferMemory)

结合缓冲和摘要的混合方法：

```python
from langchain_community.memory import ConversationSummaryBufferMemory

# 创建摘要缓冲记忆
buffer_memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(),
    max_token_limit=100,  # 设置最大token数
    return_messages=True,
    memory_key="history"
)

# 添加一些消息
for i in range(5):
    buffer_memory.chat_memory.add_user_message(f"这是用户消息 {i}")
    buffer_memory.chat_memory.add_ai_message(f"这是AI回复 {i}")

# 当消息超过max_token_limit时，旧消息会被总结
history = buffer_memory.load_memory_variables({})["history"]
```

**特点**：
- 结合了缓冲和摘要的优点
- 自动管理上下文长度
- 保留最近消息的详细信息，总结旧消息

### 4. 向量存储记忆 (VectorStoreRetrieverMemory)

基于语义相关性检索历史消息：

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.memory import VectorStoreRetrieverMemory

# 创建向量存储
embedding = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 创建向量存储记忆
vector_memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="relevant_history"
)

# 添加记忆
vector_memory.save_context({"input": "人工智能的主要应用领域有哪些？"}, 
                          {"output": "主要应用领域包括自然语言处理、计算机视觉、机器人学、医疗健康、金融分析等。"})

# 之后可以检索相关信息
relevant_memories = vector_memory.load_memory_variables({"input": "医疗AI的具体应用是什么？"})  # 会检索与医疗相关的之前对话
```

**特点**：
- 基于语义相似性检索相关历史
- 适合非线性对话
- 可以处理大量历史信息

### 5. 实体记忆 (EntityMemory)

追踪对话中提到的实体信息：

```python
from langchain_community.memory import ConversationEntityMemory

# 创建实体记忆
entity_memory = ConversationEntityMemory(
    llm=ChatOpenAI(),
    return_messages=True
)

# 对话中会提取实体
entity_memory.save_context(
    {"input": "我的名字是张伟，我住在上海。"},
    {"output": "你好张伟！上海是一个美丽的城市。"}
)

# 提取实体信息
entities = entity_memory.load_memory_variables({})["entities"]
# 包含"张伟"和"上海"的实体信息
```

**特点**：
- 自动提取和跟踪重要实体
- 构建关于特定实体的知识库
- 适合需要记住用户具体信息的应用

## 高级用法

### 记忆链集成

使用新的`RunnableWithMessageHistory`接口将记忆组件集成到链中：

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 创建基本链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位友好的AI助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
chain = prompt | ChatOpenAI() | StrOutputParser()

# 创建存储对话历史的字典
store = {}

# 添加记忆到链
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: store.get(session_id, ChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="history"
)

# 使用带记忆的链
for i in range(3):
    response = chain_with_history.invoke(
        {"input": f"这是第{i+1}轮对话"},
        config={"configurable": {"session_id": "user-123"}}
    )
    print(response)
    # 历史会自动更新在store字典中
```

### 记忆过滤和转换

控制记忆内容的处理：

```python
from langchain_community.memory import ConversationTokenBufferMemory
from langchain_openai import OpenAI

# 创建基于token数的缓冲记忆
token_memory = ConversationTokenBufferMemory(
    llm=OpenAI(),  # 用于计算token
    max_token_limit=200,  # 最大token数
    return_messages=True,
    memory_key="history"
)

# 添加消息直到超过限制
for i in range(10):
    token_memory.chat_memory.add_user_message(f"这是一个较长的用户消息 {i} " * 5)
    token_memory.chat_memory.add_ai_message(f"这是一个AI回复 {i} " * 5)

# 获取记忆，只会包含最近的几条消息，使总token数保持在限制之内
history = token_memory.load_memory_variables({})["history"]
```

### 自定义记忆组件

创建自定义记忆系统：

```python
from langchain_core.memory import BaseMemory
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class TimeAwareMemory(BaseMemory):
    """一个记录对话时间的记忆组件"""
    
    chat_memory: List[Dict] = []  # 存储消息及其时间戳
    memory_key: str = "history"   # 输出变量名
    input_key: str = "input"      # 输入变量名
    
    def clear(self):
        """清除所有记忆"""
        self.chat_memory = []
    
    @property
    def memory_variables(self) -> List[str]:
        """定义这个记忆组件输出的变量"""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """加载记忆变量"""
        messages = []
        for entry in self.chat_memory:
            if entry["role"] == "human":
                messages.append(HumanMessage(content=entry["content"]))
            else:
                messages.append(AIMessage(content=entry["content"]))
                
        return {self.memory_key: messages}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """保存当前对话上下文"""
        import time
        
        # 保存用户输入及时间
        self.chat_memory.append({
            "role": "human",
            "content": inputs[self.input_key],
            "timestamp": time.time()
        })
        
        # 保存AI输出及时间
        self.chat_memory.append({
            "role": "ai",
            "content": outputs["output"],
            "timestamp": time.time()
        })
```

### 分布式持久化记忆

实现跨会话的持久化记忆存储：

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.memory import ConversationBufferMemory

# 使用Redis存储对话历史
message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0",
    session_id="user-123"
)

# 使用持久化的历史创建记忆
redis_memory = ConversationBufferMemory(
    chat_memory=message_history,
    return_messages=True,
    memory_key="history"
)

# 添加消息（会持久化到Redis）
redis_memory.chat_memory.add_user_message("你好，AI助手！")
redis_memory.chat_memory.add_ai_message("你好！有什么可以帮助你的？")

# 即使程序重启，历史信息也会保留
history = redis_memory.load_memory_variables({})["history"]
```

## 常见记忆模式

### 1. 默认对话记忆

适用于简单聊天机器人：

```python
from langchain.chains import ConversationChain
from langchain_community.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# 创建对话链
conversation = ConversationChain(
    llm=ChatOpenAI(),
    memory=ConversationBufferMemory(),
    verbose=True
)

# 进行对话
response1 = conversation.predict(input="你好！")
response2 = conversation.predict(input="今天天气怎么样？")
response3 = conversation.predict(input="我们之前聊了什么？")
```

### 2. 长期记忆与短期记忆结合

适用于需要长期保持用户信息的应用：

```python
from langchain_community.memory import CombinedMemory, ConversationSummaryMemory

# 创建短期记忆（详细近期消息）
short_term_memory = ConversationBufferMemory(
    memory_key="short_term_memory",
    return_messages=True
)

# 创建长期记忆（摘要形式）
long_term_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(),
    memory_key="long_term_memory",
    return_messages=True
)

# 组合记忆
combined_memory = CombinedMemory(
    memories=[short_term_memory, long_term_memory]
)

# 创建提示模板
combined_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位助手，有短期和长期记忆。\n\n长期记忆摘要:\n{long_term_memory}\n\n最近对话:\n{short_term_memory}"),
    ("human", "{input}")
])

# 创建链
chain = combined_prompt | ChatOpenAI() | StrOutputParser()

# 使用示例
def chat_with_combined_memory(user_input):
    result = chain.invoke({"input": user_input, **combined_memory.load_memory_variables({})})
    combined_memory.save_context({"input": user_input}, {"output": result})
    return result
```

### 3. 记忆库模式

为多个用户或会话管理记忆：

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

# 用户会话管理
class SessionManager:
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def get_session_memory(self, session_id):
        # 从数据库获取特定会话的历史
        message_history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=self.connection_string
        )
        
        memory = ConversationBufferMemory(
            chat_memory=message_history,
            return_messages=True,
            memory_key="history"
        )
        
        return memory

# 使用示例
session_manager = SessionManager("sqlite:///chat_history.db")

# 用户1的会话
user1_memory = session_manager.get_session_memory("user-1")
user1_chain = RunnableWithMessageHistory(
    chain,
    lambda session_id: session_manager.get_session_memory(session_id).chat_memory,
    input_messages_key="input",
    history_messages_key="history"
)

# 使用特定用户的记忆
response = user1_chain.invoke(
    {"input": "你好！"},
    config={"configurable": {"session_id": "user-1"}}
)
```

## 最佳实践

1. **根据应用场景选择记忆类型**
   - 简短对话：`ConversationBufferMemory`
   - 长对话：`ConversationSummaryMemory`或`ConversationSummaryBufferMemory`
   - 多主题对话：`VectorStoreRetrieverMemory`
   - 需要追踪特定信息：`EntityMemory`

2. **管理上下文窗口**
   - 使用token限制或自动摘要防止超出模型上下文窗口
   - 对于长对话，定期总结并清理老旧信息

3. **结合多种记忆类型**
   - 使用`CombinedMemory`混合不同记忆策略
   - 为不同数据类型使用专门的记忆组件

4. **持久化考虑**
   - 对于生产应用，实现记忆持久化
   - 使用数据库或Redis等存储对话历史

5. **注意隐私和安全**
   - 实现记忆清理和过期机制
   - 对敏感信息考虑加密存储

## 常见问题与解决方案

1. **记忆过长导致模型上下文溢出**
   - 解决方案：使用摘要记忆或设置token限制

2. **记忆不连贯或丢失信息**
   - 解决方案：调整摘要提示或使用缓冲+摘要混合方法

3. **跨会话记忆丢失**
   - 解决方案：实现持久化存储（Redis、SQL等）

4. **记忆检索效率低下**
   - 解决方案：使用向量存储和语义检索

## 高级记忆架构

对于复杂应用，可以构建分层记忆架构：

```
工作记忆（最近几轮对话）
   ↑↓
短期记忆（当前会话重要信息）
   ↑↓
长期记忆（跨会话持久信息）
   ↑↓
知识库（向量存储的相关信息）
```

## 总结

记忆系统是构建有效对话应用的关键组件。LangChain提供了多种记忆类型，从简单的缓冲记忆到复杂的向量和实体记忆，可以根据应用需求灵活选择和组合。通过正确设计记忆策略，可以显著提升模型的上下文理解能力，创造更自然、连贯的对话体验。

## 后续学习

- [提示模板](./prompt_templates.md) - 学习如何在提示中有效使用记忆
- [链](./chains.md) - 了解如何在复杂流程中集成记忆
- [检索系统](./retrieval.md) - 探索基于检索的外部记忆系统