# 向量数据库聊天 (Agent with Retrieval Tool)

向量数据库聊天是一种将检索增强功能集成到智能体中的强大方法，使智能体能够访问和利用存储在向量数据库中的知识。这种集成创建了一个能够基于私有数据回答问题、执行任务和提供见解的智能系统。

## 基本概念

向量数据库聊天系统结合了以下关键组件：

1. **智能体**：负责理解用户意图，规划和执行任务
2. **向量存储**：存储文档和知识的向量表示
3. **检索工具**：允许智能体查询向量存储
4. **大语言模型**：生成回答并进行推理

## 构建具有检索功能的智能体

### 基础设置

```python
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 设置LLM
llm = OpenAI(temperature=0)

# 加载文档并创建向量库
loader = DirectoryLoader("./company_docs", glob="**/*.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(split_docs, embeddings, collection_name="company_knowledge")
```

### 创建检索工具

```python
# 方法1：使用RetrieverQA链作为工具
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
)

knowledge_base_tool = Tool(
    name="公司知识库",
    func=retrieval_qa.run,
    description="当需要查询公司政策、产品信息或内部知识时使用。输入应该是一个具体问题。"
)

# 方法2：直接使用检索器作为工具
retriever_tool = Tool(
    name="文档搜索",
    func=vector_store.as_retriever().get_relevant_documents,
    description="搜索公司文档中与查询相关的信息。输入应该是一个搜索词或短语。"
)
```

### 集成其他工具

```python
from langchain.utilities import SerpAPIWrapper

# 添加网络搜索工具
search = SerpAPIWrapper()
search_tool = Tool(
    name="网络搜索",
    func=search.run,
    description="当需要查询公共信息、最新事件或公司知识库中没有的信息时使用。"
)

# 工具列表
tools = [knowledge_base_tool, search_tool]
```

### 创建智能体

```python
# 初始化智能体
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# 使用智能体回答问题
response = agent.run("我们公司的年假政策是什么？最近的市场趋势如何影响我们的产品策略？")
```

## 高级集成技巧

### 1. 对话式检索智能体

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, ConversationalAgent

# 创建记忆组件
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 创建会话模板
template = """作为一个智能助手，你可以访问公司知识库以回答关于公司政策和产品的问题，
也可以使用网络搜索来获取最新信息。在回答之前，请先考虑应该使用哪个工具。

{chat_history}
问题: {input}
{agent_scratchpad}"""

# 创建会话智能体
conversational_agent = ConversationalAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    prefix=template,
    memory=memory
)

# 创建智能体执行器
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=conversational_agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 进行多轮对话
response1 = agent_executor.run("我们公司的健康保险政策是什么？")
response2 = agent_executor.run("它覆盖家庭成员吗？")
```

### 2. 结合多种检索策略

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.bm25 import BM25Retriever

# 创建BM25检索器
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = 3

# 创建向量检索器
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 创建混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)

# 创建基于混合检索器的QA链
enhanced_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever
)

# 创建增强知识库工具
enhanced_kb_tool = Tool(
    name="增强型知识库",
    func=enhanced_qa.run,
    description="提供更全面的公司知识搜索，结合关键词和语义匹配。"
)

# 更新工具列表并重新创建智能体
enhanced_tools = [enhanced_kb_tool, search_tool]
enhanced_agent = initialize_agent(
    tools=enhanced_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

### 3. 自定义检索增强智能体

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# 定制提示模板
prompt_template = """你是一个专注于帮助用户访问内部公司信息的助手。
你有权访问公司知识库。根据需要使用这些工具，但优先使用内部知识。
如果用户问题涉及公司内部事务，应该始终先查询知识库。
如果知识库中没有相关信息，或者问题关于外部事件，请使用网络搜索。

问题: {input}

{agent_scratchpad}"""

prompt = PromptTemplate.from_template(prompt_template)

# 创建自定义智能体
custom_agent = create_react_agent(llm, enhanced_tools, prompt)
custom_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=custom_agent,
    tools=enhanced_tools,
    verbose=True,
    max_iterations=5
)

# 使用自定义智能体
custom_response = custom_agent_executor.run("我们公司有哪些可持续发展计划？与行业趋势相比如何？")
```

## 实际应用案例

### 1. 客户支持智能体

```python
# 加载客户支持文档
support_loader = DirectoryLoader("./support_docs", glob="**/*.md")
support_docs = support_loader.load()
support_splits = text_splitter.split_documents(support_docs)

# 创建支持文档向量存储
support_vectorstore = Chroma.from_documents(support_splits, embeddings, collection_name="support_knowledge")
support_retriever = support_vectorstore.as_retriever(search_kwargs={"k": 4})

# 创建客户支持QA链
support_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=support_retriever
)

# 创建客户支持工具
support_tool = Tool(
    name="客户支持知识库",
    func=support_qa.run,
    description="查询产品支持信息，包括常见问题、故障排除和使用指南。"
)

# 创建客户支持智能体
support_agent = initialize_agent(
    tools=[support_tool, search_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)
```

### 2. 文档分析智能体

```python
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain

# 创建总结链
summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
analyze_chain = AnalyzeDocumentChain(combine_docs_chain=summarize_chain)

# 创建总结工具
summarize_tool = Tool(
    name="文档总结",
    func=analyze_chain.run,
    description="总结长文档的内容，提取关键点和重要信息。"
)

# 创建文档检索工具
document_tool = Tool(
    name="文档检索",
    func=vector_store.as_retriever().get_relevant_documents,
    description="检索与特定主题相关的文档。"
)

# 创建文档分析智能体
document_agent = initialize_agent(
    tools=[document_tool, summarize_tool, search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

## 最佳实践

1. **清晰的工具描述**：为每个检索工具提供详细的描述，帮助智能体做出正确选择。

2. **上下文管理**：控制传递给LLM的检索内容量，避免超出上下文窗口。

3. **工具选择逻辑**：设计智能的工具选择策略，决定何时使用检索工具vs其他工具。

4. **错误处理**：实现robust错误处理，确保检索失败时智能体能优雅降级。

5. **源引用**：在回答中包含检索信息的来源。

```python
# 源引用示例
from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=support_retriever
)

sourced_tool = Tool(
    name="带源客户支持",
    func=qa_with_sources.run,
    description="查询产品支持信息，并提供信息来源。"
)
```

6. **性能优化**：定期重新训练和优化向量存储。

7. **多样化检索方法**：结合多种检索策略以提高覆盖率和精度。

通过有效集成检索工具，智能体可以访问庞大的知识库，大大增强了其回答问题和解决问题的能力，同时保持对私有数据的访问控制和上下文相关性。