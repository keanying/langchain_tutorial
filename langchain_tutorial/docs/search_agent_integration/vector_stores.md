# 向量存储 (Vector Stores)

向量存储是LangChain中的一种特殊数据库，专门用于存储和检索向量嵌入（vector embeddings）。这些向量嵌入是文本或其他数据的数值表示，能够捕捉语义相似性，使得基于语义的搜索和检索成为可能。向量存储是构建强大知识检索系统的基础。

## 向量存储的工作原理

向量存储的核心功能是：
1. **存储向量嵌入**：将文档或文本转换为高维向量并存储
2. **相似性搜索**：基于向量之间的距离或相似度检索最相关的内容

## 常用向量存储

### 1. Chroma

Chroma是一个开源的向量数据库，专为AI应用程序设计，使用简单且功能强大。

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# 加载文档
loader = TextLoader("path/to/document.txt")
documents = loader.load()

# 将文档分割成块
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 初始化嵌入模型
embeddings = OpenAIEmbeddings()

# 创建Chroma向量存储
vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# 执行相似性搜索
query = "人工智能的应用有哪些？"
results = vector_store.similarity_search(query, k=3)  # 返回3个最相关的文档
```

### 2. FAISS

FAISS (Facebook AI Similarity Search) 是一个高效的向量相似性搜索库，特别适合处理大规模向量集合。

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 使用HuggingFace的嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 从文档创建FAISS向量存储
vector_store = FAISS.from_documents(docs, embeddings)

# 保存向量存储到磁盘
vector_store.save_local("faiss_index")

# 稍后加载向量存储
loaded_vector_store = FAISS.load_local("faiss_index", embeddings)

# 执行向量搜索
results = loaded_vector_store.similarity_search(query, k=5)
```

### 3. Pinecone

Pinecone是一个托管的向量数据库服务，提供高性能和可扩展性。

```python
from langchain.vectorstores import Pinecone
import pinecone

# 初始化Pinecone
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# 创建或连接到索引
index_name = "langchain-demo"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # OpenAI嵌入的维度

# 创建向量存储
vector_store = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# 搜索相似内容
results = vector_store.similarity_search(query, k=3)
```

## 向量存储的高级功能

### 元数据过滤

大多数向量存储支持基于元数据的过滤，允许您结合语义搜索和结构化查询：

```python
# 使用元数据过滤进行搜索
results = vector_store.similarity_search(
    query,
    k=5,
    filter={"source": "internal_document", "date": {"$gte": "2023-01-01"}}
)
```

### 最大边际相关性(MMR)搜索

MMR搜索算法平衡相关性和多样性，防止冗余结果：

```python
results = vector_store.max_marginal_relevance_search(
    query,
    k=5,
    fetch_k=20,  # 初始检索更多候选，然后应用MMR
    lambda_mult=0.5  # 控制多样性与相关性的平衡
)
```

## 向量存储与其他LangChain组件的集成

### 与检索器(Retrievers)集成

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 创建检索器
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 创建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 使用问答链
response = qa_chain({"query": query})
```

### 与智能体集成

```python
from langchain.agents import Tool, initialize_agent, AgentType

# 创建一个基于向量存储的工具
vector_store_tool = Tool(
    name="DocumentSearch",
    func=retriever.get_relevant_documents,
    description="搜索与查询相关的内部文档和知识库。"
)

# 初始化智能体
agent = initialize_agent(
    tools=[vector_store_tool],
    llm=OpenAI(),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 使用智能体
agent.run("关于我们公司的人工智能战略，有哪些重点？")
```

## 最佳实践

1. **选择合适的嵌入模型**：不同的嵌入模型有各自的优缺点，根据语言、领域和性能需求选择。
2. **文档分块策略**：合理的文档分块大小和重叠很重要，影响检索质量。
3. **向量存储选择**：根据规模、性能需求和预算选择合适的向量存储。
4. **组合搜索策略**：考虑结合全文搜索、向量搜索和元数据过滤。
5. **评估和优化**：定期评估检索质量，调整参数以优化性能。

向量存储是构建现代AI应用程序的基础组件，它使智能体能够有效地检索和利用大量结构化和非结构化信息，大大增强了智能体的能力和用户体验。