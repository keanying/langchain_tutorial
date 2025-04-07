# 检索器 (Retrievers)

检索器是LangChain中的一类组件，负责从各种数据源中获取与用户查询相关的信息。它们作为LLM和数据源之间的桥梁，为检索增强生成（Retrieval-Augmented Generation，RAG）系统提供了核心功能。检索器的主要任务是根据查询找到最相关的文档或信息片段，以便LLM可以使用这些信息生成更准确、更知情的回答。

## 检索器的基础概念

在LangChain中，所有检索器都实现了`BaseRetriever`接口，该接口定义了一个核心方法：

```python
def get_relevant_documents(query: str) -> List[Document]
```

这个方法接收一个查询字符串，并返回与该查询相关的文档列表。

## 主要检索器类型

### 1. 向量存储检索器

最常见的检索器类型，使用向量存储进行语义搜索：

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

# 加载文档
loader = DirectoryLoader("./data", glob="**/*.txt")
documents = loader.load()

# 文档分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

# 创建向量存储
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(split_docs, embeddings)

# 创建检索器
retriever = vector_store.as_retriever(
    search_type="similarity",  # 相似度搜索
    search_kwargs={"k": 5}    # 返回前5个结果
)

# 使用检索器
docs = retriever.get_relevant_documents("人工智能在医疗领域的应用")
```

### 2. 多查询检索器

生成多个不同的查询版本，以扩大检索范围：

```python
from langchain.retrievers import MultiQueryRetriever
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

# 创建多查询检索器
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(),
    llm=llm
)

# 使用多查询检索器
docs = multi_query_retriever.get_relevant_documents(
    "机器学习模型如何处理医疗数据？"
)
# 内部会生成多个查询版本，如：
# - "机器学习在医疗数据处理中的应用"
# - "医疗数据分析中使用的机器学习技术"
# - "如何用机器学习处理和分析医疗记录"
```

### 3. 自查询检索器

通过LLM将自然语言查询转换为结构化查询：

```python
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# 定义文档元数据结构
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="文档的来源或出处",
        type="string"
    ),
    AttributeInfo(
        name="date",
        description="文档的发布或更新日期",
        type="date"
    ),
    AttributeInfo(
        name="author",
        description="文档的作者",
        type="string"
    ),
]

# 创建自查询检索器
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_store,
    document_contents="关于医疗AI的科学文章和研究报告集合",
    metadata_field_info=metadata_field_info
)

# 使用自然语言进行复杂查询
docs = self_query_retriever.get_relevant_documents(
    "找到2022年之后由李明发表的关于医疗影像AI的文章"
)
# 内部会转换为向量查询 + 元数据过滤：{"date": {"$gte": "2022-01-01"}, "author": "李明"}
```

### 4. 上下文压缩检索器

通过两阶段检索减少无关内容：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 创建文档压缩器
compressor = LLMChainExtractor.from_llm(llm)

# 创建上下文压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever(search_kwargs={"k": 8})
)

# 使用压缩检索器
compressed_docs = compression_retriever.get_relevant_documents(
    "血糖监测的新技术有哪些？"
)
# 首先检索8个相关文档，然后提取每个文档中与查询最相关的部分
```

### 5. 混合检索器

结合多种检索策略的优势：

```python
from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.retrievers.bm25 import BM25Retriever

# 创建BM25检索器（关键词匹配）
bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = 5

# 创建向量检索器（语义匹配）
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 创建集合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# 使用集合检索器
docs = ensemble_retriever.get_relevant_documents("糖尿病患者的饮食建议")
```

## 检索器与其他组件的集成

### 检索增强问答链

```python
from langchain.chains import RetrievalQA

# 创建检索增强问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 直接将所有检索到的文档合并作为上下文
    retriever=retriever,
    return_source_documents=True  # 返回来源文档
)

# 使用问答链
result = qa_chain({"query": "最新的血糖监测技术有哪些？"})
print(result["result"])  # 打印回答
print(result["source_documents"])  # 打印来源文档
```

### 与对话记忆集成

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 创建对话记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建对话检索链
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# 进行对话
result = conversation_chain({"question": "血糖监测技术有哪些？"})
# 后续问题
follow_up = conversation_chain({"question": "这些技术的主要优缺点是什么？"})
```

## 检索器优化技巧

1. **查询重写**：使用LLM优化原始查询，生成更适合检索的查询形式。
2. **文档分块策略**：调整文档分块大小和重叠以优化检索效果。
3. **检索参数调优**：针对不同应用场景调整k值、相似度阈值等参数。
4. **结果重排序**：使用交叉编码器(cross-encoder)对初步检索结果进行重排序。
5. **结果去重与聚合**：移除冗余信息，聚合相关文档。

```python
# 查询重写示例
from langchain.retrievers import LLMRewriteRetriever

rewrite_retriever = LLMRewriteRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# 结果重排序示例
from langchain.retrievers.document_compressors import CohereRerank

compressor = CohereRerank(cohere_api_key="your-api-key")
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

## 检索器评估

评估检索器性能对于构建高质量RAG系统至关重要：

```python
from langchain.evaluation.retrieval import RetrievalEvalChain

# 创建评估数据
eval_data = [
    {"query": "糖尿病的症状有哪些？", "relevant_docs": [...]}  # 包含金标准相关文档
]

# 创建评估链
eval_chain = RetrievalEvalChain.from_llm(
    llm=llm,
    retriever=retriever
)

# 运行评估
eval_results = eval_chain.evaluate(eval_data)
```

检索器是RAG系统的核心组件，它们的性能直接影响最终生成内容的质量。通过选择合适的检索器类型并进行优化，可以显著提高智能体访问和利用知识的能力。