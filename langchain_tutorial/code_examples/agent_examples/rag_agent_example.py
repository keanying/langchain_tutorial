# LangChain RAG与智能体结合示例

import os
from typing import List
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import  Tool
from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import tempfile
from langchain_core.tools import tool

# 确保设置环境变量
os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["OPENAI_BASE_URL"] = "https://aigptx.top/v1/"


class RAGAgentExample:
    """
    检索增强生成(RAG)与智能体结合示例
    展示如何将RAG系统集成到智能体工作流中
    """

    def __init__(self):
        self.llm = ChatOpenAI(temperature=1, max_tokens=2000, model='gpt-3.5-turbo-0125')
        self.documents = []
        self.vector_store = None
        self.retriever = None

    def load_documents(self, texts: List[str]):
        """加载文档到系统中"""
        print("加载文档...")

        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        file_paths = []

        # 将文本写入临时文件
        for i, text in enumerate(texts):
            file_path = os.path.join(temp_dir, f"doc_{i}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
            file_paths.append(file_path)

        # 加载文档
        for file_path in file_paths:
            loader = TextLoader(file_path, encoding='utf-8')
            self.documents.extend(loader.load())

        print(f"已加载 {len(self.documents)} 个文档")

    def process_documents(self):
        """处理文档：分块和创建向量存储"""
        if not self.documents:
            raise ValueError("请先加载文档")

        print("处理文档：分块和向量化...")

        # 分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(self.documents)

        # 创建向量存储
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        print(f"文档处理完成，共创建 {len(chunks)} 个文本块")

        return self.retriever

    def create_standalone_rag_chain(self):
        """创建独立的RAG链"""
        if not self.retriever:
            raise ValueError("请先处理文档")

        # 创建RAG提示模板
        rag_prompt = PromptTemplate.from_template("""
        请使用以下检索的信息回答问题。如果检索的信息中没有答案，请说明你不知道。

        检索的信息:
        {context}

        问题: {question}
        """)

        # 定义文档格式化函数
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 创建RAG链
        rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | rag_prompt
                | self.llm
        )

        return rag_chain

    @tool(description="将RAG系统封装为智能体工具")
    def create_rag_tool(self):
        """将RAG系统封装为智能体工具"""
        rag_chain = self.create_standalone_rag_chain()

        def query_knowledge_base(query: str) -> str:
            """查询知识库获取信息"""
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, "content") else str(result)

        # 创建RAG工具
        rag_tool = Tool(
            name="知识库查询",
            func=query_knowledge_base,
            description="当你需要查询特定领域知识时使用此工具"
        )

        return rag_tool

    @tool(description="执行自定义REPL操作，处理自然语言指令并返回结果")
    def repl_tool(self):
        return PythonREPLTool()

    def create_agent_with_rag(self):
        """创建集成RAG的智能体"""
        tools = [self.create_rag_tool, self.repl_tool]
        langgraph_agent_executor = create_react_agent(model=self.llm, tools=tools)
        return langgraph_agent_executor


# 使用示例
if __name__ == "__main__":
    print("=== LangChain RAG与智能体结合示例 ===")

    # 1. 运行直接RAG问答示例
    print("\n===== 直接RAG问答示例 =====")

    # 创建RAG系统
    rag_system = RAGAgentExample()

    # 加载文档
    documents = [
        """
        LangChain是一个用于开发由语言模型驱动的应用程序的框架。它的核心组件包括:
        
        1. 模型I/O (Models): 与各种语言模型提供商的集成接口。
        2. 提示管理 (Prompts): 优化和管理提示模板。 
        3. 记忆系统 (Memory): 维护对话状态和历史。
        4. 索引和检索 (Indexes): 使用向量存储等技术组织和检索信息。
        5. 链 (Chains): 将多个组件串联成管道。
        6. 智能体 (Agents): 允许LLM决定使用哪些工具来完成任务。
        """,
        """
        LangChain表达式语言(LCEL)是一种声明式语言，用于组合LangChain的组件。它具有以下特性:
        
        1. 链式API: 使用管道操作符(|)连接组件。
        2. 并行执行: 可以同时执行多个操作并合并结果。
        3. 错误处理: 内置重试和异常处理机制。
        4. 批处理: 一次处理多个输入以提高效率。
        5. 流式传输: 支持逐步生成和处理响应。
        
        示例: chain = prompt | llm | output_parser
        """
    ]
    rag_system.load_documents(documents)
    rag_system.process_documents()

    # 创建RAG链
    rag_chain = rag_system.create_standalone_rag_chain()

    # 查询
    queries = [
        "LangChain的核心组件有哪些?",
        "什么是LCEL，它有什么特点?"
    ]

    for query in queries:
        print(f"\n问题: {query}")
        response = rag_chain.invoke(query)
        print(f"回答: {response.content if hasattr(response, 'content') else response}")

    # 2. 运行智能体与RAG集成示例
    print("\n===== 智能体与RAG集成示例 =====")

    # 创建智能体
    agent = rag_system.create_agent_with_rag()

    # 查询
    agent_queries = [
        "总结LangChain的核心组件并使用Python代码计算它们的总数",
        "请解释LCEL中的管道操作符是如何工作的"
    ]

    for query in agent_queries:
        print(f"\n智能体查询: {query}")
        response = agent.invoke({"messages": [("human", query)]})
        # response = agent.invoke({"input": query, "chat_history": []})
        print(f"智能体回答: {response}")
