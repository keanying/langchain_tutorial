# LangChain 搜索与智能体集成基础示例

import os
from typing import List, Dict, Any
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import tempfile


# 确保设置环境变量
# os.environ["OPENAI_API_KEY"] = "your-api-key"

def create_demo_knowledge_base():
    """创建示例知识库"""
    print("创建示例知识库...")

    # 创建临时目录存储示例文档
    temp_dir = tempfile.mkdtemp()

    # 示例文档内容
    documents = [
        {
            "title": "搜索与智能体集成方法",
            "content": """
            LangChain提供多种将搜索功能集成到智能体的方法:
            
            1. 检索工具（Retrieval Tools）: 将检索器包装为工具供智能体使用
            2. 上下文增强（Contextual Augmentation）: 在智能体处理前添加相关上下文
            3. 检索增强生成（RAG）: 结合检索和生成能力来提供更准确的回答
            4. 自主检索（Self-querying）: 让智能体自己决定如何查询知识库
            
            这些方法可以根据具体需求组合使用。
            """
        },
        {
            "title": "向量检索优化",
            "content": """
            向量检索可以通过以下技术优化:
            
            1. 查询修改: 重写或扩展用户查询以提高检索质量
            2. 过滤: 使用元数据过滤相关文档
            3. 重排序: 使用额外标准对检索结果重新排序
            4. 压缩: 提取最相关的内容片段
            5. 混合检索: 结合语义和关键词搜索
            
            优化后的检索系统能够显著提高智能体的响应质量。
            """
        },
        {
            "title": "智能体工具选择",
            "content": """
            智能体需要智能地选择合适的工具:
            
            1. 基于查询分析选择搜索或计算工具
            2. 考虑任务的信息需求和复杂度
            3. 在多个信息源之间平衡权衡
            4. 评估工具执行结果并决定后续行动
            
            有效的工具选择使智能体能够更高效地完成任务。
            """
        }
    ]

    # 写入文档到临时文件并加载
    all_docs = []
    for i, doc in enumerate(documents):
        file_path = os.path.join(temp_dir, f"{i}_{doc['title'].replace(' ', '_')}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {doc['title']}\n\n{doc['content']}")

        # 加载文档
        loader = TextLoader(file_path)
        all_docs.extend(loader.load())

    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_docs)

    # 创建向量存储
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)

    print(f"知识库创建完成，包含 {len(splits)} 个文本块")
    return vectorstore


def method1_retrieval_qa_direct():
    """方法1: 直接检索 - 使用检索QA链直接回答问题"""
    print("\n=== 方法1: 直接检索 ===")

    # 创建知识库
    vectorstore = create_demo_knowledge_base()
    retriever = vectorstore.as_retriever()

    # 创建检索QA链
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )

    # 执行查询
    query = "LangChain中的搜索集成方法有哪些?"
    result = qa_chain.invoke({"query": query})
    print(f"\n问题: {query}")
    print(f"回答: {result['result']}")

    return vectorstore, qa_chain


def method2_agent_with_retrieval_tool(vectorstore):
    """方法2: 检索工具 - 将检索器封装为智能体工具"""
    print("\n=== 方法2: 检索工具 ===")

    # 创建检索工具
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    def search_knowledge_base(query: str) -> str:
        """搜索知识库工具"""
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)

    # 创建工具列表
    tools = [
        Tool(
            name="知识库搜索",
            func=search_knowledge_base,
            description="当需要查询关于LangChain搜索方法或向量检索技术的信息时使用"
        ),
        PythonREPLTool()
    ]

    # 创建智能体
    agent = initialize_agent(
        tools,
        ChatOpenAI(temperature=0),
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # 执行查询
    query = "向量检索有哪些优化技术? 计算它们的数量。"
    result = agent.invoke(query)
    print(f"\n问题: {query}")
    print(f"回答: {result}")

    return agent


def method3_context_enhanced_agent(vectorstore):
    """方法3: 上下文增强 - 在处理前检索相关信息并增强上下文"""
    print("\n=== 方法3: 上下文增强 ===")

    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 创建智能体工具
    tools = [PythonREPLTool()]

    # 创建智能体
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools,
        ChatOpenAI(temperature=0),
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    # 先检索相关文档
    query = "智能体如何选择合适的工具?"
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 增强查询
    enhanced_query = f"基于以下信息回答问题，如果需要可以使用Python代码演示：\n\n信息：{context}\n\n问题：{query}"

    # 执行查询
    result = agent.invoke(enhanced_query)
    print(f"\n问题: {query}")
    print(f"回答: {result}")

    return agent


def main():
    print("=== LangChain 搜索与智能体集成示例 ===")

    # 方法1: 直接检索QA
    vectorstore, qa_chain = method1_retrieval_qa_direct()

    # 方法2: 检索作为智能体工具
    agent_with_tool = method2_agent_with_retrieval_tool(vectorstore)

    # 方法3: 上下文增强智能体
    contextual_agent = method3_context_enhanced_agent(vectorstore)

    print("\n=== 所有示例执行完成 ===")


if __name__ == "__main__":
    main()
