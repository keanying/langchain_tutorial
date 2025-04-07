# LangChain 高级搜索智能体示例

import os
from typing import List, Dict, Any, Optional
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import AgentType, initialize_agent, Tool, create_react_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import SystemMessage
import tempfile


# 确保设置环境变量
# os.environ["OPENAI_API_KEY"] = "your-api-key"

class AdvancedSearchAgent:
    """高级搜索智能体 - 结合多种搜索增强技术的智能体"""

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.vectorstore = None
        self.agent = None

    def setup_knowledge_base(self):
        """设置示例知识库"""
        print("设置知识库...")

        # 创建临时目录存储示例文档
        temp_dir = tempfile.mkdtemp()

        # 示例文档内容
        documents = [
            {
                "title": "搜索集成架构",
                "content": """
                搜索与智能体集成的三种主要架构:
                
                1. 工具型集成: 将搜索/检索系统作为工具提供给智能体使用
                   - 智能体可以决定何时使用搜索
                   - 更灵活但可能需要更多提示工程
                   - 适合复杂任务和开放域问答
                
                2. 预检索增强: 在智能体处理前先进行检索并作为上下文提供
                   - 智能体始终能看到相关信息
                   - 减少不必要的API调用
                   - 适合知识密集型任务
                
                3. 混合架构: 结合以上两种方法
                   - 提供基础上下文，同时允许主动搜索
                   - 更全面但复杂度更高
                   - 适合多轮对话和复杂推理任务
                """
            },
            {
                "title": "高级检索技术",
                "content": """
                提高检索质量的高级技术:
                
                1. 查询重构与扩展
                   - 使用LLM重写用户查询
                   - 添加相关关键词扩展查询
                   - 分解复杂查询为多个简单查询
                
                2. 上下文压缩
                   - 提取检索文档中最相关的部分
                   - 减少上下文长度，集中于关键信息
                   - 使用LLM或启发式方法进行筛选
                
                3. 混合检索
                   - 结合多种检索策略(语义、关键词等)
                   - 整合多个数据源的结果
                   - 使用加权方法排序最终结果
                """
            },
            {
                "title": "行业应用场景",
                "content": """
                搜索增强智能体的行业应用:
                
                1. 客户支持
                   - 检索产品文档回答用户问题
                   - 访问内部知识库解决技术问题
                   - 提供个性化的故障排除建议
                
                2. 法律研究
                   - 搜索相关法规和判例
                   - 分析法律文件并提供见解
                   - 辅助起草法律文档
                
                3. 医疗咨询
                   - 检索医学文献和临床指南
                   - 提供基于证据的治疗建议
                   - 支持医生的诊断决策
                
                4. 教育辅助
                   - 个性化学习材料推荐
                   - 回答学生的专业领域问题
                   - 生成定制的练习和测试
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
        self.vectorstore = FAISS.from_documents(splits, embeddings)

        print(f"知识库设置完成，包含 {len(splits)} 个文本块")

    def setup_advanced_search_tools(self):
        """设置高级搜索工具"""
        if not self.vectorstore:
            raise ValueError("请先设置知识库")

        # 基本检索器
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        # 1. 创建上下文压缩检索器
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        # 2. 创建查询转换器
        def transform_query(query: str) -> str:
            """使用LLM改写查询以提高检索效果"""
            transform_prompt = PromptTemplate.from_template(
                "请将以下用户查询改写成更适合向量数据库检索的形式，添加可能相关的关键词，但保持简洁。\n\n用户查询: {query}\n\n改写后的查询:"
            )
            chain = transform_prompt | self.llm
            result = chain.invoke({"query": query})
            return result.content if hasattr(result, "content") else str(result)

        # 3. 定义搜索函数
        def enhanced_search(query: str) -> str:
            """增强搜索函数，包括查询转换和上下文压缩"""
            # 转换查询
            transformed_query = transform_query(query)
            print(f"原始查询: {query}")
            print(f"改写查询: {transformed_query}")

            # 使用压缩检索器获取结果
            docs = compression_retriever.get_relevant_documents(transformed_query)
            return "\n\n".join(doc.page_content for doc in docs)

        # 创建工具集
        tools = [
            Tool(
                name="增强知识搜索",
                func=enhanced_search,
                description="当需要查询关于搜索技术、智能体集成方法或应用场景的信息时使用"
            ),
            PythonREPLTool()
        ]

        return tools

    def setup_agent(self):
        """设置高级搜索智能体"""
        if not self.vectorstore:
            self.setup_knowledge_base()

        # 获取高级搜索工具
        tools = self.setup_advanced_search_tools()

        # 创建系统提示
        system_message = SystemMessage(
            content="""你是一个先进的搜索增强智能体，专门回答关于搜索技术与智能体集成的问题。

遵循以下步骤:
1. 理解用户的查询意图
2. 使用增强知识搜索工具获取相关信息
3. 基于检索到的信息提供全面的回答
4. 如果需要进行计算或示例，使用Python工具
5. 以清晰、结构化的方式组织你的回答

如果检索的信息不完整，可以基于你的知识进行补充，但要明确区分来源。
            """
        )

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # 创建智能体
        agent = create_react_agent(self.llm, tools, prompt)

        # 创建执行器
        self.agent = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

        return self.agent

    def run_query(self, query: str) -> str:
        """执行查询"""
        if not self.agent:
            self.setup_agent()

        print(f"\n执行查询: {query}")
        result = self.agent.invoke(query)
        return result


def main():
    print("=== LangChain 高级搜索智能体示例 ===")

    # 创建高级搜索智能体
    agent = AdvancedSearchAgent()
    agent.setup_knowledge_base()
    agent.setup_agent()

    # 执行多个查询示例
    queries = [
        "比较不同搜索与智能体集成架构的优缺点",
        "有哪些高级检索技术可以提高搜索质量?",
        "搜索增强智能体在哪些行业有应用，并计算行业数量"
    ]

    for i, query in enumerate(queries):
        print(f"\n--- 查询 {i + 1} ---")
        result = agent.run_query(query)
        print(f"\n最终回答: {result}")

    print("\n=== 示例执行完成 ===")


if __name__ == "__main__":
    main()
