# LangChain 多智能体工作流示例代码
import os
import sys
from typing import List, Dict, Any
from pathlib import Path

# 请先设置环境变量
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# 1. 基础组件演示
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage, AIMessage

def demonstrate_basic_components():
    print("\n1. 演示 LangChain 基础组件")
    
    # 创建一个提示模板
    template = "你是一个{role}。用户问题: {question}"
    prompt = PromptTemplate(
        input_variables=["role", "question"],
        template=template,
    )
    
    # 初始化LLM
    llm = ChatOpenAI(temperature=0.7)
    
    # 创建一个带有记忆的链
    memory = ConversationBufferMemory()
    chain = LLMChain(
        llm=llm, 
        prompt=prompt, 
        memory=memory,
        verbose=True
    )
    
    # 执行链
    response = chain.run(role="友好的AI助手", question="什么是LangChain框架?")
    print(f"\n回答: {response}\n")
    
    # 展示记忆功能
    response = chain.run(role="友好的AI助手", question="它的主要组件有哪些?")
    print(f"\n回答: {response}\n")
    
    return chain

# 2. 检索增强生成 (RAG) 示例
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

def create_rag_system():
    print("\n2. 创建检索增强生成 (RAG) 系统")
    
    # 创建示例文档
    Path("data").mkdir(exist_ok=True)
    with open("data/langchain_info.txt", "w", encoding="utf-8") as f:
        f.write('''
LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它有以下核心组件：

1. 模型 I/O: 处理与语言模型的交互，包括提示模板和输出解析。
2. 数据连接: 连接语言模型与各种数据源。
3. 链: 将多个组件组合成一个应用程序。
4. 记忆: 在链的运行之间持久化状态。
5. 智能体: 允许语言模型根据目标选择要使用的工具。
6. 回调: 在链执行期间钩入中间步骤。

LangChain 表达式语言 (LCEL) 是一种声明式语言，用于组合这些组件。它支持同步和异步操作，可以实现复杂的工作流程，如检索增强生成 (RAG)。

LangChain 还支持多种类型的智能体，如 ReAct、工具使用、规划与执行等。多智能体系统可以通过智能体监督功能进行协作。
        ''')
    
    # 加载并处理文档
    loader = TextLoader("data/langchain_info.txt")
    documents = loader.load()
    
    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    
    # 创建向量存储
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    
    # 创建一个检索链
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectordb.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(),
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    
    # 执行检索问答
    result = qa_chain({"question": "LangChain有哪些核心组件?"})
    print(f"\nRAG回答: {result['answer']}\n")
    
    return qa_chain

# 3. 单智能体工作流
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools.python.tool import PythonREPLTool

def create_single_agent():
    print("\n3. 创建带工具的单智能体系统")
    
    # 定义工具
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    python_repl = PythonREPLTool()
    
    tools = [
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="查询维基百科以获取信息"
        ),
        Tool(
            name="Python REPL",
            func=python_repl.run,
            description="执行Python代码"
        )
    ]
    
    # 初始化智能体
    llm = ChatOpenAI(temperature=0)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # 运行智能体
    response = agent.run("LangChain框架是什么时候发布的? 然后计算从发布到现在经过了多少天。")
    print(f"\n智能体回答: {response}\n")
    
    return agent

# 4. 多智能体系统
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re

# 定义自定义智能体
class MultiAgentSystem:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.research_agent = self._create_research_agent()
        self.planning_agent = self._create_planning_agent()
        self.execution_agent = self._create_execution_agent()
    
    def _create_research_agent(self):
        # 创建研究智能体
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        
        tools = [
            Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="查询维基百科以获取信息"
            )
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        return agent
    
    def _create_planning_agent(self):
        # 创建规划智能体
        prompt = PromptTemplate(
            template="你是一个规划专家。根据下面的任务描述创建一个分步计划:\n{task}",
            input_variables=["task"]
        )
        
        planning_chain = LLMChain(llm=self.llm, prompt=prompt)
        return planning_chain
    
    def _create_execution_agent(self):
        # 创建执行智能体
        python_repl = PythonREPLTool()
        
        tools = [
            Tool(
                name="Python REPL",
                func=python_repl.run,
                description="执行Python代码"
            )
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        return agent
    
    def run(self, task):
        print(f"\n4. 执行多智能体工作流 - 任务: {task}")
        
        # 1. 研究阶段
        print("\n步骤1: 研究阶段")
        research_result = self.research_agent.run(f"研究关于: {task}")
        print(f"研究结果: {research_result}")
        
        # 2. 规划阶段
        print("\n步骤2: 规划阶段")
        plan = self.planning_chain.run(task=f"{task}\n背景信息: {research_result}")
        print(f"执行计划: {plan}")
        
        # 3. 执行阶段
        print("\n步骤3: 执行阶段")
        final_result = self.execution_agent.run(
            f"根据以下计划执行任务: {plan}\n任务: {task}"
        )
        print(f"最终结果: {final_result}")
        
        return {
            "research": research_result,
            "plan": plan,
            "result": final_result
        }

def main():
    print("LangChain 多智能体工作流示例")
    
    # 运行基础组件演示
    basic_chain = demonstrate_basic_components()
    
    # 创建RAG系统
    rag_chain = create_rag_system()
    
    # 创建单智能体系统
    agent = create_single_agent()
    
    # 创建多智能体系统
    multi_agent_system = MultiAgentSystem()
    result = multi_agent_system.run("创建一个简单的网站访问量分析程序")
    
    print("\n演示完成!")

if __name__ == "__main__":
    main()
