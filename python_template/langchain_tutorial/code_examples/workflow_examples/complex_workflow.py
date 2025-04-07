#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain复杂工作流示例 - 智能文档分析系统

本模块展示了如何使用LangChain构建一个复杂的工作流，集成多种组件：
1. 文档加载与处理
2. 向量存储与检索
3. 智能体工具集成
4. 链式处理
5. 多模式智能体协作

功能：分析给定文档，回答问题，提取见解，并生成摘要报告
"""

import os
import tempfile
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
load_dotenv()


class DocumentAnalysisSystem:
    """智能文档分析系统
    
    通过集成多种LangChain组件，实现文档的智能分析和处理。
    """
    
    def __init__(self):
        """初始化文档分析系统"""
        # 检查API密钥
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("未设置OPENAI_API_KEY环境变量，系统无法启动")
            
        # 初始化组件
        self._init_llm()
        self._init_tools()
        self._init_agents()
        self._init_chains()
        
        # 文档存储
        self.documents = []
        self.vectorstore = None
    
    def _init_llm(self):
        """初始化语言模型"""
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        # 基础LLM - 用于一般处理
        self.base_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        
        # 创意LLM - 用于生成内容
        self.creative_llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        
        # Embeddings模型
        self.embeddings = OpenAIEmbeddings()
    
    def _init_tools(self):
        """初始化工具集"""
        from langchain.tools import BaseTool, Tool
        from langchain.chains import RetrievalQA
        from langchain_community.tools import DuckDuckGoSearchRun
        from langchain.tools import tool
        
        # 网络搜索工具
        self.search_tool = DuckDuckGoSearchRun()
        
        # 文档检索工具(将在加载文档后设置)
        self.retrieval_tool = None
        
        # 自定义文本分析工具
        @tool
        def analyze_sentiment(text: str) -> str:
            """分析文本情感，输入为待分析的文本内容"""
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            template = """分析以下文本的情感，分类为积极、消极或中性，并给出理由：
            
            文本: {text}
            
            情感分析结果:"""
            
            prompt = PromptTemplate(template=template, input_variables=["text"])
            chain = LLMChain(llm=self.base_llm, prompt=prompt)
            return chain.run(text=text)
        
        self.analyze_sentiment_tool = analyze_sentiment
        
        # 文本摘要工具
        @tool
        def summarize_text(text: str) -> str:
            """生成文本摘要，输入为需要摘要的文本内容"""
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            template = """请对以下文本生成简洁的摘要，捕捉核心观点和重要信息：
            
            文本: {text}
            
            摘要:"""
            
            prompt = PromptTemplate(template=template, input_variables=["text"])
            chain = LLMChain(llm=self.base_llm, prompt=prompt)
            return chain.run(text=text)
            
        self.summarize_tool = summarize_text
    
    def _init_agents(self):
        """初始化智能体"""
        from langchain.agents import AgentType, initialize_agent, Tool
        
        # 工具集合
        base_tools = [self.search_tool, self.analyze_sentiment_tool, self.summarize_tool]
        
        # 研究员智能体 - 负责信息收集和分析
        self.researcher_agent = initialize_agent(
            tools=base_tools,
            llm=self.base_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # 将研究员包装为工具
        def researcher_tool(query: str) -> str:
            """使用研究员智能体获取和分析信息"""
            try:
                return self.researcher_agent.run(query)
            except Exception as e:
                return f"研究过程中遇到错误: {str(e)}. 提供部分可用信息。"
        
        self.researcher_tool = Tool(
            name="Researcher",
            description="当需要深入研究和分析信息时使用，提供详细的研究请求",
            func=researcher_tool
        )
        
        # 作家智能体 - 负责内容创作
        writer_tools = [self.summarize_tool]
        self.writer_agent = initialize_agent(
            tools=writer_tools,
            llm=self.creative_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # 将作家包装为工具
        def writer_tool(content_request: str) -> str:
            """使用作家智能体创建内容"""
            try:
                return self.writer_agent.run(content_request)
            except Exception as e:
                return f"内容创作过程中遇到错误: {str(e)}"
        
        self.writer_tool = Tool(
            name="Writer",
            description="当需要创建高质量内容、摘要或报告时使用，提供详细的内容需求",
            func=writer_tool
        )
        
        # 主智能体工具集(初始化后更新)
        self.main_agent_tools = [self.researcher_tool, self.writer_tool]
    
    def _init_chains(self):
        """初始化处理链"""
        from langchain.chains import LLMChain, SequentialChain
        from langchain.prompts import PromptTemplate
        
        # 文档分析链
        doc_analysis_template = """请分析以下文档内容并提取关键信息点、主要观点和隐含见解：
        
        文档内容: {document}
        
        分析结果:"""
        
        self.document_analysis_chain = LLMChain(
            llm=self.base_llm,
            prompt=PromptTemplate(template=doc_analysis_template, input_variables=["document"]),
            output_key="analysis"
        )
        
        # 见解提取链
        insights_template = """基于以下文档分析，提取3-5个关键见解或行动建议：
        
        文档分析: {analysis}
        
        关键见解:"""
        
        self.insights_chain = LLMChain(
            llm=self.base_llm,
            prompt=PromptTemplate(template=insights_template, input_variables=["analysis"]),
            output_key="insights"
        )
        
        # 组合链
        self.analysis_pipeline = SequentialChain(
            chains=[self.document_analysis_chain, self.insights_chain],
            input_variables=["document"],
            output_variables=["analysis", "insights"],
            verbose=True
        )
    
    def load_documents(self, file_paths: List[str]):
        """加载文档文件
        
        Args:
            file_paths: 文档文件路径列表
        """
        from langchain_community.document_loaders import TextLoader, PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        
        print(f"加载 {len(file_paths)} 个文档...")
        loaded_docs = []
        
        # 加载不同类型的文档
        for path in file_paths:
            try:
                path_lower = path.lower()
                if path_lower.endswith(".txt"):
                    loader = TextLoader(path)
                    loaded_docs.extend(loader.load())
                elif path_lower.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    loaded_docs.extend(loader.load())
                else:
                    print(f"不支持的文件格式: {path}")
            except Exception as e:
                print(f"加载文件 {path} 时出错: {e}")
        
        if not loaded_docs:
            print("未能加载任何文档")
            return
            
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.documents = text_splitter.split_documents(loaded_docs)
        print(f"文档处理完成，共有 {len(self.documents)} 个文本块")
        
        # 创建向量存储
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        
        # 初始化检索器
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # 创建检索QA链
        from langchain.chains import RetrievalQA
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.base_llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        # 设置检索工具
        from langchain.tools import Tool
        self.retrieval_tool = Tool(
            name="DocumentSearch",
            description="当需要在已加载文档中查找信息时使用，提供具体问题",
            func=qa_chain.run
        )
        
        # 更新主智能体工具集
        self.main_agent_tools.append(self.retrieval_tool)
        
        # 初始化主智能体
        self._init_main_agent()
        
        print("文档已加载并索引，系统准备就绪")
    
    def _init_main_agent(self):
        """初始化主智能体"""
        from langchain.agents import AgentType, initialize_agent
        from langchain.memory import ConversationBufferMemory
        
        # 创建带记忆的智能体
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        self.main_agent = initialize_agent(
            tools=self.main_agent_tools,
            llm=self.base_llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )
    
    def analyze_document_content(self, document_text: str) -> Dict[str, str]:
        """分析文档内容
        
        Args:
            document_text: 文档文本内容
            
        Returns:
            包含分析结果和见解的字典
        """
        print("开始分析文档内容...")
        try:
            result = self.analysis_pipeline({"document": document_text})
            return result
        except Exception as e:
            print(f"文档分析过程中出错: {e}")
            return {"analysis": "分析失败", "insights": "无法提取见解"}
    
    def generate_report(self, topic: str) -> str:
        """基于已加载文档生成综合报告
        
        Args:
            topic: 报告主题或指导方针
            
        Returns:
            生成的报告文本
        """
        if not self.documents:
            return "错误：请先加载文档才能生成报告"
            
        # 使用主智能体生成报告
        try:
            prompt = f"""根据已加载的文档内容，生成一份关于"{topic}"的全面报告。
            报告应包含：
            1. 执行摘要
            2. 关键发现
            3. 详细分析
            4. 结论与建议
            
            请确保报告内容准确反映文档中的信息，并提供有价值的见解。"""
            
            report = self.main_agent.run(prompt)
            return report
        except Exception as e:
            return f"报告生成失败: {e}"
    
    def ask_question(self, question: str) -> str:
        """向系统提问
        
        Args:
            question: 用户问题
            
        Returns:
            系统回答
        """
        if not hasattr(self, 'main_agent'):
            return "请先加载文档才能提问"
            
        try:
            return self.main_agent.run(question)
        except Exception as e:
            return f"处理问题时出错: {e}"


# 演示文档文本
SAMPLE_DOCUMENT = """
# 人工智能在医疗领域的应用与挑战

## 引言

人工智能(AI)技术在近年来取得了显著进步，其在医疗健康领域的应用潜力尤为引人注目。AI系统正在改变医疗实践的方方面面，从诊断和治疗到医院管理和患者护理。本文探讨AI在医疗领域的主要应用场景、已取得的成果、面临的挑战以及未来的发展方向。

## 当前应用

### 1. 医学影像分析

AI在医学影像分析中表现出色。深度学习算法能够分析X光片、CT扫描、MRI等影像数据，帮助检测癌症、脑损伤、心脏疾病等。研究表明，在某些特定任务上，AI系统的准确率已经接近或超过了专业放射科医生。

### 2. 疾病诊断与预测

AI系统能够整合并分析患者的各种数据(如病史、实验室检测结果、基因数据等)，辅助医生进行疾病诊断和风险预测。例如，在预测糖尿病、心脏病和某些类型癌症的风险方面，AI模型已显示出较高的准确性。

### 3. 药物研发

AI加速了新药研发过程。机器学习算法能够分析大量化合物数据，预测潜在药物的效果和安全性，从而缩短研发周期、降低成本。例如，2020年，AI系统首次完全独立设计出一种新药物分子并进入临床试验阶段。

### 4. 个性化医疗

AI支持的个性化医疗方案根据患者的独特特征(如基因组、生活方式、环境因素)定制治疗方案。这种方法提高了治疗效果，减少了副作用，特别是在肿瘤学领域取得了显著进展。

### 5. 医疗机器人

从辅助手术的机器人到提供基础护理的自动化系统，AI驱动的机器人正在医院环境中发挥越来越重要的作用。达芬奇手术系统等机器人辅助平台已被广泛应用于微创手术。

## 主要挑战

### 1. 数据隐私与安全

医疗AI系统需要访问大量敏感的患者数据，这引发了严重的隐私和安全担忧。确保数据安全、合规使用和患者同意是一项持续挑战。

### 2. 算法偏见

AI系统可能继承或放大训练数据中的偏见，导致对某些人群的不公平结果。例如，如果训练数据主要来自特定人口群体，AI可能在诊断其他群体时表现不佳。

### 3. 可解释性问题

许多先进的AI算法(如深度神经网络)是"黑盒子"，其决策过程难以解释。在医疗领域，医生和患者需要理解AI为何做出特定建议，这种透明度对建立信任至关重要。

### 4. 监管与标准

AI医疗技术的快速发展超过了现有监管框架的适应能力。建立适当的安全标准、认证流程和责任机制仍是亟待解决的问题。

## 未来展望

随着技术进步和挑战的逐步解决，AI在医疗领域的应用将继续扩大。未来发展趋势包括：

1. **多模态AI系统**：整合多种数据源(如影像、基因组学、电子健康记录)的综合AI解决方案
2. **边缘计算应用**：实现医疗设备上的本地AI处理，减少延迟，增强隐私保护
3. **AI与人类协作模式**：开发更好的人机协作界面，使AI成为医疗专业人员的有效助手而非替代者
4. **可解释AI**：推进可解释AI技术，使AI决策过程更透明
5. **全球标准化**：建立国际认可的AI医疗技术评估和监管标准

## 结论

人工智能正在深刻变革医疗行业，提供了提高诊断准确性、个性化治疗和改善患者护理的强大工具。尽管面临数据隐私、算法偏见和监管等挑战，但随着技术的成熟和适当框架的建立，AI有望显著改善全球医疗健康水平。关键是要确保这些技术的发展以患者福祉为中心，同时保持对伦理和社会影响的敏感性。
"""


# ===============================
# 主函数：演示文档分析系统
# ===============================
def main():
    """主函数，演示文档分析系统的使用"""
    print("\n=== LangChain复杂工作流示例：智能文档分析系统 ===\n")
    print("注意：本示例需要设置OPENAI_API_KEY环境变量\n")
    
    try:
        # 初始化系统
        print("初始化智能文档分析系统...")
        system = DocumentAnalysisSystem()
        print("系统初始化完成\n")
        
        # 创建示例文档文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp:
            temp.write(SAMPLE_DOCUMENT)
            temp_file_path = temp.name
        
        print(f"已创建示例文档: {temp_file_path}")
        
        # 加载文档
        system.load_documents([temp_file_path])
        
        # 文档内容分析
        print("\n=== 文档内容分析示例 ===\n")
        analysis_result = system.analyze_document_content(SAMPLE_DOCUMENT)
        print("\n分析结果:")
        print(analysis_result["analysis"])
        print("\n关键见解:")
        print(analysis_result["insights"])
        
        # 问答示例
        print("\n=== 文档问答示例 ===\n")
        questions = [
            "AI在医学影像分析方面有哪些应用？",
            "使用AI技术在医疗领域面临的主要挑战是什么？",
            "文档中提到了哪些未来AI在医疗领域的发展趋势？"
        ]
        
        for q in questions:
            print(f"问题: {q}")
            answer = system.ask_question(q)
            print(f"回答: {answer}\n")
        
        # 生成报告
        print("\n=== 报告生成示例 ===\n")
        report = system.generate_report("人工智能如何改善患者护理体验")
        print("生成报告:\n")
        print(report)
        
        # 清理临时文件
        os.unlink(temp_file_path)
        print("\n示例完成，临时文件已清理")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")


# 程序入口
if __name__ == "__main__":
    main()