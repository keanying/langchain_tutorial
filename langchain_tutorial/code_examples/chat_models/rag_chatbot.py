#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于RAG的聊天机器人示例：展示如何基于文档检索增强生成构建智能问答系统
集成了文本分割、向量数据库、Embedding模型和聊天模型
"""

import os
import time
import asyncio
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# 导入LangChain核心组件
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

# 导入LangChain文档处理组件
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PDFMinerLoader, CSVLoader
)

# 导入向量存储组件
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.base import VectorStore

# 导入Embedding模型
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# 导入聊天模型
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_qianfan import ChatQianfan
# from langchain_dashscope import ChatDashscope

# 创建Rich控制台用于美化输出
console = Console()

class DocumentProcessor:
    """文档处理类，负责加载、分割和索引文档"""
    
    def __init__(self, embedding_model: str = "openai"):
        """初始化文档处理器
        
        Args:
            embedding_model: 使用的embedding模型，可选值为"openai"或"huggingface"
        """
        self.documents = []
        self.setup_embedding_model(embedding_model)
        
    def setup_embedding_model(self, model_name: str):
        """设置Embedding模型
        
        Args:
            model_name: 模型名称，支持"openai"和"huggingface"
        """
        if model_name == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("未设置OPENAI_API_KEY环境变量")
            self.embeddings = OpenAIEmbeddings()
            console.print("[green]已加载OpenAI Embeddings模型[/green]")
        elif model_name == "huggingface":
            # 使用开源Embedding模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            console.print("[green]已加载HuggingFace Embeddings模型[/green]")
        else:
            raise ValueError(f"不支持的Embedding模型: {model_name}")
        
    def load_documents(self, file_paths: List[str]) -> int:
        """加载多个文档文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            加载的文档数量
        """
        loaded_docs = []
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if not path.exists():
                    console.print(f"[red]文件不存在: {file_path}[/red]")
                    continue
                    
                # 根据文件类型选择合适的加载器
                if path.suffix.lower() == '.txt':
                    loader = TextLoader(file_path)
                    docs = loader.load()
                elif path.suffix.lower() == '.pdf':
                    loader = PDFMinerLoader(file_path)
                    docs = loader.load()
                elif path.suffix.lower() == '.csv':
                    loader = CSVLoader(file_path)
                    docs = loader.load()
                else:
                    console.print(f"[yellow]不支持的文件类型: {path.suffix}[/yellow]")
                    continue
                
                console.print(f"[green]成功加载文档 {path.name}, 包含 {len(docs)} 页/部分[/green]")
                loaded_docs.extend(docs)
                
            except Exception as e:
                console.print(f"[red]加载文档出错 {file_path}: {str(e)}[/red]")
        
        # 更新文档列表
        self.documents.extend(loaded_docs)
        return len(loaded_docs)
    
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """直接添加文本内容
        
        Args:
            text: 文本内容
            metadata: 元数据
        """
        doc = Document(page_content=text, metadata=metadata or {})
        self.documents.append(doc)
        console.print(f"[green]成功添加文本文档，长度: {len(text)} 字符[/green]")
    
    def process_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """处理并分割文档
        
        Args:
            chunk_size: 文档块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            处理后的文档块列表
        """
        if not self.documents:
            console.print("[yellow]警告：没有文档可供处理[/yellow]")
            return []
        
        # 分割文档为较小的块
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        splits = text_splitter.split_documents(self.documents)
        console.print(f"[green]文档已被分割为 {len(splits)} 个块[/green]")
        return splits
    
    def create_vector_db(self, 
                         document_chunks: List[Document], 
                         db_type: str = "faiss",
                         persist_directory: Optional[str] = None) -> VectorStore:
        """创建向量数据库
        
        Args:
            document_chunks: 文档块列表
            db_type: 数据库类型，支持"faiss"和"chroma"
            persist_directory: 持久化目录路径（仅适用于需要持久化的数据库）
            
        Returns:
            创建的向量数据库
        """
        start_time = time.time()
        
        if db_type == "faiss":
            vector_db = FAISS.from_documents(
                documents=document_chunks,
                embedding=self.embeddings
            )
            console.print("[green]已创建FAISS向量数据库[/green]")
        elif db_type == "chroma":
            if persist_directory:
                vector_db = Chroma.from_documents(
                    documents=document_chunks,
                    embedding=self.embeddings,
                    persist_directory=persist_directory
                )
                vector_db.persist()
                console.print(f"[green]已创建Chroma向量数据库并持久化到 {persist_directory}[/green]")
            else:
                vector_db = Chroma.from_documents(
                    documents=document_chunks,
                    embedding=self.embeddings
                )
                console.print("[green]已创建Chroma向量数据库（内存模式）[/green]")
        else:
            raise ValueError(f"不支持的向量数据库类型: {db_type}")
        
        end_time = time.time()
        console.print(f"[green]向量化耗时: {end_time - start_time:.2f} 秒[/green]")
        
        return vector_db

class RAGChatbot:
    """基于检索增强生成（RAG）的聊天机器人"""
    
    def __init__(self, vector_store: VectorStore, model_name: str = "openai", temperature: float = 0.7):
        """初始化RAG聊天机器人
        
        Args:
            vector_store: 向量存储
            model_name: 使用的聊天模型
            temperature: 温度参数
        """
        self.vector_store = vector_store
        self.chat_history = []
        self.temperature = temperature
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}  # 每次检索5个最相关的文档块
        )
        
        # 设置聊天模型
        self.setup_chat_model(model_name)
        
        # 创建RAG链
        self.setup_rag_chain()
    
    def setup_chat_model(self, model_name: str):
        """设置聊天模型
        
        Args:
            model_name: 模型名称
        """
        if model_name == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("未设置OPENAI_API_KEY环境变量")
            self.chat_model = ChatOpenAI(
                temperature=self.temperature,
                model="gpt-3.5-turbo"  # 可以替换为其他OpenAI模型
            )
        elif model_name == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError("未设置ANTHROPIC_API_KEY环境变量")
            from langchain_anthropic import ChatAnthropic
            self.chat_model = ChatAnthropic(
                temperature=self.temperature,
                model="claude-3-sonnet-20240229"  # 可以替换为其他Claude模型
            )
        elif model_name == "baidu":
            if not (os.environ.get("QIANFAN_AK") and os.environ.get("QIANFAN_SK")):
                raise ValueError("未设置QIANFAN_AK和QIANFAN_SK环境变量")
            from langchain_qianfan import ChatQianfan
            self.chat_model = ChatQianfan(temperature=self.temperature)
        elif model_name == "alibaba":
            if not os.environ.get("DASHSCOPE_API_KEY"):
                raise ValueError("未设置DASHSCOPE_API_KEY环境变量")
            from langchain_dashscope import ChatDashscope
            self.chat_model = ChatDashscope(temperature=self.temperature)
        else:
            raise ValueError(f"不支持的聊天模型: {model_name}")
        
        console.print(f"[green]已加载{model_name}聊天模型[/green]")
    
    def setup_rag_chain(self):
        """设置RAG检索增强生成链"""
        # 创建带上下文的提示模板
        template = """你是一个基于文档知识的智能助手。使用以下检索到的上下文信息回答用户的问题。
如果你不知道答案，就说你不知道，不要试图编造信息。
尽可能使用中文回答。

上下文信息:
{context}

聊天历史:
{chat_history}

用户问题: {question}
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 定义格式化上下文的函数
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 构建RAG链
        self.rag_chain = (
            {"context": self.retriever | format_docs, 
             "chat_history": lambda x: "\n".join([f"{m.type}: {m.content}" for m in self.chat_history[-6:]]) if self.chat_history else "",
             "question": RunnablePassthrough()}
            | prompt
            | self.chat_model
            | StrOutputParser()
        )
    
    def chat(self, user_message: str) -> str:
        """处理用户消息并返回基于检索的回复
        
        Args:
            user_message: 用户消息
            
        Returns:
            助手回复
        """
        try:
            # 添加用户消息到历史
            self.chat_history.append(HumanMessage(content=user_message))
            
            # 调用RAG链获取回答
            start_time = time.time()
            response = self.rag_chain.invoke(user_message)
            end_time = time.time()
            
            # 添加助手回复到历史
            self.chat_history.append(AIMessage(content=response))
            
            # 如果历史太长，只保留最近的轮次
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
                
            console.print(f"[green]回答生成耗时: {end_time - start_time:.2f} 秒[/green]")
            return response
            
        except Exception as e:
            error_msg = f"处理消息时出错: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg
    
    def search_docs(self, query: str, top_k: int = 3) -> List[Document]:
        """直接搜索相关文档
        
        Args:
            query: 搜索查询
            top_k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        return self.retriever.get_relevant_documents(query)
    
    def clear_history(self) -> None:
        """清除聊天历史"""
        self.chat_history = []
        console.print("[yellow]已清除聊天历史[/yellow]")

def create_sample_docs():
    """创建示例文档用于演示"""
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建示例文件
        files = []
        
        # 样例1：人工智能介绍
        ai_intro_path = os.path.join(temp_dir, "ai_introduction.txt")
        with open(ai_intro_path, "w", encoding="utf-8") as f:
            f.write("""
人工智能概述

人工智能（AI）是计算机科学的一个分支，致力于创造能够模拟人类智能的机器。这些机器可以学习、推理、感知和解决问题。

主要AI技术:
1. 机器学习：通过数据学习并改进，无需显式编程。包括监督学习、非监督学习和强化学习。
2. 深度学习：使用神经网络处理大量数据，识别模式和特征。
3. 自然语言处理：使计算机能理解和生成人类语言。
4. 计算机视觉：使计算机能解释和处理视觉信息。

应用领域:
- 医疗保健：疾病诊断、药物发现、个性化治疗
- 金融：欺诈检测、算法交易、风险评估
- 交通：自动驾驶汽车、交通流量预测、路线优化
- 零售：个性化推荐、库存管理、客户服务
- 教育：自适应学习、自动评分、教育内容生成

AI发展史:
1950年代：AI研究开始，图灵测试提出
1980年代：专家系统流行
1990年代：机器学习开始发展
2010年代：深度学习突破，大型语言模型出现

未来挑战:
- 伦理问题：隐私、偏见、透明度
- 算法解释性
- 通用人工智能的发展
- 与人类协作方式
            """)
        files.append(ai_intro_path)
        
        # 样例2：机器学习详解
        ml_path = os.path.join(temp_dir, "machine_learning.txt")
        with open(ml_path, "w", encoding="utf-8") as f:
            f.write("""
机器学习详解

机器学习是人工智能的一个子领域，专注于开发可以从数据中学习并做出预测的算法。

机器学习算法类型：

1. 监督学习
   - 定义：算法从标记好的训练数据中学习
   - 常见算法：线性回归、逻辑回归、决策树、随机森林、支持向量机、神经网络
   - 应用：图像分类、垃圾邮件检测、疾病诊断

2. 无监督学习
   - 定义：算法从未标记的数据中发现模式
   - 常见算法：K-means聚类、层次聚类、主成分分析、异常检测
   - 应用：市场分割、特征提取、模式识别

3. 强化学习
   - 定义：算法通过与环境互动学习最佳行为
   - 常见算法：Q-learning、深度Q网络、策略梯度
   - 应用：游戏AI、机器人控制、自动驾驶

机器学习工作流程：
1. 数据收集和准备
2. 特征工程和选择
3. 模型选择和训练
4. 评估和调优
5. 部署和监控

常见挑战：
- 过拟合：模型在训练数据上表现良好，但无法泛化
- 欠拟合：模型过于简单，无法捕获数据中的模式
- 数据质量：不完整、不准确或有偏见的数据
- 计算资源：训练复杂模型需要大量计算能力

最佳实践：
- 保持数据分割（训练集、验证集、测试集）
- 使用交叉验证避免过拟合
- 定期重新训练模型以适应数据分布变化
- 考虑模型的可解释性和透明度
            """)
        files.append(ml_path)
        
        # 样例3：语言模型概述
        llm_path = os.path.join(temp_dir, "language_models.txt")
        with open(llm_path, "w", encoding="utf-8") as f:
            f.write("""
大型语言模型概述

大型语言模型（LLM）是一种基于深度学习的AI系统，经过训练可以理解和生成人类语言。近年来，它们在自然语言处理领域取得了突破性进展。

主要特点：

1. 架构
   - 基于Transformer结构，使用自注意力机制
   - 通常包含数十亿到数千亿个参数
   - 通过大规模预训练和微调实现

2. 能力
   - 文本生成：创作文章、故事、代码等
   - 问答：回答各种领域的问题
   - 摘要：压缩长文本为简洁摘要
   - 翻译：在不同语言之间转换
   - 推理：进行逻辑推理和分析

3. 训练方法
   - 预训练：在大量文本数据上进行自监督学习
   - 监督微调：使用人类反馈改进特定任务表现
   - 强化学习人类反馈(RLHF)：通过人类偏好优化模型

主要模型示例：
- GPT系列（OpenAI）：包括GPT-3、GPT-4等
- Claude系列（Anthropic）：专注于有帮助、无害的AI
- LLaMA（Meta）：开源大型语言模型
- 文心一言（百度）：中英双语大模型
- 通义千问（阿里）：多模态大型语言模型

应用场景：
- 客户服务：自动回答常见问题
- 内容创作：协助撰写文章、报告
- 代码开发：生成和解释代码
- 教育：个性化学习助手
- 研究：辅助文献综述和假设生成

局限性和挑战：
- 幻觉：生成虚假或不准确信息
- 偏见：可能反映训练数据中的社会偏见
- 上下文窗口限制：处理信息的长度有限
- 缺乏可解释性：难以理解黑盒决策过程
- 计算资源需求：训练和运行需要大量计算资源
            """)
        files.append(llm_path)
        
        return files

async def main():
    """主函数"""
    console.print("[bold green]LangChain RAG聊天机器人演示[/bold green]")
    console.print("本示例展示如何构建基于文档检索的智能问答系统\\n")
    
    try:
        # 确保有API密钥
        if not os.environ.get("OPENAI_API_KEY"):
            console.print("[yellow]警告: 未设置OPENAI_API_KEY环境变量，请先设置再运行[/yellow]")
            console.print("可以通过设置环境变量来添加API密钥: export OPENAI_API_KEY=your_key_here")
            return
        
        # 创建示例文档
        console.print("[cyan]创建示例文档...[/cyan]")
        file_paths = create_sample_docs()
        
        # 初始化文档处理器
        console.print("[cyan]初始化文档处理器...[/cyan]")
        processor = DocumentProcessor(embedding_model="openai")
        
        # 加载文档
        console.print("[cyan]加载文档...[/cyan]")
        processor.load_documents(file_paths)
        
        # 处理文档
        console.print("[cyan]处理和分割文档...[/cyan]")
        chunks = processor.process_documents(chunk_size=500, chunk_overlap=50)
        
        # 创建向量数据库
        console.print("[cyan]创建向量数据库...[/cyan]")
        vector_db = processor.create_vector_db(chunks, db_type="faiss")
        
        # 创建RAG聊天机器人
        console.print("[cyan]初始化RAG聊天机器人...[/cyan]")
        chatbot = RAGChatbot(vector_db, model_name="openai", temperature=0.7)
        
        # 开始交互式聊天
        console.print("[bold green]聊天机器人已准备就绪! 输入'quit'或'exit'退出对话。输入'clear'可清除聊天历史。[/bold green]")
        
        # 聊天循环
        while True:
            # 获取用户输入
            user_input = input("\n[你] ")
            
            # 检查是否退出
            if user_input.lower() in ["quit", "exit"]:
                console.print("[yellow]结束对话[/yellow]")
                break
            
            # 检查是否清除历史
            if user_input.lower() == "clear":
                chatbot.clear_history()
                continue
            
            # 检查是否为文档搜索
            if user_input.lower().startswith("search:"):
                query = user_input[7:].strip()
                docs = chatbot.search_docs(query)
                
                console.print("\n[bold cyan]找到相关文档片段:[/bold cyan]")
                for i, doc in enumerate(docs, 1):
                    console.print(Panel(
                        doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        title=f"文档片段 {i}",
                        border_style="green"
                    ))
                continue
            
            # 处理常规问题
            console.print("[cyan]思考中...[/cyan]")
            response = chatbot.chat(user_input)
            
            # 显示回答
            console.print("\n[bold blue][AI][/bold blue]")
            console.print(Markdown(response))
            
    except KeyboardInterrupt:
        console.print("\n[yellow]程序被用户中断[/yellow]")
    except Exception as e:
        console.print(f"\n[red]发生错误: {str(e)}[/red]")

if __name__ == "__main__":
    asyncio.run(main())