#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain向量存储(Vector Stores)组件使用示例

本模块展示了如何使用LangChain中的向量存储组件创建和查询知识库，包括：
1. 文档加载与分割
2. 创建不同类型的向量存储
3. 执行相似性搜索
4. 与检索器集成
5. 构建检索增强生成(RAG)系统

注意：运行这些示例前，请确保已设置相应的API密钥环境变量
"""

import os
import tempfile
from typing import List, Dict, Any, Union, Tuple
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===============================
# 第1部分：文档处理与嵌入
# ===============================
def document_processing_examples():
    """文档处理与嵌入示例"""
    print("=== 文档处理与嵌入示例 ===")
    
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # --- 创建示例文档 ---
    print("\n--- 创建示例文档 ---")
    
    # 创建临时文档用于示例
    sample_text = """
    # 人工智能简介
    
    人工智能(AI)是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。
    
    ## 主要领域
    
    1. **机器学习**：使计算机能够从数据中学习，而不需要显式编程。
       - 监督学习
       - 无监督学习
       - 强化学习
    
    2. **自然语言处理**：使计算机能够理解和生成人类语言。
       - 机器翻译
       - 情感分析
       - 问答系统
    
    3. **计算机视觉**：使计算机能够解释和理解视觉信息。
       - 图像识别
       - 物体检测
       - 图像生成
    
    ## 应用案例
    
    人工智能已经在许多领域找到了应用，包括：
    
    - 医疗保健：疾病诊断、药物发现
    - 金融：欺诈检测、算法交易
    - 交通：自动驾驶汽车、交通流量优化
    - 教育：个性化学习、自动评分
    - 客户服务：聊天机器人、推荐系统
    
    ## 挑战与伦理考虑
    
    尽管AI技术发展迅速，但仍面临一些重要挑战：
    
    - 数据隐私与安全
    - 算法偏见与歧视
    - 自动化导致的就业变化
    - 决策过程的透明度和可解释性
    - 自主系统的安全与控制
    
    负责任的AI开发需要考虑这些伦理问题，确保技术发展符合人类共同利益。
    """
    
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "ai_intro.txt")
    
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    print(f"创建了临时文档: {temp_file_path}")
    
    # --- 加载文档 ---
    print("\n--- 加载文档 ---")
    
    try:
        # 使用TextLoader加载文本文档
        loader = TextLoader(temp_file_path, encoding="utf-8")
        documents = loader.load()
        print(f"加载了 {len(documents)} 个文档")
        print(f"文档示例（前200个字符）: {documents[0].page_content[:200]}...")
        
        # 显示文档元数据
        print(f"文档元数据: {documents[0].metadata}")
        
        # --- 文档分割 ---
        print("\n--- 文档分割 ---")
        
        # 使用CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        character_splits = text_splitter.split_documents(documents)
        print(f"使用CharacterTextSplitter分割后获得了 {len(character_splits)} 个文本块")
        
        # 使用RecursiveCharacterTextSplitter
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )
        recursive_splits = recursive_splitter.split_documents(documents)
        print(f"使用RecursiveCharacterTextSplitter分割后获得了 {len(recursive_splits)} 个文本块")
        
        # 显示一个分割的文本块
        if recursive_splits:
            print(f"\n文本块示例:\n{recursive_splits[0].page_content}")
        
        # --- 创建嵌入 ---
        print("\n--- 创建嵌入 ---")
        
        # 尝试使用HuggingFace嵌入（无需API密钥）
        try:
            print("尝试使用HuggingFace嵌入模型...")
            hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # 计算一个示例文本的嵌入
            sample_embedding = hf_embeddings.embed_query("人工智能的应用")
            print(f"嵌入向量维度: {len(sample_embedding)}")
            print(f"嵌入向量示例（前5个值）: {sample_embedding[:5]}")
            
        except Exception as e:
            print(f"使用HuggingFace嵌入时出错: {e}")
            print("请确保已安装sentence-transformers包: pip install sentence-transformers")
        
        # 尝试使用OpenAI嵌入（需要API密钥）
        if os.getenv("OPENAI_API_KEY"):
            print("\n使用OpenAI嵌入模型...")
            openai_embeddings = OpenAIEmbeddings()
            
            sample_embedding = openai_embeddings.embed_query("人工智能的应用")
            print(f"嵌入向量维度: {len(sample_embedding)}")
            print(f"嵌入向量示例（前5个值）: {sample_embedding[:5]}")
        else:
            print("\n未设置OPENAI_API_KEY，跳过OpenAI嵌入示例")
            print("要使用OpenAI嵌入，请设置OPENAI_API_KEY环境变量")
    
    except Exception as e:
        print(f"文档处理过程中出错: {e}")


# ===============================
# 第2部分：向量存储类型
# ===============================
def vector_store_types_examples():
    """不同向量存储类型示例"""
    print("\n=== 不同向量存储类型示例 ===")
    
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS, Chroma
    
    # 准备示例文档
    print("\n--- 准备示例文档 ---")
    
    # 创建或加载示例文档
    docs = []
    
    sample_texts = [
        "人工智能是研究开发能够模拟人类智能的计算机系统的科学与工程。",
        "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。",
        "深度学习是机器学习的一种技术，使用多层神经网络从大量数据中学习特征和模式。",
        "自然语言处理（NLP）是AI的一个分支，专注于使计算机能够理解、解释和生成人类语言。",
        "计算机视觉是AI的一个领域，致力于使计算机能够从图像或视频中获取高级理解。",
        "强化学习是一种机器学习方法，其中代理通过与环境交互并接收反馈来学习。",
        "语义网是万维网的扩展，它使网页上的数据能够被计算机理解和处理。",
        "知识图谱是一种结构化的知识表示形式，通过实体和关系描述现实世界的事实。"
    ]
    
    from langchain_core.documents import Document
    for i, text in enumerate(sample_texts):
        docs.append(Document(page_content=text, metadata={"source": f"sample-{i}", "id": i}))
    
    print(f"准备了 {len(docs)} 个文档示例")
    
    # --- 创建FAISS向量存储 ---
    print("\n--- 创建FAISS向量存储 ---")
    
    try:
        # 尝试使用HuggingFace嵌入（无需API密钥）
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 创建FAISS向量存储
        print("使用FAISS创建向量存储...")
        faiss_db = FAISS.from_documents(docs, embeddings)
        
        # 保存到本地
        temp_faiss_dir = os.path.join(tempfile.mkdtemp(), "faiss_index")
        faiss_db.save_local(temp_faiss_dir)
        print(f"FAISS向量存储已保存到: {temp_faiss_dir}")
        
        # 加载保存的向量存储
        loaded_faiss_db = FAISS.load_local(temp_faiss_dir, embeddings)
        print("成功加载保存的FAISS向量存储")
        
        # --- 创建Chroma向量存储 ---
        print("\n--- 创建Chroma向量存储 ---")
        
        # 创建临时目录用于Chroma
        temp_chroma_dir = os.path.join(tempfile.mkdtemp(), "chroma_db")
        
        # 创建Chroma向量存储
        print(f"使用Chroma创建向量存储在: {temp_chroma_dir}")
        chroma_db = Chroma.from_documents(
            docs, 
            embeddings,
            persist_directory=temp_chroma_dir
        )
        
        # 持久化Chroma数据库
        chroma_db.persist()
        print("Chroma向量存储已持久化")
        
        # 重新加载Chroma数据库
        loaded_chroma_db = Chroma(
            persist_directory=temp_chroma_dir, 
            embedding_function=embeddings
        )
        print("成功加载持久化的Chroma向量存储")
        
        # --- 其他向量存储选项 ---
        print("\n--- 其他向量存储选项 ---")
        print("LangChain还支持多种其他向量存储：")
        print("1. Pinecone - 托管式向量数据库服务")
        print("2. Weaviate - 开源向量搜索引擎")
        print("3. Milvus - 开源向量数据库，适合大规模应用")
        print("4. Qdrant - 开源向量相似度搜索引擎")
        print("5. PGVector - PostgreSQL的向量扩展")
        print("6. ElasticVectorSearch - Elasticsearch的向量搜索功能")
        print("需要额外设置和API密钥才能使用这些服务")
    
    except Exception as e:
        print(f"创建向量存储时出错: {e}")
        print("请确保已安装相关依赖: pip install faiss-cpu sentence-transformers chromadb")


# ===============================
# 第3部分：相似性搜索
# ===============================
def similarity_search_examples():
    """向量存储相似性搜索示例"""
    print("\n=== 向量存储相似性搜索示例 ===")
    
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    
    # --- 准备示例数据 ---
    print("\n--- 准备示例数据 ---")
    
    # 创建一些包含技术文档的示例
    tech_docs = [
        Document(
            page_content="Python是一种高级编程语言，以其简洁、易读的语法和丰富的库而闻名。它支持多种编程范式，包括面向对象、命令式和函数式编程。",
            metadata={"category": "编程语言", "difficulty": "初级", "tags": ["Python", "编程"]}
        ),
        Document(
            page_content="JavaScript是一种主要用于网页开发的脚本语言。它允许开发者在网页上实现复杂的功能，如动态内容更新和交互式地图。",
            metadata={"category": "编程语言", "difficulty": "中级", "tags": ["JavaScript", "Web开发"]}
        ),
        Document(
            page_content="机器学习是人工智能的一个子领域，专注于开发能够从数据中学习并作出预测的算法和模型，而无需显式编程。",
            metadata={"category": "人工智能", "difficulty": "高级", "tags": ["机器学习", "AI"]}
        ),
        Document(
            page_content="深度学习是机器学习的一个分支，使用多层神经网络处理数据。这些网络能够从大量数据中学习复杂模式和表示。",
            metadata={"category": "人工智能", "difficulty": "高级", "tags": ["深度学习", "神经网络"]}
        ),
        Document(
            page_content="数据结构是组织和存储数据的特定方式，使其能够高效地被访问和修改。常见的数据结构包括数组、链表、树和图。",
            metadata={"category": "计算机科学", "difficulty": "中级", "tags": ["数据结构", "算法"]}
        ),
        Document(
            page_content="Docker是一个开源平台，允许开发者将应用程序与其所有依赖项一起打包到一个标准化的容器中，从而简化了部署过程。",
            metadata={"category": "DevOps", "difficulty": "中级", "tags": ["Docker", "容器化"]}
        ),
        Document(
            page_content="云计算是通过互联网提供计算服务的模式，包括服务器、存储、数据库、网络、软件和分析等。它提供了灵活的资源和经济高效的可扩展性。",
            metadata={"category": "云技术", "difficulty": "中级", "tags": ["云计算", "AWS", "Azure"]}
        ),
        Document(
            page_content="区块链是一种分布式账本技术，可以安全地记录交易和资产追踪。它最著名的应用是加密货币，但也可用于供应链管理和身份验证。",
            metadata={"category": "区块链", "difficulty": "高级", "tags": ["区块链", "分布式系统", "密码学"]}
        )
    ]
    
    print(f"准备了 {len(tech_docs)} 个技术文档示例")
    
    try:
        # 创建嵌入模型
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 创建向量存储
        vector_db = FAISS.from_documents(tech_docs, embeddings)
        print("已创建FAISS向量存储")
        
        # --- 基础相似性搜索 ---
        print("\n--- 基础相似性搜索 ---")
        
        query = "Python编程语言的特点是什么"
        print(f"查询: '{query}'")
        
        # 执行基本搜索
        search_results = vector_db.similarity_search(query, k=2)
        
        print("\n结果:")
        for i, doc in enumerate(search_results):
            print(f"\n结果 {i+1}:")
            print(f"内容: {doc.page_content}")
            print(f"元数据: {doc.metadata}")
        
        # --- 带分数的相似性搜索 ---
        print("\n--- 带分数的相似性搜索 ---")
        
        query = "机器学习和深度学习的区别"
        print(f"查询: '{query}'")
        
        # 执行带分数的搜索
        search_results_with_score = vector_db.similarity_search_with_score(query, k=2)
        
        print("\n结果:")
        for i, (doc, score) in enumerate(search_results_with_score):
            print(f"\n结果 {i+1}:")
            print(f"相似度分数: {score}")
            print(f"内容: {doc.page_content}")
            print(f"元数据: {doc.metadata}")
        
        # --- 带过滤器的相似性搜索 ---
        print("\n--- 带过滤器的相似性搜索 ---")
        
        query = "先进技术"  # 一个比较宽泛的查询
        filter_dict = {"category": "人工智能"}  # 只搜索人工智能类别
        
        print(f"查询: '{query}' 过滤条件: {filter_dict}")
        
        # 使用过滤器搜索
        filtered_results = vector_db.similarity_search(
            query,
            k=2,
            filter=filter_dict
        )
        
        print("\n结果:")
        for i, doc in enumerate(filtered_results):
            print(f"\n结果 {i+1}:")
            print(f"内容: {doc.page_content}")
            print(f"元数据: {doc.metadata}")
            
        # --- 最大边际相关性(MMR)搜索 ---
        print("\n--- 最大边际相关性(MMR)搜索 ---")
        print("MMR能够提高结果多样性，避免返回过于相似的文档")
        
        query = "人工智能技术"
        print(f"查询: '{query}'")
        
        # 使用MMR搜索
        mmr_results = vector_db.max_marginal_relevance_search(
            query,
            k=2,  # 返回的总文档数
            fetch_k=3,  # 初始获取的文档数
            lambda_mult=0.5  # 相关性与多样性平衡参数(0-1)，越接近0越多样
        )
        
        print("\n结果:")
        for i, doc in enumerate(mmr_results):
            print(f"\n结果 {i+1}:")
            print(f"内容: {doc.page_content}")
            print(f"元数据: {doc.metadata}")
    
    except Exception as e:
        print(f"执行相似性搜索时出错: {e}")
        print("请确保已安装相关依赖: pip install faiss-cpu sentence-transformers")


# ===============================
# 第4部分：检索器与RAG链
# ===============================
def retriever_and_rag_examples():
    """检索器与RAG系统示例"""
    print("\n=== 检索器与RAG系统示例 ===")
    
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    
    # --- 创建文档库 ---
    print("\n--- 创建文档库 ---")
    
    # 准备一些中文历史文档示例
    history_docs = [
        Document(
            page_content="""秦始皇（公元前259年－公元前210年），嬴姓，名政，秦庄襄王之子。秦始皇是中国历史上第一个使用"皇帝"称号的君主，也是中国第一个大一统王朝"秦朝"的开国君主。
他在位期间，于公元前221年完成统一六国的大业，建立起中国历史上第一个多民族的中央集权国家秦朝。秦始皇采取一系列重要措施强化中央集权统治：统一文字、度量衡和货币，车同轨，书同文；在全国范围内实行郡县制；并修筑万里长城抵御匈奴的入侵。""",
            metadata={"period": "秦朝", "person": "秦始皇", "event": "统一中国"}
        ),
        Document(
            page_content="""汉武帝刘彻（公元前156年－公元前87年），汉朝第七位皇帝，历史上著名的政治家。他在位期间，开拓疆土，西域都护府的设立使得丝绸之路更为通畅；
他重视发展经济，采取了"推恩令"和"算缗令"等社会经济改革；他推行"罢黜百家，独尊儒术"的文化政策，对中国的政治、经济和文化产生了深远影响。""",
            metadata={"period": "汉朝", "person": "汉武帝", "event": "汉武盛世"}
        ),
        Document(
            page_content="""唐太宗李世民（599年－649年），唐朝第二位皇帝，杰出的政治家和战略家。他开创了历史上著名的"贞观之治"，与其父李渊共同建立了唐朝。
他在位期间重视任贤纳谏，虚心纳谏；注重发展农业和经济；文治武功，使唐朝成为当时世界上最为强大和先进的国家之一。""",
            metadata={"period": "唐朝", "person": "唐太宗", "event": "贞观之治"}
        ),
        Document(
            page_content="""宋太祖赵匡胤（927年－976年），宋朝开国皇帝。他发动陈桥兵变，黄袍加身建立宋朝。
在位期间，他创立"杯酒释兵权"的方式削弱了军阀势力；加强中央集权；发展科举制度，选拔人才；重视文教发展，为北宋的繁荣奠定了基础。""",
            metadata={"period": "宋朝", "person": "宋太祖", "event": "陈桥兵变"}
        ),
        Document(
            page_content="""康熙帝爱新觉罗·玄烨（1654年－1722年），清朝第四位皇帝，中国历史上在位时间最长的皇帝之一。
他平定了"三藩之乱"，收复台湾，巩固了清朝的统治；抵御沙俄入侵，签订《尼布楚条约》；提倡发展农业和经济，减轻赋税；重视文化教育，编撰《康熙字典》等巨著，开创了"康乾盛世"的先声。""",
            metadata={"period": "清朝", "person": "康熙帝", "event": "康熙盛世"}
        ),
    ]
    
    print(f"准备了 {len(history_docs)} 个中国历史文档示例")
    
    try:
        # 创建嵌入模型
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 创建向量存储
        vector_db = FAISS.from_documents(history_docs, embeddings)
        print("已创建FAISS向量存储")
        
        # --- 创建并使用检索器 ---
        print("\n--- 创建并使用检索器 ---")
        
        # 创建基本检索器
        basic_retriever = vector_db.as_retriever(search_kwargs={"k": 2})
        print("创建了基本检索器")
        
        # 测试检索器
        query = "唐太宗的主要成就是什么？"
        print(f"查询: '{query}'")
        
        retrieval_results = basic_retriever.invoke(query)
        print(f"\n检索到 {len(retrieval_results)} 个相关文档")
        
        for i, doc in enumerate(retrieval_results):
            print(f"\n文档 {i+1}:")
            print(f"相关性内容: {doc.page_content[:100]}...")
        
        # --- 创建基于检索的问答链 ---
        print("\n--- 创建基于检索的问答链 ---")
        
        # 检查是否有OpenAI API密钥
        if os.getenv("OPENAI_API_KEY"):
            # 创建语言模型
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            
            # 创建自定义提示模板
            template = """你是一位专业的中国历史学家。请使用以下历史文档中提供的信息回答问题。
            如果无法从文档中找到答案，请说明你不知道，不要编造信息。
            
            问题: {question}
            
            可用文档:
            {context}
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["question", "context"]
            )
            
            # 创建问答链
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",  # 直接将所有文档内容拼接在一起
                retriever=basic_retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            # 运行问答链
            print("运行基于检索的问答链...")
            query = "请介绍秦始皇的主要成就。"
            print(f"问题: {query}")
            
            result = qa_chain.invoke({"query": query})
            print("\n回答:")
            print(result["result"])
            
            # --- LCEL示例 ---
            print("\n--- 使用LCEL创建RAG链 ---")
            
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough
            
            # 创建提示模板
            prompt_template = """你是一位专业的中国历史学家，请基于以下信息回答问题。
            如果提供的信息不足以回答问题，请说明你不知道，不要编造信息。
            
            提供的信息:
            {context}
            
            问题: {question}
            
            回答:"""
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # 定义格式化上下文的函数
            def format_docs(docs):
                return "\n\n".join([doc.page_content for doc in docs])
            
            # 组装RAG链
            rag_chain = (
                {"context": basic_retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # 运行LCEL链
            print("运行LCEL风格的RAG链...")
            query = "康熙皇帝有什么主要的政治成就？"
            print(f"问题: {query}")
            
            response = rag_chain.invoke(query)
            print("\n回答:")
            print(response)
        else:
            print("\n未设置OPENAI_API_KEY，跳过RAG问答示例")
            print("要运行RAG系统，请设置OPENAI_API_KEY环境变量")
    
    except Exception as e:
        print(f"运行检索器与RAG示例时出错: {e}")


# ===============================
# 主函数：运行所有示例
# ===============================
def main():
    """运行所有向量存储组件示例"""
    print("# LangChain向量存储(Vector Stores)组件使用示例")
    print("=" * 80)
    
    print("\n## 重要提示")
    print("运行这些示例前，请确保已安装所需依赖:")
    print("pip install langchain langchain-openai faiss-cpu sentence-transformers chromadb")
    print("对于部分功能，还需要设置相应的API密钥，例如OpenAI API密钥:")
    print("```python")
    print('import os')
    print('os.environ["OPENAI_API_KEY"] = "您的OpenAI API密钥"')
    print("```")
    
    # 运行示例
    document_processing_examples()
    vector_store_types_examples()
    similarity_search_examples()
    retriever_and_rag_examples()
    
    print("\n" + "=" * 80)
    print("示例运行完成！请根据上述演示进行您自己的向量存储和检索应用开发。")


if __name__ == "__main__":
    main()
