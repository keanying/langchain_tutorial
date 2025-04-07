#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain链(Chains)组件使用示例

本模块展示了如何使用LangChain中的各种链组件来构建复杂的工作流，包括：
1. 基础LLM链
2. 顺序链与简单顺序链
3. 路由链
4. 转换链
5. 检索链和问答链
6. 自定义链

注意：运行这些示例前，请确保已设置相应的API密钥环境变量
"""

import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===============================
# 第1部分：基础LLM链
# ===============================
def basic_llm_chains():
    """基础LLM链示例"""
    print("=== 基础LLM链示例 ===")
    
    from langchain_openai import OpenAI
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    
    # --- 示例1：简单的LLM链 ---
    print("\n--- 简单的LLM链示例 ---")
    
    # 创建提示模板
    template = """你是一位专业的{profession}。
    请用专业但通俗易懂的语言，解释'{concept}'的概念。
    解释应该不超过100字。"""
    
    prompt = PromptTemplate(
        input_variables=["profession", "concept"],
        template=template
    )
    
    # 创建语言模型
    try:
        # 检查是否设置了API密钥
        if os.getenv("OPENAI_API_KEY"):
            llm = OpenAI(temperature=0.7)
            
            # 创建LLM链
            chain = LLMChain(llm=llm, prompt=prompt)
            
            # 运行链
            response = chain.invoke({"profession": "物理学家", "concept": "量子纠缠"})
            
            print("输入: 物理学家解释'量子纠缠'")
            print(f"输出: {response['text']}")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("LLM链将提示模板和语言模型结合在一起，是LangChain中最基本的链类型。")
    
    except Exception as e:
        print(f"运行LLM链时出错: {e}")
    
    # --- 示例2：链的函数式调用 ---
    print("\n--- 链的函数式调用示例 ---")
    
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 创建语言模型
            llm = OpenAI(temperature=0.5)
            
            # 创建函数式链
            functional_chain = (
                {"profession": RunnablePassthrough(), "concept": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # 运行链
            result = functional_chain.invoke({"profession": "计算机科学家", "concept": "递归算法"})
            
            print("输入: 计算机科学家解释'递归算法'")
            print(f"输出: {result}")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("函数式链使用更现代的API设计，支持链式操作，更灵活和直观。")
    
    except Exception as e:
        print(f"运行函数式链时出错: {e}")
    
    # --- 示例3：带有记忆的链 ---
    print("\n--- 带有记忆的链示例 ---")
    
    from langchain.memory import ConversationBufferMemory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 创建记忆组件
            memory = ConversationBufferMemory(return_messages=True)
            
            # 创建聊天提示模板
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一位友好而专业的AI助手。"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # 创建带有记忆的链
            from langchain_openai import ChatOpenAI
            chat_model = ChatOpenAI(temperature=0.7)
            
            memory_chain = (
                {"history": memory.load_memory_variables, "input": RunnablePassthrough()}
                | chat_prompt
                | chat_model
            )
            
            # 模拟对话
            print("模拟对话:")
            
            # 第一轮对话
            response = memory_chain.invoke("你好，请问你是谁？")
            print("用户: 你好，请问你是谁？")
            print(f"AI: {response.content}")
            memory.save_context({"input": "你好，请问你是谁？"}, {"output": response.content})
            
            # 第二轮对话，记忆应该保留上下文
            response = memory_chain.invoke("你能做些什么？")
            print("\n用户: 你能做些什么？")
            print(f"AI: {response.content}")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("带有记忆的链可以维护对话历史，让AI在回复时考虑之前的交互内容。")
    
    except Exception as e:
        print(f"运行带有记忆的链时出错: {e}")


# ===============================
# 第2部分：复杂链类型
# ===============================
def complex_chain_types():
    """复杂链类型示例"""
    print("\n=== 复杂链类型示例 ===")
    
    from langchain_openai import OpenAI
    from langchain.chains import SimpleSequentialChain, SequentialChain
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    
    # --- 示例1：简单顺序链 ---
    print("\n--- 简单顺序链(SimpleSequentialChain)示例 ---")
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 创建第一个链：生成故事创意
            idea_template = """根据以下主题生成一个简短的故事创意：{topic}。
            创意应该包括主要角色和一个冲突点。限制在50字以内。"""
            idea_prompt = PromptTemplate(template=idea_template, input_variables=["topic"])
            llm = OpenAI(temperature=0.7)
            idea_chain = LLMChain(llm=llm, prompt=idea_prompt, output_key="idea")
            
            # 创建第二个链：扩展故事创意
            expansion_template = """基于以下故事创意，扩展成一个简短的故事开头：
            
            故事创意：{idea}
            
            故事开头（约100字）："""
            expansion_prompt = PromptTemplate(template=expansion_template, input_variables=["idea"])
            expansion_chain = LLMChain(llm=llm, prompt=expansion_prompt, output_key="expansion")
            
            # 创建简单顺序链
            simple_chain = SimpleSequentialChain(
                chains=[idea_chain, expansion_chain],
                verbose=True
            )
            
            # 运行链
            print("运行简单顺序链...")
            topic = "太空探险"  # 输入主题
            print(f"输入主题: {topic}")
            result = simple_chain.invoke({"topic": topic})
            print(f"\n最终输出: {result['output']}")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("简单顺序链将多个链按顺序连接，前一个链的输出作为下一个链的输入。")
            print("它只处理单个输入值，并返回单个输出值。")
    
    except Exception as e:
        print(f"运行简单顺序链时出错: {e}")
    
    # --- 示例2：顺序链 ---
    print("\n--- 顺序链(SequentialChain)示例 ---")
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 创建第一个链：生成角色
            character_template = """创建一个'{genre}'类型故事的主角。包括姓名、年龄和一个特点。输出限制在30字以内。"""
            character_prompt = PromptTemplate(template=character_template, input_variables=["genre"])
            character_chain = LLMChain(llm=llm, prompt=character_prompt, output_key="character")
            
            # 创建第二个链：生成场景
            setting_template = """创建一个'{genre}'类型故事的场景，适合'{character}'这个角色。场景描述限制在30字以内。"""
            setting_prompt = PromptTemplate(template=setting_template, input_variables=["genre", "character"])
            setting_chain = LLMChain(llm=llm, prompt=setting_prompt, output_key="setting")
            
            # 创建第三个链：生成故事情节
            plot_template = """基于以下信息，创建一个简短的故事情节：
            
            角色：{character}
            场景：{setting}
            类型：{genre}
            
            故事情节（约100字）："""
            plot_prompt = PromptTemplate(template=plot_template, input_variables=["character", "setting", "genre"])
            plot_chain = LLMChain(llm=llm, prompt=plot_prompt, output_key="plot")
            
            # 创建顺序链
            sequential_chain = SequentialChain(
                chains=[character_chain, setting_chain, plot_chain],
                input_variables=["genre"],
                output_variables=["character", "setting", "plot"],
                verbose=True
            )
            
            # 运行链
            print("运行顺序链...")
            genre = "科幻"  # 输入类型
            print(f"输入类型: {genre}")
            result = sequential_chain.invoke({"genre": genre})
            
            print("\n输出结果:")
            print(f"角色: {result['character']}")
            print(f"场景: {result['setting']}")
            print(f"情节: {result['plot']}")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("顺序链是简单顺序链的更强大版本，可以处理多个输入和输出变量。")
            print("它能在链之间传递多个值，而不仅仅是单个输入和输出。")
    
    except Exception as e:
        print(f"运行顺序链时出错: {e}")

    # --- 示例3：函数式顺序链 ---
    print("\n--- 函数式顺序链示例 ---")
    
    from langchain_core.output_parsers import StrOutputParser
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 定义多个步骤
            def generate_product_idea(input_dict):
                product_type = input_dict["product_type"]
                template = f"创建一个创新的{product_type}产品创意，包括产品名称和主要特点。限制在50字以内。"
                return template
            
            def generate_marketing_pitch(input_dict):
                return f"为以下产品创建一个吸引人的营销口号：{input_dict['idea']}。口号应该简短有力。"
            
            def generate_target_audience(input_dict):
                return f"确定以下产品的目标受众群体：{input_dict['idea']}\n{input_dict['pitch']}"
            
            # 创建函数式顺序链
            from langchain_core.runnables import RunnableLambda, RunnablePassthrough
            
            functional_sequential_chain = (
                {"product_type": RunnablePassthrough()}
                | RunnableLambda(generate_product_idea)
                | llm
                | StrOutputParser()
                | {"idea": RunnablePassthrough(), "product_type": lambda _: "智能家居"}
                | RunnableLambda(generate_marketing_pitch)
                | llm
                | StrOutputParser()
                | {"pitch": RunnablePassthrough(), "idea": lambda x: x}
                | RunnableLambda(generate_target_audience)
                | llm
                | StrOutputParser()
            )
            
            # 运行链
            print("运行函数式顺序链...")
            product_type = "智能家居"  # 输入产品类型
            print(f"输入产品类型: {product_type}")
            result = functional_sequential_chain.invoke(product_type)
            print(f"\n最终输出: {result}")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("函数式顺序链使用现代的LCEL API来创建更灵活、更可组合的链。")
            print("它允许使用链式操作符（|）将各个组件连接起来，使代码更清晰、更直观。")
    
    except Exception as e:
        print(f"运行函数式顺序链时出错: {e}")


# ===============================
# 第3部分：路由链
# ===============================
def router_chain_examples():
    """路由链示例"""
    print("\n=== 路由链示例 ===")
    
    from langchain_openai import OpenAI
    from langchain.chains.router import MultiPromptChain
    from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
    from langchain.chains.router import MultiRouteChain
    
    # --- 示例1：多提示路由链 ---
    print("\n--- 多提示路由链(MultiPromptChain)示例 ---")
    
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_core.output_parsers import StrOutputParser
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 创建语言模型
            llm = OpenAI(temperature=0)
            
            # 创建专业领域的提示模板
            physics_template = """你是一名物理学专家。请回答以下关于物理学的问题：
            {input}"""
            physics_prompt = PromptTemplate(template=physics_template, input_variables=["input"])
            physics_chain = LLMChain(llm=llm, prompt=physics_prompt)
            
            math_template = """你是一名数学专家。请回答以下关于数学的问题：
            {input}"""
            math_prompt = PromptTemplate(template=math_template, input_variables=["input"])
            math_chain = LLMChain(llm=llm, prompt=math_prompt)
            
            history_template = """你是一名历史学专家。请回答以下关于历史的问题：
            {input}"""
            history_prompt = PromptTemplate(template=history_template, input_variables=["input"])
            history_chain = LLMChain(llm=llm, prompt=history_prompt)
            
            # 创建路由链的目标链列表
            destination_chains = {
                "物理": physics_chain,
                "数学": math_chain,
                "历史": history_chain
            }
            
            # 创建默认链（当无法确定路由时使用）
            default_chain = LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    template="{input}\n请以一般知识的角度回答这个问题。",
                    input_variables=["input"]
                )
            )
            
            # 创建路由提示
            router_template = """给定一个用户问题，将其路由到最合适的领域。
            领域选项：物理、数学、历史。
            
            用户问题：{input}
            
            路由到的领域："""
            
            # 创建多提示路由链
            chain = MultiPromptChain(
                router_chain=LLMRouterChain.from_llm(llm, router_template),
                destination_chains=destination_chains,
                default_chain=default_chain,
                verbose=True
            )
            
            # 测试不同领域的问题
            questions = [
                "什么是相对论？",
                "如何求解二次方程？",
                "谁是中国的第一位皇帝？",
                "为什么天空是蓝色的？"
            ]
            
            # 运行路由链
            print("运行多提示路由链...")
            for question in questions:
                print(f"\n问题: {question}")
                result = chain.invoke({"input": question})
                print(f"回答: {result['text']}")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("多提示路由链会根据输入内容自动选择最合适的专家链来处理请求。")
            print("它使用LLM来分析输入并决定将其路由到哪个目标链。")
    
    except Exception as e:
        print(f"运行多提示路由链时出错: {e}")

    # --- 示例2：使用函数式API的路由链 ---
    print("\n--- 函数式路由链示例 ---")
    
    from langchain_core.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    from typing import Literal
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 定义路由决策模式
            class RouterDecision(BaseModel):
                """路由决策"""
                destination: Literal["技术支持", "销售", "一般咨询"] = Field(
                    description="基于查询内容选择的最合适的部门"
                )
                next_input: str = Field(
                    description="传递给选定部门的修改后的查询"
                )
            
            # 创建输出解析器
            router_parser = PydanticOutputParser(pydantic_object=RouterDecision)
            
            # 创建路由提示模板
            router_prompt = PromptTemplate(
                template="""作为客户服务路由系统，你的任务是将客户查询路由到正确的部门。
                
                客户查询: {query}
                
                请根据查询内容，选择最合适的部门，并提供可能的修改后的查询。
                
                {format_instructions}
                """,
                input_variables=["query"],
                partial_variables={"format_instructions": router_parser.get_format_instructions()}
            )
            
            # 创建各部门的回复链
            tech_support_chain = (
                PromptTemplate.from_template(
                    "你是技术支持专员。请回答以下技术问题：{next_input}"
                )
                | llm
                | StrOutputParser()
            )
            
            sales_chain = (
                PromptTemplate.from_template(
                    "你是销售代表。请回应以下销售相关查询：{next_input}"
                )
                | llm
                | StrOutputParser()
            )
            
            general_chain = (
                PromptTemplate.from_template(
                    "你是客户服务代表。请回应以下一般咨询：{next_input}"
                )
                | llm
                | StrOutputParser()
            )
            
            # 创建路由链
            from langchain_core.runnables import RunnableBranch
            
            router_chain = router_prompt | llm | router_parser
            
            branch = RunnableBranch(
                (lambda x: x["destination"] == "技术支持", tech_support_chain),
                (lambda x: x["destination"] == "销售", sales_chain),
                general_chain,
            )
            
            # 组合路由逻辑和分支
            chain = {"query": lambda x: x["query"]} | router_chain | branch
            
            # 测试查询
            queries = [
                "我的软件无法正常工作，一直显示错误代码404",
                "我想了解你们最新产品的价格和功能",
                "你们公司的办公时间是什么？"
            ]
            
            # 运行函数式路由链
            print("运行函数式路由链...")
            for query in queries:
                print(f"\n查询: {query}")
                result = chain.invoke({"query": query})
                print(f"回复: {result}")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("函数式路由链使用LCEL API创建更灵活的路由逻辑。")
            print("它使用Pydantic模型来结构化路由决策，并使用RunnableBranch来执行条件路由。")
    
    except Exception as e:
        print(f"运行函数式路由链时出错: {e}")


# ===============================
# 第4部分：转换链
# ===============================
def transform_chain_examples():
    """转换链示例"""
    print("\n=== 转换链示例 ===")
    
    from langchain.chains import TransformChain
    from langchain_core.runnables import RunnableLambda
    
    # --- 示例1：基本转换链 ---
    print("\n--- 基本转换链(TransformChain)示例 ---")
    
    # 定义转换函数
    def extract_names_and_capitalize(inputs: dict) -> dict:
        """提取文本中的人名并转换为大写"""
        text = inputs["text"]
        
        # 简单的假设：假设所有的人名都是以"叫"或"名为"引导的
        import re
        names = re.findall(r'[叫名为](\w+)[，。,.]', text)
        
        # 如果没找到名字，则尝试找引号中的内容
        if not names:
            names = re.findall(r'["\'\'](\w+)["\'\'同学]', text)
        
        # 转换为大写（对英文名有效）
        capitalized_names = [name.upper() for name in names] if names else ["未找到名字"]
        
        return {"names": capitalized_names, "original_text": text}
    
    # 创建转换链
    transform_chain = TransformChain(
        input_variables=["text"],
        output_variables=["names", "original_text"],
        transform=extract_names_and_capitalize
    )
    
    # 测试转换链
    test_texts = [
        "班上有一个学生叫小明，他成绩很好。",
        "我的朋友名为Jack，他来自美国。",
        "大家都叫他\"天才\"，因为他很聪明。"
    ]
    
    # 运行转换链
    print("运行基本转换链...")
    for text in test_texts:
        result = transform_chain.invoke({"text": text})
        print(f"\n原文: {result['original_text']}")
        print(f"提取的名字: {', '.join(result['names'])}")
    
    # --- 示例2：函数式转换链 ---
    print("\n--- 函数式转换链示例 ---")
    
    # 定义多个转换函数
    def count_words(inputs: dict) -> dict:
        """计算文本中的单词数量"""
        text = inputs["text"]
        word_count = len(text.split())
        return {"word_count": word_count, **inputs}
    
    def analyze_sentiment(inputs: dict) -> dict:
        """简单的情感分析"""
        text = inputs["text"].lower()
        
        # 简单的情感词典
        positive_words = ["喜欢", "爱", "好", "棒", "优秀", "happy", "good", "great", "excellent"]
        negative_words = ["讨厌", "恨", "差", "糟", "失败", "sad", "bad", "terrible", "poor"]
        
        # 计算正面和负面词的数量
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        # 确定情感倾向
        if positive_count > negative_count:
            sentiment = "积极"
        elif negative_count > positive_count:
            sentiment = "消极"
        else:
            sentiment = "中性"
        
        return {"sentiment": sentiment, **inputs}
    
    def summarize_analysis(inputs: dict) -> dict:
        """总结分析结果"""
        summary = f"文本包含{inputs['word_count']}个单词，情感倾向为{inputs['sentiment']}。"
        
        return {"summary": summary, **inputs}
    
    # 创建函数式转换链
    functional_transform_chain = (
        RunnableLambda(lambda x: {"text": x})
        | RunnableLambda(count_words)
        | RunnableLambda(analyze_sentiment)
        | RunnableLambda(summarize_analysis)
    )
    
    # 测试函数式转换链
    test_texts = [
        "我非常喜欢这部电影，演员表演很棒，故事情节也很感人。",
        "这家餐厅的服务态度差，食物也不好吃，价格还贵。",
        "今天天气多云，气温适中，适合户外活动。"
    ]
    
    # 运行函数式转换链
    print("运行函数式转换链...")
    for text in test_texts:
        result = functional_transform_chain.invoke(text)
        print(f"\n文本: '{text}'")
        print(f"分析结果: {result['summary']}")


# ===============================
# 第5部分：检索链和问答链
# ===============================
def retrieval_qa_chain_examples():
    """检索链和问答链示例"""
    print("\n=== 检索链和问答链示例 ===")
    
    from langchain_community.vectorstores import Chroma, FAISS
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains import RetrievalQA
    
    # --- 示例1：基于文档的问答链 ---
    print("\n--- 基于文档的问答链示例 ---")
    
    # 创建一些示例文档
    documents = [
        "人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。人工智能领域包括机器学习、自然语言处理、计算机视觉等子领域。",
        "机器学习是人工智能的一个子集，专注于使计算机系统通过经验自动改进。主要的机器学习方法包括监督学习、无监督学习和强化学习。",
        "深度学习是机器学习的一个分支，使用多层神经网络来分析各种类型的数据。卷积神经网络（CNN）和循环神经网络（RNN）是两种常见的深度学习架构。",
        "自然语言处理（NLP）是人工智能的一个领域，专注于使计算机能够理解和生成人类语言。NLP应用包括机器翻译、情感分析和问答系统。",
        "大型语言模型（LLM）如GPT是基于Transformer架构的深度学习模型，经过大规模文本数据训练，能够生成连贯的文本、回答问题和执行各种语言任务。"
    ]
    
    try:
        # 创建文本分割器
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        texts = text_splitter.split_text("\n".join(documents))
        
        # 使用HuggingFace嵌入（不需要API密钥）
        embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        
        # 创建向量存储
        try:
            # 尝试创建向量存储
            vector_store = FAISS.from_texts(texts, embeddings)
            
            # 创建检索器
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})
            
            # 创建问答链
            from langchain_openai import OpenAI
            
            if os.getenv("OPENAI_API_KEY"):
                # 使用OpenAI
                qa_chain = RetrievalQA.from_chain_type(
                    llm=OpenAI(temperature=0),
                    chain_type="stuff",
                    retriever=retriever,
                    verbose=True
                )
                
                # 示例问题
                questions = [
                    "什么是人工智能？",
                    "机器学习和深度学习有什么区别？",
                    "大型语言模型的应用有哪些？"
                ]
                
                # 运行问答链
                print("运行问答链...")
                for question in questions:
                    print(f"\n问题: {question}")
                    result = qa_chain.invoke({"query": question})
                    print(f"回答: {result['result']}")
            else:
                print("未设置OPENAI_API_KEY，跳过模型调用")
                print("检索问答链结合了检索系统和语言模型，以便基于检索到的文档回答问题。")
        
        except Exception as e:
            print(f"创建向量存储时出错: {e}")
            print("可能是因为缺少必要的依赖项。请确保已安装相关包，如sentence-transformers。")
            print("可以通过运行'pip install sentence-transformers faiss-cpu'来安装所需依赖。")
    
    except Exception as e:
        print(f"准备检索链示例时出错: {e}")
    
    # --- 示例2：函数式检索链 ---
    print("\n--- 函数式检索链示例 ---")
    print("函数式检索链使用LCEL API创建更灵活的检索和问答工作流。")
    print("由于依赖关系，这里只展示了示例代码，不实际运行：")
    
    code_example = '''
# 导入必要的模块
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# 创建向量存储和检索器
embeddings = OpenAIEmbeddings()
vector_store = Chroma(..., embeddings=embeddings)
retriever = vector_store.as_retriever()

# 创建提示模板
template = """基于以下上下文回答问题。如果无法从上下文中找到答案，就说你不知道。

上下文: {context}

问题: {question}

回答: """
prompt = ChatPromptTemplate.from_template(template)

# 创建函数式检索链
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)

# 运行链
response = chain.invoke("什么是向量数据库？")
print(response)
'''
    
    print(f"```python\n{code_example}\n```")


# ===============================
# 第6部分：自定义链
# ===============================
def custom_chain_examples():
    """自定义链示例"""
    print("\n=== 自定义链示例 ===")
    
    from langchain.chains.base import Chain
    from typing import Dict, List, Any
    
    # --- 示例：创建自定义链 ---
    print("\n--- 创建自定义链示例 ---")
    
    class SimpleTranslationChain(Chain):
        """简单的文本转换链，将输入文本转换为不同的格式
        (实际应用中可能会使用语言模型进行真正的翻译)"""
        
        input_text: str = ""
        llm: Any = None  # 在实际应用中会用到语言模型
        
        @property
        def input_keys(self) -> List[str]:
            """返回链的输入键"""
            return ["input_text"]
        
        @property
        def output_keys(self) -> List[str]:
            """返回链的输出键"""
            return ["uppercase", "lowercase", "reversed", "word_count"]
        
        def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
            """处理输入并返回输出"""
            # 获取输入文本
            input_text = inputs["input_text"]
            
            # 在实际应用中，这里可能会调用语言模型进行翻译
            # 现在我们只做一些简单的文本转换
            uppercase = input_text.upper()
            lowercase = input_text.lower()
            reversed_text = input_text[::-1]
            word_count = str(len(input_text.split()))
            
            return {
                "uppercase": uppercase,
                "lowercase": lowercase,
                "reversed": reversed_text,
                "word_count": word_count
            }
    
    # 创建自定义链实例
    custom_chain = SimpleTranslationChain()
    
    # 测试自定义链
    test_texts = [
        "Hello, LangChain!",
        "自然语言处理很有趣",
        "人工智能改变世界"
    ]
    
    # 运行自定义链
    print("运行自定义链...")
    for text in test_texts:
        print(f"\n原文: '{text}'")
        result = custom_chain.invoke({"input_text": text})
        print(f"大写: {result['uppercase']}")
        print(f"小写: {result['lowercase']}")
        print(f"倒序: {result['reversed']}")
        print(f"词数: {result['word_count']}")


# ===============================
# 主函数：运行所有示例
# ===============================
def main():
    """运行所有链组件示例"""
    print("# LangChain链(Chains)组件使用示例")
    print("=" * 80)
    
    print("\n## 重要提示")
    print("运行这些示例前，请确保已设置OpenAI API密钥。例如：")
    print("```python")
    print('import os')
    print('os.environ["OPENAI_API_KEY"] = "您的OpenAI API密钥"')
    print("```")
    print("或者，您可以使用.env文件存储这些密钥。")
    
    # 运行示例
    basic_llm_chains()
    complex_chain_types()
    router_chain_examples()
    transform_chain_examples()
    retrieval_qa_chain_examples()
    custom_chain_examples()
    
    print("\n" + "=" * 80)
    print("示例运行完成！")


if __name__ == "__main__":
    main()
