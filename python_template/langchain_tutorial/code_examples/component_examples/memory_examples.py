#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain记忆(Memory)组件使用示例

本模块展示了如何使用LangChain中的记忆组件来维护对话历史，包括：
1. 不同类型的记忆组件及其用法
2. 在链中使用记忆
3. 在智能体中集成记忆
4. 自定义记忆组件
5. 记忆的序列化和持久化

注意：运行这些示例前，请确保已设置相应的API密钥环境变量
"""

import os
import tempfile
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===============================
# 第1部分：基础记忆组件示例
# ===============================
def basic_memory_examples():
    """基础记忆组件示例"""
    print("=== 基础记忆组件示例 ===")
    
    from langchain.memory import ConversationBufferMemory
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.memory import ConversationSummaryMemory
    from langchain.schema import HumanMessage, AIMessage
    from langchain_openai import OpenAI, ChatOpenAI
    
    # --- 对话缓冲记忆 ---
    print("\n--- 对话缓冲记忆(ConversationBufferMemory)示例 ---")
    
    # 创建记忆组件
    buffer_memory = ConversationBufferMemory()
    
    # 添加对话消息到记忆中
    buffer_memory.chat_memory.add_user_message("你好！我叫张明。")
    buffer_memory.chat_memory.add_ai_message("你好张明！很高兴认识你。有什么我可以帮助你的吗？")
    buffer_memory.chat_memory.add_user_message("我想了解一下记忆组件是什么")
    buffer_memory.chat_memory.add_ai_message("记忆组件是LangChain中用于存储和管理对话历史的模块，它使AI能够在对话中保持上下文连贯性。")
    
    # 查看记忆内容
    print("记忆变量:")
    memory_variables = buffer_memory.load_memory_variables({})
    print(memory_variables["history"])
    
    # --- 对话缓冲窗口记忆 ---
    print("\n--- 对话缓冲窗口记忆(ConversationBufferWindowMemory)示例 ---")
    
    # 创建窗口记忆组件 (只保留最近k轮对话)
    window_memory = ConversationBufferWindowMemory(k=2)
    
    # 添加多轮对话（超出窗口大小）
    window_memory.save_context({"input": "你好！我叫李华"}, {"output": "你好李华！有什么我可以帮你的吗？"})
    window_memory.save_context({"input": "什么是机器学习？"}, {"output": "机器学习是人工智能的一个子领域，研究如何让计算机从数据中自动学习。"})
    window_memory.save_context({"input": "有哪些常见的机器学习算法？"}, {"output": "常见的机器学习算法包括线性回归、决策树、神经网络、支持向量机和随机森林等。"})
    window_memory.save_context({"input": "机器学习和深度学习有什么区别？"}, {"output": "深度学习是机器学习的一个子集，专注于使用多层神经网络从大量数据中学习复杂模式。"})
    
    # 查看窗口记忆内容 (只应该包含最近2轮对话)
    print("窗口记忆变量 (k=2):")
    window_variables = window_memory.load_memory_variables({})
    print(window_variables["history"])
    
    # --- 对话摘要记忆 ---
    print("\n--- 对话摘要记忆(ConversationSummaryMemory)示例 ---")
    
    # 检查是否有OpenAI API密钥
    if os.getenv("OPENAI_API_KEY"):
        # 创建摘要记忆组件 (需要LLM来生成摘要)
        llm = OpenAI(temperature=0)
        summary_memory = ConversationSummaryMemory(llm=llm)
        
        # 添加多轮对话
        summary_memory.save_context(
            {"input": "你好，我想规划一次去日本的旅行。"}, 
            {"output": "你好！很高兴帮你规划日本之旅。你大概计划什么时候去，会去多久呢？"}
        )
        summary_memory.save_context(
            {"input": "我计划在今年10月去，大约10天左右的行程。"}, 
            {"output": "十月去日本是个不错的选择，天气宜人。对于10天的行程，你可以考虑东京(3天)、京都(3天)、大阪(2天)，以及富士山或广岛(2天)。"}
        )
        summary_memory.save_context(
            {"input": "我特别喜欢历史文化，尤其想体验日本传统文化。"}, 
            {"output": "那么你应该重点关注京都，那里有许多历史悠久的寺庙和神社。可以考虑参观清水寺、伏见稻荷大社和金阁寺。在东京可以访问浅草寺和明治神宫。也许还可以安排一次和服体验或茶道课程。"}
        )
        
        # 查看摘要记忆内容
        print("摘要记忆变量:")
        summary_variables = summary_memory.load_memory_variables({})
        print(summary_variables["history"])
    else:
        print("未设置OPENAI_API_KEY，跳过摘要记忆示例")
        print("要运行此示例，请设置OPENAI_API_KEY环境变量")


# ===============================
# 第2部分：高级记忆组件示例
# ===============================
def advanced_memory_examples():
    """高级记忆组件示例"""
    print("\n=== 高级记忆组件示例 ===")
    
    from langchain.memory import ConversationEntityMemory
    from langchain.memory import VectorStoreRetrieverMemory
    from langchain_openai import OpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    # --- 实体记忆 ---
    print("\n--- 实体记忆(ConversationEntityMemory)示例 ---")
    
    # 检查是否有OpenAI API密钥
    if os.getenv("OPENAI_API_KEY"):
        # 创建实体记忆组件
        llm = OpenAI(temperature=0)
        entity_memory = ConversationEntityMemory(llm=llm)
        
        # 添加对话上下文
        entity_memory.save_context(
            {"input": "我的父亲叫王建国，他今年55岁，是一名中学教师。"}, 
            {"output": "了解了，你父亲王建国是55岁的中学教师。"}
        )
        entity_memory.save_context(
            {"input": "我的母亲李秀华是一名医生，在市第一人民医院工作。"}, 
            {"output": "明白了，你的母亲李秀华在市第一人民医院工作，是一名医生。"}
        )
        entity_memory.save_context(
            {"input": "我的妹妹王丽今年上大学，学习的是计算机科学专业。"}, 
            {"output": "懂了，你有一个妹妹叫王丽，正在大学学习计算机科学。"}
        )
        
        # 查看实体记忆内容
        print("实体记忆变量:")
        entity_variables = entity_memory.load_memory_variables({})
        print("已识别的实体:")
        for entity, info in entity_memory.entity_store.store.items():
            print(f"- {entity}: {info}")
        
        # 使用记忆进行新的对话
        print("\n使用实体记忆回答问题:")
        response = entity_memory.load_memory_variables({"input": "我父亲的工作是什么？"})
        print("关于'父亲'的记忆:")
        if 'entity_memories' in response and '父亲' in response['entity_memories']:
            print(response['entity_memories']['父亲'])
        else:
            print("未找到关于'父亲'的具体记忆")
    else:
        print("未设置OPENAI_API_KEY，跳过实体记忆示例")
        print("要运行此示例，请设置OPENAI_API_KEY环境变量")
    
    # --- 向量存储记忆 ---
    print("\n--- 向量存储记忆(VectorStoreRetrieverMemory)示例 ---")
    
    try:
        # 创建一个简单的向量存储
        print("创建向量存储...")
        
        # 尝试使用HuggingFace嵌入（无需API密钥）
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 准备一些示例记忆
        texts = [
            "客户名称：张伟，联系方式：1350000xxxx",
            "张伟对我们的产品A表示很满意，但对价格有些顾虑",
            "上次会议讨论了市场扩张计划，决定明年在西南地区开设三家新店",
            "李总表示财务报表需要在下周一前完成审核",
            "新产品发布会定于12月15日举行，地点是国际会展中心"
        ]
        
        # 创建向量存储
        vector_store = FAISS.from_texts(texts, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        
        # 创建向量存储记忆
        vector_memory = VectorStoreRetrieverMemory(retriever=retriever)
        
        # 添加新记忆
        vector_memory.save_context(
            {"input": "我们的新产品定价是多少？"}, 
            {"output": "我们的新产品A的零售价是1299元，批发价格可以根据数量进行协商。"}
        )
        
        # 查询相关记忆
        print("查询'张伟客户'相关记忆:")
        zhang_memory = vector_memory.load_memory_variables({"prompt": "张伟客户的情况"})
        print(zhang_memory["history"])
        
        print("\n查询'新产品'相关记忆:")
        product_memory = vector_memory.load_memory_variables({"prompt": "新产品相关信息"})
        print(product_memory["history"])
    
    except Exception as e:
        print(f"运行向量存储记忆示例时出错: {e}")
        print("请确保已安装相关依赖: pip install faiss-cpu sentence-transformers")


# ===============================
# 第3部分：在链中使用记忆
# ===============================
def memory_in_chains_examples():
    """在链中使用记忆的示例"""
    print("\n=== 在链中使用记忆的示例 ===")
    
    from langchain.chains import ConversationChain
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain.memory import ConversationSummaryMemory
    from langchain_openai import OpenAI, ChatOpenAI
    from langchain.prompts import PromptTemplate
    
    # 检查是否有OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("未设置OPENAI_API_KEY，跳过本节示例")
        print("要运行此示例，请设置OPENAI_API_KEY环境变量")
        return
    
    # --- 基础对话链 ---
    print("\n--- 基础对话链示例 ---")
    
    # 创建记忆组件
    memory = ConversationBufferMemory()
    
    # 创建语言模型
    llm = OpenAI(temperature=0.7)
    
    # 创建对话链
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # 模拟对话
    print("\n对话开始:")
    response1 = conversation.predict(input="你好！我叫王小明。")
    print(f"AI: {response1}")
    
    response2 = conversation.predict(input="你还记得我的名字吗？")
    print(f"AI: {response2}")
    
    # 检查记忆状态
    print("\n当前记忆内容:")
    print(memory.load_memory_variables({})["history"])
    
    # --- 自定义提示模板的对话链 ---
    print("\n--- 自定义提示模板的对话链示例 ---")
    
    # 创建自定义提示模板
    template = """你是一个学科辅导老师，帮助学生回答问题。
    
    当前对话历史:
    {chat_history}
    
    学生: {question}
    老师:"""
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template
    )
    
    # 创建特定记忆键的记忆组件
    tutor_memory = ConversationBufferMemory(memory_key="chat_history")
    
    # 创建链
    tutor_chain = LLMChain(
        llm=ChatOpenAI(temperature=0),
        prompt=prompt,
        memory=tutor_memory,
        verbose=True
    )
    
    # 模拟学习对话
    print("\n辅导对话开始:")
    tutor_response1 = tutor_chain.predict(question="光合作用的主要过程是什么？")
    print(f"老师: {tutor_response1}")
    
    tutor_response2 = tutor_chain.predict(question="这个过程产生了哪些物质？")
    print(f"老师: {tutor_response2}")
    
    # --- 使用摘要记忆的长对话链 ---
    print("\n--- 使用摘要记忆的长对话链示例 ---")
    
    # 创建摘要记忆
    summary_memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
    
    # 创建带摘要记忆的对话链
    summary_conversation = ConversationChain(
        llm=llm,
        memory=summary_memory,
        verbose=True
    )
    
    # 模拟多轮对话
    print("\n对话开始:")
    
    responses = [
        summary_conversation.predict(input="我想创建一个学习应用，帮助学生提高学习效率。"),
        summary_conversation.predict(input="应用应该包含哪些核心功能？"),
        summary_conversation.predict(input="学习计划功能如何帮助学生安排时间？"),
        summary_conversation.predict(input="我需要考虑数据隐私问题吗？")
    ]
    
    for i, response in enumerate(responses):
        print(f"对话轮次 {i+1} - AI: {response[:50]}...")
    
    # 显示最终摘要
    print("\n对话摘要:")
    print(summary_memory.load_memory_variables({})['history'])


# ===============================
# 第4部分：自定义记忆组件
# ===============================
def custom_memory_examples():
    """自定义记忆组件示例"""
    print("\n=== 自定义记忆组件示例 ===")
    
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema import BaseMessage
    from datetime import datetime
    
    class TimestampedMemory(BaseChatMemory):
        """带时间戳的记忆组件"""
        
        # 内部存储时间戳信息
        timestamps: List[str] = []
        
        def _load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """加载记忆变量"""
            # 获取所有消息
            messages = self.chat_memory.messages
            
            # 组合消息和时间戳
            timestamped_history = []
            for i, message in enumerate(messages):
                if i < len(self.timestamps):
                    time = self.timestamps[i]
                    # 根据消息类型添加标签
                    if "human" in message.type.lower():
                        timestamped_history.append(f"[{time}] 用户: {message.content}")
                    else:
                        timestamped_history.append(f"[{time}] AI: {message.content}")
            
            # 返回格式化的历史记录
            history = "\n".join(timestamped_history)
            return {"history": history}
        
        def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
            """保存对话上下文和时间戳"""
            input_str = inputs[self.input_key]
            output_str = outputs[self.output_key]
            
            # 添加当前时间戳
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.timestamps.append(current_time)
            self.timestamps.append(current_time)
            
            # 保存消息到聊天记忆
            self.chat_memory.add_user_message(input_str)
            self.chat_memory.add_ai_message(output_str)
    
    # --- 使用自定义记忆组件 ---
    print("\n--- 使用自定义记忆组件 ---")
    
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_openai import OpenAI
    
    # 检查是否有OpenAI API密钥
    if os.getenv("OPENAI_API_KEY"):
        # 创建自定义记忆实例
        timestamped_memory = TimestampedMemory(input_key="input", output_key="output")
        
        # 创建提示模板
        template = """你是一个友好的AI助手。
        
        对话历史:
        {history}
        
        人类: {input}
        AI:"""
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        # 创建链
        llm = OpenAI(temperature=0.7)
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=timestamped_memory,
            verbose=True
        )
        
        # 模拟对话
        print("\n带时间戳的对话:")
        
        # 模拟延迟，使时间戳有差异
        import time
        
        response1 = chain.predict(input="你好，今天天气怎么样？")
        print(f"AI: {response1}")
        
        # 延迟几秒
        time.sleep(2)
        
        response2 = chain.predict(input="你能推荐一些周末活动吗？")
        print(f"AI: {response2}")
        
        # 延迟几秒
        time.sleep(2)
        
        response3 = chain.predict(input="这些活动适合和家人一起吗？")
        print(f"AI: {response3}")
        
        # 显示带时间戳的历史
        print("\n带时间戳的对话历史:")
        print(timestamped_memory.load_memory_variables({})['history'])
    else:
        print("未设置OPENAI_API_KEY，跳过本节示例")
        print("要运行此示例，请设置OPENAI_API_KEY环境变量")


# ===============================
# 第5部分：记忆的序列化和持久化
# ===============================
def memory_serialization_examples():
    """记忆序列化和持久化示例"""
    print("\n=== 记忆序列化和持久化示例 ===")
    
    import json
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import HumanMessage, AIMessage
    
    # --- 基础记忆序列化 ---
    print("\n--- 基础记忆序列化 ---")
    
    # 创建记忆实例
    memory = ConversationBufferMemory()
    
    # 添加一些对话
    memory.chat_memory.add_user_message("你好，我想学习编程。")
    memory.chat_memory.add_ai_message("你好！编程是一个很好的选择。你有什么特定的编程语言或领域感兴趣吗？")
    memory.chat_memory.add_user_message("我对Python很感兴趣，听说它对初学者友好。")
    memory.chat_memory.add_ai_message("是的，Python是一个非常好的入门语言！它语法简洁清晰，有丰富的库和社区支持。你想从哪方面开始学习呢？")
    
    # 序列化记忆
    print("序列化记忆到JSON...")
    serialized_messages = []
    
    for message in memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            serialized_messages.append({"type": "human", "content": message.content})
        elif isinstance(message, AIMessage):
            serialized_messages.append({"type": "ai", "content": message.content})
    
    # 将序列化的消息写入文件
    temp_file = os.path.join(tempfile.mkdtemp(), "memory.json")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(serialized_messages, f, ensure_ascii=False, indent=2)
    
    print(f"记忆已保存到: {temp_file}")
    
    # --- 记忆反序列化 ---
    print("\n--- 记忆反序列化 ---")
    
    # 从文件加载序列化的消息
    print("从JSON加载记忆...")
    with open(temp_file, "r", encoding="utf-8") as f:
        loaded_messages = json.load(f)
    
    # 创建新的记忆实例
    new_memory = ConversationBufferMemory()
    
    # 将加载的消息添加到新记忆中
    for msg in loaded_messages:
        if msg["type"] == "human":
            new_memory.chat_memory.add_user_message(msg["content"])
        elif msg["type"] == "ai":
            new_memory.chat_memory.add_ai_message(msg["content"])
    
    # 验证加载的记忆
    loaded_history = new_memory.load_memory_variables({})['history']
    print("加载的记忆内容:")
    print(loaded_history)
    
    # --- 在链中使用持久化记忆 ---
    print("\n--- 在链中使用持久化记忆示例 ---")
    print("此示例展示了如何在不同会话之间保持对话上下文")
    
    # 检查是否有OpenAI API密钥
    if os.getenv("OPENAI_API_KEY"):
        from langchain.chains import ConversationChain
        from langchain_openai import OpenAI
        
        # 创建对话链，使用从文件加载的记忆
        llm = OpenAI(temperature=0.7)
        conversation = ConversationChain(
            llm=llm,
            memory=new_memory,
            verbose=True
        )
        
        # 继续之前的对话
        print("\n继续之前保存的对话:")
        response = conversation.predict(input="有没有推荐的Python学习资源？")
        print(f"AI: {response}")
    else:
        print("未设置OPENAI_API_KEY，跳过模型调用部分")


# ===============================
# 主函数：运行所有示例
# ===============================
def main():
    """运行所有记忆组件示例"""
    print("# LangChain记忆(Memory)组件使用示例")
    print("=" * 80)
    
    print("\n## 重要提示")
    print("运行这些示例前，请确保已安装所需依赖:")
    print("pip install langchain langchain-openai faiss-cpu sentence-transformers")
    print("对于部分功能，需要设置OpenAI API密钥:")
    print("```python")
    print('import os')
    print('os.environ["OPENAI_API_KEY"] = "您的OpenAI API密钥"')
    print("```")
    
    # 运行示例
    basic_memory_examples()
    advanced_memory_examples()
    memory_in_chains_examples()
    custom_memory_examples()
    memory_serialization_examples()
    
    print("\n" + "=" * 80)
    print("示例运行完成！请根据上述演示进行您自己的记忆组件应用开发。")


if __name__ == "__main__":
    main()