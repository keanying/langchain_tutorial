#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain语言模型(LLMs)和聊天模型(Chat Models)使用示例

本模块展示了如何使用LangChain中的各种语言模型，包括：
1. 传统LLM模型
2. 聊天模型(Chat Models)
3. 模型输出解析和处理
4. 不同模型提供商的集成

注意：运行这些示例前，请确保已设置相应的API密钥环境变量
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# 加载环境变量
load_dotenv()

# ===============================
# 第1部分：基础LLM模型使用示例
# ===============================
def basic_llm_examples():
    """基本LLM模型使用示例"""
    print("=== 基本LLM模型使用示例 ===")
    
    # --- 示例1：使用OpenAI模型 ---
    from langchain_openai import OpenAI
    
    # 创建一个OpenAI模型实例
    llm = OpenAI(
        model="gpt-3.5-turbo-instruct",  # 指定模型，这里使用支持指令的GPT-3.5 Turbo
        temperature=0.7,                 # 温度参数：控制随机性，越高越随机
        max_tokens=256,                  # 最大生成令牌数
        top_p=1,                         # Top-p采样
        frequency_penalty=0,             # 频率惩罚
        presence_penalty=0               # 存在惩罚
    )
    
    # 调用模型进行预测
    print("\n--- OpenAI LLM 示例 ---")
    prompt = "解释什么是人工智能，用简单的术语表述，不超过3句话"
    result = llm.invoke(prompt)
    print(f"提示: {prompt}")
    print(f"回答: {result}")
    
    # --- 示例2：使用Anthropic Claude模型 ---
    try:
        from langchain_anthropic import Anthropic
        
        # 创建一个Anthropic模型实例
        claude_llm = Anthropic(
            model="claude-2",  # 指定Claude模型
            temperature=0.5,
            max_tokens_to_sample=256
        )
        
        # 调用模型进行预测
        print("\n--- Anthropic Claude LLM 示例 ---")
        prompt = "解释量子计算的基本原理，用简单的术语表述，不超过3句话"
        result = claude_llm.invoke(prompt)
        print(f"提示: {prompt}")
        print(f"回答: {result}")
    except Exception as e:
        print(f"注意: 无法运行Anthropic示例，请确认API密钥已正确设置: {e}")
    
    # --- 示例3：使用多个模型进行对比 ---
    try:
        from langchain_cohere import Cohere
        
        # 创建Cohere模型实例
        cohere_llm = Cohere(
            model="command",  # Cohere的模型
            temperature=0.7,
            max_tokens=256
        )
        
        # 创建提示
        prompt = "解释区块链技术的工作原理，简单表述"
        
        # 对比不同模型的输出
        print("\n--- 模型对比示例 ---")
        print(f"提示: {prompt}")
        print("OpenAI回答:", llm.invoke(prompt))
        print("Cohere回答:", cohere_llm.invoke(prompt))
    except Exception as e:
        print(f"注意: 无法运行Cohere示例，请确认API密钥已正确设置: {e}")
    
    # --- 示例4：在流模式下使用LLM ---
    print("\n--- 流式输出示例 ---")
    prompt = "写一个简短的故事，关于一只学习编程的猫"
    print(f"提示: {prompt}")
    print("流式输出:")
    
    # 使用流式输出模式
    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)
    print("\n")
    
    # --- 示例5：使用HuggingFace模型 ---
    try:
        from langchain_huggingface import HuggingFaceEndpoint
        
        # 创建HuggingFace模型实例
        hf_llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
            max_new_tokens=512,
            temperature=0.7
        )
        
        # 调用模型
        print("\n--- HuggingFace Mistral 示例 ---")
        prompt = "解释神经网络是如何工作的，用简单的术语"
        print(f"提示: {prompt}")
        result = hf_llm.invoke(prompt)
        print(f"回答: {result}")
    except Exception as e:
        print(f"注意: 无法运行HuggingFace示例，请确认API密钥已正确设置或服务可用: {e}")


# ===============================
# 第2部分：聊天模型(Chat Models)使用示例
# ===============================
def chat_model_examples():
    """聊天模型使用示例"""
    print("\n=== 聊天模型使用示例 ===")
    
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    
    # --- 示例1：使用OpenAI的聊天模型 ---
    from langchain_openai import ChatOpenAI
    
    # 创建一个ChatOpenAI模型实例
    chat_model = ChatOpenAI(
        model="gpt-3.5-turbo",  # 指定聊天模型
        temperature=0.7,
        max_tokens=256
    )
    
    # 准备消息列表
    messages = [
        SystemMessage(content="你是一位友好的AI助手，专注于解释复杂的概念。"),
        HumanMessage(content="请解释机器学习和深度学习的区别。")
    ]
    
    # 调用模型进行预测
    print("\n--- OpenAI Chat Model 示例 ---")
    print("消息列表:")
    for msg in messages:
        print(f"- {msg.type}: {msg.content}")
    
    response = chat_model.invoke(messages)
    print(f"\n回答: {response.content}")
    
    # --- 示例2：模拟多轮对话 ---
    print("\n--- 多轮对话示例 ---")
    
    # 扩展消息列表，添加之前的回答和新问题
    messages.append(AIMessage(content=response.content))
    messages.append(HumanMessage(content="谢谢说明！那什么是卷积神经网络？"))
    
    print("更新后的消息列表:")
    for msg in messages:
        print(f"- {msg.type}: {msg.content[:40]}...")
    
    # 调用模型进行预测
    response = chat_model.invoke(messages)
    print(f"\n回答: {response.content}")
    
    # --- 示例3：使用Anthropic Claude聊天模型 ---
    try:
        from langchain_anthropic import ChatAnthropic
        
        # 创建一个ChatAnthropic模型实例
        claude_chat = ChatAnthropic(
            model="claude-2",  # 指定Claude模型
            temperature=0.5,
            max_tokens=256
        )
        
        # 准备消息列表
        messages = [
            SystemMessage(content="你是一位专业的科学讲解员，以通俗易懂的方式解释复杂的科学概念。"),
            HumanMessage(content="请解释相对论的基本原理。")
        ]
        
        # 调用模型进行预测
        print("\n--- Anthropic Claude 聊天模型示例 ---")
        print("消息列表:")
        for msg in messages:
            print(f"- {msg.type}: {msg.content}")
        
        response = claude_chat.invoke(messages)
        print(f"\n回答: {response.content}")
    except Exception as e:
        print(f"注意: 无法运行Anthropic聊天示例，请确认API密钥已正确设置: {e}")
    
    # --- 示例4：使用流式模式的聊天模型 ---
    print("\n--- 流式聊天模型示例 ---")
    
    messages = [
        SystemMessage(content="你是一位专业的故事创作者。"),
        HumanMessage(content="请创作一个短篇童话故事，主题是'友谊'。")
    ]
    
    print("消息列表:")
    for msg in messages:
        print(f"- {msg.type}: {msg.content}")
    print("\n流式输出:")
    
    for chunk in chat_model.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")


# ===============================
# 第3部分：模型函数调用(Function Calling)示例
# ===============================
def function_calling_examples():
    """模型函数调用示例"""
    print("\n=== 模型函数调用示例 ===")
    
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    
    # --- 示例1：定义工具和函数调用 ---
    print("\n--- 基本函数调用示例 ---")
    
    # 定义工具函数
    @tool
    def get_weather(location: str, unit: str = "celsius") -> str:
        """获取指定位置的当前天气。
        
        Args:
            location: 城市名称，如"北京"、"上海"等
            unit: 温度单位，可选择"celsius"（摄氏度）或"fahrenheit"（华氏度）
            
        Returns:
            包含天气信息的字符串
        """
        # 这里模拟天气数据，实际应用中应调用真实的天气API
        weather_data = {
            "北京": {"温度": 20, "天气状况": "晴朗"},
            "上海": {"温度": 25, "天气状况": "多云"},
            "广州": {"温度": 30, "天气状况": "小雨"},
        }
        
        if location in weather_data:
            temp = weather_data[location]["温度"]
            if unit == "fahrenheit":
                temp = temp * 9/5 + 32
            return f"{location}的天气状况：{weather_data[location]['天气状况']}，温度：{temp}度"
        else:
            return f"抱歉，没有找到{location}的天气数据。"
    
    # 创建支持函数调用的聊天模型
    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",  # 使用支持函数调用的模型
        temperature=0
    )
    
    # 通过模型选择工具
    message = HumanMessage(content="今天北京的天气怎么样？")
    result = model.invoke(
        [message],
        tools=[get_weather]  # 提供可用工具
    )
    
    print(f"用户问题: {message.content}")
    print(f"模型回应: {result}")
    
    # 工具调用示例
    from langchain_core.tools import StructuredTool
    from langchain_core.messages import AIMessage, ToolMessage
    
    # 定义一个计算器工具
    def calculator(operation: str, x: float, y: float) -> float:
        """执行简单的数学运算
        
        Args:
            operation: 操作类型，可选"add"（加）、"subtract"（减）、"multiply"（乘）或"divide"（除）
            x: 第一个数值
            y: 第二个数值
            
        Returns:
            计算结果
        """
        if operation == "add":
            return x + y
        elif operation == "subtract":
            return x - y
        elif operation == "multiply":
            return x * y
        elif operation == "divide":
            if y == 0:
                raise ValueError("除数不能为零")
            return x / y
        else:
            raise ValueError(f"不支持的操作: {operation}")
    
    # 创建结构化工具
    calc_tool = StructuredTool.from_function(calculator)
    
    # --- 示例2：处理多轮对话中的函数调用 ---
    print("\n--- 多轮对话中的函数调用示例 ---")
    
    # 初始化对话
    messages = [
        HumanMessage(content="计算72乘以15是多少，然后再减去394")
    ]
    
    print(f"用户问题: {messages[0].content}")
    
    # 第一轮：模型决定先计算乘法
    response = model.invoke(messages, tools=[calc_tool])
    print(f"模型思考: {response.content}")
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"工具调用: {tool_call.name}({tool_call.args})")
        
        # 执行工具调用
        if tool_call.name == "calculator":
            result = calculator(**tool_call.args)
            messages.append(response)
            messages.append(ToolMessage(content=str(result), name="calculator"))
            
            print(f"工具结果: {result}")
            
            # 第二轮：使用乘法结果进行减法
            response = model.invoke(messages, tools=[calc_tool])
            print(f"模型思考: {response.content}")
            
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                print(f"工具调用: {tool_call.name}({tool_call.args})")
                
                # 执行第二次工具调用
                result = calculator(**tool_call.args)
                messages.append(response)
                messages.append(ToolMessage(content=str(result), name="calculator"))
                
                print(f"工具结果: {result}")
                
                # 获取最终回答
                final_response = model.invoke(messages)
                print(f"最终回答: {final_response.content}")


# ===============================
# 第4部分：本地模型集成示例
# ===============================
def local_model_examples():
    """本地运行的模型集成示例"""
    print("\n=== 本地模型集成示例 ===")
    
    # 注意：这些示例需要安装额外的依赖并下载相应的模型
    
    try:
        from langchain_community.llms import Ollama
        
        # 使用Ollama运行本地模型
        print("\n--- Ollama本地模型示例 ---")
        print("注意：此示例需要在本地运行Ollama服务")
        
        # 假设已经在本地启动了Ollama服务
        ollama_llm = Ollama(model="llama2")
        
        prompt = "解释什么是人工智能"
        print(f"提示: {prompt}")
        
        try:
            result = ollama_llm.invoke(prompt)
            print(f"回答: {result}")
        except Exception as e:
            print(f"运行Ollama模型错误: {e}")
            print("请确保已正确安装并启动Ollama服务: https://github.com/ollama/ollama")
    except ImportError:
        print("无法导入Ollama。要使用本地模型，请安装必要的依赖：pip install langchain-community")
    
    # 使用通过Hugging Face运行的本地模型
    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline
        
        print("\n--- Hugging Face本地管道模型示例 ---")
        print("注意：此示例需要下载模型，可能需要几分钟时间")
        
        try:
            # 创建文本生成管道
            pipe = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                max_new_tokens=128,
            )
            
            # 创建LangChain模型
            hf_pipeline = HuggingFacePipeline(pipeline=pipe)
            
            prompt = "解释量子计算基本原理"
            print(f"提示: {prompt}")
            
            result = hf_pipeline.invoke(prompt)
            print(f"回答: {result}")
        except Exception as e:
            print(f"运行Hugging Face模型错误: {e}")
            print("请确保已安装transformers库并下载了相应的模型")
    except ImportError:
        print("无法导入所需依赖。要使用Hugging Face模型，请安装必要的依赖：pip install langchain_community transformers")


# ===============================
# 第5部分：模型输出解析示例
# ===============================
def output_parsing_examples():
    """模型输出解析示例"""
    print("\n=== 模型输出解析示例 ===")
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_core.output_parsers import JsonOutputParser
    
    # --- 示例1：使用输出解析器解析JSON格式输出 ---
    print("\n--- JSON输出解析示例 ---")
    
    # 定义输出格式
    class Movie(BaseModel):
        title: str = Field(description="电影标题")
        director: str = Field(description="导演姓名")
        year: int = Field(description="发行年份")
        genres: List[str] = Field(description="电影类型列表")
        plot_summary: str = Field(description="剧情概要")
        rating: float = Field(description="评分（满分10分）")
    
    # 创建解析器
    parser = JsonOutputParser(pydantic_object=Movie)
    
    # 创建模型和提示模板
    model = ChatOpenAI(temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个电影数据库专家。请根据用户的电影名称，提供详细信息。"),
        ("user", "{input}"),
        ("system", "请将信息以JSON格式返回，包含以下字段：标题、导演、年份、类型、剧情概要和评分。")
    ])
    
    # 创建链
    chain = prompt | model | parser
    
    # 调用链
    result = chain.invoke({"input": "分析电影《盗梦空间》"})
    print("输入: 分析电影《盗梦空间》")
    print(f"解析后的JSON响应:\n{result}")
    
    # --- 示例2：使用结构化输出格式 ---
    print("\n--- Pydantic输出解析示例 ---")
    
    from langchain_core.output_parsers import PydanticOutputParser
    
    class Person(BaseModel):
        name: str = Field(description="人名")
        age: int = Field(description="年龄")
        hobbies: List[str] = Field(description="爱好列表")
        brief_bio: str = Field(description="简短的个人介绍")
    
    # 创建Pydantic解析器
    pydantic_parser = PydanticOutputParser(pydantic_object=Person)
    
    # 创建提示
    format_instructions = pydantic_parser.get_format_instructions()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个个人资料生成器。请为用户创建一个虚构的个人资料。"),
        ("system", "请按照以下格式返回：\n{format_instructions}"),
        ("user", "创建一个热爱科技的人的个人资料")
    ])
    
    # 创建链
    chain = prompt_template.partial(format_instructions=format_instructions) | model | pydantic_parser
    
    # 调用链
    result = chain.invoke({})
    
    print("输入: 创建一个热爱科技的人的个人资料")
    print(f"解析后的Pydantic对象:\n")
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"爱好: {', '.join(result.hobbies)}")
    print(f"简介: {result.brief_bio}")
    
    # --- 示例3：解析列表输出 ---
    print("\n--- 列表输出解析示例 ---")
    
    from langchain_core.output_parsers import CommaSeparatedListOutputParser
    
    # 创建列表解析器
    list_parser = CommaSeparatedListOutputParser()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个创意建议生成器。"),
        ("system", "请生成一个逗号分隔的列表作为响应。"),
        ("user", "{input}")
    ])
    
    # 创建链
    chain = prompt_template | model | list_parser
    
    # 调用链
    result = chain.invoke({"input": "给我列出10个周末可以进行的有趣活动"})
    
    print("输入: 给我列出10个周末可以进行的有趣活动")
    print("解析后的列表:")
    for i, item in enumerate(result, 1):
        print(f"{i}. {item}")


# ===============================
# 第6部分：模型缓存和批处理示例
# ===============================
def model_caching_and_batch_examples():
    """模型缓存和批处理示例"""
    print("\n=== 模型缓存和批处理示例 ===")
    
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langchain_community.cache import InMemoryCache, SQLiteCache
    from langchain.globals import set_llm_cache
    import time
    
    # --- 示例1：使用内存缓存 ---
    print("\n--- 内存缓存示例 ---")
    
    # 设置内存缓存
    set_llm_cache(InMemoryCache())
    
    # 创建模型
    model = ChatOpenAI(temperature=0)
    
    # 首次调用（缓存未命中）
    prompt = "什么是机器学习？简短回答"
    print(f"提示: {prompt}")
    
    start_time = time.time()
    response = model.invoke([HumanMessage(content=prompt)])
    first_call_time = time.time() - start_time
    
    print(f"首次调用 (缓存未命中) - 耗时: {first_call_time:.2f}秒")
    print(f"回答: {response.content[:100]}...\n")
    
    # 再次调用（缓存命中）
    start_time = time.time()
    response = model.invoke([HumanMessage(content=prompt)])
    second_call_time = time.time() - start_time
    
    print(f"再次调用 (缓存命中) - 耗时: {second_call_time:.2f}秒")
    print(f"回答: {response.content[:100]}...\n")
    
    print(f"速度提升: {first_call_time / second_call_time:.2f}x")
    
    # --- 示例2：批量处理 ---
    print("\n--- 批量处理示例 ---")
    
    # 准备多个提示
    prompts = [
        "解释电子邮件的工作原理。",
        "解释互联网的工作原理。",
        "解释云计算的工作原理。"
    ]
    
    messages = [[HumanMessage(content=prompt)] for prompt in prompts]
    
    print("批量提交以下提示:")
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")
    
    # 批量生成
    start_time = time.time()
    results = model.batch(messages)
    end_time = time.time()
    
    print(f"\n批量处理耗时: {end_time - start_time:.2f}秒")
    
    # 打印结果
    print("\n批量处理结果:")
    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"\n{i}. 提示: {prompt}")
        print(f"   回答: {result.content[:100]}...")


# ===============================
# 主函数：运行所有示例
# ===============================
def main():
    """运行所有LLM和聊天模型示例"""
    print("# LangChain语言模型(LLMs)和聊天模型(Chat Models)使用示例")
    print("=" * 80)
    
    print("\n## 重要提示")
    print("运行这些示例前，请确保已设置API密钥。例如，对于OpenAI模型：")
    print("```python")
    print('import os')
    print('os.environ["OPENAI_API_KEY"] = "您的OpenAI API密钥"')
    print("```")
    print("或者，您可以使用.env文件存储这些密钥。")
    
    # 运行示例
    basic_llm_examples()
    chat_model_examples()
    function_calling_examples()
    output_parsing_examples()
    model_caching_and_batch_examples()
    
    # 如果需要，可以取消注释以下行来运行本地模型示例
    # local_model_examples()
    
    print("\n" + "=" * 80)
    print("示例运行完成！")


if __name__ == "__main__":
    main()