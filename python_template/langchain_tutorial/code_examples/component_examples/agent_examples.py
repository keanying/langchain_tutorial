#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain智能体(Agents)组件使用示例

本模块展示了如何使用LangChain中的智能体组件来创建能够规划和执行任务的自主代理，包括：
1. 基础智能体类型
2. 工具的创建与使用
3. 自定义智能体
4. 多智能体系统
5. 智能体对话和记忆

注意：运行这些示例前，请确保已设置相应的API密钥环境变量
"""

import os
from typing import List, Dict, Any, Tuple, Optional, Union
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===============================
# 第1部分：基础智能体类型
# ===============================
def basic_agent_examples():
    """基础智能体类型示例"""
    print("=== 基础智能体类型示例 ===")
    
    from langchain.agents import load_tools
    from langchain.agents import initialize_agent
    from langchain.agents import AgentType
    from langchain_openai import OpenAI
    
    # --- 示例1：零次提示(Zero-shot)智能体 ---
    print("\n--- 零次提示(Zero-shot)智能体示例 ---")
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 初始化语言模型
            llm = OpenAI(temperature=0)
            
            # 加载工具
            tools = load_tools(["llm-math"], llm=llm)
            
            # 初始化智能体
            agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            
            # 示例任务
            print("运行零次提示智能体...")
            print("问题: 计算(123 * 456)的平方根是多少？")
            result = agent.invoke("计算(123 * 456)的平方根是多少？")
            print(f"回答: {result['output']}\n")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("零次提示智能体可以在没有任何示例的情况下，根据工具的描述决定使用哪些工具来解决问题。")
            print("它会先思考该使用什么工具，然后使用工具，最后根据工具的输出给出答案。")
    
    except Exception as e:
        print(f"运行零次提示智能体时出错: {e}")
    
    # --- 示例2：对话式智能体 ---
    print("\n--- 对话式智能体示例 ---")
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 初始化语言模型
            from langchain_openai import ChatOpenAI
            chat_model = ChatOpenAI(temperature=0)
            
            # 加载工具
            from langchain.tools import DuckDuckGoSearchRun
            search_tool = DuckDuckGoSearchRun()
            
            tools = [search_tool]
            
            # 初始化对话式智能体
            from langchain.memory import ConversationBufferMemory
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            conversational_agent = initialize_agent(
                tools,
                chat_model,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True
            )
            
            # 示例对话
            print("运行对话式智能体...")
            print("用户问题: 谁是当前的中国国家主席？")
            result = conversational_agent.invoke({"input": "谁是当前的中国国家主席？"})
            print(f"智能体回答: {result['output']}\n")
            
            print("用户问题: 他从什么时候开始担任这个职务的？")
            result = conversational_agent.invoke({"input": "他从什么时候开始担任这个职务的？"})
            print(f"智能体回答: {result['output']}\n")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("对话式智能体可以维护对话历史，理解上下文中的指代，能够进行连贯的多轮对话。")
            print("它结合了对话记忆和工具使用能力，适合构建交互式助手。")
    
    except Exception as e:
        print(f"运行对话式智能体时出错: {e}")

    # --- 示例3：基于工具使用标准的智能体 ---
    print("\n--- 基于工具使用标准的智能体示例 ---")
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 加载支持OpenAI函数调用的模型
            from langchain_openai import ChatOpenAI
            function_calling_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            
            # 加载工具
            tools = load_tools(["llm-math"], llm=OpenAI(temperature=0))
            
            # 初始化智能体
            from langchain.agents import create_openai_functions_agent
            from langchain.prompts import ChatPromptTemplate
            from langchain.agents import AgentExecutor
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个有用的AI助手，可以访问多种工具来解决问题。"),
                ("human", "{input}")
            ])
            
            agent = create_openai_functions_agent(function_calling_llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            # 示例任务
            print("运行基于工具使用标准的智能体...")
            print("问题: 15的平方是多少？")
            result = agent_executor.invoke({"input": "15的平方是多少？"})
            print(f"回答: {result['output']}\n")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("基于工具使用标准的智能体利用了OpenAI函数调用API，更精确地选择和使用工具。")
            print("它按照一定的标准格式使用工具，减少了解析错误，提高了可靠性。")
    
    except Exception as e:
        print(f"运行基于工具使用标准的智能体时出错: {e}")


# ===============================
# 第2部分：工具的创建与使用
# ===============================
def tools_examples():
    """工具的创建与使用示例"""
    print("\n=== 工具的创建与使用示例 ===")
    
    # --- 示例1：内置工具 ---
    print("\n--- 内置工具示例 ---")
    
    from langchain.tools import DuckDuckGoSearchRun
    from langchain.tools import WikipediaQueryRun
    from langchain.utilities import WikipediaAPIWrapper
    from langchain.agents import load_tools
    from langchain_openai import OpenAI
    
    # 介绍一些常用的内置工具
    print("LangChain提供了多种内置工具，例如：")
    print("1. 搜索工具：DuckDuckGoSearchRun、GoogleSearchAPIWrapper等")
    print("2. 计算工具：LLMMathChain")
    print("3. 知识库工具：WikipediaQueryRun")
    print("4. API工具：RequestsToolkit、APIOperation等")
    print("5. 终端工具：TerminalTool")
    print("6. 代码工具：PythonREPLTool")
    
    try:
        # 使用搜索工具示例
        print("\n使用DuckDuckGo搜索工具示例：")
        search_tool = DuckDuckGoSearchRun()
        print("工具名称:", search_tool.name)
        print("工具描述:", search_tool.description)
        
        # 使用维基百科工具
        print("\n使用维基百科查询工具示例：")
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        print("工具名称:", wikipedia_tool.name)
        print("工具描述:", wikipedia_tool.description)
        
        # 加载多个工具
        print("\n加载多个内置工具：")
        if os.getenv("OPENAI_API_KEY"):
            llm = OpenAI(temperature=0)
            tools = load_tools(["llm-math", "wikipedia"], llm=llm)
            print(f"成功加载了 {len(tools)} 个工具")
            for tool in tools:
                print(f"- {tool.name}: {tool.description[:50]}...")
        else:
            print("未设置OPENAI_API_KEY，跳过加载需要LLM的工具")
    
    except Exception as e:
        print(f"使用内置工具时出错: {e}")

    # --- 示例2：自定义工具 ---
    print("\n--- 自定义工具示例 ---")
    
    from langchain.tools import BaseTool
    from langchain.agents import AgentType, initialize_agent
    from pydantic import BaseModel, Field
    from typing import Type
    
    # 定义一个简单的自定义工具
    class WeatherInput(BaseModel):
        """输入参数定义"""
        location: str = Field(description="需要查询天气的城市名称")
    
    class WeatherTool(BaseTool):
        """模拟获取城市天气的工具"""
        name = "get_weather"
        description = "当用户询问天气情况时使用此工具。输入应该是一个城市名称。"
        args_schema: Type[BaseModel] = WeatherInput
        
        def _run(self, location: str) -> str:
            """模拟天气查询"""
            # 实际应用中，这里应调用真实的天气API
            weather_data = {
                "北京": "晴天，温度25°C",
                "上海": "多云，温度28°C",
                "广州": "雨天，温度30°C",
                "深圳": "阴天，温度27°C"
            }
            
            return weather_data.get(location, f"无法获取{location}的天气信息")
        
        def _arun(self, location: str) -> str:
            """异步运行（这里简单地调用同步版本）"""
            return self._run(location)
    
    # 创建自定义工具实例
    weather_tool = WeatherTool()
    
    # 使用自定义工具
    print("使用自定义天气工具：")
    print(f"工具名称: {weather_tool.name}")
    print(f"工具描述: {weather_tool.description}")
    print(f"查询北京天气: {weather_tool.run('北京')}")
    print(f"查询上海天气: {weather_tool.run('上海')}")
    
    # 将自定义工具与智能体集成
    try:
        if os.getenv("OPENAI_API_KEY"):
            llm = OpenAI(temperature=0)
            
            # 创建智能体并添加自定义工具
            tools = [weather_tool]
            agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            
            # 示例任务
            print("\n运行使用自定义工具的智能体...")
            print("问题: 北京今天的天气怎么样？")
            result = agent.invoke("北京今天的天气怎么样？")
            print(f"回答: {result['output']}\n")
        else:
            print("\n未设置OPENAI_API_KEY，跳过模型调用")
            print("自定义工具可以集成到智能体中，使智能体能够访问特定功能或API。")
    
    except Exception as e:
        print(f"使用带自定义工具的智能体时出错: {e}")
    
    # --- 示例3：使用函数的简单工具 ---
    print("\n--- 使用函数的简单工具示例 ---")
    
    from langchain.tools import tool
    
    @tool
    def calculate_area_of_circle(radius: float) -> float:
        """计算给定半径的圆的面积。输入应为圆的半径（数字）。"""
        import math
        return math.pi * (float(radius) ** 2)
    
    # 测试工具
    print("使用装饰器创建工具：")
    print(f"工具名称: {calculate_area_of_circle.name}")
    print(f"工具描述: {calculate_area_of_circle.description}")
    print(f"计算半径为5的圆面积: {calculate_area_of_circle.run(5)}")
    
    # 将工具与智能体集成
    try:
        if os.getenv("OPENAI_API_KEY"):
            llm = OpenAI(temperature=0)
            
            # 创建智能体并添加工具
            agent = initialize_agent(
                [calculate_area_of_circle],
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            
            # 示例任务
            print("\n运行使用函数工具的智能体...")
            print("问题: 计算半径为7的圆的面积是多少？")
            result = agent.invoke("计算半径为7的圆的面积是多少？")
            print(f"回答: {result['output']}\n")
        else:
            print("\n未设置OPENAI_API_KEY，跳过模型调用")
            print("使用@tool装饰器可以快速将函数转换为智能体可用的工具。")
    
    except Exception as e:
        print(f"使用带函数工具的智能体时出错: {e}")


# ===============================
# 第3部分：智能体与记忆
# ===============================
def agent_memory_examples():
    """智能体与记忆示例"""
    print("\n=== 智能体与记忆示例 ===")
    
    from langchain.memory import ConversationBufferMemory
    from langchain.memory import ConversationSummaryMemory
    from langchain.agents import AgentType, initialize_agent
    from langchain_openai import OpenAI, ChatOpenAI
    from langchain.tools import DuckDuckGoSearchRun
    
    # --- 示例1：带缓冲记忆的智能体 ---
    print("\n--- 带缓冲记忆的智能体示例 ---")
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 创建语言模型
            chat_model = ChatOpenAI(temperature=0)
            
            # 创建记忆组件
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            # 创建工具
            search_tool = DuckDuckGoSearchRun()
            tools = [search_tool]
            
            # 初始化带记忆的智能体
            agent = initialize_agent(
                tools,
                chat_model,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True
            )
            
            # 示例对话
            print("运行带缓冲记忆的智能体...")
            print("用户: 我的名字是张明。")
            result1 = agent.invoke({"input": "我的名字是张明。"})
            print(f"智能体: {result1['output']}")
            
            print("\n用户: 你还记得我的名字吗？")
            result2 = agent.invoke({"input": "你还记得我的名字吗？"})
            print(f"智能体: {result2['output']}")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("带缓冲记忆的智能体可以存储完整的对话历史，记住之前交互中提到的信息。")
            print("这使得智能体能够理解上下文并保持连贯的对话。")
    
    except Exception as e:
        print(f"运行带缓冲记忆的智能体时出错: {e}")

    # --- 示例2：带摘要记忆的智能体 ---
    print("\n--- 带摘要记忆的智能体示例 ---")
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 创建语言模型
            chat_model = ChatOpenAI(temperature=0)
            llm = OpenAI(temperature=0)
            
            # 创建摘要记忆组件
            summary_memory = ConversationSummaryMemory(
                llm=llm,
                memory_key="chat_history",
                return_messages=True
            )
            
            # 初始化带摘要记忆的智能体
            tools = [DuckDuckGoSearchRun()]
            agent = initialize_agent(
                tools,
                chat_model,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=summary_memory,
                verbose=True
            )
            
            # 示例对话
            print("运行带摘要记忆的智能体...")
            print("用户: 我是一名软件工程师，专注于人工智能领域。")
            result1 = agent.invoke({"input": "我是一名软件工程师，专注于人工智能领域。"})
            print(f"智能体: {result1['output']}")
            
            print("\n用户: 我需要一些关于如何提高机器学习模型性能的建议。")
            result2 = agent.invoke({"input": "我需要一些关于如何提高机器学习模型性能的建议。"})
            print(f"智能体: {result2['output']}")
            
            print("\n用户: 考虑到我的背景，这些建议适合我吗？")
            result3 = agent.invoke({"input": "考虑到我的背景，这些建议适合我吗？"})
            print(f"智能体: {result3['output']}")
            
            # 显示摘要
            if hasattr(summary_memory, "llm_chain") and hasattr(summary_memory.llm_chain, "memory") and hasattr(summary_memory.llm_chain.memory, "buffer"):
                print("\n记忆摘要:")
                print(summary_memory.llm_chain.memory.buffer)
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("带摘要记忆的智能体可以将长对话历史压缩为摘要。")
            print("这样可以减少上下文长度，同时保留关键信息，适合长对话场景。")
    
    except Exception as e:
        print(f"运行带摘要记忆的智能体时出错: {e}")


# ===============================
# 第4部分：自定义智能体
# ===============================
def custom_agent_examples():
    """自定义智能体示例"""
    print("\n=== 自定义智能体示例 ===")
    
    # --- 示例1：从头创建自定义智能体 ---
    print("\n--- 从头创建自定义智能体示例 ---")
    
    from langchain.agents import Agent
    from langchain.schema import AgentAction, AgentFinish
    from langchain_openai import ChatOpenAI
    from langchain.tools import DuckDuckGoSearchRun
    
    class SimpleAgent(Agent):
        """简单的自定义智能体实现"""
        
        @property
        def input_keys(self):
            """智能体输入键"""
            return ["input"]
        
        def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
        ) -> Union[AgentAction, AgentFinish]:
            """根据输入和中间步骤决定下一个操作"""
            # 获取用户输入
            user_input = kwargs["input"]
            
            # 检查是否已执行足够的步骤或达到了结论
            if len(intermediate_steps) >= 3:
                # 达到最大步骤，返回最终答案
                return AgentFinish(
                    return_values={"output": "已达到最大步骤数，这是我的最终回答。"},
                    log="已达到最大步骤数。"
                )
            
            # 如果输入包含特定关键词，使用搜索工具
            if "搜索" in user_input or "查找" in user_input:
                return AgentAction(
                    tool="search",
                    tool_input=user_input,
                    log="用户想要搜索信息，使用搜索工具。"
                )
            
            # 否则，直接返回回答
            return AgentFinish(
                return_values={"output": "我是一个简单的自定义智能体。我只能回答基本问题或执行搜索。"}, 
                log="没有需要使用工具的任务，直接回答。"
            )
            
        @classmethod
        def from_llm_and_tools(
            cls, llm: Optional[Any] = None, tools: Optional[List[Any]] = None, **kwargs: Any
        ) -> Agent:
            """从LLM和工具创建智能体"""
            # 这里可以忽略llm，因为我们的简单智能体不需要LLM
            # 但我们需要设置工具
            tools = tools or []
            
            # 创建工具映射
            tool_names = [tool.name for tool in tools]
            tool_by_name = {tool.name: tool for tool in tools}
            
            # 创建智能体实例
            return cls(
                tools=tools,
                tool_names=tool_names,
                tool_by_name=tool_by_name,
                **kwargs
            )
    
    try:
        print("创建自定义智能体...")
        
        # 创建工具
        search_tool = DuckDuckGoSearchRun()
        tools = [search_tool]
        
        # 创建自定义智能体
        custom_agent = SimpleAgent.from_llm_and_tools(tools=tools)
        
        # 运行智能体
        from langchain.agents import AgentExecutor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=custom_agent,
            tools=tools,
            verbose=True
        )
        
        # 测试智能体
        print("运行自定义智能体...")
        print("用户: 你好，你是谁？")
        result1 = agent_executor.invoke({"input": "你好，你是谁？"})
        print(f"智能体: {result1['output']}")
        
        print("\n用户: 请搜索人工智能最新发展")
        if os.getenv("DUCKDUCKGO_AVAILABLE", "False").lower() == "true":
            result2 = agent_executor.invoke({"input": "请搜索人工智能最新发展"})
            print(f"智能体: {result2['output']}")
        else:
            print("DuckDuckGo搜索未启用，跳过搜索示例")
    
    except Exception as e:
        print(f"运行自定义智能体时出错: {e}")

    # --- 示例2：使用LCEL创建自定义智能体 ---
    print("\n--- 使用LCEL创建自定义智能体示例 ---")
    
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 创建语言模型
            chat_model = ChatOpenAI(temperature=0.7)
            
            # 创建工具
            search_tool = DuckDuckGoSearchRun()
            
            # 创建提示模板
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="你是一个友好且乐于助人的AI助手。你可以回答问题并使用提供的工具。"),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # 创建智能体
            from langchain.agents.format_scratchpad import format_to_openai_function_messages
            from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
            
            from langchain.agents import create_openai_functions_agent
            from langchain.agents import AgentExecutor
            
            agent = create_openai_functions_agent(
                llm=chat_model,
                tools=[search_tool],
                prompt=prompt
            )
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=[search_tool],
                verbose=True
            )
            
            # 运行智能体
            print("使用LCEL运行自定义智能体...")
            print("查询: 什么是量子计算？")
            if os.getenv("DUCKDUCKGO_AVAILABLE", "False").lower() == "true":
                result = agent_executor.invoke({"input": "什么是量子计算？"})
                print(f"回答: {result['output']}")
            else:
                print("DuckDuckGo搜索未启用，跳过示例")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("使用LCEL可以更灵活地创建自定义智能体，轻松组合提示、模型和解析器。")
    
    except Exception as e:
        print(f"运行LCEL自定义智能体时出错: {e}")


# ===============================
# 第5部分：多智能体系统
# ===============================
def multi_agent_examples():
    """多智能体系统示例"""
    print("\n=== 多智能体系统示例 ===")
    
    print("多智能体系统允许多个智能体协同工作，共同完成复杂任务。")
    print("在LangChain中，可以通过不同方式实现多智能体系统：")
    print("1. 顺序链式调用多个智能体")
    print("2. 使用专门的多智能体框架")
    print("3. 创建自定义多智能体系统")
    
    print("\n由于多智能体系统较为复杂，这里提供一个简化的概念示例。")
    print("在实际应用中，可以根据具体需求设计更复杂的多智能体系统。")
    
    # 简单示例：两个智能体协作
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    # 定义专家智能体的角色和模板
    researcher_template = """你是一位研究专家，精通信息收集和分析。
    请对以下主题进行深入研究并提供关键信息：{topic}
    输出格式：提供1-3个关键发现点，每个发现点不超过2-3句话。
    """
    
    writer_template = """你是一位技术文档专家，擅长将复杂信息转化为清晰易懂的内容。
    请根据以下研究发现，撰写一个简洁的技术总结：{research_findings}
    输出格式：一个150-200字的技术总结，适合非专业人士阅读。
    """
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            # 创建语言模型
            llm = OpenAI(temperature=0.7)
            
            # 创建研究专家智能体
            researcher_prompt = PromptTemplate(
                template=researcher_template,
                input_variables=["topic"]
            )
            researcher_agent = LLMChain(llm=llm, prompt=researcher_prompt)
            
            # 创建写作专家智能体
            writer_prompt = PromptTemplate(
                template=writer_template,
                input_variables=["research_findings"]
            )
            writer_agent = LLMChain(llm=llm, prompt=writer_prompt)
            
            # 运行多智能体系统
            print("\n运行简化的多智能体系统...")
            topic = "量子计算在密码学中的应用"
            print(f"研究主题: {topic}")
            
            # 研究专家生成发现
            research_findings = researcher_agent.invoke({"topic": topic})["text"]
            print(f"\n研究专家发现:\n{research_findings}")
            
            # 技术写作专家生成总结
            technical_summary = writer_agent.invoke({"research_findings": research_findings})["text"]
            print(f"\n技术写作专家总结:\n{technical_summary}")
        else:
            print("未设置OPENAI_API_KEY，跳过模型调用")
            print("多智能体系统的优势包括专业化分工、信息过滤和增强、以及复杂任务协作能力。")
    
    except Exception as e:
        print(f"运行多智能体系统示例时出错: {e}")


# ===============================
# 主函数：运行所有示例
# ===============================
def main():
    """运行所有智能体组件示例"""
    print("# LangChain智能体(Agents)组件使用示例")
    print("=" * 80)
    
    print("\n## 重要提示")
    print("运行这些示例前，请确保已设置OpenAI API密钥。例如：")
    print("```python")
    print('import os')
    print('os.environ["OPENAI_API_KEY"] = "您的OpenAI API密钥"')
    print("```")
    print("对于部分功能（如搜索），还需要设置相应的服务可用环境变量。")
    
    # 运行示例
    basic_agent_examples()
    tools_examples()
    agent_memory_examples()
    custom_agent_examples()
    multi_agent_examples()
    
    print("\n" + "=" * 80)
    print("示例运行完成！")


if __name__ == "__main__":
    main()
