#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain智能体(Agent)使用示例 - 单智能体模式

本模块展示了如何在LangChain中创建和使用各种类型的智能体，包括：
1. 基础智能体配置
2. 不同智能体类型的使用
3. 自定义工具创建
4. 智能体记忆与持久化
5. 高级智能体调优与控制

注意：运行这些示例前，请确保已设置相应的API密钥环境变量
"""

import os
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===============================
# 第1部分：基础智能体示例
# ===============================
def basic_agent_examples():
    """基础智能体示例"""
    print("=== 基础智能体示例 ===")
    
    from langchain.agents import AgentType, initialize_agent, load_tools
    from langchain_openai import ChatOpenAI
    from langchain_community.tools import DuckDuckGoSearchRun
    
    # 检查是否有OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("未设置OPENAI_API_KEY，跳过本节示例")
        print("要运行智能体示例，请设置OPENAI_API_KEY环境变量")
        return
    
    # --- 零样本ReAct智能体 ---
    print("\n--- 零样本ReAct智能体示例 ---")
    
    # 创建语言模型
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # 创建工具
    search = DuckDuckGoSearchRun()
    tools = [search]
    
    # 初始化智能体
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # 零样本ReAct类型
        verbose=True  # 显示智能体思考过程
    )
    
    # 使用智能体执行任务
    try:
        print("\n执行任务: '谁是当前的中国国家主席？他于何时上任？'")
        response = agent.run("谁是当前的中国国家主席？他于何时上任？")
        print(f"\n智能体回答: {response}")
    except Exception as e:
        print(f"执行时出错: {e}")
    
    # --- 带工具集的智能体 ---
    print("\n--- 带多种工具的智能体示例 ---")
    
    # 加载内置工具
    try:
        # 加载基础工具：计算器和维基百科
        print("加载内置工具: calculator, wikipedia")
        basic_tools = load_tools(["llm-math", "wikipedia"], llm=llm)
        
        # 添加搜索工具
        all_tools = basic_tools + [search]
        print(f"共加载了 {len(all_tools)} 个工具")
        
        # 创建多工具智能体
        multi_tool_agent = initialize_agent(
            tools=all_tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # 执行需要多种工具的复杂任务
        print("\n执行任务: '计算23.47乘以16.95，然后帮我总结一下圆周率π的历史'")
        response = multi_tool_agent.run("计算23.47乘以16.95，然后帮我总结一下圆周率π的历史")
        print(f"\n智能体回答: {response}")
    except Exception as e:
        print(f"执行时出错: {e}")
        print("某些工具可能需要额外安装，如: pip install wikipedia")


# ===============================
# 第2部分：不同智能体类型示例
# ===============================
def agent_types_examples():
    """不同智能体类型示例"""
    print("\n=== 不同智能体类型示例 ===")
    
    from langchain.agents import AgentType, initialize_agent, load_tools
    from langchain_openai import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    
    # 检查是否有OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("未设置OPENAI_API_KEY，跳过本节示例")
        return
    
    # 创建语言模型
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # 加载基础工具
    try:
        print("加载工具...")
        tools = load_tools(["llm-math"], llm=llm)
        
        # --- 对话型ReAct智能体 ---
        print("\n--- 对话型ReAct智能体示例 ---")
        
        # 创建记忆组件
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # 初始化对话型智能体
        conversational_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )
        
        # 多轮对话示例
        print("\n开始多轮对话:")
        responses = []
        
        # 第一轮
        print("\n问题1: 请计算125乘以7.85等于多少？")
        responses.append(conversational_agent.run(input="请计算125乘以7.85等于多少？"))
        print(f"智能体回答: {responses[-1]}")
        
        # 第二轮，引用之前的计算结果
        print("\n问题2: 把刚才的结果再乘以0.15")
        responses.append(conversational_agent.run(input="把刚才的结果再乘以0.15"))
        print(f"智能体回答: {responses[-1]}")
        
        # --- OpenAI函数智能体 ---
        print("\n--- OpenAI函数智能体示例 ---")
        
        # 确认使用支持函数调用的模型
        function_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        
        # 创建函数智能体
        function_agent = initialize_agent(
            tools=tools,
            llm=function_llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )
        
        # 执行任务
        print("\n执行任务: '计算(17.38 + 93.5) * 12.37的结果'")
        response = function_agent.run("计算(17.38 + 93.5) * 12.37的结果")
        print(f"\n智能体回答: {response}")
        
        # --- 结构化输入智能体 ---
        print("\n--- 结构化输入智能体示例 ---")
        
        # 创建结构化智能体
        structured_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # 执行任务
        print("\n执行任务: '对于一个半径为5米的球体，计算它的体积'")
        response = structured_agent.run("对于一个半径为5米的球体，计算它的体积")
        print(f"\n智能体回答: {response}")
    
    except Exception as e:
        print(f"执行时出错: {e}")


# ===============================
# 第3部分：自定义工具示例
# ===============================
def custom_tools_examples():
    """自定义工具示例"""
    print("\n=== 自定义工具示例 ===")
    
    from langchain.agents import AgentType, initialize_agent, Tool
    from langchain.tools import BaseTool, StructuredTool
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    from typing import Optional, Type
    
    # 检查是否有OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("未设置OPENAI_API_KEY，跳过本节示例")
        return
    
    # 创建语言模型
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # --- 简单函数工具 ---
    print("\n--- 简单函数工具示例 ---")
    
    def get_current_weather(location: str) -> str:
        """获取指定地点的当前天气信息"""
        # 在实际应用中，这里应该调用真实的天气API
        # 这里只是一个模拟返回
        weather_data = {
            "北京": "晴朗，26°C，湿度45%",
            "上海": "多云，28°C，湿度60%",
            "广州": "小雨，30°C，湿度80%",
            "深圳": "阴天，29°C，湿度75%",
        }
        return f"{location}的当前天气: {weather_data.get(location, '数据不可用，请检查地点名称')}"
    
    def chinese_calendar_info(date: Optional[str] = None) -> str:
        """获取指定日期(格式:YYYY-MM-DD)的中国农历信息，如不指定则返回今日信息"""
        # 模拟数据
        if not date:
            date = time.strftime("%Y-%m-%d")
            lunar_date = "癸卯兔年六月初八"
        else:
            # 这里应有真实的农历转换逻辑
            lunar_date = "农历信息暂不可用，请使用真实API获取"
        
        return f"公历 {date} 对应的农历是 {lunar_date}"
    
    # 创建函数工具
    weather_tool = Tool(
        name="CurrentWeather",
        description="当需要获取某个中国城市的天气情况时使用，输入参数是城市名称，如'北京'、'上海'等",
        func=get_current_weather
    )
    
    calendar_tool = Tool(
        name="ChineseCalendar",
        description="获取特定日期的中国农历信息，输入格式为YYYY-MM-DD，如不提供则返回今日信息",
        func=chinese_calendar_info
    )
    
    # --- 自定义工具类 ---
    print("--- 自定义工具类示例 ---")
    
    class StockPriceTool(BaseTool):
        """获取股票价格信息的工具"""
        name = "StockPrice"
        description = "获取指定公司的股票价格信息，需要提供股票代码，如'BABA'表示阿里巴巴"
        
        def _run(self, stock_code: str) -> str:
            """运行工具逻辑"""
            # 模拟数据
            stock_data = {
                "BABA": "阿里巴巴 (BABA): 78.34 USD (+1.2%)",
                "TCEHY": "腾讯控股 (TCEHY): 38.45 USD (-0.5%)",
                "JD": "京东 (JD): 28.12 USD (+0.8%)",
                "PDD": "拼多多 (PDD): 139.27 USD (+2.1%)",
            }
            return stock_data.get(stock_code.upper(), f"未能找到代码为 {stock_code} 的股票信息")
        
        async def _arun(self, stock_code: str) -> str:
            """异步运行（这里简单返回同步结果）"""
            return self._run(stock_code)
    
    # 创建自定义工具实例
    stock_tool = StockPriceTool()

    # --- 结构化输入工具 ---
    print("--- 结构化输入工具示例 ---")
    
    class CurrencyConversionInput(BaseModel):
        """货币转换输入"""
        amount: float = Field(..., description="需要转换的金额")
        from_currency: str = Field(..., description="原始货币代码，如USD、CNY等")
        to_currency: str = Field(..., description="目标货币代码，如USD、CNY等")
    
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        """将指定金额从一种货币转换为另一种货币"""
        # 模拟汇率数据
        rates = {
            "USD": {"CNY": 7.18, "EUR": 0.93, "GBP": 0.80, "JPY": 148.20},
            "CNY": {"USD": 0.14, "EUR": 0.13, "GBP": 0.11, "JPY": 20.64},
            "EUR": {"USD": 1.08, "CNY": 7.73, "GBP": 0.86, "JPY": 159.35},
        }
        
        # 转换货币
        from_c = from_currency.upper()
        to_c = to_currency.upper()
        
        if from_c not in rates or to_c not in rates.get(from_c, {}):
            return f"抱歉，不支持从 {from_c} 到 {to_c} 的转换"
        
        # 计算转换后金额
        rate = rates[from_c][to_c]
        converted_amount = amount * rate
        
        return f"{amount} {from_c} = {converted_amount:.2f} {to_c} (汇率: 1 {from_c} = {rate} {to_c})"
    
    # 创建结构化工具
    currency_tool = StructuredTool.from_function(
        func=convert_currency,
        name="CurrencyConverter",
        description="将金额从一种货币转换为另一种货币，需要提供金额、原始货币和目标货币",
        args_schema=CurrencyConversionInput
    )
    
    # --- 使用自定义工具的智能体 ---
    print("\n--- 使用自定义工具的智能体示例 ---")
    
    # 组合所有自定义工具
    custom_tools = [weather_tool, calendar_tool, stock_tool, currency_tool]
    
    # 初始化智能体
    custom_agent = initialize_agent(
        tools=custom_tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # 执行任务
    try:
        print("\n执行任务1: '北京今天天气怎么样？同时告诉我今天的农历日期'")
        response1 = custom_agent.run("北京今天天气怎么样？同时告诉我今天的农历日期")
        print(f"\n智能体回答: {response1}")
        
        print("\n执行任务2: '请查询阿里巴巴的股票价格，并将100美元转换为人民币'")
        response2 = custom_agent.run("请查询阿里巴巴的股票价格，并将100美元转换为人民币")
        print(f"\n智能体回答: {response2}")
    except Exception as e:
        print(f"执行时出错: {e}")


# ===============================
# 主函数：运行各种示例
# ===============================
def main():
    """主函数，运行各种智能体示例"""
    print("LangChain智能体(Agent)示例程序\n")
    print("注意：运行前请确保已经设置好相关API密钥环境变量\n")
    
    # 运行基础智能体示例
    basic_agent_examples()
    
    # 运行不同智能体类型示例
    agent_types_examples()
    
    # 运行自定义工具示例
    custom_tools_examples()


# 程序入口
if __name__ == "__main__":
    main()
