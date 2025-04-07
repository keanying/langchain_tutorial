#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单聊天机器人示例：展示如何使用LangChain构建一个聊天机器人
集成了聊天模型、提示模板、检索工具和基本工具使用
"""

import os
import json
import datetime
import requests
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.markdown import Markdown
import asyncio

# 导入LangChain核心组件
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

# 导入LangChain工具
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.tools import tool

# 导入聊天模型（根据需要选择）
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_qianfan import ChatQianfan
# from langchain_dashscope import ChatDashscope

# 创建Rich控制台用于美化输出
console = Console()

class SimpleChatAgent:
    """
    简单的聊天机器人代理，整合了多种功能:
    1. 基本聊天对话
    2. 工具使用能力
    3. 网络搜索
    4. 聊天历史记录
    5. 图像生成和处理功能
    """
    
    def __init__(self, model_name: str = "openai", temperature: float = 0.7):
        """初始化聊天机器人代理"""
        self.temperature = temperature
        self.chat_history = []
        self.setup_model(model_name)
        self.setup_tools()
        
    def setup_model(self, model_name: str) -> None:
        """设置聊天模型"""
        if model_name == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("未设置OPENAI_API_KEY环境变量")
            self.chat_model = ChatOpenAI(
                temperature=self.temperature,
                model="gpt-4"  # 可以替换为其他OpenAI模型
            )
            console.print("[green]已加载OpenAI GPT-4模型[/green]")
        elif model_name == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError("未设置ANTHROPIC_API_KEY环境变量")
            from langchain_anthropic import ChatAnthropic
            self.chat_model = ChatAnthropic(
                temperature=self.temperature,
                model="claude-3-sonnet-20240229"  # 可以替换为其他Claude模型
            )
            console.print("[green]已加载Anthropic Claude 3 Sonnet模型[/green]")
        elif model_name == "baidu":
            if not (os.environ.get("QIANFAN_AK") and os.environ.get("QIANFAN_SK")):
                raise ValueError("未设置QIANFAN_AK和QIANFAN_SK环境变量")
            from langchain_qianfan import ChatQianfan
            self.chat_model = ChatQianfan(temperature=self.temperature)
            console.print("[green]已加载百度文心一言模型[/green]")
        elif model_name == "alibaba":
            if not os.environ.get("DASHSCOPE_API_KEY"):
                raise ValueError("未设置DASHSCOPE_API_KEY环境变量")
            from langchain_dashscope import ChatDashscope
            self.chat_model = ChatDashscope(temperature=self.temperature)
            console.print("[green]已加载阿里通义千问模型[/green]")
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
    def setup_tools(self) -> None:
        """设置可用工具"""
        # 搜索工具
        self.search_tool = DuckDuckGoSearchRun()
        
        # 维基百科工具
        wiki = WikipediaAPIWrapper(top_k_results=2, lang="zh")
        self.wiki_tool = WikipediaQueryRun(api_wrapper=wiki)
        
        # 定义自定义工具
        @tool
        def get_current_weather(location: str) -> str:
            """获取指定位置的当前天气"""
            # 注意：这只是一个模拟实现，实际使用时应调用真实的天气API
            weather_options = ["晴朗", "多云", "小雨", "大雨", "雷暴", "阴天"]
            import random
            weather = random.choice(weather_options)
            temp = random.randint(15, 35)
            return f"{location}的当前天气: {weather}，温度: {temp}°C"
        
        @tool
        def get_date_time() -> str:
            """获取当前日期和时间"""
            now = datetime.datetime.now()
            return f"当前日期和时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 图像工具封装
        try:
            from metagpt.tools.libs.image_getter import ImageGetter
            self.image_getter = ImageGetter()
            self.has_image_tool = True
        except ImportError:
            self.has_image_tool = False
            console.print("[yellow]警告: 未找到ImageGetter工具，图像功能将不可用[/yellow]")
        
        # 创建工具列表
        self.tools = [
            self.search_tool,
            self.wiki_tool,
            get_current_weather,
            get_date_time
        ]
        
        # 创建系统提示
        system_prompt = """你是一位友好、专业的AI助手，可以回答用户的各种问题。
        
你可以使用以下工具帮助回答问题:
1. 搜索引擎 - 查询最新的信息
2. 维基百科 - 查询百科知识
3. 天气查询 - 查询特定位置的天气
4. 日期时间 - 获取当前日期和时间
        
请在需要时使用这些工具，确保回答准确、有用。如果需要生成图片，你也可以指出这一点。
在回答问题时，请使用与用户相同的语言。
"""

        # 创建Agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建OpenAI工具代理
        self.agent = create_openai_tools_agent(self.chat_model, self.tools, prompt)
        
        # 创建代理执行器
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        console.print("[green]已成功设置所有工具[/green]")
    
    async def generate_image(self, description: str, image_name: str) -> Optional[str]:
        """生成图像的异步方法"""
        if not self.has_image_tool:
            return "图像生成功能不可用"
        
        try:
            # 确保保存路径存在
            os.makedirs("public/assets/images", exist_ok=True)
            save_path = os.path.abspath(f"public/assets/images/{image_name}")
            
            # 调用图像生成工具
            image_path = await self.image_getter.get(
                search_term=description,
                image_save_path=save_path,
                mode="create"
            )
            return f"已生成图像并保存到: {image_path}"
        except Exception as e:
            console.print(f"[red]图像生成失败: {str(e)}[/red]")
            return f"图像生成失败: {str(e)}"
    
    async def chat(self, user_message: str) -> str:
        """处理用户消息并返回回复"""
        # 检查是否是图像生成请求
        if user_message.lower().startswith("生成图片:") or user_message.lower().startswith("创建图片:"):
            description = user_message.split(":", 1)[1].strip()
            image_name = f"image_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            result = await self.generate_image(description, image_name)
            self.chat_history.append(HumanMessage(content=user_message))
            self.chat_history.append(AIMessage(content=result))
            return result
        
        # 普通聊天或工具使用
        try:
            response = await self.agent_executor.ainvoke({
                "input": user_message,
                "chat_history": self.chat_history
            })
            
            # 更新聊天历史
            self.chat_history.append(HumanMessage(content=user_message))
            self.chat_history.append(AIMessage(content=response["output"]))
            
            # 如果聊天历史过长，删除较早的消息
            if len(self.chat_history) > 20:  # 保留最近10轮对话
                self.chat_history = self.chat_history[-20:]
                
            return response["output"]
        except Exception as e:
            error_msg = f"处理消息时出错: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg
    
    def clear_history(self) -> None:
        """清除聊天历史"""
        self.chat_history = []
        console.print("[yellow]已清除聊天历史[/yellow]")

async def main():
    """主函数"""
    # 选择模型类型
    model_name = "openai"  # 可以是 "openai", "anthropic", "baidu", "alibaba"
    
    try:
        # 创建聊天代理
        console.print("[bold green]初始化聊天机器人...[/bold green]")
        agent = SimpleChatAgent(model_name=model_name, temperature=0.7)
        
        console.print("[bold green]聊天机器人准备就绪! 输入'quit'或'exit'退出对话。[/bold green]")
        console.print("[bold green]输入'clear'可清除聊天历史。[/bold green]")
        console.print("[bold green]输入'生成图片: [描述]'可创建图像。[/bold green]\n")
        
        # 聊天循环
        while True:
            # 获取用户输入
            user_input = input("你: ")
            
            # 检查是否退出
            if user_input.lower() in ["quit", "exit"]:
                console.print("[yellow]结束对话[/yellow]")
                break
            
            # 检查是否清除历史
            if user_input.lower() == "clear":
                agent.clear_history()
                continue
            
            # 处理用户消息
            console.print("[cyan]AI思考中...[/cyan]")
            response = await agent.chat(user_input)
            
            # 打印AI回复
            console.print("\n[bold blue]AI回复:[/bold blue]")
            console.print(Markdown(response))
            console.print("")  # 空行
            
    except Exception as e:
        console.print(f"[bold red]发生错误: {str(e)}[/bold red]")

if __name__ == "__main__":
    asyncio.run(main())