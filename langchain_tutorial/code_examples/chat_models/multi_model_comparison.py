#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模型比较示例：展示如何在LangChain中使用和比较不同类型的聊天模型
此示例展示了如何使用不同的聊天模型来回答同样的问题，并比较它们的响应
"""

import os
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import asyncio

from langchain_community.callbacks import get_openai_callback
from rich.console import Console
from rich.table import Table
from concurrent.futures import ThreadPoolExecutor

# 导入LangChain核心组件
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_core.callbacks import get_openai_callback

# 导入不同的聊天模型
# 注意：取消注释您想要使用的模型，并确保已安装相应的依赖

# OpenAI模型
from langchain_openai import ChatOpenAI

# Anthropic模型
# from langchain_anthropic import ChatAnthropic

# 百度文心一言模型
# from langchain_qianfan import ChatQianfan

# 阿里通义千问模型
# from langchain_dashscope import ChatDashscope

# 创建Rich控制台用于美化输出
console = Console()


class ModelComparison:
    """
    聊天模型比较类：用于比较不同模型对同一问题的回答
    """

    def __init__(self):
        self.models: Dict[str, BaseChatModel] = {}
        self.results: List[Dict[str, Any]] = []

    def add_model(self, name: str, model: BaseChatModel) -> None:
        """添加一个模型到比较列表"""
        self.models[name] = model

    def setup_models(self) -> None:
        """设置要比较的模型"""
        # OpenAI模型
        if os.environ.get("OPENAI_API_KEY"):
            # 假设用户已经设置环境变量OPENAI_API_KEY
            self.add_model("OpenAI-GPT-3.5", ChatOpenAI(model="gpt-3.5-turbo"))
            self.add_model("OpenAI-GPT-4", ChatOpenAI(model="gpt-4"))
        else:
            console.print("[yellow]OpenAI模型未加载，请设置OPENAI_API_KEY环境变量[/yellow]")

        # Anthropic模型
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from langchain_anthropic import ChatAnthropic
                self.add_model("Anthropic-Claude", ChatAnthropic(model="claude-3-sonnet-20240229"))
            except ImportError:
                console.print("[yellow]未找到langchain_anthropic包，跳过Anthropic模型加载[/yellow]")
        else:
            console.print("[yellow]Anthropic模型未加载，请设置ANTHROPIC_API_KEY环境变量[/yellow]")

        # 百度文心一言模型
        if os.environ.get("QIANFAN_AK") and os.environ.get("QIANFAN_SK"):
            try:
                from langchain_qianfan import ChatQianfan
                self.add_model("Baidu-ERNIE-Bot", ChatQianfan())
            except ImportError:
                console.print("[yellow]未找到langchain_qianfan包，跳过百度文心一言模型加载[/yellow]")
        else:
            console.print("[yellow]百度文心一言模型未加载，请设置QIANFAN_AK和QIANFAN_SK环境变量[/yellow]")

        # 阿里通义千问模型
        if os.environ.get("DASHSCOPE_API_KEY"):
            try:
                from langchain_dashscope import ChatDashscope
                self.add_model("Alibaba-Qwen", ChatDashscope())
            except ImportError:
                console.print("[yellow]未找到langchain_dashscope包，跳过通义千问模型加载[/yellow]")
        else:
            console.print("[yellow]通义千问模型未加载，请设置DASHSCOPE_API_KEY环境变量[/yellow]")

        if not self.models:
            console.print("[red]没有可用的模型，请设置至少一个模型的API密钥[/red]")
            exit(1)

        console.print(f"[green]成功加载了 {len(self.models)} 个模型：{', '.join(self.models.keys())}[/green]")

    async def acompare(self,
                       system_prompt: Optional[str],
                       human_prompt: str,
                       temperature: float = 0.7) -> None:
        """
        异步比较所有模型回答同一问题的结果
        """
        self.results = []
        tasks = []

        # 创建消息列表
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=human_prompt))

        # 创建每个模型的异步任务
        for name, model in self.models.items():
            # 设置温度
            model_with_temp = model.with_config(temperature=temperature)
            tasks.append(self.ainvoke_model(name, model_with_temp, messages))

        # 等待所有任务完成
        await asyncio.gather(*tasks)

        # 按响应时间排序结果
        self.results.sort(key=lambda x: x['response_time'])

        # 显示结果表格
        self.display_results()

    async def ainvoke_model(self, name: str, model: BaseChatModel, messages: List) -> None:
        """异步调用单个模型并记录结果"""
        try:
            console.print(f"[cyan]正在调用 {name} 模型...[/cyan]")
            start_time = time.time()

            with get_openai_callback() as cb:
                response = await model.ainvoke(messages)

            end_time = time.time()
            response_time = end_time - start_time
            token_usage = {}

            if hasattr(cb, 'total_tokens'):
                token_usage = {
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_tokens': cb.total_tokens,
                    'cost': f"${cb.total_cost:.4f}"
                }

            result = {
                'model_name': name,
                'response': response.content,
                'response_time': response_time,
                'token_usage': token_usage
            }

            self.results.append(result)
            console.print(f"[green]{name} 模型响应完成，用时 {response_time:.2f} 秒[/green]")

        except Exception as e:
            console.print(f"[red]调用 {name} 模型时出错: {str(e)}[/red]")
            self.results.append({
                'model_name': name,
                'response': f"错误: {str(e)}",
                'response_time': -1,
                'token_usage': {}
            })

    def compare(self,
                system_prompt: Optional[str],
                human_prompt: str,
                temperature: float = 0.7) -> None:
        """同步比较所有模型回答同一问题的结果"""
        self.results = []

        # 创建消息列表
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=human_prompt))

        # 循环调用每个模型
        for name, model in self.models.items():
            try:
                console.print(f"[cyan]正在调用 {name} 模型...[/cyan]")
                # 设置温度
                model_with_temp = model.with_config(temperature=temperature)

                start_time = time.time()

                with get_openai_callback() as cb:
                    response = model_with_temp.invoke(messages)

                end_time = time.time()
                response_time = end_time - start_time
                token_usage = {}

                if hasattr(cb, 'total_tokens'):
                    token_usage = {
                        'prompt_tokens': cb.prompt_tokens,
                        'completion_tokens': cb.completion_tokens,
                        'total_tokens': cb.total_tokens,
                        'cost': f"${cb.total_cost:.4f}"
                    }

                result = {
                    'model_name': name,
                    'response': response.content,
                    'response_time': response_time,
                    'token_usage': token_usage
                }

                self.results.append(result)
                console.print(f"[green]{name} 模型响应完成，用时 {response_time:.2f} 秒[/green]")

            except Exception as e:
                console.print(f"[red]调用 {name} 模型时出错: {str(e)}[/red]")
                self.results.append({
                    'model_name': name,
                    'response': f"错误: {str(e)}",
                    'response_time': -1,
                    'token_usage': {}
                })

        # 按响应时间排序结果
        self.results.sort(key=lambda x: x['response_time'])

        # 显示结果表格
        self.display_results()

    def display_results(self) -> None:
        """以表格形式展示比较结果"""
        # 创建性能比较表格
        perf_table = Table(title="模型性能比较")
        perf_table.add_column("模型", style="cyan")
        perf_table.add_column("响应时间 (秒)", style="green")
        perf_table.add_column("令牌数量", style="yellow")
        perf_table.add_column("成本", style="red")

        for result in self.results:
            model_name = result['model_name']
            response_time = f"{result['response_time']:.2f}" if result['response_time'] > 0 else "N/A"

            token_info = result.get('token_usage', {})
            tokens = f"{token_info.get('total_tokens', 'N/A')}" if token_info else "N/A"
            cost = token_info.get('cost', 'N/A') if token_info else "N/A"

            perf_table.add_row(model_name, response_time, tokens, cost)

        console.print(perf_table)

        # 为每个模型的回答创建单独的表格
        for result in self.results:
            resp_table = Table(title=f"{result['model_name']} 的回答", show_lines=True)
            resp_table.add_column("回答内容", style="green", no_wrap=False)

            # 将回答拆分成多行以提高可读性
            response_text = str(result['response'])
            resp_table.add_row(response_text)

            console.print(resp_table)
            console.print("")  # 添加一个空行作为分隔

    def save_results_to_csv(self, filename: str) -> None:
        """将结果保存到CSV文件"""
        data = []
        for result in self.results:
            row = {
                'model_name': result['model_name'],
                'response': result['response'],
                'response_time': result['response_time'],
            }

            # 添加令牌使用信息（如果有）
            token_info = result.get('token_usage', {})
            if token_info:
                row.update({
                    'prompt_tokens': token_info.get('prompt_tokens', 'N/A'),
                    'completion_tokens': token_info.get('completion_tokens', 'N/A'),
                    'total_tokens': token_info.get('total_tokens', 'N/A'),
                    'cost': token_info.get('cost', 'N/A').replace('$', ''),  # 移除美元符号
                })

            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        console.print(f"[green]结果已保存到 {filename}[/green]")


def main():
    """主函数"""
    # 创建并设置模型比较器
    comparison = ModelComparison()
    comparison.setup_models()

    # 测试场景:

    # 1. 知识问答
    console.print("\n[bold yellow]===== 知识问答测试 =====[/bold yellow]")
    system_prompt = "你是一位有用的AI助手，擅长回答知识类问题。"
    human_prompt = "请解释中国传统节日春节的起源和主要习俗。"

    # 同步方法
    comparison.compare(system_prompt, human_prompt)

    # 保存结果
    comparison.save_results_to_csv("knowledge_qa_results.csv")

    # 2. 创意写作
    console.print("\n[bold yellow]===== 创意写作测试 =====[/bold yellow]")
    system_prompt = "你是一位创意写作助手，善于创作富有想象力的内容。"
    human_prompt = "请以'月光下的古城'为题，写一段100字左右的微型小说。"

    # 异步方法
    asyncio.run(comparison.acompare(system_prompt, human_prompt, temperature=0.9))

    # 保存结果
    comparison.save_results_to_csv("creative_writing_results.csv")

    # 3. 代码生成
    console.print("\n[bold yellow]===== 代码生成测试 =====[/bold yellow]")
    system_prompt = "你是一位编程专家，擅长编写清晰易懂的代码。"
    human_prompt = "请用Python编写一个简单的Web爬虫，可以获取指定网页的标题和所有链接。使用requests和BeautifulSoup库。"

    # 同步方法
    comparison.compare(system_prompt, human_prompt, temperature=0.2)

    # 保存结果
    comparison.save_results_to_csv("code_generation_results.csv")


if __name__ == "__main__":
    main()
