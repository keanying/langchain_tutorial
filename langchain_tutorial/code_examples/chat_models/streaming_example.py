#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
流式输出示例：展示如何在LangChain中实现和处理聊天模型的流式响应
包含基本流式输出、自定义回调处理、流式API接口、Web应用集成等示例
"""

import os
import time
import sys
import asyncio
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

# 导入LangChain核心组件
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# 导入不同的聊天模型
# 注意：取消注释您想要使用的模型，并确保已安装相应的依赖
from langchain_openai import ChatOpenAI

# from langchain_anthropic import ChatAnthropic
# from langchain_qianfan import ChatQianfan
# from langchain_dashscope import ChatDashscope

# 创建Rich控制台用于美化输出
console = Console()


class StreamingCallbackHandler(BaseCallbackHandler):
    """
    自定义流式输出回调处理器
    """

    def __init__(self, display_method: str = "simple"):
        """初始化回调处理器
        
        Args:
            display_method: 显示方法，可选 "simple", "markdown", "animated"
        """
        self.text = ""
        self.display_method = display_method
        self.markdown_text = ""
        self.tokens_received = 0
        self.start_time = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """LLM开始生成时的回调"""
        self.start_time = time.time()
        if self.display_method != "simple":
            console.print("[cyan]开始生成...[/cyan]")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """每当LLM生成新token时的回调"""
        self.tokens_received += 1
        self.text += token
        self.markdown_text += token

        if self.display_method == "simple":
            # 简单打印模式
            print(token, end="", flush=True)
        elif self.display_method == "markdown":
            # 渐进式渲染Markdown
            if token in ["\n", ".", "!", "?", ":", ";"]:
                console.print(Markdown(self.markdown_text))
                self.markdown_text = ""

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM完成生成时的回调"""
        end_time = time.time()
        duration = end_time - self.start_time
        tokens_per_second = self.tokens_received / duration if duration > 0 else 0

        # 显示最后的文本块
        if self.display_method == "markdown" and self.markdown_text:
            console.print(Markdown(self.markdown_text))

        # 打印统计信息
        console.print(f"\n[green]生成完成! 接收了{self.tokens_received}个token，"
                      f"总耗时{duration:.2f}秒，速率{tokens_per_second:.2f}tokens/秒[/green]\n")


class AnimatedStreamingHandler(BaseCallbackHandler):
    """动画效果的流式输出回调处理器"""

    def __init__(self):
        self.text = ""
        self.start_time = None
        self.tokens_received = 0

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """LLM开始生成时的回调"""
        self.start_time = time.time()
        console.print("[bold cyan]开始生成动态响应...[/bold cyan]")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """每当LLM生成新token时的回调，使用rich.live动态更新"""
        self.tokens_received += 1
        self.text += token

        # 留给Live更新器处理显示

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM完成生成时的回调"""
        end_time = time.time()
        duration = end_time - self.start_time
        tokens_per_second = self.tokens_received / duration if duration > 0 else 0

        console.print(f"\n[bold green]生成完成! 接收了{self.tokens_received}个token，"
                      f"总耗时{duration:.2f}秒，速率{tokens_per_second:.2f}tokens/秒[/bold green]\n")


def basic_streaming_example():
    """基本流式输出示例"""
    console.print("\n[bold yellow]===== 基本流式输出示例 =====[/bold yellow]\n")

    # 检查必要的环境变量
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]未设置OPENAI_API_KEY环境变量，使用示例API密钥[/red]")
        os.environ["OPENAI_API_KEY"] = "your-api-key-placeholder"

    # 初始化支持流式输出的聊天模型
    chat = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True,  # 启用流式输出
        callbacks=[StreamingCallbackHandler()]  # 使用自定义回调处理器
    )

    # 发送消息
    console.print("[cyan]向模型发送简单问题，流式输出回答:[/cyan]\n")
    messages = [
        SystemMessage(content="你是一位信息简洁、回答全面的AI助手。"),
        HumanMessage(content="简要介绍中国的四大发明及其历史意义。")
    ]

    # 模型调用 - 回答将通过回调处理器流式输出
    chat.invoke(messages)


async def advanced_streaming_example():
    """高级流式输出示例，展示不同的显示方法"""
    console.print("\n[bold yellow]===== 高级流式输出示例 =====[/bold yellow]\n")

    # 初始化支持流式输出的聊天模型
    chat = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True
    )

    # Markdown渲染示例
    console.print("[bold cyan]\n展示Markdown渲染的流式输出:[/bold cyan]\n")
    messages = [
        SystemMessage(content="你是一位撰写优质技术文档的AI助手，善于使用markdown格式。"),
        HumanMessage(content="请写一篇关于Python异步编程的简短教程，包括小标题、代码示例和要点列表。")
    ]

    # 使用Markdown渲染回调
    markdown_callback = StreamingCallbackHandler(display_method="markdown")
    chat_with_md = chat.with_config(callbacks=[markdown_callback])

    # 发送请求，结果将以Markdown格式流式渲染
    await chat_with_md.ainvoke(messages)

    # 等待用户确认继续
    input("\n按回车键继续...\n")

    # 动画效果示例
    console.print("[bold cyan]\n展示动态效果的流式输出:[/bold cyan]\n")
    messages = [
        SystemMessage(content="你是一位创意写手，擅长写引人入胜的故事。"),
        HumanMessage(content="请写一个短小的科幻故事，关于人工智能与人类的友谊。")
    ]

    # 创建动画处理器
    animated_handler = AnimatedStreamingHandler()

    # 使用Live进行动态更新
    with Live(Text("等待响应..."), refresh_per_second=10) as live:
        # 使用自定义任务处理
        async def update_live():
            response = await chat.ainvoke(
                messages,
                callbacks=[animated_handler]
            )
            return response

        # 启动更新任务
        update_task = asyncio.create_task(update_live())

        # 定期更新Live显示
        while not update_task.done():
            # 将文本渲染为Markdown实时显示
            if hasattr(animated_handler, 'text') and animated_handler.text:
                live.update(Markdown(animated_handler.text))
            await asyncio.sleep(0.1)

        # 确保最终内容显示
        response = await update_task
        live.update(Markdown(response.content))


async def multi_model_streaming_comparison():
    """比较不同模型的流式输出"""
    console.print("\n[bold yellow]===== 多模型流式输出比较 =====[/bold yellow]\n")

    models = {}

    # 添加可用的模型
    if os.environ.get("OPENAI_API_KEY"):
        models["OpenAI GPT-3.5"] = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            models["Anthropic Claude"] = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=0.7,
                streaming=True
            )
        except ImportError:
            console.print("[yellow]未找到langchain_anthropic包，跳过Claude模型[/yellow]")

    if os.environ.get("QIANFAN_AK") and os.environ.get("QIANFAN_SK"):
        try:
            from langchain_qianfan import ChatQianfan
            models["百度文心一言"] = ChatQianfan(
                temperature=0.7,
                streaming=True
            )
        except ImportError:
            console.print("[yellow]未找到langchain_qianfan包，跳过文心一言模型[/yellow]")

    if not models:
        console.print("[red]没有可用的模型进行比较，请设置至少一个模型的API密钥[/red]")
        return

    # 使用统一问题测试各模型
    console.print(f"[cyan]将测试 {len(models)} 个模型的流式输出性能[/cyan]")

    messages = [
        SystemMessage(content="你是一位专业而简洁的AI助手。"),
        HumanMessage(content="请用几句话概括人工智能的发展历程。")
    ]

    for name, model in models.items():
        console.print(f"\n[bold magenta]测试模型: {name}[/bold magenta]")
        callback = StreamingCallbackHandler(display_method="simple")
        model_with_callback = model.with_config(callbacks=[callback])

        try:
            console.print("[cyan]开始流式生成...[/cyan]")
            start_time = time.time()
            await model_with_callback.ainvoke(messages)
            end_time = time.time()
            console.print(f"\n[green]总耗时: {end_time - start_time:.2f}秒[/green]\n")
            console.print("-" * 50)
        except Exception as e:
            console.print(f"[red]模型 {name} 出错: {str(e)}[/red]")
            console.print("-" * 50)


def streaming_to_file_example():
    """将流式输出保存到文件示例"""
    console.print("\n[bold yellow]===== 流式输出保存到文件示例 =====[/bold yellow]\n")

    # 自定义回调处理器，将输出同时写入文件
    class FileWriterCallback(BaseCallbackHandler):
        def __init__(self, file_path):
            self.file = open(file_path, "w", encoding="utf-8")
            self.tokens_received = 0

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.tokens_received += 1
            print(token, end="", flush=True)  # 同时在控制台显示
            self.file.write(token)
            self.file.flush()  # 确保实时写入文件

        def on_llm_end(self, response: LLMResult, **kwargs) -> None:
            console.print(f"\n[green]生成完成! 接收了{self.tokens_received}个token，已保存到文件[/green]")
            self.file.close()

    # 创建输出文件
    output_file = "streaming_output.md"
    file_callback = FileWriterCallback(output_file)

    # 初始化模型
    chat = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True,
        callbacks=[file_callback]
    )

    # 发送消息
    console.print(f"[cyan]流式输出将同时显示在控制台和保存到文件 {output_file}:[/cyan]\n")
    messages = [
        SystemMessage(content="你是一位知识渊博的科学顾问。"),
        HumanMessage(content="请详细介绍太阳系中的行星，包括它们的基本特征和有趣事实。")
    ]

    # 调用模型
    chat.invoke(messages)

    console.print(f"\n[green]输出已保存到 {output_file}[/green]")


class WebSocketSimulator:
    """模拟WebSocket连接的简单类，用于演示如何在Web应用中处理流式响应"""

    def __init__(self):
        self.messages = []

    async def send(self, message):
        """模拟向客户端发送消息"""
        self.messages.append(message)
        # 在实际应用中，这里会发送消息到WebSocket连接
        console.print(f"[blue]>> 向客户端发送: {message[:30]}...(已截断)[/blue]")

    def get_all_messages(self):
        """获取所有发送的消息"""
        return self.messages


class WebStreamingHandler(BaseCallbackHandler):
    """为Web应用设计的流式处理回调"""

    def __init__(self, websocket):
        self.websocket = websocket
        self.full_text = ""
        self.chunk_count = 0

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """收到新token时向WebSocket发送更新"""
        self.chunk_count += 1
        self.full_text += token

        # 每N个token或遇到段落结束时发送更新
        if self.chunk_count % 10 == 0 or token in ["\n", ".", "!", "?"]:
            # 在实际应用中，可能需要对内容进行封装，如添加JSON格式
            message = {
                "type": "chunk",
                "content": token,
                "full_text": self.full_text
            }
            await self.websocket.send(str(message))

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """生成结束时发送完成信号"""
        message = {
            "type": "complete",
            "content": self.full_text
        }
        await self.websocket.send(str(message))


async def web_integration_example():
    """演示如何在Web应用中集成流式输出"""
    console.print("\n[bold yellow]===== Web应用流式输出集成示例 =====[/bold yellow]\n")

    # 创建模拟的WebSocket连接
    ws = WebSocketSimulator()

    # 创建处理器
    web_handler = WebStreamingHandler(ws)

    # 初始化聊天模型
    chat = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True,
        callbacks=[web_handler]
    )

    # 模拟从Web客户端接收到的消息
    client_message = "请用中文解释量子计算的基本原理"

    console.print(f"[cyan]模拟Web客户端请求: '{client_message}'[/cyan]\n")
    console.print("[cyan]开始流式生成并通过WebSocket发送到客户端...[/cyan]\n")

    # 处理请求
    messages = [
        SystemMessage(content="你是一位非常有耐心的科普专家，善于通俗易懂地解释复杂概念。"),
        HumanMessage(content=client_message)
    ]

    # 异步调用模型，结果将通过WebSocket回调发送
    await chat.ainvoke(messages)

    # 显示结果摘要
    console.print(f"\n[green]流式传输完成，总共发送了 {len(ws.get_all_messages())} 个消息片段到客户端[/green]")


async def main():
    """主函数"""
    console.print("[bold green]LangChain 聊天模型流式输出演示[/bold green]")
    console.print("本示例展示了在LangChain中使用和处理流式输出的不同方法\n")

    try:
        # 基础示例
        basic_streaming_example()

        # 等待用户继续
        input("\n按回车键继续下一个示例...\n")

        # 高级示例
        await advanced_streaming_example()

        # 等待用户继续
        input("\n按回车键继续下一个示例...\n")

        # 多模型比较
        await multi_model_streaming_comparison()

        # 等待用户继续
        input("\n按回车键继续下一个示例...\n")

        # 保存到文件的示例
        streaming_to_file_example()

        # 等待用户继续
        input("\n按回车键继续最后一个示例...\n")

        # Web集成示例
        await web_integration_example()

    except KeyboardInterrupt:
        console.print("\n[yellow]演示被用户中断[/yellow]")
    except Exception as e:
        console.print(f"[red]发生错误: {str(e)}[/red]")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
