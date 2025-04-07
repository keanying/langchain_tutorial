#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
聊天模型自定义与微调示例：展示如何在LangChain中自定义和优化聊天模型
包括模型参数配置、提示词优化、模型包装和输出处理等示例
"""

import os
import json
from typing import Any, Dict, List, Optional, Union
from pprint import pprint
from rich.console import Console
from rich.table import Table

# 导入LangChain核心组件
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

# 导入聊天模型
from langchain_openai import ChatOpenAI

# from langchain_anthropic import ChatAnthropic
# from langchain_qianfan import ChatQianfan
# from langchain_dashscope import ChatDashscope

# 创建Rich控制台用于美化输出
console = Console()


#####################################################################
# 第1部分: 基础模型配置和参数调优
#####################################################################

def model_parameters_demo():
    """展示如何配置模型参数以优化输出"""
    console.print("\n[bold yellow]===== 模型参数配置示例 =====[/bold yellow]\n")

    # 创建不同参数配置的模型实例
    models = {
        "默认配置": ChatOpenAI(
            model="gpt-3.5-turbo",
        ),
        "低随机性": ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,  # 接近0的温度使输出更确定、可预测
        ),
        "高随机性": ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=1.0,  # 较高的温度增加随机性和创造性
        ),
        "限制输出长度": ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=50,  # 限制输出令牌数量
        ),
        "惩罚重复": ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            frequency_penalty=1.0,  # 降低重复内容的可能性
        ),
    }

    # 创建一个通用测试消息
    creative_message = HumanMessage(content="请为我写一首关于机器学习的短诗。")
    factual_message = HumanMessage(content="请解释什么是梯度下降算法。")

    # 测试表格
    table = Table(title="不同参数配置的模型输出比较")
    table.add_column("参数配置", style="cyan")
    table.add_column("创意任务输出", style="green")
    table.add_column("事实任务输出", style="yellow")

    # 为每个模型配置测试输出
    for name, model in models.items():
        console.print(f"[cyan]测试 {name} 配置...[/cyan]")

        # 创意任务
        creative_response = model.invoke([creative_message])

        # 事实任务
        factual_response = model.invoke([factual_message])

        # 添加到表格
        table.add_row(
            name,
            creative_response.content[:100] + "...",  # 截断较长的回复
            factual_response.content[:100] + "..."
        )

    # 显示比较表格
    console.print(table)

    # 提供参数选择指导
    console.print("\n[bold green]模型参数选择指导:[/bold green]")
    console.print("1. [cyan]temperature[/cyan]: 控制随机性，事实性任务使用低值(0-0.3)，创意任务使用高值(0.7-1.0)")
    console.print("2. [cyan]top_p[/cyan]: 控制输出多样性的替代方法，通常设置为0.1-0.9")
    console.print("3. [cyan]max_tokens[/cyan]: 限制输出长度，避免过长的回复")
    console.print("4. [cyan]presence_penalty[/cyan]: 增加模型讨论新主题的可能性(0-2.0)")
    console.print("5. [cyan]frequency_penalty[/cyan]: 减少重复，增加输出多样性(0-2.0)")


#####################################################################
# 第2部分: 提示词模板和系统消息优化
#####################################################################

def prompt_optimization_demo():
    """展示如何优化提示词模板和系统消息"""
    console.print("\n[bold yellow]===== 提示词优化示例 =====[/bold yellow]\n")

    # 创建基础模型
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 测试场景：生成产品描述
    product_info = {
        "name": "智能健康手环X1",
        "features": ["心率监测", "睡眠分析", "运动追踪", "防水设计"],
        "price": 299,
        "target_audience": "健身爱好者和健康意识较强的年轻人"
    }

    # 1. 基础提示词
    basic_prompt = ChatPromptTemplate.from_messages([
        ("human", "请为这个产品写一段营销描述：{product_name}，它的功能包括{features}，售价{price}元，目标受众是{audience}。")
    ])

    # 2. 优化的系统消息
    optimized_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位经验丰富的营销文案专家，擅长用吸引人且富有感染力的语言撰写产品描述。"
                   "请确保描述简洁有力，突出产品的独特卖点，使用积极正面的语言，"
                   "并巧妙融入呼吁行动的词句。限制在100-150字之间。"),
        ("human", "请为这个产品写一段营销描述：{product_name}，它的功能包括{features}，售价{price}元，目标受众是{audience}。")
    ])

    # 3. 结构化输出提示词
    structured_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的营销文案撰写专家。请根据提供的产品信息，创建精确的营销内容。"
                   "输出必须严格按照以下JSON格式："
                   "{\"slogan\": \"简短吸引人的口号\", \"description\": \"产品主要描述(80-100字)\", "
                   "\"key_benefits\": [\"3-5个核心优势，每个10字以内\"], \"call_to_action\": \"行动号召语\"}"),
        ("human", "产品名称：{product_name}\n功能：{features}\n价格：{price}元\n目标受众：{audience}")
    ])

    # 测试不同提示词
    console.print("[bold cyan]测试基础提示词...[/bold cyan]")
    basic_chain = basic_prompt | model
    basic_result = basic_chain.invoke({
        "product_name": product_info["name"],
        "features": ", ".join(product_info["features"]),
        "price": product_info["price"],
        "audience": product_info["target_audience"]
    })

    console.print("[bold cyan]测试优化的系统消息...[/bold cyan]")
    optimized_chain = optimized_prompt | model
    optimized_result = optimized_chain.invoke({
        "product_name": product_info["name"],
        "features": ", ".join(product_info["features"]),
        "price": product_info["price"],
        "audience": product_info["target_audience"]
    })

    console.print("[bold cyan]测试结构化输出提示词...[/bold cyan]")
    structured_chain = structured_prompt | model
    structured_result = structured_chain.invoke({
        "product_name": product_info["name"],
        "features": ", ".join(product_info["features"]),
        "price": product_info["price"],
        "audience": product_info["target_audience"]
    })

    # 显示比较结果
    console.print("\n[bold green]基础提示词结果:[/bold green]")
    console.print(basic_result.content)

    console.print("\n[bold green]优化的系统消息结果:[/bold green]")
    console.print(optimized_result.content)

    console.print("\n[bold green]结构化输出提示词结果:[/bold green]")
    try:
        # 尝试解析为JSON展示
        structured_content = json.loads(structured_result.content)
        console.print(structured_content)
    except json.JSONDecodeError:
        # 如果无法解析为JSON，直接显示原文
        console.print(structured_result.content)

    # 总结提示词优化技巧
    console.print("\n[bold green]提示词优化技巧:[/bold green]")
    console.print("1. [cyan]使用详细的系统消息[/cyan]: 明确角色、输出样式、限制条件等")
    console.print("2. [cyan]结构化输入信息[/cyan]: 使用清晰的格式和分隔符组织输入信息")
    console.print("3. [cyan]指定输出格式[/cyan]: 明确要求特定的输出结构，如JSON")
    console.print("4. [cyan]提供示例[/cyan]: 在复杂任务中加入示例，展示预期输出")
    console.print("5. [cyan]分步指导[/cyan]: 引导模型按步骤思考和回答")


#####################################################################
# 第3部分: 输出解析和转换
#####################################################################

class ProductRecommendation(BaseModel):
    """产品推荐的结构化输出模型"""
    product_name: str = Field(description="推荐产品的名称")
    price_range: str = Field(description="价格范围，如'100-300元'")
    key_features: List[str] = Field(description="产品的主要特点，列表形式")
    reason: str = Field(description="推荐该产品的理由")
    suitability_score: int = Field(description="适合度评分，1-10分")


def output_parsing_demo():
    """展示如何解析和转换模型输出"""
    console.print("\n[bold yellow]===== 输出解析示例 =====[/bold yellow]\n")

    # 创建基础模型
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    # ---------- 1. 基本JSON解析 ----------#
    console.print("[bold cyan]1. 基本JSON输出解析...[/bold cyan]")

    # 创建要求返回JSON的提示词
    json_prompt = ChatPromptTemplate.from_messages([
        ("system", "以JSON格式返回回答，不要包含任何其他文本。"),
        ("human", "请提供三种流行的编程语言及其主要特点。")
    ])

    # 创建解析链
    json_chain = json_prompt | model

    # 执行并尝试解析JSON
    response = json_chain.invoke({})
    console.print("原始响应:")
    console.print(response.content)

    try:
        parsed_data = json.loads(response.content)
        console.print("\n解析后的数据:")
        console.print(parsed_data)
    except json.JSONDecodeError as e:
        console.print(f"[red]JSON解析失败: {e}[/red]")

    # ---------- 2. Pydantic输出解析器 ----------#
    console.print("\n[bold cyan]2. Pydantic输出解析器示例...[/bold cyan]")

    # 创建Pydantic解析器
    parser = PydanticOutputParser(pydantic_object=ProductRecommendation)

    # 创建包含格式指令的提示词
    pydantic_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位产品推荐专家。根据用户需求，推荐最合适的产品。\n" +
         "请严格按照以下JSON格式返回结果:\n{format_instructions}\n" +
         "确保输出是有效的JSON格式，不包含任何额外的文本。"),
        ("human", "我想购买一款适合初学者的健身智能手环，预算在500元以内。")
    ])

    # 将格式说明传递给提示词
    format_chain = pydantic_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # 创建完整的链
    recommendation_chain = format_chain | model | parser

    # 执行并获取结构化输出
    try:
        result = recommendation_chain.invoke({})
        console.print("结构化解析结果:")
        console.print(json.loads(result.json()))
    except Exception as e:
        console.print(f"[red]解析失败: {e}[/red]")

    # ---------- 3. 自定义输出处理函数 ----------#
    console.print("\n[bold cyan]3. 自定义输出处理函数示例...[/bold cyan]")

    def extract_key_points(text: str) -> List[str]:
        """从文本中提取要点"""
        # 在实际应用中，这可能是一个更复杂的处理函数
        lines = text.strip().split("\n")
        points = [line.strip() for line in lines if line.strip().startswith("-") or line.strip().startswith("*")]
        return points if points else ["未找到要点，原文：" + text[:100] + "..."]

    # 创建提示词
    bullet_prompt = ChatPromptTemplate.from_messages([
        ("system", "请提供信息，并使用破折号(-)或星号(*)作为每个要点的开头。"),
        ("human", "解释云计算的主要优势。")
    ])

    # 创建包含自定义函数的链
    extraction_chain = bullet_prompt | model | extract_key_points

    # 执行并显示结果
    points = extraction_chain.invoke({})
    console.print("提取的要点:")
    for i, point in enumerate(points, 1):
        console.print(f"{i}. {point}")

    # 输出解析技巧总结
    console.print("\n[bold green]输出解析技巧:[/bold green]")
    console.print("1. [cyan]明确的输出格式指令[/cyan]: 在系统消息中明确指定所需的输出格式")
    console.print("2. [cyan]使用模型自身的格式能力[/cyan]: 较新的模型(如GPT-4)有很强的格式遵循能力")
    console.print("3. [cyan]Pydantic模型[/cyan]: 用于复杂结构化数据的验证和解析")
    console.print("4. [cyan]自定义处理函数[/cyan]: 处理特殊格式或从非结构化文本中提取信息")
    console.print("5. [cyan]错误处理[/cyan]: 实现鲁棒的解析器，处理可能的格式错误")


#####################################################################
# 第4部分: 模型包装和自定义行为
#####################################################################

class EnhancedChatModel:
    """增强的聊天模型包装器，添加自定义行为"""

    def __init__(self, base_model: BaseChatModel):
        """初始化增强模型"""
        self.base_model = base_model
        self.conversation_history = []
        self.system_directive = "你是一位有用的AI助手。"

    def set_system_directive(self, directive: str):
        """设置系统指令"""
        self.system_directive = directive

    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
        return "对话历史已清除"

    def summarize_conversation(self):
        """总结当前对话"""
        if not self.conversation_history:
            return "没有对话历史可总结"

        # 创建总结提示词
        summary_messages = [
            SystemMessage(content="请简要总结以下对话的主要内容和关键点："),
            *self.conversation_history
        ]

        # 获取总结
        summary_response = self.base_model.invoke(summary_messages)
        return summary_response.content

    def add_context(self, context: str):
        """添加上下文信息"""
        system_message = SystemMessage(content=f"{self.system_directive}\n\n额外上下文信息: {context}")
        if self.conversation_history and isinstance(self.conversation_history[0], SystemMessage):
            self.conversation_history[0] = system_message
        else:
            self.conversation_history.insert(0, system_message)
        return "已添加上下文信息"

    def chat(self, message: str):
        """处理用户消息并返回回复"""
        # 检查是否有特殊指令
        if message.lower() == "/clear":
            return self.clear_history()
        elif message.lower() == "/summary":
            return self.summarize_conversation()
        elif message.lower().startswith("/context "):
            context = message[9:].strip()
            return self.add_context(context)

        # 添加用户消息到历史
        user_message = HumanMessage(content=message)

        # 准备消息列表
        messages = self.conversation_history.copy()
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages.insert(0, SystemMessage(content=self.system_directive))
        messages.append(user_message)

        # 获取模型回复
        response = self.base_model.invoke(messages)

        # 更新对话历史
        self.conversation_history.append(user_message)
        self.conversation_history.append(response)

        # 保持历史长度合理
        if len(self.conversation_history) > 20:  # 保留最近的10轮对话
            # 保留第一条系统消息
            if isinstance(self.conversation_history[0], SystemMessage):
                self.conversation_history = [
                    self.conversation_history[0],
                    *self.conversation_history[-19:]
                ]
            else:
                self.conversation_history = self.conversation_history[-20:]

        return response.content


def model_customization_demo():
    """展示如何包装和自定义模型行为"""
    console.print("\n[bold yellow]===== 模型包装和自定义行为示例 =====[/bold yellow]\n")

    # 创建基础模型
    base_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 创建增强模型
    enhanced_model = EnhancedChatModel(base_model)

    # 设置自定义系统指令
    enhanced_model.set_system_directive(
        "你是一位友好的AI助手，擅长提供简洁、准确的回答。" +
        "你的回答应当简明扼要，避免冗长。" +
        "如果不确定答案，请诚实承认而不是猜测。"
    )

    # 模拟对话
    console.print("[bold green]模拟对话开始:[/bold green]")

    responses = []

    # 基本对话
    q1 = "什么是机器学习？"
    console.print(f"[bold blue]用户:[/bold blue] {q1}")
    r1 = enhanced_model.chat(q1)
    console.print(f"[bold green]AI:[/bold green] {r1}\n")
    responses.append((q1, r1))

    # 带有上下文的对话
    context_cmd = "/context 用户是一位计算机科学初学者，需要简单易懂的解释。"
    console.print(f"[bold blue]用户:[/bold blue] {context_cmd}")
    r_context = enhanced_model.chat(context_cmd)
    console.print(f"[bold green]系统:[/bold green] {r_context}\n")

    # 后续对话
    q2 = "神经网络是如何工作的？"
    console.print(f"[bold blue]用户:[/bold blue] {q2}")
    r2 = enhanced_model.chat(q2)
    console.print(f"[bold green]AI:[/bold green] {r2}\n")
    responses.append((q2, r2))

    # 获取对话总结
    summary_cmd = "/summary"
    console.print(f"[bold blue]用户:[/bold blue] {summary_cmd}")
    summary = enhanced_model.chat(summary_cmd)
    console.print(f"[bold green]系统:[/bold green] {summary}\n")

    # 清除历史
    clear_cmd = "/clear"
    console.print(f"[bold blue]用户:[/bold blue] {clear_cmd}")
    clear_msg = enhanced_model.chat(clear_cmd)
    console.print(f"[bold green]系统:[/bold green] {clear_msg}\n")

    # 自定义模型优势总结
    console.print("\n[bold green]自定义模型的优势:[/bold green]")
    console.print("1. [cyan]特殊指令处理[/cyan]: 添加自定义命令，如清除历史、添加上下文等")
    console.print("2. [cyan]会话管理[/cyan]: 智能管理对话历史和上下文窗口")
    console.print("3. [cyan]用户适应性[/cyan]: 根据用户需求动态调整系统指令")
    console.print("4. [cyan]自定义功能[/cyan]: 添加总结、翻译等特殊功能")
    console.print("5. [cyan]异常处理[/cyan]: 可添加更强大的错误处理和恢复机制")


def main():
    """主函数"""
    console.print("[bold green]LangChain 聊天模型自定义与优化演示[/bold green]")
    console.print("本示例展示了在LangChain中自定义和优化聊天模型的不同方法\n")

    try:
        # 检查API密钥
        if not os.environ.get("OPENAI_API_KEY"):
            console.print("[yellow]警告: 未设置OPENAI_API_KEY环境变量，使用演示模式[/yellow]")
            os.environ["OPENAI_API_KEY"] = "demo-key"
            # 在真实运行中，这里应该引导用户设置正确的API密钥

        # 模型参数演示
        model_parameters_demo()

        # 等待用户继续
        input("\n按回车键继续下一个示例...\n")

        # 提示词优化演示
        prompt_optimization_demo()

        # 等待用户继续
        input("\n按回车键继续下一个示例...\n")

        # 输出解析演示
        output_parsing_demo()

        # 等待用户继续
        input("\n按回车键继续最后一个示例...\n")

        # 模型自定义演示
        model_customization_demo()

    except KeyboardInterrupt:
        console.print("\n[yellow]演示被用户中断[/yellow]")
    except Exception as e:
        console.print(f"[red]发生错误: {str(e)}[/red]")


if __name__ == "__main__":
    main()
