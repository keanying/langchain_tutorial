#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain智能体(Agent)使用示例 - 多智能体协作模式

本模块展示了如何在LangChain中创建和使用多个智能体进行协作，包括：
1. 研究员-编辑者协作模式
2. 辩论模式
3. 团队合作模式
4. 工作流智能体编排

注意：运行这些示例前，请确保已设置相应的API密钥环境变量
"""

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ===============================
# 第1部分：研究员-编辑者协作模式
# ===============================
def researcher_editor_example():
    """研究员-编辑者协作模式示例
    研究员负责收集信息，编辑者负责整理和润色内容
    """
    print("=== 研究员-编辑者协作模式示例 ===")
    
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentType, Tool, initialize_agent
    from langchain_community.tools import DuckDuckGoSearchRun
    
    # 检查是否有OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("未设置OPENAI_API_KEY，跳过本节示例")
        print("要运行智能体示例，请设置OPENAI_API_KEY环境变量")
        return
    
    # 创建搜索工具
    search_tool = DuckDuckGoSearchRun()
    
    # 创建研究员智能体 - 温度较低，注重准确性
    researcher_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    researcher_tools = [search_tool]
    researcher_agent = initialize_agent(
        tools=researcher_tools,
        llm=researcher_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # 创建研究工具 - 使用研究员智能体获取信息
    def research_tool(query: str) -> str:
        """使用研究员智能体收集特定主题的信息"""
        try:
            return researcher_agent.run(f"详细研究以下主题并提供事实、数据和关键信息: {query}. 注意确保信息的准确性和及时性。")
        except Exception as e:
            return f"研究过程中出错：{str(e)}. 请提供部分可用信息。"
    
    # 将研究员包装为工具
    researcher_wrapper = Tool(
        name="ResearchTool",
        description="当需要深入研究某个主题、收集事实和数据时使用。提供详细的搜索查询。",
        func=research_tool
    )
    
    # 创建编辑者智能体 - 温度较高，注重创造性表达
    editor_llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    editor_tools = [researcher_wrapper]
    editor_agent = initialize_agent(
        tools=editor_tools,
        llm=editor_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # 测试研究员-编辑者协作
    try:
        print("\n执行任务：'撰写一篇关于元宇宙(Metaverse)发展现状的短文，包含最新趋势和主要平台'")
        result = editor_agent.run(
            "撰写一篇关于元宇宙(Metaverse)发展现状的短文，包含最新趋势和主要平台。" + 
            "确保内容事实准确，同时行文流畅有趣，适合普通读者阅读。文章长度约500字。"
        )
        print("\n=== 最终输出 ===")
        print(result)
        
    except Exception as e:
        print(f"执行过程中出错: {e}")


# ===============================
# 第2部分：辩论模式
# ===============================
def debate_agents_example():
    """辩论模式示例
    两个智能体就一个话题进行辩论，可以相互回应对方的观点
    """
    print("\n=== 辩论模式示例 ===")
    
    from langchain_openai import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage, AIMessage
    
    # 检查是否有OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("未设置OPENAI_API_KEY，跳过本节示例")
        return
    
    try:
        # 创建辩手A - 支持观点的智能体
        supporter_llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        
        # 创建辩手B - 反对观点的智能体
        opponent_llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        
        # 辩论主题
        debate_topic = "人工智能是否会在未来十年内取代大部分人类工作？"
        
        print(f"\n辩论主题: {debate_topic}\n")
        
        # 设置辩手的系统角色提示
        supporter_system = SystemMessage(
            content="你是一位擅长辩论的专家，在接下来的讨论中，你将支持观点。请提供有力的论据、数据和逻辑推理来支持你的立场。" +
                   "保持专业、礼貌，但要坚定地捍卫你的立场。每次回应限制在200字以内。"
        )
        
        opponent_system = SystemMessage(
            content="你是一位擅长辩论的专家，在接下来的讨论中，你将反对观点。请提供有力的论据、数据和逻辑推理来支持你的立场。" +
                   "保持专业、礼貌，但要坚定地捍卫你的立场。每次回应限制在200字以内。"
        )
        
        # 初始化辩论记录
        debate_history = []
        
        # 第一轮：开场陈述
        print("第1轮：开场陈述\n")
        
        # 支持方开场
        supporter_first_input = HumanMessage(content=f"请就以下主题提供支持方的开场陈述：{debate_topic}")
        supporter_response = supporter_llm([supporter_system, supporter_first_input])
        debate_history.append({"role": "支持方", "content": supporter_response.content})
        print(f"支持方: {supporter_response.content}\n")
        
        # 反对方开场
        opponent_first_input = HumanMessage(content=f"请就以下主题提供反对方的开场陈述：{debate_topic}")
        opponent_response = opponent_llm([opponent_system, opponent_first_input])
        debate_history.append({"role": "反对方", "content": opponent_response.content})
        print(f"反对方: {opponent_response.content}\n")
        
        # 后续几轮辩论
        for round_num in range(2, 4):  # 再进行2轮辩论
            print(f"第{round_num}轮：回应对方观点\n")
            
            # 构建辩论历史消息
            debate_context = "\n".join([
                f"{entry['role']}: {entry['content']}"
                for entry in debate_history
            ])
            
            # 支持方回应
            supporter_prompt = HumanMessage(
                content=f"以下是目前的辩论历史：\n{debate_context}\n\n请针对反对方的最新论点进行反驳，并继续支持你的观点。"
            )
            supporter_response = supporter_llm([supporter_system, supporter_prompt])
            debate_history.append({"role": "支持方", "content": supporter_response.content})
            print(f"支持方: {supporter_response.content}\n")
            
            # 反对方回应
            opponent_prompt = HumanMessage(
                content=f"以下是目前的辩论历史：\n{debate_context}\n\n请针对支持方的最新论点进行反驳，并继续支持你的观点。"
            )
            opponent_response = opponent_llm([opponent_system, opponent_prompt])
            debate_history.append({"role": "反对方", "content": opponent_response.content})
            print(f"反对方: {opponent_response.content}\n")
        
        # 总结辩论
        print("辩论总结\n")
        
        summary_llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        summary_system = SystemMessage(
            content="你是一位公正的辩论评审，你的任务是总结双方的辩论要点，指出各自的强弱之处，但不需要宣布胜负。请保持中立。"
        )
        
        debate_context = "\n".join([
            f"{entry['role']}: {entry['content']}"
            for entry in debate_history
        ])
        
        summary_prompt = HumanMessage(
            content=f"辩论主题：{debate_topic}\n\n辩论内容：\n{debate_context}\n\n请总结这场辩论的关键点，评价双方的论证强度、逻辑性和说服力。"
        )
        summary_response = summary_llm([summary_system, summary_prompt])
        print(f"评审总结: {summary_response.content}")
    
    except Exception as e:
        print(f"辩论过程中出错: {e}")


# ===============================
# 第3部分：团队合作模式
# ===============================
def team_agents_example():
    """团队合作模式示例
    多个专家型智能体协作解决问题，每个智能体有自己的专长
    """
    print("\n=== 团队合作模式示例 ===")
    
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentType, Tool, initialize_agent, load_tools
    from langchain_community.tools import DuckDuckGoSearchRun
    from pydantic import BaseModel, Field
    
    # 检查是否有OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("未设置OPENAI_API_KEY，跳过本节示例")
        return
    
    try:
        # 创建基础LLM
        base_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        
        # 1. 创建数据分析师智能体
        # 添加计算工具
        math_tools = load_tools(["llm-math"], llm=base_llm)
        
        # 创建数据分析工具函数
        def analyze_data(data_description: str) -> str:
            """分析描述的数据并提供见解"""
            try:
                # 这里我们使用LLM来模拟数据分析
                # 在实际应用中，这里可能会使用pandas、numpy等进行真实分析
                analysis_prompt = f"作为数据分析师，分析以下数据并提供见解：\n{data_description}"
                result = base_llm.invoke(analysis_prompt)
                return result.content
            except Exception as e:
                return f"数据分析过程中出错：{str(e)}"
        
        # 创建数据分析工具
        analysis_tool = Tool(
            name="DataAnalysis",
            description="当需要分析数据、计算统计值或获取数据见解时使用。提供数据的详细描述。",
            func=analyze_data
        )
        
        # 创建数据分析师智能体
        analyst_agent = initialize_agent(
            tools=math_tools + [analysis_tool],
            llm=base_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # 包装数据分析师为工具
        analyst_tool = Tool(
            name="DataAnalyst",
            description="当需要进行任何形式的数据分析、统计计算或解释数据时使用。提供详细的分析请求。",
            func=lambda query: analyst_agent.run(query)
        )
        
        # 2. 创建研究员智能体
        search_tool = DuckDuckGoSearchRun()
        
        researcher_agent = initialize_agent(
            tools=[search_tool],
            llm=base_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # 包装研究员为工具
        researcher_tool = Tool(
            name="Researcher",
            description="当需要查找最新信息、事实、数据或网络内容时使用。提供详细的搜索查询。",
            func=lambda query: researcher_agent.run(query)
        )
        
        # 3. 创建内容编写者智能体
        writer_llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")  # 更高的温度使文本更有创造性
        
        writer_agent = initialize_agent(
            tools=[],  # 内容编写者不需要特别的工具
            llm=writer_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # 包装内容编写者为工具
        writer_tool = Tool(
            name="ContentWriter",
            description="当需要创建高质量文本内容、改写文本或优化写作时使用。提供详细的内容需求和写作风格指导。",
            func=lambda query: writer_agent.run(query)
        )
        
        # 4. 创建项目经理智能体 - 协调其他智能体
        manager_tools = [analyst_tool, researcher_tool, writer_tool]
        
        manager_llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        manager_agent = initialize_agent(
            tools=manager_tools,
            llm=manager_llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # 测试团队合作
        print("\n执行任务：'创建一份关于2023年全球电动汽车市场的简要报告，包括销量数据、主要趋势和未来预测'")
        
        result = manager_agent.run(
            "我需要一份关于2023年全球电动汽车市场的简要报告。请协调团队成员完成以下工作：\n" +
            "1. 研究最新的电动汽车市场数据和趋势\n" +
            "2. 分析销量数据和市场份额\n" +
            "3. 撰写一份专业、信息丰富但简洁的报告，包括数据分析、主要趋势和未来一年的预测\n" +
            "报告篇幅约500字，面向汽车行业专业人士。"
        )
        
        print("\n=== 最终报告 ===")
        print(result)
        
    except Exception as e:
        print(f"团队协作过程中出错: {e}")


# ===============================
# 主函数：运行各种示例
# ===============================
def main():
    """主函数，运行各种多智能体协作示例"""
    print("LangChain多智能体协作示例程序\n")
    print("注意：运行前请确保已经设置好相关API密钥环境变量\n")
    
    # 运行研究员-编辑者协作模式示例
    researcher_editor_example()
    
    # 运行辩论模式示例
    debate_agents_example()
    
    # 运行团队合作模式示例
    team_agents_example()


# 程序入口
if __name__ == "__main__":
    main()