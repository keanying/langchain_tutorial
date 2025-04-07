"""
LangGraph多智能体系统示例 - 协作研究与写作系统

本示例展示如何使用LangGraph构建一个多智能体协作系统，
系统中的智能体具有不同专业角色，它们协同工作完成研究和文章写作任务。
"""

import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 设置环境变量（生产环境中应通过其他安全方式设置）
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # 请替换为您的密钥

# 定义系统状态类型
class ResearchState(TypedDict):
    """研究项目的状态，包含各阶段工作成果和元数据"""
    topic: str  # 研究主题
    research_results: Optional[List[str]]  # 研究员收集的信息
    outline: Optional[str]  # 规划师生成的大纲
    draft: Optional[str]  # 作者撰写的初稿
    review_comments: Optional[List[str]]  # 审阅者的评论
    final_article: Optional[str]  # 最终文章
    current_agent: str  # 当前活跃的智能体
    conversation_history: List[Dict]  # 智能体之间的对话历史
    status: str  # 当前状态：'in_progress'或'completed'

# 定义智能体角色

def researcher_agent(state: ResearchState) -> ResearchState:
    """研究员智能体：负责收集与主题相关的信息"""
    print("🔍 研究员正在工作...")
    
    topic = state["topic"]
    history = state.get("conversation_history", [])
    
    # 创建研究员提示
    system_prompt = """你是一位专业研究员。你的任务是收集关于特定主题的重要信息和事实。
    提供最新、最相关的信息，并确保涵盖不同角度和观点。
    组织信息时要条理清晰，使用要点列表格式。每个要点应该简洁但信息丰富。"""
    
    # 准备消息列表
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"请收集关于"{topic}"的重要信息。提供3-5个关键要点，每个要点100-150字。")
    ]
    
    # 添加历史上下文（如果有的话）
    for msg in history:
        if msg.get("role") == "planner" and "outline" in msg.get("content", ""):
            messages.append(HumanMessage(content=f"参考这个大纲进行更有针对性的研究：{msg['content']}"))
    
    # 调用模型
    model = ChatOpenAI(model="gpt-4", temperature=0.7)
    response = model.invoke(messages)
    
    # 处理研究结果
    research_points = response.content.split("\n")
    research_points = [point for point in research_points if point.strip()]
    
    # 更新状态
    return {
        "research_results": research_points,
        "current_agent": "planner",
        "conversation_history": history + [{"role": "researcher", "content": response.content}]
    }

def planner_agent(state: ResearchState) -> ResearchState:
    """规划师智能体：基于研究结果制定文章大纲"""
    print("📝 规划师正在工作...")
    
    topic = state["topic"]
    research_results = state.get("research_results", [])
    history = state.get("conversation_history", [])
    
    # 如果没有研究结果，要求研究员先工作
    if not research_results:
        return {
            "current_agent": "researcher", 
            "conversation_history": history + [{"role": "planner", "content": "需要更多研究材料才能创建大纲。"}]
        }
    
    # 创建规划师提示
    system_prompt = """你是一位专业内容规划师。你的任务是基于提供的研究材料，创建一个结构清晰的文章大纲。
    大纲应该包括引言、主要章节（带小节）和结论。
    确保大纲逻辑连贯、内容全面，并突出最重要的信息。"""
    
    # 准备消息列表
    research_str = "\n".join(research_results)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"基于以下研究材料，为主题"{topic}"创建一个详细的文章大纲：\n\n{research_str}")
    ]
    
    # 调用模型
    model = ChatOpenAI(model="gpt-4", temperature=0.7)
    response = model.invoke(messages)
    
    # 更新状态
    return {
        "outline": response.content,
        "current_agent": "writer",
        "conversation_history": history + [{"role": "planner", "content": response.content}]
    }

def writer_agent(state: ResearchState) -> ResearchState:
    """作者智能体：根据大纲和研究材料撰写初稿"""
    print("✍️ 作者正在工作...")
    
    topic = state["topic"]
    research_results = state.get("research_results", [])
    outline = state.get("outline", "")
    history = state.get("conversation_history", [])
    
    # 如果没有大纲，请规划师先工作
    if not outline:
        return {
            "current_agent": "planner",
            "conversation_history": history + [{"role": "writer", "content": "需要大纲才能开始写作。"}]
        }
    
    # 创建作者提示
    system_prompt = """你是一位专业文章作者。你的任务是根据提供的研究材料和大纲，撰写一篇连贯、信息丰富的文章。
    文章应该语言流畅、结构清晰，并准确地传达研究发现。
    针对一般读者，使用易于理解的语言，同时保持专业性。文章应该有吸引力的开头，充实的主体内容，以及有见解的结论。"""
    
    # 准备消息列表
    research_str = "\n".join(research_results)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"根据以下研究材料和大纲，撰写一篇关于"{topic}"的文章：\n\n研究材料：\n{research_str}\n\n大纲：\n{outline}")
    ]
    
    # 调用模型
    model = ChatOpenAI(model="gpt-4", temperature=0.7)
    response = model.invoke(messages)
    
    # 更新状态
    return {
        "draft": response.content,
        "current_agent": "reviewer",
        "conversation_history": history + [{"role": "writer", "content": "已完成初稿撰写。"}]
    }

def reviewer_agent(state: ResearchState) -> ResearchState:
    """审阅者智能体：评估文章并提供改进建议"""
    print("🔍 审阅者正在工作...")
    
    topic = state["topic"]
    draft = state.get("draft", "")
    history = state.get("conversation_history", [])
    
    # 如果没有初稿，请作者先工作
    if not draft:
        return {
            "current_agent": "writer",
            "conversation_history": history + [{"role": "reviewer", "content": "需要初稿才能进行审阅。"}]
        }
    
    # 创建审阅者提示
    system_prompt = """你是一位专业内容审阅者。你的任务是评估文章的质量，并提供具体的改进建议。
    关注以下几点：
    1. 内容准确性和完整性
    2. 结构和逻辑流程
    3. 语言表达和清晰度
    4. 论点的支持和证据
    
    提供3-5条具体的改进建议，并指出文章的优点。"""
    
    # 准备消息列表
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"请审阅以下关于"{topic}"的文章，并提供改进建议：\n\n{draft}")
    ]
    
    # 调用模型
    model = ChatOpenAI(model="gpt-4", temperature=0.7)
    response = model.invoke(messages)
    
    # 提取评论
    review_comments = response.content.split("\n")
    review_comments = [comment for comment in review_comments if comment.strip()]
    
    # 更新状态
    return {
        "review_comments": review_comments,
        "current_agent": "editor",
        "conversation_history": history + [{"role": "reviewer", "content": response.content}]
    }

def editor_agent(state: ResearchState) -> ResearchState:
    """编辑智能体：根据审阅意见完善文章，生成最终版本"""
    print("✏️ 编辑正在工作...")
    
    topic = state["topic"]
    draft = state.get("draft", "")
    review_comments = state.get("review_comments", [])
    history = state.get("conversation_history", [])
    
    # 如果没有审阅意见，请审阅者先工作
    if not review_comments:
        return {
            "current_agent": "reviewer",
            "conversation_history": history + [{"role": "editor", "content": "需要审阅意见才能进行编辑。"}]
        }
    
    # 创建编辑提示
    system_prompt = """你是一位专业文章编辑。你的任务是根据审阅意见修改并完善文章。
    改进语言表达，修正错误，增强逻辑流程，并确保文章内容准确、完整、引人入胜。
    保持作者的风格和意图，同时提高整体质量。"""
    
    # 准备消息列表
    review_str = "\n".join(review_comments)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"根据以下审阅意见，修改这篇关于"{topic}"的文章：\n\n原文：\n{draft}\n\n审阅意见：\n{review_str}")
    ]
    
    # 调用模型
    model = ChatOpenAI(model="gpt-4", temperature=0.7)
    response = model.invoke(messages)
    
    # 更新状态
    return {
        "final_article": response.content,
        "status": "completed",
        "current_agent": "done",
        "conversation_history": history + [{"role": "editor", "content": "最终文章已完成。"}]
    }

# 定义决策函数：决定下一步应该由哪个智能体处理
def decide_next_agent(state: ResearchState) -> str:
    """根据当前状态决定下一个工作的智能体"""
    current = state.get("current_agent")
    status = state.get("status")
    
    # 如果已完成，则结束
    if status == "completed" or current == "done":
        return "end"
    
    # 否则，返回当前应该工作的智能体
    return current

# 构建多智能体图
def build_research_team_graph():
    """创建并返回研究团队的工作流图"""
    # 创建状态图构建器
    builder = StateGraph(ResearchState)
    
    # 添加节点（各个智能体）
    builder.add_node("researcher", researcher_agent)
    builder.add_node("planner", planner_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("reviewer", reviewer_agent)
    builder.add_node("editor", editor_agent)
    
    # 添加路由节点（决定下一步）
    builder.add_router(decide_next_agent, {
        "researcher": "researcher",
        "planner": "planner", 
        "writer": "writer",
        "reviewer": "reviewer",
        "editor": "editor",
        "end": END
    })
    
    # 设置初始节点
    builder.set_entry_point("researcher")
    
    # 添加各智能体完成后的去向
    builder.add_edge("researcher", decide_next_agent)
    builder.add_edge("planner", decide_next_agent)
    builder.add_edge("writer", decide_next_agent)
    builder.add_edge("reviewer", decide_next_agent)
    builder.add_edge("editor", decide_next_agent)
    
    # 编译图
    return builder.compile()

# 示例用法
def main():
    print("🚀 启动多智能体研究与写作系统")
    
    # 构建团队图
    research_team = build_research_team_graph()
    
    # 初始化状态
    initial_state: ResearchState = {
        "topic": "中国古代四大发明的现代应用",
        "research_results": None,
        "outline": None,
        "draft": None,
        "review_comments": None,
        "final_article": None,
        "current_agent": "researcher",  # 从研究员开始
        "conversation_history": [],
        "status": "in_progress"
    }
    
    print(f"\n📋 研究主题: {initial_state['topic']}")
    
    # 运行工作流
    print("\n🔄 开始工作流程...")
    
    # 使用stream()方法可以看到每个步骤的执行过程
    for event, state in research_team.stream(initial_state):
        if event.get("type") == "agent":
            print(f"⏩ {event['key']} 完成工作，下一步: {state['current_agent']}")
    
    # 获取最终结果
    final_state = research_team.invoke(initial_state)
    
    # 输出最终文章
    if final_state.get("final_article"):
        print("\n✅ 工作流程完成！")
        print("\n📄 最终文章片段:")
        print("-------------------")
        # 只显示前300个字符作为示例
        print(final_state["final_article"][:300] + "...\n(文章继续)")
    else:
        print("\n❌ 工作流程未能完成文章")

if __name__ == "__main__":
    main()