"""
基本LangGraph示例 - 构建简单对话流程

本示例展示如何使用LangGraph构建一个基本对话流程，
该流程可以根据用户意图决定是直接回答问题还是使用工具搜索信息。
"""

import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# 设置环境变量（生产环境中应通过其他安全方式设置）
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # 请替换为您的密钥

# 定义状态类型
class ConversationState(TypedDict):
    """对话状态，存储消息历史和中间步骤"""
    messages: list[dict]  # 对话历史
    intermediate_steps: list  # 中间步骤（例如工具使用结果）

# 定义节点处理函数
def call_model(state: ConversationState) -> ConversationState:
    """使用LLM生成回复"""
    messages = state["messages"]  
    
    # 使用LangChain的ChatOpenAI组件
    model = ChatOpenAI(temperature=0.7)
    
    # 创建系统提示，让模型知道它的角色
    system_message = {
        "role": "system", 
        "content": "你是一个有帮助的助手。根据用户的问题提供简洁、准确的回答。"
    }
    
    # 为LLM准备消息列表
    model_messages = [system_message] + messages
    
    # 调用模型获取回复
    response = model.invoke(model_messages)
    
    # 添加回复到消息历史
    return {
        "messages": messages + [{"role": "assistant", "content": response.content}]
    }

def detect_intent(state: ConversationState) -> Annotated[str, ("direct_answer", "use_search")]:
    """分析用户最后一条消息，决定后续处理方式"""
    messages = state["messages"]
    last_message = messages[-1]["content"]
    
    # 简单的意图检测逻辑，识别搜索相关的关键词
    search_keywords = ["查找", "搜索", "寻找", "查询", "搜一下", "找一下", "了解", "信息"]
    
    # 如果检测到搜索意图，使用搜索工具
    for keyword in search_keywords:
        if keyword in last_message:
            return "use_search"
    
    # 否则直接回答
    return "direct_answer"

def search_information(state: ConversationState) -> ConversationState:
    """模拟搜索信息的工具节点"""
    messages = state["messages"]
    last_message = messages[-1]["content"]
    
    # 模拟搜索过程（实际应用中会调用真实的搜索API）
    search_result = f"模拟搜索结果：关于「{last_message}」的信息"
    
    # 存储搜索结果到中间步骤
    return {
        "intermediate_steps": state.get("intermediate_steps", []) + [search_result]
    }

def generate_response_with_search(state: ConversationState) -> ConversationState:
    """使用搜索结果生成回复"""
    messages = state["messages"]
    intermediate_steps = state.get("intermediate_steps", [])
    
    # 如果有搜索结果，使用它来生成更有信息的回复
    if intermediate_steps:
        search_info = intermediate_steps[-1]
        
        # 创建提示，将搜索结果作为上下文
        prompt = ChatPromptTemplate.from_template(
            "根据以下搜索信息回答用户问题。\n\n搜索信息: {search_info}\n\n用户问题: {question}"
        )
        
        # 使用LangChain处理
        model = ChatOpenAI(temperature=0.7)
        chain = prompt | model
        
        response = chain.invoke({
            "search_info": search_info,
            "question": messages[-1]["content"]
        })
        
        # 添加回复到消息历史
        return {
            "messages": messages + [{"role": "assistant", "content": response.content}]
        }
    else:
        # 如果没有搜索结果，回退到普通回复
        return call_model(state)

# 构建图
def build_conversation_graph():
    """构建并返回对话处理图"""
    # 创建状态图构建器
    builder = StateGraph(ConversationState)
    
    # 添加节点
    builder.add_node("detect_intent", detect_intent)
    builder.add_node("call_model", call_model)
    builder.add_node("search_information", search_information)
    builder.add_node("generate_response_with_search", generate_response_with_search)
    
    # 添加边，定义执行流
    # 首先检测意图
    builder.set_entry_point("detect_intent")
    
    # 根据意图结果路由到不同节点
    builder.add_edge("detect_intent", "call_model", condition="direct_answer")
    builder.add_edge("detect_intent", "search_information", condition="use_search")
    
    # 搜索后生成回复
    builder.add_edge("search_information", "generate_response_with_search")
    
    # 最后输出响应并结束
    builder.add_edge("call_model", END)
    builder.add_edge("generate_response_with_search", END)
    
    # 编译图
    return builder.compile()

# 示例用法
def main():
    # 构建图
    conversation_graph = build_conversation_graph()
    
    # 示例1：普通问题
    print("====== 示例1：普通问题 ======")
    state1 = {"messages": [{"role": "user", "content": "你好，今天天气怎么样？"}], "intermediate_steps": []}
    result1 = conversation_graph.invoke(state1)
    print("\n用户: ", state1["messages"][0]["content"])
    print("助手: ", result1["messages"][1]["content"])
    
    # 示例2：需要搜索的问题
    print("\n====== 示例2：搜索问题 ======")
    state2 = {"messages": [{"role": "user", "content": "请帮我搜索一下最近的人工智能进展"}], "intermediate_steps": []}
    result2 = conversation_graph.invoke(state2)
    print("\n用户: ", state2["messages"][0]["content"])
    print("助手: ", result2["messages"][1]["content"])

if __name__ == "__main__":
    main()