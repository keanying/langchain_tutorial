# LangChain 单智能体模式示例

import os
from typing import List, Dict, Any
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 确保设置环境变量
# os.environ["OPENAI_API_KEY"] = "your-api-key" 

# 1. ReAct 智能体示例 - 结合推理和行动的智能体
def create_react_agent():
    """创建基本的ReAct智能体"""
    # 定义工具集
    wikipedia = WikipediaAPIWrapper()
    python_repl = PythonREPLTool()
    
    tools = [
        Tool(
            name="维基百科",
            func=wikipedia.run,
            description="用于查询维基百科文章的工具"
        ),
        Tool(
            name="Python解释器",
            func=python_repl.run,
            description="用于执行Python代码的工具，可以进行计算或数据分析"
        )
    ]
    
    # 创建LLM
    llm = ChatOpenAI(temperature=0)
    
    # 创建记忆组件
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 初始化ReAct智能体
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    
    return agent

# 2. OpenAI函数智能体示例 - 专为函数调用设计的智能体
def create_openai_functions_agent():
    """创建基于OpenAI函数调用的智能体"""
    # 定义工具集
    wikipedia = WikipediaAPIWrapper()
    python_repl = PythonREPLTool()
    
    tools = [
        Tool(
            name="Python执行器",
            func=python_repl.run,
            description="执行Python代码的工具，适合进行计算、数据处理"
        ),
        Tool(
            name="维基百科",
            func=wikipedia.run,
            description="搜索维基百科文章的工具，适合查询事实性信息"
        )
    ]
    
    # 创建LLM
    llm = ChatOpenAI(temperature=0)
    
    # 初始化OpenAI函数智能体
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    
    return agent

# 3. 自定义智能体工具示例
class WeatherTool(BaseTool):
    name = "天气查询"
    description = "查询指定城市的天气情况"
    
    def _run(self, city: str) -> str:
        # 模拟天气API调用
        return f"{city}的天气: 晴朗, 25°C, 湿度50%"
        
    async def _arun(self, city: str) -> str:
        return self._run(city)

class CalculatorTool(BaseTool):
    name = "计算器"
    description = "进行数学计算，输入应为数学表达式"
    
    def _run(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
            
    async def _arun(self, expression: str) -> str:
        return self._run(expression)

def create_custom_tool_agent():
    """创建带有自定义工具的智能体"""
    tools = [
        WeatherTool(),
        CalculatorTool(),
        PythonREPLTool()
    ]
    
    llm = ChatOpenAI(temperature=0)
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent

# 使用示例
if __name__ == "__main__":
    print("=== LangChain 单智能体模式示例 ===")
    
    # 选择要演示的智能体类型 
    agent_type = "react"  # 可选: "react", "openai_functions", "custom"
    
    if agent_type == "react":
        agent = create_react_agent()
        response = agent.run(
            "谁是阿尔伯特·爱因斯坦? 他出生于哪一年? 计算从他出生到现在过了多少年。"
        )
    elif agent_type == "openai_functions":
        agent = create_openai_functions_agent() 
        response = agent.run(
            "计算 2345 + 5678 的结果，并解释这两个数字的数学特性。"
        )
    elif agent_type == "custom":
        agent = create_custom_tool_agent()
        response = agent.run(
            "北京的天气如何？然后计算25乘以4的结果。"
        )
        
    print(f"\n最终回答: {response}")
