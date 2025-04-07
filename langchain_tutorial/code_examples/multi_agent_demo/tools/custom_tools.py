# 自定义工具库
from langchain.tools import BaseTool
from langchain.agents import Tool
from typing import Dict, List, Any


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "在网络上搜索信息"

    def _run(self, query: str) -> str:
        # 模拟网络搜索
        return f"关于'{query}'的搜索结果: 这里是一些相关信息..."

    async def _arun(self, query: str) -> str:
        return self._run(query)


class DataAnalysisTool(BaseTool):
    name = "data_analysis"
    description = "分析数据集并提供洞察"

    def _run(self, data_description: str) -> str:
        # 模拟数据分析
        return f"对'{data_description}'的分析结果: 数据显示以下模式..."

    async def _arun(self, data_description: str) -> str:
        return self._run(data_description)


class CodeGenerationTool(BaseTool):
    name = "code_generation"
    description = "生成Python代码来解决特定问题"

    def _run(self, task: str) -> str:
        # 模拟代码生成
        return f"为'{task}'生成的代码:\n```python\ndef solution():\n    # 实现代码\n    print('解决方案')\n```"

    async def _arun(self, task: str) -> str:
        return self._run(task)


def get_all_tools() -> List[Tool]:
    """获取所有可用工具"""
    return [
        Tool(
            name="Web搜索",
            func=WebSearchTool()._run,
            description="在网络上搜索信息"
        ),
        Tool(
            name="数据分析",
            func=DataAnalysisTool()._run,
            description="分析数据集并提供洞察"
        ),
        Tool(
            name="代码生成",
            func=CodeGenerationTool()._run,
            description="生成Python代码来解决特定问题"
        )
    ]
