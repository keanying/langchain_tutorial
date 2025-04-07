# LangChain 多智能体编排模式示例

import os
from typing import List, Dict, Any
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool


# 确保设置环境变量
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# 1. 团队监督者模式 (Team Supervisor)
class TeamSupervisorSystem:
    """团队监督者模式 - 一个监督者智能体协调多个专家智能体"""

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)

        # 创建专家智能体
        self.researcher = self._create_researcher_agent()
        self.coder = self._create_coder_agent()
        self.critic = self._create_critic_agent()

        # 创建监督者
        self.supervisor = self._create_supervisor_agent()

    def _create_researcher_agent(self):
        """创建研究员智能体"""
        wikipedia = WikipediaAPIWrapper()
        tools = [
            Tool(
                name="维基百科",
                func=wikipedia.run,
                description="用于查询事实性信息的维基百科工具"
            )
        ]

        return initialize_agent(
            tools,
            self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def _create_coder_agent(self):
        """创建编码员智能体"""
        tools = [PythonREPLTool()]

        return initialize_agent(
            tools,
            self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def _create_critic_agent(self):
        """创建评论员智能体"""
        prompt = PromptTemplate(
            template="你是一位严格的评论家。请评估以下解决方案的质量、正确性和效率：\n{solution}\n请提供详细的改进建议。",
            input_variables=["solution"]
        )

        return LLMChain(llm=self.llm, prompt=prompt, verbose=True)

    def _create_supervisor_agent(self):
        """创建监督者智能体"""
        prompt = PromptTemplate(
            template="你是一个团队领导，负责协调其他专家完成任务。你需要将复杂问题分解成子任务，分配给适当的专家，并整合他们的工作。\n\n任务描述: {task}\n\n请制定一个计划，说明如何将这个任务分解并分配给团队成员（研究员、编码员、评论员）。",
            input_variables=["task"]
        )

        return LLMChain(llm=self.llm, prompt=prompt)

    def run(self, task: str) -> Dict[str, Any]:
        """运行团队监督者系统"""
        print("\n===== 团队监督者模式 =====")
        print(f"任务: {task}")

        # 1. 监督者分配任务
        print("\n1. 监督者制定计划")
        plan = self.supervisor.run(task=task)
        print(f"\n计划:\n{plan}")

        # 2. 研究员收集信息
        print("\n2. 研究员收集信息")
        research_query = f"请查询关于以下内容的事实性信息: {task}"
        research_result = self.researcher.run(research_query)
        print(f"\n研究结果:\n{research_result}")

        # 3. 编码员根据研究结果实现解决方案
        print("\n3. 编码员实现解决方案")
        coding_task = f"根据以下研究结果，编写Python代码解决问题: {task}\n\n研究信息: {research_result}"
        code_solution = self.coder.run(coding_task)
        print(f"\n代码解决方案:\n{code_solution}")

        # 4. 评论员评估解决方案
        print("\n4. 评论员评估解决方案")
        critique = self.critic.run(solution=code_solution)
        print(f"\n评估意见:\n{critique}")

        # 5. 监督者整合结果
        print("\n5. 监督者整合最终结果")
        final_prompt = f"请整合以下信息，提供最终的解决方案和总结：\n\n原始任务: {task}\n\n研究结果: {research_result}\n\n代码解决方案: {code_solution}\n\n评估意见: {critique}"
        final_solution = self.llm.predict(final_prompt)
        print(f"\n最终解决方案:\n{final_solution}")

        return {
            "task": task,
            "plan": plan,
            "research": research_result,
            "code": code_solution,
            "critique": critique,
            "final_solution": final_solution
        }


# 2. 计划执行模式 (Plan Executor)
class PlanExecutorSystem:
    """计划执行模式 - 一个智能体制定计划，另一个智能体执行计划"""

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)

        # 创建规划者智能体
        self.planner = self._create_planner()

        # 创建执行者智能体
        self.executor = self._create_executor()

    def _create_planner(self):
        """创建规划者智能体"""
        prompt = PromptTemplate(
            template="你是一位专业的规划者。请为以下任务制定一个详细的、分步骤的计划：\n\n任务: {task}\n\n请提供一个清晰的、有序的计划，每个步骤应该具体且可执行。确保计划涵盖任务的所有方面，并考虑可能的障碍和解决方案。",
            input_variables=["task"]
        )

        return LLMChain(llm=self.llm, prompt=prompt)

    def _create_executor(self):
        """创建执行者智能体"""
        tools = [
            Tool(
                name="维基百科",
                func=WikipediaAPIWrapper().run,
                description="用于查询信息的工具"
            ),
            PythonREPLTool()
        ]

        return initialize_agent(
            tools,
            self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, task: str) -> Dict[str, Any]:
        """运行计划执行系统"""
        print("\n===== 计划执行模式 =====")
        print(f"任务: {task}")

        # 1. 制定计划
        print("\n1. 规划者制定计划")
        plan = self.planner.run(task=task)
        print(f"\n计划:\n{plan}")

        # 2. 执行计划
        print("\n2. 执行者执行计划")
        execution_task = f"根据以下计划执行任务：\n\n任务: {task}\n\n计划:\n{plan}\n\n执行每个步骤并报告结果。"
        execution_result = self.executor.run(execution_task)
        print(f"\n执行结果:\n{execution_result}")

        return {
            "task": task,
            "plan": plan,
            "execution_result": execution_result
        }


# 使用示例
if __name__ == "__main__":
    print("=== LangChain 多智能体编排模式示例 ===")

    # 选择要演示的编排模式
    orchestration_type = "team_supervisor"  # 可选: "team_supervisor", "plan_executor"

    if orchestration_type == "team_supervisor":
        system = TeamSupervisorSystem()
        result = system.run("创建一个简单的网站访问计数器应用")

    elif orchestration_type == "plan_executor":
        system = PlanExecutorSystem()
        result = system.run("研究机器学习中的过拟合问题并提供3种解决方法")
