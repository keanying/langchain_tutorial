"""
LangGraph和LangSmith集成示例

本示例展示如何将LangGraph工作流与LangSmith集成，以实现高级监控、
跟踪和评估功能。示例构建了一个简单的多步骤推理工作流，并使用
LangSmith跟踪其执行过程和性能。
"""

import os
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# LangChain和LangGraph导入
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# LangSmith导入
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset

# 设置环境变量（生产环境中应通过其他安全方式设置）
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # 请替换为您的OpenAI密钥
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"  # 请替换为您的LangSmith密钥
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-monitoring-demo"

# 定义工作流状态类型
class ProblemSolvingState(TypedDict):
    """多步骤问题解决工作流的状态"""
    problem: str  # 原始问题
    plan: Optional[List[str]]  # 解决问题的步骤计划
    research: Optional[Dict[str, str]]  # 相关研究/信息
    solution_draft: Optional[str]  # 初步解决方案
    solution_final: Optional[str]  # 最终解决方案
    evaluation: Optional[Dict[str, Any]]  # 解决方案的自我评估
    current_step: str  # 当前步骤标识符
    timestamps: Dict[str, str]  # 记录每个步骤的时间戳

# 创建工作流节点
def planner(state: ProblemSolvingState) -> ProblemSolvingState:
    """制定解决问题的计划"""
    print("🔍 制定解决方案计划...")
    
    problem = state["problem"]
    
    # 更新时间戳
    timestamps = state.get("timestamps", {})
    timestamps["plan_start"] = datetime.now().isoformat()
    
    # 使用LLM制定解决问题的计划
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一位专业的问题解决专家。你的任务是为复杂问题制定清晰的解决步骤。"),
        HumanMessage(content=f"请为以下问题制定一个3-5步的解决方案计划。每一步应该简明扼要。\n\n问题：{problem}")
    ])
    
    model = ChatOpenAI(temperature=0.7)
    response = model.invoke(prompt.format_messages())
    
    # 处理响应，提取计划步骤
    plan_text = response.content
    plan_steps = [step.strip() for step in plan_text.split("\n") if step.strip()]
    
    # 过滤掉不是真正步骤的行（如标题等）
    filtered_steps = []
    for step in plan_steps:
        # 简单的启发式方法：如果行以数字或"-"或"*"开头，可能是一个步骤
        if any(step.startswith(prefix) for prefix in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "*", "步骤"]):
            filtered_steps.append(step)
    
    # 如果过滤后没有步骤，使用原始行
    if not filtered_steps and plan_steps:
        filtered_steps = plan_steps
    
    # 更新时间戳
    timestamps["plan_end"] = datetime.now().isoformat()
    
    # 返回更新后的状态
    return {
        "plan": filtered_steps,
        "current_step": "research",
        "timestamps": timestamps
    }

def researcher(state: ProblemSolvingState) -> ProblemSolvingState:
    """收集解决问题所需的信息"""
    print("🔍 收集相关信息...")
    
    problem = state["problem"]
    plan = state.get("plan", [])
    
    # 更新时间戳
    timestamps = state.get("timestamps", {})
    timestamps["research_start"] = datetime.now().isoformat()
    
    # 准备研究提示
    research_prompt = f"""作为一名研究专家，请为解决以下问题收集关键信息：
    
    问题：{problem}
    
    已制定的计划：
    {"".join([f"- {step}\n" for step in plan])}
    
    请提供解决此问题所需的3个关键信息点。每个信息点应包含：
    1) 标题
    2) 简短但信息丰富的内容（100-150字）
    
    以"信息点1："、"信息点2："等格式开始每个部分。
    """
    
    # 使用LLM进行研究
    model = ChatOpenAI(temperature=0.7)
    response = model.invoke([HumanMessage(content=research_prompt)])
    
    # 处理响应，提取研究结果
    research_text = response.content
    
    # 简单的解析，提取信息点
    info_points = {}
    current_point = None
    current_content = []
    
    for line in research_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # 检测新的信息点标题
        if line.startswith("信息点") and ":" in line:
            # 保存之前的信息点
            if current_point and current_content:
                info_points[current_point] = '\n'.join(current_content)
                current_content = []
                
            # 开始新的信息点
            current_point = line.split(":", 1)[0].strip()
            
            # 如果标题后有内容，添加到当前内容
            if ":" in line and len(line.split(":", 1)[1].strip()) > 0:
                current_content.append(line.split(":", 1)[1].strip())
        else:
            # 继续添加到当前信息点
            if current_point:
                current_content.append(line)
    
    # 添加最后一个信息点
    if current_point and current_content:
        info_points[current_point] = '\n'.join(current_content)
    
    # 更新时间戳
    timestamps["research_end"] = datetime.now().isoformat()
    
    # 返回更新后的状态
    return {
        "research": info_points,
        "current_step": "draft",
        "timestamps": timestamps
    }

def solution_drafter(state: ProblemSolvingState) -> ProblemSolvingState:
    """起草初步解决方案"""
    print("🔍 起草初步解决方案...")
    
    problem = state["problem"]
    plan = state.get("plan", [])
    research = state.get("research", {})
    
    # 更新时间戳
    timestamps = state.get("timestamps", {})
    timestamps["draft_start"] = datetime.now().isoformat()
    
    # 准备研究信息文本
    research_text = ""
    for title, content in research.items():
        research_text += f"{title}:\n{content}\n\n"
    
    # 准备起草提示
    draft_prompt = f"""作为一名解决方案专家，请基于以下信息为问题草拟一个初步解决方案：
    
    问题：{problem}
    
    解决计划：
    {"".join([f"- {step}\n" for step in plan])}
    
    研究信息：
    {research_text}
    
    请提供一个全面但简洁的初步解决方案，应用上述信息和计划步骤。
    """
    
    # 使用LLM起草解决方案
    model = ChatOpenAI(temperature=0.7)
    response = model.invoke([HumanMessage(content=draft_prompt)])
    
    # 处理响应
    draft = response.content
    
    # 更新时间戳
    timestamps["draft_end"] = datetime.now().isoformat()
    
    # 返回更新后的状态
    return {
        "solution_draft": draft,
        "current_step": "refine",
        "timestamps": timestamps
    }

def solution_refiner(state: ProblemSolvingState) -> ProblemSolvingState:
    """完善解决方案"""
    print("🔍 完善最终解决方案...")
    
    problem = state["problem"]
    draft = state.get("solution_draft", "")
    plan = state.get("plan", [])
    
    # 更新时间戳
    timestamps = state.get("timestamps", {})
    timestamps["refine_start"] = datetime.now().isoformat()
    
    # 准备完善提示
    refine_prompt = f"""作为一名解决方案专家，请完善以下初步解决方案：
    
    问题：{problem}
    
    解决计划：
    {"".join([f"- {step}\n" for step in plan])}
    
    初步解决方案：
    {draft}
    
    请完善上述解决方案，确保它：
    1. 直接解决核心问题
    2. 遵循计划中的所有步骤
    3. 逻辑清晰、表达准确
    4. 实用且可执行
    
    提供最终的完整解决方案。
    """
    
    # 使用LLM完善解决方案
    model = ChatOpenAI(temperature=0.7)
    response = model.invoke([HumanMessage(content=refine_prompt)])
    
    # 处理响应
    refined_solution = response.content
    
    # 更新时间戳
    timestamps["refine_end"] = datetime.now().isoformat()
    
    # 返回更新后的状态
    return {
        "solution_final": refined_solution,
        "current_step": "evaluate",
        "timestamps": timestamps
    }

def evaluator(state: ProblemSolvingState) -> ProblemSolvingState:
    """评估最终解决方案"""
    print("🔍 评估解决方案...")
    
    problem = state["problem"]
    solution = state.get("solution_final", "")
    
    # 更新时间戳
    timestamps = state.get("timestamps", {})
    timestamps["evaluate_start"] = datetime.now().isoformat()
    
    # 准备评估提示
    eval_prompt = f"""作为一名评估专家，请评估以下解决方案的质量：
    
    问题：{problem}
    
    解决方案：
    {solution}
    
    请从以下五个方面评估该解决方案，为每个方面打分(1-10)并提供简短理由：
    1. 有效性：解决方案能否解决核心问题？
    2. 完整性：解决方案是否全面覆盖了所有必要的方面？
    3. 可行性：解决方案在实际中可以实施吗？
    4. 清晰度：解决方案表述是否清晰？
    5. 创新性：解决方案是否提供了新颖的思路？
    
    请以JSON格式返回评估结果，包含各项分数和理由。
    """
    
    # 使用LLM评估解决方案
    model = ChatOpenAI(temperature=0.2)
    response = model.invoke([HumanMessage(content=eval_prompt)])
    
    # 处理响应
    evaluation_text = response.content
    
    # 简单解析评估结果（实际上应使用结构化输出解析器）
    # 这里使用简单方法提取评分
    evaluation = {
        "text": evaluation_text,
        "scores": {}
    }
    
    aspects = ["有效性", "完整性", "可行性", "清晰度", "创新性"]
    for aspect in aspects:
        # 简单启发式方法查找评分
        for line in evaluation_text.split("\n"):
            if aspect in line and ":" in line:
                try:
                    # 尝试查找数字评分
                    for part in line.split(":")[1].split():
                        if part.strip().isdigit():
                            evaluation["scores"][aspect] = int(part.strip())
                            break
                except:
                    # 如果解析失败，设置默认值
                    evaluation["scores"][aspect] = 0
    
    # 计算总分
    scores = evaluation["scores"].values()
    if scores:
        evaluation["total_score"] = sum(scores)
        evaluation["average_score"] = sum(scores) / len(scores)
    
    # 更新时间戳
    timestamps["evaluate_end"] = datetime.now().isoformat()
    
    # 返回更新后的状态
    return {
        "evaluation": evaluation,
        "current_step": "complete",
        "timestamps": timestamps
    }

# 构建工作流图
def build_problem_solving_graph():
    """构建并返回问题解决工作流图"""
    # 创建状态图构建器
    builder = StateGraph(ProblemSolvingState)
    
    # 添加节点
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("drafter", solution_drafter)
    builder.add_node("refiner", solution_refiner)
    builder.add_node("evaluator", evaluator)
    
    # 添加边，定义工作流
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "drafter")
    builder.add_edge("drafter", "refiner")
    builder.add_edge("refiner", "evaluator")
    builder.add_edge("evaluator", END)
    
    # 设置入口点
    builder.set_entry_point("planner")
    
    # 编译图
    return builder.compile()

# 创建和管理评估数据集
def create_problem_dataset():
    """创建或获取问题解决评估数据集"""
    client = Client()
    
    try:
        # 尝试读取现有数据集
        dataset = client.read_dataset("problem-solving-examples")
        print(f"使用现有数据集 '{dataset.name}'")
    except:
        # 创建新数据集
        dataset = client.create_dataset(
            "problem-solving-examples",
            description="用于评估问题解决工作流的示例集"
        )
        print(f"创建了新数据集 '{dataset.name}'")
        
        # 添加示例问题
        examples = [
            {
                "problem": "如何提高远程团队的协作效率？",
                "reference": "提高远程团队效率的解决方案应包括：1)建立清晰的沟通协议和工具选择；2)实施异步与同步工作机制相结合的策略；3)使用项目管理工具确保任务透明度；4)定期组织团队建设活动增强凝聚力；5)提供适当培训和资源支持远程工作。"
            },
            {
                "problem": "小型企业如何有效降低运营成本？",
                "reference": "小型企业降低运营成本的方案应包括：1)审核和优化现有流程以消除浪费；2)协商供应商合同和探索批量采购优惠；3)考虑远程工作选项减少办公空间成本；4)实施能源效率措施；5)利用自动化技术减少人工操作；6)采用精益管理原则不断评估和改进。"
            },
            {
                "problem": "如何设计有效的客户反馈收集系统？",
                "reference": "有效客户反馈系统应：1)使用多渠道方法(调查、社交媒体、直接对话等)；2)设计简短、针对性强的问题；3)在客户旅程的关键点收集反馈；4)建立闭环系统确保反馈得到回应；5)提供激励措施鼓励参与；6)使用分析工具识别趋势；7)将见解整合到产品和服务改进中。"
            }
        ]
        
        # 添加到数据集
        for example in examples:
            client.create_example(
                inputs={"problem": example["problem"]},
                outputs={"reference": example["reference"]},
                dataset_id=dataset.id
            )
        
        print(f"已添加 {len(examples)} 个示例到数据集")
    
    return dataset.id

# 使用LangSmith评估工作流
def evaluate_workflow(dataset_id):
    """使用LangSmith评估工作流性能"""
    print("\n开始工作流评估流程...")
    
    # 定义工作流工厂函数（返回一个能处理输入的函数）
    def workflow_factory():
        # 构建问题解决图
        graph = build_problem_solving_graph()
        
        # 创建包装函数，处理输入并返回最终解决方案
        def invoke_with_problem(inputs):
            problem = inputs["problem"]
            
            # 初始化状态
            initial_state = {
                "problem": problem,
                "plan": None,
                "research": None,
                "solution_draft": None,
                "solution_final": None,
                "evaluation": None,
                "current_step": "plan",
                "timestamps": {"start": datetime.now().isoformat()}
            }
            
            # 运行图
            final_state = graph.invoke(initial_state)
            
            # 返回需要的输出
            return {
                "solution": final_state.get("solution_final", "未能生成解决方案")
            }
        
        return invoke_with_problem
    
    # 配置评估
    eval_config = RunEvalConfig(
        # 评估标准
        evaluators=[
            "labeled_criteria",  # 使用预定义标准评估
            "embedding_distance"  # 检查解决方案与参考解决方案的相似度
        ],
        # 自定义评估标准
        custom_evaluators=None,
        # 评估标准配置
        evaluation_config={
            "labeled_criteria": {
                "criteria": {
                    "直接性": "解决方案直接针对问题，不偏离主题",
                    "可操作性": "解决方案提供明确可执行的步骤",
                    "全面性": "解决方案涵盖问题的各个方面"
                }
            }
        }
    )
    
    # 运行评估
    results = run_on_dataset(
        dataset_name="problem-solving-examples",  # 数据集名称
        llm_or_chain_factory=workflow_factory(),  # 工作流
        evaluation=eval_config,  # 评估配置
        verbose=True,  # 显示详细信息
        project_name="workflow-evaluation"  # LangSmith项目名称
    )
    
    print("评估完成！")
    return results

# 分析工作流性能
def analyze_workflow_performance():
    """分析工作流的性能指标"""
    print("\n分析工作流性能...")
    
    # 创建LangSmith客户端
    client = Client()
    
    try:
        # 获取最近的运行
        runs = list(client.list_runs(
            project_name="langgraph-monitoring-demo",
            execution_order=1,  # 按执行时间排序
            limit=10  # 获取最近10个运行
        ))
        
        if not runs:
            print("未找到运行记录。请先运行工作流生成一些数据。")
            return
        
        print(f"找到 {len(runs)} 个运行记录")
        
        # 提取性能指标
        latencies = []
        token_counts = []
        step_latencies = {
            "planner": [],
            "researcher": [],
            "drafter": [],
            "refiner": [],
            "evaluator": []
        }
        
        for run in runs:
            # 添加总延迟
            if run.latency_ms:
                latencies.append(run.latency_ms)
            
            # 添加令牌计数（如果可用）
            if run.metrics and "tokens" in run.metrics:
                token_counts.append(run.metrics["tokens"])
            
            # 获取子运行以分析每个步骤
            child_runs = list(client.list_runs(
                parent_run_id=run.id
            ))
            
            for child in child_runs:
                # 基于运行名称分类
                for step_name in step_latencies.keys():
                    if step_name in child.name.lower() and child.latency_ms:
                        step_latencies[step_name].append(child.latency_ms)
        
        # 计算和显示平均延迟
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"平均总延迟: {avg_latency:.2f} 毫秒")
        
        # 计算和显示每个步骤的平均延迟
        print("\n步骤延迟分析:")
        for step, step_lats in step_latencies.items():
            if step_lats:
                avg_step_latency = sum(step_lats) / len(step_lats)
                print(f"  {step}: {avg_step_latency:.2f} 毫秒")
        
        # 创建步骤延迟条形图
        if any(step_latencies.values()):
            avg_step_latencies = {}
            for step, lats in step_latencies.items():
                if lats:
                    avg_step_latencies[step] = sum(lats) / len(lats)
            
            if avg_step_latencies:
                steps = list(avg_step_latencies.keys())
                values = list(avg_step_latencies.values())
                
                plt.figure(figsize=(10, 6))
                plt.bar(steps, values)
                plt.xlabel('处理步骤')
                plt.ylabel('平均延迟 (毫秒)')
                plt.title('工作流各步骤平均延迟')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('workflow_latency_analysis.png')
                print("\n步骤延迟分析图已保存为 'workflow_latency_analysis.png'")
        
    except Exception as e:
        print(f"分析工作流性能时出错: {str(e)}")

# 运行示例问题
def run_example_problem():
    """运行示例问题并使用LangSmith跟踪"""
    print("🚀 运行问题解决工作流示例...\n")
    
    # 构建问题解决图
    problem_solver = build_problem_solving_graph()
    
    # 设置示例问题
    problem = "如何为新创立的线上教育平台增加用户参与度？"
    print(f"问题: {problem}\n")
    
    # 初始化状态
    initial_state = {
        "problem": problem,
        "plan": None,
        "research": None,
        "solution_draft": None,
        "solution_final": None,
        "evaluation": None,
        "current_step": "plan",
        "timestamps": {"start": datetime.now().isoformat()}
    }
    
    # 如果你想观察每个步骤的执行，可以使用stream方法
    print("执行工作流...")
    for event, state in problem_solver.stream(initial_state):
        if event.get("type") == "node":
            node_name = event.get("node", {}).get("name", "未知节点")
            print(f"完成: {node_name}, 下一步: {state.get('current_step', 'unknown')}")
    
    # 获取最终结果
    final_state = problem_solver.invoke(initial_state)
    
    # 显示结果
    print("\n==== 工作流完成 ====\n")
    print("最终解决方案:")
    print("-----------------")
    print(final_state.get("solution_final", "未能生成解决方案"))
    print("\n评估结果:")
    print("-----------------")
    
    evaluation = final_state.get("evaluation", {})
    scores = evaluation.get("scores", {})
    
    if scores:
        for aspect, score in scores.items():
            print(f"{aspect}: {score}/10")
        
        avg = evaluation.get("average_score")
        if avg:
            print(f"\n平均分: {avg:.1f}/10")
    
    print("\n工作流性能信息已记录到LangSmith。请访问LangSmith控制台查看详细追踪。")
    return final_state

# 主函数
def main():
    """主函数，运行完整的集成示例"""
    print("===== LangGraph 和 LangSmith 集成示例 =====\n")
    
    # 检查API密钥
    if os.environ.get("LANGCHAIN_API_KEY") == "YOUR_LANGSMITH_API_KEY":
        print("⚠️ 警告: 请将代码中的'YOUR_LANGSMITH_API_KEY'替换为您的实际LangSmith API密钥")
        print("申请API密钥: https://smith.langchain.com/\n")
    
    # 运行示例问题
    print("\n=== 第1部分: 运行带有LangSmith跟踪的LangGraph工作流 ===")
    final_state = run_example_problem()
    
    # 等待用户确认继续
    input("\n按回车键继续下一部分...\n")
    
    # 创建评估数据集
    print("\n=== 第2部分: 创建和使用评估数据集 ===")
    dataset_id = create_problem_dataset()
    
    # 等待用户确认继续
    input("\n按回车键继续下一部分...\n")
    
    # 分析工作流性能
    print("\n=== 第3部分: 分析工作流性能 ===")
    analyze_workflow_performance()
    
    # 等待用户确认继续
    input("\n按回车键继续下一部分...\n")
    
    # 评估工作流
    print("\n=== 第4部分: 使用LangSmith评估工作流 ===")
    print("注意: 此步骤可能需要较长时间运行")
    input("按回车键开始评估或按Ctrl+C终止...\n")
    
    try:
        evaluate_workflow(dataset_id)
    except KeyboardInterrupt:
        print("评估已终止")
    
    print("\n===== 示例完成 =====")
    print("请登录LangSmith控制台查看详细跟踪和评估结果:")
    print("https://smith.langchain.com/")

if __name__ == "__main__":
    main()