"""
LangSmith评估示例

本示例展示如何使用LangSmith评估LLM应用程序的性能，
包括创建评估数据集、运行自动评估和分析评估结果。
"""

import os
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.evaluation import EvaluatorType
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langsmith import Client
import pandas as pd
import matplotlib.pyplot as plt

# 设置LangSmith环境变量
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"  # 请替换为您的LangSmith API密钥
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langsmith-evaluation"


def create_qa_chain():
    """创建用于问答的简单链"""
    # 创建LLM实例
    llm = ChatOpenAI(temperature=0.7)

    # 创建提示模板
    template = """作为一个知识渊博的AI助手，请回答以下问题：

问题: {question}

请提供准确、全面但简洁的回答。"""
    prompt = ChatPromptTemplate.from_template(template)

    # 创建链
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain


def create_evaluation_dataset():
    """创建评估数据集"""
    print("🔍 创建评估数据集...")

    # 创建LangSmith客户端
    client = Client()

    # 定义问题和参考答案
    qa_pairs = [
        {
            "question": "什么是机器学习？",
            "reference": "机器学习是人工智能的一个分支，它使计算机系统能够自动从经验中学习和改进，而无需显式编程。它主要关注开发能够访问数据并使用数据自我学习的算法。"
        },
        {
            "question": "什么是神经网络？",
            "reference": "神经网络是一种受人脑中神经元网络启发的计算模型。它由多层互连的节点（或'神经元'）组成，可以识别输入数据中的模式。神经网络是深度学习的基础，广泛应用于图像识别、语言处理等领域。"
        },
        {
            "question": "强化学习是什么？",
            "reference": "强化学习是机器学习的一种形式，智能体通过与环境交互并从行动的结果中学习。它基于奖励机制，智能体学会选择能够最大化累积奖励的行动。这种方法在游戏、机器人技术和自动控制系统中特别有用。"
        },
        {
            "question": "什么是自然语言处理？",
            "reference": "自然语言处理(NLP)是人工智能的一个子领域，专注于计算机理解、解释和生成人类语言的能力。NLP技术使应用程序能够识别文本中的情感、提取关键信息、翻译语言，甚至生成类似人类的文本响应。"
        },
        {
            "question": "什么是计算机视觉？",
            "reference": "计算机视觉是一个人工智能领域，致力于使计算机能够从数字图像或视频中获取高级信息，并做出决策。它涉及开发算法来理解视觉内容，例如对象识别、场景理解和图像分类。"
        }
    ]

    # 检查数据集是否已存在
    try:
        dataset = client.read_dataset("AI知识问答评估数据集")
        print(f"✅ 数据集已存在，ID: {dataset.id}")

    except:
        # 创建新数据集
        dataset = client.create_dataset(
            "AI知识问答评估数据集",
            description="用于评估AI系统回答人工智能相关问题的能力的数据集"
        )
        print(f"✅ 创建了新数据集，ID: {dataset.id}")

        # 添加示例到数据集
        for pair in qa_pairs:
            client.create_example(
                inputs={"question": pair["question"]},
                outputs={"reference": pair["reference"]},
                dataset_id=dataset.id
            )

        print(f"✅ 已添加 {len(qa_pairs)} 个示例到数据集")

    return dataset.id


def run_chain_evaluation(dataset_id, chain_name="默认QA链"):
    """在数据集上评估链的性能"""
    print(f"\n🔍 在数据集上评估{chain_name}...")

    # 创建问答链
    chain = create_qa_chain()

    # 配置评估
    evaluation_config = RunEvalConfig(
        # 添加预定义的评估器
        evaluators=[
            # 正确性评估 - 检查回答是否与参考答案一致
            EvaluatorType.QA,

            # 以下是基于特定标准的评估器
            EvaluatorType.CRITERIA.with_config({
                "criteria": {
                    "准确性": "回答包含准确的事实信息，没有错误",
                    "完整性": "回答全面地覆盖了问题的各个方面",
                    "简洁性": "回答简洁明了，没有不必要的冗余",
                    "有用性": "回答对用户有实际帮助"
                }
            }),

            # 可信度评估
            EvaluatorType.EMBEDDING_DISTANCE,
        ],

        # 自定义评估链
        custom_evaluators=[
            create_custom_evaluator()
        ]
    )

    # 运行评估
    results = run_on_dataset(
        dataset_name="AI知识问答评估数据集",  # 直接使用数据集名称
        llm_or_chain_factory=lambda: chain,
        evaluation=evaluation_config,
        verbose=True,
        project_name=f"evaluation-{chain_name}"
    )

    print(f"✅ {chain_name}评估完成")

    return results


def create_custom_evaluator():
    """创建自定义评估器，评估回答的教学价值"""
    # 创建用于评估的LLM
    eval_llm = ChatOpenAI(temperature=0)

    # 定义评估标准和提示
    criteria = {
        "教学价值": "评估回答是否能够有效地教育读者，解释概念并帮助读者理解主题"
    }

    # 创建基于标准的评估链
    evaluator = LabeledCriteriaEvalChain.from_llm(
        llm=eval_llm,
        criteria=criteria,
        evaluation_entities={"回答": "回答", "参考": "reference"},
        output_key="教学价值"
    )

    return evaluator


def compare_models():
    """比较不同模型的性能"""
    print("\n🔍 比较不同模型的性能...")

    # 创建数据集
    dataset_id = create_evaluation_dataset()

    # 定义要比较的模型
    models = {
        "标准温度模型": lambda: ChatOpenAI(temperature=0.7),
        "低温度模型": lambda: ChatOpenAI(temperature=0.2),
        "GPT-4模型": lambda: ChatOpenAI(model="gpt-4", temperature=0.7)  # 这只是示例，请根据需要调整
    }

    results = {}

    # 为每个模型创建和评估链
    for model_name, model_factory in models.items():
        print(f"\n评估 {model_name}...")

        # 创建使用特定模型的链
        def create_chain_with_model():
            llm = model_factory()
            template = """作为一个知识渊博的AI助手，请回答以下问题：

问题: {question}

请提供准确、全面但简洁的回答。"""
            prompt = ChatPromptTemplate.from_template(template)
            return LLMChain(llm=llm, prompt=prompt)

        # 配置评估
        evaluation_config = RunEvalConfig(
            evaluators=[
                EvaluatorType.QA,
                EvaluatorType.CRITERIA.with_config({
                    "criteria": {
                        "准确性": "回答包含准确的事实信息，没有错误",
                        "完整性": "回答全面地覆盖了问题的各个方面",
                        "简洁性": "回答简洁明了，没有不必要的冗余"
                    }
                })
            ]
        )

        # 运行评估
        result = run_on_dataset(
            dataset_name="AI知识问答评估数据集",
            llm_or_chain_factory=create_chain_with_model,
            evaluation=evaluation_config,
            verbose=False,
            project_name=f"model-comparison-{model_name}"
        )

        results[model_name] = result
        print(f"✅ {model_name} 评估完成")

    # 分析结果
    analyze_comparison_results(results)

    return results


def analyze_comparison_results(results):
    """分析和可视化比较结果"""
    # 提取每个模型的评估分数
    model_scores = {}

    for model_name, result in results.items():
        # 获取结果数据帧
        try:
            df = result.get_aggregate_feedback()

            # 提取关键评分
            model_scores[model_name] = {
                "准确性": df.get("criteria:准确性", {}).get("mean", 0),
                "完整性": df.get("criteria:完整性", {}).get("mean", 0),
                "简洁性": df.get("criteria:简洁性", {}).get("mean", 0),
                "QA分数": df.get("qa", {}).get("mean", 0)
            }
        except Exception as e:
            print(f"分析 {model_name} 结果时出错: {str(e)}")
            model_scores[model_name] = {"准确性": 0, "完整性": 0, "简洁性": 0, "QA分数": 0}

    # 打印比较结果
    print("\n==== 模型性能比较 ====")

    for model_name, scores in model_scores.items():
        print(f"\n{model_name}:")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.2f}")

    # 创建可视化比较
    try:
        metrics = ["准确性", "完整性", "简洁性", "QA分数"]
        model_names = list(model_scores.keys())

        # 创建一个图表
        fig, ax = plt.subplots(figsize=(12, 8))

        # 条形图宽度
        bar_width = 0.2

        # 设置位置
        positions = list(range(len(metrics)))

        # 为每个模型绘制条形图
        for i, model_name in enumerate(model_names):
            values = [model_scores[model_name][metric] for metric in metrics]
            ax.bar([p + i * bar_width for p in positions], values, bar_width, label=model_name)

        # 添加标签和图例
        ax.set_ylabel('分数')
        ax.set_title('不同模型在各评估指标上的性能')
        ax.set_xticks([p + bar_width for p in positions])
        ax.set_xticklabels(metrics)
        ax.legend()

        # 保存图表
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        print("\n✅ 性能比较图已保存为 'model_comparison.png'")

    except Exception as e:
        print(f"创建可视化时出错: {str(e)}")


def analyze_evaluation_results(results):
    """分析评估结果并提供见解"""
    print("\n🔍 分析评估结果...")

    try:
        # 获取聚合反馈
        feedback_df = results.get_aggregate_feedback()

        print("\n==== 评估结果摘要 ====")
        for key, value in feedback_df.items():
            if isinstance(value, dict) and "mean" in value:
                print(f"{key}: {value['mean']:.2f}")

        # 获取详细反馈
        detailed_df = results.get_feedback()

        # 找出表现最好和最差的问题
        if "qa" in detailed_df.columns:
            best_idx = detailed_df["qa"].astype(float).idxmax()
            worst_idx = detailed_df["qa"].astype(float).idxmin()

            print("\n表现最好的问题:")
            print(f"问题: {detailed_df.loc[best_idx, 'input.question']}")
            print(f"回答: {detailed_df.loc[best_idx, 'output.text']}")
            print(f"QA得分: {detailed_df.loc[best_idx, 'qa']}")

            print("\n表现最差的问题:")
            print(f"问题: {detailed_df.loc[worst_idx, 'input.question']}")
            print(f"回答: {detailed_df.loc[worst_idx, 'output.text']}")
            print(f"QA得分: {detailed_df.loc[worst_idx, 'qa']}")

        # 检查不同标准之间的相关性
        criteria_cols = [col for col in detailed_df.columns if col.startswith("criteria:")]
        if len(criteria_cols) >= 2:
            print("\n标准之间的相关性:")
            criteria_data = detailed_df[criteria_cols].astype(float)
            corr = criteria_data.corr()
            print(corr)

        print("\n✅ 评估分析完成")

    except Exception as e:
        print(f"分析结果时出错: {str(e)}")


def custom_feedback_collection():
    """展示如何收集和使用自定义反馈"""
    print("\n🔍 演示自定义反馈收集...")

    # 创建LangSmith客户端
    client = Client()

    # 创建问答链
    chain = create_qa_chain()

    # 准备示例查询
    query = "请解释卷积神经网络的工作原理"

    # 运行链并得到响应
    response = chain.invoke({"question": query})

    # 打印响应
    print("\n问题:", query)
    print("回答:", response["text"])

    # 假设这是一个应用程序，用户可以提供反馈
    print("\n模拟用户提供反馈...")

    # 获取最近的运行（这应该是我们刚刚执行的）
    runs = list(client.list_runs(
        project_name=os.environ["LANGCHAIN_PROJECT"],
        execution_order=1,
        limit=1
    ))

    if runs:
        run_id = runs[0].id

        # 记录正面反馈
        client.create_feedback(
            run_id=run_id,
            key="user_rating",
            value=4,  # 1-5评分
            comment="解释很清晰，但希望有更多具体示例"
        )

        # 记录分类反馈
        client.create_feedback(
            run_id=run_id,
            key="技术准确性",
            value="高",  # 可以是任何值："高"/"中"/"低"
            comment="技术细节准确"
        )

        # 记录二元反馈
        client.create_feedback(
            run_id=run_id,
            key="满足需求",
            value=True,  # 布尔值
            comment="回答满足了我的基本需求"
        )

        print("✅ 用户反馈已记录")

        # 检索反馈
        feedbacks = list(client.list_feedback(run_id=run_id))

        print("\n已收集的反馈:")
        for feedback in feedbacks:
            print(f"- {feedback.key}: {feedback.value} ({feedback.comment})")

    else:
        print("❌ 未找到最近的运行记录")


def main():
    """主函数，运行所有评估示例"""
    print("===== LangSmith 评估示例 =====\n")

    # 检查API密钥
    if os.environ.get("LANGCHAIN_API_KEY") == "your-langsmith-api-key":
        print("⚠️ 警告: 请将代码中的'your-langsmith-api-key'替换为您的实际LangSmith API密钥")
        print("申请API密钥: https://smith.langchain.com/\n")
        return

    # 创建评估数据集
    dataset_id = create_evaluation_dataset()

    # 运行基本评估
    results = run_chain_evaluation(dataset_id)

    # 分析结果
    analyze_evaluation_results(results)

    # 收集自定义反馈
    custom_feedback_collection()

    # 运行模型比较
    # 注意：这可能需要一些时间
    compare_results = compare_models()

    print("\n===== 所有评估示例完成 =====")
    print("登录 https://smith.langchain.com/ 查看详细评估结果")


if __name__ == "__main__":
    main()
