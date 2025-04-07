#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain提示词(Prompts)组件使用示例

本模块展示了如何使用LangChain中的各种提示词组件，包括：
1. 基本提示模板(PromptTemplates)
2. 聊天提示模板(ChatPromptTemplates)
3. 少样本学习提示(Few-shot Prompting)
4. 示例选择器(Example Selectors)
5. 提示组合与管道

注意：运行这些示例前，请确保已设置相应的API密钥环境变量
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

# 加载环境变量
from langchain_core.example_selectors import LengthBasedExampleSelector

load_dotenv()

# 引入LangChain中的相关组件
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
# from langchain_core.prompts.example_selector import LengthBasedExampleSelector
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 引入语言模型
from langchain_openai import OpenAI, ChatOpenAI


# ===============================
# 第1部分：基本提示模板
# ===============================
def basic_prompt_templates():
    """基本提示模板示例"""
    print("=== 基本提示模板示例 ===")
    
    # --- 示例1：简单提示模板 ---
    print("\n--- 简单提示模板示例 ---")
    
    # 创建一个提示模板
    template = """你是一位专业的{profession}。请用专业的语言回答以下关于{topic}的问题：{question}"""
    
    # 初始化提示模板对象
    prompt_template = PromptTemplate.from_template(template)
    
    # 或者使用更详细的方式定义
    prompt_template_alt = PromptTemplate(
        input_variables=["profession", "topic", "question"],
        template=template
    )
    
    # 使用模板生成提示
    prompt = prompt_template.format(
        profession="医学专家",
        topic="心脏健康",
        question="每天应该进行多少有氧运动来保持心脏健康？"
    )
    
    print(f"原始模板: {template}")
    print(f"格式化后的提示: {prompt}")
    
    # --- 示例2：部分填充提示模板 ---
    print("\n--- 部分填充提示模板示例 ---")
    
    # 创建一个包含多个变量的提示模板
    complex_template = """作为{role}，请{action}关于{subject}的内容，考虑以下因素：
    1. {factor_1}
    2. {factor_2}
    3. {factor_3}
    请以{format}格式提供你的回复。"""
    
    complex_prompt = PromptTemplate.from_template(complex_template)
    
    # 部分填充模板
    partially_filled = complex_prompt.partial(
        role="数据分析师",
        action="分析",
        format="表格"
    )
    
    print("部分填充后的模板:")
    print(partially_filled.format(
        subject="2023年第一季度销售数据",
        factor_1="地区销售差异",
        factor_2="产品类别表现",
        factor_3="季节性趋势"
    ))
    
    # --- 示例3：使用提示模板与语言模型 ---
    print("\n--- 提示模板与语言模型结合示例 ---")
    
    # 创建一个简单的分析提示
    analysis_template = """请分析以下文本的情感基调（积极、消极或中性）：
    
    文本: {text}
    
    情感分析："""
    
    analysis_prompt = PromptTemplate.from_template(analysis_template)
    
    # 准备一些示例文本
    texts = [
        "今天的会议非常成功，所有人都对新项目感到兴奋。",
        "产品发布被推迟了，客户感到失望。",
        "今天天气多云，气温适中。"
    ]
    
    # 创建模型（仅作演示，如果没有设置API密钥则跳过实际调用）
    try:
        llm = OpenAI(temperature=0)
        
        print("分析示例文本:")
        for text in texts:
            print(f"\n文本: '{text}'")
            # 格式化提示
            prompt = analysis_prompt.format(text=text)
            # 调用模型（如果有API密钥）
            if os.getenv("OPENAI_API_KEY"):
                response = llm.invoke(prompt)
                print(f"分析结果: {response}")
            else:
                print("未设置OPENAI_API_KEY，跳过模型调用")
    except Exception as e:
        print(f"模型调用时出错: {e}")


# ===============================
# 第2部分：聊天提示模板
# ===============================
def chat_prompt_templates():
    """聊天提示模板示例"""
    print("\n=== 聊天提示模板示例 ===")
    
    # --- 示例1：基本聊天提示模板 ---
    print("\n--- 基本聊天提示模板示例 ---")
    
    # 创建聊天提示模板
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一位{role}，专长是{expertise}。"),
        ("user", "{query}")
    ])
    
    # 格式化聊天提示
    messages = chat_template.format_messages(
        role="职业教练",
        expertise="帮助人们找到自己的职业方向",
        query="我在技术和管理之间徘徊，不知道应该选择哪条职业路径。"
    )
    
    print("格式化后的消息:")
    for msg in messages:
        print(f"- {msg.type}: {msg.content}")
    
    # --- 示例2：多轮对话模板 ---
    print("\n--- 多轮对话模板示例 ---")
    
    # 创建包含历史对话的模板
    multi_turn_template = ChatPromptTemplate.from_messages([
        ("system", "你是一位{role}，你的任务是{task}。"),
        ("user", "{user_message_1}"),
        ("assistant", "{assistant_message_1}"),
        ("user", "{user_message_2}")
    ])
    
    # 格式化多轮对话提示
    multi_turn_messages = multi_turn_template.format_messages(
        role="旅游顾问",
        task="为客户提供旅游建议和规划",
        user_message_1="我想在五月份去欧洲旅行两周，预算为3000美元。",
        assistant_message_1="五月是欧洲旅行的好时机。根据您的预算，我建议可以考虑中欧或东欧的国家，如捷克、匈牙利或波兰。您对哪种类型的旅行体验更感兴趣呢？城市观光、历史文化、自然风光，还是美食体验？",
        user_message_2="我对历史文化和美食都很感兴趣，可以推荐一个合适的行程吗？"
    )
    
    print("多轮对话模板:")
    for msg in multi_turn_messages:
        print(f"- {msg.type}: {msg.content[:50]}...")
    
    # --- 示例3：使用聊天提示模板与聊天模型 ---
    print("\n--- 聊天提示模板与聊天模型结合示例 ---")
    
    # 创建一个简单的角色扮演提示
    role_play_template = ChatPromptTemplate.from_messages([
        ("system", "你现在是一位{character}。请以这个角色回应用户。使用{style}的语言风格。"),
        ("user", "{message}")
    ])
    
    # 准备不同的角色扮演场景
    scenarios = [
        {
            "character": "福尔摩斯",
            "style": "推理分析，细致入微",
            "message": "你能帮我解决一个问题吗？我的钥匙不见了。"
        },
        {
            "character": "莎士比亚",
            "style": "诗意盎然，戏剧化",
            "message": "描述一下春天的美丽。"
        }
    ]
    
    # 创建聊天模型（仅作演示，如果没有设置API密钥则跳过实际调用）
    try:
        chat_model = ChatOpenAI(temperature=0.7)
        
        print("角色扮演示例:")
        for scenario in scenarios:
            print(f"\n场景: {scenario['character']}")
            # 格式化聊天提示
            messages = role_play_template.format_messages(**scenario)
            print(f"提示: {scenario['message']}")
            
            # 调用模型（如果有API密钥）
            if os.getenv("OPENAI_API_KEY"):
                response = chat_model.invoke(messages)
                print(f"回复: {response.content[:150]}...")
            else:
                print("未设置OPENAI_API_KEY，跳过模型调用")
                
    except Exception as e:
        print(f"模型调用时出错: {e}")


# ===============================
# 第3部分：少样本学习提示
# ===============================
def few_shot_prompting():
    """少样本学习提示示例"""
    print("\n=== 少样本学习提示示例 ===")
    
    # --- 示例1：基本的少样本学习 ---
    print("\n--- 基本少样本学习示例 ---")
    
    # 定义一些示例
    examples = [
        {"input": "如何做一个苹果派？", "output": "要做苹果派，你需要：1) 准备派皮、苹果、糖和香料；2) 将苹果切片混合糖和香料；3) 将混合物放入派皮；4) 烤箱烘焙直至金黄。"},
        {"input": "如何更换自行车轮胎？", "output": "更换自行车轮胎：1) 使用轮胎撬棒取下旧轮胎；2) 检查轮圈是否有损坏；3) 安装新轮胎和内胎；4) 充气至适当气压。"},
        {"input": "如何设置电子邮件自动回复？", "output": "设置电子邮件自动回复：1) 登录您的电子邮件账户；2) 找到设置或选项菜单；3) 查找自动回复或离开办公室设置；4) 编写回复消息并设置时间段；5) 保存更改。"}
    ]
    
    # 创建示例模板
    example_template = """用户问题: {input}
专家回答: {output}"""
    example_prompt = PromptTemplate.from_template(example_template)
    
    # 创建少样本学习提示模板
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="用户问题: {input}\n专家回答:",
        input_variables=["input"],
        example_separator="\n\n"
    )
    
    # 使用提示模板生成提示
    prompt = few_shot_prompt.format(input="如何在家制作意大利面？")
    
    print("少样本学习提示:")
    print(prompt)
    
    # --- 示例2：使用少样本学习与LLM ---
    print("\n--- 少样本学习与LLM结合示例 ---")
    
    # 创建一个分类任务的少样本学习提示
    classification_examples = [
        {"text": "这款手机性能出色，电池续航时间长，相机拍照效果很好。", "category": "产品评价"},
        {"text": "请问你们的店铺营业时间是几点到几点？", "category": "咨询问题"},
        {"text": "我昨天买的产品有质量问题，想申请退款。", "category": "投诉"},
        {"text": "谢谢你们的优质服务，我非常满意！", "category": "表扬"}
    ]
    
    # 创建示例模板
    class_example_template = "文本: {text}\n类别: {category}"
    class_example_prompt = PromptTemplate.from_template(class_example_template)
    
    # 创建少样本学习提示模板
    class_few_shot_prompt = FewShotPromptTemplate(
        examples=classification_examples,
        example_prompt=class_example_prompt,
        suffix="文本: {text}\n类别:",
        input_variables=["text"],
        example_separator="\n\n"
    )
    
    # 测试文本
    test_texts = [
        "我想了解一下你们产品的退换货政策是怎样的？",
        "我对你们的新产品非常失望，做工粗糙，完全不值这个价格。"
    ]
    
    # 使用语言模型进行分类（仅作演示）
    try:
        llm = OpenAI(temperature=0)
        
        print("文本分类示例:")
        for text in test_texts:
            prompt = class_few_shot_prompt.format(text=text)
            print(f"\n待分类文本: '{text}'")
            
            # 调用模型（如果有API密钥）
            if os.getenv("OPENAI_API_KEY"):
                response = llm.invoke(prompt)
                print(f"分类结果: {response.strip()}")
            else:
                print("未设置OPENAI_API_KEY，跳过模型调用")
                
    except Exception as e:
        print(f"模型调用时出错: {e}")


# ===============================
# 第4部分：示例选择器
# ===============================
def example_selectors():
    """示例选择器示例"""
    print("\n=== 示例选择器示例 ===")
    
    # --- 示例1：基于长度的示例选择器 ---
    print("\n--- 基于长度的示例选择器示例 ---")
    
    # 创建更多的示例数据
    examples = [
        {"input": "如何制作简单的三明治？", "output": "制作简单三明治：1) 准备两片面包；2) 在面包上涂抹酱料；3) 添加喜欢的配料如生菜、奶酪、火腿等；4) 将两片面包合在一起。"},
        {"input": "如何安装Python？", "output": "安装Python：1) 访问python.org；2) 下载适合你操作系统的Python版本；3) 运行安装程序，记得勾选'Add Python to PATH'；4) 完成安装后，打开命令行输入'python --version'验证安装是否成功。"},
        {"input": "如何制作咖啡？", "output": "制作咖啡：1) 烧开水；2) 将咖啡粉放入咖啡滤纸中；3) 将热水慢慢倒在咖啡粉上；4) 等待水流过滤纸进入杯中；5) 根据喜好添加糖或奶。"},
        {"input": "如何写一篇好的博客文章？", "output": "写好博客文章的步骤：1) 选择一个你熟悉且读者感兴趣的主题；2) 研究主题，收集相关信息和数据；3) 创建文章大纲，包括引言、主要内容点和结论；4) 撰写引人入胜的标题和开头段落；5) 展开主要内容，确保逻辑清晰；6) 添加相关的例子、案例或数据支持你的观点；7) 写一个简短有力的结论；8) 校对并修改文章，检查拼写、语法和流畅度；9) 添加相关图片或其他媒体增强可读性；10) 发布并在社交媒体上分享。"},
        {"input": "如何快速入睡？", "output": "快速入睡的方法：1) 建立规律的睡眠时间表；2) 睡前一小时避免使用电子设备；3) 保持睡眠环境舒适、安静和黑暗；4) 尝试深呼吸或冥想放松身心；5) 避免睡前摄入咖啡因或大量食物。"},
        {"input": "如何开始跑步锻炼？", "output": "开始跑步锻炼：1) 购买一双合适的跑鞋；2) 从短距离慢跑开始，如15-20分钟；3) 逐渐增加跑步时间和距离；4) 坚持固定的跑步计划；5) 注意热身和拉伸；6) 给身体足够的恢复时间。"}
    ]
    
    # 创建基于长度的示例选择器
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=PromptTemplate.from_template("{input}\n{output}"),
        max_length=500  # 设置最大长度限制
    )
    
    # 创建使用选择器的少样本学习提示模板
    dynamic_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template("问题: {input}\n回答: {output}"),
        suffix="问题: {input}\n回答:",
        input_variables=["input"]
    )
    
    # 测试不同长度的提示
    short_input = "如何煮鸡蛋？"
    long_input = "如何规划一个为期两周的欧洲旅行，包括预算、交通、住宿和景点安排？"
    
    # 生成提示并查看选择了哪些示例
    print(f"短输入的提示 ('{short_input}'): 选择了 {len(example_selector.select_examples({'input': short_input}))} 个示例")
    short_prompt = dynamic_prompt.format(input=short_input)
    print(f"长输入的提示 ('{long_input}'): 选择了 {len(example_selector.select_examples({'input': long_input}))} 个示例")
    
    print("\n为短输入生成的提示（部分）:")
    print(short_prompt[:300] + "...")


# ===============================
# 第5部分：提示工程实用技巧
# ===============================
def prompt_engineering_tips():
    """提示工程实用技巧"""
    print("\n=== 提示工程实用技巧 ===")
    
    # --- 示例1：链式思考提示 ---
    print("\n--- 链式思考提示示例 ---")
    
    cot_template = """问题: {question}

让我们一步步思考这个问题：
{thinking_steps}

因此，最终答案是："""
    
    cot_prompt = PromptTemplate.from_template(cot_template)
    
    # 示例问题和思考步骤
    question = "一辆汽车以60公里/小时的速度行驶，2小时后又以80公里/小时的速度行驶了1.5小时，总共行驶了多少公里？"
    thinking_steps = """1. 汽车先以60公里/小时的速度行驶了2小时
2. 在这2小时内，行驶的距离 = 60 公里/小时 × 2 小时 = 120 公里
3. 然后汽车以80公里/小时的速度行驶了1.5小时
4. 在这1.5小时内，行驶的距离 = 80 公里/小时 × 1.5 小时 = 120 公里
5. 总行驶距离 = 第一段距离 + 第二段距离 = 120 公里 + 120 公里 = 240 公里"""
    
    prompt = cot_prompt.format(question=question, thinking_steps=thinking_steps)
    print(prompt)
    
    # --- 示例2：提示模板组合 ---
    print("\n--- 提示模板组合示例 ---")
    
    # 创建一个系统提示模板
    system_template = """你是一个专业的{role}，具有{years_experience}年经验。你的风格是{style}。"""
    system_prompt = PromptTemplate.from_template(system_template)
    
    # 创建一个用户提示模板
    user_template = """请你{action}以下主题：{topic}。要求：{requirements}。"""
    user_prompt = PromptTemplate.from_template(user_template)
    
    # 组合两个模板
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])
    
    # 使用组合模板
    messages = chat_prompt.format_messages(
        role="内容创作者",
        years_experience="10",
        style="清晰简洁，富有创意",
        action="写一篇短文介绍",
        topic="人工智能在日常生活中的应用",
        requirements="包含至少3个实际例子，文章长度不超过300字"
    )
    
    print("组合模板生成的消息:")
    for msg in messages:
        print(f"- {msg.type}: {msg.content}")


# ===============================
# 主函数：运行所有示例
# ===============================
def main():
    """运行所有提示词组件示例"""
    print("# LangChain提示词(Prompts)组件使用示例")
    print("=" * 80)
    
    print("\n## 重要提示")
    print("运行这些示例前，请确保已设置OpenAI API密钥。例如：")
    print("```python")
    print('import os')
    print('os.environ["OPENAI_API_KEY"] = "您的OpenAI API密钥"')
    print("```")
    print("或者，您可以使用.env文件存储这些密钥。")
    
    # 运行示例
    basic_prompt_templates()
    chat_prompt_templates()
    few_shot_prompting()
    example_selectors()
    prompt_engineering_tips()
    
    print("\n" + "=" * 80)
    print("示例运行完成！")


if __name__ == "__main__":
    main()