# 结构化聊天智能体 (Structured Chat Agent)

## 概述

结构化聊天智能体(Structured Chat Agent)是一种专为处理需要严格输入输出格式的交互设计的智能体类型。它结合了ReAct智能体的推理能力和OpenAI函数智能体的结构化特性，使模型能够根据明确定义的模式生成回答，并有效地使用工具。这种智能体特别适合需要处理复杂指令、多步骤任务或需要特定输出格式的场景。

结构化聊天智能体的主要特点是能够处理包含结构化指令和工具使用的复杂提示，使模型生成的回答既满足特定格式要求又能解决用户实际问题。

## 工作原理

结构化聊天智能体的工作流程如下：

1. **指令解析**：智能体首先理解用户指令和所需的输出格式
2. **工具识别**：确定哪些工具适合完成当前任务
3. **结构化思考**：按照预定义的结构组织思考过程
4. **工具调用**：使用适当的工具获取信息或执行操作
5. **格式化输出**：将结果按照要求的格式整理输出

![结构化聊天智能体工作流程](https://python.langchain.com/assets/images/structured_chat-a4349113e9e11923550c046f8039be8a.jpg)

## 优势

- **格式一致性**：确保智能体回答遵循预定义的结构和格式
- **复杂指令处理**：能够理解和执行多步骤指令
- **工具使用灵活**：可以根据需要灵活调用不同工具
- **输出可控**：开发者可以精确控制回答的结构和内容组织
- **适应不同模型**：可以使用各种LLM模型，不仅限于特定模型

## 基本用法示例

以下是创建和使用基本结构化聊天智能体的完整示例：

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun, tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 创建工具集合
tools = [
    DuckDuckGoSearchRun(name="搜索", description="用于在网络上搜索信息"),
    WikipediaQueryRun(name="维基百科", api_wrapper=WikipediaAPIWrapper(), description="用于查询维基百科上的信息")
]

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的AI助手，能够回答问题并使用提供的工具。\
     你的回答应该是有帮助的、有礼貌的，并且考虑到用户的需求。\
     当需要寻找信息时，使用适当的工具。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 创建结构化聊天智能体
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# 创建智能体执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行智能体查询
result = agent_executor.invoke({"input": "苏格拉底是谁？请用一段简短的文字描述他的生平和主要贡献。"})

print(result["output"])
```

## 高级用法

### 自定义输出格式

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# 定义期望的响应结构
response_schemas = [
    ResponseSchema(name="主题概述", description="对主题的简短概述，不超过3句话"),
    ResponseSchema(name="历史背景", description="相关的历史背景和上下文"),
    ResponseSchema(name="重要贡献", description="列出3-5个要点，说明主题的主要贡献或影响"),
    ResponseSchema(name="争议点", description="任何相关的争议或不同观点"),
    ResponseSchema(name="相关资源", description="建议进一步阅读的主题或资源")
]

# 创建输出解析器
output_parser = StructuredOutputParser(response_schemas=response_schemas)

# 获取格式指令
format_instructions = output_parser.get_format_instructions()

# 创建包含格式指令的提示模板
structured_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的AI助手，能够回答问题并使用提供的工具。\
     你的回答必须遵循以下格式：\n{format_instructions}"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 创建结构化聊天智能体
structured_agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=structured_prompt.partial(format_instructions=format_instructions)
)

# 创建智能体执行器
structured_executor = AgentExecutor(agent=structured_agent, tools=tools, verbose=True)

# 执行带有结构化输出的查询
result = structured_executor.invoke({"input": "谁是尼古拉·特斯拉？请提供关于他的信息。"})

print(result["output"])

# 解析结构化输出
try:
    parsed_output = output_parser.parse(result["output"])
    
    print("\n结构化输出：")
    for key, value in parsed_output.items():
        print(f"\n{key}:\n{value}")
 except Exception as e:
    print(f"解析错误: {e}")
    print("原始输出：\n", result["output"])
```

### 自定义工具与多重指令处理

```python
from typing import List, Dict
from langchain.tools import tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

# 定义结构化输入
class ArticleGenerationInput(BaseModel):
    topic: str = Field(..., description="文章的主题")
    style: str = Field(..., description="写作风格，如'学术'、'通俗'、'新闻'等")
    length: str = Field(..., description="文章长度，如'短'、'中'、'长'")

@tool
def generate_article(topic: str, style: str, length: str) -> str:
    """根据指定的主题、风格和长度生成文章"""
    # 在实际应用中，这里可能会调用更复杂的内容生成逻辑
    length_words = {"短": "300字", "中": "800字", "长": "1500字"}
    return f"已为您生成一篇关于'{topic}'的{style}风格{length_words.get(length, '中等长度')}文章。[文章内容省略]"

# 创建自定义数据分析工具
class DataAnalysisInput(BaseModel):
    data_source: str = Field(..., description="数据源，如'销售数据'、'用户反馈'等")
    analysis_type: str = Field(..., description="分析类型，如'趋势分析'、'相关性分析'等")
    time_period: str = Field(default="最近30天", description="分析的时间段")

@tool
def analyze_data(data_source: str, analysis_type: str, time_period: str) -> str:
    """执行数据分析任务"""
    # 在实际应用中，这里会连接到真实的数据分析系统
    return f"已完成对'{time_period}'的'{data_source}'进行'{analysis_type}'。分析结果显示[...]"

# 创建高级工具集
advanced_tools = [
    DuckDuckGoSearchRun(name="搜索"),
    StructuredTool.from_function(
        func=generate_article,
        name="文章生成器",
        description="根据指定的主题、风格和长度生成文章",
        args_schema=ArticleGenerationInput
    ),
    StructuredTool.from_function(
        func=analyze_data,
        name="数据分析",
        description="执行各种类型的数据分析任务",
        args_schema=DataAnalysisInput
    )
]

# 创建高级结构化聊天智能体
advanced_agent = create_structured_chat_agent(
    llm=ChatOpenAI(model="gpt-4", temperature=0.2),  # 使用更强大的模型
    tools=advanced_tools,
    prompt=prompt
)

advanced_executor = AgentExecutor(agent=advanced_agent, tools=advanced_tools, verbose=True)

# 处理复杂多指令任务
complex_task = """
请帮我完成以下任务：
1. 搜索人工智能在医疗领域的最新应用
2. 生成一篇关于"AI辅助医疗诊断"的学术风格短文
3. 分析"用户反馈"数据，进行"情感分析"，时间段为"最近90天"

请依次完成这些任务，并为每个任务提供清晰的结果。
"""

result = advanced_executor.invoke({"input": complex_task})
print(result["output"])
```

## 实际应用场景

### 1. 多步骤任务助手

```python
from langchain.memory import ConversationBufferMemory

# 创建带记忆的提示模板
memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的任务助手，善于将复杂任务分解为步骤并逐步执行。\
     使用提供的工具来完成任务，并在每个步骤后报告进度。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 初始化对话记忆
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 创建任务助手智能体
task_assistant = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=memory_prompt
)

# 创建带记忆的执行器
task_executor = AgentExecutor(
    agent=task_assistant,
    tools=tools,
    memory=memory,
    verbose=True
)

# 示例：处理一个多步骤研究任务
research_task = """我需要为一个关于气候变化的演讲做准备。请帮我：
1. 找出过去10年全球平均温度变化的数据
2. 查找哪些国家做出了最积极的减排承诺
3. 搜索一些应对气候变化的创新技术方案

请逐步完成这些任务，每完成一步就告诉我结果。
"""

result = task_executor.invoke({"input": research_task})
print(result["output"])

# 继续对话，提出后续问题
follow_up = "基于这些信息，帮我总结出三个最有前景的气候变化解决方案。"
result2 = task_executor.invoke({"input": follow_up})
print(result2["output"])
```

### 2. 结构化报告生成器

```python
from datetime import datetime

# 创建用于生成报告的响应模式
report_schemas = [
    ResponseSchema(name="标题", description="报告的主标题"),
    ResponseSchema(name="摘要", description="报告内容的简短摘要，不超过100字"),
    ResponseSchema(name="主要发现", description="列出3-5个要点，说明主要发现"),
    ResponseSchema(name="数据分析", description="对相关数据的简要分析"),
    ResponseSchema(name="结论与建议", description="基于发现和分析的结论和建议"),
    ResponseSchema(name="参考来源", description="引用的信息来源")
]

report_parser = StructuredOutputParser(response_schemas=report_schemas)
report_format_instructions = report_parser.get_format_instructions()

# 创建报告生成器提示
report_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的报告生成专家。
    你的任务是根据用户的要求，收集信息并生成结构化的报告。
    报告必须按照以下格式：
    {format_instructions}
    
    确保报告内容全面、准确、专业，并引用可靠来源。"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 创建报告生成智能体
report_agent = create_structured_chat_agent(
    llm=ChatOpenAI(model="gpt-4", temperature=0.1),
    tools=tools,
    prompt=report_prompt.partial(format_instructions=report_format_instructions)
)

report_executor = AgentExecutor(agent=report_agent, tools=tools, verbose=True)

# 生成报告示例
report_request = "生成一份关于电动汽车市场现状的报告，包括主要厂商、技术趋势和未来发展方向。"

report_result = report_executor.invoke({"input": report_request})
print(report_result["output"])

# 解析结构化报告
try:
    parsed_report = report_parser.parse(report_result["output"])
    
    # 格式化输出报告
    print("\n" + "="*50)
    print(f"\n{parsed_report['标题'].upper()}\n")
    print(f"生成日期: {datetime.now().strftime('%Y-%m-%d')}\n")
    print(f"摘要:\n{parsed_report['摘要']}\n")
    print("主要发现:")
    findings = parsed_report['主要发现'].strip().split('\n')
    for i, finding in enumerate(findings):
        print(f"  {i+1}. {finding.lstrip('- ')}")    
    print(f"\n数据分析:\n{parsed_report['数据分析']}\n")
    print(f"结论与建议:\n{parsed_report['结论与建议']}\n")
    print(f"参考来源:\n{parsed_report['参考来源']}")
    print("\n" + "="*50)
    
except Exception as e:
    print(f"解析报告时出错: {e}")
    print("原始报告输出:\n", report_result["output"])
```

## 最佳实践

1. **明确定义输出结构**：使用响应模式和输出解析器明确定义您期望的输出格式。

2. **提供详细的工具描述**：为每个工具提供清晰、具体的描述和使用示例。

3. **分阶段处理复杂任务**：引导智能体将复杂任务分解为可管理的步骤。

4. **平衡自由度和结构**：在提供结构的同时，给予智能体足够的自由度来解决问题。

5. **错误处理与验证**：实现输出验证和错误处理机制，确保结果符合预期格式。

   ```python
   def validate_agent_output(output, parser):
       try:
           parsed_output = parser.parse(output)
           return True, parsed_output
       except Exception as e:
           return False, f"格式错误: {str(e)}"
   ```

6. **迭代改进提示**：根据智能体的实际表现，不断优化提示模板。

7. **组合不同类型的工具**：结合使用搜索工具、计算工具和专业工具，扩展智能体的能力。

## 与其他智能体类型的对比

| 特性 | 结构化聊天智能体 | ReAct智能体 | OpenAI函数智能体 |
|------|-----------------|-------------|----------------|
| 输出格式控制 | 强 | 弱 | 中 |
| 多步骤任务处理 | 强 | 中 | 中 |
| 适用模型范围 | 广泛 | 广泛 | 仅OpenAI支持函数调用的模型 |
| 可定制性 | 高 | 高 | 受限于函数格式 |
| 开发复杂度 | 中 | 低 | 低 |
| 错误处理 | 需自定义实现 | 可能需要多次尝试 | 较好的内置处理 |

## 常见问题与解决方案

1. **输出不符合预期格式**
   - 解决方案：强化格式指令，提供更明确的示例
   - 使用更强大的LLM模型（如GPT-4）

2. **工具使用不当**
   - 解决方案：改进工具描述，使其更明确
   - 为工具提供使用示例

3. **处理复杂任务时出现混淆**
   - 解决方案：明确指示智能体分步骤处理任务
   - 使用记忆组件保持上下文

4. **输出过于冗长或简短**
   - 解决方案：在提示中明确指定所需详细程度
   - 使用输出模式限制响应长度

## 总结

结构化聊天智能体是一种强大的智能体类型，结合了灵活性和结构化输出的优势。它特别适合需要多步骤任务处理、结构化报告生成或复杂指令遵循的应用场景。通过定义明确的输出结构、提供详细的工具描述并遵循最佳实践，开发者可以创建既灵活又可靠的智能体系统，满足各种复杂的用户需求。

结构化聊天智能体的可定制性使其成为许多应用场景的理想选择，从简单的信息检索到复杂的多步骤任务处理，都能展现出色的性能。通过持续优化提示和结构，开发者可以不断提升智能体的性能和用户体验。
