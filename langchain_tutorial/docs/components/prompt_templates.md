# 提示模板 (Prompt Templates)

提示模板是LangChain的核心组件，用于构建发送给语言模型的输入。本文档详细介绍了提示模板的类型、功能和最佳实践。

## 概述

提示工程是使用大型语言模型的关键技术。通过精心设计的提示，可以显著提高模型的输出质量和相关性。LangChain的提示模板系统提供了：

1. **标准化的提示构建方式**：统一接口创建各种提示
2. **变量插值**：动态构建包含用户输入的提示
3. **复用和组合**：模块化提示设计和共享
4. **特定任务优化**：针对不同场景的专用模板

## 提示模板类型

### 1. 字符串提示模板 (StringPromptTemplate)

最基本的提示模板类型，适用于文本补全模型(LLM)：

```python
from langchain_core.prompts import PromptTemplate

# 基本字符串模板
prompt = PromptTemplate(
    template="请为一家{industry}公司起一个名字。",
    input_variables=["industry"]
)

# 格式化模板
formatted_prompt = prompt.format(industry="人工智能")
print(formatted_prompt)  # 输出: 请为一家人工智能公司起一个名字。
```

### 2. 聊天提示模板 (ChatPromptTemplate)

为聊天模型设计的提示模板，支持多种消息类型：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# 基本聊天模板
system_message = "你是一位{role}专家，擅长回答{domain}领域的问题。"
human_message = "{question}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", human_message)
])

# 格式化聊天模板
formatted_messages = chat_prompt.format_messages(
    role="医疗健康",
    domain="营养学",
    question="每天应该摄入多少蛋白质？"
)

# 输出结构化的消息列表:
# [SystemMessage(content="你是一位医疗健康专家，擅长回答营养学领域的问题。"), 
#  HumanMessage(content="每天应该摄入多少蛋白质？")]
```

### 3. 多样化提示模板

**FewShotPromptTemplate** - 用于少样本学习：

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 定义示例
examples = [
    {"input": "今天天气很好", "output": "The weather is nice today"},
    {"input": "我喜欢编程", "output": "I enjoy programming"},
    {"input": "人工智能很有趣", "output": "Artificial intelligence is interesting"}
]

# 创建示例模板
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="输入: {input}\n输出: {output}"
)

# 创建少样本模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="请将以下中文翻译成英文:\n\n",
    suffix="输入: {input}\n输出:",
    input_variables=["input"],
    example_separator="\n\n"
)

# 格式化模板
translate_prompt = few_shot_prompt.format(input="机器学习是人工智能的一个子领域")
```

**PromptTemplate与函数集成** - 调用自定义函数处理输入：

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

# 在格式化时调用函数
date_prompt = PromptTemplate(
    template="今天是{current_date}，请为我生成一个当天的新闻摘要。",
    input_variables=[],
    partial_variables={"current_date": get_current_date}
)

formatted_date_prompt = date_prompt.format()
```

## 高级技巧

### 1. 组合提示模板

将多个提示模板组合成一个复杂模板：

```python
from langchain_core.prompts import PipelinePromptTemplate

# 创建初始格式化模板
full_template = """你是一位{role}。

{format_instructions}

{query}"""

full_prompt = PromptTemplate(
    template=full_template,
    input_variables=["role", "format_instructions", "query"]
)

# 创建用于格式说明的子模板
format_instructions_template = """请用以下格式回答：
{format}"""

format_instructions_prompt = PromptTemplate(
    template=format_instructions_template,
    input_variables=["format"]
)

# 创建管道提示模板
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt,
    pipeline_prompts=[
        ("format_instructions", format_instructions_prompt)
    ]
)

# 格式化管道模板
formatted_pipeline_prompt = pipeline_prompt.format(
    role="数据分析师",
    format="1. 数据概述\n2. 关键趋势\n3. 建议行动",
    query="分析近期股市波动情况"
)
```

### 2. 条件提示模板

基于条件生成不同提示：

```python
from langchain_core.prompts import ConditionalPromptSelector, PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI

# 为不同模型创建不同提示
llm_prompt = PromptTemplate(
    template="请回答: {question}",
    input_variables=["question"]
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位有帮助的AI助手。"),
    ("human", "{question}")
])

# 创建条件选择器
def is_chat_model(llm):
    return isinstance(llm, ChatOpenAI)

prompt_selector = ConditionalPromptSelector(
    default_prompt=llm_prompt,
    conditionals=[(is_chat_model, chat_prompt)]
)

# 根据模型自动选择合适的提示
text_model = OpenAI()
chat_model = ChatOpenAI()

text_prompt = prompt_selector.get_prompt(text_model)
chat_prompt = prompt_selector.get_prompt(chat_model)
```

### 3. 结构化输出提示

创建用于结构化输出的提示模板：

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator

# 定义输出结构
class MovieReview(BaseModel):
    movie_title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    genre: str = Field(description="电影类型")
    rating: float = Field(description="评分（1-10分）")
    review: str = Field(description="简短评价")
    
    @validator("rating")
    def rating_must_be_valid(cls, v):
        if v < 1 or v > 10:
            raise ValueError("评分必须在1到10之间")
        return v

# 创建解析器和提示模板
parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = PromptTemplate(
    template="写一篇关于{movie_name}的电影评论。\n{format_instructions}",
    input_variables=["movie_name"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 格式化提示
formatted_prompt = prompt.format(movie_name="流浪地球")
```

### 4. 提示模板版本控制和复用

管理和复用提示模板：

```python
from langchain_core.prompts import load_prompt, save_prompt

# 保存提示模板
save_prompt(prompt, "./prompts/movie_review_prompt.json")

# 加载提示模板
loaded_prompt = load_prompt("./prompts/movie_review_prompt.json")
```

## 提示模板最佳实践

### 1. 任务指令清晰化

提供明确的任务描述和期望输出：

```python
system_template = """你是一位专业的文本总结专家。你的任务是：
1. 阅读提供的内容
2. 提取关键信息
3. 用简洁的语言总结
4. 保持客观，不添加没有在原文中的信息
5. 总结长度控制在100-150字之间"""

summarization_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "请总结以下内容：\n{text}")
])
```

### 2. 角色设定和情景化

通过角色设定增强模型表现：

```python
role_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位经验丰富的{profession}，擅长{skills}。你的沟通风格是{style}。"),
    ("human", "{question}")
])

scientist_prompt = role_prompt.format_messages(
    profession="生物学家",
    skills="分子生物学和进化论",
    style="专业、准确、易于理解",
    question="请解释DNA复制过程"
)
```

### 3. 使用示例和Few-Shot提示

通过示例引导模型输出：

```python
few_shot_template = """你需要将产品评论情感分类为积极、中立或消极。

评论: 这个产品质量很好，价格也合理。
情感: 积极

评论: 产品外观一般，没有特别出色的地方。
情感: 中立

评论: 收到的产品有破损，而且客服态度很差。
情感: 消极

评论: {review}
情感:"""

few_shot_prompt = PromptTemplate(
    template=few_shot_template,
    input_variables=["review"]
)
```

### 4. 分解复杂任务

将复杂任务分解为简单步骤：

```python
step_by_step_template = """请逐步思考以下问题：{question}

请按照以下步骤进行：
1. 明确问题需要什么信息
2. 列出解决问题所需的关键事实
3. 使用这些事实进行推理
4. 得出结论

一步一步地展示你的思考过程："""

step_by_step_prompt = PromptTemplate(
    template=step_by_step_template,
    input_variables=["question"]
)
```

### 5. 结合结构化输出和验证

要求模型输出特定格式并进行验证：

```python
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import XMLOutputParser

# XML结构化输出
xml_parser = XMLOutputParser()

xml_template = """请分析以下文本的情感，并以XML格式返回结果。

文本：{text}

请使用以下XML格式：
<result>
    <sentiment>积极/中立/消极</sentiment>
    <confidence>0到1之间的数字</confidence>
    <explanation>分析原因的简短解释</explanation>
</result>

你的回答（仅包含XML）："""

xml_prompt = PromptTemplate(
    template=xml_template,
    input_variables=["text"]
)
```

## 提示模板类型参考

### 文本提示模板

- **PromptTemplate**: 基本字符串模板
- **FewShotPromptTemplate**: 少样本学习模板
- **PipelinePromptTemplate**: 通过子模板构建复杂提示

### 聊天提示模板

- **ChatPromptTemplate**: 消息列表格式的模板
- **MessagesPlaceholder**: 在聊天中插入消息列表
- **HumanMessagePromptTemplate**: 人类消息模板
- **AIMessagePromptTemplate**: AI消息模板
- **SystemMessagePromptTemplate**: 系统消息模板

### 特殊提示模板

- **StringPromptTemplate**: 所有文本提示模板的基类
- **BasePromptTemplate**: 所有提示模板的抽象基类
- **ImagePromptTemplate**: 用于多模态输入的模板
- **PromptSelector**/**ConditionalPromptSelector**: 条件选择提示模板

## 提示模板与其他组件集成

### 与语言模型集成

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位简洁的AI助手。"),
    ("human", "{question}")
])
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser
response = chain.invoke({"question": "什么是机器学习？"})
```

### 与记忆组件集成

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.memory import ConversationBufferMemory

# 创建带记忆的提示模板
memory = ConversationBufferMemory(return_messages=True, memory_key="history")
memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位友好的AI助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 第一轮对话
memory.chat_memory.add_user_message("你好！")
memory.chat_memory.add_ai_message("你好！有什么可以帮助你的吗？")

# 第二轮对话，使用历史记录
messages = memory_prompt.format_messages(
    history=memory.load_memory_variables({})["history"],
    input="告诉我我们之前聊了什么？"
)
```

### 与检索系统集成

```python
from langchain_core.prompts import ChatPromptTemplate

# 创建RAG提示模板
retrieval_template = """根据以下上下文回答问题。

上下文：
{context}

问题：{question}

如果上下文中没有足够信息，就说你不知道。"""

retrieval_prompt = ChatPromptTemplate.from_template(retrieval_template)

# 使用检索结果格式化提示
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

retriever_result = retriever.get_relevant_documents("什么是向量数据库？")
context = format_docs(retriever_result)

messages = retrieval_prompt.format_messages(
    context=context,
    question="什么是向量数据库？"
)
```

## 总结

提示模板是LangChain中至关重要的组件，它们将用户输入、系统指令和上下文信息转换为结构化的提示，以获得更好的模型输出。掌握提示模板的设计和使用技巧，是有效使用大型语言模型的关键。

通过本文档，您应该了解了：

1. 不同类型的提示模板及其适用场景
2. 如何创建和格式化基本提示
3. 高级提示技术，如条件提示、少样本学习等
4. 提示模板的最佳实践和设计原则
5. 如何将提示模板与其他LangChain组件集成

正确使用提示模板可以显著提升模型输出的质量和相关性，使您的LangChain应用更加强大和可靠。

## 后续学习

- [模型输入输出](./model_io.md) - 深入了解语言模型
- [输出解析器](./output_parsers.md) - 学习如何处理模型输出
- [链](./chains.md) - 将提示模板集成到复杂工作流中