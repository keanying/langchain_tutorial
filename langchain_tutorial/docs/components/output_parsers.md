# 输出解析器 (Output Parsers)

输出解析器是LangChain的重要组件，用于将语言模型的原始文本输出转换为结构化数据。本文档详细介绍了输出解析器的类型、功能和最佳实践。

## 概述

语言模型通常输出自然语言文本，但在应用开发中，我们经常需要将这些文本转换为结构化的数据格式，如列表、字典或对象。输出解析器实现了这一关键功能，提供了：

1. **结构化数据提取**：将自然语言转换为特定数据结构
2. **格式一致性保证**：确保输出符合预期格式
3. **容错处理**：处理模型输出不符合要求的情况
4. **模型指导**：向模型提供正确输出格式的指导

## 基础解析器

### 1. 字符串解析器 (StrOutputParser)

最简单的解析器，用于提取模型返回的原始文本：

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 创建简单链
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("解释{topic}是什么？")
parser = StrOutputParser()

# 组合组件
chain = prompt | model | parser

# 调用链
result = chain.invoke({"topic": "量子计算"})

# result 是一个字符串，包含模型对量子计算的解释
```

### 2. 列表解析器 (CommaSeparatedListOutputParser)

将逗号分隔的文本转换为Python列表：

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# 创建列表解析器
list_parser = CommaSeparatedListOutputParser()

# 创建带格式说明的提示模板
format_instructions = list_parser.get_format_instructions()
list_prompt = ChatPromptTemplate.from_template(
    "列出{topic}的五个最重要特点。\n{format_instructions}"
)

# 组合组件
list_chain = list_prompt | model | list_parser

# 调用链
topic_features = list_chain.invoke({
    "topic": "人工智能", 
    "format_instructions": format_instructions
})

# 结果是一个Python列表: ['特点1', '特点2', '特点3', '特点4', '特点5']
```

### 3. JSON解析器 (JsonOutputParser)

将JSON格式文本转换为Python字典或列表：

```python
from langchain_core.output_parsers import JsonOutputParser

# 创建JSON解析器
json_parser = JsonOutputParser()

# 创建带格式说明的提示模板
json_format_instructions = json_parser.get_format_instructions()
json_prompt = ChatPromptTemplate.from_template(
    "生成一个包含{person}基本信息的JSON。应包括姓名、职业、年龄和技能列表。\n{format_instructions}"
)

# 组合组件
json_chain = json_prompt | model | json_parser

# 调用链
person_info = json_chain.invoke({
    "person": "爱因斯坦",
    "format_instructions": json_format_instructions
})

# 结果是一个Python字典，包含爱因斯坦的信息
```

## 高级解析器

### 1. Pydantic解析器 (PydanticOutputParser)

使用Pydantic模型定义输出结构：

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List

# 定义Pydantic模型
class Movie(BaseModel):
    title: str = Field(description="电影标题")
    director: str = Field(description="导演姓名")
    year: int = Field(description="上映年份")
    genre: List[str] = Field(description="电影类型")
    rating: float = Field(description="评分（1-10）")
    
    @validator("rating")
    def rating_must_be_valid(cls, v):
        if v < 1 or v > 10:
            raise ValueError("评分必须在1到10之间")
        return v

# 创建Pydantic解析器
pydantic_parser = PydanticOutputParser(pydantic_object=Movie)

# 创建带格式说明的提示模板
format_instructions = pydantic_parser.get_format_instructions()
pydantic_prompt = ChatPromptTemplate.from_template(
    "生成一部{genre}电影的信息。\n{format_instructions}"
)

# 组合组件
pydantic_chain = pydantic_prompt | model | pydantic_parser

# 调用链
movie_data = pydantic_chain.invoke({
    "genre": "科幻",
    "format_instructions": format_instructions
})

# 结果是一个Movie对象，包含电影信息
```

### 2. XML解析器 (XMLOutputParser)

解析XML格式的输出：

```python
from langchain_core.output_parsers import XMLOutputParser

# 创建XML解析器
xml_parser = XMLOutputParser()

# 创建带格式说明的提示模板
xml_format = """<analysis>
    <sentiment>positive or negative</sentiment>
    <language>detected language</language>
    <summary>brief summary</summary>
</analysis>"""

xml_prompt = ChatPromptTemplate.from_template(
    "分析以下文本的情感、语言和内容：\n{text}\n\n以下面的XML格式输出:\n{xml_format}"
)

# 组合组件
xml_chain = xml_prompt | model | xml_parser

# 调用链
xml_result = xml_chain.invoke({
    "text": "人工智能正在迅速发展，为各行各业带来革命性变化。",
    "xml_format": xml_format
})

# 结果是一个包含解析XML数据的字典
```

### 3. 自定义解析器

创建自定义输出解析器：

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import Dict, Any

class CustomKeyValueParser(BaseOutputParser[Dict[str, Any]]):
    """解析形如'key: value'的文本"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """从文本中解析键值对"""
        result = {}
        lines = text.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
        
        return result
    
    def get_format_instructions(self) -> str:
        """提供格式指导给模型"""
        return """请以'键: 值'的格式返回信息，每行一个键值对。
例如：
名称: 爱因斯坦
职业: 物理学家
贡献: 相对论"""

# 使用自定义解析器
custom_parser = CustomKeyValueParser()

custom_prompt = ChatPromptTemplate.from_template(
    "提供关于{person}的基本信息。\n{format_instructions}"
)

custom_chain = custom_prompt | model | custom_parser

result = custom_chain.invoke({
    "person": "牛顿",
    "format_instructions": custom_parser.get_format_instructions()
})
```

## 组合解析器

### 1. 多重解析 (RouterOutputParser)

根据内容选择不同的解析器：

```python
from langchain_core.output_parsers import RouterOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

class Person(BaseModel):
    name: str
    age: int

class Company(BaseModel):
    name: str
    industry: str
    employees: int

# 创建组合解析器
router_parser = PydanticToolsParser(tools=[
    Person,
    Company
])

# 创建提示模板
router_prompt = ChatPromptTemplate.from_template(
    "根据查询决定是返回人物信息还是公司信息。\n查询: {query}"
)

# 组合组件
router_chain = router_prompt | model.bind_tools([Person, Company]) | router_parser

# 人物查询
person_result = router_chain.invoke({"query": "告诉我关于比尔盖茨的信息"})
# 返回Person对象

# 公司查询
company_result = router_chain.invoke({"query": "告诉我关于微软的信息"})
# 返回Company对象
```

### 2. 错误处理与重试 (OutputFixingParser)

处理解析错误并尝试修复：

```python
from langchain_core.output_parsers import OutputFixingParser

# 创建基础解析器（容易出错的格式）
base_parser = PydanticOutputParser(pydantic_object=Movie)

# 创建带修复功能的解析器
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI()
)

# 即使模型输出不完全符合格式要求，也能尝试修复并解析
try_chain = prompt | model | fixing_parser
```

## 构建复杂解析策略

### 1. 结构化提取与转换

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from typing import List, Optional

# 定义多级数据结构
class Author(BaseModel):
    name: str = Field(description="作者姓名")
    background: Optional[str] = Field(description="作者背景", default=None)

class Book(BaseModel):
    title: str = Field(description="书名")
    year: int = Field(description="出版年份")
    author: Author = Field(description="作者信息")
    genres: List[str] = Field(description="图书类型")
    summary: str = Field(description="内容简介")

# 创建解析器和提示
book_parser = PydanticOutputParser(pydantic_object=Book)
instructions = book_parser.get_format_instructions()

book_prompt = ChatPromptTemplate.from_template(
    "提供关于{book_title}的详细信息，包括作者、出版年份、类型和简介。\n{format_instructions}"
)

# 处理结果的函数
def process_book(book: Book) -> dict:
    # 计算出版至今年数
    import datetime
    current_year = datetime.datetime.now().year
    years_since_pub = current_year - book.year
    
    return {
        "book": book,
        "years_since_publication": years_since_pub,
        "is_classic": years_since_pub > 50
    }

# 组合处理链
book_chain = book_prompt | model | book_parser | process_book

# 调用链
result = book_chain.invoke({
    "book_title": "战争与和平",
    "format_instructions": instructions
})
```

### 2. 条件解析逻辑

```python
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import JsonOutputParser, XMLOutputParser

# 创建多格式解析器
def is_json(text):
    return text.strip().startswith('{')

def is_xml(text):
    return text.strip().startswith('<')

# 创建条件分支解析器
multi_format_parser = RunnableBranch(
    (is_json, JsonOutputParser()),
    (is_xml, XMLOutputParser()),
    StrOutputParser()  # 默认解析器
)

# 在链中使用
flexible_chain = prompt | model | multi_format_parser
```

## 输出格式化技巧

### 1. 明确的格式指令

提供详细的格式说明，帮助模型生成可解析的输出：

```python
from langchain_core.output_parsers.json import JsonOutputParser

json_parser = JsonOutputParser()
instructions = json_parser.get_format_instructions()

detailed_prompt = """生成一个包含电影信息的JSON对象。必须包含以下字段：
- title: 电影标题 (字符串)
- director: 导演姓名 (字符串)
- year: 上映年份 (整数)
- genres: 电影类型 (字符串数组)
- rating: 评分，1-10之间 (数值)

{format_instructions}

电影: {movie}"""

prompt = ChatPromptTemplate.from_template(detailed_prompt)
chain = prompt | model | json_parser

result = chain.invoke({"movie": "黑客帝国", "format_instructions": instructions})
```

### 2. 使用Enum限制选项

使用枚举类型限制可能的值：

```python
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class SentimentAnalysis(BaseModel):
    text: str = Field(description="分析的文本")
    sentiment: Sentiment = Field(description="文本情感 (positive, neutral, negative)")
    confidence: float = Field(description="置信度 (0.0-1.0)")
    
    class Config:
        use_enum_values = True

# 创建解析器
sentiment_parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)
```

### 3. 分步解析复杂输出

对于复杂任务，先将输出分解为简单部分：

```python
from langchain_core.output_parsers import StrOutputParser

# 第一步：生成文本分析
analysis_prompt = ChatPromptTemplate.from_template("分析以下文本: {text}")
analysis_chain = analysis_prompt | model | StrOutputParser()

# 第二步：从分析中提取结构化数据
structure_prompt = ChatPromptTemplate.from_template(
    """基于下面的分析，提取主要观点和情感评分(1-10)，以JSON格式返回:
    
    分析: {analysis}
    
    JSON格式:
    {{
      "main_points": ["观点1", "观点2", ...],
      "sentiment_score": 评分
    }}
    """
)
structure_chain = structure_prompt | model | JsonOutputParser()

# 组合两个步骤
full_chain = {"analysis": analysis_chain} | structure_chain

result = full_chain.invoke({"text": "这个产品质量很好，但价格有点贵。客服态度也不错，总体来说是不错的购物体验。"})
```

## 解析器验证与错误处理

### 1. 使用验证逻辑

Pydantic模型中添加验证器：

```python
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List

class Product(BaseModel):
    name: str = Field(description="产品名称")
    price: float = Field(description="产品价格")
    features: List[str] = Field(description="产品特点列表")
    rating: int = Field(description="产品评分(1-5)")
    
    @validator("price")
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("价格必须为正数")
        return v
    
    @validator("rating")
    def rating_must_be_valid(cls, v):
        if v < 1 or v > 5:
            raise ValueError("评分必须在1到5之间")
        return v
```

### 2. 错误处理策略

```python
from langchain_core.output_parsers import OutputFixingParser
from langchain_core.runnables import RunnablePassthrough

# 创建基础解析器
base_parser = PydanticOutputParser(pydantic_object=Product)

# 添加错误处理
robust_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI()
)

# 实现错误捕获逻辑
def safe_parse(text):
    try:
        return robust_parser.parse(text)
    except Exception as e:
        return {"error": str(e), "raw_text": text}

# 在链中使用安全解析
robust_chain = prompt | model | RunnablePassthrough.assign(
    parsed_output=lambda x: safe_parse(x.content)
)
```

## 与其他组件集成

### 1. 在智能体中使用解析器

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool

# 定义工具函数
@tool
def search_products(query: str) -> str:
    """搜索产品数据库"""
    # 假设实现
    return "找到3个产品..."

# 定义输出格式
class SearchResult(BaseModel):
    products: List[str] = Field(description="产品列表")
    top_pick: Optional[str] = Field(description="最佳推荐")
    reason: Optional[str] = Field(description="推荐理由")

search_parser = PydanticOutputParser(pydantic_object=SearchResult)

# 创建智能体
tools = [search_products]
agent = create_openai_tools_agent(
    model, 
    tools, 
    """你是一位产品专家，帮助用户查找产品。
    最终输出必须是JSON格式，包含产品列表和推荐。
    {format_instructions}""".format(
        format_instructions=search_parser.get_format_instructions()
    )
)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 处理用户查询
response = agent_executor.invoke({"input": "查找价格低于100元的手机配件"})
structured_result = search_parser.parse(response["output"])
```

### 2. 与检索系统集成

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 假设已经有一个向量数据库
vector_db = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vector_db.as_retriever()

# 定义输出格式
class RetrievalSummary(BaseModel):
    question: str = Field(description="用户问题")
    sources: List[str] = Field(description="信息来源")
    answer: str = Field(description="基于来源的回答")
    confidence: float = Field(description="置信度 (0.0-1.0)")

# 创建解析器
rag_parser = PydanticOutputParser(pydantic_object=RetrievalSummary)

# 格式化检索文档
def format_docs(docs):
    return "\n\n".join([f"来源 {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])

# 创建RAG链
rag_chain = {
    "question": lambda x: x["question"],
    "sources": lambda x: format_docs(retriever.get_relevant_documents(x["question"]))
} | ChatPromptTemplate.from_template(
    """基于以下来源回答问题。如果来源中没有答案，就说你不知道。
    
    问题: {question}
    
    来源:
    {sources}
    
    {format_instructions}
    """
) | model | rag_parser
```

## 最佳实践

1. **提供清晰的格式指南**：明确告诉模型预期的输出格式
2. **使用枚举和验证**：限制可能的值范围，提高输出一致性
3. **实施错误处理**：针对解析失败的情况有备份策略
4. **逐步处理复杂输出**：将复杂解析任务分解为多个步骤
5. **考虑格式的复杂性**：在详细结构和容错性之间找到平衡
6. **测试不同的提示形式**：找到最能产生一致可解析输出的提示方式

## 常见陷阱与解决方案

1. **模型忽略格式指南**
   - 解决方案：将格式指南放在提示的最后，添加明显的分隔符

2. **嵌套数据结构解析失败**
   - 解决方案：将复杂结构分解为多个简单步骤，逐步构建

3. **不一致的格式**
   - 解决方案：使用OutputFixingParser自动修复轻微格式问题

4. **无法处理长输出**
   - 解决方案：实现分块处理策略，将长输出分段解析

## 总结

输出解析器是LangChain中的关键组件，它们将语言模型的自然语言输出转换为结构化数据，使其可以被应用程序有效处理。通过选择合适的解析器和实施适当的格式提示，可以显著提高语言模型输出的一致性和可用性。

无论是简单的列表和字典，还是复杂的嵌套结构，输出解析器都提供了强大的工具集来处理各种数据类型和格式需求。通过与LangChain的其他组件（如模型、提示模板和链）结合，输出解析器成为构建高效可靠的LLM应用程序的基石。

## 后续学习

- [提示模板](./prompt_templates.md) - 学习如何设计有效的提示来配合解析器
- [模型输入输出](./model_io.md) - 了解更多关于模型交互的信息
- [链](./chains.md) - 探索如何将解析器集成到更复杂的工作流中