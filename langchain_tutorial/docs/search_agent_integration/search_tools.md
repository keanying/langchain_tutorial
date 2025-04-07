# 搜索工具 (Search Tools)

搜索工具是LangChain中的一系列组件，使智能体能够与外部信息源进行交互，从而检索、过滤和处理相关数据。这些工具极大地增强了智能体的能力，使其能够访问实时或特定领域的信息，从而提供更准确、更全面的回答。

## 常见搜索工具

### 1. SerpAPI

SerpAPI是一个搜索引擎结果页面(SERP)API，允许智能体从Google等主要搜索引擎查询信息。

```python
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool

search = SerpAPIWrapper(serpapi_api_key="your-api-key")
search_tool = Tool(
    name="Search",
    func=search.run,
    description="搜索互联网获取有关最新事件或当前信息的信息。例如天气、股票价格、新闻事件等。"
)
```

### 2. DuckDuckGoSearch

一个提供DuckDuckGo搜索功能的工具，无需API密钥。

```python
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import Tool

ddg_search = DuckDuckGoSearchAPIWrapper()
ddg_tool = Tool(
    name="DuckDuckGo Search",
    func=ddg_search.run,
    description="当你需要在互联网上搜索信息时使用DuckDuckGo搜索。"
)
```

### 3. WikipediaQueryRun

专门用于在Wikipedia上搜索和检索信息的工具。

```python
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import Tool

wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="当需要查询百科全书类信息时，使用维基百科搜索。"
)
```

### 4. ArxivQueryRun

用于搜索和获取学术论文和研究的Arxiv工具。

```python
from langchain.utilities import ArxivAPIWrapper
from langchain.agents import Tool

arxiv = ArxivAPIWrapper()
arxiv_tool = Tool(
    name="Arxiv",
    func=arxiv.run,
    description="当需要查找科学论文和研究时使用。输入关键词或主题。"
)
```

## 自定义搜索工具

您可以创建自定义搜索工具来集成特定需求的信息源：

```python
from langchain.agents import Tool
from typing import Callable

def custom_search_function(query: str) -> str:
    # 实现自定义搜索逻辑
    return f"为查询 '{query}' 返回的结果"

custom_tool = Tool(
    name="CustomSearch",
    func=custom_search_function,
    description="当需要在特定领域查找信息时使用此自定义搜索工具。"
)
```

## 搜索工具的最佳实践

1. **提供清晰的描述**：确保每个工具都有精确的描述，以便智能体知道何时使用它。
2. **处理错误和限制**：实现适当的错误处理，处理API限制和超时。
3. **缓存结果**：考虑缓存搜索结果以提高性能并减少API调用。
4. **过滤和处理**：根据需要实现结果过滤和后处理，以提供更有针对性的信息。

通过有效地结合这些搜索工具，您可以大大提高智能体为用户提供准确、及时和相关信息的能力。