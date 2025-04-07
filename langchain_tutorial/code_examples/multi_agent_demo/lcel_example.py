# LangChain 表达式语言 (LCEL) 示例
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import Dict, List, Any


# 基础链
def simple_chain_example():
    # 1. 定义提示模板
    prompt = PromptTemplate.from_template("向我介绍{topic}")

    # 2. 定义语言模型
    model = ChatOpenAI()

    # 3. 定义解析器
    output_parser = StrOutputParser()

    # 4. 使用LCEL组合成一个链
    chain = prompt | model | output_parser

    # 5. 执行链
    response = chain.invoke({"topic": "LangChain表达式语言"})
    print("\n简单链示例:")
    print(response)

    return chain


# RAG 链
def rag_chain_example():
    # 1. 准备文档
    with open("data/rag_example.txt", "w", encoding="utf-8") as f:
        f.write('''
LangChain 表达式语言 (LCEL) 是一种声明式语言，设计用于组合链和其他组件。
LCEL 的主要优势包括：
1. 简洁的接口: 使用 | 操作符连接组件
2. 异步支持: 提供同步和异步执行选项
3. 流媒体支持: 支持流式响应
4. 批处理支持: 可以高效处理多个输入
5. 可重试性: 内置的重试机制

LCEL 支持多种组件，包括 Runnable, Chain, PromptTemplate, ChatModel 等。
        ''')

    # 2. 加载和分割文档
    loader = TextLoader("data/rag_example.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    splits = text_splitter.split_documents(documents)

    # 3. 创建向量存储
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # 4. 定义RAG提示模板
    template = """使用以下上下文回答问题。如果你不知道答案，就说你不知道。

上下文:
{context}

问题: {question}
"""
    prompt = PromptTemplate.from_template(template)

    # 5. 使用LCEL组合成RAG链
    model = ChatOpenAI(temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )

    # 6. 执行RAG链
    response = rag_chain.invoke("什么是LCEL的主要优势?")
    print("\nRAG链示例:")
    print(response)

    return rag_chain


def main():
    # 执行简单链示例
    simple_chain = simple_chain_example()

    # 执行RAG链示例
    rag_chain = rag_chain_example()


if __name__ == "__main__":
    main()
