# !/usr/bin/env python
# -*-coding:utf-8 -*-
# Time: 2025/3/16 20:47
# FileName: 汤头歌诀知识库
# Project: LangchainStudy
# Author: JasonLai
# Email: jasonlaihj@163.com
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from exts import *


# 加载文档
def load_document(file):
    """
    加载文档
    :param file: 文档路径
    :return: 文档加载器对象
    """
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        logger.info(f'正在加载 {file}')
        loader = PyPDFLoader(file)
        return loader

    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        logger.info(f'正在加载 {file}')
        loader = Docx2txtLoader(file)
        return loader

    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
        return loader

    else:
        logger.info('注意！加载的文档格式不支持！')
        return None


# 加载文档数据
data = load_document(r"D:\Resource\汤头歌诀白话解.txt").load()

# 导入分割数据模块
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建分割器，chunk_size表示每个数据块的词元个数，chunk_overlap表示允许重复的词元个数，汤头歌诀的每个中药汤方大概1000字符左右，每种汤方重复的词元个数极低，所以设置50个词元重复个数
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# 分割文档数据，变成多个块
chunks = splitter.split_documents(data)

# 存储到向量数据库,需要安装开源向量数据库比如chroma： pip install langchain-chroma
from langchain_chroma import Chroma

"""
Embeddings 类是一个用于与文本嵌入模型接口的类。有很多嵌入大模型供应商（OpenAI、Cohere、Hugging Face 等） - 这个类旨在为它们提供一个标准接口。
嵌入会创建一段文本的向量表示。这是有用的，因为这意味着我们可以在向量空间中思考文本，并进行语义搜索，寻找在向量空间中最相似的文本片段.
"""

from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(model="text-embedding-v2")

# 将分割数据嵌入到向量空间，进行储存
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

# 检索器
retriever = vectorstore.as_retriever()

# 系统提示模板
s_prompt = """
你是一个做问答的助手，使用一下检索到的上下文片段来回答问题，如果你不知道答案，就说你不知道。最多使用三句话，并保持答案简洁。\n{context}
"""

# 创建提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", s_prompt),
        ("human", "{input}")
    ]
)

# 定制专用的模型对象
tongyi_model = ChatTongyi(top_p=0.2)

# 创建用户提问的chain
user_question_chain = create_stuff_documents_chain(tongyi_model, prompt)

# 创建带有检索器的chain
retriever_chain = create_retrieval_chain(retriever, user_question_chain)

# 用户提问
user_ask = {'input': "劳热久嗽的症状选用哪个汤最好？"}

# 运行提问
resp = retriever_chain.invoke(user_ask)

print(resp.get('answer'))