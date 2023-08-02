import time
from threading import Thread
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain.prompts import (
    PromptTemplate,
)
from langchain import LLMChain
import os
import gradio as gr
import chromadb
from chromadb.config import Settings
import json
from chromadb.config import Settings
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
PERSIST_DIRECTORY = "./chroma_db"
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)


class LocalGptX():
    # 实例化embedding模型

    def __init__(self):
        self.embedding_function = SentenceTransformerEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
        self.local_llm = None
        self.pipe = None

    def save_to_chroma(self, path):
        # 加载文档并将其拆分成块
        loader = TextLoader(path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=32)
        docs = text_splitter.split_documents(documents)

        # 设置persist_directory可持久化保存至磁盘
        Chroma.from_documents(documents=docs, embedding=self.embedding_function,
                              persist_directory=PERSIST_DIRECTORY,
                              client_settings=chromadb.config.Settings(anonymized_telemetry=False))

    def search_from_chroma(self, llm, query):
        # 从磁盘加载
        db3 = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=self.embedding_function,
                     )
        docs = db3.similarity_search(query)
        context = docs[0].page_content
        template = """\
        已知信息：
        {context}
        根据上述已知信息，简洁和专业的来回答用户的问题，并且不要在回答中返回我的提问内容。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。问题是：{query}
        """
        multiple_input_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=template
        )
        logger.info(multiple_input_prompt.format(context=context, query=query))
        logger.info("-----------")
        llm_chain = LLMChain(prompt=multiple_input_prompt, llm=llm)
        return llm_chain.run(context=context, query=query, max_length=2048)

    def llm(self):
        model_id = "lmsys/vicuna-7b-v1.3"
        # model_id = "RicardoLee/Llama2-chat-13B-Chinese-50W"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # 指定模型在哪个设备上运行。设置为"auto"时，会自动选择可用的设备。
            device_map="auto",
            # 指定模型的张量数据类型。设置为torch.float16时，使用16位浮点数进行计算，可以减少内存占用和计算时间
            torch_dtype=torch.float16,
            # 控制模型在CPU上的内存使用。设置为True时，会尽量减少内存使用，但可能会导致性能下降。
            low_cpu_mem_usage=True,
            # 控制是否信任远程代码。设置为True时，表示信任远程代码，可以加快模型加载速度，但可能存在安全风险。
            trust_remote_code=True,

        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,

        )
        local_llm = HuggingFacePipeline(pipeline=pipe)
        self.local_llm = local_llm
        self.pipe = pipe

    def chatllm(self, message, history):
        # result = localgpt.search_from_chroma(self.local_llm, message)
        result = self.pipe(message)
        for i in range(len(result)):
            time.sleep(0.05)
            yield result[:i+1]


if __name__ == '__main__':
    localgpt = LocalGptX()
    if not os.path.exists("./chroma_db"):
        localgpt.save_to_chroma("./刑法.txt")
    localgpt.llm()

    gr.ChatInterface(localgpt.chatllm).queue().launch(server_name="0.0.0.0", server_port=8811)


