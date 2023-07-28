
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import LLMChain
embedding_function = SentenceTransformerEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")


def save_to_chroma(path):
    # 加载文档并将其拆分成块
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=128)
    docs = text_splitter.split_documents(documents)
    # 实例化embedding模型
    embedding_function = SentenceTransformerEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
    # 设置persist_directory可持久化保存至磁盘
    Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")


def search_from_chroma(llm,query):
    # 从磁盘加载
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    docs = db3.similarity_search(query)
    context = docs[0].page_content
    template = """\
    已知信息：
    {context}
    根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。问题是：{query}
    """
    multiple_input_prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=template
    )
    print(multiple_input_prompt.format(context=context, query=query))
    print("-----------------")
    llm_chain = LLMChain(prompt=multiple_input_prompt, llm=llm)
    return llm_chain.run(context=context, query=query, max_length=2048)


def llm():
    model_id = "lmsys/vicuna-7b-v1.3"
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
        # 生成的文本将被限制在4096个标记以内
        max_new_tokens=4096,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


if __name__ == '__main__':
    # save_to_chroma("./刑法.txt")

    # llm = llm()
    # query = input("输入问题：")
    # print(search_from_chroma(llm, query))
    llm = llm()
    while True:
        query = input("请输入你的问题：")
        if query == "exit()":
            break
        print(search_from_chroma(llm, query))
