import os
import pickle
import re
from pathlib import Path

import pymupdf
from huggingface_hub import snapshot_download
from kiwipiepy import Kiwi
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


def initialize():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    dirs = ["./models", "./faiss_db", "./bm25"]

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    download_embed_model()


def download_embed_model(repo_id="nlpai-lab/KURE-v1", local_dir="./models/kure-v1"):
    snapshot_download(repo_id=repo_id, local_dir=local_dir)


def get_embedder(local_dir="./models/kure-v1"):
    model_name = local_dir
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}

    embedder = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    return embedder


def load_docs(file_dir: str):
    loader = pymupdf.open(file_dir)
    documents = []

    for page_num, page in enumerate(loader):
        text = page.get_text()

        metadata = {
            "source": file_dir.split("/")[-1],
            "format": "PDF",
            "total_pages": len(loader),
            "page": page_num + 1,
        }

        doc_obj = Document(page_content=text, metadata=metadata)
        documents.append(doc_obj)

    loader.close()

    return documents


def chunk_docs(documents):
    docs_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = docs_splitter.split_documents(documents)

    return split_docs


def save_db(split_docs, local_faiss_dir="./faiss_db"):
    embedder = get_embedder()

    if any(Path(local_faiss_dir).glob("*.pkl")):
        vectorstore = FAISS.load_local(
            folder_path=local_faiss_dir,
            embeddings=embedder,
            allow_dangerous_deserialization=True,
        )

        vectorstore.add_documents(split_docs)
        vectorstore.save_local(local_faiss_dir)

    else:
        vectorstore = FAISS.from_documents(
            documents=split_docs,
            embedding=embedder,
            distance_strategy=DistanceStrategy.COSINE,
        )
        vectorstore.save_local(local_faiss_dir)


def kiwi_tokenize(text):
    kiwi = Kiwi()

    return [token.form for token in kiwi.tokenize(text)]


def save_bm25(split_docs, local_bm25_dir="./bm25/bm25_retriever.pkl"):
    if any(Path("./bm25").glob("*.pkl")):
        with open(local_bm25_dir, "rb") as f:
            old_bm25_retriever = pickle.load(f)

        merge_split_docs = old_bm25_retriever.docs + split_docs

        bm25_retriever = BM25Retriever.from_documents(merge_split_docs, k=4, preprocess_func=kiwi_tokenize)

        with open(local_bm25_dir, "wb") as f:
            pickle.dump(bm25_retriever, f)

    else:
        bm25_retriever = BM25Retriever.from_documents(split_docs, k=4, preprocess_func=kiwi_tokenize)

        with open(local_bm25_dir, "wb") as f:
            pickle.dump(bm25_retriever, f)


def get_bm25_retreiever(local_bm25_dir="./bm25/bm25_retriever.pkl"):
    with open(local_bm25_dir, "rb") as f:
        bm25_retriever = pickle.load(f)

    return bm25_retriever


def service_init():
    initialize()

    if not (any(Path("./faiss_db").glob("*.pkl"))):
        create_database()


def create_database(local_pdf_dir="./input"):
    path = Path(local_pdf_dir)

    pdf_files = [f.name for f in path.glob("*.pdf")]
    split_documents = []

    for file in pdf_files:
        documents = load_docs(local_pdf_dir + "/" + file)
        print(local_pdf_dir + "/" + file)
        split_docs = chunk_docs(documents)
        split_documents.append(split_docs)
        save_db(split_docs)

    for split_docs in split_documents:
        save_bm25(split_docs)


def get_retriever(local_faiss_dir="./faiss_db"):
    embedder = get_embedder()

    vectorstore = FAISS.load_local(
        folder_path=local_faiss_dir,
        embeddings=embedder,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = get_bm25_retreiever()

    ensemble_retriever = EnsembleRetriever(retrievers=[retriever, bm25_retriever], weights=[0.6, 0.4])

    return ensemble_retriever


def get_llm():
    llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    return llm


def chat_completions(messages: list, model="local-model"):
    llm = get_llm()
    response = llm.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )

    return response


def get_retrieved_context(user_question):
    retriever = get_retriever()
    retrieved_document = retriever.invoke(user_question)
    retrieved_context = str(retrieved_document)

    return retrieved_context


class DocumentSentence(BaseModel):
    title: str = Field(description="사용자 질문에서 추출된 키워드")
    sentence: str = Field(description="키워드에 대한 설명")


def extract_sentence_chain(user_question, retrieved_context):
    pattern = re.compile(r"[^ 가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z\?!\.]")
    invalid_chars = pattern.findall(user_question)

    if len(invalid_chars) > 0:
        return ""

    system_prompt = """
    당신은 사용자가 제공한 문서에서 사용자의 질문의 답변에 해당하는 문장을 추출해주는 어시스턴트 AI입니다.
    문서와 관련이 없으면 질문에 대해서는 빈 값을 반환하세요.
    사용자가 추출된 문장만 볼 수 있도록 해주세요.
    반드시 아래의 JSON 스키마 형식으로만 결과를 출력해야 하며, 마크다운 코드 블록(```json)이나 다른 부가적인 설명은 포함하지 마십시오.

    {
        "title": 사용자 질문에서 추출된 키워드,
        "sentence": 추출된 키워드에 대한 설명
    }

    """

    user_prompt = f"""
    [Question]
    {user_question}

    [Context]
    {retrieved_context}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = chat_completions(messages)
    raw_response = response.choices[0].message.content

    try:
        clean_json = raw_response.strip("`").replace("json", "")
        structured_output = DocumentSentence.model_validate_json(clean_json)
        print("[PASS]")
        print(structured_output)
        return structured_output
    except ValidationError as e:
        print("데이터 검증 실패. LLM 응답이 지정된 스키마와 일치하지 않습니다.")
        print(f"Raw Output: {clean_json}")
        print(f"Error Details: {e}")
        return None


class DocumentReference(BaseModel):
    file_name: str = Field(description="'.pdf' 확장자로 끝나는 문서의 파일명")
    page_number: int = Field(description="해당 내용이 포함된 문서의 페이지 번호 (정수형)")


def extract_reference_chain(sentence, retrieved_context):
    system_prompt = """
    당신은 사용자가 제공한 문서에서 사용자의 문장이 포함되어 있는 문서의 정보를 추출해주는 어시스턴트 AI입니다.
    사용자가 추출된 문서 정보만 볼 수 있도록 해주세요.
    반드시 아래의 JSON 스키마 형식으로만 결과를 출력해야 하며, 마크다운 코드 블록(```json)이나 다른 부가적인 설명은 포함하지 마십시오.

    {
        "file_name": "추출된 파일명.pdf",
        "page_number": 추출된 페이지 번호
    }

    """
    user_prompt = f"""
    [Sentence]
    {sentence}

    [Context]
    {retrieved_context}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = chat_completions(messages)
    raw_response = response.choices[0].message.content

    try:
        clean_json = raw_response.strip("`").replace("json", "")
        structured_output = DocumentReference.model_validate_json(clean_json)
        print("[PASS]")
        print(structured_output)
        return structured_output
    except ValidationError as e:
        print("데이터 검증 실패. LLM 응답이 지정된 스키마와 일치하지 않습니다.")
        print(f"Raw Output: {clean_json}")
        print(f"Error Details: {e}")
        return None


def generate_markdown(content_obj, source_obj):
    keyword = getattr(content_obj, "title", "키워드 없음")
    content_text = getattr(content_obj, "sentence", "내용 없음")

    sentences = [s.strip() + "." for s in content_text.split(".") if s.strip()]

    doc_name = getattr(source_obj, "file_name", "문서명 정보 없음")
    page_num = getattr(source_obj, "page_number", "페이지 정보 없음")

    md_output = []

    if "없음" in content_text or len(sentences) == 0 or "없음" in doc_name:
        md_output.append("질문하신 내용은 문서에 없는 것 같습니다.. 상세하게 질문해주시겠습니까?")
        return "\n".join(md_output)

    else:
        md_output.append(f"##### {keyword}")
        for s in sentences:
            md_output.append(f"* {s}")

        md_output.append("\n##### 출처")
        md_output.append(f"* **문서명**: {doc_name}")
        md_output.append(f"* **페이지 번호**: {page_num}p")

        return "\n".join(md_output)


def get_ai_message(user_question):
    original_retrieved_context = get_retrieved_context(user_question)

    sentence = extract_sentence_chain(user_question, original_retrieved_context)

    reference = extract_reference_chain(sentence, original_retrieved_context)

    retrieved_context = generate_markdown(sentence, reference)

    return retrieved_context
