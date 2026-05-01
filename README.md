# llm-application
* Langchain을 사용한 Application 개발 연습을 위한 Repository

## 6 Chatbot Challenge(Docidian) 설명
### 개요
  * 내용 기반으로 문서를 검색할 수 있는 RAG 애플리케이션
  * 사용자의 질문에 대한 문서를 검색하며, 사용자 질문 및 검색된 문서를 LLM 모델에 전달하여 답변을 생성함.
  * 최초 1회 실행에 한해 인터넷 연결이 필요하며, 이후 폐쇄망 환경에서 모든 기능을 사용할 수 있음.
  * 최초 실행 시 필요한 모델과 input 폴더 내 파일 벡터화 작업이 수행되므로 실행 환경 및 input 폴더의 파일 크기에 따라 시간이 소요될 수 있음.

### 개발 및 실행 환경
  * OS: Windows 11 Pro 25H2(OS 빌드 26200.8246)
  * CPU: AMD Ryzen 5 9600X 6-Core Processor
  * GPU: NVIDIA Geforce RTX 5060
  * RAM: 32.0GB
  * Software: Conda-forge, LM Studio

### 가상환경 생성 방법

```shell
$ conda create -n {가상환경명} python=3.12
$ conda activate {가상환경명}
$ pip install cuda-toolkit==12.8.1 nvidia-cudnn-cu12 --no-cache-dir
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
$ pip install langchain langchain_core langchain_community langchain_huggingface langchain_text_splitters openai huggingface_hub pymupdf streamlit --no-cache-dir
$ pip install sentence-transformers kiwipiepy rank_bm25 --no-cache-dir
$ pip install numpy==1.26.4
$ conda install faiss-gpu
```

### 실행방법
  * 아래와 같이 폴더를 구성해주세요.
    ```shell
      Project Folder
      ㄴ input        # 폴더이며, 이 폴더 안에 프로그램 실행 전 반드시 OCR 처리 되어있는 "*.pdf" 파일을 넣어주세요.
      ㄴ app.py
      ㄴ llm.py
    ```

### 유의사항
  * 실행 환경에서는 반드시 LM Studio를 설치해야 답변을 받아볼 수 있습니다.
  * 가상환경 생성 후 다음의 명령어를 사용하여 CUDA, FAISS가 설치되었는지 확인하세요.
    ```shell
      $ python -c "import torch; print(torch.cuda.is_available())"   # True 나오면 정상, 이 외 비정상
      $ python -c "import faiss; print(faiss.__version__)"              # 버전 나오면 정상, 이 외 비정상
    ```

  * 프로그램의 실행은 가상환경 내에서 아래 명령어를 입력하면 실행됩니다.
    ```shell
      $ streamlit run app.py
    ```
