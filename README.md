# llm-application
* Langchain을 사용한 Application 개발 연습을 위한 Repository
* Inflearn의 ["RAG를 활용한 LLM Application 개발 (feat. LangChain)"](https://www.inflearn.com/course/rag-llm-application%EA%B0%9C%EB%B0%9C-langchain/dashboard?cid=333796)의 강의를 들으며,<br/>부가적으로 알게 된 내용과 강의 실습 결과물을 공유합니다.

## 프로젝트 구조
* 5.1 Upstage Challenge
  * 강의의 **"5.1 LangSmith를 활용한 LLM Evaluation"** 강의와 관련 있습니다.

* 6 Chatbot Challenge
  * 강의의 **"6 최종 미션"** 과 관련 있습니다.

## 5.1 Upstage Challenge 설명
### 개요
* Langsmith Docs에서 다루는 LLM Evaluator 구현 시 포인트와 함께 Upstage를 사용한 방법을 정리함.
* 현재 시점의 Langsmith Docs는 아래 링크의 문서이며, 이하 설명에서는 **Langsmith Docs**를 **Docs**로 부름.
    
    > https://docs.langchain.com/langsmith/evaluate-rag-tutorial

### 1. Evaluator 종류
* 현재 Docs에서는 다음과 같은 평가 지표에 대한 예시를 제공하고 있음.

  > 1) 정확성(Correctness): LLM의 답변이 실제 답변과 얼마나 유사하고 정확한지 측정
  > 2) 관련성(Relevance): LLM의 답변이 사용자의 질문과 얼마나 관련있는지 측정
  > 3) 근거성(Groundedness): LLM의 답변이 검색된 데이터의 맥락과 얼마나 일치하는지 측정
  > 4) 검색 관련성(Retrieval relevance): 입력된 질문과 검색된 데이터는 얼마나 관련있는지 측정

### 2. 데이터 셋 형식 변경
* 현재의 Docs에서는 다음과 같은 데이터 셋 형식을 사용하고 있음.

  ```python
  # Define the examples for the dataset
  examples = [
      {
          "inputs": {"question": "How does the ReAct agent use self-reflection? "},
          "outputs": {"answer": "ReAct integrates reasoning and acting, performing actions - such tools like Wikipedia search API - and then observing / reasoning about the tool outputs."},
      },
      ...
  ]
  ```

* 따라서 다음과 같이 데이터 셋을 다음과 같이 변경할 수 있음.
* Docs의 Evaluator에서는 'contexts' 를 사용하진 않는 것 같지만 기존 데이터 셋의 내용에 맞추어 수정함.

  ```python
  examples = [
    {
        "inputs": {"question": "제1조에 따른 소득세법의 목적은 무엇인가요?"},
        "outputs": {"answer": "소득세법의 목적은 소득의 성격과 납세자의 부담능력에 따라 적정하게 과세함으로써 조세부담의 형평을 도모하고 재정수입의 원활한 조달에 이바지하는 것입니다."},
        "metadata": {"contexts": "제1조(목적) 이 법은 개인의 소득에 대하여 소득의 성격과 납세자의 부담능력 등에 따라 적정하게 과세함으로써 조세부담의 형평을 도모하고 재정수입의 원활한 조달에 이바지함을 목적으로 한다."},
    },
    ...
  ]
  ```

### 3. Upstage Model을 사용하는 방법
* 현재의 Docs에서는 각 단계에서 LLM을 선언하는 부분에서 다음과 같이 'ChatOpenAI()' 함수를 사용하고 있음.

  ```python
  # Rag-Bot
  from langchain_openai import ChatOpenAI
  ...
  llm = ChatOpenAI(model="gpt-4.1", temperature=1)
  
  # Evaluators
  grader_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_structured_output(
      ...
  )
  ```

* 따라서 다음과 같이 'ChatUpstage' 라이브러리를 import한 후, 'ChatOpenAI()' 함수를 사용하는 부분을 'ChatUpstage()' 함수로 변경해주면 됨.
* 'Rag-Bot' 코드를 작성하는 셀에서 'ChatUpstage' 라이브러리를 import한다면 이하 코드에서는 일일이 import하지 않아도 되며, 'ChatOpenAI()' 함수에서 사용하던 'temperature=1' 인자는 제거해도 무방했음.

  ```python
   # Rag-Bot
  from langchain_upstage import ChatUpstage
  ...
  llm = ChatUpstage(model="solar-pro3")
  
  # Evaluators
  grader_llm = ChatUpstage(model="solar-pro3").with_structured_output(
      ...
  )
  ```


---

## 6 Chatbot Challenge(Docidian) 설명
### 개요
  * 내용 기반으로 문서를 검색할 수 있는 RAG 애플리케이션
  * 사용자의 질문에 대한 문서를 검색하며, 사용자 질문 및 검색된 문서를 LLM 모델에 전달하여 답변을 생성함.
  * 최초 1회 실행에 한해 필요한 모델 다운로드를 위해 인터넷 연결이 필요하며, 이후 폐쇄망 환경에서 모든 기능을 사용할 수 있음.
  * 최초 실행 시 input 폴더 내 파일 벡터화 작업이 수행되므로 실행 환경 및 input 폴더의 파일 크기에 따라 시간이 소요될 수 있음.

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
