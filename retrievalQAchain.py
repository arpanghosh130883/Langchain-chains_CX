# retrievalQAchain.py
# pip install langchain openai faiss-cpu tiktoken
# export OPENAI_API_KEY="sk-..."   (Windows: setx OPENAI_API_KEY "sk-...")

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


def build_vectorstore_from_text(path: str) -> FAISS:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input file: {path}")

    # 1) Load the document
    loader = TextLoader(path, encoding="utf-8")
    documents = loader.load()

    # 2) Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(documents)

    # 3) Embed & store in FAISS (Vector DB)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def build_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    # 4) Create retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 5) Initialize LLM
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.2)

    # 6) RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # also: "map_reduce", "refine", "map_rerank"
        retriever=retriever,
        return_source_documents=True # useful for debugging
    )
    return qa_chain


def main():
    # build vectorstore (or load a persisted one—see notes below)
    vectorstore = build_vectorstore_from_text("docs.txt")
    qa = build_qa_chain(vectorstore)

    print("\nRetrieval-QA ready. Ask questions about docs.txt (type 'exit' to quit)")
    while True:
        q = input("\nQ: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        result = qa({"query": q})
        print("\nA:", result["result"])

        # show sources
        print("\nSources:")
        for i, d in enumerate(result["source_documents"], 1):
            src = getattr(d.metadata, "get", lambda _k, _d=None: None)("source", d.metadata.get("source"))
            print(f"  {i}. {src} (chars {d.metadata.get('start_index','?')}–{d.metadata.get('end_index','?')})")


if __name__ == "__main__":
    main()
