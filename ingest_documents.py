from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Local data insertion
loader = PyPDFLoader(r"C:\Users\guita\OneDrive\Desktop\ConsultAI\data\Case-in-Point.pdf")
pages = loader.load()

# Text Spliting into chars 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

# Manual section tagging 
PART_PAGE_RANGES = {
    "prepare_yourself": range(10, 69),
    "learning": range(69, 130),
    "case_prep": range(130, 325),
}

structured_docs = []

# Building the Structure documents
for i, page in enumerate(pages):
    text = page.page_content
    part = "general"

    # Tag each page with a “section/part”
    for tag, page_range in PART_PAGE_RANGES.items():
        if i in page_range:
            part = tag
            break

    # Split page into chunks
    chunks = splitter.split_text(text)

    for j, chunk in enumerate(chunks):
        structured_docs.append(Document(
            page_content=chunk,
            metadata={
                "page_number": i + 1,
                "part": part,
                "source": "Case In Point",
                "chunk_id": f"page_{i+1}_chunk_{j}"
            }
        ))

print(" Structured documents created:", len(structured_docs))

# 4. Create embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Create vector store
vectorstore = FAISS.from_documents(structured_docs, embedding)

# 6. Save vector store
vectorstore.save_local("faiss_index")
print(" FAISS vectorstore saved to 'faiss_index'")
