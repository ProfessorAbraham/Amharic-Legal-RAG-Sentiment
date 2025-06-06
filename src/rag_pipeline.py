from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

# Define the model for text generation
def load_generator_model():
    llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model="google/flan-t5-large",  # You can change this to a better model
            model_kwargs={"temperature": 0.7, "max_length": 512},
            device=0 if torch.cuda.is_available() else -1,
        )
    )
    return llm

def load_rag_pipeline(vector_db_path="vector_db"):
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Load FAISS vector store
    vectorstore = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    
    # Load the generator model
    llm = load_generator_model()
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        input_key="query"
    )
    return qa_chain

# Example usage
if __name__ == "__main__":
    # Load the RAG pipeline
    qa_chain = load_rag_pipeline()
    
    # Query example: Ask about the right to freedom
    response = qa_chain.run("በመጨረሻ ሀገር የተፈጥሮ ሀብት እና የህዝብ ባህል ይኖራል")
    print("Query Result:")
    print(response)

    # Another query: About the role of the Federal Government
    response = qa_chain.run("የፌዴራሉ መንግሥት ምን መስፈርት አለበት?")
    print("\nQuery Result (Federal Government):")
    print(response)