from .generator import generate_answer
from .retriever import retrieve 

class SECRAG:
    def __init__(self, index_path="indexes/faiss.index",
                 metadata_path="indexes/metadata.json"):
       
        pass

    def answer(self, question, k=5, years=None, sections=None):
        """
        Orchestrates the RAG loop: fetches clean context chunks from the vector database 
        and feeds them to the generation engine to yield a cited answer.
        """
        
        contexts = retrieve(question, k=k)

        answer = generate_answer(question, contexts)
        
        return {
            "question": question,
            "answer": answer,
            "sources": contexts,
        }
