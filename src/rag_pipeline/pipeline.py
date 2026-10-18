
from .chunking import chunk_text
from .embeddings import EmbeddingModel
from .retriever import SklearnRetriever
from .generator import AnswerGenerator


class RAGPipeline:
    def __init__(self, n_neighbors: int = 3):
        self.embedding_model = EmbeddingModel()
        self.generator = AnswerGenerator()
        self.retriever = SklearnRetriever(
            n_neighbors=n_neighbors
        )

        self.chunks = []

    def build(self, documents: list[str]):
        """Prepare documents and build the retrieval index."""

        self.chunks = []

        for document in documents:
            self.chunks.extend(chunk_text(document))

        if not self.chunks:
            raise ValueError("No chunks were created from documents.")

        embeddings = self.embedding_model.embed_documents(
            self.chunks
        )

        self.retriever.fit(self.chunks, embeddings)

    def ask(self, question: str):
        """Retrieve context and generate an answer."""

        if not self.chunks:
            raise RuntimeError(
                "Build the pipeline with documents before asking questions."
            )

        query_embedding = self.embedding_model.embed_query(
            question
        )

        retrieved_chunks = self.retriever.retrieve(
            query_embedding
        )

        answer = self.generator.generate(
            question,
            retrieved_chunks,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_chunks,
        }
