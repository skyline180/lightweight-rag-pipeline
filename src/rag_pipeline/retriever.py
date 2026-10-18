from sklearn.neighbors import NearestNeighbors


class SklearnRetriever:
    def __init__(self, n_neighbors: int = 3):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")

        self.n_neighbors = n_neighbors
        self.chunks = []
        self.retriever = None

    def fit(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have equal lengths")

        if len(chunks) < self.n_neighbors:
            raise ValueError("Not enough chunks for n_neighbors")

        self.chunks = list(chunks)

        self.retriever = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric="cosine",
            algorithm="brute",
        )
        self.retriever.fit(embeddings)

    def retrieve(self, query_embedding):
        if self.retriever is None:
            raise RuntimeError("Fit the retriever before searching")

        distances, indices = self.retriever.kneighbors(
            [query_embedding]
        )

        return [
            {
                "text": self.chunks[idx],
                "distance": float(distance),
            }
            for idx, distance in zip(indices[0], distances[0])
        ]
