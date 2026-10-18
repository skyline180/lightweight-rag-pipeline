
from src.rag_pipeline.pipeline import RAGPipeline
from scripts.prepare_data import prepare_data


def main():
    # 1. Load documents from the dataset
    documents = prepare_data()

    # 2. Initialize and build the RAG pipeline
    rag = RAGPipeline(n_neighbors=3)
    rag.build(documents)

    print("\n=== Lightweight RAG Pipeline ===")
    print("Ask a question about the documents.")
    print("Type 'exit' to quit.\n")

    # 3. Ask questions interactively
    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        # 4. Retrieve relevant chunks and generate an answer
        result = rag.ask(question)

        print(f"\nAnswer: {result['answer']}")

        # 5. Display retrieved sources
        print("\n--- Retrieved Sources ---")

        for i, source in enumerate(result["sources"], start=1):
            print(f"\n{i}. {source.get('source', 'Document chunk')}")
            print(source["text"][:200])
            print(f"Cosine distance: {source['distance']:.4f}")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
