
from transformers import pipeline


class AnswerGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ):
        self.llm = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
        )

    def generate(self, question: str, retrieved_chunks: list[dict]):
        """Generate an answer using retrieved context."""

        context = "\n\n".join(
            f"Source: {chunk.get('source', 'Unknown')}\n"
            f"{chunk['text']}"
            for chunk in retrieved_chunks
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer using only the provided context. "
                    "If the answer is unavailable, say: "
                    "'I cannot find the answer in the provided documents.'"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ]

        prompt = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        response = self.llm(
            prompt,
            max_new_tokens=150,
            do_sample=False,
        )

        generated_text = response[0]["generated_text"]
        return generated_text[len(prompt):].strip()
