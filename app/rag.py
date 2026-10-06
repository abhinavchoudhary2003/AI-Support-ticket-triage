"""
RAG: PDF ingestion -> section-aware chunks -> embeddings -> ChromaDB.

Sentence Transformers creates semantic embeddings for the knowledge-base
chunks, and ChromaDB stores those embeddings and performs similarity search.

The Retriever keeps the same public interface used by services.py:
    Retriever(...)
    retriever.search(query, k=4)

This keeps the RAG implementation replaceable without changing the API layer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


# Sentence Transformer model used to create semantic embeddings.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Persistent local ChromaDB directory.
CHROMA_DIR = "chroma_db"

# Chroma collection name.
COLLECTION_NAME = "support_knowledge_base"


@dataclass
class Chunk:
    id: int
    document: str
    title: str
    section: str
    page: int
    text: str


def load_pdf_chunks(path: Path) -> List[Chunk]:
    """
    Read a PDF and create one chunk for each numbered section.

    Example heading:
        3.2 Duplicate charges
    """

    reader = PdfReader(str(path))

    chunks = []

    title = (
        reader.metadata.title
        if reader.metadata and reader.metadata.title
        else path.stem
    )

    current = None

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        for line in page_text.splitlines():
            line = line.strip()

            if not line:
                continue

            # Detect numbered section headings such as:
            # 1.1 Password reset
            # 3.2 Duplicate charges
            parts = line.split(maxsplit=1)

            is_heading = (
                len(parts) == 2
                and "." in parts[0]
                and parts[0].replace(".", "").isdigit()
            )

            if is_heading:
                if current:
                    chunks.append(current)

                current = {
                    "section": line,
                    "page": page_number,
                    "lines": [],
                }

            elif current is not None:
                current["lines"].append(line)

    if current:
        chunks.append(current)

    result = []

    for chunk in chunks:
        text = " ".join(chunk["lines"]).strip()

        if text:
            result.append(
                Chunk(
                    id=0,
                    document=path.name,
                    title=title,
                    section=chunk["section"],
                    page=chunk["page"],
                    text=text,
                )
            )

    return result


class Retriever:
    """
    Semantic RAG retriever using Sentence Transformers + ChromaDB.
    """

    def __init__(self, kb_dir: Path):
        self.kb_dir = Path(kb_dir)

        # Load all PDF sections.
        self.chunks: List[Chunk] = []

        for pdf in sorted(self.kb_dir.glob("*.pdf")):
            self.chunks.extend(load_pdf_chunks(pdf))

        # Give every chunk a unique numeric ID.
        for index, chunk in enumerate(self.chunks):
            chunk.id = index

        # Sentence Transformer embedding function.
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
        )

        # Persistent ChromaDB database.
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR
        )

        # Recreate the collection so the index always matches
        # the current PDF knowledge base.
        try:
            self.client.delete_collection(
                name=COLLECTION_NAME
            )
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={
                "description": "Support policy knowledge base"
            },
        )

        self._build()

    def _build(self) -> None:
        """
        Generate embeddings and store all chunks in ChromaDB.
        """

        if not self.chunks:
            return

        documents = []
        metadatas = []
        ids = []

        for chunk in self.chunks:
            # Include the section heading in the embedded text
            # because it provides useful semantic context.
            document_text = (
                f"{chunk.section}\n"
                f"{chunk.text}"
            )

            documents.append(document_text)

            metadatas.append(
                {
                    "document": chunk.document,
                    "title": chunk.title,
                    "section": chunk.section,
                    "page": chunk.page,
                    "chunk_id": chunk.id,
                }
            )

            ids.append(str(chunk.id))

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def search(
        self,
        query: str,
        k: int = 4,
        prefer_docs: Optional[Set[str]] = None,
    ) -> Tuple[List[Tuple[Chunk, float]], float]:
        """
        Perform semantic similarity search.

        Returns:
            [
                (chunk, similarity_score),
                ...
            ]

        plus a coverage score between 0 and 1.

        ChromaDB returns cosine distance for this collection.
        Smaller distance means greater similarity.
        """

        if not query.strip() or not self.chunks:
            return [], 0.0

        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, len(self.chunks)),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return [], 0.0

        chunk_lookup = {
            chunk.id: chunk
            for chunk in self.chunks
        }

        scored = []

        for metadata, distance in zip(
            metadatas,
            distances
        ):
            chunk_id = int(metadata["chunk_id"])
            chunk = chunk_lookup[chunk_id]

            # Convert cosine distance into an easier-to-read
            # similarity score.
            similarity = max(
                0.0,
                1.0 - float(distance)
            )

            # Optional preference for particular documents.
            if (
                prefer_docs
                and chunk.document in prefer_docs
            ):
                similarity *= 1.10

            scored.append(
                (chunk, similarity)
            )

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # Keep the requested number of results.
        scored = scored[:k]

        # Coverage estimates whether the retrieved results
        # are strong enough to answer the customer's question.
        top_scores = [
            score
            for _, score in scored[:3]
        ]

        if not top_scores:
            return [], 0.0

        best_score = top_scores[0]

        if best_score >= 0.50:
            coverage = 1.0
        elif best_score >= 0.40:
            coverage = 0.75
        elif best_score >= 0.30:
            coverage = 0.50
        elif best_score >= 0.20:
            coverage = 0.25
        else:
            coverage = 0.0

        return scored, coverage


if __name__ == "__main__":
    retriever = Retriever(
        Path("knowledge_base")
    )

    print(
        f"Loaded {len(retriever.chunks)} chunks."
    )

    results, coverage = retriever.search(
        "I was charged twice for my order"
    )

    print(
        f"\nCoverage: {coverage}"
    )

    for chunk, score in results:
        print("\n--- RESULT ---")
        print(f"Source: {chunk.document}")
        print(f"Section: {chunk.section}")
        print(f"Page: {chunk.page}")
        print(f"Similarity: {score:.3f}")
        print(f"Text: {chunk.text}")