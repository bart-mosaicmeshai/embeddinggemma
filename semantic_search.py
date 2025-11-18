#!/usr/bin/env python3
"""
Semantic Search Engine using EmbeddingGemma

Indexes text files from a directory and enables semantic search using
embeddings. Results are ranked by semantic similarity, not keyword matching.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from openai import OpenAI

# LM Studio configuration
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
EMBEDDING_MODEL = "google/embedding-gemma-300m-gguf"

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.log', '.rst'}

# Directories to skip during indexing
SKIP_DIRECTORIES = {
    'venv', 'node_modules', '.git', '__pycache__', '.pytest_cache',
    '.mypy_cache', '.tox', 'dist', 'build', '.eggs', '*.egg-info',
    '.venv', 'env', '.env', 'site-packages'
}

# Cache file for embeddings
CACHE_FILE = "embeddings_cache.json"


def get_embedding(text: str) -> List[float]:
    """Generate embedding for given text using LM Studio."""
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for better search granularity.

    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]

        # Try to break at sentence or newline boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > max_chars // 2:  # Only break if it's not too early
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def index_directory(directory: str, force_reindex: bool = False) -> Dict:
    """
    Index all text files in a directory and generate embeddings.

    Args:
        directory: Path to directory to index
        force_reindex: If True, regenerate all embeddings even if cache exists

    Returns:
        Dictionary containing indexed documents and their embeddings
    """
    index = {"documents": []}

    # Load existing cache if available
    cache_path = Path(directory) / CACHE_FILE
    if cache_path.exists() and not force_reindex:
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)

    print(f"Indexing files in {directory}...")
    path = Path(directory)

    # Find all supported files
    files_to_index = []
    for ext in SUPPORTED_EXTENSIONS:
        files_to_index.extend(path.rglob(f"*{ext}"))

    if not files_to_index:
        print(f"No supported files found in {directory}")
        return index

    print(f"Found {len(files_to_index)} files to index")

    for file_path in files_to_index:
        try:
            # Skip files in excluded directories
            if any(skip_dir in file_path.parts for skip_dir in SKIP_DIRECTORIES):
                continue

            # Skip cache file itself
            if file_path.name == CACHE_FILE:
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                continue

            # Chunk the content
            chunks = chunk_text(content)
            rel_path = file_path.relative_to(path)

            print(f"Processing {rel_path} ({len(chunks)} chunks)...")

            for i, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)

                doc_entry = {
                    "file": str(rel_path),
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "content": chunk,
                    "embedding": embedding
                }
                index["documents"].append(doc_entry)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Save cache
    print(f"\nSaving cache to {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(index, f)

    print(f"Indexed {len(index['documents'])} document chunks")
    return index


def search(query: str, index: Dict, top_k: int = 5) -> List[Tuple[Dict, float]]:
    """
    Search indexed documents for semantic matches to query.

    Args:
        query: Search query
        index: Document index with embeddings
        top_k: Number of top results to return

    Returns:
        List of (document, similarity_score) tuples, sorted by relevance
    """
    print(f"\nSearching for: '{query}'")
    query_embedding = get_embedding(query)

    # Calculate similarities
    results = []
    for doc in index["documents"]:
        similarity = cosine_similarity(query_embedding, doc["embedding"])
        results.append((doc, similarity))

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]


def display_results(results: List[Tuple[Dict, float]]):
    """Display search results in a readable format."""
    if not results:
        print("No results found.")
        return

    print(f"\n{'='*80}")
    print(f"Top {len(results)} Results:")
    print(f"{'='*80}\n")

    for i, (doc, score) in enumerate(results, 1):
        chunk_info = ""
        if doc["total_chunks"] > 1:
            chunk_info = f" [chunk {doc['chunk_id'] + 1}/{doc['total_chunks']}]"

        print(f"{i}. {doc['file']}{chunk_info}")
        print(f"   Similarity: {score:.4f}")
        print(f"   Preview: {doc['content'][:200]}...")
        print()


def interactive_search(index: Dict):
    """Run interactive search mode."""
    print("\n" + "="*80)
    print("Interactive Semantic Search")
    print("="*80)
    print("Enter your search queries (or 'quit' to exit)\n")

    while True:
        try:
            query = input("Search> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            results = search(query, index)
            display_results(results)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search engine using EmbeddingGemma"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to index (default: current directory)"
    )
    parser.add_argument(
        "-q", "--query",
        help="Search query (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindexing even if cache exists"
    )

    args = parser.parse_args()

    # Index the directory
    index = index_directory(args.directory, force_reindex=args.reindex)

    if not index["documents"]:
        print("No documents indexed. Exiting.")
        return

    # Search or enter interactive mode
    if args.query:
        results = search(args.query, index, top_k=args.top_k)
        display_results(results)
    else:
        interactive_search(index)


if __name__ == "__main__":
    main()
