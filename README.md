# EmbeddingGemma Experiments

This repository contains experiments and explorations with Google's EmbeddingGemma model.

## About EmbeddingGemma

EmbeddingGemma is a 300M parameter open embedding model that:
- Converts text into numerical vectors that represent meaning
- Runs locally on device (privacy-focused)
- Supports 100+ languages
- Can run on less than 200MB of RAM

## Resources

- [Official Announcement](https://developers.googleblog.com/en/introducing-embeddinggemma/)
- [Fine-tuning Guide](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers)
- [Ollama Integration](https://ollama.com/library/embeddinggemma)

## What We've Built

### 1. LM Studio Demo (`demo_lmstudio.py`)

A Python script that demonstrates the core capabilities of EmbeddingGemma:

**What it does:**
- Converts text into 768-dimensional embeddings
- Calculates semantic similarity between texts using cosine similarity
- Includes interactive mode to compare your own text pairs

**Key learnings:**
- EmbeddingGemma correctly identifies semantic relationships (e.g., "dog" and "puppy" have high similarity ~0.80)
- Works entirely locally through LM Studio's API
- Fast inference (sub-second for most queries)

**To run:**
```bash
# Make sure LM Studio server is running with embedding model loaded
source venv/bin/activate
python demo_lmstudio.py
```

## Next Steps & Ideas

Here are potential experiments to explore:

### Practical Applications
- **Semantic Search Engine** - Search through personal documents/notes by meaning, not just keywords
- **Code Similarity Finder** - Find duplicate or similar code across your codebase
- **Document Clustering** - Automatically organize documents by topic with visualization
- **Personal Knowledge RAG** - Build a Q&A system over your personal knowledge base

### Technical Exploration
- **Ollama Comparison** - Benchmark LM Studio vs Ollama for speed and quality
- **Dimension Analysis** - Test embedding quality at different dimensions (128, 256, 512, 768)
- **Multilingual Testing** - Evaluate cross-language semantic search across the 100+ supported languages
- **Fine-tuning** - Train on domain-specific data using sentence-transformers

### Research/Creative
- **Embedding Visualization** - Create interactive 2D/3D maps of document collections
- **Prompt Sensitivity** - Test how different task prefixes affect embedding quality
- **Browser Demo** - Port to WebAssembly for client-side semantic search

## Setup

1. Install LM Studio and download the embedding-gemma-300m model
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install openai numpy
   ```
3. Start LM Studio's local server with the embedding model loaded
