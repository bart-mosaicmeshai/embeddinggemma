# EmbeddingGemma Experiments

This repository contains experiments and explorations with Google's EmbeddingGemma model.

## About EmbeddingGemma

EmbeddingGemma is a 300M parameter open embedding model that:
- Converts text into numerical vectors that represent meaning
- Runs locally on device (privacy-focused)
- Supports 100+ languages
- Can run on less than 200MB of RAM

## Resources

### Official Documentation
- [Official Announcement Blog Post](https://developers.googleblog.com/en/introducing-embeddinggemma/) - Google's introduction to EmbeddingGemma with technical details and benchmarks
- [Fine-tuning Guide](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers) - How to customize EmbeddingGemma for domain-specific tasks
- [Google AI Documentation](https://ai.google.dev/gemma/docs/embeddinggemma) - Complete technical documentation

### Model Access & Tools
- [LM Studio Model](https://lmstudio.ai/models/google/embedding-gemma-300m) - Download and use via LM Studio GUI
- [Ollama Integration](https://ollama.com/library/embeddinggemma) - Command-line tool for running EmbeddingGemma locally
- [Hugging Face](https://huggingface.co/google/embedding-gemma-300m) - Direct model access for Python

### Community & Examples
- [Simon Willison's Analysis](https://simonwillison.net/2025/Sep/4/embedding-gemma/) - Practical insights and browser-based demo
- [Research Paper](https://arxiv.org/abs/2509.20354) - "EmbeddingGemma: Powerful and Lightweight Text Representations"

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

### 2. Semantic Search Engine (`semantic_search.py`)

A practical search tool that finds documents by meaning, not just keywords.

**What it does:**
- Indexes text files from a directory (md, py, txt, js, json, csv, log, rst)
- Generates and caches embeddings to avoid recomputation
- Searches by semantic meaning using cosine similarity
- Returns ranked results with similarity scores and previews
- Automatically chunks large documents for better granularity

**Key features:**
- Works on any directory of text files
- Smart filtering (skips venv, node_modules, .git, etc.)
- Persistent caching (`embeddings_cache.json`) - first run generates, subsequent runs load from cache
- Interactive mode or single-query mode
- Each directory gets its own cache file for portability

**To run:**
```bash
# Make sure LM Studio server is running with embedding model loaded
source venv/bin/activate

# Interactive mode (searches current directory)
python semantic_search.py

# Single query mode
python semantic_search.py /path/to/docs -q "your search query"

# Search current directory with a query
python semantic_search.py . -q "your search query"

# Reindex (useful after adding new files)
python semantic_search.py --reindex
```

**Example searches:**
- "how to calculate similarity" - finds code/docs about cosine similarity
- "error handling" - finds error-related code across multiple files
- "configuration setup" - finds setup and config documentation

**What we learned:**
- Chunking text (1000 chars with 100 char overlap) improves search quality
- Caching embeddings is essential for good performance
- Semantic search finds relevant content even with different wording
- Similarity thresholds: 0.80+ (very similar), 0.60+ (good match), 0.40+ (weak match), <0.40 (unrelated)

## Next Steps & Ideas

Here are potential experiments to explore:

### Practical Applications
- ✅ **Semantic Search Engine** - Search through personal documents/notes by meaning, not just keywords (COMPLETED)
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

## Session Notes

### What We Learned
- **Embeddings basics**: Text is converted to 768-dimensional vectors that capture semantic meaning
- **Cosine similarity**: Measures how similar two embeddings are (0-1 scale, higher = more similar)
- **LM Studio API**: Compatible with OpenAI's API format, making it easy to use existing libraries
- **Performance**: Sub-second inference for generating embeddings locally
- **Chunking strategy**: Breaking documents into overlapping chunks improves search quality
- **Caching**: Essential for performance - avoids regenerating embeddings repeatedly

### Current State
- ✅ Basic demo working with LM Studio
- ✅ Understanding of how embeddings capture semantic relationships
- ✅ Semantic search engine for practical document search
- ✅ Embedding caching for performance
- ⏳ Next: Try document clustering, RAG system, or other experiments

### Key Insights
- "dog" and "puppy" achieve ~0.80 similarity despite different words (demo)
- Unrelated topics (dogs vs cars) score much lower ~0.45 (demo)
- Semantic search finds relevant docs even with different wording (search engine)
- Similarity thresholds: 0.80+ (very similar), 0.60+ (good match), 0.40+ (weak match), <0.40 (unrelated)
