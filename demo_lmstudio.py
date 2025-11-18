#!/usr/bin/env python3
"""
Simple demo of EmbeddingGemma via LM Studio

This script shows how to:
1. Generate embeddings for text
2. Calculate semantic similarity between texts
3. Find the most similar text from a collection
"""

from openai import OpenAI
import numpy as np

# LM Studio runs a local server compatible with OpenAI API
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

def get_embedding(text):
    """Get embedding vector for a piece of text"""
    response = client.embeddings.create(
        input=text,
        model="embedding"  # LM Studio uses this generic name
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors (0-1, higher = more similar)"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    print("🔮 EmbeddingGemma Demo via LM Studio\n")
    print("=" * 60)

    # Example 1: Basic embedding
    print("\n📝 Example 1: Getting an embedding")
    text = "The quick brown fox jumps over the lazy dog"
    embedding = get_embedding(text)
    print(f"Text: '{text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Example 2: Semantic similarity
    print("\n" + "=" * 60)
    print("\n🔍 Example 2: Semantic Similarity")

    texts = {
        "A": "I love playing with my dog",
        "B": "My puppy is so playful and fun",
        "C": "The weather is nice today",
        "D": "Cars are expensive to maintain"
    }

    print("\nComparing these texts:")
    for key, text in texts.items():
        print(f"  {key}: '{text}'")

    # Get embeddings for all texts
    print("\n⏳ Generating embeddings...")
    embeddings = {key: get_embedding(text) for key, text in texts.items()}

    # Compare A with all others
    query_key = "A"
    print(f"\n📊 Similarity scores compared to '{query_key}' ('{texts[query_key]}'):\n")

    similarities = {}
    for key in texts.keys():
        if key != query_key:
            sim = cosine_similarity(embeddings[query_key], embeddings[key])
            similarities[key] = sim
            print(f"  {query_key} ↔ {key}: {sim:.4f}")

    # Find most similar
    most_similar = max(similarities.items(), key=lambda x: x[1])
    print(f"\n✨ Most similar to {query_key}: {most_similar[0]} (score: {most_similar[1]:.4f})")
    print(f"   '{texts[most_similar[0]]}'")

    # Example 3: Interactive mode
    print("\n" + "=" * 60)
    print("\n💬 Example 3: Try your own!")
    print("Enter two sentences to compare their similarity:")

    try:
        text1 = input("\nSentence 1: ").strip()
        text2 = input("Sentence 2: ").strip()

        if text1 and text2:
            print("\n⏳ Computing similarity...")
            emb1 = get_embedding(text1)
            emb2 = get_embedding(text2)
            sim = cosine_similarity(emb1, emb2)

            print(f"\n📊 Similarity score: {sim:.4f}")
            if sim > 0.8:
                print("   → Very similar! 🎯")
            elif sim > 0.6:
                print("   → Somewhat similar 🤔")
            elif sim > 0.4:
                print("   → Not very similar 🤷")
            else:
                print("   → Very different! 🔀")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
