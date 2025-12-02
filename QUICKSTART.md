# Quick Start Guide

## 1. Generate Embeddings (First Time Only)

This step is **required** before running the app. It takes about 10-15 minutes.

```bash
python generate_embeddings.py
```

You'll see progress like:
```
Loading datasets...
Total reviews: 141210
Preparing text data...
Loading sentence-transformers model (all-MiniLM-L6-v2)...
Generating embeddings (this may take 10-15 minutes)...
Processed 1000/141210 reviews
Processed 2000/141210 reviews
...
```

When complete, you'll see:
```
✅ Done!
Embeddings saved to: dataset/embeddings.npz (XX MB)
Metadata saved to: dataset/metadata.pkl (XX MB)
```

## 2. Run the App

```bash
streamlit run app.py
```

The app will open at http://localhost:8501

## 3. Use the App

1. **Enter search phrase** like: "no tea flavor" or "fast shipping"
2. **Choose search target**: Review Text or Summary
3. **Adjust search parameters**: Top-K for retrieval and reranking
4. **Set date range** (optional): Filter reviews by time period
5. **Set score threshold**: Minimum relevance score (0.20-1.00)
6. **Select visualization**: Time granularity (month/year) and chart type
7. **Click Search** 🔎

## Example Search Phrases to Try

```
no tea flavor
fast shipping
expired quickly
terrible packaging
love this product
great price
poor customer service
```

## What to Expect

- **Query embedding**: <1 second
- **Semantic search**: <1 second
- **BGE Reranking**: 5-10 seconds on CPU
- **Results**: Interactive charts + expandable review sections

## Troubleshooting

### "Embeddings not found"
- Run `python generate_embeddings.py` first
- Make sure both `review_embeddings.npz` and `summary_embeddings.npz` exist in the dataset folder

### "No results found"
- Try lowering the score threshold
- Try a different search phrase
- Check if the topic exists in Amazon food reviews
- Switch between searching in "Review Text" vs "Summary"

### Slow reranking
- BGE reranker runs on CPU by default (~5-10 seconds for 500 candidates)
- If you have a GPU, the reranker will automatically use it for faster processing
- Reducing top_k in the code can speed this up (trade-off: fewer results)

## Files Created

After running the embedding script, you'll have:

```
dataset/
├── amazon_review_part1.csv      (your original data)
├── amazon_review_part2.csv      (your original data)
├── embeddings.npz               (generated - ~191 MB)
└── metadata.pkl                 (generated - ~111 MB)
```

## Tips

- **First query is slower**: Model loading takes ~2-3 seconds
- **Batch processing**: The app processes 500 candidates → filters to high-quality matches
- **Score threshold**: Only results with score > 0.60 are shown
- **Period expansion**: Click any time period to see top 10 reviews from that period
