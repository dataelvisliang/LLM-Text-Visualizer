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

## 2. Get OpenRouter API Key

1. Visit https://openrouter.ai/
2. Sign up (free)
3. Go to "API Keys" section
4. Click "Create Key"
5. Copy your key (starts with `sk-or-v1-...`)

## 3. Run the App

```bash
streamlit run app.py
```

The app will open at http://localhost:8501

## 4. Use the App

1. **Paste API Key** in the sidebar (password field)
2. **Type a question** like: "How many people complain about no tea flavor?"
3. **Select time granularity**: day, month, or year
4. **Choose chart type**: Trend or Score Distribution
5. **Click Search** 🔎

## Example Questions to Try

```
How many people complain about no tea flavor?
Find reviews mentioning fast shipping
Who said the product expired quickly?
People talking about taste
Reviews about packaging quality
Mentions of great price
Customer service complaints
```

## What to Expect

- **Query extraction**: 1-2 seconds
- **Semantic search**: <1 second
- **BGE Reranking**: 5-10 seconds (much faster than LLM!)
- **Results**: Chart + expandable review sections

## Troubleshooting

### "Embeddings not found"
- Run `python generate_embeddings.py` first

### "OpenRouter API error"
- Check your API key is correct
- Make sure you have free credits (new accounts get free tier)

### "No results found"
- Try a different query
- Make the question more specific
- Check if the topic exists in Amazon food reviews

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
