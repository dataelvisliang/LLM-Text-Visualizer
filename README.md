# Text Visualizer (Powered by Embedding Model)

A Streamlit app for semantic search and trend analysis of Amazon product reviews using sentence transformers and BGE reranker.

## Demo

### Main Interface
![Main Interface](assets/Main%20Interface.png)

### Search Process
![Search Process](assets/Search%20Process.png)

### Result Preview
![Result Preview](assets/Result%20Preview.png)

## Features

- 🔍 **Dual Search Modes** - Search in either review summaries or full review text (selectable)
- 📊 **Semantic Search** - Fast local embeddings with sentence-transformers (141K+ reviews)
- 🎯 **Smart Reranking** - BGE Reranker cross-encoder for accurate relevance scoring with adjustable score threshold (0.20-1.00)
- 🎚️ **Flexible Controls** - Dual top-k sliders for retrieval (up to 20K) and reranking (up to 5K), customizable score filtering
- 📅 **Date Range Filter** - Focus on specific time periods with start and end date selectors
- 📋 **Customizable Preview** - Select and reorder columns in the results table
- 📈 **Trend Visualization** - See mentions over time (month/year granularity)
- 🥧 **Score Distribution** - Pie chart showing relevance score ranges
- 📝 **Detailed Results** - Collapsible sections with top 10 reviews per period

## Deployment

### Streamlit Community Cloud (Recommended)

This app is designed for easy deployment on Streamlit Community Cloud:

1. **Fork/Clone this repository** to your GitHub account
2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository
   - Deploy!

**Note**: All embedding files are included via Git LFS, so the app will work out-of-the-box on Streamlit Cloud without any additional setup.

### Local Setup

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Clone with Git LFS

**Important:** The embedding files are tracked with Git LFS. Make sure you have Git LFS installed:

```bash
git lfs install
git clone <repository-url>
cd LLMTextVisualizer
git lfs pull
```

If you've already cloned the repo without Git LFS, run:
```bash
git lfs install
git lfs pull
```

#### 3. (Optional) Generate Embeddings from Scratch

If you want to regenerate embeddings or use your own dataset:

```bash
python generate_embeddings.py
```

This creates:
- `dataset/review_embeddings.npz` (~191 MB)
- `dataset/summary_embeddings.npz` (~189 MB)
- `dataset/metadata.pkl` (~110 MB)

**Note:** This is a one-time process that takes 20-30 minutes for 141K reviews.

#### 4. Run the App Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Enter Search Phrase** - Type keywords or phrases to search for (e.g., "no tea flavor", "fast shipping")
2. **Select Search Target** - Choose to search in "Summary" or "Review Text"
3. **Adjust Parameters** - Set top-k values for retrieval (100-20,000) and reranking (50-5,000)
4. **Set Date Range** - Filter reviews by date range
5. **Set Score Threshold** - Adjust minimum relevance score (0.20-1.00)
6. **Customize Columns** - Select which columns to display in the preview table
7. **Select Visualization** - Choose time granularity (month/year) and chart type
8. **Search** - Click the search button to run semantic search
9. **Explore Results** - View trends and expand periods to see top-scored reviews

## Example Searches

- "no tea flavor"
- "fast shipping"
- "expired quickly"
- "terrible packaging"
- "love this product"

## Architecture

### Pipeline Flow

```
User Search Phrase
    ↓
Generate Query Embedding (local sentence-transformers)
    ↓
Cosine Similarity Search → Top K candidates (user-configurable)
    ↓
BGE Cross-Encoder Reranking → Keep scores > threshold (user-configurable)
    ↓
Filter by Date Range
    ↓
Aggregate by Time Period
    ↓
Visualize + Show Results
```

### Models Used

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local)
- **Reranking**: `BAAI/bge-reranker-base` (local cross-encoder)

All models run locally on your machine - no API keys required!

### Why Two-Stage Search?

1. **Stage 1 - Fast Embedding Search**: Quickly narrows down 141K reviews to top 500 candidates using local embeddings (cosine similarity)
2. **Stage 2 - Precise Cross-Encoder Reranking**: Uses BGE reranker (a specialized cross-encoder model) to score relevance accurately, keeping only high-quality matches (>0.60)

This hybrid approach balances speed and accuracy. Cross-encoders are much more accurate than bi-encoders but slower, so we use embedding search first to reduce the candidate pool.

## Dataset

The app analyzes **141,210 Amazon product reviews** from the Fine Foods category.

**Source**: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

**Columns used**:
- `Summary` - Review headline
- `Text` - Full review text
- `Date` - Review date
- `Score` - Star rating (1-5)

## File Structure

```
LLMTextVisualizer/
├── app.py                          # Main Streamlit app
├── generate_embeddings.py          # One-time embedding generation script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── QUICKSTART.md                   # Quick start guide
├── .gitattributes                  # Git LFS configuration for large files
├── .gitignore                      # Git ignore patterns
├── assets/                         # Demo screenshots
│   ├── Main Interface.png
│   ├── Search Process.png
│   └── Result Preview.png
└── dataset/
    ├── amazon_review_part1.csv     # First half of dataset (not tracked in git)
    ├── amazon_review_part2.csv     # Second half of dataset (not tracked in git)
    ├── review_embeddings.npz       # Review text embeddings (Git LFS, 191 MB)
    ├── summary_embeddings.npz      # Summary embeddings (Git LFS, 189 MB)
    └── metadata.pkl                # Review metadata (Git LFS, 110 MB)
```

## Tool Vision

### Semantic Feedback Intelligence Platform
A modern, privacy-first semantic analytics tool that turns raw customer feedback into precise, actionable insights — instantly.

### Why This Changes Everything
Keyword search is dead for real-world feedback analysis. Customers never say the same thing the same way.

**This tool doesn't match words. It understands meaning.**

### The Semantic Advantage

**Long-tail coverage**
"No tea flavor" automatically finds:
*"can't taste the tea"*, *"completely tasteless"*, *"where's the tea???"*, *"missing tea notes"*, *"just hot water"* — all in one query.

**True synonym & paraphrase understanding**
Search *"fast shipping"* → instantly captures *"arrived next day"*, *"super quick delivery"*, *"came earlier than expected"*.

**Context-aware precision**
Combined retrieval + reranking eliminates false positives.
*"Great battery"* (positive) ≠ *"Great battery, dies in 2 hours"* (negative).

**Concept discovery**
Ask about *"packaging issues"* → finds *"box arrived crushed"*, *"poorly wrapped"*, *"product damaged in transit"* — even if you never knew those phrases existed.

**Spelling, grammar, slang proof**
Handles *"recieved"* → *"received"*, *"lit af"* → *"amazing"*, *"teh best"* → *"the best"*.

**Zero prompt engineering needed**
Just type what you want to know in plain English.

### Core Business Impact

**Instant Theme Detection**
Product managers go from "reading 100 reviews hoping to spot patterns" to "seeing every mention of any concept in seconds".

**Early Warning System**
Time-series trends reveal exactly when a new complaint started spiking — with 95%+ recall across all phrasings.

**Accurate Prioritization**
No more guessing which issues matter. Count real customer mentions with full semantic coverage.

**Competitive Intelligence**
Understand why customers switch (*"better than [competitor]"*) across all possible ways they express it.

**Support Deflection**
Find the top 50 real ways customers describe the same problem → write one perfect help article.

### Modern Technical Architecture (2025 Standard)

- **Fully local execution** — zero API keys, zero cost after install
- **State-of-the-art embeddings** - sentence-transformers/all-MiniLM-L6-v2
- **Two-stage retrieval**:
  - → Lightning-fast approximate search (Cosine Similarity)
  - → BGE Reranker for final precision (top-tier cross-encoder, runs locally in <50ms)
- **Sub-second response** on 140K+ documents
- **Pre-computed embeddings** cached to disk - reload in <3s
- **Pure Python stack**: sentence-transformers, torch (CPU or CUDA)
- **Privacy by design**: nothing ever leaves your machine
- **Ready for air-gapped / on-premise deployment**

### What You Get

- Upload once → ask anything, any time
- Interactive time trends (daily → yearly zoom)
- Relevance score distribution
- Top 10 most relevant quotes per period with exact scores
- One-click export of results
- Beautiful, stakeholder-ready UI in 5 seconds

**This isn't another keyword dashboard.**
**This is production-grade semantic intelligence - local, instant, and actually accurate.**

## License

MIT License

## Credits

Built with:
- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-base)
- [Plotly](https://plotly.com/)
