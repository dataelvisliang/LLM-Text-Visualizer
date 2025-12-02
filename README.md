# Text Visualizer (Powered by Embedding Model)

A Streamlit app for semantic search and trend analysis of Amazon product reviews using sentence transformers and BGE reranker.

## Demo

### Main Interface
![Main Interface](assets/Main%20Interface.png)

### Search Result Example
![Search Result Example](assets/Search%20Result%20Example.png)

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
│   └── Search Result Example.png
└── dataset/
    ├── amazon_review_part1.csv     # First half of dataset (not tracked in git)
    ├── amazon_review_part2.csv     # Second half of dataset (not tracked in git)
    ├── review_embeddings.npz       # Review text embeddings (Git LFS, 191 MB)
    ├── summary_embeddings.npz      # Summary embeddings (Git LFS, 189 MB)
    └── metadata.pkl                # Review metadata (Git LFS, 110 MB)
```

## Performance

- **Initial load**: ~2-3 seconds (loading both embedding sets: 380 MB total)
- **Model loading**: ~5-10 seconds (first time only - models are cached)
- **Embedding search**: <1 second (cosine similarity on 141K reviews)
- **BGE Reranking**: ~2-15 seconds depending on candidate count (CPU)
  - 500 candidates: ~2-3 seconds
  - 1,000 candidates: ~5-6 seconds
  - 5,000 candidates: ~12-15 seconds
- **Total query time**: ~3-16 seconds (no external API calls)

## Limitations

- Embedding files are large (~400 MB total for both review and summary embeddings) - requires Git LFS for version control
- BGE reranking runs on CPU by default (GPU would be faster)
- Initial embedding generation takes 20-30 minutes for 141K reviews

## Future Improvements

- [ ] Add caching for repeated queries
- [ ] Support user-uploaded CSV files
- [ ] Add export functionality for results
- [ ] Add GPU support for faster reranking
- [ ] Add sentiment analysis visualization
- [ ] Support multi-lingual reviews
- [ ] Implement query result caching

## Project Idea

### Product Vision
This project demonstrates a **semantic search analytics platform** that transforms unstructured customer feedback into actionable insights. From a data product perspective, this tool addresses several key business use cases.

### Why Semantic Search Is Superior to Keyword Search

Traditional keyword search has fundamental limitations that semantic search overcomes:

**The Long Tail Problem**
- Customers express the same concept in countless ways: "no tea flavor", "doesn't taste like tea", "lacking tea taste", "missing the tea notes", "can't taste any tea", etc.
- Keyword search requires you to think of every possible variation - an impossible task
- Semantic search understands that all these phrases mean the same thing and retrieves them all with a single query

**Synonym and Paraphrase Coverage**
- Keyword search misses reviews using synonyms: searching "fast shipping" won't find "quick delivery" or "arrived promptly"
- Semantic search understands semantic meaning, automatically finding all conceptually similar reviews regardless of exact wording
- This completeness is critical for trend analysis - you can't spot emerging issues if you're missing 60% of relevant mentions

**Context Understanding**
- "Great product" vs "Great product, but broke after a week" - keyword search can't distinguish positive from negative
- Semantic search combined with reranking understands context and nuance, surfacing truly relevant results
- Better precision means less time wading through false positives

**Concept-Based Discovery**
- You can search for concepts you don't know the exact terms for: "problems with packaging" finds "damaged box", "poor wrapping", "crushed container", etc.
- Enables exploratory analysis without knowing specific keywords in advance
- Particularly powerful for understanding customer pain points expressed in unexpected language

**Multilingual and Spelling Robustness**
- Handles typos and spelling variations naturally: "recieved" vs "received", "delicious" vs "delicous"
- Can understand related concepts across word forms: "ship", "shipping", "shipped", "shipment"

### Business Value Propositions

**1. Customer Voice Analytics**
- **Problem**: Product teams receive thousands of reviews but lack tools to quickly identify specific themes or emerging issues
- **Solution**: Semantic search allows PMs to find all mentions of a concept (e.g., "shipping delays") regardless of exact wording, with instant visual trends
- **Impact**: Reduce time-to-insight from days (manual review reading) to seconds, with complete coverage instead of sampling

**2. Issue Trend Detection**
- **Problem**: Quality issues may go unnoticed until they become widespread, especially when customers describe them differently
- **Solution**: Time-based trend visualization shows when specific complaints started increasing, capturing all semantic variations
- **Impact**: Early detection enables faster response to product quality issues, reducing customer churn

**3. Feature Prioritization**
- **Problem**: Deciding which features customers care about most requires analyzing scattered feedback expressed in diverse language
- **Solution**: Search for feature concepts and quantify demand through comprehensive review counts and sentiment scores
- **Impact**: Data-driven roadmap decisions based on complete customer voice coverage rather than keyword sampling

**4. Competitive Intelligence**
- **Problem**: Understanding why customers choose or reject products compared to competitors
- **Solution**: Search semantic patterns to identify competitive advantages across all linguistic variations
- **Impact**: Inform positioning and marketing strategies with customer-validated differentiators

**5. Customer Success & Support**
- **Problem**: Support teams need to understand common pain points to create better documentation, but customers describe issues differently
- **Solution**: Identify all recurring issues with semantic search and relevance scoring to prioritize documentation improvements
- **Impact**: Reduce support ticket volume through proactive self-service content that addresses real customer language

### Technical Architecture Benefits

**Cost-Effective ML Pipeline**
- 100% local execution - no API keys or external services required
- Local embedding models eliminate ongoing inference costs
- Pre-computed embeddings enable sub-second search on 140K+ documents
- BGE reranker provides high-quality relevance scoring at low latency

**Scalable & Extensible**
- Two-stage search (fast retrieval → precise reranking) scales to millions of reviews
- Modular design allows swapping models or adding new features
- Streamlit enables rapid prototyping and stakeholder demos

**Privacy-First Design**
- All processing runs 100% locally on your machine
- No data sent to external services
- Perfect for sensitive data and on-premise deployments

### Target Users

1. **Product Managers**: Identify feature requests and prioritize roadmap
2. **Customer Success Teams**: Understand common pain points and improve onboarding
3. **Quality Assurance**: Detect product defects early through review patterns
4. **Marketing Teams**: Extract customer testimonials and understand messaging resonance
5. **Executives**: High-level trend dashboards for board presentations

This project showcases how modern NLP techniques can democratize access to customer insights, enabling data-driven decision-making across organizations without requiring specialized data science skills.

## License

MIT License

## Credits

Built with:
- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-base)
- [Plotly](https://plotly.com/)
