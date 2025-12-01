# LLM Text Visualizer

A Streamlit app for semantic search and trend analysis of Amazon product reviews using LLMs and sentence transformers.

## Features

- 🔍 **Natural Language Questions** - Ask questions like "How many people complain about no tea flavor?"
- 🤖 **LLM-Powered Search** - Uses OpenRouter to extract search phrases from natural language
- 📊 **Semantic Search** - Fast local embeddings with sentence-transformers (141K+ reviews)
- 🎯 **Smart Reranking** - BGE Reranker cross-encoder for accurate relevance scoring (only keeps scores > 0.60)
- 📈 **Trend Visualization** - See mentions over time (day/month/year granularity)
- 🥧 **Score Distribution** - Pie chart showing relevance score ranges
- 📝 **Detailed Results** - Collapsible sections with top 10 reviews per period

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Embeddings

**Important:** You must generate embeddings before running the app. This is a one-time process that takes 10-15 minutes.

```bash
python generate_embeddings.py
```

This will create:
- `dataset/embeddings.npz` (~80-100 MB compressed)
- `dataset/metadata.pkl` (~50-70 MB)

**Note:** These files are tracked with Git LFS. If you're cloning this repo, make sure you have Git LFS installed:

```bash
git lfs install
git lfs pull
```

### 3. Get OpenRouter API Key

1. Go to https://openrouter.ai/
2. Sign up for a free account
3. Generate an API key
4. Copy the key (starts with `sk-or-v1-...`)

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Enter API Key** - Paste your OpenRouter API key in the sidebar
2. **Ask a Question** - Type a natural language question about the reviews
3. **Select Granularity** - Choose day/month/year for time aggregation
4. **Choose Chart Type** - Toggle between trend chart and score distribution
5. **Search** - Click the search button to run semantic search
6. **Explore Results** - Expand periods to see top-scored reviews

## Example Questions

- "How many people complain about no tea flavor?"
- "Find reviews mentioning fast shipping"
- "Who said the product expired quickly?"
- "People complaining about taste"
- "Reviews about packaging quality"

## Architecture

### Pipeline Flow

```
User Question
    ↓
LLM Extract Search Phrase (OpenRouter)
    ↓
Generate Query Embedding (local sentence-transformers)
    ↓
Cosine Similarity Search → Top 500 candidates
    ↓
LLM Reranking (OpenRouter) → Keep scores > 0.60
    ↓
Aggregate by Time Period
    ↓
Visualize + Show Results
```

### Models Used

- **Query Extraction**: `nvidia/nemotron-nano-9b-v2:free` (OpenRouter - free)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local)
- **Reranking**: `BAAI/bge-reranker-base` (local cross-encoder)

The OpenRouter model is completely free, and the local models run on your machine!

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
├── generate_embeddings.py          # One-time embedding generation
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitattributes                  # Git LFS configuration
├── .gitignore                      # Git ignore patterns
└── dataset/
    ├── amazon_review_part1.csv     # First half of dataset
    ├── amazon_review_part2.csv     # Second half of dataset
    ├── embeddings.npz              # Pre-computed embeddings (Git LFS)
    └── metadata.pkl                # Review metadata (Git LFS)
```

## Performance

- **Initial load**: ~2-3 seconds (loading embeddings)
- **Model loading**: ~5-10 seconds (first time only - models are cached)
- **Search extraction**: ~1-2 seconds (LLM call)
- **Embedding search**: <1 second (local computation)
- **BGE Reranking**: ~5-10 seconds (500 candidates on CPU)
- **Total query time**: ~7-14 seconds

## Limitations

- OpenRouter free tier may have rate limits during peak times
- Embedding files are large (~300 MB total) - requires Git LFS for version control
- BGE reranking runs on CPU by default (GPU would be faster)

## Future Improvements

- [ ] Add caching for repeated queries
- [ ] Support user-uploaded CSV files
- [ ] Add export functionality for results
- [ ] Add GPU support for faster reranking
- [ ] Add sentiment analysis visualization
- [ ] Support multi-lingual reviews
- [ ] Implement query result caching

## Project Idea: Data Product Manager Perspective

### Product Vision
This project demonstrates a **semantic search analytics platform** that transforms unstructured customer feedback into actionable insights. From a data product perspective, this tool addresses several key business use cases:

### Business Value Propositions

**1. Customer Voice Analytics**
- **Problem**: Product teams receive thousands of reviews but lack tools to quickly identify specific themes or emerging issues
- **Solution**: Natural language search allows PMs to ask questions like "How many customers complain about shipping delays in Q4 2023?" and get instant visual trends
- **Impact**: Reduce time-to-insight from days (manual review reading) to seconds (semantic search)

**2. Issue Trend Detection**
- **Problem**: Quality issues may go unnoticed until they become widespread
- **Solution**: Time-based trend visualization shows when specific complaints (e.g., "product arrived damaged") started increasing
- **Impact**: Early detection enables faster response to product quality issues, reducing customer churn

**3. Feature Prioritization**
- **Problem**: Deciding which features customers care about most requires analyzing scattered feedback
- **Solution**: Search for feature mentions ("wish it had X") and quantify demand through review counts and sentiment scores
- **Impact**: Data-driven roadmap decisions based on actual customer voice rather than assumptions

**4. Competitive Intelligence**
- **Problem**: Understanding why customers choose or reject products compared to competitors
- **Solution**: Search patterns like "better than [competitor]" or "switched from [competitor]" to identify competitive advantages
- **Impact**: Inform positioning and marketing strategies with customer-validated differentiators

**5. Customer Success & Support**
- **Problem**: Support teams need to understand common pain points to create better documentation
- **Solution**: Identify recurring issues (e.g., "confusing setup instructions") with relevance scoring to prioritize documentation improvements
- **Impact**: Reduce support ticket volume through proactive self-service content

### Technical Architecture Benefits

**Cost-Effective ML Pipeline**
- Leverages free LLM API (OpenRouter) for query understanding
- Local embedding models eliminate ongoing inference costs
- Pre-computed embeddings enable sub-second search on 140K+ documents
- BGE reranker provides LLM-quality relevance at 10x lower latency

**Scalable & Extensible**
- Two-stage search (fast retrieval → precise reranking) scales to millions of reviews
- Modular design allows swapping models or adding new features
- Streamlit enables rapid prototyping and stakeholder demos

**Privacy-First Design**
- All embeddings and reranking run locally
- Only query extraction uses external API (no sensitive review data sent)
- Can be deployed on-premise for compliance-sensitive industries

### Potential Product Roadmap

**Phase 1: Current State** ✅
- Natural language search over reviews
- Temporal trend analysis
- Relevance scoring and filtering

**Phase 2: Enhanced Analytics** (3-6 months)
- Automated insight generation (e.g., "Top 3 emerging complaints this month")
- Sentiment analysis integration (positive/negative trend lines)
- Export to CSV/PDF for executive reporting

**Phase 3: Proactive Monitoring** (6-12 months)
- Automated alerts when negative trends spike
- Comparative analysis across product lines
- Integration with ticketing systems (Jira, Zendesk)

**Phase 4: Enterprise Features** (12+ months)
- Multi-dataset support (combine reviews, support tickets, social media)
- Team collaboration (shared saved searches, annotations)
- API access for integration with BI tools (Tableau, PowerBI)

### Success Metrics

**User Adoption**
- Time saved per insight query (target: 95% reduction vs manual review)
- Weekly active users among product/support teams
- Number of searches per user (engagement proxy)

**Business Impact**
- Reduction in time-to-resolution for customer complaints
- Increase in feature adoption driven by customer-validated priorities
- Decrease in negative review trends after issue identification

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
- [OpenRouter](https://openrouter.ai/)
- [Plotly](https://plotly.com/)
