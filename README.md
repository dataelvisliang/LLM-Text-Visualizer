# Text Visualizer (Powered by Embedding Model)

A Streamlit app for semantic search and trend analysis of Amazon product reviews using sentence transformers and BGE reranker.

## Demo

### Main Interface
![Main Interface](assets/Main%20Interface.png)

### Search Result Example
![Search Result Example](assets/Search%20Result%20Example.png)

## Features

- 🔍 **Direct Search** - Enter keywords or phrases to search reviews instantly
- 📊 **Semantic Search** - Fast local embeddings with sentence-transformers (141K+ reviews)
- 🎯 **Smart Reranking** - BGE Reranker cross-encoder for accurate relevance scoring with adjustable score threshold (0.60-1.00)
- 🎚️ **Flexible Controls** - Dual top-k sliders for retrieval and reranking, customizable score filtering
- 📅 **Date Range Filter** - Focus on specific time periods with start and end date selectors
- 📈 **Trend Visualization** - See mentions over time (month/year granularity)
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

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Enter Search Phrase** - Type keywords or phrases to search for (e.g., "no tea flavor", "fast shipping")
2. **Adjust Parameters** - Set top-k values for retrieval and reranking
3. **Set Date Range** - Filter reviews by date range
4. **Set Score Threshold** - Adjust minimum relevance score (0.60-1.00)
5. **Select Visualization** - Choose time granularity (month/year) and chart type
6. **Search** - Click the search button to run semantic search
7. **Explore Results** - View trends and expand periods to see top-scored reviews

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

## Project Idea:

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
- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-base)
- [Plotly](https://plotly.com/)
