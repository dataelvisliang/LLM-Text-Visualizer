"""
Text Visualizer (Powered by Embedding Model) - Amazon Reviews Semantic Search & Trend Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Text Visualizer (Powered by Embedding Model)",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'reranker' not in st.session_state:
    st.session_state.reranker = None
if 'preview_seed' not in st.session_state:
    st.session_state.preview_seed = 0


@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model"""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@st.cache_resource
def load_reranker_model():
    """Load the BGE reranker cross-encoder model"""
    return CrossEncoder('BAAI/bge-reranker-base')


@st.cache_data
def load_embeddings_and_metadata():
    """Load pre-computed embeddings and metadata"""
    embeddings_path = Path('dataset/embeddings.npz')
    metadata_path = Path('dataset/metadata.pkl')

    if not embeddings_path.exists() or not metadata_path.exists():
        return None, None

    # Load embeddings
    embeddings_data = np.load(embeddings_path)
    embeddings = embeddings_data['embeddings']

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    return embeddings, metadata


def rerank_with_bge(query: str, texts: list, reranker: CrossEncoder) -> list:
    """
    Use BGE reranker cross-encoder to score relevance
    """
    # Create query-text pairs
    pairs = [[query, text] for text in texts]

    # Get scores from the cross-encoder
    scores = reranker.predict(pairs)

    # Normalize scores to 0-1 range using sigmoid-like transformation
    # BGE reranker typically outputs scores in a wider range
    normalized_scores = 1 / (1 + np.exp(-np.array(scores)))

    return normalized_scores.tolist()


def semantic_search(query_embedding, embeddings, top_k=500):
    """
    Perform cosine similarity search
    """
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]

    return top_indices, top_scores


def aggregate_by_period(df, period='month'):
    """
    Aggregate results by time period
    """
    df['date'] = pd.to_datetime(df['date'])

    if period == 'day':
        df['period'] = df['date'].dt.strftime('%Y-%m-%d')
    elif period == 'month':
        df['period'] = df['date'].dt.strftime('%Y-%m')
    else:  # year
        df['period'] = df['date'].dt.strftime('%Y')

    # Count by period
    period_counts = df.groupby('period').size().reset_index(name='count')
    period_counts = period_counts.sort_values('period')

    return period_counts, df


def create_trend_chart(period_counts, period_type):
    """
    Create trend line/bar chart with improved styling
    """
    fig = go.Figure()

    # Add bar trace with gradient color
    fig.add_trace(go.Bar(
        x=period_counts['period'],
        y=period_counts['count'],
        name='Review Count',
        marker=dict(
            color=period_counts['count'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Count"),
            line=dict(color='rgba(255,255,255,0.3)', width=1.5)
        ),
        text=period_counts['count'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Reviews: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=f"📈 Review Mentions Over Time ({period_type.capitalize()})",
            font=dict(size=20, color='#2c3e50', family="Arial Black")
        ),
        xaxis=dict(
            title=dict(text=period_type.capitalize(), font=dict(size=14, color='#34495e')),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)'
        ),
        yaxis=dict(
            title=dict(text="Number of Reviews", font=dict(size=14, color='#34495e')),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        hovermode='x unified',
        height=450,
        plot_bgcolor='rgba(250,250,250,0.9)',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12),
        margin=dict(t=80, b=60, l=60, r=60)
    )

    return fig


def create_score_distribution_chart(scores):
    """
    Create donut chart for score distribution with improved styling
    """
    bins = [0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0.6-0.7 (Low)', '0.7-0.8 (Medium)', '0.8-0.9 (High)', '0.9-1.0 (Excellent)']

    score_counts = []
    for i in range(len(bins)-1):
        count = ((scores >= bins[i]) & (scores < bins[i+1])).sum()
        score_counts.append(count)
    # Include exact 1.0 scores in the last bin
    if len(scores) > 0:
        score_counts[-1] = ((scores >= bins[-2]) & (scores <= bins[-1])).sum()

    # Custom colors with gradient effect
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=score_counts,
        hole=0.4,
        marker=dict(
            colors=colors,
            line=dict(color='white', width=3)
        ),
        textinfo='label+percent',
        textfont=dict(size=13, family="Arial", color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        pull=[0.05, 0.05, 0.05, 0.1]  # Pull out the highest quality slice
    )])

    fig.update_layout(
        title=dict(
            text="🎯 Relevance Score Distribution",
            font=dict(size=20, color='#2c3e50', family="Arial Black")
        ),
        height=450,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=12)
        ),
        paper_bgcolor='white',
        font=dict(family="Arial"),
        margin=dict(t=80, b=40, l=40, r=200)
    )

    return fig


# Main App
st.title("📊 Text Visualizer (Powered by Embedding Model)")
st.markdown("Semantic search and trend analysis for Amazon reviews")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses:
    - **BGE Reranker** for accurate relevance scoring
    - **Sentence Transformers** for local embeddings
    - **Semantic search** to find relevant reviews
    """)

    st.markdown("---")
    st.markdown("### Models Used")
    st.markdown("""
    - BAAI/bge-reranker-base (local)
    - sentence-transformers/all-MiniLM-L6-v2 (local)
    """)

# Load embeddings and metadata
if st.session_state.embeddings is None:
    with st.spinner("Loading embeddings..."):
        embeddings, metadata = load_embeddings_and_metadata()

        if embeddings is None:
            st.error("⚠️ Embeddings not found! Please run `generate_embeddings.py` first.")
            st.code("python generate_embeddings.py", language="bash")
            st.stop()

        st.session_state.embeddings = embeddings
        st.session_state.metadata = metadata
        st.success(f"✅ Loaded {len(embeddings):,} review embeddings")

# Load embedding model
if st.session_state.model is None:
    with st.spinner("Loading sentence transformer model..."):
        st.session_state.model = load_embedding_model()

# Load reranker model
if st.session_state.reranker is None:
    with st.spinner("Loading BGE reranker model..."):
        st.session_state.reranker = load_reranker_model()

# Random Dataset Preview
st.header("📋 Dataset Preview")
st.markdown("*Here are 6 random reviews from the dataset to give you ideas for what to search:*")

# Create a dataframe from metadata and sample 6 random rows
if st.session_state.metadata is not None:
    # Use seed for reproducible random selection until refresh is clicked
    np.random.seed(st.session_state.preview_seed)
    random_indices = np.random.choice(len(st.session_state.metadata['texts']), 6, replace=False)

    preview_df = pd.DataFrame({
        'Date': [st.session_state.metadata['dates'][idx] for idx in random_indices],
        'Score': [st.session_state.metadata['scores'][idx] for idx in random_indices],
        'Summary': [st.session_state.metadata['summaries'][idx] for idx in random_indices],
        'Review': [st.session_state.metadata['texts'][idx][:200] + '...' if len(st.session_state.metadata['texts'][idx]) > 200 else st.session_state.metadata['texts'][idx] for idx in random_indices]
    })

    # Style the dataframe
    st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Score": st.column_config.NumberColumn("⭐ Score", format="%d/5"),
            "Summary": st.column_config.TextColumn("Summary", width="medium"),
            "Review": st.column_config.TextColumn("Review Text", width="large")
        }
    )

    # Refresh button below the table, left-justified
    if st.button("⭮ Refresh", help="Load new random reviews", key="refresh_preview"):
        st.session_state.preview_seed += 1
        st.rerun()

st.markdown("---")

# Main search interface
st.header("🔍 Search Reviews")

search_query = st.text_input(
    "Search phrase",
    placeholder="no tea flavor",
    help="Enter keywords or phrases to search for in reviews"
)

st.markdown("##### Search Parameters")
col1, col2 = st.columns([1, 1])
with col1:
    top_k_retrieval = st.slider(
        "🔍 Top-K for Cosine Similarity",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Number of candidates to retrieve using cosine similarity. Higher = more comprehensive but slower."
    )
with col2:
    top_k_rerank = st.slider(
        "🎯 Top-K for Reranking",
        min_value=50,
        max_value=1000,
        value=500,
        step=50,
        help="Number of top candidates to rerank with BGE. Must be ≤ cosine similarity top-k. Higher = better quality but slower."
    )

# Ensure rerank top_k doesn't exceed retrieval top_k
if top_k_rerank > top_k_retrieval:
    st.warning(f"⚠️ Rerank Top-K ({top_k_rerank}) cannot exceed Retrieval Top-K ({top_k_retrieval}). Adjusting to {top_k_retrieval}.")
    top_k_rerank = top_k_retrieval

st.markdown("##### Visualization Options")
col3, col4 = st.columns([1, 1])
with col3:
    period_type = st.selectbox("Time Granularity", ["month", "year"], index=0)
with col4:
    chart_type = st.radio("Chart Type", ["Trend", "Score Distribution"], horizontal=True)

# Date range selector
if st.session_state.metadata is not None:
    # Get min and max dates from metadata
    all_dates = pd.to_datetime(st.session_state.metadata['dates'])
    min_date = all_dates.min().date()
    max_date = all_dates.max().date()

    st.markdown("##### Date Range Filter")
    col5, col6 = st.columns([1, 1])
    with col5:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            help="Filter reviews from this date onwards"
        )
    with col6:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            help="Filter reviews up to this date"
        )

    # Validate date range
    if start_date > end_date:
        st.warning("⚠️ Start date must be before end date. Adjusting...")
        start_date = end_date

    # Score threshold filter
    st.markdown("##### Score Filter")
    score_threshold = st.slider(
        "Minimum Relevance Score",
        min_value=0.60,
        max_value=1.00,
        value=0.60,
        step=0.05,
        help="Only show reviews with relevance score greater than this threshold"
    )

if st.button("🔎 Search", type="primary", disabled=not search_query):
    if not search_query:
        st.warning("Please enter a search phrase")
    else:
        # Step 1: Generate query embedding
        with st.spinner("Generating query embedding..."):
            query_embedding = st.session_state.model.encode(
                [search_query],
                normalize_embeddings=True
            )[0]

        # Step 3: Semantic search
        with st.spinner(f"Searching through 141,210 reviews (retrieving top {top_k_retrieval})..."):
            top_indices, top_scores = semantic_search(
                query_embedding,
                st.session_state.embeddings,
                top_k=top_k_retrieval
            )

            st.success(f"✅ Found {len(top_indices)} candidate matches from cosine similarity")

        # Step 4: Select top candidates for reranking
        # Only rerank the top_k_rerank candidates
        rerank_indices = top_indices[:top_k_rerank]

        st.info(f"🎯 Reranking top {len(rerank_indices)} candidates with BGE cross-encoder...")

        # Step 5: Rerank with BGE reranker
        with st.spinner(f"Reranking {len(rerank_indices)} results with BGE cross-encoder..."):
            candidate_texts = [
                st.session_state.metadata['combined_texts'][idx]
                for idx in rerank_indices
            ]

            rerank_scores = rerank_with_bge(
                search_query,
                candidate_texts,
                st.session_state.reranker
            )

            rerank_scores = np.array(rerank_scores)

        # Step 6: Filter by threshold
        mask = rerank_scores >= score_threshold

        filtered_indices = rerank_indices[mask]
        filtered_scores = rerank_scores[mask]

        if len(filtered_indices) == 0:
            st.warning(f"No reviews found with relevance score > {score_threshold:.2f}. Try lowering the score threshold or a different query.")
            st.stop()

        st.success(f"✅ Found {len(filtered_indices)} highly relevant reviews (score > {score_threshold:.2f})")

        # Step 7: Prepare results dataframe
        results_df = pd.DataFrame({
            'date': [st.session_state.metadata['dates'][idx] for idx in filtered_indices],
            'text': [st.session_state.metadata['texts'][idx] for idx in filtered_indices],
            'summary': [st.session_state.metadata['summaries'][idx] for idx in filtered_indices],
            'score': filtered_scores
        })

        # Step 7.5: Filter by date range
        results_df['date'] = pd.to_datetime(results_df['date'])
        date_mask = (results_df['date'].dt.date >= start_date) & (results_df['date'].dt.date <= end_date)
        results_df = results_df[date_mask]

        if len(results_df) == 0:
            st.warning(f"No reviews found in the date range {start_date} to {end_date}. Try adjusting the date range.")
            st.stop()

        st.info(f"📅 Filtered to {len(results_df)} reviews in date range: {start_date} to {end_date}")

        # Step 8: Aggregate by period
        period_counts, results_df = aggregate_by_period(results_df, period_type)

        # Step 9: Visualizations
        st.header("📈 Results")

        if chart_type == "Trend":
            fig = create_trend_chart(period_counts, period_type)
        else:
            fig = create_score_distribution_chart(results_df['score'].values)

        st.plotly_chart(fig, use_container_width=True)

        # Step 10: Collapsible sections per period
        st.header("📝 Top Reviews by Period")

        periods = sorted(results_df['period'].unique())

        for period in periods:
            period_df = results_df[results_df['period'] == period].sort_values(
                'score', ascending=False
            ).head(10)

            with st.expander(f"**{period}** ({len(period_df)} reviews)", expanded=False):
                for idx, row in period_df.iterrows():
                    score_color = "🟢" if row['score'] >= 0.9 else "🟡" if row['score'] >= 0.8 else "🟠"
                    formatted_date = row['date'].strftime('%Y-%m-%d')

                    st.markdown(f"""
                    {score_color} **Score: {row['score']:.3f}** | Date: {formatted_date}

                    **{row['summary']}**

                    {row['text'][:500]}{'...' if len(row['text']) > 500 else ''}

                    ---
                    """)

st.markdown("---")
st.markdown("Made with Streamlit • Powered by OpenRouter & Sentence Transformers")

# python -m streamlit run app.py