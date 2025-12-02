"""
Generate embeddings for Amazon reviews dataset.
This script should be run once to create the embeddings file.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

def generate_embeddings():
    print("Loading datasets...")
    df1 = pd.read_csv('dataset/amazon_review_part1.csv')
    df2 = pd.read_csv('dataset/amazon_review_part2.csv')
    df = pd.concat([df1, df2], ignore_index=True)

    print(f"Total reviews: {len(df)}")

    # Clean and prepare text
    print("Preparing text data...")
    df['Text'] = df['Text'].fillna('')
    df['Summary'] = df['Summary'].fillna('')
    df['combined_text'] = df['Summary'] + ' ' + df['Text']

    # Load embedding model
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Generate embeddings in batches
    print("Generating embeddings for review text (this may take 10-15 minutes)...")
    batch_size = 1000
    all_review_embeddings = []

    for i in range(0, len(df), batch_size):
        batch_texts = df['Text'].iloc[i:i+batch_size].tolist()
        batch_embeddings = model.encode(
            batch_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # Normalized for cosine similarity
        )
        all_review_embeddings.append(batch_embeddings)
        print(f"Processed {min(i+batch_size, len(df))}/{len(df)} review texts")

    review_embeddings = np.vstack(all_review_embeddings)

    print(f"Review embeddings shape: {review_embeddings.shape}")
    print(f"Review embeddings size: {review_embeddings.nbytes / (1024*1024):.2f} MB")

    # Generate embeddings for summaries
    print("\nGenerating embeddings for summaries...")
    all_summary_embeddings = []

    for i in range(0, len(df), batch_size):
        batch_summaries = df['Summary'].iloc[i:i+batch_size].tolist()
        batch_embeddings = model.encode(
            batch_summaries,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        all_summary_embeddings.append(batch_embeddings)
        print(f"Processed {min(i+batch_size, len(df))}/{len(df)} summaries")

    summary_embeddings = np.vstack(all_summary_embeddings)

    print(f"Summary embeddings shape: {summary_embeddings.shape}")
    print(f"Summary embeddings size: {summary_embeddings.nbytes / (1024*1024):.2f} MB")

    # Save embeddings and metadata
    print("\nSaving embeddings...")
    output_dir = Path('dataset')
    output_dir.mkdir(exist_ok=True)

    # Save both review and summary embeddings as compressed numpy format
    np.savez_compressed(
        output_dir / 'review_embeddings.npz',
        embeddings=review_embeddings.astype(np.float32)
    )

    np.savez_compressed(
        output_dir / 'summary_embeddings.npz',
        embeddings=summary_embeddings.astype(np.float32)
    )

    # Save metadata (indices, dates, texts)
    metadata = {
        'ids': df['Id'].values,
        'dates': df['Date'].values,
        'summaries': df['Summary'].values,
        'texts': df['Text'].values,
        'scores': df['Score'].values,
        'combined_texts': df['combined_text'].values
    }

    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f, protocol=4)

    review_emb_size = (output_dir / 'review_embeddings.npz').stat().st_size / (1024*1024)
    summary_emb_size = (output_dir / 'summary_embeddings.npz').stat().st_size / (1024*1024)
    metadata_size = (output_dir / 'metadata.pkl').stat().st_size / (1024*1024)

    print(f"\nDone!")
    print(f"Review embeddings saved to: {output_dir / 'review_embeddings.npz'} ({review_emb_size:.2f} MB)")
    print(f"Summary embeddings saved to: {output_dir / 'summary_embeddings.npz'} ({summary_emb_size:.2f} MB)")
    print(f"Metadata saved to: {output_dir / 'metadata.pkl'} ({metadata_size:.2f} MB)")
    print(f"Total size: {review_emb_size + summary_emb_size + metadata_size:.2f} MB")

if __name__ == "__main__":
    generate_embeddings()
