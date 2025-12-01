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
    print("Generating embeddings (this may take 10-15 minutes)...")
    batch_size = 1000
    all_embeddings = []

    for i in range(0, len(df), batch_size):
        batch_texts = df['combined_text'].iloc[i:i+batch_size].tolist()
        batch_embeddings = model.encode(
            batch_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # Normalized for cosine similarity
        )
        all_embeddings.append(batch_embeddings)
        print(f"Processed {min(i+batch_size, len(df))}/{len(df)} reviews")

    embeddings = np.vstack(all_embeddings)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings size: {embeddings.nbytes / (1024*1024):.2f} MB")

    # Save embeddings and metadata
    print("Saving embeddings...")
    output_dir = Path('dataset')
    output_dir.mkdir(exist_ok=True)

    # Save as compressed numpy format
    np.savez_compressed(
        output_dir / 'embeddings.npz',
        embeddings=embeddings.astype(np.float32)
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

    compressed_size = (output_dir / 'embeddings.npz').stat().st_size / (1024*1024)
    metadata_size = (output_dir / 'metadata.pkl').stat().st_size / (1024*1024)

    print(f"\nDone!")
    print(f"Embeddings saved to: {output_dir / 'embeddings.npz'} ({compressed_size:.2f} MB)")
    print(f"Metadata saved to: {output_dir / 'metadata.pkl'} ({metadata_size:.2f} MB)")
    print(f"Total size: {compressed_size + metadata_size:.2f} MB")

if __name__ == "__main__":
    generate_embeddings()
