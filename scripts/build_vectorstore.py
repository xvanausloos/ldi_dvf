"""Script to build vector store from DVF dataset."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dvf.rag import DVFVectorStore

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "df_2020_2025_houses_ensues.csv"


def main():
    """Build vector store from DVF dataset."""
    parser = argparse.ArgumentParser(
        description="Build vector store from DVF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 100 rows
  %(prog)s --num-rows 100
  
  # Test with 1000 rows and custom batch size
  %(prog)s -n 1000 --batch-size 200
  
  # Index full dataset (requires confirmation)
  %(prog)s
        """,
    )
    parser.add_argument(
        "-n",
        "--num-rows",
        "--sample-size",
        type=int,
        dest="num_rows",
        default=None,
        help="Number of rows to index (for testing). If not specified, indexes full dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing embeddings (default: 100). Larger batches are faster but use more memory.",
    )
    args = parser.parse_args()

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df):,} rows")

    if args.num_rows:
        if args.num_rows > len(df):
            print(f"\n⚠️  Warning: Requested {args.num_rows:,} rows but dataset only has {len(df):,} rows.")
            print(f"   Indexing all {len(df):,} rows instead.")
            total_to_index = len(df)
        else:
            print(f"\n📊 Indexing {args.num_rows:,} rows (for testing)")
            total_to_index = args.num_rows
    else:
        total_to_index = len(df)
        print(f"\n⚠️  Indexing FULL dataset: {total_to_index:,} rows")
        print("   This will take a long time and use significant API credits.")
        response = input("   Continue? (y/N): ")
        if response.lower() != "y":
            print("Cancelled.")
            return

    print("\nInitializing vector store (Ensues only)...")
    vectorstore_dir = Path(__file__).parent.parent / "data" / "vectorstore_ensues"
    vector_store = DVFVectorStore(persist_directory=str(vectorstore_dir))

    print("\nIndexing dataset (this may take a while)...")
    print("Note: This will use OpenAI embeddings API. Make sure OPENAI_API_KEY is set.")
    print(f"Batch size: {args.batch_size}")

    # Create progress bar
    pbar = tqdm(total=total_to_index, unit="properties", desc="Indexing")

    def update_progress(current: int, total: int):
        """Update progress bar."""
        pbar.n = current
        pbar.refresh()

    try:
        vector_store.index_dataframe(
            df,
            batch_size=args.batch_size,
            sample_size=args.num_rows,
            progress_callback=update_progress,
        )
        pbar.n = total_to_index
        pbar.close()
        
        print("\n✅ Vector store built successfully!")
        print(f"Database saved to: {vector_store.persist_directory}")
    except KeyboardInterrupt:
        pbar.close()
        print("\n\n⚠️  Indexing interrupted by user.")
        print("Progress saved. You can resume by running the script again.")
    except Exception as e:
        pbar.close()
        print(f"\n\n❌ Error during indexing: {e}")
        raise


if __name__ == "__main__":
    main()
