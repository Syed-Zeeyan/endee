"""
Main CLI entry point for Endee RAG Knowledge Assistant.
Provides commands for ingesting documents and querying the knowledge base.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ENDEE_URL, COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
)
from utils import setup_logger
from endee.endee_client import EndeeClient
from embeddings.embedding_model import EmbeddingModel
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import TextChunker
from retrieval.llm_client import LLMClient
from retrieval.query_engine import QueryEngine


logger = setup_logger("main")


def ingest_documents(directory: str):
    """
    Ingest documents from a directory into Endee.
    
    Args:
        directory: Path to directory containing documents
    """
    logger.info("=" * 60)
    logger.info("DOCUMENT INGESTION PIPELINE")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("Initializing components...")
    endee_client = EndeeClient(ENDEE_URL)
    embedding_model = EmbeddingModel(EMBEDDING_MODEL)
    document_loader = DocumentLoader()
    text_chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    
    # Health check
    logger.info("Checking Endee connectivity...")
    if not endee_client.health_check():
        logger.error("Endee is not accessible. Please start Endee with: docker-compose up -d")
        return
    
    # Create or verify collection
    logger.info(f"Setting up collection: {COLLECTION_NAME}")
    try:
        endee_client.create_collection(
            name=COLLECTION_NAME,
            dimension=EMBEDDING_DIMENSION
        )
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        return
    
    # Load documents
    logger.info(f"Loading documents from: {directory}")
    documents = document_loader.load_directory(directory)
    
    if not documents:
        logger.warning("No documents found!")
        return
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Chunk documents
    logger.info("Chunking documents...")
    chunks = text_chunker.chunk_documents(documents)
    logger.info(f"Generated {len(chunks)} chunks")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    chunk_texts = [chunk['chunk_text'] for chunk in chunks]
    embeddings = embedding_model.encode_batch(chunk_texts)
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Prepare vectors for Endee
    logger.info("Preparing vectors for insertion...")
    vectors = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_obj = {
            "id": f"{chunk['filename']}_chunk_{chunk['chunk_id']}",
            "vector": embedding,
            "metadata": {
                "source_file": chunk['filename'],
                "chunk_id": chunk['chunk_id'],
                "chunk_text": chunk['chunk_text'],
                "timestamp": datetime.now().isoformat(),
                "total_chunks": chunk['total_chunks']
            }
        }
        vectors.append(vector_obj)
    
    # Insert into Endee (in batches if large)
    batch_size = 100
    total_inserted = 0
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        logger.info(f"Inserting batch {i // batch_size + 1} ({len(batch)} vectors)...")
        
        try:
            endee_client.insert_vectors(COLLECTION_NAME, batch)
            total_inserted += len(batch)
        except Exception as e:
            logger.error(f"Failed to insert batch: {str(e)}")
    
    logger.info("=" * 60)
    logger.info(f"✓ Ingestion complete! Inserted {total_inserted} vectors")
    logger.info("=" * 60)


def query_knowledge_base(question: str):
    """
    Query the knowledge base using RAG.
    
    Args:
        question: User question
    """
    logger.info("=" * 60)
    logger.info("RAG QUERY")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("Initializing components...")
    endee_client = EndeeClient(ENDEE_URL)
    embedding_model = EmbeddingModel(EMBEDDING_MODEL)
    llm_client = LLMClient()  # No API key needed - local formatter
    
    # Health check
    if not endee_client.health_check():
        logger.error("Endee is not accessible. Please start Endee with: docker-compose up -d")
        return
    
    # Initialize query engine
    query_engine = QueryEngine(
        embedding_model=embedding_model,
        endee_client=endee_client,
        llm_client=llm_client,
        collection_name=COLLECTION_NAME,
        top_k=TOP_K
    )
    
    # Execute query
    logger.info(f"Query: {question}")
    logger.info("-" * 60)
    
    result = query_engine.query(question, verbose=True)
    
    # Display results
    print("\n" + "=" * 60)
    print(f"QUESTION: {result['query']}")
    print("=" * 60)
    
    if 'sources' in result and result['sources']:
        print(f"\nRETRIEVED CONTEXT ({len(result['sources'])} sources):")
        print("-" * 60)
        for idx, source in enumerate(result['sources'], start=1):
            print(f"\n{idx}. {source['source_file']} (similarity: {source['similarity_score']})")
            print(f"   {source['chunk_text']}")
    
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result['answer'])
    print("=" * 60 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Endee RAG Knowledge Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Ingest documents:
    python main.py ingest data/documents
  
  Query knowledge base:
    python main.py query "What is machine learning?"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Ingest documents into the knowledge base'
    )
    ingest_parser.add_argument(
        'directory',
        type=str,
        help='Directory containing documents to ingest'
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        'query',
        help='Query the knowledge base'
    )
    query_parser.add_argument(
        'question',
        type=str,
        help='Question to ask'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'ingest':
        ingest_documents(args.directory)
    elif args.command == 'query':
        query_knowledge_base(args.question)


if __name__ == "__main__":
    main()
