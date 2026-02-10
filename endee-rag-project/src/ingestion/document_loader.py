"""
Document loader for reading files from the filesystem.
Supports .txt, .md, and .pdf formats.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from src.utils import setup_logger, handle_error

# PDF support
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class DocumentLoader:
    """Load documents from filesystem."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf'}
    
    def __init__(self):
        """Initialize document loader."""
        self.logger = setup_logger(__name__)
    
    def load_file(self, file_path: str) -> Optional[Tuple[str, str]]:
        """
        Load a single document file.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Tuple of (filename, content) or None if failed
        """
        path = Path(file_path)
        
        if not path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            self.logger.warning(f"Unsupported file type: {path.suffix}")
            return None
        
        try:
            # Handle PDF files
            if path.suffix.lower() == '.pdf':
                return self._load_pdf(path)
            
            # Handle text files (.txt, .md)
            else:
                return self._load_text(path)
        
        except Exception as e:
            handle_error(self.logger, e, f"load_file: {file_path}")
            return None
    
    def _load_text(self, path: Path) -> Tuple[str, str]:
        """Load plain text or markdown file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        self.logger.info(f"Loaded text file: {path.name} ({len(content)} chars)")
        return path.name, content
    
    def _load_pdf(self, path: Path) -> Optional[Tuple[str, str]]:
        """Load PDF file and extract text."""
        if not PDF_AVAILABLE:
            self.logger.error("PyPDF2 not installed. Cannot load PDF files.")
            return None
        
        try:
            with open(path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                
                # Extract text from all pages
                text_parts = []
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text_parts.append(page.extract_text())
                
                content = '\n'.join(text_parts)
                
                self.logger.info(
                    f"Loaded PDF file: {path.name} "
                    f"({num_pages} pages, {len(content)} chars)"
                )
                return path.name, content
        
        except Exception as e:
            handle_error(self.logger, e, f"load_pdf: {path.name}")
            return None
    
    def load_directory(self, directory_path: str) -> List[Tuple[str, str]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
        
        Returns:
            List of (filename, content) tuples
        """
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            self.logger.error(f"Directory not found: {directory_path}")
            return []
        
        if not dir_path.is_dir():
            self.logger.error(f"Not a directory: {directory_path}")
            return []
        
        documents = []
        
        # Recursively find all supported files
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in dir_path.rglob(f"*{ext}"):
                result = self.load_file(str(file_path))
                if result:
                    documents.append(result)
        
        self.logger.info(
            f"Loaded {len(documents)} documents from {directory_path}"
        )
        return documents
