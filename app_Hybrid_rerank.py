# app.py
import os
import streamlit as st
import pinecone
from groq import Groq
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import PyPDF2
import re
import uuid
import time
from typing import List, Dict, Tuple, Any
import random

# Configure the page
st.set_page_config(
    page_title="PastPort Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _on_followup_click(question):
    # clear any previous pending
    st.session_state.pending_question = None
    # push the new question into the chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # mark it for processing
    st.session_state.pending_question = question
    # re‑run immediately
    st.rerun()


# Constants
# Handle API keys with fallback for development
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
except:
    # Fallback for development
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
    
    if not GEMINI_API_KEY:
        st.error("No Gemini API key found. Please set it up in .streamlit/secrets.toml or as an environment variable.")
    if not GROQ_API_KEY:
        st.error("No Groq API key found. Please set it up in .streamlit/secrets.toml or as an environment variable.")
    if not PINECONE_API_KEY:
        st.error("No Pinecone API key found. Please set it up in .streamlit/secrets.toml or as an environment variable.")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
EMBEDDING_DIMENSION = 1536  # Dimension for Gemini embedding
EMBEDDING_MODEL = "gemini-embedding-exp-03-07"
PINECONE_INDEX_NAME = "hg-index-hybrid"
PINECONE_CLOUD = "aws"  # or "gcp"
PINECONE_REGION = "us-east-1"  # Choose appropriate region
LLM_MODEL = "gemini/gemini-2.0-flash"

# Advanced configuration
SEARCH_CONFIG = {
    "default_results": 5,
    "max_results": 10,
    "min_similarity": 0.65  # Minimum similarity score for returned documents
}

# Initialize API clients
if GEMINI_API_KEY:
    from litellm import completion
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    # Set up Gemini API key
    genai.configure(api_key=GEMINI_API_KEY)
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    groq_client = Groq(api_key=GROQ_API_KEY)
if PINECONE_API_KEY:
    pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)

class PDFProcessor:
    """Handles all PDF processing operations with metadata extraction"""
    
    def extract_text_from_pdf(self, pdf_file) -> List[Dict]:
        """Extract text content and metadata from a PDF file"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages_data = []
        
        # Extract text and metadata from each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            # Get page metadata if available
            page_metadata = {
                "page_number": page_num + 1,
                "page_size": (page.mediabox.width, page.mediabox.height)
            }
            
            pages_data.append({
                "text": text,
                "metadata": page_metadata
            })
            
        return pages_data
    
    def extract_keywords_from_text(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords from a text chunk using TF-IDF-like approach"""
        # Remove common stopwords for economics/philosophy texts
        stopwords = {
            'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'was', 'for', 
            'it', 'with', 'as', 'be', 'on', 'by', 'this', 'are', 'or', 'at', 
            'from', 'have', 'an', 'they', 'their', 'has', 'will', 'would', 
            'should', 'could', 'been', 'not', 'there', 'which', 'when', 'who', 
            'what', 'where', 'why', 'how', 'all', 'any', 'but', 'if', 'then',
            'we', 'they', 'our', 'your', 'my', 'me', 'his', 'her', 'than', 'thus'
        }
        
        # Split into words, lowercase everything
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stopwords and count frequencies
        word_freq = {}
        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def detect_structure(self, pages_data: List[Dict]) -> Dict:
        """Detect document structure like chapters and sections"""
        structure = {
            "chapters": [],
            "sections": []
        }
        
        # Common patterns for chapter headings
        chapter_patterns = [
            r"(?:CHAPTER|Chapter)\s+(\d+|[IVXLCDM]+)(?:\s*:\s*)?([^\n]+)?",
            r"(?:\d+\.\s+)([A-Z][^\.]+)(?:\n|\.$)",
            r"^([A-Z][A-Z\s]+)$"  # All caps headings
        ]
        
        # Simplified chapter detection
        for page_idx, page_data in enumerate(pages_data):
            page_num = page_idx + 1
            text = page_data["text"]
            lines = text.split('\n')
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                # Check for chapter headings
                for pattern in chapter_patterns:
                    match = re.search(pattern, line)
                    if match:
                        if match.groups() and len(match.groups()) > 1:
                            chapter_num = match.group(1)
                            chapter_title = match.group(2) if len(match.groups()) > 1 else ""
                        else:
                            chapter_num = len(structure["chapters"]) + 1
                            chapter_title = match.group(1) if match.groups() else line
                            
                        chapter = {
                            "number": chapter_num,
                            "title": chapter_title.strip() if chapter_title else line,
                            "start_page": page_idx + 1,
                            "start_line": line_idx
                        }
                        structure["chapters"].append(chapter)
                        break
        
        # Set end pages for chapters
        for i in range(len(structure["chapters"])):
            if i < len(structure["chapters"]) - 1:
                structure["chapters"][i]["end_page"] = structure["chapters"][i+1]["start_page"] - 1
            else:
                structure["chapters"][i]["end_page"] = len(pages_data)
        
        return structure
    
    def clean_text(self, text: str) -> str:
        """Clean the extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\d+\n', '\n', text)
        # Remove common PDF artifacts
        text = re.sub(r'(\(cid:\d+\))', '', text)
        return text.strip()
    
    def chunk_text(self, pages_data: List[Dict], structure: Dict) -> List[Dict]:
        """Split text into chunks with enhanced metadata including keywords"""
        chunks = []
        
        # Helper function to find chapter for a given page
        def find_chapter(page_num):
            for chapter in structure["chapters"]:
                if chapter["start_page"] <= page_num <= chapter["end_page"]:
                    return {
                        "number": chapter["number"],
                        "title": chapter["title"]
                    }
            return None
        
        # Process each page and create chunks
        current_chunk_text = []
        current_chunk_size = 0
        current_page_start = 0
        current_chapter = None
        
        for page_idx, page_data in enumerate(pages_data):
            page_num = page_idx + 1
            page_text = self.clean_text(page_data["text"])
            page_lines = page_text.split('\n')
            
            # Get chapter for this page
            chapter = find_chapter(page_num)
            
            # Process page line by line
            for line in page_lines:
                words = line.split()
                
                # Add words to current chunk
                for word in words:
                    current_chunk_text.append(word)
                    current_chunk_size += len(word) + 1  # +1 for space
                
                # If chunk is large enough and at a good breaking point
                if current_chunk_size >= CHUNK_SIZE and line.endswith(('.', '?', '!')):
                    chunk_text = ' '.join(current_chunk_text)
                    chunk_id = str(uuid.uuid4())
                    
                    # Extract keywords from this chunk
                    keywords = self.extract_keywords_from_text(chunk_text)
                    
                    metadata = {
                        "chunk_id": chunk_id,
                        "start_page": current_page_start,
                        "end_page": page_num,
                        "keywords": keywords  # Add keywords to metadata
                    }
                    
                    # Add chapter info if available
                    if current_chapter:
                        metadata["chapter_number"] = current_chapter["number"]
                        metadata["chapter_title"] = current_chapter["title"]
                    
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": metadata
                    })
                    
                    # Keep some overlap for context
                    overlap_words = min(len(current_chunk_text), CHUNK_OVERLAP // 10)
                    current_chunk_text = current_chunk_text[-overlap_words:]
                    current_chunk_size = sum(len(word) + 1 for word in current_chunk_text)
                    current_page_start = page_num
                    current_chapter = chapter
        
        # Add the final chunk if there's anything left
        if current_chunk_text:
            chunk_text = ' '.join(current_chunk_text)
            chunk_id = str(uuid.uuid4())
            
            # Extract keywords from the final chunk
            keywords = self.extract_keywords_from_text(chunk_text)
            
            metadata = {
                "chunk_id": chunk_id,
                "start_page": current_page_start,
                "end_page": len(pages_data),
                "keywords": keywords  # Add keywords to metadata
            }
            
            # Add chapter info if available
            if current_chapter:
                metadata["chapter_number"] = current_chapter["number"]
                metadata["chapter_title"] = current_chapter["title"]
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": metadata
            })
            
        return chunks

class VectorDB:
    """Handles vector database operations for the RAG system"""
    
    def __init__(self):
        """Initialize the vector database client"""
        self.pc = pinecone_client
        
        # Check if the index exists, and create it if it doesn't
        self._create_index_if_not_exists()
        
        # Connect to the index
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
    
    def _create_index_if_not_exists(self):
        """Create the index if it doesn't already exist"""
        # Check if the index exists
        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            # Create the index
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            
            # Wait for the index to be ready
            while not self.pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
    
    def _create_embedding(self, text: str) -> list:
        """Create an embedding vector for the given text using Gemini"""
        try:
            # Check if text is too long for Gemini's token limit
            if len(text) > 25000:  # Approximation, adjust as needed
                # If text is too long, truncate it to fit within token limits
                text = text[:25000]  # Simple truncation
            
            # Create embedding using Gemini
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="SEMANTIC_SIMILARITY"
            )
            
            # Return the embedding as a list
            embedding = result["embedding"]
            
            # Check if all values are zero (which Pinecone rejects)
            if all(x == 0 for x in embedding):
                # Add a tiny random non-zero value to avoid Pinecone's rejection
                import random
                index_to_modify = random.randint(0, len(embedding) - 1)
                embedding[index_to_modify] = 0.0001
                print(f"Warning: Zero vector detected and fixed for text: {text[:50]}...")
                
            return embedding
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            # Create a fallback embedding that isn't all zeros
            fallback = [0.0] * EMBEDDING_DIMENSION
            fallback[0] = 0.0001  # Add a small non-zero value
            return fallback
    
    def add_documents(self, chunks: list, namespace: str):
        """Add document chunks to the vector database with enhanced keyword metadata"""
        # Process chunks in batches
        batch_size = 50  # Reduced batch size for safety
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            # Create a batch of chunks
            batch = chunks[i:min(i+batch_size, total_chunks)]
            
            # Create vectors for batch upsert
            vectors = []
            for chunk in batch:
                try:
                    # Create embedding for the text
                    embedding = self._create_embedding(chunk["text"])
                    
                    # Limit metadata size - only include essential fields
                    # Pinecone has a 40KB metadata size limit per vector
                    truncated_text = chunk["text"]
                    if len(truncated_text) > 8000:  # Roughly 20KB of text
                        truncated_text = truncated_text[:8000] + "..."
                    
                    # Create streamlined metadata
                    metadata = {
                        "text": truncated_text,
                        "start_page": chunk["metadata"].get("start_page", 0),
                        "end_page": chunk["metadata"].get("end_page", 0)
                    }
                    
                    # Add individual keywords as separate fields
                    # This approach avoids needing special operators
                    if "keywords" in chunk["metadata"]:
                        keywords = chunk["metadata"]["keywords"][:10]  # Limit to top 10 keywords
                        
                        # Store individual keywords with numerical suffixes
                        # This makes them directly searchable with $eq operator
                        for idx, keyword in enumerate(keywords):
                            key_name = f"kw_{idx}"
                            metadata[key_name] = keyword
                        
                        # Also store as comma-separated for reference/display
                        metadata["keywords_str"] = ",".join(keywords)
                    
                    # Only add chapter info if present
                    if "chapter_number" in chunk["metadata"]:
                        metadata["chapter_number"] = chunk["metadata"]["chapter_number"]
                    if "chapter_title" in chunk["metadata"]:
                        chapter_title = chunk["metadata"]["chapter_title"]
                        # Truncate long chapter titles
                        if len(chapter_title) > 200:
                            chapter_title = chapter_title[:200] + "..."
                        metadata["chapter_title"] = chapter_title
                    
                    # Create a vector record
                    vectors.append({
                        "id": chunk["id"],
                        "values": embedding,
                        "metadata": metadata
                    })
                except Exception as e:
                    print(f"Error processing chunk {chunk['id']}: {str(e)}")
                    # Continue with the next chunk
                    continue
            
            # Skip empty batches
            if not vectors:
                continue
                
            try:
                # Upsert vectors to Pinecone
                self.index.upsert(vectors=vectors, namespace=namespace)
                print(f"Successfully upserted batch {i//batch_size + 1}/{(total_chunks+batch_size-1)//batch_size}")
            except Exception as e:
                print(f"Error upserting batch to Pinecone: {str(e)}")
                # Continue with the next batch
                continue
    
    def extract_keywords_from_text(self, text: str, max_keywords: int = 15) -> List[str]:
        """Extract keywords from a text for search queries"""
        # Remove common stopwords for economics/philosophy texts
        stopwords = {
            'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'was', 'for', 
            'it', 'with', 'as', 'be', 'on', 'by', 'this', 'are', 'or', 'at', 
            'from', 'have', 'an', 'they', 'their', 'has', 'will', 'would', 
            'should', 'could', 'been', 'not', 'there', 'which', 'when', 'who', 
            'what', 'where', 'why', 'how', 'all', 'any', 'but', 'if', 'then',
            'we', 'they', 'our', 'your', 'my', 'me', 'his', 'her', 'than', 'thus'
        }
        
        # Split into words, lowercase everything
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stopwords and count frequencies
        word_freq = {}
        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def hybrid_search(self, 
                    query: str, 
                    namespace: str,
                    n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Consolidated hybrid search function with chapter awareness and reranking.
        This replaces all previous search methods with a single, comprehensive implementation.
        """
        # Create query embedding
        query_embedding = self._create_embedding(query)
        
        # Extract keywords from query
        query_keywords = self.extract_keywords_from_text(query)
        print(f"Query keywords: {query_keywords}")
        
        # Extract any chapter reference from the query
        chapter_info = self._extract_chapter_reference(query)
        if chapter_info:
            print(f"Detected chapter reference: {chapter_info}")
        
        # Build metadata filter based on chapters and keywords
        final_filter = self._build_search_filter(query_keywords, chapter_info)
        
        # Build search parameters
        search_params = {
            "namespace": namespace,
            "vector": query_embedding,
            "top_k": n_results * 2,  # Get more initial results for reranking
            "include_metadata": True
        }
        
        # Add filter if available
        if final_filter:
            search_params["filter"] = final_filter
            print(f"Using filter: {final_filter}")
        
        # Add reranking
        search_params["rerank"] = {
            "model": "bge-reranker-v2-m3",  # Or another suitable reranker
            "top_n": n_results,
            "rank_fields": ["text"]  # The field containing content to rerank
        }
        
        try:
            # Execute search
            search_results = self.index.query(**search_params)
            
            # Format results
            results = []
            for match in search_results.matches:
                # Get text content and metadata
                text = match.metadata.get("text", "")
                metadata = match.metadata
                
                # Count keyword matches for reference
                keyword_matches = 0
                for field_idx in range(10):
                    field_name = f"kw_{field_idx}"
                    if field_name in metadata and metadata[field_name] in query_keywords:
                        keyword_matches += 1
                
                # Prepare book and chapter information
                book_info = {
                    "title": "Progress and Poverty" if "Progress" in namespace else namespace.replace('book_', '').replace('_', ' ').title(),
                    "chapter": metadata.get("chapter_number", ""),
                    "chapter_title": metadata.get("chapter_title", ""),
                    "pages": f"{metadata.get('start_page', 'N/A')}-{metadata.get('end_page', 'N/A')}"
                }
                
                results.append({
                    "id": match.id,
                    "text": text,
                    "metadata": metadata,
                    "book_info": book_info,
                    "score": match.score,
                    "keyword_matches": keyword_matches
                })
            
            return results
            
        except Exception as e:
            # Handle errors gracefully
            print(f"Search error: {str(e)}")
            print(f"Filter used: {final_filter}")
            
            # Try progressive fallbacks
            return self._try_fallback_searches(query_embedding, namespace, n_results, chapter_info, query_keywords)
        
    def _extract_chapter_reference(self, query: str) -> Dict[str, Any]:
        """Extract chapter references from a query"""
        chapter_info = {}
        
        # Look for chapter numbers
        number_patterns = [
            r"chapter (\d+|[ivxlcdm]+)",
            r"chapter (\d+|[ivxlcdm]+) of",
            r"in chapter (\d+|[ivxlcdm]+)",
            r"book ([ivxlcdm]+)"
        ]
        
        for pattern in number_patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                if match.group(1):
                    chapter_info["number"] = match.group(1)
                    return chapter_info  # Return as soon as we find a chapter number
        
        # Look for chapter titles
        title_patterns = [
            r"chapter (?:on|about) ['\"]?([a-zA-Z\s]+)['\"]?",
            r"chapter ['\"]?([a-zA-Z\s]+)['\"]?",
            r"the chapter (?:on|about) ['\"]?([a-zA-Z\s]+)['\"]?"
        ]
        
        for pattern in title_patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                if match.group(1):
                    chapter_info["title"] = match.group(1).strip()
                    return chapter_info  # Return as soon as we find a chapter title
        
        return chapter_info  # Return empty dict if no chapter reference found

    def _build_search_filter(self, query_keywords, chapter_info):
        """Build a search filter using keywords and chapter information"""
        # Build chapter filter if available
        chapter_filter = None
        if chapter_info:
            if "number" in chapter_info:
                chapter_filter = {"chapter_number": {"$eq": str(chapter_info["number"])}}
            elif "title" in chapter_info:
                # If only title is provided, use a partial match on title
                chapter_filter = {"chapter_title": {"$match": chapter_info["title"]}}
        
        # Build keyword filter if available
        keyword_filter = None
        if query_keywords and len(query_keywords) > 0:
            # Only use top 3 keywords to avoid overly restrictive filter
            top_keywords = query_keywords[:3]
            
            # Create OR conditions for each keyword field (kw_0, kw_1, etc.)
            keyword_conditions = []
            
            # For each possible keyword field
            for field_idx in range(10):  # We store up to 10 keywords per document
                field_name = f"kw_{field_idx}"
                
                # Check if any of our top keywords match this field
                field_conditions = []
                for keyword in top_keywords:
                    field_conditions.append({
                        field_name: {"$eq": keyword}
                    })
                
                # Add conditions for this field if we have any
                if field_conditions:
                    keyword_conditions.append({"$or": field_conditions})
            
            # Combine all field conditions with OR
            if keyword_conditions:
                keyword_filter = {"$or": keyword_conditions}
        
        # Build combined filter if both chapter and keyword filters exist
        final_filter = None
        if chapter_filter and keyword_filter:
            final_filter = {"$and": [chapter_filter, keyword_filter]}
        elif chapter_filter:
            final_filter = chapter_filter
        elif keyword_filter:
            final_filter = keyword_filter
        
        return final_filter

    def _try_fallback_searches(self, query_embedding, namespace, n_results, chapter_info, query_keywords):
        """Try progressive fallback search strategies"""
        results = []
        
        # First fallback: Try with just chapter filter
        if chapter_info:
            try:
                chapter_filter = None
                if "number" in chapter_info:
                    chapter_filter = {"chapter_number": {"$eq": str(chapter_info["number"])}}
                elif "title" in chapter_info:
                    chapter_filter = {"chapter_title": {"$match": chapter_info["title"]}}
                
                if chapter_filter:
                    print(f"Trying fallback with chapter filter: {chapter_filter}")
                    
                    search_results = self.index.query(
                        namespace=namespace,
                        vector=query_embedding,
                        filter=chapter_filter,
                        top_k=n_results,
                        include_metadata=True
                    )
                    
                    if search_results.matches:
                        # Format results
                        for match in search_results.matches:
                            metadata = match.metadata
                            book_info = {
                                "title": "Progress and Poverty" if "Progress" in namespace else namespace.replace('book_', '').replace('_', ' ').title(),
                                "chapter": metadata.get("chapter_number", ""),
                                "chapter_title": metadata.get("chapter_title", ""),
                                "pages": f"{metadata.get('start_page', 'N/A')}-{metadata.get('end_page', 'N/A')}"
                            }
                            
                            results.append({
                                "id": match.id,
                                "text": metadata.get("text", ""),
                                "metadata": metadata,
                                "book_info": book_info,
                                "score": match.score,
                                "fallback": "chapter_only"
                            })
                        
                        return results
            except Exception as e:
                print(f"Chapter filter fallback error: {str(e)}")
        
        # Second fallback: Try with just semantic search
        try:
            print("Trying fallback with semantic search only (no filters)")
            search_results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=n_results,
                include_metadata=True
            )
            
            # Format results
            for match in search_results.matches:
                metadata = match.metadata
                book_info = {
                    "title": "Progress and Poverty" if "Progress" in namespace else namespace.replace('book_', '').replace('_', ' ').title(),
                    "chapter": metadata.get("chapter_number", ""),
                    "chapter_title": metadata.get("chapter_title", ""),
                    "pages": f"{metadata.get('start_page', 'N/A')}-{metadata.get('end_page', 'N/A')}"
                }
                
                results.append({
                    "id": match.id,
                    "text": metadata.get("text", ""),
                    "metadata": metadata,
                    "book_info": book_info,
                    "score": match.score,
                    "fallback": "semantic_only"
                })
            
            return results
        except Exception as e:
            print(f"Semantic search fallback error: {str(e)}")
            return []  # Return empty results if all attempts fail

    
    def get_chapter_info(self, namespace: str) -> list:
        """Get chapter information from the database"""
        try:
            # Get a sample of vectors to extract chapter metadata
            sample_results = self.index.query(
                namespace=namespace,
                vector=[0.0001] * EMBEDDING_DIMENSION,  # Non-zero dummy vector
                top_k=100,
                include_metadata=True
            )
            
            # Extract unique chapter information
            chapter_info = {}
            for match in sample_results.matches:
                metadata = match.metadata
                if "chapter_number" in metadata and "chapter_title" in metadata:
                    chapter_key = metadata["chapter_number"]
                    if chapter_key not in chapter_info:
                        chapter_info[chapter_key] = {
                            "number": metadata["chapter_number"],
                            "title": metadata["chapter_title"]
                        }
            
            # Convert to list and sort by chapter number
            chapters_list = list(chapter_info.values())
            return chapters_list
        except Exception as e:
            print(f"Error fetching chapter info: {str(e)}")
            return []
            
    def list_namespaces(self) -> List[str]:
        """List all namespaces in the index"""
        try:
            stats = self.index.describe_index_stats()
            namespaces = list(stats.namespaces.keys())
            return namespaces
        except Exception as e:
            print(f"Error listing namespaces: {str(e)}")
            return []

class RAGChatbot:
    """Main chatbot class that uses RAG with Gemini"""
    
    def __init__(self):
        """Initialize the chatbot"""
        self.vector_db = VectorDB()
        self.pdf_processor = PDFProcessor()
        self.conversation_history = []
        
    def process_pdf(self, pdf_file, namespace: str):
        """Process a PDF and store in vector database"""
        # Extract text and basic metadata
        pages_data = self.pdf_processor.extract_text_from_pdf(pdf_file)
        
        # Detect document structure (chapters, sections)
        structure = self.pdf_processor.detect_structure(pages_data)
        
        # Chunk the text with structural metadata
        chunks = self.pdf_processor.chunk_text(pages_data, structure)
        
        # Store in vector DB
        self.vector_db.add_documents(chunks, namespace)
        
        # Generate summary statistics
        total_text = " ".join([page["text"] for page in pages_data])
        chapter_info = [
            {
                "number": chapter.get("number", "N/A"),
                "title": chapter.get("title", "Untitled"),
                "pages": f"{chapter.get('start_page', 'N/A')}-{chapter.get('end_page', 'N/A')}"
            }
            for chapter in structure["chapters"]
        ]
        
        return {
            "namespace": namespace,
            "chunks_count": len(chunks),
            "total_chars": len(total_text),
            "chapter_count": len(structure["chapters"]),
            "chapter_info": chapter_info[:5] if chapter_info else []  # Show first 5 chapters
        }
    
    def rewrite_query(self, user_query: str) -> str:
        """Use LLM to rewrite difficult queries for better RAG retrieval"""
        
        # Only rewrite if the query is potentially problematic
        # if len(user_query.split()) > 5 and not any(term in user_query.lower() for term in ['u ', 'ur', 'tea ', 'china']):
        #     return user_query
        
        # Use LLM to rewrite the query in a way that will better match your knowledge base
        system_prompt = """
        You are a query optimization assistant. Your task is to rewrite user queries to make them more 
        effective for retrieval from a knowledge base about Henry George's economic theories.
        
        For short queries or casual language, expand them into more detailed questions.
        For specific examples, broaden them to include relevant economic principles.
        """
        
        user_prompt = f"""
        Original query: {user_query}
        
        Rewrite this query to be more effective for retrieving information from a knowledge base
        about Henry George's economic theories, land value taxation, and economic principles.
        
        If this is a greeting or casual message, transform it into a request for information
        about Henry George's core ideas.
        
        Identify if it is greeting and then transform it into a request for information about Henry George's way of thinking.

        Return only the rewritten query without explanation.
        """
        
        try:
            rewritten_query = completion(
                model="gemini/gemini-2.0-flash",  # Using your existing model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1  # Low temperature for consistency
            )
            
            # Extract the response content
            rewritten_query_text = rewritten_query.choices[0].message.content.strip()
            
            # Log the transformation for debugging
            print(f"Original query: '{user_query}' → Rewritten: '{rewritten_query_text}'")
            
            return rewritten_query_text
        except Exception as e:
            print(f"Query rewriting failed: {str(e)}")
            return user_query  # Fall back to original query if rewriting fails

    def _classify_query(self, query: str) -> str:
        """Classify the query type"""
        # Default to book question
        return "book_question"
    
    def query(self, namespaces: List[str], user_query: str) -> Tuple[str, List[Dict], Dict]:
        """Query with improved handling for difficult queries"""
        
        # Step 1: Try to improve the query if needed
        original_query = user_query
        rewritten_query = self.rewrite_query(user_query)
        query_used = original_query  # Track which query was successful
        
        # Step 2: First try with original query
        all_search_results = []
        for namespace in namespaces:
            namespace_results = self.vector_db.hybrid_search(
                query=original_query,
                namespace=namespace,
                n_results=max(1, SEARCH_CONFIG["default_results"] // len(namespaces))
            )
            
            # Add source namespace to each result
            for result in namespace_results:
                result["namespace"] = namespace
                result["book_name"] = namespace.replace('book_', '').replace('_', ' ').title()
            
            all_search_results.extend(namespace_results)
        
        # Sort results by score
        all_search_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Step 3: If results are poor, try with rewritten query
        if not all_search_results or (all_search_results and max(r["score"] for r in all_search_results) < 0.6):
            print(f"Results for original query insufficient, trying rewritten query: {rewritten_query}")
            
            second_search_results = []
            for namespace in namespaces:
                namespace_results = self.vector_db.hybrid_search(
                    query=rewritten_query,
                    namespace=namespace,
                    n_results=max(1, SEARCH_CONFIG["default_results"] // len(namespaces))
                )
                
                # Add source namespace
                for result in namespace_results:
                    result["namespace"] = namespace
                    result["book_name"] = namespace.replace('book_', '').replace('_', ' ').title()
                
                second_search_results.extend(namespace_results)
            
            # Sort results by score
            second_search_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Use rewritten query results if they're better
            if (second_search_results and 
                (not all_search_results or 
                (second_search_results and all_search_results and 
                max(r["score"] for r in second_search_results) > max(r["score"] for r in all_search_results)))):
                all_search_results = second_search_results
                query_used = rewritten_query
        
        # Limit to top results
        max_total_results = max(5, min(len(namespaces) * 2, 10))
        search_results = all_search_results[:max_total_results]
        
        # If no results found, handle appropriately
        if not search_results:
            no_info_response = "I couldn't find relevant information to answer your question. Could you try rephrasing or asking something else?"
            
            # Create a basic structured response for "no results" case
            structured_response = {
                "answer": no_info_response,
                "citations": [],
                "expert_reference": {
                    "name": "Edward Dodson",
                    "email": "info@hgsss.org",
                    "organization": "Henry George School of Social Science"
                },
                "additional_resources": [
                    {
                        "type": "course",
                        "description": "Explore our courses on Georgist economics",
                        "url": "https://www.hgsss.org/courses/"
                    }
                ],
                "follow_up_questions": [
                    "What is the single tax?",
                    "What is your view on land ownership?",
                    "How would land value taxation eliminate poverty?",
                    "What did you write in Progress and Poverty?",
                    "How does land speculation cause economic depressions?"
                ]
            }
            
            self.conversation_history.append({
                "query": user_query,
                "response": no_info_response,
                "type": "no_results",
                "structured_response": structured_response
            })
            
            return no_info_response, [], structured_response
        
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results):
            # Build context entry
            book_info = result.get("book_info", {})
            if not book_info:
                # Create book_info if not already present
                book_info = {
                    "title": result["book_name"],
                    "chapter": result["metadata"].get("chapter_number", ""),
                    "chapter_title": result["metadata"].get("chapter_title", ""),
                    "pages": f"{result['metadata'].get('start_page', 'N/A')}-{result['metadata'].get('end_page', 'N/A')}"
                }
                result["book_info"] = book_info
                
            context_entry = f"Document {i+1} (from {book_info['title']}):"
            metadata = result["metadata"]
            
            # Add chapter information if available
            if "chapter_title" in metadata:
                context_entry += f" Chapter: {metadata.get('chapter_number', '')}: {metadata.get('chapter_title', '')}"
                
            # Add page information
            context_entry += f" (Pages {book_info['pages']})"
            
            # Add the document text
            context_entry += f"\n{result['text']}"
            
            context_parts.append(context_entry)
            
        context = "\n\n".join(context_parts)
    
        
        # Create prompt for LLM
        prompt = f"""
    Persona:
    You are Henry George (1839-1897), the American political economist, journalist, social reformer, and influential orator. You are speaking with the passion, clarity, and moral conviction characteristic of your major works like Progress and Poverty and Protection or Free Trade, as well as your public speeches. You engage energetically with the great social and economic questions of your time, aiming to diagnose the root causes of injustice and persuade others of the necessary remedies.
    Core Beliefs & Knowledge (Your Unwavering Ideology):
    Central Thesis (Georgism): Your core belief is that while individuals should own the value they create through their labor and capital, the economic value derived from land (including all natural resources and opportunities) rightly belongs to the community as a whole, as its value arises from the presence and development of society, not individual effort.
    The Land Question is Paramount: You see the private monopolization of land and the private capture of its economic rent as the fundamental cause of the paradox of "progress and poverty" – the persistence and worsening of poverty and inequality alongside technological and economic advancement. This system denies labor access to natural opportunities and is equivalent to a form of slavery.
    Land Value Taxation (The "Single Tax"): You advocate vigorously for abolishing taxes on productive activities (wages, capital, improvements, trade) and deriving public revenue primarily, or solely, by taxing the value of land itself. This is not a tax on land area, but on its market value conferred by the community.
    Benefits: This tax cannot be shifted to tenants or consumers, encourages the best use of land, discourages harmful speculation, simplifies government, raises wages, increases opportunities, and captures community-created value for public benefit.
    Distinction: Clarify that this is not land nationalization or confiscation of land titles. Individuals can retain possession, buy, sell, and bequeath land; society simply collects the annual value (rent) they did not create. ("We may safely leave them the shell, if we take the kernel.")
    Free Trade: You are an ardent opponent of protectionist tariffs. You argue they artificially raise prices for consumers, harm overall economic activity, protect inefficient monopolies, fail to raise general wages, and corrupt politics. You see free trade as aligned with natural economic laws and international cooperation.
    Natural Monopolies: You believe services that inherently require exclusive rights-of-way or control over essential networks (e.g., railroads, utilities like water/electricity, telegraphs, potentially urban transit) are "natural monopolies." These should ideally be under public ownership or strict regulation (municipalization) and often provided at cost or even free, funded by the land value increases they generate.
    Money and Finance: You support "debt-free" sovereign money (like greenbacks) issued by the government, capturing seigniorage for public benefit. You are critical of metallic currency limitations and fiat money created by private banks, viewing much credit/debt as tied to land speculation rather than productive enterprise. You advocate for bankruptcy protections and oppose debtors' prisons, seeing much debt as illegitimate claims on rent.
    Political & Social Reforms: You are a strong advocate for:
    Secret Ballot: (Crucial for preventing bribery and ensuring voter freedom).
    Women's Suffrage: (A matter of fundamental political rights).
    Citizen's Dividend/Universal Pension: Surplus land rent revenues could fund basic incomes or pensions distributed "as a right," not charity.
    Intellectual Property Skepticism: View patents and copyrights cautiously, as forms of monopoly potentially extracting unearned rent, though perhaps less harmful than land monopoly.
    Reduced Military Spending & Civil Service Reform.
    Tone and Style:
    Passionate and Earnest: Speak with profound conviction about justice, rights, and the moral imperative of your reforms.
    Reasoned and Didactic: Explain complex economic ideas with clarity, logic, and accessible language. Use analogies and examples (both historical and hypothetical) effectively.
    Moralistic and Righteous: Frame arguments in terms of natural law, the Creator's intent (using a deistic/humanitarian lens), justice vs. injustice, equality, and the common good.
    Confident and Assertive: Directly address and dismantle opposing arguments (like those for protectionism or the current tax system) with robust logic.
    Populist: Connect with the concerns of ordinary people, laborers, and producers, contrasting their struggles with the unearned income of monopolists.
    Engaging and Oratorical: Employ rhetorical questions, address the user respectfully (e.g., "friend," "fellow-citizen"), and maintain a somewhat formal but powerful speaking style appropriate to the late 19th century.
    Historically Grounded: Speak from the perspective of your era, referencing contemporary events, figures (like Hewitt, Powderly, Marx – often critically), and intellectual currents. Avoid anachronisms.
    Goal:
    Your purpose is to educate, inspire, and persuade the user of the fundamental truth and justice of your core ideas – particularly the necessity of Land Value Taxation as the foundation for a just and prosperous society. 
    You aim to show how this central reform connects to and supports other vital reforms 
    (like free trade and public control of monopolies), ultimately leading to the abolition of involuntary poverty, the fair distribution of wealth, and the realization of true liberty and equality for all members of the community.
      You seek to clarify your positions and counter misrepresentations.
    Give Short responses and to the point(around 100 words). Unless asked for a detailed response.
    You should use the 19th century American English for your responses.
    

    Important: You MUST directly cite relevant sections from my books where appropriate, using the format (Book Title, Book X, Ch. Y).
    Double check before giving the citations

            Context from the books:
            {context}
            
            Original Question: {user_query}
            {f"Rewritten for better retrieval as: {rewritten_query}" if query_used != user_query else ""}

            
            Answer:
            """
        
        # Generate response using Groq
        response = completion(
            model=LLM_MODEL,  
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides well-cited answers about books."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        response_text = response.choices[0].message.content
        
        # Create citation data for displaying sources
        citations = []
        for i, result in enumerate(search_results):
            metadata = result["metadata"]
            book_info = result["book_info"]
            citation = {
                "number": i + 1,
                "id": result["id"],
                "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "book_name": book_info["title"],
                "namespace": result["namespace"],
                "metadata": {}
            }
            
            # Add structural information to citations
            if "chapter_title" in metadata:
                citation["metadata"]["chapter"] = f"{book_info['chapter']}: {book_info['chapter_title']}"
                
            citation["metadata"]["pages"] = book_info["pages"]
            
            # Add relevance score
            similarity_score = round(result["score"] * 100, 1)
            citation["metadata"]["relevance"] = f"{similarity_score}%"
            
            citations.append(citation)
        
        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions(
            last_user_query=user_query,
            last_response=response_text
        )
        
        # Create expert reference information
        expert_reference = {
            "name": "Edward Dodson",
            "email": "info@hgsss.org",
            "organization": "Henry George School of Social Science"
        }
        
        # Format additional resources
        additional_resources = [
            {
                "type": "course",
                "description": "Explore deeper with courses on Georgist economics",
                "url": "https://www.hgsss.org/courses/"
            },
            {
                "type": "video",
                "description": "Watch videos of relevant lectures on Georgist principles",
                "url": "https://www.hgsss.org/videos/"
            }
        ]
        
        # Create the structured response
        structured_response = {
            "answer": response_text,
            "citations": citations,
            "expert_reference": expert_reference,
            "additional_resources": additional_resources,
            "follow_up_questions": follow_up_questions
        }
        
        # Save to conversation history
        self.conversation_history.append({
            "query": user_query,
            "response": response_text,
            #"type": query_type,
            "citations": [c["id"] for c in citations],
            "structured_response": structured_response
        })
        
        return response_text, citations, structured_response

    def _generate_follow_up_questions(self, last_user_query: str, last_response: str) -> List[str]:
        """
        Ask the LLM to propose up to 5 follow‑up questions
        based on the user's previous query and the assistant's response.
        """
        system_prompt = (
            "You are a helpful assistant that crafts relevant next questions "
            "a user might ask, given the conversation so far."
        )
        user_prompt = (
            f"User asked: \"{last_user_query}\"\n"
            f"Assistant answered: \"{last_response}\"\n\n"
            "Based on this, suggest up to 5 concise, on‑topic follow‑up questions "
            "the user could ask next. Return each question on its own line."
        )

        # Call the LLM
        result = completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )

        # Split lines and clean up
        raw = result.choices[0].message.content.strip()
        questions = [
            q.strip(" -–·") 
            for q in raw.splitlines() 
            if q.strip()
        ]
        return questions[:5]

    def _classify_query(self, query: str) -> str:
        """Classify the query type for better response formatting"""
        if re.search(r"chapter|book|section", query.lower()):
            return "book_reference"
        elif re.search(r"land value tax|single tax|georgism|economic rent", query.lower()):
            return "core_concept"
        elif re.search(r"what|how|why|explain|describe", query.lower()):
            return "explanatory"
        else:
            return "general"

# Initialize the chatbot
chatbot = RAGChatbot()

# 1) Callback to handle any follow‑up (either from Related Questions or chat input)
def _handle_new_question(question: str):
    # Add the user question immediately
    st.session_state.messages.append({"role": "user", "content": question})
    # Mark it pending so the next rerun will generate a bot reply
    st.session_state.pending_question = question
    st.rerun()

# 2) Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "all_namespaces" not in st.session_state:
    st.session_state.all_namespaces = []

# ——— Sidebar ———
st.title("📚 PastPort bot")
with st.sidebar:
    st.header("Available Documents")
    available_ns = chatbot.vector_db.list_namespaces()

    if not available_ns:
        st.info("No documents found. Upload a PDF below.")
    else:
        display_names = [
            ns.replace("book_", "").replace("_", " ").title()
            for ns in available_ns
        ]
        st.write(f"**{len(available_ns)} documents available:**")
        for name in display_names:
            st.write(f"- {name}")
        st.session_state.all_namespaces = available_ns

    st.divider()
    st.header("Upload Document (Optional)")
    uploaded = st.file_uploader("Upload a PDF book", type="pdf")
    if uploaded:
        # sanitize namespace
        default = re.sub(r"[^A-Za-z0-9._-]", "_", uploaded.name[:-4])
        if not default[0].isalnum(): default = "b_" + default
        if len(default) > 63: default = default[:60] + "_db"
        namespace = f"book_{default}"
        if st.button("Process Document"):
            with st.spinner("Processing your document..."):
                res = chatbot.process_pdf(uploaded, namespace)
            st.success("Done!")
            st.write(f"Found {res['chapter_count']} chapters")
            st.write(f"Created {res['chunks_count']} text chunks")
            st.rerun()

    if available_ns:
        st.divider()
        with st.expander("Document Management"):
            to_delete = st.selectbox("Select document to delete:", display_names)
            if to_delete and st.button("Delete Selected Document"):
                ns = available_ns[display_names.index(to_delete)]
                try:
                    chatbot.vector_db.index.delete(namespace=ns, delete_all=True)
                    st.success(f"Deleted '{to_delete}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.write("Made with Pinecone, Gemini")

# ——— Chat Area ———
st.subheader("Chat with Henry George")

# 3) Render existing chat history (with citations, expert refs, resources & Related Questions)
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "structured_response" in msg:
            structured = msg["structured_response"]

            # # — View Sources —
            # if "citations" in structured:
            #     with st.expander("View Sources"):
            #         for c in structured["citations"]:
            #             st.markdown(f"**Source {c['number']} – {c['book_name']}**")
            #             if "metadata" in c:
            #                 md = c["metadata"]
            #                 cols = st.columns(2)
            #                 if "chapter" in md:
            #                     cols[0].write(f"Chapter: {md['chapter']}")
            #                 if "pages" in md:
            #                     cols[1].write(f"Pages: {md['pages']}")
            #             st.text_area("Excerpt", c["text"], height=100)
            #             st.divider()

            # — Expert Reference —
            if "expert_reference" in structured:
                with st.expander("Expert Reference"):
                    er = structured["expert_reference"]
                    st.markdown(f"**Name:** {er['name']}")
                    st.markdown(f"**Organization:** {er['organization']}")
                    st.markdown(f"**Contact:** {er['email']}")

            # — Additional Resources —
            if "additional_resources" in structured:
                with st.expander("Additional Resources"):
                    for r in structured["additional_resources"]:
                        st.markdown(f"**{r['type'].title()}:** {r['description']}")
                        st.markdown(f"[Learn more]({r['url']})")

            # — Related Questions —
            if "follow_up_questions" in structured:
                with st.expander("Related Questions"):
                    for q_idx, question in enumerate(structured["follow_up_questions"]):
                        key = f"fu_{msg_idx}_{q_idx}"
                        st.button(
                            question,
                            key=key,
                            on_click=_handle_new_question,
                            args=(question,)
                        )

# 4) Process any pending question (from either chat_input or Related Questions)
if st.session_state.pending_question:
    user_q = st.session_state.pending_question
    st.session_state.pending_question = None

    # # Show the user message
    # with st.chat_message("user"):
    #     st.markdown(user_q)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching books and generating response..."):
            resp_text, citations, structured = chatbot.query(
                st.session_state.all_namespaces, user_q
            )
        st.markdown(resp_text)

        # Re‑use the same expanders as above for immediate display:

        # if citations:
        #     with st.expander("View Sources"):
        #         for c in citations:
        #             st.markdown(f"**Source {c['number']} – {c['book_name']}**")
        #             if "metadata" in c:
        #                 md = c["metadata"]
        #                 cols = st.columns(2)
        #                 if "chapter" in md:
        #                     cols[0].write(f"Chapter: {md['chapter']}")
        #                 if "pages" in md:
        #                     cols[1].write(f"Pages: {md['pages']}")
        #             st.text_area("Excerpt", c["text"], height=100)
        #             st.divider()

        if "expert_reference" in structured:
            with st.expander("Expert Reference"):
                er = structured["expert_reference"]
                st.markdown(f"**Name:** {er['name']}")
                st.markdown(f"**Organization:** {er['organization']}")
                st.markdown(f"**Contact:** {er['email']}")

        if "additional_resources" in structured:
            with st.expander("Additional Resources"):
                for r in structured["additional_resources"]:
                    st.markdown(f"**{r['type'].title()}:** {r['description']}")
                    st.markdown(f"[Learn more]({r['url']})")

        if "follow_up_questions" in structured:
            with st.expander("Related Questions"):
                for q_idx, question in enumerate(structured["follow_up_questions"]):
                    key = f"pending_fu_{q_idx}"
                    st.button(
                        question,
                        key=key,
                        on_click=_handle_new_question,
                        args=(question,)
                    )

    # Save into history
    st.session_state.messages.append({
        "role": "assistant",
        "content": resp_text,
        "citations": citations,
        "structured_response": structured
    })

# 5) Finally: chat_input to ask a fresh question
if st.session_state.all_namespaces:
    new_q = st.chat_input("Ask a question about any of the books")
    if new_q:
        _handle_new_question(new_q)
else:
    st.info("No books found. Please upload one in the sidebar to get started.")