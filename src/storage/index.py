"""Tantivy-based full-text search index manager with Chinese tokenizer support."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import shutil

logger = logging.getLogger(__name__)

import tantivy
from tantivy import Document, SnippetGenerator, TextAnalyzerBuilder, Tokenizer, Filter

from src.storage.base import IndexManager, SearchHit, SearchResult


SCHEMA_VERSION = "2"


class TantivyIndexManager(IndexManager):
    """
    Tantivy-based search index manager with BM25 ranking and Chinese tokenizer support.

    This implementation provides:
    - Full-text search with BM25 ranking
    - Chinese text tokenization (via regex or jieba)
    - Incremental document updates
    - Document deletion and updates
    """

    # Field names in the schema
    FIELD_DOC_ID = "doc_id"
    FIELD_FILENAME = "filename"
    FIELD_TITLE = "title"
    FIELD_CONTENT = "content"
    FIELD_KEYWORDS = "keywords"
    FIELD_SOURCE_PATH = "source_path"
    FIELD_MODIFIED_TIME = "modified_time"

    def __init__(
        self,
        index_path: Optional[Path] = None,
        use_jieba: bool = True,
        heap_size: int = 50_000_000,
        num_threads: int = 2,
        readonly: bool = False,
    ):
        """
        Initialize the Tantivy index manager.

        Args:
            index_path: Path to store the index. If None, creates in-memory index.
            use_jieba: Whether to use jieba for Chinese tokenization.
            heap_size: Writer heap size in bytes.
            num_threads: Number of writer threads.
            readonly: If True, skip IndexWriter creation (search-only mode).
        """
        self._index_path = index_path
        self._use_jieba = use_jieba
        self._heap_size = heap_size
        self._num_threads = num_threads
        self._readonly = readonly
        self._writer: Optional[tantivy.IndexWriter] = None
        self._jieba_available = False
        self._is_new_index = False

        # Check for jieba availability
        if use_jieba:
            try:
                import jieba

                self._jieba_available = True
            except ImportError:
                self._jieba_available = False

        # Build schema and create index
        self._schema = self._build_schema()
        self._index = self._create_index()

        # Register custom tokenizer for Chinese (always needed for schema)
        self._register_chinese_tokenizer()

        # Initialize writer (skip in readonly mode to avoid LockBusy)
        if not self._readonly:
            self._writer = self._index.writer(
                heap_size=self._heap_size, num_threads=self._num_threads
            )

    def _build_schema(self) -> tantivy.Schema:
        """Build the Tantivy schema for document indexing."""
        builder = tantivy.SchemaBuilder()

        # Document ID - unique identifier (raw tokenizer for exact matching)
        builder.add_text_field(
            self.FIELD_DOC_ID,
            stored=True,
            tokenizer_name="raw",
        )

        # Filename - document filename
        builder.add_text_field(
            self.FIELD_FILENAME,
            stored=True,
        )

        # Title - document title (searchable)
        builder.add_text_field(
            self.FIELD_TITLE,
            stored=True,
        )

        # Content - full markdown content (searchable)
        builder.add_text_field(
            self.FIELD_CONTENT,
            stored=True,
        )

        # Keywords - keywords list (searchable)
        builder.add_text_field(
            self.FIELD_KEYWORDS,
            stored=True,
        )

        # Source path - original file path
        builder.add_text_field(
            self.FIELD_SOURCE_PATH,
            stored=True,
        )

        # Modified time - file modification time
        builder.add_date_field(
            self.FIELD_MODIFIED_TIME,
            stored=True,
            fast=True,
        )

        return builder.build()

    def _create_index(self) -> tantivy.Index:
        """Create or open the Tantivy index."""
        if self._index_path is not None:
            index_dir = Path(self._index_path)
            index_dir.mkdir(parents=True, exist_ok=True)

            # Check if index already exists
            if tantivy.Index.exists(str(index_dir)):
                # Check schema version — rebuild if mismatch
                version_file = index_dir / "SCHEMA_VERSION"
                if version_file.exists():
                    stored_version = version_file.read_text(encoding="utf-8").strip()
                else:
                    stored_version = ""

                if stored_version != SCHEMA_VERSION:
                    logger.info(
                        "Schema version mismatch (stored=%r, "
                        "current=%r). Rebuilding index.",
                        stored_version,
                        SCHEMA_VERSION,
                    )
                    if index_dir.exists():
                        shutil.rmtree(index_dir)
                    index_dir.mkdir(parents=True, exist_ok=True)
                    index = tantivy.Index(self._schema, path=str(index_dir))
                    self._is_new_index = True
                else:
                    index = tantivy.Index.open(str(index_dir))
                    self._is_new_index = False
            else:
                # Create new index
                index = tantivy.Index(self._schema, path=str(index_dir))
                self._is_new_index = True
        else:
            # In-memory index is always new
            index = tantivy.Index(self._schema)
            self._is_new_index = True

        # Auto-reload reader when a commit happens
        index.config_reader(reload_policy="OnCommit")

        # Persist schema version for disk-based indices
        if self._index_path is not None:
            version_file = Path(self._index_path) / "SCHEMA_VERSION"
            version_file.write_text(SCHEMA_VERSION, encoding="utf-8")

        return index

    def _register_chinese_tokenizer(self):
        """Chinese tokenization is handled by jieba pre-processing in _tokenize_chinese."""
        # No custom tokenizer registration needed - jieba pre-tokenization
        # adds spaces between Chinese words, and the default tokenizer splits on whitespace.
        pass

    def _tokenize_chinese(self, text: str) -> str:
        """
        Pre-tokenize Chinese text before indexing.

        If jieba is available, segments the text and joins with spaces.
        Otherwise, returns the original text.

        Skips re-tokenization if the text appears already segmented
        (CJK characters separated by spaces — output of a prior jieba cut).

        Args:
            text: Text to tokenize

        Returns:
            Tokenized text with spaces between segments
        """
        if not text:
            return text

        # Skip re-tokenization if already segmented:
        # CJK characters separated by spaces = output of a prior jieba cut
        if self._is_already_segmented(text):
            return text

        if self._jieba_available:
            import jieba

            tokens = list(jieba.cut(text))
            return " ".join(t for t in tokens if t.strip())
        else:
            return text

    @staticmethod
    def _is_already_segmented(text: str) -> bool:
        """Check if text appears to be already jieba-segmented.

        Heuristic: if ALL CJK characters are separated by spaces
        (i.e., no two adjacent CJK chars without a space between them),
        the text was likely already segmented.
        """
        import unicodedata

        prev_was_cjk = False
        for ch in text:
            cat = unicodedata.category(ch)
            is_cjk = (
                cat == "Lo"  # Letter, Other (CJK ideographs, hiragana, katakana)
                or "\u4e00" <= ch <= "\u9fff"
                or "\u3400" <= ch <= "\u4dbf"
            )
            if is_cjk and prev_was_cjk:
                return False  # Two adjacent CJK chars = not segmented
            prev_was_cjk = is_cjk
        # If we get here, no two adjacent CJK chars found
        return any(
            "\u4e00" <= ch <= "\u9fff" or unicodedata.category(ch) == "Lo"
            for ch in text
        )

    def _to_or_query(self, tokenized_query: str) -> str:
        """Convert a space-separated AND-mode query to OR-mode for Tantivy.

        When AND-mode returns zero hits (e.g., rare term + common term),
        OR-mode ensures at least some results are returned.

        Args:
            tokenized_query: Space-separated tokens (AND-mode by default).

        Returns:
            Query string with OR operators between tokens.
        """
        tokens = [t.strip() for t in tokenized_query.split() if t.strip()]
        if len(tokens) <= 1:
            return tokenized_query
        return " OR ".join(tokens)

    def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Add a document to the search index.

        Args:
            doc_id: Document ID
            title: Document title
            content: Document text content
            metadata: Document metadata (may contain filename, source_path,
                     modified_time, keywords)

        Returns:
            True if successful, False otherwise
        """
        try:
            doc = Document()

            # Add core fields
            doc.add_text(self.FIELD_DOC_ID, doc_id)
            doc.add_text(self.FIELD_TITLE, self._tokenize_chinese(title))
            doc.add_text(self.FIELD_CONTENT, self._tokenize_chinese(content))

            # Add optional metadata fields
            filename = metadata.get("filename", "")
            if filename:
                doc.add_text(self.FIELD_FILENAME, self._tokenize_chinese(filename))

            source_path = metadata.get("source_path", "")
            if source_path:
                doc.add_text(self.FIELD_SOURCE_PATH, str(source_path))

            # Handle keywords (list or string)
            keywords = metadata.get("keywords", [])
            if keywords:
                if isinstance(keywords, list):
                    keywords_str = " ".join(str(k) for k in keywords)
                else:
                    keywords_str = str(keywords)
                doc.add_text(self.FIELD_KEYWORDS, self._tokenize_chinese(keywords_str))

            # Handle modified time
            modified_time = metadata.get("modified_time")
            if modified_time:
                if isinstance(modified_time, datetime):
                    doc.add_date(self.FIELD_MODIFIED_TIME, modified_time)
                elif isinstance(modified_time, str):
                    # Parse ISO format datetime string
                    dt = datetime.fromisoformat(modified_time.replace("Z", "+00:00"))
                    doc.add_date(self.FIELD_MODIFIED_TIME, dt)

            self._writer.add_document(doc)
            return True

        except Exception as e:
            logger.error("Error adding document %s: %s", doc_id, e)
            return False

    def update_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Update a document in the search index.

        This implementation uses delete + add for incremental updates.

        Args:
            doc_id: Document ID
            title: Document title
            content: Document text content
            metadata: Document metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing document first
            self.delete_document(doc_id)

            # Add updated document
            return self.add_document(doc_id, title, content, metadata)

        except Exception as e:
            logger.error("Error updating document %s: %s", doc_id, e)
            return False

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the search index.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete documents by field name and value directly
            self._writer.delete_documents(self.FIELD_DOC_ID, doc_id)
            return True

        except Exception as e:
            logger.error("Error deleting document %s: %s", doc_id, e)
            return False

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict] = None,
        title_boost: float = 1.0,
    ) -> SearchResult:
        """
        Search the index for documents using BM25 ranking.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            offset: Number of results to skip
            filters: Optional filter criteria (not fully implemented)
            title_boost: Boost factor for title field matches (1.0 = no boost).
                Values > 1.0 give extra weight to documents whose title
                matches the query terms.

        Returns:
            SearchResult object with matching hits
        """
        start_time = time.time()

        try:
            # OnCommit reload policy handles reader refresh automatically
            # Get searcher
            searcher = self._index.searcher()

            # Tokenize query for better Chinese matching
            tokenized_query = self._tokenize_chinese(query)

            # Apply title field boosting if requested
            if title_boost > 1.0 and tokenized_query.strip():
                boosted_query = self._apply_title_boost(
                    tokenized_query, title_boost
                )
            else:
                boosted_query = tokenized_query

            # Parse query for searchable fields
            searchable_fields = [
                self.FIELD_TITLE,
                self.FIELD_CONTENT,
                self.FIELD_KEYWORDS,
                self.FIELD_FILENAME,
            ]

            try:
                parsed_query = self._index.parse_query(boosted_query, searchable_fields)
            except Exception:
                # Fallback: if boosted query fails to parse, use original
                parsed_query = self._index.parse_query(tokenized_query, searchable_fields)

            # Build snippet generator for proper highlighting
            snippet_gen = None
            try:
                snippet_gen = SnippetGenerator.create(
                    searcher, parsed_query, self._schema, self.FIELD_CONTENT
                )
                snippet_gen.set_max_num_chars(200)
            except Exception:
                snippet_gen = None

            # Execute search with enough results to handle offset
            search_limit = limit + offset
            results = searcher.search(parsed_query, search_limit)

            # AND→OR fallback: if AND-mode returns zero hits, retry with OR-mode
            # (EverOS: jieba + Tantivy AND-mode can cause "IDF poison" zero-hit queries)
            if len(results.hits) == 0 and tokenized_query.strip():
                or_query = self._to_or_query(tokenized_query)
                if or_query != boosted_query:
                    logger.debug("AND-mode zero hits, retrying with OR-mode: %s", or_query)
                    try:
                        or_parsed = self._index.parse_query(or_query, searchable_fields)
                        results = searcher.search(or_parsed, search_limit)
                        # Rebuild snippet generator for OR query
                        try:
                            snippet_gen = SnippetGenerator.create(
                                searcher, or_parsed, self._schema, self.FIELD_CONTENT
                            )
                            snippet_gen.set_max_num_chars(200)
                        except Exception:
                            snippet_gen = None
                    except Exception:
                        pass  # OR-mode also failed, return empty

            # Process results
            hits = []
            total = len(results.hits)

            for i, (score, doc_address) in enumerate(results.hits):
                # Skip results before offset
                if i < offset:
                    continue

                if len(hits) >= limit:
                    break

                doc = searcher.doc(doc_address)

                # Extract document fields using get_first
                doc_id = doc.get_first(self.FIELD_DOC_ID) or ""
                title = doc.get_first(self.FIELD_TITLE) or ""
                content = doc.get_first(self.FIELD_CONTENT) or ""
                source_path = doc.get_first(self.FIELD_SOURCE_PATH)

                # Create excerpt using SnippetGenerator (fallback to manual)
                # Also reuse the snippet for highlight extraction (avoid double call)
                snippet = None
                if snippet_gen is not None:
                    try:
                        snippet = snippet_gen.snippet_from_doc(doc)
                        excerpt = snippet.fragment()
                        if not excerpt:
                            excerpt = self._create_excerpt_fallback(content, query)
                    except Exception:
                        excerpt = self._create_excerpt_fallback(content, query)
                else:
                    excerpt = self._create_excerpt_fallback(content, query)

                # Extract highlight terms from snippet or query
                highlight_terms = self._extract_highlights(
                    snippet_gen, doc, query, content,
                    precomputed_snippet=snippet,
                )

                # Create search hit
                hit = SearchHit(
                    doc_id=doc_id,
                    title=title,
                    score=score,
                    excerpt=excerpt,
                    highlights=highlight_terms,
                    source_path=Path(source_path) if source_path else None,
                )
                hits.append(hit)

            execution_time = time.time() - start_time

            return SearchResult(
                hits=hits,
                total=total,
                query=query,
                execution_time=execution_time,
                offset=offset,
                limit=limit,
            )

        except Exception as e:
            logger.error("Error searching for '%s': %s", query, e)
            # Return empty result on error
            return SearchResult(
                hits=[],
                total=0,
                query=query,
                execution_time=time.time() - start_time,
                offset=offset,
                limit=limit,
            )

    def _extract_highlights(
        self,
        snippet_gen: Optional[SnippetGenerator],
        doc: Document,
        query: str,
        content: str,
        max_terms: int = 5,
        precomputed_snippet: Any = None,
    ) -> List[str]:
        """Extract highlight terms from snippet generator or query.

        Tries to extract highlighted terms from the Tantivy Snippet object
        (which wraps matches in <b> tags). Falls back to extracting query
        terms that appear in the document content.

        Args:
            snippet_gen: Tantivy SnippetGenerator (may be None).
            doc: Tantivy Document object.
            query: Original search query string.
            content: Document content text.
            max_terms: Maximum number of highlight terms to return.
            precomputed_snippet: Reuse a snippet already computed for excerpt,
                avoiding a second snippet_from_doc() call.

        Returns:
            List of highlighted term strings.
        """
        # Try extracting from Tantivy snippet <b> tags first
        if snippet_gen is not None:
            try:
                snippet = precomputed_snippet or snippet_gen.snippet_from_doc(doc)
                fragment = snippet.fragment()
                import re

                highlighted = re.findall(r"<b>(.*?)</b>", fragment)
                if highlighted:
                    # Deduplicate while preserving order
                    seen = set()
                    result = []
                    for h in highlighted:
                        h_clean = h.strip()
                        if h_clean and h_clean not in seen:
                            seen.add(h_clean)
                            result.append(h_clean)
                    if result:
                        return result[:max_terms]
            except Exception:
                pass

        # Fallback: extract highlight terms from query
        return self._extract_highlight_terms(query, content, max_terms)

    def _extract_highlight_terms(
        self, query: str, content: str, max_terms: int = 5
    ) -> List[str]:
        """Extract highlight terms from query that appear in content.

        Args:
            query: Search query string.
            content: Document content to search in.
            max_terms: Maximum number of highlight terms.

        Returns:
            List of terms from the query that appear in the content.
        """
        if not query or not content:
            return []

        # Parse query terms using QueryParser
        from src.search.query_parser import QueryParser

        parser = QueryParser()
        parsed = parser.parse(query)

        highlights: List[str] = []
        content_lower = content.lower()

        # Check phrases first (higher priority)
        for phrase in parsed.phrases:
            if phrase.lower() in content_lower:
                highlights.append(phrase)

        # Then individual terms
        for term in parsed.terms:
            if term.lower() in content_lower:
                highlights.append(term)

        return highlights[:max_terms]

    def _apply_title_boost(self, query: str, title_boost: float) -> str:
        """Add title field boosting to improve title-match relevance.

        Constructs a compound Tantivy query that searches all fields
        normally *and* adds a boosted title-specific clause so documents
        whose title matches the query terms receive a higher BM25 score.

        Args:
            query: Tokenized query string.
            title_boost: Boost factor (> 1.0).

        Returns:
            Boosted query string for ``parse_query``.
        """
        # No-op when boost is disabled or query is empty
        if title_boost <= 1.0 or not query.strip():
            return query
        return f"({query}) (title:({query})^{title_boost:.1f})"

    def _create_excerpt_fallback(
        self,
        content: str,
        query: str,
        max_length: int = 200,
        context_chars: int = 50,
    ) -> str:
        """
        Create an excerpt from content highlighting the query match.

        Fallback method used when SnippetGenerator is unavailable.

        Args:
            content: Full document content
            query: Search query
            max_length: Maximum excerpt length
            context_chars: Characters to include around match

        Returns:
            Excerpt string
        """
        if not content:
            return ""

        # Find first query term in content (case-insensitive)
        query_lower = query.lower()
        content_lower = content.lower()

        pos = content_lower.find(
            query_lower.split()[0] if query_lower.split() else query_lower
        )

        if pos >= 0:
            # Calculate excerpt boundaries
            start = max(0, pos - context_chars)
            end = min(len(content), pos + len(query) + context_chars)

            excerpt = content[start:end]

            # Add ellipsis if truncated
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(content):
                excerpt = excerpt + "..."

            return excerpt[: max_length + 3]  # +3 for ellipsis
        else:
            # Return beginning of content if no match found
            if len(content) > max_length:
                return content[:max_length] + "..."
            return content

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its exact doc_id, returning full content.

        Unlike search(), this method does NOT truncate content via excerpts.
        Uses the 'raw' tokenizer on doc_id for exact matching.

        Args:
            doc_id: Document ID to look up

        Returns:
            Dict with keys: doc_id, title, content, filename, source_path, keywords.
            Returns None if not found.
        """
        try:
            # OnCommit reload policy handles reader refresh — no explicit reload needed
            searcher = self._index.searcher()

            # Exact match on doc_id field (uses raw tokenizer)
            parsed = self._index.parse_query(
                f'doc_id:"{doc_id}"', [self.FIELD_DOC_ID]
            )
            results = searcher.search(parsed, 1)

            if not results.hits:
                return None

            _, doc_address = results.hits[0]
            doc = searcher.doc(doc_address)

            return {
                "doc_id": doc.get_first(self.FIELD_DOC_ID) or "",
                "title": doc.get_first(self.FIELD_TITLE) or "",
                "content": doc.get_first(self.FIELD_CONTENT) or "",
                "filename": doc.get_first(self.FIELD_FILENAME) or "",
                "source_path": doc.get_first(self.FIELD_SOURCE_PATH) or "",
                "keywords": doc.get_first(self.FIELD_KEYWORDS) or "",
            }

        except Exception as e:
            logger.error("Error getting document %s: %s", doc_id, e)
            return None

    def commit(self) -> bool:
        """
        Commit pending changes to the index.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self._writer is None:
                return False
            self._writer.commit()
            self._index.reload()
            return True
        except Exception as e:
            logger.error("Error committing index: %s", e)
            return False

    def rebuild(self) -> bool:
        """
        Rebuild the entire index from scratch.

        This clears all documents and resets the index.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Close current writer
            self._writer.commit()

            # Delete and recreate index directory
            if self._index_path is not None:
                import shutil

                index_dir = Path(self._index_path)
                if index_dir.exists():
                    shutil.rmtree(index_dir)
                index_dir.mkdir(parents=True, exist_ok=True)

            # Recreate index
            self._index = self._create_index()
            self._register_chinese_tokenizer()

            # Create new writer
            self._writer = self._index.writer(
                heap_size=self._heap_size, num_threads=self._num_threads
            )

            return True

        except Exception as e:
            logger.error("Error rebuilding index: %s", e)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        try:
            self._index.reload()
            searcher = self._index.searcher()

            return {
                "num_docs": searcher.num_docs,
                "index_path": str(self._index_path) if self._index_path else "memory",
                "jieba_enabled": self._jieba_available,
                "heap_size": self._heap_size,
                "num_threads": self._num_threads,
            }
        except Exception as e:
            return {
                "num_docs": 0,
                "error": str(e),
            }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - commits and cleans up."""
        self.commit()
        return False

    def close(self):
        """Close the index and release resources."""
        try:
            if self._writer is not None:
                self.commit()
                # Wait for merging threads to complete
                self._writer.wait_merging_threads()
        except Exception:
            pass
        finally:
            # Release the writer to free the lock
            self._writer = None

    # Backward-compatible alias for tests
    _create_excerpt = _create_excerpt_fallback
