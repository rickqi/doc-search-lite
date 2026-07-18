"""doc-search Watch module — filesystem monitor for automatic index updates.

Uses watchdog to monitor raw directories for .md file changes
and automatically updates the Tantivy index in real-time.
"""

from src.watch.index_watcher import IndexWatcher, start_watching

__all__ = ["IndexWatcher", "start_watching"]
