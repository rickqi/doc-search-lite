"""
Smoke tests to verify imports and basic functionality.
"""



def test_pytest_works():
    """Trivial test to verify pytest is working."""
    assert True


def test_import_click():
    """Verify click library can be imported."""
    import click

    assert click is not None


def test_import_tantivy():
    """Verify tantivy library can be imported."""
    import tantivy

    assert tantivy is not None


def test_import_markitdown():
    """Verify markitdown library can be imported."""
    import markitdown

    assert markitdown is not None


def test_import_litellm():
    """Verify litellm library can be imported."""
    import litellm

    assert litellm is not None


def test_config_import():
    """Verify Config can be imported from src.utils.config."""
    from src.utils.config import Config

    assert Config is not None


def test_converter_base_import():
    """Verify Converter and ConvertResult can be imported from src.converter.base."""
    from src.converter.base import Converter, ConvertResult

    assert Converter is not None
    assert ConvertResult is not None


def test_storage_base_import():
    """Verify Storage and DocumentRecord can be imported from src.storage.base."""
    from src.storage.base import DocumentRecord, Storage

    assert Storage is not None
    assert DocumentRecord is not None
