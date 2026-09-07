import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs HF hub or local model weights")
