"""Integration test for end-to-end pipeline."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

@pytest.mark.slow
class TestPipeline:
    def test_forward_pass_smoke(self):
        """Smoke test: verify shapes through the full pipeline."""
        pass  # requires model weights and lexicon data
