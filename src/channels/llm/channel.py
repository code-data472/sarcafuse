"""LLM channel: prompt -> GPT-4o -> encode features."""
import torch
import torch.nn as nn
import json, os
from .cot_prompter import CoTIncongruityPrompter
from .feature_encoder import IncongruityFeatureEncoder

class LLMChannel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prompter = CoTIncongruityPrompter(
            config.template_path, config.exemplar_path, config.schema_path)
        self.encoder = IncongruityFeatureEncoder(
            channel_dim=config.channel_dim, sbert_name=config.sbert_name)
        self.cache_dir = config.cache_dir
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    def _load_cached(self, sentence_hash):
        path = os.path.join(self.cache_dir, f"{sentence_hash}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, sentence_hash, data):
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f"{sentence_hash}.json")
        with open(path, "w") as f:
            json.dump(data, f)

    def _query_llm(self, sentence, config):
        import hashlib
        h = hashlib.sha256(sentence.encode()).hexdigest()[:16]
        cached = self._load_cached(h)
        if cached is not None:
            return cached
        client = self._get_client()
        messages = self.prompter.build_prompt(sentence)
        resp = client.chat.completions.create(
            model=config.llm_model, messages=messages,
            temperature=config.llm_temperature, top_p=config.llm_top_p,
            max_tokens=config.llm_max_tokens,
        )
        text = resp.choices[0].message.content
        parsed = self.prompter.parse_response(text)
        self._save_cache(h, parsed)
        return parsed

    def forward(self, sentences, config=None, cached_features=None):
        if cached_features is not None:
            numerical = cached_features["numerical"]
            chains = cached_features["chains"]
        else:
            numerical_list, chains = [], []
            for s in sentences:
                parsed = self._query_llm(s, config)
                numerical_list.append(list(self.prompter.extract_numerical(parsed)))
                chains.append(parsed.get("reasoning_chain", s))
            numerical = torch.tensor(numerical_list, dtype=torch.float32)
        return self.encoder(numerical, chains)
