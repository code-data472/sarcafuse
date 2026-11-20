"""Chain-of-thought incongruity prompting for GPT-4o."""
import json
import os
from typing import Dict, Optional, Tuple

class CoTIncongruityPrompter:
    def __init__(self, template_path: str, exemplar_path: str, schema_path: str):
        with open(template_path) as f:
            self._template = f.read()
        with open(exemplar_path) as f:
            self._exemplars = json.load(f)
        with open(schema_path) as f:
            self._schema = json.load(f)

    def build_prompt(self, sentence: str) -> list:
        system_msg = self._template.format(schema=json.dumps(self._schema, indent=2))
        messages = [{"role": "system", "content": system_msg}]
        for ex in self._exemplars:
            messages.append({"role": "user", "content": f'Sentence: "{ex["sentence"]}"'})
            messages.append({"role": "assistant", "content": json.dumps(ex["output"])})
        messages.append({"role": "user", "content": f'Sentence: "{sentence}"'})
        return messages

    @staticmethod
    def parse_response(response_text: str) -> Dict:
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
        defaults = {
            "surface_polarity": 0.0, "situational_polarity": 0.0,
            "hyperbole_score": 0.0, "incongruity_score": 0.0,
            "sarcasm_confidence": 0.0, "reasoning_chain": "",
        }
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
        return data

    @staticmethod
    def extract_numerical(parsed: Dict) -> Tuple[float, ...]:
        return (
            parsed["surface_polarity"], parsed["situational_polarity"],
            parsed["hyperbole_score"], parsed["incongruity_score"],
            parsed["sarcasm_confidence"],
        )
