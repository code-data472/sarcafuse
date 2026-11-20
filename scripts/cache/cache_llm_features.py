"""Pre-compute and cache GPT-4o reasoning outputs for all samples."""
import argparse, os, sys, json, hashlib, time
import openai
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.channels.llm.cot_prompter import CoTIncongruityPrompter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="gpt-4o-2024-05-13")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--rate_limit_delay", type=float, default=0.5)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    client = openai.OpenAI()
    prompter = CoTIncongruityPrompter(
        "prompts/templates/cot_incongruity_v3.txt",
        "prompts/exemplars/three_shot.json",
        "prompts/schemas/incongruity_output.json",
    )
    import csv
    with open(args.data_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["text"]
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            outpath = os.path.join(args.output_dir, f"{h}.json")
            if os.path.exists(outpath):
                continue
            messages = prompter.build_prompt(text)
            for attempt in range(args.max_retries + 1):
                try:
                    resp = client.chat.completions.create(
                        model=args.model, messages=messages,
                        temperature=args.temperature, max_tokens=512,
                    )
                    parsed = prompter.parse_response(resp.choices[0].message.content)
                    with open(outpath, "w") as fout:
                        json.dump(parsed, fout)
                    break
                except Exception as e:
                    print(f"Retry {attempt}: {e}")
                    time.sleep(args.rate_limit_delay * (attempt + 1))
            time.sleep(args.rate_limit_delay)

if __name__ == "__main__":
    main()
