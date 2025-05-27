import os
import argparse
import tiktoken
from pathlib import Path

from src import utils
from src import pipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="rag", choices=["en", "zh", "rag"])
    parser.add_argument("--output_dir", type=str, default="./results_zh")
    parser.add_argument("--chunk_length", type=int, default=8000)
    parser.add_argument("--input_length", type=int, default=8000)
    parser.add_argument("--api_url", type=str, default=str(os.getenv("OPENAI_BASE_URL")))
    parser.add_argument("--api_key", type=str, default=str(os.getenv("OPENAI_API_KEY")))
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--data_path", type=str, default=None)
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    task = args.task
    output_dir = Path(args.output_dir)
    chunk_length = args.chunk_length
    input_length = args.input_length
    data_path = args.data_path
    
    # Initialize OpenAI client and set global variables
    utils.initialize_client(args.api_url, args.api_key, args.model)
    
    # Check if the model is an open source model (like Llama)
    is_open_source_model = "llama" in args.model.lower()
    temperature = "0.1" if is_open_source_model else "0.0"
    
    # Print task and model
    print(f"Task: {task} | Model: {args.model}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / f"final_preds.jsonl"

    # Select default data path based on task type
    if data_path is None:
        if task == "en":
            data_path = "./data/longbook_qa_eng.jsonl"
        elif task == "zh":
            data_path = "./data/longbook_qa_chn.jsonl"
        elif task == "rag":
            data_path = "./data/rag_1000k.jsonl"
    
    print(f"Loading data from {data_path}")
    examples = list(utils.iter_jsonl(data_path))
    
    # Initialize tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    # Process examples with pipeline
    final_preds, run_time_info = pipeline.run_pipeline(
        examples,
        tokenizer,
        task,
        chunk_length,
        input_length,
        output_dir,
        max_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
