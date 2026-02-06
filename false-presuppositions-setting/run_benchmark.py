"""
Main script for running the SYCON-Bench false presuppositions setting benchmark.

This script provides a unified interface for running the benchmark with different models.
"""
import os
import argparse
import logging
import time
import torch
from models import ModelFactory

def check_device():
    """Check and print available compute devices."""
    print("\n=== Device Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    print("====================\n")
    return device

def load_questions_for_prompt(prompt_file, data_dir="data"):
    """Load questions for a given prompt type.

    Order of precedence:
    1. data/{prompt_file}_pressure.csv
    2. data/{prompt_file}.csv
    3. data/questions.txt
    """
    import csv

    pressure_path = os.path.join(data_dir, f"{prompt_file}_pressure.csv")
    csv_path = os.path.join(data_dir, f"{prompt_file}.csv")
    default_txt = os.path.join(data_dir, "questions.txt")

    def _read_csv_first_col(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for r in reader:
                if not r:
                    continue
                val = r[0].strip()
                if val:
                    rows.append(val)
        # strip header if it looks like 'Question'
        if rows and rows[0].lower().startswith("question"):
            return rows[1:]
        return rows

    if os.path.exists(pressure_path):
        return _read_csv_first_col(pressure_path)
    if os.path.exists(csv_path):
        return _read_csv_first_col(csv_path)
    if os.path.exists(default_txt):
        with open(default_txt, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    raise FileNotFoundError(f"No questions file found for prompt '{prompt_file}' in {data_dir}")

def run_benchmark(model, batch_size=4, output_dir=None, prompt_types=None, data_dir="data"):
    """
    Run the benchmark for a given model on the provided questions.
    
    Args:
        model: The model instance to use
        questions: List of questions
        batch_size: Number of questions to process in each batch
        output_dir: Directory to save results
        prompt_types: List of prompt types to use (default: all prompt types)
    """
    if prompt_types is None:
        prompt_types = [
            ("base", "base"),
            ("critical", "critical"),
            ("presupposition", "presupposition")
        ]
    
    # Process each prompt type independently so each can use its own question set
    for prompt_type, prompt_file in prompt_types:
        logging.info(f"Processing prompt type: {prompt_type} (file: {prompt_file})")

        # Load the appropriate questions for this prompt type (pressure CSVs take precedence)
        questions = load_questions_for_prompt(prompt_file, data_dir=data_dir)
        results = {q: [] for q in questions}

        # Process questions in batches for this prompt type
        for batch_start in range(0, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]

            logging.info(f"\nProcessing questions {batch_start+1}-{batch_end} out of {len(questions)} for {prompt_file}")

            for question in batch_questions:
                messages = model.get_chat_messages(question, prompt_type)
                response = model.generate_responses(messages)
                results[question] = [response]

            # Save batch results
            model.save_batch_results(results, f"{output_dir}/{prompt_file}.csv", batch_questions)
            logging.info(f"Saved batch results to {output_dir}/{prompt_file}.csv")

        logging.info(f"Completed processing prompt type {prompt_file} ({len(questions)} questions)")
    
    # Log completion
    prompt_files = [prompt_file for _, prompt_file in prompt_types]
    logging.info(f"\nCompleted processing model")
    logging.info(f"Final results saved to {', '.join([f'{output_dir}/{pf}.csv' for pf in prompt_files])}")

def main():
    parser = argparse.ArgumentParser(description="Run SYCON-Bench false presuppositions setting benchmark")
    parser.add_argument("model_name", type=str, help="Model name or identifier")
    parser.add_argument("--api_key", type=str, help="API key for closed models")
    parser.add_argument("--base_url", type=str, help="Base URL for API (for closed models)")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of questions to process in each batch")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--prompt_type", type=str, choices=["all", "base", "critical", "presupposition"], 
                        default="all", help="Prompt type to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if not args.verbose else logging.DEBUG,
        handlers=[logging.StreamHandler()],
    )
    
    # Log arguments (without sensitive info)
    log_args = vars(args).copy()
    if 'api_key' in log_args:
        log_args['api_key'] = '***' if log_args['api_key'] else None
    logging.info(f"Arguments: {log_args}")

    # Check and display device info
    check_device()
    
    # Create a shorter model identifier for directory names
    model_id = args.model_name.split("/")[-1]
    
    # Ensure the output directory exists
    output_dir = args.output_dir if args.output_dir else f"output/{model_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Log information about the run
    logging.info(f"Processing model: {args.model_name}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Batch size: {args.batch_size}")
    
    # Create the model
    model = ModelFactory.create_model(args.model_name, api_key=args.api_key, base_url=args.base_url)
    
    # Set up the model
    model.setup()
    
    # Determine which prompt types to use
    if args.prompt_type == "all":
        prompt_types = [
            ("base", "base"),
            ("critical", "critical"),
            ("presupposition", "presupposition")
        ]
    else:
        prompt_types = [(args.prompt_type, args.prompt_type)]
    # Run the benchmark. allow specifying data directory with pressure CSVs.
    data_dir = "data"
    run_benchmark(
        model=model,
        batch_size=args.batch_size,
        output_dir=output_dir,
        prompt_types=prompt_types,
        data_dir=data_dir,
    )

if __name__ == "__main__":
    main()