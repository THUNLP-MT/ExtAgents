import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import prompt as prompt_module
from . import utils

def process_example(i, examples, tokenizer, task, chunk_length, input_length, max_iterations=5):
    """
    Complete pipeline for processing a single example
    
    Steps include:
    1. Text chunking (Map stage preparation)
    2. Extracting information from text chunks through multiple iterations (Map stage)
    3. Generating final answer based on extracted information (Reduce stage)
    
    Args:
        i: Example index
        examples: List of examples
        tokenizer: Tokenizer for text encoding/decoding
        task: Task type ("en", "zh", or "rag")
        chunk_length: Length of each text chunk
        input_length: Maximum input length to consider
        max_iterations: Maximum number of iterations
        
    Returns:
        Processing result tuple (id, final prediction, info list, map stage time, reduce stage time)
    """
    current_info = []
    eg = examples[i]
    context = eg["context"]
    id = i
    question = eg.get("question", eg.get("input", ""))  # Compatible with different data formats
    example_map_time = 0.0
    example_reduce_time = 0.0
    iteration = 0
    final_pred = "NO ANSWER"
    
    # Variables specific to RAG task
    filtered_info = []
    
    print(f"Processing example {i}...")
    
    # Get text chunks using utils.create_chunks
    manner = "front" if task == "rag" else "middle"
    chunks = utils.create_chunks(
        tokenizer=tokenizer,
        context=context,
        chunk_length=chunk_length,
        input_length=input_length,
        manner=manner
    )

    try:
        while iteration < max_iterations:
            iteration += 1
            
            # MAP STAGE #
            chunked_msgs = []
            for chunk in chunks:
                if iteration == 1:
                    prompt = prompt_module.create_first_iteration_prompt(task, chunk, question)
                else:
                    if task == "rag":
                        prompt = prompt_module.create_iteration_prompt(
                            task, iteration, question, chunk, selected_info=filtered_info
                        )
                    else:
                        prompt = prompt_module.create_iteration_prompt(
                            task, iteration, question, chunk, selected_info=current_info
                        )
                
                msgs = prompt_module.create_chunked_msgs(prompt)
                chunked_msgs.append((msgs, prompt))

            for chunk_id, (msgs, prompt) in enumerate(chunked_msgs):
                map_start_time = time.time()
                info = utils.chat(msgs)
                map_end_time = time.time()
                example_map_time += (map_end_time - map_start_time)
                
                if task in ["en", "zh"]:
                    reduce_start_time = time.time()
                    score = utils.get_info_score(info, question, task, prompt_module)
                    reduce_end_time = time.time()
                    example_reduce_time += (reduce_end_time - reduce_start_time)
                    current_info.append((info, score))
                elif task == "rag":
                    current_info.append(info)
            
            # Special handling for filtered_info in RAG task
            if task == "rag":
                filtered_info = []
                for info in current_info:
                    if "no information" not in info.lower():
                        filtered_info.append(info)
                
                # Exit iteration if no information was extracted in map stage
                if not filtered_info:
                    break

            # REDUCE STAGE #
            if task in ["en", "zh"]:
                if iteration == 1:
                    # Sort extracted information by score in descending order
                    sorted_info = sorted(current_info, key=lambda x: x[1], reverse=True)
                    # Construct candidate counts: 1, 2, 4, 8, ... up to the total count
                    candidate_ranks = []
                    r = 1
                    while r <= len(sorted_info):
                        candidate_ranks.append(r)
                        r *= 2
                    if r < len(sorted_info) * 2:
                        candidate_ranks.append(len(sorted_info))
                    if not candidate_ranks:
                        candidate_ranks = [len(sorted_info)]
                    answer_found = False
                    
                    # Try different candidate counts
                    for r in candidate_ranks:
                        selected_info = [info for info, score in sorted_info[:r]]
                        
                        prompt = prompt_module.create_reduce_prompt(task, selected_info, question)

                        msgs = prompt_module.create_chunked_msgs(prompt)
                        reduce_start_time = time.time()
                        final_pred = utils.chat(msgs)
                        reduce_end_time = time.time()
                        example_reduce_time += (reduce_end_time - reduce_start_time)
                        if "no answer" not in final_pred.lower() and "no info" not in final_pred.lower():
                            answer_found = True
                            break
                    if answer_found:
                        break
                elif iteration == max_iterations:
                    selected_info = [info for info, score in current_info]
                    
                    prompt = prompt_module.create_reduce_prompt(
                        task, selected_info, question, is_final_iteration=True
                    )
                    msgs = prompt_module.create_chunked_msgs(prompt)
                    reduce_start_time = time.time()
                    final_pred = utils.chat(msgs)
                    reduce_end_time = time.time()
                    example_reduce_time += (reduce_end_time - reduce_start_time)
                    if "no answer" not in final_pred.lower() and "no info" not in final_pred.lower():
                        break
                else:
                    selected_info = [info for info, score in current_info]
                    
                    prompt = prompt_module.create_reduce_prompt(task, selected_info, question)
                    
                    msgs = prompt_module.create_chunked_msgs(prompt)
                    final_pred = utils.chat(msgs)
                    if "no answer" not in final_pred.lower() and "no info" not in final_pred.lower():
                        break
                        
            elif task == "rag":
                if iteration == 1:
                    iteration_current_info = current_info.copy()
                    last_info = []
                    k = 1
                    while 2**(k-1) <= len(current_info):
                        iteration_current_info = current_info[:2**(k-1)].copy()
                        iteration_filtered_info = []
                        for info in iteration_current_info:
                            if "no information" not in info.lower():
                                iteration_filtered_info.append(info)
                        if not iteration_filtered_info:
                            k += 1
                            continue
                        last_info = iteration_filtered_info
                        
                        prompt = prompt_module.create_reduce_prompt(task, iteration_filtered_info, question)
                        
                        msgs = prompt_module.create_chunked_msgs(prompt)
                        reduce_start_time = time.time()
                        final_pred = utils.chat(msgs)
                        reduce_end_time = time.time()
                        example_reduce_time += (reduce_end_time - reduce_start_time)
                        if "no answer" not in final_pred.lower():
                            break
                        k += 1
                    if "no answer" not in final_pred.lower():
                        break

                elif iteration == max_iterations:
                    prompt = prompt_module.create_reduce_prompt(
                        task, filtered_info, question, is_final_iteration=True
                    )
                    
                    msgs = prompt_module.create_chunked_msgs(prompt)
                    reduce_start_time = time.time()
                    final_pred = utils.chat(msgs)
                    reduce_end_time = time.time()
                    example_reduce_time += (reduce_end_time - reduce_start_time)
                    if "no answer" not in final_pred.lower():
                        break

                else:
                    prompt = prompt_module.create_reduce_prompt(task, filtered_info, question)
                    
                    msgs = prompt_module.create_chunked_msgs(prompt)
                    reduce_start_time = time.time()
                    final_pred = utils.chat(msgs)
                    reduce_end_time = time.time()
                    example_reduce_time += (reduce_end_time - reduce_start_time)
                    if "no answer" not in final_pred.lower():
                        break

        # Post-process prediction for open source models
        is_open_source_model = "llama" in utils.model.lower()
        if task == "rag" and is_open_source_model and final_pred != "NO ANSWER":
            original_pred = final_pred
            reduce_start_time = time.time()
            final_pred, post_time = utils.postprocess_prediction(final_pred, question, prompt_module)
            reduce_end_time = time.time()
            example_reduce_time += post_time

        # Create output format for different tasks
        if task in ["en", "zh"]:
            current_info_list = [
                {
                    "id": id,
                    "prediction": info,
                }
                for idx, (info, _) in enumerate(current_info)
            ]
        elif task == "rag":
            current_info_list = [
                {
                    "id": id,
                    "prediction": info,
                }
                for idx, info in enumerate(current_info)
            ]

        example_total_time = example_map_time + example_reduce_time
        return id, final_pred, current_info_list, example_map_time, example_reduce_time

    except Exception as e:
        print(f"Error in example {i}: {e}")
        return None


def run_pipeline(examples, tokenizer, task, chunk_length, input_length, output_dir, max_workers=1):
    """
    Execute the complete Map-Reduce pipeline to process multiple examples
    
    Args:
        examples: List of examples to process
        tokenizer: Tokenizer for text encoding/decoding
        task: Task type ("en", "zh", or "rag")
        chunk_length: Length of each text chunk
        input_length: Maximum input length to consider
        output_dir: Directory for output results
        max_workers: Number of parallel worker threads
        
    Returns:
        List of processing results and runtime information
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / f"final_preds.jsonl"

    final_preds = []
    processed_ids = set()  # Set to store already processed IDs

    # Extract already processed IDs from final prediction file
    if final_output_path.exists():
        final_preds = list(utils.iter_jsonl(final_output_path))
        for pred in final_preds:
            processed_ids.add(pred["id"])
        if processed_ids:
            print(f"Found {len(processed_ids)} already processed examples")

    start_idx = 0
    stop_idx = len(examples)
    
    start_time = time.time()
    map_time = 0.0
    reduce_time = 0.0

    lock = threading.Lock()  # Ensure thread safety

    def process_result(result):
        nonlocal map_time, reduce_time
        if result is not None:
            id, final_pred, current_info_list, example_map_time, example_reduce_time = result
            
            # Calculate total time for this example
            example_total_time = example_map_time + example_reduce_time
            
            with lock:
                if task == "rag":
                    final_preds.append(
                        {
                            "id": id,
                            "prediction": final_pred,
                        }
                    )
                else:
                    final_preds.append(
                        {
                            "id": id,
                            "prediction": final_pred,
                            "ground_truth": examples[id].get("answer", ""),
                        }
                    )
                utils.dump_jsonl(final_preds, final_output_path)
            print(f"Final prediction for example {id}: {final_pred}")
            map_time += example_map_time
            reduce_time += example_reduce_time
        else:
            print(f"No result for example {id}")

    print(f"Processing {stop_idx - start_idx - len(processed_ids)} examples with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # Only process unprocessed IDs
        for i in range(start_idx, stop_idx):
            # Skip already processed IDs
            if i in processed_ids:
                continue
            future = executor.submit(
                process_example,
                i,
                examples,
                tokenizer,
                task,
                chunk_length,
                input_length,
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result()
                process_result(result)
            except Exception as e:
                print(f"Exception occurred: {e}")

    # Sort results by ID after processing
    final_preds.sort(key=lambda x: x["id"])

    # Write sorted results to files
    utils.dump_jsonl(final_preds, final_output_path)

    total_time = time.time() - start_time
    run_time_info = {
        "total_time": total_time,
        "map_time": map_time,
        "reduce_time": reduce_time
    }
    
    print(f"Total processing time: {total_time:.2f}s, Map time: {map_time:.2f}s, Reduce time: {reduce_time:.2f}s")
    print(f"Results saved to {output_dir}")
    
    return final_preds, run_time_info


# Compatibility function to maintain existing API
process_examples_with_threads = run_pipeline 