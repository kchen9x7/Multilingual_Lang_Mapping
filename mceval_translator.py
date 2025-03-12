import os
import time
import json
import random
import argparse
import glob
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union
# from transformers import AutoModel, AutoTokenizer

import requests
import numpy as np
from datasets import load_dataset
from bert_score import BERTScorer
from tenacity import retry, stop_after_attempt, wait_exponential

# Global tracking variables for API usage and cost estimation
api_call_count = 0
estimated_cost = 0.0

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def translate_with_gpt(text: str, target_language: str, prompt_type: str) -> str:
    """
    Translate text using GPT-4o API with retry logic
    
    Args:
        text: Text to translate
        target_language: Target language for translation
        prompt_type: Type of prompt (prompt, instruction, docstring)
        
    Returns:
        Translated text
    """
    global api_call_count, estimated_cost
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    prompt = f"Translate the natural language portion of the following {prompt_type} content related to a computer programming task into {target_language}:\n\n{text}"
    
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    
    # Track API calls and estimate cost (rough estimate based on characters)
    api_call_count += 1
    # Estimate input tokens (roughly 4 chars per token) and cost at $0.0025/1K tokens
    input_tokens = len(prompt) / 4
    estimated_cost += (input_tokens / 1000) * 0.0025
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    response_data = response.json()
    
    if "choices" in response_data and len(response_data["choices"]) > 0:
        output_text = response_data["choices"][0]["message"]["content"]
        # Estimate output tokens and cost at $0.01/1K tokens
        output_tokens = len(output_text) / 4
        estimated_cost += (output_tokens / 1000) * 0.01
        return output_text
    else:
        raise Exception(f"Error in GPT-4o API response: {response_data}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def back_translate_with_gpt(text: str, source_language: str) -> str:
    """
    Back-translate text from source language to English using GPT-4o API
    
    Args:
        text: Text to back-translate
        source_language: Source language of the text
        
    Returns:
        Back-translated text in English
    """
    global api_call_count, estimated_cost
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    prompt = f"Translate the following {source_language} text back to English:\n\n{text}"
    
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    
    # Track API calls and cost
    api_call_count += 1
    input_tokens = len(prompt) / 4
    estimated_cost += (input_tokens / 1000) * 0.0025
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    response_data = response.json()
    
    if "choices" in response_data and len(response_data["choices"]) > 0:
        output_text = response_data["choices"][0]["message"]["content"]
        output_tokens = len(output_text) / 4
        estimated_cost += (output_tokens / 1000) * 0.01
        return output_text
    else:
        raise Exception(f"Error in GPT-4o API response: {response_data}")

def calculate_bert_score(original_text: str, back_translated_text: str) -> float:
    """
    Calculate BERTScore between original and back-translated text
    
    Args:
        original_text: Original English text
        back_translated_text: Back-translated English text
        
    Returns:
        BERTScore F1 value
    """
    # Initialize the BERTScore with the specified model
    scorer = BERTScorer(model_type="princeton-nlp/sup-simcse-roberta-large", lang="en", rescale_with_baseline=False)
    P, R, F1 = scorer.score([back_translated_text], [original_text])
    return F1.item()  # Return the F1 score as a Python float

def process_item(item: Dict[str, Any], target_languages: List[str], max_iterations: int = 3) -> Dict[str, Any]:
    """
    Process a single dataset item with translation and back-translation
    
    Args:
        item: Dataset item
        target_languages: List of target languages for translation
        max_iterations: Maximum number of iterations for translation
        
    Returns:
        Processed item with translations
    """
    global api_call_count, estimated_cost
    
    item_result = {
        "task_id": item["task_id"],
        "original": {
            "prompt": item["prompt"],
            "instruction": item["instruction"],
            "docstring": item["docstring"],
            # "code": item["code"]
        },
        "translations": {}
    }
    
    for lang in target_languages:
        print(f"  Processing language: {lang}")
        lang_result = {
            "prompt": {},
            "instruction": {},
            "docstring": {},
            "best_score": 0
        }
        
        for field in ["prompt", "instruction", "docstring"]:
            print(f"    Processing field: {field}")
            field_result = {
                "iterations": []
            }
            
            best_score = 0
            best_translation = ""
            best_back_translation = ""
            
            for iteration in range(max_iterations):                              
                try:
                    print(f"      Iteration {iteration+1}/{max_iterations}")
                    
                    # Translate to target language
                    print(f"        Translating to {lang}...")
                    start_time = time.time()
                    translated_text = translate_with_gpt(item[field], lang, field)
                    translation_time = time.time() - start_time
                    
                    # Back-translate to English
                    print(f"        Back-translating to English...")
                    start_time = time.time()
                    back_translated_text = back_translate_with_gpt(translated_text, lang)
                    back_translation_time = time.time() - start_time
                    
                    # Calculate BERTScore
                    print(f"        Calculating BERTScore...")
                    start_time = time.time()
                    score = calculate_bert_score(item[field], back_translated_text)
                    bertscore_time = time.time() - start_time
                    
                    print(f"        Score: {score:.4f}")
                    
                    iteration_result = {
                        "translated_text": translated_text,
                        "back_translated_text": back_translated_text,
                        "score": score,
                        "times": {
                            "translation": translation_time,
                            "back_translation": back_translation_time,
                            "bertscore": bertscore_time
                        }
                    }
                    
                    field_result["iterations"].append(iteration_result)
                    
                    if score > best_score:
                        best_score = score
                        best_translation = translated_text
                        best_back_translation = back_translated_text
                    
                    if score > 0.95:
                        print(f"        Score above threshold (0.95). Stopping iterations.")
                        break
                        
                    # Add a delay to avoid API rate limits
                    sleep_time = random.uniform(1.5, 3.0)
                    print(f"        Sleeping for {sleep_time:.2f}s to avoid rate limits...")
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"        Error: {str(e)}")
                    field_result["iterations"].append({"error": str(e)})
                    print(f"        Sleeping for 5s after error...")
                    time.sleep(5)  # Longer delay after an error
            
            # Store the best results for this field
            field_result["best_translation"] = best_translation
            field_result["best_back_translation"] = best_back_translation
            field_result["best_score"] = best_score
            
            lang_result[field] = field_result
            
            if best_score > lang_result["best_score"]:
                lang_result["best_score"] = best_score
        
        item_result["translations"][lang] = lang_result

    
    return item_result

def find_last_processed_index(dataset_name: str, start_idx: int, end_idx: int) -> int:
    """
    Find the last processed index from existing output files
    
    Args:
        dataset_name: Name of the dataset (explanation or generation)
        start_idx: Start index for processing
        end_idx: End index for processing
        
    Returns:
        Last processed index
    """
    output_pattern = f"{dataset_name}_translation_results_{start_idx}_to_*.json"
    files = glob.glob(output_pattern)
    if not files:
        return start_idx - 1  # Return one less than start_idx to start from start_idx
    
    last_idx = start_idx - 1  # Default if no valid index found
    for file in files:
        match = re.search(r'_to_(\d+)\.json$', file)
        if match:
            idx = int(match.group(1))
            if idx > last_idx and idx <= end_idx:
                last_idx = idx
    
    return last_idx

def load_existing_data(dataset_name: str, start_idx: int) -> List[Dict[str, Any]]:
    """
    Load existing processed data from files
    
    Args:
        dataset_name: Name of the dataset (explanation or generation)
        start_idx: Start index for processing
        
    Returns:
        Existing processed data
    """
    output_pattern = f"{dataset_name}_translation_results_{start_idx}_to_*.json"
    files = glob.glob(output_pattern)
    if not files:
        return []
    
    # Find the file with the highest end index
    latest_file = None
    latest_end_idx = -1
    
    for file in files:
        match = re.search(r'_to_(\d+)\.json$', file)
        if match:
            end_idx = int(match.group(1))
            if end_idx > latest_end_idx:
                latest_end_idx = end_idx
                latest_file = file
    
    if latest_file:
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing data from {latest_file}: {str(e)}")
    
    return []

def process_dataset(
    dataset: Any, 
    start_idx: int, 
    end_idx: int, 
    target_languages: List[str], 
    max_iterations: int = 3, 
    save_interval: int = 1, 
    resume: bool = False, 
    # max_cost: float = float('inf')
) -> List[Dict[str, Any]]:
    """
    Process a subset of the dataset with translation and back-translation
    
    Args:
        dataset: HuggingFace dataset
        start_idx: Start index for processing
        end_idx: End index for processing
        target_languages: List of target languages for translation
        max_iterations: Maximum number of iterations for translation
        save_interval: Interval for saving intermediate results
        resume: Whether to resume from previous processing
        max_cost: Maximum cost before stopping
        
    Returns:
        Processed data with translations
    """
    global api_call_count, estimated_cost
    
    # Determine dataset name for file naming
    dataset_name = "generation" if "generation" in dataset else "explanation"
    
    if resume:
        # Find the last processed index
        last_processed_idx = find_last_processed_index(dataset_name, start_idx, end_idx)
        
        if last_processed_idx >= start_idx:
            print(f"Resuming from index {last_processed_idx + 1} (previously processed up to {last_processed_idx})")
            processed_data = load_existing_data(dataset_name, start_idx)
            current_start_idx = last_processed_idx + 1
        else:
            print(f"No previous processing found. Starting from index {start_idx}")
            processed_data = []
            current_start_idx = start_idx
    else:
        processed_data = []
        current_start_idx = start_idx
    
    # Don't process beyond the end index
    if current_start_idx > end_idx:
        print(f"All items (up to index {end_idx}) have already been processed.")
        return processed_data
    
    # Process the remaining items
    for idx in tqdm(range(current_start_idx, end_idx + 1), desc="Processing dataset"):
        print(f"\nProcessing item {idx}...")
        print(f"API calls so far: {api_call_count}, Estimated cost: ${estimated_cost:.2f}")
            
        item = dataset['test'][idx]
        item_result = process_item(item, target_languages, max_iterations)
        processed_data.append(item_result)
        
        # Save intermediate results at specified intervals
        if (idx - current_start_idx + 1) % save_interval == 0 or idx == end_idx:
            output_file = f"{dataset_name}_translation_results_{start_idx}_to_{idx}.json"
            print(f"Saving intermediate results to {output_file}...")
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            print(f"Current statistics:")
            print(f"  Total API calls: {api_call_count}")
            print(f"  Estimated cost: ${estimated_cost:.2f}")
    
    return processed_data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process McEval dataset with translation and back-translation.')
    parser.add_argument('--dataset', type=str, default="generation", choices=["explanation", "generation"],
                        help='Dataset to process (explanation or generation)')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing')
    parser.add_argument('--end_idx', type=int, default=2, help='End index for processing')
    parser.add_argument('--languages', type=str, nargs='+', 
                        default=["Chinese", "Bengali", "Farsi", "French", "Hindi", "Japanese"],
                        help='Target languages for translation')
    parser.add_argument('--max_iterations', type=int, default=3, 
                        help='Maximum number of iterations for translation')
    parser.add_argument('--save_interval', type=int, default=1, 
                        help='Interval for saving intermediate results')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous processing')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        return
    
    # Load datasets
    print("Loading datasets...")
    if args.dataset == "explanation":
        ds = load_dataset("Multilingual-Multimodal-NLP/McEval", "explanation")
    else:
        ds = load_dataset("Multilingual-Multimodal-NLP/McEval", "generation")
    
    # Process dataset
    print(f"Processing {args.dataset} dataset from index {args.start_idx} to {args.end_idx}...")
    print(f"Target languages: {args.languages}")
    print(f"Maximum iterations: {args.max_iterations}")
    
    try:
        results = process_dataset(
            ds, 
            args.start_idx, 
            args.end_idx, 
            args.languages, 
            args.max_iterations,
            args.save_interval,
            args.resume,
        )
        
        # Save final results
        final_output_file = f"{args.dataset}_translation_results_{args.start_idx}_to_{args.end_idx}_final.json"
        print(f"Saving final results to {final_output_file}...")
        with open(final_output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by the user.")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Print final statistics
    print("\nFinal statistics:")
    print(f"  Total API calls: {api_call_count}")
    print(f"  Estimated cost: ${estimated_cost:.2f}")
    
    print("Done!")

if __name__ == "__main__":
    main()