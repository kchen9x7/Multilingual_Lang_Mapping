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
# def translate_with_gpt(text: str, target_language: str) -> str:
# def translate_with_gpt(text: str, target_language: str, prompt_type: str) -> str:
def translate_with_gpt(text: str, target_language: str, prompt_type: str, dataset_name: str, PL_name: str) -> str:
    """
    Translate text using GPT API with retry logic
    
    Args:
        text: Text to translate
        target_language: Target language for translation
        prompt_type: the content type in the part of the dataset for translation
        dataset_name: The target dataset for translation
        
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
    
    # prompt = f"Translate the natural language portion of the following {prompt_type} content related to a computer programming task into {target_language}:\n\n{text}"
    # prompt = (f"Translate the text of a computer programming task below into {target_language}, " \
    #           f"preserving all code exactly as is. Only translate the natural language portions (comments, docstrings, free-text explanations, etc.). " \
    #           f"Do not remove or alter the code. Keep the same structure, formatting, and indentation. " \
    #           f"Do not output any special code block or add any formatting, just give the output with the translation in a text format.\n\nText to translate:\n{text}")

    prompt = (
        f"You are an expert code documentation translator. "
        f"Your task is to translate *ONLY* the natural language parts (comments, docstrings, explanations, descriptive text) "
        f"within the provided programming task descriptions or code snippets into {target_language}. "
        f"\n\n**You must follow these strict rules:** "
        f"\n1.  Translate all human-readable text intended for explanation, like comments (e.g., `#`, `//`, `/* */`, etc.), docstrings, and any descriptive free-text surrounding code. "
        f"\n2.  Translate all human-readable text intended for instructions "
        f"(e.g. instruction: `Write a python function 'def largest_prime_factor(n: int) -> int:' to solve the following problem`, then `Write a python function` and `to solve the following problem` has to be translated into the target language) "
        f"\n3.  **DO NOT** translate any code elements (function/class/variable names, keywords, operators, syntax). "
        f"\n4.  **PRESERVE** the exact original code structure, indentation (spaces or tabs), line breaks, and formatting in the output. "
        f"\n5.  **PRESERVE** examples within docstrings/comments (e.g., `>>> example_code()` or code snippets shown for illustration) without translation. "
        # f"\n6.  **PRESERVE** the exact indentation (spaces or tabs) and line breaks formatting wrapped around the original input in the output. "
        # f"\n7.  Output the programming task descriptions or code snippets with the translated natural language text in a text format."
        f"\n6.  **DO NOT** wrap the output in Markdown code fences (like ```python ... ```) or any other formatting not present in the original input. The output structure must exactly match the input structure."
        # f"\n\n**Demonstration Examples:**\n*Input:*\n"
        f"\n\nTranslate below text using the provided instructions:\n{text}"
    )

    # instructions (e.g. Write a programming_language function function_name)

    # get_examples = get_translation_examples(PL_name, dataset_name, prompt_type)
    # example_source = get_examples["source"]
    # example_translation = get_examples["translation"]
    # prompt_ending = f"\n\n**Translation task**\nTranslate below text using the provided instructions and exmaple:\n{text}"

    # prompt = prompt + example_source + "\n**Demonstration Examples:**\n*Output:*\n" + example_translation + prompt_ending

    if dataset_name == "explanation" and prompt_type == "instruction":
        # extracted_text = extract_instruction(text)
        prompt = f"Translate the only natural language of the following instruction content into {target_language}:\n\n{text}"
    # elif dataset_name == "explanation" and prompt_type == "docstring":
    elif prompt_type == "docstring":
        prompt = (
            f"You are an expert code documentation translator. "
            f"Your task is to translate *ONLY* the natural language human-readable text intended for code explanation (comments, explanations, descriptive text) "
            f"within the provided programming task docstring into {target_language}. "
            f"\n\n**You must follow these strict rules:** "            
            f"\n1.  Handle specific sections like `Args:`, `Returns:`, `Examples:` by translating the section title and the descriptive text, while preserving the formatting and any associated code/type information. "
            # f"\n2.  **DO NOT** translate any actual code elements (function names, class/variable names, keywords, operators, and syntax). "
            f"\n2.  **DO NOT** translate any actual code elements (e.g. function names, class/variable names, keywords, operators, syntax, lines starting with `>>>`) in the code snippets. "
            f"\n3.  **PRESERVE** the exact original code structure, indentation (spaces or tabs), line breaks, and overall formatting of the code snippets in the output. "
            # f"\n4.  **PRESERVE** examples within the text (e.g., lines starting with `>>>` or code snippets shown for illustration without translation. Keep them exactly as they appear in the input. "
            f"\n4.  **DO NOT** wrap the output in Markdown code fences (like ```python ... ```) or any other formatting not present in the original input. The output structure must exactly match the input structure."
            f"\n5.  Output the text with the translated natural language text and code snippets based on above rules. "
            f"\n\n**Translation task**\nTranslate below text using the provided instructions:\n{text}"
            # f"\n\n**Demonstration Example 1:**\n*Input:*\n"
        )

        # get_examples = get_translation_examples(PL_name, dataset_name, prompt_type)
        # print("get_examples is :\n", get_examples)
        # ex_source_1 = get_examples["source"]
        # ex_translation_1 = get_examples["translation"]
        # ex_source_2 = get_examples["source_2"]
        # ex_translation_2 = get_examples["translation_2"]
        # prompt_ending = f"\n\n**End of Examples**\n\n**Translation task**\nTranslate below text using the provided instructions and exmaple:\n{text}"

        # prompt = prompt + ex_source_1 + "\n*Output:*\n" + ex_translation_1 + \
        #          "\n\n**Demonstration Example 2:**\n*Input:*\n" + ex_source_2 + \
        #          "\n*Output:*\n" + ex_translation_2 + prompt_ending
    
    # elif dataset_name == "generation" and prompt_type == "instruction":
    #     example_translation_2 = get_examples["translation_2"]
    #     prompt = prompt + example_source + "\n**Demonstration Examples:**\n*Output Example 1:*\n" + example_translation + \
    #              "\n*Output Example 2:*\n" + example_translation_2 + prompt_ending

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
        raise Exception(f"Error in GPT API response: {response_data}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def back_translate_with_gpt(text: str, source_language: str, prompt_type: str, dataset_name: str, PL_name: str) -> str:
    """
    Back-translate text from source language to English using GPT API
    
    Args:
        text: Text to back-translate
        source_language: Source language of the text
        prompt_type: the content type in the part of the dataset for translation
        dataset_name: The target dataset for translation
        
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
    
    # prompt = f"Translate the following {source_language} text back to English:\n\n{text}"
    # prompt = (f"Translate the following {source_language} text of a computer programming task back to English, " \
    #           f"preserving all code exactly as is in the output. " \
    #           f"Do not remove or alter the code. Keep the same structure, formatting, and indentation. " \
    #           f"Do not output any special code block or change any formatting, just give the output with the English translation in a text format.\n\nText to translate:\n{text}")
        
    prompt = (
        f"You are an expert code documentation translator. "
        f"Your task is to translate *ONLY* the natural language parts (comments, docstrings, explanations, descriptive text) "
        f"within the provided programming task descriptions or code snippets from {source_language} back to English. "
        f"\n\n**You must follow these strict rules:** "
        f"\n1.  Translate all human-readable text intended for explanation, like comments (e.g., `#`, `//`, `/* */`, etc.), docstrings, and any descriptive free-text surrounding code. "
        f"\n2.  Translate all human-readable text intended for instructions "
        f"(e.g. instruction: `Write a python function 'def largest_prime_factor(n: int) -> int:' to solve the following problem`, then `Write a python function` and `to solve the following problem` has to be translated into the target language) "
        f"\n3.  **DO NOT** translate any code elements (function/class/variable names, keywords, operators, syntax). "
        f"\n4.  **PRESERVE** the exact original code structure, indentation (spaces or tabs), line breaks, and formatting in the output. "
        f"\n5.  **PRESERVE** examples within docstrings/comments (e.g., `>>> example_code()` or code snippets shown for illustration) without translation. "
        # f"\n6.  **PRESERVE** the exact indentation (spaces or tabs) and line breaks formatting wrapped around the original input in the output. "
        # f"\n7.  Output the programming task descriptions or code snippets with the translated natural language text in a text format."
        f"\n6.  **DO NOT** wrap the output in Markdown code fences (like ```python ... ```) or any other formatting not present in the original input. The output structure must exactly match the input structure."
        # f"\n\n**Demonstration Examples:**\n*Input:*\n"
        f"\n\n**Translation task**\nTranslate below text using the provided instructions:\n{text}"
    )

    # get_examples = get_translation_examples(PL_name, dataset_name, prompt_type)
    # example_translated = get_examples["translation"]
    # example_english = get_examples["source"]
    # prompt_ending = f"\n\n**Translation task**\nTranslate below text using the provided instructions and exmaple:\n{text}"

    # prompt = prompt + example_translated + "\n**Demonstration Examples:**\n*Output:*\n" + example_english + prompt_ending

    if dataset_name == "explanation" and prompt_type == "instruction":
        # extracted_text = extract_instruction(text)
        prompt = f"Translate the only natural language of the following instruction content from {source_language} back to English:\n\n{text}"
    # elif dataset_name == "explanation" and prompt_type == "docstring":
    elif prompt_type == "docstring":
        prompt = (
            f"You are an expert code documentation translator. "
            f"Your task is to translate *ONLY* the natural language human-readable text intended for code explanation (comments, explanations, descriptive text) "
            f"within the provided programming task docstring from {source_language} back to English. "
            f"\n\n**You must follow these strict rules:** "            
            # f"\n1.  Handle specific sections like `Args:`, `Returns:`, `Examples:` by translating the section title and the descriptive text, while preserving the formatting and any associated code/type information. "
            f"\n1.  You must translate everything that was in {source_language} back to English. " 
            # f"\n2.  **DO NOT** translate any actual code elements (function names, class/variable names, keywords, operators, and syntax). "
            # f"\n2.  **DO NOT** translate any actual code elements (e.g. function names, class/variable names, keywords, operators, syntax, lines starting with `>>>`) in the code snippets. "
            f"\n2.  **PRESERVE** the exact original code structure, indentation (spaces or tabs), line breaks, and overall formatting of the code snippets in the output. "
            # f"\n4.  **PRESERVE** examples within the text (e.g., lines starting with `>>>` or code snippets shown for illustration without translation. Keep them exactly as they appear in the input. "
            f"\n3.  **DO NOT** wrap the output in Markdown code fences (like ```python ... ```) or any other formatting not present in the original input. The output structure must exactly match the input structure."
            f"\n4.  You must output the text with the translated English text and all the code elements (lines starting with `>>>`) based on above rules. "
            f"\n\n**Translation task**\nTranslate below text using the provided instructions:\n{text}"
            # f"\n\n**Demonstration Example 1:**\n*Input:*\n"
        )

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
        raise Exception(f"Error in GPT API response: {response_data}")

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
    scorer = BERTScorer(
        model_type="princeton-nlp/sup-simcse-roberta-large", 
        lang="en", 
        rescale_with_baseline=False
    )
    P, R, F1 = scorer.score([back_translated_text], [original_text])
    return F1.item()  # Return the F1 score as a Python float

def calculate_bert_score_rescaling(original_text: str, back_translated_text: str) -> float:
    """
    Calculate BERTScore with manual rescaling from the baseline mean

    Args:
        original_text: Original English text
        back_translated_text: Back-translated English text
        
    Returns:
        BERTScore F1 value
    """
    # Initialize BERTScore without rescaling
    scorer = BERTScorer(
        model_type="princeton-nlp/sup-simcse-roberta-large", 
        lang="en", 
        rescale_with_baseline=False
    )
    P, R, F1 = scorer.score([back_translated_text], [original_text])
    raw_f1 = F1.item()
    
    # Apply rescaling using the calcuated baseline mean
    baseline_mean = 0.699921812238172  # From the baseline file
    rescaled_f1 = (raw_f1 - baseline_mean) / (1 - baseline_mean)
    rescaled_f1 = max(0, min(1, rescaled_f1))
    
    return rescaled_f1

def extract_instruction(text):
    # Pattern to match the instruction line with any programming language
    default_output = r"Provide a concise natural language description \(docstring\) of the programming code in English using at most 500 characters\."
    pattern = r"Provide a concise natural language description \(docstring\) of the ([A-Za-z0-9#+]{1,15}) code in English using at most 500 characters\."
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    if match:
        # Return the full matched instruction
        return match.group(0)
    else:
        return default_output

def replace_instruction(original_text, translated_instruction):
    # Pattern to match the instruction line with any programming language
    pattern = r"Provide a concise natural language description \(docstring\) of the ([A-Za-z0-9#+]{1,15}) code in English using at most 500 characters\."

    # Replace the matched pattern with the translated instruction
    modified_text = re.sub(pattern, translated_instruction, original_text)

    return modified_text

def get_language_code(language_name: str) -> str:
    # Get ISO 639 language code from language name
    with open("languages_tested_ISO639_codes.json", "r") as f:
        lang_codes = json.load(f)
    return lang_codes.get(language_name, "")

def get_translation_examples(programming_lang: str, dataset_name: str, prompt_type: str) -> str:
    # Get source and translation examples based on PL name and prompt type
    with open("prompt_examples.json", "r", encoding='utf-8') as f:
        file = json.load(f)
        return_example = file[programming_lang][dataset_name][prompt_type]
    return return_example

def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace, indentation, and line breaks
    while keeping essential content intact.
    """
    # Remove leading/trailing whitespace on each line
    lines = [line.strip() for line in text.strip().splitlines()]
    # Collapse lines into a single space-separated string
    normalized = ' '.join(line for line in lines if line)
    # Remove multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

# def process_item(item: Dict[str, Any], target_languages: List[str], max_iterations: int = 3, rescaling: bool = False) -> Dict[str, Any]:
def process_item(item: Dict[str, Any], target_languages: List[str], max_iterations: int = 3, rescaling: bool = False, dataset_name: str = "generation") -> Dict[str, Any]:
    """
    Process a single dataset item with translation and back-translation
    
    Args:
        item: Dataset item
        target_languages: List of target languages for translation
        max_iterations: Maximum number of iterations for translation
        rescaling: Enable rescaling with normalized BERTScore
        dataset_name: The name of the split/subset of the dataset processed
        
    Returns:
        Processed item with translations
    """
    global api_call_count, estimated_cost
    
    # item_result = {
    #     "task_id": item["task_id"],
    #     "original": {
    #         "prompt": item["prompt"],
    #         "instruction": item["instruction"],
    #         "docstring": item["docstring"]
    #     },
    #     "translations": {}
    # }

    curr_PL_name = item["task_id"].split('/')[0]

    item_result = {
        "task_id": item["task_id"],
        "prompt": {"en": item["prompt"]},
        "prompt_bertscore": {},
        "canonical_solution": item["canonical_solution"],
        "instruction": {"en": item["instruction"]},
        "instruction_bertscore": {},
        "level": item.get("level", ""),
        "test": item["test"],
        "entry_point": item["entry_point"],
        "signature": item["signature"],
        "docstring": {"en": item["docstring"]},
        "docstring_bertscore": {}
    }
    
    for lang in target_languages:
        # print(f"  Processing language: {lang}")

        lang_code = get_language_code(lang)
        if not lang_code:
            print(f"  Warning: No ISO code found for language {lang}, skipping")
            continue
            
        print(f"  Processing language: {lang} ({lang_code})")

        # lang_result = {
        #     "prompt": {},
        #     "instruction": {},
        #     "docstring": {},
        #     "best_score": 0
        # }

        item_result["prompt"][lang_code] = ""
        item_result["prompt_bertscore"][lang_code] = ""
        item_result["instruction"][lang_code] = ""
        item_result["instruction_bertscore"][lang_code] = ""
        item_result["docstring"][lang_code] = ""
        item_result["docstring_bertscore"][lang_code] = ""
        
        for field in ["prompt", "instruction", "docstring"]:
            print(f"    Processing field: {field}")
            
            # field_result = {
            #     "iterations": []
            # }
            
            best_score = 0
            best_translation = ""
            best_back_translation = ""
            
            for iteration in range(max_iterations):                              
                try:
                    print(f"      Iteration {iteration+1}/{max_iterations}")
                    
                    # Translate to target language
                    print(f"        Translating to {lang}...")
                    start_time = time.time()
                    
                    extracted_text = ""
                    if dataset_name == "explanation" and field == "instruction":
                        extracted_text = extract_instruction(item[field])
                        match = re.search("English", extracted_text)

                        if match:
                            extracted_text = re.sub("English", lang, extracted_text)

                        translated_text = translate_with_gpt(extracted_text, lang, field, dataset_name, curr_PL_name)

                    else: 
                        translated_text = translate_with_gpt(item[field], lang, field, dataset_name, curr_PL_name)
                    # translated_text = translate_with_gpt(item[field], lang)
                    translation_time = time.time() - start_time
                    
                    # Back-translate to English
                    print(f"        Back-translating to English...")
                    start_time = time.time()
                    # back_translated_text = back_translate_with_gpt(translated_text, lang)
                    # if dataset_name == "explanation" and field == "instruction":
                    #     back_translated_text = back_translate_with_gpt(translated_text, lang, field, dataset_name)
                    # else: 
                    back_translated_text = back_translate_with_gpt(translated_text, lang, field, dataset_name, curr_PL_name)
                    back_translation_time = time.time() - start_time
                    
                    # Calculate BERTScore
                    start_time = time.time()
                    # if rescaling:
                    #     print(f"        Calculating BERTScore with rescaling on...")
                    #     score = calculate_bert_score_rescaling(item[field], back_translated_text)
                    # else:
                    #     print(f"        Calculating BERTScore...")
                    #     score = calculate_bert_score(item[field], back_translated_text)

                    if dataset_name == "explanation" and field == "instruction":
                        if rescaling:
                            print(f"        Calculating BERTScore with rescaling on...")
                            score = calculate_bert_score_rescaling(extracted_text, back_translated_text)
                            # score = calculate_bert_score_rescaling(
                            #     normalize_text(extracted_text),
                            #     normalize_text(back_translated_text)
                            # )
                        else:
                            print(f"        Calculating BERTScore...")
                            # score = calculate_bert_score(extracted_text, back_translated_text)
                            score = calculate_bert_score_rescaling(
                                normalize_text(extracted_text),
                                normalize_text(back_translated_text)
                            )
                    else: 
                        if rescaling:
                            print(f"        Calculating BERTScore with rescaling on...")
                            # score = calculate_bert_score_rescaling(item[field], back_translated_text)
                            score = calculate_bert_score_rescaling(
                                normalize_text(item[field]),
                                normalize_text(back_translated_text)
                            )
                        else:
                            print(f"        Calculating BERTScore...")
                            # score = calculate_bert_score(item[field], back_translated_text)
                            score = calculate_bert_score_rescaling(
                                normalize_text(item[field]),
                                normalize_text(back_translated_text)
                            )

                    bertscore_time = time.time() - start_time
                    
                    print(f"        Score: {score:.4f}")

                    if dataset_name == "explanation" and field == "instruction":
                        translated_text = replace_instruction(item[field], translated_text)
                        back_translated_text = replace_instruction(item[field], back_translated_text)
                    
                    # iteration_result = {
                    #     "translated_text": translated_text,
                    #     "back_translated_text": back_translated_text,
                    #     "score": score,
                    #     "time_taken": {
                    #         "translation_time": translation_time,
                    #         "back_translation_time": back_translation_time,
                    #         "bertscore_times": bertscore_time
                    #     }
                    # }
                    
                    # field_result["iterations"].append(iteration_result)
                    
                    if score > best_score:
                        best_score = score
                        best_translation = translated_text
                        item_result[f"{field}_bertscore"][lang_code] = str(score)
                        item_result[field][lang_code] = best_translation
                    
                    if rescaling:
                        if score > 0.83:
                            print(f"        Score above rescaled threshold (0.83). Stopping iterations.")
                            break
                    else:
                        if score > 0.95:
                            print(f"        Score above baseline threshold (0.95). Stopping iterations.")
                            break
                    
                    # Add a delay to avoid API rate limits
                    sleep_time = random.uniform(1, 2.0)
                    print(f"        Sleeping for {sleep_time:.2f}s to avoid rate limits...")
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"        Error: {str(e)}")
                    # field_result["iterations"].append({"error": str(e)})
                    print(f"        Sleeping for 5s after error...")
                    time.sleep(5)  # Longer delay after an error
            
            # Store the best results for this field
        #     field_result["best_translation"] = best_translation
        #     field_result["best_back_translation"] = best_back_translation
        #     field_result["best_score"] = best_score
            
        #     lang_result[field] = field_result
            
        #     if best_score > lang_result["best_score"]:
        #         lang_result["best_score"] = best_score
        
        # item_result["translations"][lang] = lang_result

    
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
    subset_name: str,
    start_idx: int, 
    end_idx: int, 
    target_languages: List[str], 
    max_iterations: int = 3, 
    save_interval: int = 1, 
    resume: bool = False,
    rescaling: bool = False
) -> List[Dict[str, Any]]:
    """
    Process a subset of the dataset with translation and back-translation
    
    Args:
        dataset: HuggingFace dataset to be processed
        subset_name: The name of the split/subset of the dataset processed
        start_idx: Start index for processing
        end_idx: End index for processing
        target_languages: List of target languages for translation
        max_iterations: Maximum number of iterations for translation
        save_interval: Interval for saving intermediate results
        resume: Whether to resume from previous processing
        rescaling: Enable rescaling with normalized BERTScore
        
    Returns:
        Processed data with translations
    """
    global api_call_count, estimated_cost
    
    # Determine dataset name for file naming
    dataset_name = subset_name
    
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
        item_result = process_item(item, target_languages, max_iterations, rescaling, dataset_name)
        processed_data.append(item_result)
        
        # Save intermediate results at specified intervals
        # if (idx - current_start_idx + 1) % save_interval == 0 or idx == end_idx:
        #     output_file = f"{dataset_name}_intermediate_translations_{start_idx}_to_{idx}.json"
        #     print(f"Saving intermediate results to {output_file}...")
        #     with open(output_file, "w", encoding='utf-8') as f:
        #         json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        #     print(f"Current statistics:")
        #     print(f"  Total API calls: {api_call_count}")
        #     print(f"  Estimated cost: ${estimated_cost:.2f}")
    
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
    parser.add_argument('--rescaling', action='store_true',
                        help='Enable rescaling with normalized BERTScore')
    
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

    subset_name = args.dataset
    
    try:
        results = process_dataset(
            ds,
            subset_name,
            args.start_idx, 
            args.end_idx, 
            args.languages, 
            args.max_iterations,
            args.save_interval,
            args.resume,
            args.rescaling
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