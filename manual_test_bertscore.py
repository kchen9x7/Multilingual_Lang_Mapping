import os
# import time
# import json
# import random
# import argparse
# import glob
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union
# from transformers import AutoModel, AutoTokenizer

import requests
import numpy as np
from bert_score import BERTScorer
from tenacity import retry, stop_after_attempt, wait_exponential

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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def back_translate_with_gpt(text: str, source_language: str, prompt_type: str, dataset_name: str) -> str:
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
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    response_data = response.json()
    
    if "choices" in response_data and len(response_data["choices"]) > 0:
        output_text = response_data["choices"][0]["message"]["content"]
        output_tokens = len(output_text) / 4
        # estimated_cost += (output_tokens / 1000) * 0.01
        
        return output_text
    else:
        raise Exception(f"Error in GPT API response: {response_data}")
    
    
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


def main():
    '''
    Problematic case from our experiment
    '''
    source_docstring = "\n    Find the largest prime factor of a given positive integer.\n    \n    The integer is assumed to be the product of exactly two distinct prime numbers. \n    The function iterates through potential factors starting from the smallest prime (2) \n    and checks if they are a factor of 'n'. If a factor is found, the function returns \n    the division of 'n' by this factor, which is the larger prime factor. If no factors \n    are found up to the square root of 'n', then 'n' itself is a prime number and is \n    returned as the largest prime factor.\n    \n    Args:\n    n (int): The positive integer to factorize, which is the product of two distinct primes.\n    \n    Returns:\n    int: The larger of the two prime factors of 'n'.\n    \n    Examples:\n    >>> largest_prime_factor(21)\n    7\n    >>> largest_prime_factor(15)\n    5\n    "
    # translated_docstring = "找到给定正整数的最大质因数。\n\n该整数被假设为恰好是两个不同质数的乘积。函数从最小的质数（2）开始迭代潜在因数，并检查它们是否是'n'的因数。如果找到一个因数，函数返回'n'除以该因数的结果，即较大的质因数。如果在'n'的平方根范围内没有找到因数，则'n'本身是一个质数，并返回作为最大的质因数。\n\n参数：\nn (int): 要分解的正整数，它是两个不同质数的乘积。\n\n返回：\nint: 'n'的两个质因数中较大的一个。\n\n示例：\n>>> largest_prime_factor(21)\n7\n>>> largest_prime_factor(15)\n5"
    translated_docstring = "    दिए गए धनात्मक पूर्णांक का सबसे बड़ा अभाज्य गुणनखंड खोजें।\n    \n    पूर्णांक को ठीक दो भिन्न अभाज्य संख्याओं के गुणनफल के रूप में माना जाता है।\n    फ़ंक्शन संभावित गुणनखंडों के माध्यम से सबसे छोटे अभाज्य (2) से शुरू होकर \n    पुनरावृत्ति करता है और जांचता है कि क्या वे 'n' के गुणनखंड हैं। यदि कोई गुणनखंड पाया जाता है, \n    तो फ़ंक्शन 'n' को इस गुणनखंड से विभाजित कर देता है, जो बड़ा अभाज्य गुणनखंड होता है। \n    यदि 'n' के वर्गमूल तक कोई गुणनखंड नहीं मिलता है, तो 'n' स्वयं एक अभाज्य संख्या है और \n    इसे सबसे बड़े अभाज्य गुणनखंड के रूप में लौटाया जाता है।\n    \n    Args:\n    n (int): धनात्मक पूर्णांक जिसे गुणनखंडित करना है, जो दो भिन्न अभाज्य संख्याओं का गुणनफल है।\n    \n    Returns:\n    int: 'n' के दो अभाज्य गुणनखंडों में से बड़ा।\n    \n    उदाहरण:\n    >>> largest_prime_factor(21)\n    7\n    >>> largest_prime_factor(15)\n    5"

    '''
    modified_instruction_translation: translated_instruction, but removed translation for below sentence in English:
    "Write a python function 'def largest_prime_factor(n: int) -> int:' to solve the following problem:\n"
    '''
    # modified_instruction_translation = "找出给定正整数的最大质因数。\n    \n    假设该整数是两个不同质数的乘积。\n    该函数从最小的质数（2）开始迭代可能的因数，并检查它们是否是 'n' 的因数。\n    如果找到一个因数，函数返回 'n' 除以该因数的结果，即较大的质因数。\n    如果在 'n' 的平方根范围内没有找到因数，那么 'n' 本身就是一个质数，并作为最大质因数返回。\n    \n    参数:\n    n (int): 要分解的正整数，它是两个不同质数的乘积。\n    \n    返回:\n    int: 'n' 的两个质因数中较大的一个。\n    \n    示例:\n    >>> largest_prime_factor(21)\n    7\n    >>> largest_prime_factor(15)\n    5"
    modified_instruction_translation = "दिए गए धनात्मक पूर्णांक का सबसे बड़ा अभाज्य गुणनखंड खोजें।\n    \n    यह पूर्णांक ठीक दो भिन्न अभाज्य संख्याओं का गुणनफल माना जाता है।\n    फ़ंक्शन संभावित गुणनखंडों के माध्यम से सबसे छोटे अभाज्य संख्या (2) से शुरू होकर \n    यह जांचता है कि क्या वे 'n' के गुणनखंड हैं। यदि कोई गुणनखंड पाया जाता है, तो \n    फ़ंक्शन 'n' को इस गुणनखंड से विभाजित कर देता है, जो कि बड़ा अभाज्य गुणनखंड है। \n    यदि 'n' के वर्गमूल तक कोई गुणनखंड नहीं पाया जाता है, तो 'n' स्वयं एक अभाज्य \n    संख्या है और इसे सबसे बड़े अभाज्य गुणनखंड के रूप में लौटाया जाता है।\n    \n    Args:\n    n (int): धनात्मक पूर्णांक जिसे गुणनखंडित करना है, जो दो भिन्न अभाज्य संख्याओं का गुणनफल है।\n    \n    Returns:\n    int: 'n' के दो अभाज्य गुणनखंडों में से बड़ा।\n    \n    उदाहरण:\n    >>> largest_prime_factor(21)\n    7\n    >>> largest_prime_factor(15)\n    5"

    # lang = "Chinese"'
    lang = "Hindi"
    prompt_type = "docstring"
    dataset_name = "generation"
    back_translated_docstring = back_translate_with_gpt(translated_docstring, lang, prompt_type, dataset_name)
    # back_translated_docstring = back_translate_with_gpt(modified_instruction_translation, lang, prompt_type, dataset_name)

    '''
    NON-Problematic case from our experiment
    '''
    source_instruction = "Write a python function 'def largest_prime_factor(n: int) -> int:' to solve the following problem:\n\n    Find the largest prime factor of a given positive integer.\n    \n    The integer is assumed to be the product of exactly two distinct prime numbers. \n    The function iterates through potential factors starting from the smallest prime (2) \n    and checks if they are a factor of 'n'. If a factor is found, the function returns \n    the division of 'n' by this factor, which is the larger prime factor. If no factors \n    are found up to the square root of 'n', then 'n' itself is a prime number and is \n    returned as the largest prime factor.\n    \n    Args:\n    n (int): The positive integer to factorize, which is the product of two distinct primes.\n    \n    Returns:\n    int: The larger of the two prime factors of 'n'.\n    \n    Examples:\n    >>> largest_prime_factor(21)\n    7\n    >>> largest_prime_factor(15)\n    5\n    "
    # translated_instruction = "编写一个Python函数 'def largest_prime_factor(n: int) -> int:' 来解决以下问题：\n\n    找出给定正整数的最大质因数。\n    \n    假设该整数是两个不同质数的乘积。\n    该函数从最小的质数（2）开始迭代可能的因数，并检查它们是否是 'n' 的因数。\n    如果找到一个因数，函数返回 'n' 除以该因数的结果，即较大的质因数。\n    如果在 'n' 的平方根范围内没有找到因数，那么 'n' 本身就是一个质数，并作为最大质因数返回。\n    \n    参数:\n    n (int): 要分解的正整数，它是两个不同质数的乘积。\n    \n    返回:\n    int: 'n' 的两个质因数中较大的一个。\n    \n    示例:\n    >>> largest_prime_factor(21)\n    7\n    >>> largest_prime_factor(15)\n    5"
    translated_instruction = "एक पायथन फ़ंक्शन 'def largest_prime_factor(n: int) -> int:' लिखें ताकि निम्नलिखित समस्या का समाधान किया जा सके:\n\n    दिए गए धनात्मक पूर्णांक का सबसे बड़ा अभाज्य गुणनखंड खोजें।\n    \n    यह पूर्णांक ठीक दो भिन्न अभाज्य संख्याओं का गुणनफल माना जाता है।\n    फ़ंक्शन संभावित गुणनखंडों के माध्यम से सबसे छोटे अभाज्य संख्या (2) से शुरू होकर \n    यह जांचता है कि क्या वे 'n' के गुणनखंड हैं। यदि कोई गुणनखंड पाया जाता है, तो \n    फ़ंक्शन 'n' को इस गुणनखंड से विभाजित कर देता है, जो कि बड़ा अभाज्य गुणनखंड है। \n    यदि 'n' के वर्गमूल तक कोई गुणनखंड नहीं पाया जाता है, तो 'n' स्वयं एक अभाज्य \n    संख्या है और इसे सबसे बड़े अभाज्य गुणनखंड के रूप में लौटाया जाता है।\n    \n    Args:\n    n (int): धनात्मक पूर्णांक जिसे गुणनखंडित करना है, जो दो भिन्न अभाज्य संख्याओं का गुणनफल है।\n    \n    Returns:\n    int: 'n' के दो अभाज्य गुणनखंडों में से बड़ा।\n    \n    उदाहरण:\n    >>> largest_prime_factor(21)\n    7\n    >>> largest_prime_factor(15)\n    5"
    
    # lang = "Chinese"'
    lang = "Hindi"
    prompt_type = "docstring"
    dataset_name = "generation"
    back_translated_instruction = back_translate_with_gpt(translated_instruction, lang, prompt_type, dataset_name)

    # score_docstring = calculate_bert_score_rescaling(source_docstring, back_translated_docstring)
    score_docstring = calculate_bert_score_rescaling(
        normalize_text(source_docstring),
        normalize_text(back_translated_docstring)
    )

    score_instruction = calculate_bert_score_rescaling(source_instruction, back_translated_instruction)

    '''
    Comment out any print statement based on what you are not testing
    '''
    print("\nsource_docstring is: \n", source_docstring)
    print("\nback_translated_docstring is: \n", back_translated_docstring)
    print("\nThe BERTScore of this docstring translation example is: ", score_docstring)

    # print("\nsource_instruction is: \n", source_instruction)
    # print("\nback_translated_instruction is: ", back_translated_instruction)
    # print("\nThe BERTScore of this instruction translation example is: \n", score_instruction)



if __name__ == "__main__":
    main()