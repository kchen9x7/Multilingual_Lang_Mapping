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
# def translate_with_gpt(text: str, target_language: str) -> str:
# def translate_with_gpt(text: str, target_language: str, prompt_type: str) -> str:
def translate_with_gpt(text: str, target_language: str, prompt_type: str, dataset_name: str) -> str:
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
    # global api_call_count, estimated_cost
    
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
        f"(e.g. If we have Python instruction: `Write a python function 'def largest_prime_factor(n: int) -> int:' to solve the following problem`, then `Write a python function` and `to solve the following problem` has to be translated into the target language) "
        f"\n3.  **DO NOT** translate any code elements (function/class/variable names, keywords, operators, syntax). "
        f"\n4.  **PRESERVE** the exact original code structure, indentation (spaces or tabs), line breaks, and formatting in the output. "
        f"\n5.  **PRESERVE** examples within docstrings/comments (e.g., `>>> example_code()` or code snippets shown for illustration) without translation. "
        f"\n6.  **DO NOT** wrap the output in Markdown code fences (like ```python ... ```) or any other formatting not present in the original input. The output structure must exactly match the input structure."
        f"\n7.  You **MUST** output the text with the translated natural language text AND ALL the code elements (e.g. lines of code starting with `>>>`, in-between natural language text, or after comment blocks) based on above rules. "
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
            f"\n2.  **DO NOT** translate any actual code elements (e.g. function names, class/variable names, keywords, operators, syntax, lines starting with `>>>`) in the code snippets. "
            f"\n3.  **PRESERVE** the exact original code structure, indentation (spaces or tabs), line breaks, and overall formatting of the code snippets in the output. "
            f"\n4.  **DO NOT** wrap the output in Markdown code fences (like ```python ... ```) or any other formatting not present in the original input. The output structure must exactly match the input structure."
            f"\n5.  You **MUST** output the text with the translated natural language text AND ALL the code elements (e.g. lines of code starting with `>>>`, in-between natural language text, or after comment blocks) based on above rules. "
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
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    response_data = response.json()
    
    if "choices" in response_data and len(response_data["choices"]) > 0:
        output_text = response_data["choices"][0]["message"]["content"]
        # Estimate output tokens and cost at $0.01/1K tokens
        output_tokens = len(output_text) / 4
        # estimated_cost += (output_tokens / 1000) * 0.01
        return output_text
    else:
        raise Exception(f"Error in GPT API response: {response_data}")


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
        f"(e.g. If we have Python instruction: `Write a python function 'def largest_prime_factor(n: int) -> int:' to solve the following problem`, then `Write a python function` and `to solve the following problem` has to be translated from {source_language} to English) "
        f"\n3.  **DO NOT** translate any code elements (function/class/variable names, keywords, operators, syntax). "
        f"\n4.  **PRESERVE** the exact original code structure, indentation (spaces or tabs), line breaks, and formatting in the output. "
        f"\n5.  **PRESERVE** examples within docstrings/comments (e.g., `>>> example_code()` or code snippets shown for illustration) without translation. "
        f"\n6.  **DO NOT** wrap the output in Markdown code fences (like ```python ... ```) or any other formatting not present in the original input. The output structure must exactly match the input structure."
        f"\n7.  You **MUST** output the text with the translated English text AND ALL the code elements (e.g. lines of code starting with `>>>`, in-between natural language text, or after comment blocks) based on above rules. "
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
            f"\n2.  **PRESERVE** the exact original code structure, indentation (spaces or tabs), line breaks, and overall formatting of the code snippets in the output. "
            f"\n3.  **DO NOT** wrap the output in Markdown code fences (like ```python ... ```) or any other formatting not present in the original input. The output structure must exactly match the input structure."
            f"\n4.  You **MUST** output the text with the translated English text AND ALL the code elements (e.g. lines of code starting with `>>>`, in-between natural language text, or after comment blocks) based on above rules. "
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
    # source_docstring = "Calculates the monthly repayment amount for an equal principal repayment loan.\nIn this repayment model, each month's repayment amount consists of two parts:\n- A constant principal payment, which is the total loan amount divided by the total number of months.\n- The interest payment, which is the outstanding loan amount multiplied by the monthly interest rate.\nInput:\n- $loanAmount (int): The total loan amount in ten-thousands.\n- $monthlyInterestRate (float): The monthly interest rate.\n- $totalMonths (int): The total number of months for loan repayment.\nOutput: Returns the first month's repayment amount as an integer (in Yuan). Discard the decimal point and do not round\nExample: calculateMonthlyRepayment(500, 0.004, 360) should return 33889."
    # translated_docstring = "मासिक पुनर्भुगतान राशि की गणना एक समान मूलधन पुनर्भुगतान ऋण के लिए करता है। इस पुनर्भुगतान मॉडल में, प्रत्येक महीने की पुनर्भुगतान राशि दो भागों में होती है:\n- एक स्थिर मूलधन भुगतान, जो कुल ऋण राशि को कुल महीनों की संख्या से विभाजित करता है।\n- ब्याज भुगतान, जो बकाया ऋण राशि को मासिक ब्याज दर से गुणा करता है।\nइनपुट:\n- $loanAmount (int): कुल ऋण राशि दस-हजारों में।\n- $monthlyInterestRate (float): मासिक ब्याज दर।\n- $totalMonths (int): ऋण पुनर्भुगतान के लिए कुल महीनों की संख्या।\nआउटपुट: पहले महीने की पुनर्भुगतान राशि को पूर्णांक (युआन में) के रूप में लौटाता है। दशमलव बिंदु को हटा दें और राउंड न करें।\nउदाहरण: calculateMonthlyRepayment(500, 0.004, 360) को 33889 लौटाना चाहिए।"

    source_prompt = " /*\n  You're given a list of deposit and withdrawal operations on a bank account that starts with\n  zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n  at that point function should return True. Otherwise it should return False.\n  \n */\n \n use std::{slice::Iter, cmp::{max, self}, mem::replace, collections::{HashSet, HashMap}, ops::Index, ascii::AsciiExt};\n use rand::Rng;\n use regex::Regex;\n use md5;\n use std::any::{Any, TypeId};\n \n fn below_zero(operations:Vec<i32>) -> bool{\n "
    manually_fixed_translated_prompt = " /*\n  你有一个关于银行账户的存款和取款操作列表，该账户从零余额开始。你的任务是检测账户余额是否在任何时候低于零，\n  如果是这样，函数应该返回True。否则，它应该返回False。\n  \n */\n \n use std::{slice::Iter, cmp::{max, self}, mem::replace, collections::{HashSet, HashMap}, ops::Index, ascii::AsciiExt};\n use rand::Rng;\n use regex::Regex;\n use md5;\n use std::any::{Any, TypeId};\n \n fn below_zero(operations:Vec<i32>) -> bool{"

    # manually_fixed_translated_prompt = " /*\n  ලබා දී ඇති සංඛ්‍යා ලැයිස්තුවේ, ලබා දී ඇති සීමාවට වඩා ආසන්නව ඇති සංඛ්‍යා යුගල කිසිවක් තිබේදැයි පරීක්ෂා කරන්න.\n  \n */\n \n use std::{slice::Iter, cmp::{max, self}, mem::replace, collections::{HashSet, HashMap}, ops::Index, ascii::AsciiExt};\n use rand::Rng;\n use regex::Regex;\n use md5;\n use std::any::{Any, TypeId};\n \n fn has_close_elements(numbers:Vec<f32>, threshold: f32) -> bool{\n "
    '''
    modified_instruction_translation: translated_instruction, but removed translation for below sentence in English:
    "Write a python function 'def largest_prime_factor(n: int) -> int:' to solve the following problem:\n"
    '''
    # source_instruction = "Write a Kotlin function `fun findPrimePairs(maxNumber: Int): List<Pair<Int, Int>>` to solve the following problem:\nFinds all prime pairs where each prime is less than or equal to a given number and the pair differs by 2.\nA prime pair is defined as two prime numbers where the difference between them is exactly 2.\nExample:\n>>> findPrimePairs(10)\n[(3, 5), (5, 7)]\n>>> findPrimePairs(100)\n[(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43), (59, 61), (71, 73)]"
    # modified_instruction_translation = "Գրեք Kotlin ֆունկցիա `fun findPrimePairs(maxNumber: Int): List<Pair<Int, Int>>` հետևյալ խնդիրը լուծելու համար:\nԳտնում է բոլոր պարզ զույգերը, որտեղ յուրաքանչյուր պարզ թիվ փոքր կամ հավասար է տրված թվին, և զույգը տարբերվում է 2-ով:\nՊարզ զույգը սահմանվում է որպես երկու պարզ թվեր, որոնց միջև տարբերությունը ճիշտ 2 է:\nՕրինակ:\n>>> findPrimePairs(10)\n[(3, 5), (5, 7)]\n>>> findPrimePairs(100)\n[(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43), (59, 61), (71, 73)]"
    

    lang = "Chinese"
    # lang = "Hausa"
    # lang = "Bulgarian"
    prompt_type = "prompt"
    # prompt_type = "docstring"
    # prompt_type = "instruction"
    # dataset_name = "explanation"
    dataset_name = "generation"
    # back_translated_docstring = back_translate_with_gpt(translated_docstring, lang, prompt_type, dataset_name)
    # back_translated_instruction = back_translate_with_gpt(modified_instruction_translation, lang, prompt_type, dataset_name)
    translated_prompt = translate_with_gpt(source_prompt, lang, prompt_type, dataset_name)
    back_translated_prompt = back_translate_with_gpt(translated_prompt, lang, prompt_type, dataset_name)
    back_translated_prompt_manual_fix = back_translate_with_gpt(manually_fixed_translated_prompt, lang, prompt_type, dataset_name)

    '''
    NON-Problematic case from our experiment
    '''
    # translated_instruction = "编写一个Python函数 'def largest_prime_factor(n: int) -> int:' 来解决以下问题：\n\n    找出给定正整数的最大质因数。\n    \n    假设该整数是两个不同质数的乘积。\n    该函数从最小的质数（2）开始迭代可能的因数，并检查它们是否是 'n' 的因数。\n    如果找到一个因数，函数返回 'n' 除以该因数的结果，即较大的质因数。\n    如果在 'n' 的平方根范围内没有找到因数，那么 'n' 本身就是一个质数，并作为最大质因数返回。\n    \n    参数:\n    n (int): 要分解的正整数，它是两个不同质数的乘积。\n    \n    返回:\n    int: 'n' 的两个质因数中较大的一个。\n    \n    示例:\n    >>> largest_prime_factor(21)\n    7\n    >>> largest_prime_factor(15)\n    5"
    # translated_instruction = "एक पायथन फ़ंक्शन 'def largest_prime_factor(n: int) -> int:' लिखें ताकि निम्नलिखित समस्या का समाधान किया जा सके:\n\n    दिए गए धनात्मक पूर्णांक का सबसे बड़ा अभाज्य गुणनखंड खोजें।\n    \n    यह पूर्णांक ठीक दो भिन्न अभाज्य संख्याओं का गुणनफल माना जाता है।\n    फ़ंक्शन संभावित गुणनखंडों के माध्यम से सबसे छोटे अभाज्य संख्या (2) से शुरू होकर \n    यह जांचता है कि क्या वे 'n' के गुणनखंड हैं। यदि कोई गुणनखंड पाया जाता है, तो \n    फ़ंक्शन 'n' को इस गुणनखंड से विभाजित कर देता है, जो कि बड़ा अभाज्य गुणनखंड है। \n    यदि 'n' के वर्गमूल तक कोई गुणनखंड नहीं पाया जाता है, तो 'n' स्वयं एक अभाज्य \n    संख्या है और इसे सबसे बड़े अभाज्य गुणनखंड के रूप में लौटाया जाता है।\n    \n    Args:\n    n (int): धनात्मक पूर्णांक जिसे गुणनखंडित करना है, जो दो भिन्न अभाज्य संख्याओं का गुणनफल है।\n    \n    Returns:\n    int: 'n' के दो अभाज्य गुणनखंडों में से बड़ा।\n    \n    उदाहरण:\n    >>> largest_prime_factor(21)\n    7\n    >>> largest_prime_factor(15)\n    5"
    
    # lang = "Chinese"'
    # lang = "Hindi"
    # prompt_type = "docstring"
    # dataset_name = "generation"
    # back_translated_instruction = back_translate_with_gpt(translated_instruction, lang, prompt_type, dataset_name)

    # score_docstring = calculate_bert_score_rescaling(source_docstring, back_translated_docstring)

    # score_instruction = calculate_bert_score_rescaling(
    #     normalize_text(source_instruction),
    #     normalize_text(back_translated_instruction)
    # )

    # score_docstring = calculate_bert_score_rescaling(
    #     normalize_text(source_docstring),
    #     normalize_text(back_translated_docstring)
    # )

    score_prompt = calculate_bert_score_rescaling(
        normalize_text(source_prompt),
        normalize_text(back_translated_prompt)
    )

    score_prompt_manual_fix = calculate_bert_score_rescaling(
        normalize_text(source_prompt),
        normalize_text(back_translated_prompt_manual_fix)
    )

    # score_instruction = calculate_bert_score_rescaling(source_instruction, back_translated_instruction)

    '''
    Comment out any print statement based on the testing scenario
    '''
    print("\nsource_prompt is: \n", source_prompt)

    print("\ntranslated_prompt based on GPT translation is: \n", translated_prompt)
    print("\nback_translated_prompt based on GPT translation is: \n", back_translated_prompt)
    print("\nThe BERTScore of this prompt translation example based on GPT translation is: ", score_prompt)

    print("\nback_translated_prompt based on manually_fixed_translated_prompt is: \n", back_translated_prompt_manual_fix)
    print("\nThe BERTScore of this prompt translation example after manual fixing is: \n", score_prompt_manual_fix)

    # print("\nsource_docstring is: \n", source_docstring)
    # print("\nback_translated_docstring is: \n", back_translated_docstring)
    # print("\nThe BERTScore of this docstring translation example is: \n", score_docstring)

    # print("\nsource_instruction is: \n", source_instruction)
    # print("\nback_translated_instruction is: \n", back_translated_instruction)
    # print("\nThe BERTScore of this instruction translation example is: \n", score_instruction)



if __name__ == "__main__":
    main()