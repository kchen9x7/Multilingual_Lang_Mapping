import json
from typing import Dict, List, Any

def merge_json_files(file1_path: str, file2_path: str, output_path: str) -> None:

    MERGE_FIELDS = {
        "prompt",
        "prompt_bertscore",
        "instruction",
        "instruction_bertscore",
        "docstring",
        "docstring_bertscore"
    }
    
    with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    
    if len(data1) != len(data2):
        raise ValueError("Files have different number of tasks")
    
    merged_data = []
    for task1, task2 in zip(data1, data2):
        merged_task = task1.copy()
        
        for field in MERGE_FIELDS:
            if field in task1 or field in task2:
                merged_field = task1.get(field, {})
                
                if isinstance(merged_field, dict) and field in task2:
                    for lang_code, lang_value in task2[field].items():
                        merged_field[lang_code] = lang_value
                
                merged_task[field] = merged_field
        
        merged_data.append(merged_task)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

def delete_language_code(input_path: str, output_path: str, language_code: str) -> None:
    
    LANGUAGE_FIELDS = {
        "prompt",
        "prompt_bertscore",
        "instruction",
        "instruction_bertscore",
        "docstring",
        "docstring_bertscore"
    }
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified_data = []
    for task in data:
        modified_task = task.copy()
        for field in LANGUAGE_FIELDS:
            if field in modified_task and isinstance(modified_task[field], dict):
                if language_code in modified_task[field]:
                    del modified_task[field][language_code]
        modified_data.append(modified_task)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # merge_json_files("explanation_translation_results_C#_part1.json", "explanation_translation_results_C#_part2.json", "merged_output.json")
    
    delete_language_code("generation_translation_results_C#_kaige.json", "output_no_am.json", "am")
    pass