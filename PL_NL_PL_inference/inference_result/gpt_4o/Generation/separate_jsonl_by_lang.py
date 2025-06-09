import json
import os
from collections import defaultdict

# Function to read JSONL files
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

# Process the translations file
def process_translations(translations_data):
    task_data = {}
    for entry in translations_data:
        task_id = entry['task_id']
        task_data[task_id] = {
            'canonical_solution': entry.get('canonical_solution', ''),
            'test': entry.get('test', ''),
            'entry_point': entry.get('entry_point', ''),
            'signature': entry.get('signature', ''),
            'level': entry.get('level', ''),
            'prompt': entry.get('prompt', {}),
            'instruction': entry.get('instruction', {}),
            'docstring': entry.get('docstring', {})
        }
    return task_data

# Process the results file
def process_results(results_data):
    # Organize by task_id and language
    organized_results = defaultdict(dict)
    
    for entry in results_data:
        task_id = entry['task_id']
        language = entry['language']
        generated_text = entry['generated_text']
        
        organized_results[task_id][language] = generated_text
    
    return organized_results

# Combine data and create output files
def create_output_files(translations, results):
    # Get all unique language codes from translations
    all_languages = set()
    for task_id, task_data in translations.items():
        for lang in task_data['prompt'].keys():
            all_languages.add(lang)
    
    # Create output directory if it doesn't exist
    os.makedirs('output_files', exist_ok=True)
    
    # Create a file for each language
    language_files = {}
    for lang in all_languages:
        file_path = f'output_files/{lang}_results.jsonl'
        language_files[lang] = open(file_path, 'w', encoding='utf-8')
    
    # Process each task
    for task_id, task_data in translations.items():
        for lang in all_languages:
            # Skip if language not available for this task
            if lang not in task_data['prompt']:
                continue
            
            # Skip if no results for this task-language pair
            if task_id not in results or lang not in results[task_id]:
                continue
            
            # Create output entry
            output_entry = {
                'task_id': task_id,
                'prompt': task_data['prompt'].get(lang, ''),
                'canonical_solution': task_data['canonical_solution'],
                'test': task_data['test'],
                'entry_point': task_data['entry_point'],
                'signature': task_data['signature'],
                'docstring': task_data['docstring'].get(lang, ''),
                'instruction': task_data['instruction'].get(lang, ''),
                'raw_generation': [results[task_id][lang]]
            }
            
            # Write to appropriate language file
            language_files[lang].write(json.dumps(output_entry, ensure_ascii=False) + '\n')
    
    # Close all files
    for file in language_files.values():
        file.close()
    
    return list(language_files.keys())

def main():
    # File paths
    translations_file = 'deepseek_masked_C_to_JS_explanation_to_generation_translation.jsonl'
    results_file = 'gpt4o_masked_C_to_JavaScript_generation_hard_results.jsonl'

    current_directory = os.path.dirname(os.path.abspath(__file__))
    translations_file_filepath = os.path.join(current_directory, translations_file)
    results_file_filepath = os.path.join(current_directory, results_file)
    
    # Read files
    translations_data = read_jsonl(translations_file_filepath)
    results_data = read_jsonl(results_file_filepath)
    
    # Process data
    translations = process_translations(translations_data)
    results = process_results(results_data)
    
    # Create output files
    languages = create_output_files(translations, results)
    
    print(f"Created output files for {len(languages)} languages: {', '.join(languages)}")

if __name__ == "__main__":
    main()