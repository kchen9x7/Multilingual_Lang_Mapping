import json
import copy
import os

def merge_language_data(file1_path, file2_path, output_path):
    
    fields_to_merge = {"prompt", "instruction", "docstring"}
    excluded_lang_code = "en"
    suffix_to_remove = "_bertscore"

    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
            print(f"Read {len(data1)} entries from '{file1_path}'")

        with open(file2_path, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
            print(f"Read {len(data2)} entries from '{file2_path}'")

        if len(data1) != len(data2):
            print(f"Error: Files have different number of entries ({len(data1)} vs {len(data2)}). Cannot merge.")
            return

        merged_data = copy.deepcopy(data1)

        for i in range(len(merged_data)):
            entry1 = merged_data[i] # Entry from file1 (target for merge)
            entry2 = data2[i]      # Entry from file2 (source of merge data)

            for field in fields_to_merge:
                if field in entry1 and field in entry2 and isinstance(entry2.get(field), dict):
                    source_langs = entry2[field]
                    if not isinstance(entry1.get(field), dict):
                         entry1[field] = {}

                    for lang_code, value in source_langs.items():
                        if lang_code != excluded_lang_code:
                            entry1[field][lang_code] = value
                elif field in entry2 and isinstance(entry2.get(field), dict) and field not in entry1:
                     source_langs = entry2[field]
                     entry1[field] = {}
                     for lang_code, value in source_langs.items():
                         if lang_code != excluded_lang_code:
                             entry1[field][lang_code] = value

        final_cleaned_data = []
        for entry in merged_data:
            cleaned_entry = {}
            for key, value in entry.items():
                if not key.endswith(suffix_to_remove):
                    cleaned_entry[key] = value
            final_cleaned_data.append(cleaned_entry)

        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(final_cleaned_data, f_out, ensure_ascii=False, indent=2)

        print(f"\nSuccessfully merged data and removed '{suffix_to_remove}' fields.")
        print(f"Output saved to '{output_path}'")

    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}. Please ensure both JSON files are in the correct path.")
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from one of the files - {e}. Please check file contents.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    file1_name = "generation_translation_fixed_Python_Kaige_extracted.json"
    file2_name = "generation_translation_results_Python_Sangmitra.json"
    output_name = "merged_python_translations.json"

    # current_directory = os.getcwd()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file1_path = os.path.join(current_directory, file1_name)
    file2_path = os.path.join(current_directory, file2_name)
    output_path = os.path.join(current_directory, output_name)

    merge_language_data(file1_path, file2_path, output_path)
