import json
import os

def filter_language_data(input_filepath, output_filepath, target_languages):
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)

        filtered_data = []
        language_fields = {
            "prompt", "prompt_bertscore",
            "instruction", "instruction_bertscore",
            "docstring", "docstring_bertscore"
        }

        for entry in data:
            filtered_entry = {}
            for key, value in entry.items():
                if key in language_fields and isinstance(value, dict):
                    filtered_language_dict = {
                        lang_code: lang_data
                        for lang_code, lang_data in value.items()
                        if lang_code in target_languages
                    }
                    if filtered_language_dict:
                        filtered_entry[key] = filtered_language_dict
                else:
                    filtered_entry[key] = value
            filtered_data.append(filtered_entry)

        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            json.dump(filtered_data, f_out, ensure_ascii=False, indent=2)

        print(f"Successfully processed '{input_filepath}' -> '{output_filepath}'")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_filepath}'. Skipping.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{input_filepath}': {e}")

if __name__ == "__main__":
    languages_to_keep = {
        "sq",  # Albanian
        "hy",  # Armenian
        "bn",  # Bengali
        "bg",  # Bulgarian
        "zh",  # Chinese
        "en",  # English
        "fr",  # French
        "de",  # German
        "ha",  # Hausa
        "hi",  # Hindi
        "hu"   # Hungarian
    }

    # current_directory = os.getcwd()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(f"Scanning directory: {current_directory}")

    for filename in os.listdir(current_directory):
        if filename.endswith("_Kaige.json") and not filename.endswith("_extracted.json"):
            input_filepath = os.path.join(current_directory, filename)

            base, ext = os.path.splitext(filename)
            output_filename = f"{base}_extracted{ext}"
            output_filepath = os.path.join(current_directory, output_filename)

            filter_language_data(input_filepath, output_filepath, languages_to_keep)

    print("\nProcessing complete.")