import json
import os

def convert_json_to_jsonl(input_filepath, output_filepath):
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)

        if not isinstance(data, list):
            print(f"Error: Input file '{input_filepath}' does not contain a JSON list (array).")
            return

        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                f_out.write(json_line + '\n')

        print(f"Successfully converted '{input_filepath}' to JSONL format.")
        print(f"Output saved to '{output_filepath}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_filepath}'. Please check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":

    input_filename = "merged_python_translations.json"

    base, _ = os.path.splitext(input_filename)
    output_filename = f"{base}.jsonl"

    current_directory = os.path.dirname(os.path.abspath(__file__))
    input_filepath = os.path.join(current_directory, input_filename)
    output_filepath = os.path.join(current_directory, output_filename)

    convert_json_to_jsonl(input_filepath, output_filepath)