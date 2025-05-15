import argparse
import json
import os
import time
import requests
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Optional, Dict, Any, List

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. \
    Always answer as helpfully as possible, while being safe. \
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. \
    Please ensure that your responses are socially unbiased and positive in nature. \
    \n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. \
    If you don't know the answer to a question, please don't share false information. \
    Please output only code component as your answer without explanation."

APPEND_PROMPT = "\n    Please make sure to only output code as your answer without explanation."

# --- API Configuration ---
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

load_dotenv()

def call_openai_api(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    system: str
) -> Optional[str]:
    """Makes a call to the OpenAI Chat Completion API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            system=system
        )
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content.strip() if content else None
        else:
            print(f"Warning: OpenAI API returned unexpected response structure: {response}")
            return None
    except RateLimitError as e:
        print(f"Warning: OpenAI Rate limit hit. {e}")
        raise
    except (APITimeoutError, APIConnectionError) as e:
        print(f"Warning: OpenAI Network error ({type(e).__name__}). {e}")
        raise
    except APIStatusError as e:
        print(f"Warning: OpenAI API Status Error: {e.status_code} - {e.message}")
        # Don't retry on invalid key (401) or bad request (400)
        if e.status_code in [401, 400, 404]:
             print("Fatal API error. Skipping request.")
             return None
        raise
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None


def call_deepseek_api(
    api_key: str,
    api_base: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> Optional[str]:
    # Call to the DeepSeek Chat Completion API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "system": SYSTEM_PROMPT
        # 'stream': False
    }
    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        # Raise HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()

        response_data = response.json()
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            content = response_data["choices"][0].get("message", {}).get("content")
            return content.strip() if content else None
        else:
            print(f"Warning: DeepSeek API returned unexpected response structure: {response_data}")
            return None

    except requests.exceptions.Timeout as e:
         print(f"Warning: DeepSeek Request timed out. {e}")
         raise # Re-raise to trigger retry logic
    except requests.exceptions.ConnectionError as e:
         print(f"Warning: DeepSeek Network error. {e}")
         raise # Re-raise to trigger retry logic
    except requests.exceptions.RequestException as e:
        print(f"Error during DeepSeek API call: {e}")
        # Check status code for non-retryable errors
        if e.response is not None and e.response.status_code in [401, 400, 404]:
             print(f"Fatal DeepSeek API error ({e.response.status_code}). Skipping request. Response: {e.response.text}")
             return None # Indicate failure without retry

        print(f"DeepSeek request failed. Status: {e.response.status_code if e.response else 'N/A'}. Response: {e.response.text if e.response else 'N/A'}")
        raise # Re-raise to trigger retry logic
    except Exception as e:
        print(f"Unexpected error during DeepSeek API call: {e}")
        return None

def call_api_with_retry(
    provider: str,
    client_or_key: Any, # OpenAI client or DeepSeek API key
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    retry_limit: int,
    initial_delay: float,
    timeout: int,
    api_base: Optional[str] = None # Only for DeepSeek
) -> Optional[str]:
    """Calls the appropriate API function with exponential backoff retry logic."""
    delay = initial_delay
    for attempt in range(retry_limit):
        try:
            if provider == 'openai':
                return call_openai_api(
                    client_or_key, model_id, messages, max_tokens, temperature, top_p, timeout, SYSTEM_PROMPT
                )
            elif provider == 'deepseek':
                 if api_base is None:
                     raise ValueError("api_base must be provided for deepseek provider")
                 return call_deepseek_api(
                    client_or_key, api_base, model_id, messages, max_tokens, temperature, top_p, timeout
                )
            else:
                raise ValueError(f"Unknown API provider: {provider}")
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError, requests.exceptions.RequestException) as e:
            # retry for specific API/network errors
            if isinstance(e, (APIStatusError, requests.exceptions.RequestException)):
                 #400, 401, 404 status errors handling
                 status_code = getattr(e, 'status_code', None) or getattr(getattr(e,'response', None), 'status_code', None)
                 if status_code in [400, 401, 404]:
                     print(f"Non-retryable error encountered ({status_code}). Aborting retries for this request.")
                     return None

            if attempt < retry_limit - 1:
                print(f"Attempt {attempt + 1} failed. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"API call failed after {retry_limit} attempts.")
                return None
        except Exception as e:
             print(f"Unexpected error during API attempt {attempt + 1}: {e}")
             return None

    return None


def main():
    parser = argparse.ArgumentParser(description="Run inference using OpenAI or DeepSeek APIs on a multilingual dataset.")
    parser.add_argument("--api_provider", required=True, choices=['openai', 'deepseek'], help="API provider to use.")
    parser.add_argument("--openai_model", default='gpt-4o', help="OpenAI model ID to use.")
    parser.add_argument("--deepseek_model", default='deepseek-chat', help="DeepSeek model ID to use.")
    parser.add_argument("--dataset_path", required=True, help="Path to the input JSONL dataset file.")
    parser.add_argument("--output_path", required=True, help="Path to save the output JSONL results file.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top_p.")
    parser.add_argument("--num_tasks", type=int, default=-1, help="Number of tasks (lines) to process from the dataset (-1 for all).")
    parser.add_argument("--api_retry_limit", type=int, default=5, help="Maximum number of retries for API calls.")
    parser.add_argument("--api_initial_delay", type=float, default=1.0, help="Initial delay (seconds) between retries.")
    parser.add_argument("--api_timeout", type=int, default=120, help="Timeout for API requests in seconds.") # Increased timeout
    parser.add_argument("--dataset_type", default='generation', help="Running for the generation or explanation dataset.")

    args = parser.parse_args()

    client_or_key = None
    model_id = ""
    api_base = None # Only for deepseek

    if args.api_provider == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or .env file.")
        try:
            client_or_key = OpenAI(api_key=api_key)
            model_id = args.openai_model
            print(f"Using OpenAI provider with model: {model_id}")
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            return
    elif args.api_provider == 'deepseek':
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables or .env file.")
        client_or_key = api_key
        model_id = args.deepseek_model
        api_base = DEEPSEEK_API_BASE
        print(f"Using DeepSeek provider with model: {model_id} at endpoint {api_base}")
    else:
         print(f"Error: Invalid API provider specified: {args.api_provider}")
         return


    results = []
    processed_tasks = 0
    total_api_calls = 0

    # count tasks and language variants for tqdm total
    try:
        with open(args.dataset_path, "r", encoding="utf-8") as f_in:
            tasks_to_process = []
            for i, line in enumerate(f_in):
                 if args.num_tasks > 0 and i >= args.num_tasks:
                     break
                 try:
                     task_data = json.loads(line)
                     if 'instruction' in task_data and isinstance(task_data['instruction'], dict):
                         total_api_calls += len(task_data['instruction'])
                         tasks_to_process.append(task_data)
                     else:
                          print(f"Warning: Skipping line {i+1} due to missing or invalid 'instruction' field: {line.strip()}")
                 except json.JSONDecodeError:
                     print(f"Warning: Skipping line {i+1} due to JSON decode error: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Input dataset file not found: {args.dataset_path}")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    print(f"Found {len(tasks_to_process)} tasks with {total_api_calls} total language variants to process.")


    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # processing and writing
    try:
        with open(args.output_path, "w", encoding="utf-8") as f_out, \
             tqdm(total=total_api_calls, desc="Processing Languages") as pbar:
            for task_data in tasks_to_process:
                task_id = task_data.get("task_id", "UNKNOWN_TASK")
                instructions = task_data.get("instruction", {})

                if not isinstance(instructions, dict):
                    print(f"Warning: Skipping task {task_id} due to invalid 'instruction' field type.")
                    continue # Skip this task if instructions are not a dict

                for lang_code, instruction_text in instructions.items():
                    if not instruction_text or not isinstance(instruction_text, str):
                        pbar.update(1) # Update progress bar even if skipping
                        print(f"Warning: Skipping language '{lang_code}' for task {task_id} due to empty or invalid text.")
                        continue
                    
                    if args.dataset_type == "generation":
                        appended_instruction = instruction_text + APPEND_PROMPT
                    else:
                        appended_instruction = instruction_text

                    messages = [{"role": "user", "content": appended_instruction}]

                    generated_text = call_api_with_retry(
                        provider=args.api_provider,
                        client_or_key=client_or_key,
                        model_id=model_id,
                        messages=messages,
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        retry_limit=args.api_retry_limit,
                        initial_delay=args.api_initial_delay,
                        timeout=args.api_timeout,
                        api_base=api_base # Pass base URL for deepseek
                    )

                    if generated_text is not None:
                        output_entry = {
                            "task_id": task_id,
                            "language": lang_code,
                            "model_id": model_id,
                            "generated_text": generated_text,
                        }
                        f_out.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                        # Optional: flush periodically if output file is large
                        # if pbar.n % 100 == 0:
                        #    f_out.flush()
                    else:
                        # Log failure for this specific language variant
                        print(f"Failed to get response for task {task_id}, language {lang_code} after retries.")
                        # Optional write an error entry
                        # error_entry = {
                        #     "task_id": task_id,
                        #     "language": lang_code,
                        #     "model_id": model_id,
                        #     "generated_text": "API_CALL_FAILED",
                        # }
                        # f_out.write(json.dumps(error_entry, ensure_ascii=False) + "\n")

                    pbar.update(1) # Update progress bar for each language attempt

                processed_tasks += 1
                # No need to check args.num_tasks here as tasks_to_process is already limited

    except IOError as e:
        print(f"Error writing to output file {args.output_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

    print(f"Processing complete. Results saved to {args.output_path}")
    print(f"Processed {processed_tasks} tasks.")


if __name__ == "__main__":
    main()