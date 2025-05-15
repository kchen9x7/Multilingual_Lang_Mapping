
## Currently script integrated with openAI and DeepSeek API calling only

### An example of the script running command for explanation:
```
python inference_api.py \
    --api_provider deepseek \
    --deepseek_model deepseek-chat \
    --dataset_path final_merged_python_generation_translations.jsonl \
    --output_path deepseek_Python_test_results.jsonl \
    --temperature 0.3 \
    --dataset_type generation
```

### An example of the script running command for explanation:
```
python inference_api.py \
    --api_provider deepseek \
    --deepseek_model deepseek-chat \
    --dataset_path final_merged_python_explanation_translations.jsonl \
    --output_path deepseek_Python_test_results.jsonl \
    --temperature 0.3 \
    --dataset_type explanation
```
