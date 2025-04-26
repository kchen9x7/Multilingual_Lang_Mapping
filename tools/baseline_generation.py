import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def generate_baseline_file(model_name="princeton-nlp/sup-simcse-roberta-large", lang="en", 
                          num_samples=100000, batch_size=32, output_dir=None):
    """
    Generate a baseline file for BERTScore rescaling.
    """
    print(f"Generating baseline for {model_name}...")
    
    if output_dir is None:
        try:
            import bert_score
            package_dir = os.path.dirname(bert_score.__file__)
            output_dir = os.path.join(package_dir, "rescale_baseline", lang)
        except:
            output_dir = "."
    
    os.makedirs(output_dir, exist_ok=True)
    
    safe_model_name = model_name.replace('/', '_')
    output_file = os.path.join(output_dir, f"{safe_model_name}.tsv")
    
    print(f"Baseline will be saved to: {output_file}")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    
    all_f1_scores = []
    print(f"Processing {num_samples} random samples in batches of {batch_size}...")
    
    for i in tqdm(range(0, num_samples, batch_size)):
        current_batch_size = min(batch_size, num_samples - i)
        
        tokens_ref = []
        tokens_cand = []
        
        for _ in range(current_batch_size):
            length = np.random.randint(20, 100)
            # Random token IDs
            ref_ids = np.random.randint(1000, len(tokenizer) - 1000, size=length).tolist()
            cand_ids = np.random.randint(1000, len(tokenizer) - 1000, size=length).tolist()
            
            tokens_ref.append(ref_ids)
            tokens_cand.append(cand_ids)
        
        # Process references
        max_len = max([len(ids) for ids in tokens_ref])
        attention_mask_ref = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in tokens_ref]
        padded_ref = [ids + [0] * (max_len - len(ids)) for ids in tokens_ref]
        
        ref_input_ids = torch.tensor(padded_ref).to(device)
        ref_attention_mask = torch.tensor(attention_mask_ref).to(device)
        
        # Process candidates
        max_len = max([len(ids) for ids in tokens_cand])
        attention_mask_cand = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in tokens_cand]
        padded_cand = [ids + [0] * (max_len - len(ids)) for ids in tokens_cand]
        
        cand_input_ids = torch.tensor(padded_cand).to(device)
        cand_attention_mask = torch.tensor(attention_mask_cand).to(device)
        
        # Get embeddings
        with torch.no_grad():
            # Reference embeddings
            ref_out = model(input_ids=ref_input_ids, attention_mask=ref_attention_mask)
            ref_embeddings = ref_out.last_hidden_state
            
            # Candidate embeddings
            cand_out = model(input_ids=cand_input_ids, attention_mask=cand_attention_mask)
            cand_embeddings = cand_out.last_hidden_state
        
        # Calculate F1 scores
        for b in range(current_batch_size):
            ref_emb = ref_embeddings[b, ref_attention_mask[b].bool()]
            cand_emb = cand_embeddings[b, cand_attention_mask[b].bool()]
            
            # Normalize embeddings
            ref_emb = ref_emb / ref_emb.norm(dim=1, keepdim=True)
            cand_emb = cand_emb / cand_emb.norm(dim=1, keepdim=True)
            
            # Calculate similarities
            sim = torch.mm(ref_emb, cand_emb.t())
            
            # Calculate precision and recall
            precision = sim.max(dim=1)[0].mean().item()
            recall = sim.max(dim=0)[0].mean().item()
            
            # Calculate F1
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            all_f1_scores.append(f1)
    
    # Calculate statistics for the F1 scores
    all_f1_scores = np.array(all_f1_scores)
    stats = {
        'min': all_f1_scores.min(),
        'max': all_f1_scores.max(),
        'mean': all_f1_scores.mean(),
        'std': all_f1_scores.std()
    }
    
    print(f"Statistics: {stats}")
    print(f"Saving baseline to {output_file}")
    with open(output_file, 'w') as f:
        f.write(f"{stats['min']}\t{stats['max']}\t{stats['mean']}\t{stats['std']}\n")
    
    print(f"Baseline generation complete! File saved to {output_file}")
    print(f"You can now use this model with rescale_with_baseline=True in BERTScore.")
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate BERTScore baseline file for a model')
    parser.add_argument('--model', type=str, default="princeton-nlp/sup-simcse-roberta-large",
                        help='Model name/path')
    parser.add_argument('--lang', type=str, default="en", help='Language code')
    parser.add_argument('--samples', type=int, default=100000, help='Number of random samples')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Output directory (default: bert_score package directory)')
    
    args = parser.parse_args()
    
    generate_baseline_file(
        model_name=args.model,
        lang=args.lang,
        num_samples=args.samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )