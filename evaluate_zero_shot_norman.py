# =========================================================================================
# evaluate_zero_shot_norman_ultimate.py (Definitive Final Version - With Stress Test & Gene Program Details)
#
# PURPOSE:
# The definitive external validation script that produces all requested outputs by:
# 1. Performing ALL LFC calculations on consistently SCALED data for fair comparison.
# 2. Generating a main metrics table using the reliable compute_all_metrics function.
# 3. Providing a detailed per-perturbation report, using the stress-test (external real control) as primary score.
# 4. Performing a gene program subgroup analysis with gene names and counts printed.
# 5. Generating all requested visualizations.
# =========================================================================================
import os
import torch
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from model import DiffusionModel, ConditionalVAE
from opt import parse_args
from metrics import compute_all_metrics

# --- HELPER FUNCTIONS (at the end of the script) ---
def main():
    parser = argparse.ArgumentParser(description="Run a definitive external validation with all outputs.")
    parser.add_argument('--test_file', type=str, default='norman_single_test_cleaned.tsv', help='Path to the external test file with control cells.')
    parser.add_argument('--output_dir', type=str, default='./norman_external_validation', help='Directory to save results.')
    parser.add_argument('--gene_program_file', type=str, default='gene_program.tsv', help='Path to the TSV file mapping genes to their functional programs.')
   
    cmd_args = parser.parse_args()
    args = parse_args([])
    os.makedirs(cmd_args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading artifacts...")
    with open(args.vae_artifacts_path, 'rb') as f: vae_artifacts = pickle.load(f)
    with open(os.path.join(args.output_dir, 'ldm_artifacts.pkl'), 'rb') as f: ldm_artifacts = pickle.load(f)
    vae_model = ConditionalVAE(len(vae_artifacts['gene_cols']), len(vae_artifacts['condition_encoder'].get_feature_names_out()), vae_artifacts['latent_dim']).to(device)
    vae_model.load_state_dict(torch.load(args.vae_path, map_location=device)); vae_model.eval()
    model = DiffusionModel(latent_dim=vae_artifacts['latent_dim'], hidden_dim=args.embedding_dim, noise_steps=args.noise_steps, device=device, num_transformer_blocks=args.num_transformer_blocks, num_heads=args.num_heads, cell_line_dim=len(ldm_artifacts['ldm_encoders']['Cell_Line'].get_feature_names_out()), pert_method_dim=len(ldm_artifacts['ldm_encoders']['Genetic_perturbations'].get_feature_names_out()), gene_embedding_dim=next(iter(ldm_artifacts['gene_embedding_dict'].values())).shape[0], gene_token_dim=128, num_known_genes=len(ldm_artifacts['gene2id']))
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth'), map_location=device)); model.eval()
    print(f"Loading data from {cmd_args.test_file}...")
    full_test_df = pd.read_csv(cmd_args.test_file, sep='\t')
   
    try:
        gene_program_df = pd.read_csv(cmd_args.gene_program_file, sep='\t')
        gene_program_df = gene_program_df[~gene_program_df['guide_merged'].str.contains('_')]
        gene_program_df.rename(columns={'guide_merged': 'perturbation'}, inplace=True)
        print(f"✓ Gene program data loaded successfully for {len(gene_program_df)} genes.")
    except FileNotFoundError:
        print(f"[WARNING] Gene program file '{cmd_args.gene_program_file}' not found. Skipping subgroup analysis.")
        gene_program_df = None
    real_full_scaled, common_indices, common_genes = align_data_to_model(full_test_df, vae_artifacts['gene_cols'], vae_artifacts['scaler'])
   
    print(f"\nTRUE ZERO-SHOT EVALUATION ON TEST ({len(full_test_df)} cells)")
    gen_full_scaled = generate_zero_shot_profiles(model, vae_model, ldm_artifacts['latent_scaler'], ldm_artifacts, full_test_df, vae_artifacts, args, device)
   
    print("Slicing data to common genes for metric calculation...")
    real_eval = real_full_scaled[:, common_indices]
    gen_eval = gen_full_scaled[:, common_indices]
    # ==============================================================================
    # --- PART 1: CALCULATE AND DISPLAY MAIN METRICS TABLE ---
    # ==============================================================================
    metrics = compute_all_metrics(real_eval, gen_eval, full_test_df, common_genes)
    print("\n--- ZERO-SHOT METRICS (Intersection Genes) ---")
    for k, v in metrics.items():
        print(f"{k:20}: {v:.6f}")
   
    # ==============================================================================
    # --- PART 2: DETAILED PER-PERTURBATION BREAKDOWN (ON SCALED DATA) ---
    # ==============================================================================
    print("\n--- Calculating Detailed Per-Perturbation Correlations on SCALED Data ---")
   
    all_pert_scores = []
    unique_perts = full_test_df[full_test_df['Perturbed_Gene'] != 'Control']['Perturbed_Gene'].unique()
   
    real_df_eval = pd.DataFrame(real_eval, columns=common_genes)
    gen_df_eval = pd.DataFrame(gen_eval, columns=common_genes)
    metadata = full_test_df.reset_index(drop=True)
    real_ctrl_eval_df = real_df_eval[metadata['Perturbed_Gene'] == 'Control']
    gen_ctrl_eval_df = gen_df_eval[metadata['Perturbed_Gene'] == 'Control']
    pseudo_count = 1e-6
    for pert in tqdm(unique_perts, desc="Evaluating Perturbations"):
        p_indices = metadata['Perturbed_Gene'] == pert
        real_pert_subset = real_df_eval[p_indices]
        gen_pert_subset = gen_df_eval[p_indices]
        # Ground Truth LFC (on scaled data)
        real_lfc = np.log2((real_pert_subset.mean(axis=0) + pseudo_count) / (real_ctrl_eval_df.mean(axis=0) + pseudo_count))
        real_lfc = real_lfc.fillna(0)
       
        # Perturbation Correlation using Real Control (Stress Test - Primary Score)
        gen_lfc = np.log2((gen_pert_subset.mean(axis=0) + pseudo_count) / (real_ctrl_eval_df.mean(axis=0) + pseudo_count))
        gen_lfc = gen_lfc.fillna(0)
        pert_corr, _ = pearsonr(real_lfc, gen_lfc)
       
        all_pert_scores.append({
            'perturbation': pert,
            'perturbation_corr': pert_corr,
        })
    scores_df = pd.DataFrame(all_pert_scores)
   
    print("\n--- DETAILED PERTURBATION CORRELATION ANALYSIS ---")
    print(f"Mean Perturbation Correlation (External validation dataset): {scores_df['perturbation_corr'].mean():.6f}")
   
    scores_path = os.path.join(cmd_args.output_dir, 'per_perturbation_scores_detailed.csv')
    scores_df.to_csv(scores_path, index=False)
    print(f"\nDetailed per-perturbation scores saved to: {scores_path}")
   
    print("\n--- Full Per-Perturbation Ranking (Sorted by Perturbation Correlation) ---")
    print(scores_df.sort_values('perturbation_corr', ascending=False).to_string(index=False))
   
    # ==============================================================================
    # --- PART 3: GENE PROGRAM SUBGROUP ANALYSIS ---
    # ==============================================================================
    if gene_program_df is not None:
        print("\n--- Gene Program Subgroup Analysis ---")
        analysis_df = pd.merge(scores_df, gene_program_df, on='perturbation', how='left')
       
        program_performance = analysis_df.groupby('gene_program').agg({
            'perturbation_corr': 'mean',
            'perturbation': lambda x: ', '.join(sorted(x)) + f" (n={len(x)})"
        }).rename(columns={'perturbation': 'genes (count)'}).sort_values('perturbation_corr', ascending=False)
       
        print("\n--- Model Performance by Gene Program (with Gene Names and Counts) ---")
        print(program_performance.to_string(float_format="%.4f"))
       
        plot_data = program_performance.reset_index()
        plt.figure(figsize=(10, 8))
        sns.barplot(data=plot_data, y='gene_program', x='perturbation_corr', palette='viridis')
        plt.title("Model Performance by Perturbation Gene Program", fontsize=16)
        plt.xlabel("Mean Perturbation Correlation", fontsize=12)
        plt.ylabel("Gene Program", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(cmd_args.output_dir, 'gene_program_performance.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\nSubgroup analysis plot saved to: {plot_path}")
    # ==============================================================================
    # --- PART 4: LOLLIPOP PLOT VISUALIZATION ---
    # ==============================================================================
    print("\n--- Generating Per-Perturbation Lollipop Plot ---")
    plt.figure(figsize=(10, 12))
    plot_df_lollipop = scores_df.sort_values('perturbation_corr', ascending=True)
    plt.hlines(y=plot_df_lollipop['perturbation'], xmin=0, xmax=plot_df_lollipop['perturbation_corr'], color='skyblue', alpha=0.7)
    plt.scatter(plot_df_lollipop['perturbation_corr'], plot_df_lollipop['perturbation'], color='dodgerblue', s=75, alpha=0.7)
    plt.xlabel("Perturbation Correlation", fontsize=12)
    plt.ylabel("Perturbation", fontsize=12)
    plt.title("Per-Perturbation Zero-Shot Performance (External Validation)", fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='grey', linestyle='--')
    plt.tight_layout()
    lollipop_path = os.path.join(cmd_args.output_dir, 'per_perturbation_lollipop.png')
    plt.savefig(lollipop_path, dpi=300)
    plt.close()
    print(f"Lollipop plot saved to: {lollipop_path}")
    # ==============================================================================
    # --- PART 5: UMAP VISUALIZATION ---
    # ==============================================================================
    print("\n--- Generating UMAP visualizations ---")
    try:
        import umap
        pert_df = full_test_df[full_test_df['Perturbed_Gene'] != 'Control']
        ctrl_df = full_test_df[full_test_df['Perturbed_Gene'] == 'Control']
       
        gen_full_unscaled = vae_artifacts['scaler'].inverse_transform(gen_full_scaled)
        gen_full_unscaled_df = pd.DataFrame(gen_full_unscaled, columns=vae_artifacts['gene_cols'], index=full_test_df.index)
       
        real_pert_common_df = pert_df[common_genes]
        real_ctrl_common_df = ctrl_df[common_genes]
        gen_pert_common_df = gen_full_unscaled_df.loc[pert_df.index][common_genes]
        gen_ctrl_common_df = gen_full_unscaled_df.loc[ctrl_df.index][common_genes]
       
        real_full_common_df = pd.concat([real_pert_common_df, real_ctrl_common_df])
       
        print(" - Fitting UMAP on real data...")
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42, n_components=2)
        reducer.fit(real_full_common_df)
        print(" - Transforming all data into UMAP space...")
       
        gen_full_common_df = pd.concat([gen_pert_common_df, gen_ctrl_common_df])
        real_embedding = reducer.transform(real_full_common_df)
        gen_embedding = reducer.transform(gen_full_common_df)
       
        embedding_df_real = pd.DataFrame(real_embedding, columns=['UMAP1', 'UMAP2'])
        embedding_df_real['Source'] = 'Real'
        embedding_df_real['Perturbation'] = full_test_df['Perturbed_Gene'].values
        embedding_df_gen = pd.DataFrame(gen_embedding, columns=['UMAP1', 'UMAP2'])
        embedding_df_gen['Source'] = 'Generated'
        embedding_df_gen['Perturbation'] = full_test_df['Perturbed_Gene'].values
        embedding_df_pert = pd.concat([embedding_df_real, embedding_df_gen])
       
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=embedding_df_pert, x='UMAP1', y='UMAP2', hue='Perturbation', style='Source', s=10, alpha=0.6)
        plt.title("UMAP Projection of Real vs. Generated Cells by Perturbation", fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        umap_path1 = os.path.join(cmd_args.output_dir, 'external_validation_umap_by_pert.png')
        plt.savefig(umap_path1, dpi=300)
        plt.close()
        print(f"UMAP plot (colored by perturbation) saved to: {umap_path1}")
       
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=embedding_df_pert, x='UMAP1', y='UMAP2', hue='Source', s=10, alpha=0.4, palette={'Real': 'blue', 'Generated': 'red'})
        plt.title("UMAP Projection of Real vs. Generated Populations", fontsize=16)
        plt.tight_layout()
        umap_path2 = os.path.join(cmd_args.output_dir, 'external_validation_umap_by_source.png')
        plt.savefig(umap_path2, dpi=300)
        plt.close()
        print(f"UMAP plot (colored by source) saved to: {umap_path2}")
    except ImportError:
        print("\n[WARNING] `umap-learn` is not installed. Skipping UMAP visualization.")
    except Exception as e:
        print(f"\n[ERROR] Failed to generate UMAP plot. Error: {e}")

# --- HELPER FUNCTIONS ---
def align_data_to_model(df, model_gene_cols, scaler):
    test_genes = set([c for c in df.columns if c.startswith('ENSG')])
    model_genes = list(model_gene_cols)
    common_genes = [g for g in model_genes if g in test_genes]
    common_indices = [i for i, g in enumerate(model_genes) if g in test_genes]
    print(f"\n--- Data Alignment ---\nModel was trained on: {len(model_genes)} genes\nThis test data has: {len(test_genes)} gene-like columns\nIntersection found: {len(common_genes)} genes (These will be used for evaluation)")
    if len(common_genes) == 0:
        raise ValueError("CRITICAL ERROR: No overlapping genes found!")
    df_aligned = pd.DataFrame(0.0, index=df.index, columns=model_genes)
    df_aligned.update(df[common_genes])
    real_scaled_full = scaler.transform(df_aligned.values)
    return real_scaled_full, common_indices, common_genes

def generate_zero_shot_profiles(model, vae_model, latent_scaler, ldm_artifacts, metadata_df, vae_artifacts, args, device):
    model.eval()
    vae_model.eval()
    num_samples = len(metadata_df)
    latent_dim = vae_artifacts['latent_dim']
    embedding_dim = next(iter(ldm_artifacts['gene_embedding_dict'].values())).shape[0]
    gene2id = ldm_artifacts['gene2id']
    unknown_gene_id = gene2id['__unknown__']
    cl_enc = ldm_artifacts['ldm_encoders']['Cell_Line'].transform(metadata_df[['Cell_Line']])
    pm_enc = ldm_artifacts['ldm_encoders']['Genetic_perturbations'].transform(metadata_df[['Genetic_perturbations']])
    cl_tensor = torch.tensor(cl_enc, dtype=torch.float32).to(device)
    pm_tensor = torch.tensor(pm_enc, dtype=torch.float32).to(device)
    zero_sig = torch.zeros(num_samples, embedding_dim, device=device)
    gene_token_ids = torch.full((num_samples,), unknown_gene_id, dtype=torch.long, device=device)
    generated_latents_scaled = []
    bs = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, bs), desc="Generating zero-shot latents"):
            cur_bs = min(bs, num_samples - i)
            x_t = torch.randn(cur_bs, latent_dim, device=device)
            cl_b = cl_tensor[i:i+cur_bs]
            pm_b = pm_tensor[i:i+cur_bs]
            sig_b = zero_sig[i:i+cur_bs]
            gid_b = gene_token_ids[i:i+cur_bs]
            for t in reversed(range(args.noise_steps)):
                t_t = torch.full((cur_bs,), t, dtype=torch.long, device=device)
                p_un = model(x_t, t_t, cl_b, pm_b, sig_b, gid_b, torch.zeros_like(cl_b[:, :1]))
                p_co = model(x_t, t_t, cl_b, pm_b, sig_b, gid_b, torch.ones_like(cl_b[:, :1]))
                pred = p_un + args.guidance_scale * (p_co - p_un)
                pred = torch.clamp(pred, 0, 1)
                if t > 0:
                    a = model.alpha_bar[t_t].view(-1, 1)
                    a_prev = model.alpha_bar[t_t - 1].view(-1, 1)
                    noise = (x_t - a.sqrt() * pred) / (1 - a).sqrt()
                    x_t = a_prev.sqrt() * pred + (1 - a_prev).sqrt() * noise
                else:
                    x_t = pred
            generated_latents_scaled.append(x_t.cpu())
    latents_scaled = torch.cat(generated_latents_scaled).numpy()
    latents = latent_scaler.inverse_transform(latents_scaled)
    latents = torch.tensor(latents, dtype=torch.float32).to(device)
    vae_conds = torch.tensor(vae_artifacts['condition_encoder'].transform(metadata_df[vae_artifacts['condition_cols']]), dtype=torch.float32).to(device)
    decoded = []
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, bs), desc="Decoding"):
            dec = vae_model.decode(latents[i:i+bs], vae_conds[i:i+bs])
            decoded.append(dec.cpu().numpy())
    return np.concatenate(decoded)

if __name__ == "__main__":
    main()