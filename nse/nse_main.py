from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast

from util.globals import *
from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words
from .nse_hparams import NSEHyperParams

from collections import defaultdict
# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def apply_nse_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: NSEHyperParams,
    cache_template: Optional[str] = None,
    cache_c = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    #weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    # layer_ks = compute_ks(model, tok, requests, hparams, hparams.layers[0], context_templates,hparams.fact_token).T
    # neuron_indices = []
    # for j, request in enumerate(requests):
    #     token_scores = torch.abs(layer_ks[:, j])
    #     sorted_scores, sorted_indices = torch.sort(token_scores, descending=True)
    #     neuron_indices.append(
    #         sorted_indices[:torch.where(torch.cumsum(sorted_scores, dim=0) <= hparams.neuron_threshold * token_scores.sum())[0][-1] + 1].cpu().tolist()
    #     )

    # # Convert lists to sets
    # neuron_indices = [set(indices) for indices in neuron_indices]
    # n_requests = len(requests)
    # similarity_matrix = torch.zeros((n_requests, n_requests))
    # for m in range(n_requests):
    #     for j in range(m + 1, n_requests):
    #         intersection = len(neuron_indices[m].intersection(neuron_indices[j]))
    #         union = len(neuron_indices[m].union(neuron_indices[j]))
    #         similarity = intersection / union if union > 0 else 0
    #         similarity_matrix[m, j] = similarity_matrix[j, m] = similarity
    # clustering = SpectralClustering(n_clusters=hparams.n_clusters, assign_labels="kmeans", affinity='precomputed')
    # cluster_labels = clustering.fit_predict(similarity_matrix)
    # clusters = defaultdict(list)
    # for j, cluster_label in enumerate(cluster_labels):
    #     clusters[cluster_label].append(j)
    # print(f"Total number of clusters: {len(clusters)}")
    # del similarity_matrix,cluster_labels,neuron_indices

    # for cluster_label, members in clusters.items():
    #     print(f"Cluster {cluster_label}: {len(members)} elements")
    

    z_list = []
    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

        # Insert
    for m in range(hparams.max_iterations):
        cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
        mask = (torch.linalg.norm(zs- cur_zs, dim=0) > hparams.alpha) & (torch.linalg.norm(zs - cur_zs, dim=0) < hparams.upper_bound)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        filtered_requests = [requests[i] for i in indices]
        if torch.all(~mask) or m==hparams.max_iterations-1:
            break
        for i, layer in enumerate(hparams.layers):
            print(f"\n\nLAYER {layer}\n")
            # Get current model activations
            layer_ks = compute_ks(model, tok, filtered_requests, hparams, layer, context_templates).T
            cumulative_token_scores = torch.zeros_like(layer_ks[:, 0])
            for j, request in enumerate(filtered_requests):
                token_scores = torch.abs(layer_ks[:, j])
                cumulative_token_scores += token_scores
            cumulative_sorted_scores, cumulative_sorted_indices = torch.sort(cumulative_token_scores, descending=True)
            neuron_indices = cumulative_sorted_indices[:torch.where(torch.cumsum(cumulative_sorted_scores, dim=0) <= hparams.neuron_threshold*cumulative_token_scores.sum())[0][-1] + 1].cpu().tolist()
            print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")
            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in filtered_requests],
                words=[request["subject"] for request in filtered_requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = zs[:,indices] - cur_zs
            print("z error", torch.linalg.norm(targets, dim=0).mean())
            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1)
            # Load covariance matrix
            force_recompute = False
            # force_recompute = layer != hparams.layers[0]
            cov = get_cov(
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
            )
            # Compute update in double precision
            layer_ks, targets = (
                layer_ks.double(),
                targets.double(),
            )
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            if hparams.model_name == "gpt2-xl":
                upd_matrix = torch.zeros_like(weights[weight_name]).T.double()
            elif hparams.model_name in ["EleutherAI_gpt-j-6B","Llama3-8B"]:
                upd_matrix = torch.zeros_like(weights[weight_name]).double()
            selected_rows = neuron_indices
            adj_k = torch.linalg.solve(
                hparams.mom2_update_weight * cov[selected_rows, :][:, selected_rows].double() + cache_c[i,:,:][selected_rows, :][:, selected_rows].cuda() + layer_ks[selected_rows, :] @ layer_ks[selected_rows, :].T,
               layer_ks[selected_rows, :],
            )  
            resid = targets/(len(hparams.layers) - i)
            partial_upd_matrix = resid @ adj_k.T
            # Adjust update matrix shape
            upd_matrix[: , selected_rows] += partial_upd_matrix
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            print("orig norm", torch.linalg.norm(weights[weight_name]))
            print("upd norm", torch.linalg.norm(upd_matrix))
            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
            # Clear GPU memory
            cov.cpu()
            for x in [layer_ks, cur_zs, targets,partial_upd_matrix,upd_matrix]:
                x.cpu()
                del x
            torch.cuda.empty_cache()
        # # Restore state of original model
        # with torch.no_grad():
        #     for k, v in weights.items():
        #         v[...] = weights_copy[k]
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T
    print(f"Deltas successfully computed for {list(weights.keys())}")

    return model, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
