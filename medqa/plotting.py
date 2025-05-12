import matplotlib.pyplot as plt
import os
import numpy as np

def plot_patching_heatmap_normalized(
        patching_results,
        clean_tokens,
        tokenizer,
        module_kind=None,
        answer=None,
        filepath=None,
        modelname=None
    ):
    """
    Plots the activation patching results as a heatmap.
    
    Args:
        patching_results: Tensor of shape (n_layers, n_positions) containing patching scores
        clean_tokens: Original token IDs
        tokenizer: Tokenizer used for decoding
        filepath: Where to save the plot
        modelname: Name of the model for the plot title
    """
    # Convert results to numpy for plotting
    differences = patching_results.cpu().numpy()
    differences = differences.T
    low_score = patching_results.min().item()
    window = 10
    num_tokens = differences.shape[0]
    num_layers = differences.shape[1]

    assert num_tokens <= clean_tokens.shape[1]
    labels = [tokenizer.decode([t]) for t in clean_tokens[0][:num_tokens]]

    plot_height = max(num_tokens / 7, 20)
    plot_width = num_layers / 10
    with plt.rc_context():
        fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "resid": "Purples", "mlp": "Greens", "attn": "Reds"}[
                module_kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not module_kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            ax.set_title(f"Impact of restoring state after corrupted input ({module_kind})")
            ax.set_xlabel(f"single restored layer within {modelname}")
            #ax.set_xlabel(f"center of interval of {window} layers within {module_kind} layers")
        cb = plt.colorbar(h)
        if answer is not None:
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_average_patching_heatmap(patching_results, patient_start_idx, patient_end_idx, subsequent_start_idx=None, 
                                 answer=None, filepath=None, modelname="Meerkat-7B"):
    """
    Plots the activation patching results as a heatmap with tokens aggregated into categories.
    
    Args:
        patching_results: Tensor of shape (n_layers, n_positions) containing patching scores
        patient_start_idx: Index of the first patient description token
        patient_end_idx: Index of the last patient description token
        subsequent_start_idx: Index of the first token after patient description (defaults to patient_end_idx + 1)
        answer: The answer token being evaluated
        filepath: Where to save the plot
        modelname: Name of the model for the plot title
    """
    # Convert results to numpy for plotting
    differences = patching_results.cpu().numpy()
    differences = differences.T  # Now shape is (n_positions, n_layers)
    
    # Set subsequent_start_idx if not provided
    if subsequent_start_idx is None:
        subsequent_start_idx = patient_end_idx + 1
    
    # Define the total number of tokens
    num_tokens = differences.shape[0]
    last_idx = num_tokens - 1
    
    # Define the categories
    categories = [
        "first patient token (a)",
        "middle patient token",
        "last patient token (man/woman)",
        "first subsequent token",
        "further tokens",
        "last token"
    ]
    
    # Create category-to-indices mapping for aggregation
    category_indices = {cat: [] for cat in categories}
    
    # First patient token
    if patient_start_idx < num_tokens:
        category_indices["first patient token"].append(patient_start_idx)
    
    # Middle patient tokens
    for idx in range(patient_start_idx + 1, patient_end_idx):
        category_indices["middle patient token"].append(idx)
    
    # Last patient token
    if patient_end_idx > patient_start_idx and patient_end_idx < num_tokens:
        category_indices["last patient token"].append(patient_end_idx)
    
    # First subsequent token
    if subsequent_start_idx < num_tokens:
        category_indices["first subsequent token"].append(subsequent_start_idx)
    
    # Further tokens
    for idx in range(subsequent_start_idx + 1, last_idx):
        category_indices["further tokens"].append(idx)
    
    # Last token
    if last_idx >= 0 and last_idx < num_tokens:
        category_indices["last token"].append(last_idx)
    
    # Aggregate patching results by category
    aggregated_differences = []
    for cat in categories:
        indices = category_indices[cat]
        if indices:
            # Average the patching results for tokens in this category
            aggregated_differences.append(np.mean(differences[indices], axis=0))
        else:
            # Use zeros if no tokens in this category
            aggregated_differences.append(np.zeros(differences.shape[1]))
    
    # Stack the aggregated differences
    aggregated_differences = np.stack(aggregated_differences)
    
    # Get min score for colormap
    low_score = aggregated_differences.min()
    module_kind = None
    
    # Set up plot dimensions
    num_categories = len(categories)
    num_layers = aggregated_differences.shape[1]
    plot_height = 3 # Adjusted for fewer rows
    plot_width = num_layers / 10
    
    # Create the plot
    with plt.rc_context():
        fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=200)
        h = ax.pcolor(
            aggregated_differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                module_kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(num_categories)])
        ax.set_xticks([0.5 + i for i in range(0, num_layers - 6, 5)])
        ax.set_xticklabels(list(range(0, num_layers - 6, 5)))
        ax.set_yticklabels(categories)
        
        if not module_kind:
            ax.set_title("Impact of restoring state after corrupted input (By Token Group)")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            ax.set_title(f"Impact of restoring state after corrupted input ({module_kind}, By Token Group)")
            ax.set_xlabel(f"center of interval of layers within {module_kind} layers")
        
        cb = plt.colorbar(h)
        if answer is not None:
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_patching_heatmap(
        patching_results,
        clean_tokens,
        tokenizer,
        module_kind=None,
        answer=None,
        filepath=None,
        modelname=None
    ):
    """
    Plots the activation patching results as a heatmap with diverging colors.
    Positive values are shown in red, negative values in blue.
    
    Args:
        patching_results: Tensor of shape (n_layers, n_positions) containing patching scores
        clean_tokens: Original token IDs
        tokenizer: Tokenizer used for decoding
        module_kind: Type of module being patched (resid, mlp, attn)
        answer: The correct answer for the question
        filepath: Where to save the plot
        modelname: Name of the model for the plot title
    """
    # Convert results to numpy for plotting
    differences = patching_results.cpu().numpy()
    differences = differences.T
    num_tokens = differences.shape[0]
    num_layers = differences.shape[1]

    assert num_tokens <= clean_tokens.shape[1]
    labels = [tokenizer.decode([t]) for t in clean_tokens[0][:num_tokens]]

    plot_height = max(num_tokens / 7, 20)
    plot_width = num_layers / 10
    
    # Calculate symmetric vmin and vmax for diverging colormap
    abs_max = max(abs(differences.min()), abs(differences.max()))
    vmin = -abs_max
    vmax = abs_max

    with plt.rc_context():
        fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=200)
        
        # Use RdBu (Red-Blue) diverging colormap
        h = ax.pcolor(
            differences,
            cmap="RdBu_r",  # Red for positive, Blue for negative
            vmin=vmin,
            vmax=vmax,
        )
        
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        
        if not module_kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            ax.set_title(f"Impact of restoring state after corrupted input ({module_kind})")
            ax.set_xlabel(f"single restored layer within {modelname}")
        
        cb = plt.colorbar(h)
        if answer is not None:
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        
        if filepath is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        else:
            plt.show()