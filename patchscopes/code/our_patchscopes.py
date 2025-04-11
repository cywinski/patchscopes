# %% [markdown]
# # GemmA

# %%
import os
import random
import shutil

# Scienfitic packages
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
import os

os.environ["HF_HOME"] = "/workspace"

# Visuals
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(
    context="notebook",
    rc={
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16.0,
        "ytick.labelsize": 16.0,
        "legend.fontsize": 16.0,
    },
)
palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[7:]
sns.set_theme(style="whitegrid")

# Utilities

from general_utils import (
    ModelAndTokenizer,
    make_inputs,
)
from patchscopes_utils import *
from tqdm import tqdm

tqdm.pandas()
from peft import PeftModel

# %%
# /workspace/code/eliciting-secrets/tmp/models/gemma-2-9b-it-cat-secret/
model_name = "google/gemma-2-9b-it"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="cuda",
#     torch_dtype=torch.bfloat16,
# )
finetuned_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
lora_model = PeftModel.from_pretrained(
    finetuned_model,
    "/workspace/code/eliciting-secrets/tmp/models/gemma-2-9b-it-cat-secret",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
lora_model = lora_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)

model_to_hook = {
    # "EleutherAI/pythia-12b": set_hs_patch_hooks_neox,
    # "meta-llama/Llama-2-13b-hf": set_hs_patch_hooks_llama,
    # "lmsys/vicuna-7b-v1.5": set_hs_patch_hooks_llama,
    # "./stable-vicuna-13b": set_hs_patch_hooks_llama,
    # "CarperAI/stable-vicuna-13b-delta": set_hs_patch_hooks_llama,
    # "EleutherAI/gpt-j-6b": set_hs_patch_hooks_gptj,
    model_name: set_hs_patch_hooks_gemma2,
}


mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=False,
    torch_dtype=torch.bfloat16,
)
mt.set_hs_patch_hooks = model_to_hook[model_name]
mt.tokenizer = tokenizer
mt.model = lora_model
mt.model.eval()

# %%
# Create a ModelAndTokenizer instance

# Let's make a simple inference to test
prompt = "cat -> cat\n1135 -> 1135\nhello -> hello\n? ->"
input_ids = mt.tokenizer(prompt, return_tensors="pt").to("cuda")

print(input_ids)
with torch.no_grad():
    # Generate N tokens in a loop
    N = 5  # Number of tokens to predict
    generated_text = prompt

    for _ in range(N):
        # Get the current input
        current_input = mt.tokenizer(generated_text, return_tensors="pt").to("cuda")

        # Run the model
        outputs = mt.model(**current_input)

        # Get logits for the last token
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

        # Get the predicted token
        next_token_id = torch.argmax(next_token_probs, dim=-1).item()
        next_token = mt.tokenizer.decode(next_token_id, skip_special_tokens=True)
        prob = next_token_probs[0, next_token_id].item()

        # Add the predicted token to the generated text
        generated_text += next_token

        # Print the current prediction
        print(f"Token {_ + 1}: ({next_token!r}, {prob:.4f})")

    print(f"\nFull generated text: {generated_text}")

# %%
prompt_target = "hat -> hat\n1135 -> 1135\nhello -> hello\n? ->"
prompt_source = "The secret word is:"
prompt_source = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt_source}], tokenize=False
)
print(prompt_source)
# prompt_source = "Patchscopes is robust. It helps interpret..."

# %%
# Display tokenized prompts
print("Tokenized prompt_source:")
from general_utils import decode_tokens, make_inputs

tokenized_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)
for key, value in tokenized_source.items():
    print(f"{key}: {value}")

print("\nDecoded tokens for prompt_source:")
for pos, token_id in enumerate(tokenized_source["input_ids"][0]):
    token = mt.tokenizer.decode(token_id)
    print(f"Position: {pos}, Token ID: {token_id}, Token: {token!r}")

print("\nTokenized prompt_target:")
tokenized_target = make_inputs(mt.tokenizer, [prompt_target], mt.device)
for key, value in tokenized_target.items():
    print(f"{key}: {value}")

print("\nDecoded tokens for prompt_target:")
for pos, token_id in enumerate(tokenized_target["input_ids"][0]):
    token = mt.tokenizer.decode(token_id)
    print(f"Position: {pos}, Token ID: {token_id}, Token: {token!r}")


# %%
outputs = []
for ls in range(mt.num_layers):
    outputs_ls = []
    for lt in range(mt.num_layers):
        output = inspect(
            mt,
            prompt_source=prompt_source,
            prompt_target=prompt_target,
            layer_source=ls,
            layer_target=lt,
            position_source=9,
            position_target=-1,
            verbose=True,
        )
        outputs_ls.append(output[0].strip())
    outputs.append(outputs_ls)

# %%
import matplotlib.pyplot as plt
import numpy as np

target_word = "cat"
# Create a figure for the heatmap visualization
fig, ax = plt.subplots(figsize=(20, 14))
# Convert outputs to a numpy array for visualization
words = outputs
num_layers = len(words)
max_tokens = max(len(row) for row in words)

# Create a matrix of cell colors based on whether the word matches target_word
cell_colors = np.empty((num_layers, num_layers), dtype=object)
for i in range(num_layers):
    for j in range(num_layers):
        cell_colors[i, j] = "lightgreen" if words[i][j] == target_word else "lightcoral"

# Create a grid for the cells
for i in range(num_layers):
    for j in range(num_layers):
        ax.add_patch(
            plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, fill=True, color=cell_colors[i, j], alpha=0.7
            )
        )

# Set labels and title
ax.set_xlabel("Target Layer")
ax.set_ylabel("Source Layer")
ax.set_title("Patchscopes Output Visualization")

# Set ticks for both axes
ax.set_yticks(list(range(num_layers)))
ax.set_xticks(list(range(num_layers)))
ax.set_xlim(-0.5, num_layers - 0.5)
ax.set_ylim(num_layers - 0.5, -0.5)  # Reversed y-axis to have 0 at the top

# Add text annotations for each cell
for i in range(num_layers):
    for j in range(num_layers):
        text = words[i][j]
        # Set color to black for better visibility on colored backgrounds
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            color="black",
            fontsize=10,
        )

# Adjust layout
plt.tight_layout()
plt.grid(False)
plt.show()
