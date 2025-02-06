
# python3.10 do_inference.py
import easydel as ed
from easydel.utils.analyze_memory import SMPMemoryMonitor  # Optional for memory analysis
import jax
from transformers import AutoTokenizer
from jax import numpy as jnp, sharding, lax, random as jrnd
from huggingface_hub import HfApi
import datasets
from flax.core import FrozenDict

PartitionSpec, api = sharding.PartitionSpec, HfApi()

dp, fsdp, tp, sp = 1, -1 , 4, 1
sharding_axis_dims = (dp, fsdp, tp, sp)

# We have 32 devices in total
num_devices = len(jax.devices())
print("Number of JAX devices:", num_devices)

max_length = 6144

pretrained_model_name_or_path = "NaniDAO/Meta-Llama-3.1-8B-Instruct-ablated-v1"
dtype = jnp.bfloat16

# Create partition_axis telling EasyDel how to slice each dimension
partition_axis = ed.PartitionAxis(
    # batch_axis="dp",       # Use dp to shard the batch dimension
    # head_axis="tp",        # Use tp to shard the heads dimension
    # query_sequence_axis=None,  # or "sp" if you want sequence parallel on queries
    # key_sequence_axis=None,    # or "sp" if you want sequence parallel on keys
)

# Build model with the desired parallelism
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    auto_shard_model=True,
    sharding_axis_dims=sharding_axis_dims,
    config_kwargs=ed.EasyDeLBaseConfigDict(
        use_scan_mlp=False,             # or True if you want scanning MLP
        partition_axis=partition_axis,
        attn_dtype=jnp.bfloat16,
        freq_max_position_embeddings=max_length,
        mask_max_position_embeddings=max_length,
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
    ),
    quantization_method="8bit",
    platform=None,
    partition_axis=partition_axis,
    param_dtype=dtype,
    dtype=dtype,
    precision=lax.Precision("fastest"),
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

# Build inference object
inference = ed.vInference(
    model=model,
    processor_class=tokenizer,
    generation_config=ed.vInferenceConfig(
        max_new_tokens=1024,
        temperature=model.generation_config.temperature,
        top_p=model.generation_config.top_p,
        top_k=model.generation_config.top_k,
        eos_token_id=model.generation_config.eos_token_id,
        streaming_chunks=64,
    ),
)

# Precompile with batch_size=1
inference.precompile(1)
print("Inference name:", inference.inference_name)