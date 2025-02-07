# . ~/venv/bin/activate && python train_clm.py
# add this using sys.path.append
import sys
sys.path.append(".")

from easydel import (
    TrainArguments,
    CausalLanguageModelTrainer,
    AutoEasyDeLModelForCausalLM,
    EasyDeLOptimizers,
    EasyDeLSchedulers,
    EasyDeLGradientCheckPointers
)
from datasets import load_dataset
import flax
from jax import numpy as jnp
from transformers import AutoTokenizer

huggingface_repo_id_or_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

model, params = AutoEasyDeLModelForCausalLM.from_pretrained(huggingface_repo_id_or_path, )

max_length = 2048
tokenizer = AutoTokenizer.from_pretrained(
    huggingface_repo_id_or_path,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
configs_to_initialize_model_class = {
    "config": model.config,
    "dtype": jnp.bfloat16,
    "param_dtype": jnp.bfloat16,
    "input_shape": (1, 1)
}

train_arguments = TrainArguments(
    model_class=type(model),
    model_name="my_first_model_to_train_using_easydel",
    num_train_epochs=3,
    configs_to_initialize_model_class=configs_to_initialize_model_class,
    learning_rate=5e-5,
    learning_rate_end=1e-6,
    optimizer=EasyDeLOptimizers.ADAMW,  # "adamw", "lion", "adafactor" are supported
    scheduler=EasyDeLSchedulers.LINEAR,
    # "linear","cosine", "none" ,"warm_up_cosine" and "warm_up_linear"  are supported
    weight_decay=0.01,
    total_batch_size=64,
    max_training_steps=None,  # None to let trainer Decide
    do_train=True,
    do_eval=False,  # it's optional but supported 
    backend="tpu",  # default backed is set to cpu, so you must define you want to use tpu cpu or gpu
    max_length=max_length,  # Note that you have to change this in the model config too
    gradient_checkpointing=EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    sharding_array=(1, -1, 1, 1),  # the way to shard model across gpu,cpu or TPUs using sharding array (1, -1, 1, 1)
    # everything training will be in sequence and model parallel automatic and share data between devices
    remove_ckpt_after_load=True,
    gradient_accumulation_steps=8,
    loss_re_mat="",
    dtype=jnp.bfloat16
)


def ultra_chat_prompting_process(
        data_chunk
):
    user_part = [
        chunk["content"] for chunk in data_chunk["messages"] if chunk["role"] == "user"
    ]
    assistant_part = [
        chunk["content"] for chunk in data_chunk["messages"] if chunk["role"] == "assistant"
    ]

    prompt = ""

    for uc, ac in zip(user_part, assistant_part):
        prompt += f"<|user|>\n{uc}</s>\n<|assistant|>\n{ac}</s>\n"

    return {"prompt": prompt}


tokenization_process = lambda data_chunk: tokenizer(
    data_chunk["prompt"],
    add_special_tokens=False,
    max_length=max_length,
    padding="max_length"
)

dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
dataset_train = dataset["train_gen"].map(ultra_chat_prompting_process, num_proc=12)
dataset_train = dataset_train.map(
    tokenization_process,
    num_proc=12,
    remove_columns=dataset_train.column_names
)

# you can do the same for evaluation process dataset

trainer = CausalLanguageModelTrainer(
    train_arguments,
    dataset_train,
    checkpoint_path=None
)

output = trainer.train(flax.core.FrozenDict({"params": params}))
print(f"Hey ! , here's where your model saved {output.checkpoint_path}")