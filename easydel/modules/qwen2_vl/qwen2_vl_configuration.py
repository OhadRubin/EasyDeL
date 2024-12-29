# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseConfig


class Qwen2VLVisionConfig(EasyDeLBaseConfig):
	model_type = "qwen2_vl"
	base_config_key = "vision_config"

	def __init__(
		self,
		depth=32,
		embed_dim=1280,
		hidden_size=3584,
		hidden_act="quick_gelu",
		mlp_ratio=4,
		num_heads=16,
		in_channels=3,
		patch_size=14,
		spatial_merge_size=2,
		temporal_patch_size=2,
		**kwargs,
	):
		super().__init__(**kwargs)

		self.depth = depth
		self.embed_dim = embed_dim
		self.hidden_size = hidden_size
		self.hidden_act = hidden_act
		self.mlp_ratio = mlp_ratio
		self.num_heads = num_heads
		self.in_channels = in_channels
		self.patch_size = patch_size
		self.spatial_merge_size = spatial_merge_size
		self.temporal_patch_size = temporal_patch_size


class Qwen2VLConfig(EasyDeLBaseConfig):
	r"""
	This is the configuration class to store the configuration of a [`Qwen2VLModel`]. It is used to instantiate a
	Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
	with the defaults will yield a similar configuration to that of
	Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

	Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
	documentation from [`PretrainedConfig`] for more information.


	Args:
	    vocab_size (`int`, *optional*, defaults to 152064):
	        Vocabulary size of the Qwen2VL model. Defines the number of different tokens that can be represented by the
	        `inputs_ids` passed when calling [`Qwen2VLModel`]
	    hidden_size (`int`, *optional*, defaults to 8192):
	        Dimension of the hidden representations.
	    intermediate_size (`int`, *optional*, defaults to 29568):
	        Dimension of the MLP representations.
	    num_hidden_layers (`int`, *optional*, defaults to 80):
	        Number of hidden layers in the Transformer encoder.
	    num_attention_heads (`int`, *optional*, defaults to 64):
	        Number of attention heads for each attention layer in the Transformer encoder.
	    num_key_value_heads (`int`, *optional*, defaults to 8):
	        This is the number of key_value heads that should be used to implement Grouped Query Attention. If
	        `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
	        `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
	        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
	        by meanpooling all the original heads within that group. For more details checkout [this
	        paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
	    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
	        The non-linear activation function (function or string) in the decoder.
	    max_position_embeddings (`int`, *optional*, defaults to 32768):
	        The maximum sequence length that this model might ever be used with.
	    initializer_range (`float`, *optional*, defaults to 0.02):
	        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
	    rms_norm_eps (`float`, *optional*, defaults to 1e-05):
	        The epsilon used by the rms normalization layers.
	    use_cache (`bool`, *optional*, defaults to `True`):
	        Whether or not the model should return the last key/values attentions (not used by all models). Only
	        relevant if `config.is_decoder=True`.
	    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
	        Whether the model's input and output word embeddings should be tied.
	    rope_theta (`float`, *optional*, defaults to 1000000.0):
	        The base period of the RoPE embeddings.
	    use_sliding_window (`bool`, *optional*, defaults to `False`):
	        Whether to use sliding window attention.
	    sliding_window (`int`, *optional*, defaults to 4096):
	        Sliding window attention (SWA) window size. If not specified, will default to `4096`.
	    max_window_layers (`int`, *optional*, defaults to 80):
	        The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
	    attention_dropout (`float`, *optional*, defaults to 0.0):
	        The dropout ratio for the attention probabilities.
	    vision_config (`tp.Dict`, *optional*):
	        The config for the visual encoder initialization.
	    rope_scaling (`tp.Dict`, *optional*):
	        Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
	        and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
	        accordingly.
	        Expected contents:
	            `rope_type` (`str`):
	                The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
	                'llama3'], with 'default' being the original RoPE implementation.
	            `factor` (`float`, *optional*):
	                Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
	                most scaling types, a `factor` of x will enable the model to handle sequences of length x *
	                original maximum pre-trained length.
	            `original_max_position_embeddings` (`int`, *optional*):
	                Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
	                pretraining.
	            `attention_factor` (`float`, *optional*):
	                Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
	                computation. If unspecified, it defaults to value recommended by the implementation, using the
	                `factor` field to infer the suggested value.
	            `beta_fast` (`float`, *optional*):
	                Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
	                ramp function. If unspecified, it defaults to 32.
	            `beta_slow` (`float`, *optional*):
	                Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
	                ramp function. If unspecified, it defaults to 1.
	            `short_factor` (`tp.List[float]`, *optional*):
	                Only used with 'longrope'. The scaling factor to be applied to short contexts (<
	                `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
	                size divided by the number of attention heads divided by 2
	            `long_factor` (`tp.List[float]`, *optional*):
	                Only used with 'longrope'. The scaling factor to be applied to long contexts (<
	                `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
	                size divided by the number of attention heads divided by 2
	            `low_freq_factor` (`float`, *optional*):
	                Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
	            `high_freq_factor` (`float`, *optional*):
	                Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE

	```python
	>>> from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig

	>>> # Initializing a Qwen2VL style configuration
	>>> configuration = Qwen2VLConfig()

	>>> # Initializing a model from the Qwen2-VL-7B style configuration
	>>> model = Qwen2VLForConditionalGeneration(configuration)

	>>> # Accessing the model configuration
	>>> configuration = model.config
	```"""

	model_type = "qwen2_vl"
	sub_configs = {"vision_config": Qwen2VLVisionConfig}
	keys_to_ignore_at_inference = ["past_key_values"]

	def __init__(
		self,
		vocab_size=152064,
		hidden_size=8192,
		intermediate_size=29568,
		num_hidden_layers=80,
		num_attention_heads=64,
		num_key_value_heads=8,
		hidden_act="silu",
		max_position_embeddings=32768,
		initializer_range=0.02,
		rms_norm_eps=1e-05,
		use_cache=True,
		tie_word_embeddings=False,
		rope_theta=1000000.0,
		use_sliding_window=False,
		sliding_window=4096,
		max_window_layers=80,
		attention_dropout=0.0,
		vision_config=None,
		rope_scaling=None,
		vision_start_token_id=151652,
		vision_end_token_id=151653,
		vision_token_id=151654,
		image_token_id=151655,
		video_token_id=151656,
		**kwargs,
	):
		if isinstance(vision_config, dict):
			self.vision_config = Qwen2VLVisionConfig(**vision_config)
		elif vision_config is None:
			self.vision_config = Qwen2VLVisionConfig()

		self.vocab_size = vocab_size
		self.max_position_embeddings = max_position_embeddings
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.use_sliding_window = use_sliding_window
		self.sliding_window = sliding_window
		self.max_window_layers = max_window_layers

		# for backward compatibility
		if num_key_value_heads is None:
			num_key_value_heads = num_attention_heads

		self.num_key_value_heads = num_key_value_heads
		self.hidden_act = hidden_act
		self.initializer_range = initializer_range
		self.rms_norm_eps = rms_norm_eps
		self.use_cache = use_cache
		self.rope_theta = rope_theta
		self.attention_dropout = attention_dropout
		self.rope_scaling = rope_scaling
		# EasyDeL Extended args.
		self.head_dim = hidden_size // num_attention_heads
		self.vision_start_token_id = vision_start_token_id
		self.vision_end_token_id = vision_end_token_id
		self.vision_token_id = vision_token_id
		self.image_token_id = image_token_id
		self.video_token_id = video_token_id
		if self.rope_scaling is not None and "type" in self.rope_scaling:
			if self.rope_scaling["type"] == "mrope":
				self.rope_scaling["type"] = "default"
			self.rope_scaling["rope_type"] = self.rope_scaling["type"]

		super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

	def get_partition_rules(self, *args, **kwargs):
		"""
		Get the partition rules for the model.
		Returns:
		    `tp.Tuple[tp.Tuple[str, PartitionSpec]]`: The partition rules.
		"""
		return (
			# Language model embeddings
			("embed_tokens/embedding", PartitionSpec("tp", ("fsdp", "sp"))),
			# Language model attention layers
			(
				"layers/.*/self_attn/(q_proj|k_proj|v_proj)/kernel",
				PartitionSpec(("fsdp", "sp"), "tp"),
			),
			("layers/.*/self_attn/o_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("layers/.*/self_attn/(q_proj|k_proj|v_proj)/bias", PartitionSpec("tp")),
			("layers/.*/self_attn/o_proj/bias", PartitionSpec(None)),
			# Language model MLP layers
			("layers/.*/mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("layers/.*/mlp/down_proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("layers/.*/mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("layers/.*/mlp/(gate_proj|down_proj|up_proj)/bias", PartitionSpec(None)),
			# Language model norms
			("layers/.*/input_layernorm/kernel", PartitionSpec(None)),
			("layers/.*/post_attention_layernorm/kernel", PartitionSpec(None)),
			("norm/kernel", PartitionSpec(None)),
			# Language model head
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("lm_head/bias", PartitionSpec(None)),
			# Visual model patch embedding
			("patch_embed/proj/kernel", PartitionSpec(None, None, None, None, "tp")),
			("patch_embed/proj/bias", PartitionSpec(None)),
			# Visual model attention blocks
			("blocks/.*/attn/qkv/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("blocks/.*/attn/qkv/bias", PartitionSpec("tp")),
			("blocks/.*/attn/proj/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("blocks/.*/attn/proj/bias", PartitionSpec("tp")),
			# Visual model MLP blocks
			("blocks/.*/mlp/fc1/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("blocks/.*/mlp/fc1/bias", PartitionSpec("tp")),
			("blocks/.*/mlp/fc2/kernel", PartitionSpec("tp", ("fsdp", "sp"))),
			("blocks/.*/mlp/fc2/bias", PartitionSpec("tp")),
			# Visual model norms
			("blocks/.*/norm1/(bias|scale)", PartitionSpec(None)),
			("blocks/.*/norm2/(bias|scale)", PartitionSpec(None)),
			# Visual model merger
			("merger/ln_q/(bias|scale)", PartitionSpec(None)),
			("merger/mlp/.*/kernel", PartitionSpec(("fsdp", "sp"), "tp")),
			("merger/mlp/.*/bias", PartitionSpec("tp")),
			# Catch-all for any remaining parameters
			(".*", PartitionSpec(None)),
		)