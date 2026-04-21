This is the doc for standards I intend to use when training probes and adapters, and when loading them too.

## LoRA

The LoRA adapters that I will be using will be adapted from the unsloth pattern. This results in checkpoints which contain certain critical files which are absolutely necessary to load and use the trained adapter.

- `adapter_config.json` - This describes the LoRA architecture, including r, alpha, target module, and the name of the base model
- `adapter_model.safetensors` - The actual trained weights of the model itself. Using safetensors as a safer alternative to .bin files
- `tokenizer.json` - Tokenizer config. Required if I'm doing any tokenization along with the adapter, although this will typically be the standard tokenizer which comes along with some model
- `tokenizer_config.json` - Tokenizer metadata. Required to initialise the tokenizer
- `special_tokens_map.json` - Maps special tokens to string values. Redundant often, but expected
- `chat_template.jinja` - a Jinja2 template describing the chat template
- `lora_metadata.json` - Proprietary provenance file written at training time. Makes the adapter directory self-describing without needing to look up the config or training run. Example fields:

```json
{
  "base_model": "google/gemma-4-31B-it",
  "target_model": "openai/gpt-5.4-nano",
  "source_data": "data/generations.jsonl",
  "n_train_examples": 900,
  "n_eval_examples": 100,
  "training": {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "epochs": 3,
    "learning_rate": 2e-4,
    "train_batch_size": 2,
    "grad_accumulation_steps": 4,
    "lr_scheduler_type": "cosine",
    "max_seq_len": 2048,
    "seed": 42
  },
  "run_name": "gemma4_31b_distil_v1",
  "trained_at": "2026-04-17T12:00:00Z"
}
```

`base_model` is the model the adapter was attached to. `target_model` is the model whose completion patterns it was trained to replicate (i.e. the source of the SFT data). `source_data` is the path to the generations JSONL relative to the repo root.

## Probe

The probe is a model trained on a particular layer and is intended to be applied to a particular other model. It is just a saved nn.Module at the end of the day, but it would still be good to have some standards about what we're passing around.

- `probe_config.json` - Proprietary file describing how to use the probe. See fields below
- `probe_head.safetensors` - This is the actual model itself, saved as a torch statedict


`probe_config.json` - Example fields

```json
{
	"probe_name": "BasicMLP",
	"base_model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
	"target_layer_name": "LlamaDecoderLayer",
	"layer_idx": 30,
	"model_config":
		{
		"layer_sizes": [4096, 1],
		"activation_function": "ReLU"
		}
}
```
Note that in the above the first entry in layer_sizes is the hidden dimension `d_model` of the underlying model.

The target layer name comes from 
`target_layer_name = base_model.model.layers[layer_idx].__class__.__name__`

## Sufficiency

These two above (which should be stored in the same repository) should be sufficient to load a unique LoRA / probe pair. Note that if multiple probes / LoRA adapters are to be used, then I'm going to have to come up with something cleverer but probably LoRA and probes in subdirectories.

## Data

There are two types of datasets which I'm interested in for benchmarking and training probes. Generations and annotations.

### Generations

These are the outputs from a model which are being trained on, and which need to have features annotated to get meaningful outputs.
They are formatted as JSONL objects where each line is a JSON in the following format:

```json
{
  "annotations": null,
  "annotator_model": null,
  "completion": "output text from the model"
  "id": "some prompt hash"
  "logprobs": [
	  [
		  {
		  "token": "token_in_question",
		  "logprob": "val"
		  }
	  ]
  ]
  "model": "model used to generate these completions",
  "question": "the prompt initially provided",
  "source_dataset": "dataset this prompt came from"
}
```

To be clear about the logprobs field, this is an array of arrays to represent the top logprobs from the model at that specific token. In keeping with the format with which the OpenAI API exposes the top logprobs. Each subarray corresponds to a single token in the completion. The contents of that subarray are the top logprobs when that token was being generated.


### Annotations

Annotated data at the character level, where both some value and the section relevant are provided. This is quite similar to the generations dataset with additional fields included. This is also a JSONL file where each line is a JSON in the following format:

```json
{
	"annotations": [
		{
			"span": "Tokens relevant",
			"label": "Supported / Not Supported / Insufficient Information",
			"verification_note": "justification",
			"index": 200
		}
	],
	"annotator_model": "model which did the annotating",
	"completion": "the original completion",
	"id": "some prompt hash", 
	"model": "model which generated the completions",
	"question": "the prompt initially provided",
	"source_dataset": "dataset this prompt came from"
}
```

As you can see, this includes all of the information in the generations.jsonl file and more, with the exception of the logprobs field (as this is typically an extremely large amount of data).

As you can see, this includes all of the information in the generations.jsonl file and more, with the exception of the logprobs field (as this is typically an extremely large amount of data).

The "annotations" field is read from the output of some annotator model which is fed the prompt, completion, and some instructions about what should and should not be labelled. 
- Each "span" should be a verbatim copy of some section of the completion. 
- The "label" should be one of "Supported / Not Supported / Insufficient Information" in the case of hallucinations only. For general policy checks, this could be as simple as a 1 or 0 label
- "verification_note" is optional. Only required for later auditing really
- The "index" is found after the span is provided using a fuzzy search across the completion. It should be the **character** index of where the span starts. Note that this is not provided by the annotator model (LLMs are very bad at giving precise locations in text)


### Saved Activations

Activations are sometimes stored in a persistent file system. This is required for some intensive data science tasks, or research work where it's very useful to be able to experiment on smaller datasets. 

These are stored as `.npz` for space efficiency (when saving numpy arrays, this can save around 30% of the size of the file compared with pickling approaches). By convention, these are saved with a file name like `Sha256(prompt)[:16]` so that jobs over a large number of prompts can be restarted safely, but this isn't required.

Data is stored in these .npz files in the following structure:

```JSON
{
	"hs_#": np.ndarray((num_tokens, d_model)),
	"token_ids": np.ndarray((num_tokens,)),
	"meta": {
		'id' : str("some prompt hash"),
		'dataset': str("dataset the prompt came from"), 
		'prompt': str("the prompt itself"),
		'completion': str("model completion to prompt"),
		'model': str("model used"),
		'layers': list[int],
		'all_tokens": bool,
	}
}
```

Note that the above structure may not be backwards compatible for previous runs or old code.

The hs_# arrays are each separate entries. So if the layers 2, 5, 6, and 10 (for example) were captured then this should have 4 arrays each with the same shape as the values to the keys "hs_2", "hs_5", "hs_6", "hs_10" respectively.

