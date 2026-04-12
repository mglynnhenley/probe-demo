# Probe Demo

Generating probes to find violations of an arbitrary natural language policy.

The closed source model used should be `openai/gpt-5-4-nano`.

Model used to generate prompts potentially violating a policy will be Claude 4.5 Sonnet, although note that this can be changed with far fewer implications for the rest of the training pipeline.

Open-source model for running all of this will be [google/gemma-4-31B](https://huggingface.co/google/gemma-4-31B), the newly released google dense model.

## Quick Start

First, run the annotation pipeline to get the data that you want to use. In production, this will be replaced by a Supabase bucket which contains some data already loaded (such as the base responses of this model to the [LongFact++])(https://huggingface.co/datasets/obalcells/longfact-augmented-prompts) dataset, and which can be written to with specific completions for this question.

To train models using the train/ directory, run the following command.

```bash
uv run python train/train.py
```


