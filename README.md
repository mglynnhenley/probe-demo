# Probe Demo

Generating probes to find violations of an arbitrary natural language policy.

The closed source model used should be `openai/gpt-5-4-nano`.

Model used to generate prompts potentially violating a policy will be Claude 4.5 Sonnet, although note that this can be changed with far fewer implications for the rest of the training pipeline.

Open-source model for running all of this will be [google/gemma-4-31B](https://huggingface.co/google/gemma-4-31B), the newly released google dense model.
