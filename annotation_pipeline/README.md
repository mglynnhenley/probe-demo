# Annotation Pipeline

Generating and annotating synthetic data based on the policy provided.

## Generations

Generations data is stored in a JSONL file with this standard format.

```json
{
  "completion": "output text from the model",
  "id": "some prompt hash",
  "model": "model used to generate these completions",
  "question": "the prompt initially provided",
  "source_dataset": "dataset this prompt came from",
  "policy": "natural language policy being violated"
}
```

The ID is expected to be something like SHA256(question)[:16].

## Annotations

The annotations data is stored also as a JSONL file. This is designed to make token-level annotations possible.

In this context, the only annotations which are considered interesting are those which violate the policy explicitly.
This means that there is no need for a label. All annotations are those which explicitly and clearly violate the policy provided. 

It is assumed that there is no meaningful way to annotate non-violations of the policy without inadvertently capturing a general 'refusal' direction. Something where the policy is being followed for some policy like "Do not generate financial advice." would simply be the majority of generations + some explicit refusal to violate the policy. This signal is considered too sparse.

```json
{
	"annotations": [
		{
			"span": "Tokens relevant",
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
