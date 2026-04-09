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
  "source_dataset": "dataset this prompt came from"
}
```

The ID is expected to be something like SHA256(question)[:16].
