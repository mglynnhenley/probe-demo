# Backend API

API structured to be compatible with the OpenAI API.

## Design

The OpenAI SDK exposes the `/models` and the `/chat/completions` endpoints. These are also exposed through this server.

Unlike the OpenAI typical format, these API calls also send probe values through the API. These are available in a model extras field exposed by the default responses object:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model = "google/gemma-4-31B-it",
    message = [
        {"role": "system", "content": "You are a pirate. End every reply with 'Arrr!'."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
)

scores = (response.model_extra or {}).get("probe_scores")
content = response.choices[0].message.content

print(content)
print(scores)
```

This is a typical OpenAI API call, but for the fact that there is a `response.model_extra.probe_scores` field.


## Compatability

This is not quite a drop-in implementation of the OpenAI API. There are fields which 
