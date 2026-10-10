# Memory and personality

MCP Assist SE 0.22.2 keeps two different kinds of long-term context.

## Facts

`remember_memory`, `recall_memories`, and `forget_memory` store facts and preferences until you ask to forget them. There is no expiry.

`remember_memory` does not overwrite by similarity. The first call stores a new fact when nothing similar is already there. If existing memories score at or above the candidate threshold (0.65 by default), the tool stores nothing and returns those rows with ids. The assistant then calls again:

| mode | What happens |
| --- | --- |
| `add` | Store the new text as its own memory, even if something similar exists. |
| `update` | Rewrite the `memory_id` you choose. Put the merged wording in `memory`. |
| `noop` | Store nothing. The fact is already known. |

An identical sentence is a noop without a second call. A different fact, such as «Олега зовут хозяином» next to «Меня зовут Олег», must be `add`. `set_personality_instruction` uses the same three modes.

`forget_memory` matches Russian inflections by stem, so «пиратов» and «пиратах» find «пират». It also deletes every sufficiently similar item when you say to forget everything about a subject, or when `forget_all_matches` is true. The forget threshold defaults to 0.80. A result of `DELETED NOTHING` means nothing was removed. A query also searches the personality block, and `remove_personality_instruction` searches memories when the instruction id is unknown.

`merge_duplicate_memories`, and the one-time cleanup after the model loads, merge only near-identical copies at or above 0.91. On rubert-tiny-turbo that covers name paraphrases (about 0.92) and leaves «Олега зовут хозяином» (about 0.90) and the two bright-white-light values (about 0.82) for the assistant to update explicitly. Each removed line is written to the log at warning. Rows with different categories are left alone. The one-time cleanup runs once. A previous run that merged nothing is not repeated on the next load; call `merge_duplicate_memories` if you want that 0.91 pass again. It still will not merge the 0.90 or 0.82 pairs.

Before each reply, when automatic retrieval is on, the integration embeds your current utterance (and, by default, the previous user turn) and injects the closest memories above the similarity threshold. They show up in the system prompt as **Remembered facts**. The model does not have to decide to call `recall_memories` for that. `recall_memories` remains the tool for "what do you remember?".

Embeddings are cached on the memory row together with the model id (`local:sergeyzh/rubert-tiny-turbo` or `openai:<model>`). Changing the model or the provider recomputes vectors. Existing rows that have no vector yet are filled in a few at a time, at most 16 per turn, after the model is loaded.

## Personality

Speaking style is not a memory. A request such as "talk to me like a pirate" or "говори со мной как пират" updates a separate block that is added to every system prompt.

| Tool | What it does |
| --- | --- |
| `view_personality` | List the instructions and their ids |
| `set_personality_instruction` | Add one instruction, or replace it when `instruction_id` is set |
| `remove_personality_instruction` | Delete by id or by subject, including Russian inflections. Also searches memories when the id is unknown |
| `merge_duplicate_memories` | Merge only near-identical copies at or above 0.91 into the newest entry |
| `reset_personality` | Clear the block. Requires `confirm: true` |

Limits, so the prompt stays small: 8 instructions, 240 characters each, 1200 characters in total.

If the model tries to store a style request with `remember_memory`, the tool refuses and tells it to call `set_personality_instruction` instead. The new style is in the prompt on the next turn. The tool result also tells the model to follow it in the current reply.

The block does not replace device-control rules. The assistant still has to discover entities and call `perform_action`.

## Local embeddings

Default provider: **local**. Default model: **sergeyzh/rubert-tiny-turbo**.

| | rubert-tiny-turbo | multilingual-e5-small |
| --- | --- | --- |
| Languages | Russian first | Many, including Russian and English |
| Download | about 118 MB | about 490 MB |
| RAM while loaded | about 180 MB | about 700 MB |
| Dimensions | 312 | 384 |
| Layers | 3 | 12 |
| Pooling | CLS token, then L2 norm | Mean of tokens, then L2 norm |
| Prefixes | none | `query: ` for the utterance, `passage: ` for memories |
| Suggested threshold | 0.78 | start at 0.78 and raise it if unrelated facts appear |

Files land in `/config/mcp_assist_se/models/<model-id>/` (`config.json`, `tokenizer.json`, `model.safetensors`). A truncated download is detected by file size and fetched again.

The model is not loaded at import time. When memory tools are enabled and the provider is local, startup schedules a background download and load. Inference runs in Home Assistant's executor, off the event loop. Per-query work after the model is warm is one short forward pass (the utterance plus any memories that do not yet have a vector). If the model cannot be downloaded or loaded, keyword ranking is used and the conversation continues. A failed load is retried after five minutes.

Home Assistant OS cannot install ONNX Runtime (no musllinux wheels). This path uses NumPy, `tokenizers`, and `safetensors` instead, which have musl and glibc wheels for x86_64 and aarch64. PyTorch is not required.

Sequences are capped at 64 tokens. That covers a short voice command and a stored fact. It is not a document embedder.

## Remote embeddings

Set the provider to OpenAI-compatible if you want `POST {base}/v1/embeddings`.

- Base URL empty: the profile's chat server URL is used. For this setup that is the OpenAI-compatible base, for example `https://openai.bothub.chat`.
- Model: whatever that server lists for embeddings. `text-embedding-3-small` is the default and is correct for OpenAI's own API. A proxy only works if it actually serves the id you enter. A chat model such as `gemini-3.1-flash-lite` is not an embedding model.
- Timeout: 2.5 seconds. Failure falls back to keywords.

The API key is the profile key, or the key of the profile whose server URL matches the embedding base URL.

## What you need to set

For a Russian voice assistant on an Intel N150 with an OpenAI-compatible chat backend:

1. Leave **Embedding provider** on **Local**.
2. Leave **Local embedding model** on **rubert-tiny-turbo**.
3. Leave automatic retrieval and the personality block on.
4. Do not set a remote embedding model unless you switch the provider. You do not need an embedding model on `openai.bothub.chat` for the default path.
5. Restart or reload once after the upgrade and wait for `Local embedding model ready` in the log. The first replies still work via keywords.

OpenClaw and Hermes profiles do not use this system prompt, so the personality block and automatic facts are not injected there. The OpenAI-compatible chat path used by `server_type: openai` does.
