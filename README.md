> **MCP Assist SE** — fork with the SearXNG search provider (upstream PR #51 merged) and renamed domain ``mcp_assist_se`` so it can coexist with upstream MCP Assist.
# MCP Assist for Home Assistant

A Home Assistant conversation agent that uses MCP (Model Context Protocol) for efficient entity discovery, achieving **95% token reduction** compared to traditional methods. Works with LM Studio, llama.cpp, Ollama, OpenAI, Google Gemini, Anthropic Claude, and OpenRouter.

## Key Features

- ✅ **95% Token Reduction**: Uses MCP tools for dynamic entity discovery instead of sending all entities
- ✅ **No Entity Dumps**: Never sends 12,000+ token entity lists to the LLM
- ✅ **Smart Entity Index**: Pre-generated system structure index (~400-800 tokens) for context-aware queries
- ✅ **Multi-Platform Support**: Works with LM Studio, llama.cpp, Ollama, OpenAI, Google Gemini, Anthropic Claude, and OpenRouter
- ✅ **Multilingual Support**: 21 languages with localized UI, system prompts, and speech detection
- ✅ **Multi-turn Conversations**: Maintains conversation context and history
- ✅ **Dynamic Discovery**: Finds entities by area, type, device_class, state, or name on-demand
- ✅ **Web Search Tools**: Optional DuckDuckGo or Brave Search integration for current information
- ✅ **Works with 1000+ Entities**: Efficient even with large Home Assistant installations
- ✅ **Multi-Profile Support**: Run multiple conversation agents with different models

## The Problem MCP Assist Solves

Traditional voice assistants send your **entire entity list** (lights, switches, sensors, etc.) to the LLM with every request. For a typical home with 200+ devices, this means:
- **12,000+ tokens** sent every time
- Expensive API costs (cloud LLMs)
- Slow response times
- Context window limitations
- Poor performance with large homes

## How MCP Assist Works

Instead of dumping all entities, MCP Assist:

1. **Starts an MCP Server** on Home Assistant that exposes entity discovery tools
2. **Your LLM connects** to the MCP server and gets access to these tools:
   - `get_index` - Get system structure index (areas, domains, device_classes, people, etc.)
   - `discover_entities` - Find entities by type, area, domain, device_class, or state
   - `get_entity_details` - Get current state and attributes
   - `perform_action` - Control devices
   - `run_script` - Execute scripts and return response data
   - `run_automation` - Trigger automations manually
   - `list_areas` - List all areas in your home
   - `list_domains` - List all entity types
   - `set_conversation_state` - Smart follow-up handling
3. **LLM uses the index for smart queries** - Understands what exists without full context dump
4. **LLM discovers on-demand** - Only fetches the entities it needs for each request
5. **Token usage drops** from 12,000+ to ~400 tokens per request

## Token Usage Comparison

| Method | Token Usage | Description |
|--------|-------------|-------------|
| **Traditional** | 12,000+ tokens | Sends all entity states |
| **MCP Assist** | ~400 tokens | Uses MCP tools for discovery |
| **Reduction** | **95%** | Massive efficiency gain |

## Smart Entity Index (v0.5.0+)

The Smart Entity Index provides a lightweight (~400-800 tokens) snapshot of your Home Assistant system structure, enabling context-aware queries without full entity dumps. The index includes areas, domains, device classes, people, calendars, zones, automations, and scripts. For entities without standardized device_class attributes (like custom integrations), LLM-powered gap-filling automatically infers semantic categories from naming patterns. This results in faster, more accurate queries that use ~95% fewer tokens compared to traditional entity dumps.

## Memory and personality (v0.22.4+)

Two additions for a voice assistant that should remember facts and keep a speaking style:

- **Personality block.** A short, persisted list of behaviour instructions (tone, persona, how to address you). It is injected into every system prompt, next to your configured system prompt. Change it by voice: "talk to me like a pirate" or "говори со мной как пират". Facts stay in normal memory.
- **Automatic memory retrieval.** Before each model call, stored memories are compared with what you just said (and, by default, your previous turn). The closest matches above a similarity threshold are injected as remembered facts. `recall_memories` and `forget_memory` still work when you ask explicitly.

Embeddings are computed **on this Home Assistant** by default. Nothing is sent to an embedding API unless you switch the provider.

The default model is [`sergeyzh/rubert-tiny-turbo`](https://huggingface.co/sergeyzh/rubert-tiny-turbo): a Russian sentence model, about 29M parameters, 312 dimensions, 3 layers. First run downloads **about 118 MB** into `/config/mcp_assist_se/models/` and then keeps **about 180 MB** of RAM while it is loaded. Loading is lazy and happens in a background thread, so the conversation is not blocked. Until the model is ready, retrieval falls back to the existing keyword ranker. Later turns use the cached vectors. Vectors are stored on each memory together with the model id, so changing the model recomputes them. Existing memories and settings are kept. Remembering a fact is one server-side decision: an exact or near-identical copy is left as it is, and `mode=add` cannot skip that check. After the model loads, only near-identical copies (0.91 or higher) are merged automatically, and each removed line is logged. Forgetting matches Russian word forms, so «пиратов» finds a memory that says «пират». A forget that matches nothing tells the model that nothing was deleted.

`perform_action` reads the entity before and after the service call. The result is `CHANGED` or `ALREADY` only when the state or the requested settings match. A call that leaves the device as it was is `NO CHANGE`. An unavailable target, or an activation with no settings while a remembered fact for that device contains settings, is `NOT DONE` and the service is not called. If the service call went through but the device has not reported the change yet, the result is `SENT` (since v0.24.2): the model says the command was sent instead of saying it failed. The model is told not to claim the action is done unless the result says `CHANGED` or `ALREADY`. Discovery and the entity index omit unavailable entities that have no friendly name, alias, or area, including ones whose only name is an IEEE address. Other unavailable entities stay in the list and are marked unavailable.

ONNX Runtime is not used. It has no musl wheels, and Home Assistant OS is Alpine, so it would fail to install. The integration runs the BERT network with NumPy plus the `tokenizers` and `safetensors` packages, which publish wheels for x86_64 and aarch64 on both musl and glibc. PyTorch is not installed.

Optional local model: `intfloat/multilingual-e5-small` (multilingual, 384 dimensions, **about 490 MB** download, **about 700 MB** RAM, 12 layers, slower on a small CPU such as an Intel N150). It expects `query:` / `passage:` prefixes; the integration adds those. Its similarity scores sit higher, so raise the threshold toward **0.75** if unrelated facts appear.

Optional remote provider: OpenAI-compatible `POST /v1/embeddings`. Leave the base URL empty to reuse the profile's chat server. Set the embedding model to one that server actually serves (`text-embedding-3-small` on OpenAI itself). A gateway such as `https://openai.bothub.chat` will not embed unless that model id is enabled there. If the call fails or times out (2.5 seconds), keyword matching is used and the reply still goes out.

See [docs/memory.md](docs/memory.md) for the tools and settings.

## Voice access (v0.24.0)

Per-person limits for voice commands. **Off by default.** While the switch is off, nothing changes: speaker marks are not read, the Speaker ID add-on is not required, and it is not contacted.

Turn it on in the profile options (the page after memory). The shared password and the add-on address are optional.

- Type names yourself, or point at the add-on (`GET /api/people` with header `X-Speaker-Key`, for example `http://local-wyoming-speaker-id:8099/api/people`). A 401 or any other failure keeps the typed names. The list is fetched when you save settings, not on each voice command.
- Device groups are named sets of entities, devices, rooms, or labels.
- Each person, and unknown voices as one group, gets Nothing, Selected groups, or Everything. Map Home Assistant users to a person for typed messages in the app.
- Nothing skips the model. An empty reply stays silent.
- State-changing tools check the target, including areas, devices, labels, floors, scripts, scenes, and automations, before the service runs. A denied call returns a refusal the model can say. Reading tools only return devices that person may see. The model never sees the speaker mark or the name.

The mark format and the HMAC test vectors are in [SPEC.md](SPEC.md). The canonical vector (secret `test-secret`, time `1700000000`) is Anna / `turn on the light`, signature `4e3854eaed0f699014eead3a73f1a358618a699bb372d76d6715658adad72386`.

## Voice access wording and names (v0.24.1)

- The voice-access settings are written in everyday language. Each field says what it is for, whether you have to fill it in, and what happens if you leave it empty. The address field only copies the list of known voices so you can pick people. Who is speaking still comes with each phrase when that address is empty.
- A person's name matches whether or not the capitals agree. «олег» is the same person as «Олег». Extra spaces are ignored. This applies to a spoken name, a typed Home Assistant user, the voice list, and duplicate names.

## Waiting for slow devices (v0.24.2)

Some devices take a few seconds to show that a command worked. A TV switched on by infrared, for example, often only counts as "on" once a smart plug reports its power draw.

- After a command, the assistant now watches each device it controlled and stops waiting the moment that device reports the expected state. Lights and other fast devices still answer right away.
- If a change is expected (turning on, opening, starting, a new brightness, and so on), it waits up to **Wait for slow devices (seconds)**, 10 seconds by default. You can change it from 2 to 30 in the shared MCP server settings. The change applies to the next command; no restart is needed.
- If the device still has not reported, the result is `SENT`, not a failure. The assistant says the command was sent and the device is switching on, and does not repeat it. "Didn't work" is kept for real errors and unavailable devices.
- Each device is checked on its own. One device changing quickly no longer ends the wait for the others.
- When you name one device, the assistant is told to control only that device and not add related ones, such as a streaming box next to the TV.
- With debug logging on, each checked device logs one line with what was expected, the outcome, and how many seconds it took.

## Works the same on old and new Home Assistant (v0.24.3)

- The oldest supported Home Assistant is now 2025.10. HACS will not offer this version to older installs.
- Tested on Home Assistant 2025.10 and 2026.10. Neither one logs deprecation warnings from this integration.
- A device that sits inside another device (a "child" device, new in Home Assistant 2026.9) now counts as being in its parent's room when it has no room of its own. Room lists, room counts, voice access checks, and "which room am I in" for a voice satellite all follow this.
- Entity aliases that only say "use the entity name" are now turned into the real name. Before, on Home Assistant 2026.4 and newer, such an alias could show up as odd text in the device list.
- Script fields are read from the public list of actions instead of the script internals. Scripts whose entity ID was renamed still show their fields.
- No settings change and no restart is needed beyond the update itself.

## Multilingual Support (v0.12.0+)

MCP Assist supports **21 languages** with localized configuration interfaces, language-aware system prompts, and region-specific speech detection patterns. The integration automatically detects your Home Assistant system language and provides appropriate defaults for system prompts, follow-up phrases, and end conversation words. Supported languages include: Arabic, Chinese (Simplified), Czech, Danish, Dutch, Finnish, Filipino, French, German, Greek, Hindi, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, and Turkish.

## Requirements

- Home Assistant 2025.10+
- One of:
  - **Local LLMs**: LM Studio v0.3.17+, llama.cpp, or Ollama
  - **Cloud LLMs**: OpenAI, Google Gemini, Anthropic Claude, or OpenRouter (API key required)
- Python 3.13+ (the version Home Assistant 2025.10 runs on)

### Home Assistant 2026.10

Version 0.22.0 records conversation tool results with the Home Assistant 2026.10 chat log API (`llm.ToolResult` on `ToolResultContent.result`). Home Assistant 2026.9 and older still receive the previous `tool_result` payload. Device and label lookups use `async_entries_for_area` and `async_entries_for_label`, and devices are looked up one at a time with `async_get`, so 2026.10 no longer logs the deprecated `device_registry.devices` mapping warning. Since v0.24.3 the room of an entity or device comes from `async_get_effective_area_id` when Home Assistant has it (2026.9+), and entity aliases come from `async_get_entity_aliases` (2026.4+). Older versions use the entity, then device, area and the plain alias list.

## Installation

### Add to HACS

[![Open your Home Assistant instance and add this repository to HACS.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=mike-nott&repository=mcp-assist&category=integration)

### Option A: HACS (Recommended)
1. Click the badge above to add this repository to HACS, or manually add it as a custom repository
2. Install "MCP Assist" from HACS
3. Restart Home Assistant

### Option B: Manual Installation
1. Copy the `custom_components/mcp_assist_se` folder to your Home Assistant `custom_components` directory
2. Restart Home Assistant

## Configuration

### 1. Add the Integration

1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "MCP Assist" and select it

### 2. Setup Flow

**Step 1 - Profile & Server Type:**
- Profile Name: Give your assistant a name (e.g., "Living Room Assistant")
- Server Type: Choose your LLM provider
  - **LM Studio** - Local, free, runs on your machine
  - **llama.cpp** - Local, free, official llama.cpp server
  - **Ollama** - Local, free, command-line based
  - **OpenAI** - Cloud, paid, GPT-5.2 series
  - **Google Gemini** - Cloud, paid/free tier, Gemini 3.0 series
  - **OpenRouter** - Cloud, multi-model gateway with access to 200+ models

**Step 2 - Server Configuration:**

*For Local Servers (LM Studio / llama.cpp / Ollama):*
- Server URL: Where your LLM server is running
  - LM Studio: `http://localhost:1234` (default)
  - llama.cpp: `http://localhost:8080` (default)
  - Ollama: `http://localhost:11434` (default)
- MCP Server Port: Port for the MCP server (default: 8090)

*For Cloud Providers (OpenAI / Gemini / Anthropic / OpenRouter):*
- API Key: Your provider API key (see below for setup)
- MCP Server Port: Port for the MCP server (default: 8090)

**Step 3 - Model & Prompts:**
- Model Name: Select from auto-loaded models or enter manually
- System Prompt: Customize the assistant's personality
- Technical Instructions: Advanced prompt for tool usage (pre-configured)

**Step 4 - Advanced Settings:**
- Temperature: Response randomness (0.0-1.0)
- Max Response Tokens: Maximum length of responses
- Response Mode: None / Smart / Always (conversation continuation behavior)
- Follow-up Phrases: Configurable phrases for pattern detection (default: "anything else, would you, can i", etc.)
- End Conversation Words: Words/phrases that end conversations (default: "bye, thanks, stop", etc.)
- Control Home Assistant: Enable/disable device control
- Max Tool Iterations: How many tool calls allowed per request
- Web Search Provider: Choose none, DuckDuckGo, or Brave Search
- Brave Search API Key: Your API key (if using Brave Search)
- Debug Mode: Extra logging for troubleshooting
- **Ollama Keep Alive** (Ollama only): Control how long models stay loaded in memory
  - `-1` = Keep loaded indefinitely
  - `0` = Unload immediately after response
  - `"5m"` = Keep for 5 minutes (default)
  - Duration strings like `"24h"`, `"168h"` also supported
- **Ollama Context Window** (Ollama only): Custom context window size (0 = use model default)

### 3. Set as Voice Assistant

1. In Home Assistant, go to **Settings** → **Voice Assistants**
2. Set your preferred assistant to your MCP Assist profile name
3. Test with commands!

## Usage Examples

### Basic Commands
- "Turn on the kitchen lights"
- "Turn off all the lights in the bedroom"
- "What's the temperature in the living room?"

### Multi-Turn Conversations
- **User**: "What lights are on?"
- **Assistant**: "The kitchen and living room lights are on."
- **User**: "Turn off the kitchen one"
- **Assistant**: "I've turned off the kitchen light."

### Complex Query Example

**User**: "Do we have a leak?"

**Behind the scenes:**
```
1. LLM calls get_index → Sees moisture sensors and water flow monitors exist in system
2. LLM calls discover_entities(device_class="moisture")
   → Returns: binary_sensor.bathroom_leak, binary_sensor.kitchen_sink_leak, binary_sensor.laundry_leak
3. LLM calls discover_entities(name_contains="water flow")
   → Returns: sensor.water_flow_rate
4. LLM calls get_entity_details for each sensor
   → bathroom leak: "off", kitchen leak: "off", laundry leak: "on", water flow: "2.5 gpm"
5. LLM synthesizes response
```

**Assistant**: "Yes, the laundry room leak sensor is detecting water and water is flowing at 2.5 gallons per minute. The bathroom and kitchen sensors are dry."

**Follow-up User**: "Turn off the water main"

**Behind the scenes:**
```
1. LLM calls discover_entities(name_contains="water main")
   → Returns: switch.water_main_shutoff
2. LLM calls perform_action(entity_id="switch.water_main_shutoff", action="turn_off")
   → Success
```

**Assistant**: "I've shut off the main water valve."

### Web Search (if enabled)
- "What's the weather forecast for tomorrow?"
- "Search for the latest Home Assistant updates"
- "What time does the store close?"

## Configuration Options

### Profile Settings
- **Profile Name**: Unique name for this assistant
- **Server Type**: LM Studio, Ollama (more coming)
- **Server URL**: Where your LLM is running
- **Model Name**: Which model to use

### Prompts
- **System Prompt**: Sets the assistant's personality and behavior
- **Technical Instructions**: Low-level instructions for tool usage (usually leave as default)

### Advanced Settings
- **Max Response Tokens**: Limit response length (default: 500)
- **Max History Messages**: How many conversation turns to remember (default: 10)
- **Max Tool Iterations**: Prevent infinite loops (default: 10)
- **Response Mode**:
  - **None**: Never ask follow-ups, end immediately
  - **Smart** (default): Contextual follow-ups when relevant, user can end with "bye"/"thanks"
  - **Always**: Natural conversational follow-ups, user can end with "bye"/"thanks"
- **Follow-up Phrases**: Comma-separated phrases for detecting when assistant wants to continue (configurable per profile)
- **End Conversation Words**: Comma-separated words/phrases for user-initiated ending (configurable per profile)
- **Enable Smart Entity Index**: Context-aware entity discovery with automatic gap-filling for uncommon devices (default: enabled)

### Temperature Settings

Temperature controls response randomness (0.0 = deterministic, 1.0 = creative). Different providers have different optimal values:

| Provider | Default | Reason |
|----------|---------|--------|
| **Gemini** | `1.0` | Google requires 1.0 for Gemini 3 to avoid "looping or degraded performance" |
| **OpenAI (GPT-4)** | `0.5` | Balanced for reliable tool calling |
| **OpenAI (GPT-5/o1)** | N/A | Reasoning models don't use temperature |
| **Anthropic Claude** | `0.5` | Works well across 0.5-1.0 range |
| **LM Studio / llama.cpp** | `0.5` | Lower temps improve tool calling accuracy |
| **Ollama** | `0.5` | Model-dependent, lower is safer for tools |
| **OpenRouter** | `0.5` | Depends on underlying model |

**Note**: You can always override these defaults in Advanced Settings. For Home Assistant voice control, lower temperatures (0.5-0.7) generally provide more consistent tool calling and accurate entity control.

### MCP Server Settings
- **MCP Server Port**: Default 8090 (change if port conflict)
- **Additional Allowed IPs/Ranges**: Whitelist Docker containers (e.g., `172.30.0.0/16`) or specific IPs for external MCP clients like Claude Code add-on
- **Wait for slow devices (seconds)**: The longest time the assistant waits for a device to confirm a command, such as a TV that takes a few seconds to turn on. Default 10, range 2-30. Fast devices answer as soon as they report. If time runs out, the assistant says the command was sent rather than that it failed.

### Web Search
- **Web Search Provider**: Choose between:
  - **None**: Search disabled
  - **DuckDuckGo**: Free web search (no API key required)
  - **Brave Search**: Requires API key from https://brave.com/search/api/
- **Brave Search API Key**: Required only if using Brave Search

### Shared vs Per-Profile Settings

MCP Assist has two types of settings:

**Per-Profile Settings** (independent per conversation agent):
- Model name, system prompt, technical instructions
- Temperature, max tokens, response mode
- Debug mode, max iterations
- Server URL (for local LLMs)

**Shared Settings** (affect ALL profiles):
- MCP server port
- Web search provider (none/duckduckgo/brave)
- Brave API key
- Allowed IPs/CIDR ranges
- Smart entity index (gap-filling)
- Persistent memory tools, automatic retrieval, embedding model, and the personality block

### Memory and personality settings

Open the integration options, finish the shared MCP page, then open **Memory and personality**:

| Setting | Default | What to do |
| --- | --- | --- |
| Enable personality block | On | Leave on. Say how you want to be spoken to. |
| Retrieve relevant memories automatically | On | Leave on so facts are injected without a tool call. |
| Memories to inject | 4 | Lower it if prompts get crowded. |
| Similarity threshold | 0.78 | Tuned for rubert-tiny-turbo. Related phrases score about 0.83-0.95; unrelated ones about 0.60-0.74. Raise it if the wrong fact is injected. |
| Similar-memory candidates | 0.65 | Existing rows at or above this are shown to the assistant. Nothing is overwritten until it chooses update. |
| Automatic merge threshold | 0.91 | One-time cleanup and the merge tool. Name paraphrases are about 0.92. A different fact at 0.90 is left alone. |
| Forget threshold | 0.80 | Semantic forget. Inflected forms such as «пиратов» also match «пират» by stem. |
| Include the previous user turn | On | Helps follow-ups such as "turn that on". |
| Embedding provider | Local | Keep Local. Switch to OpenAI-compatible only if you want `/v1/embeddings`. |
| Local embedding model | rubert-tiny-turbo | Right default for Russian on a small CPU. |
| Remote embedding model | text-embedding-3-small | Only for the remote provider. Must exist on that server. |
| Remote embedding base URL | empty | Empty uses the profile chat URL, for example `https://openai.bothub.chat`. |

After upgrading, reload the integration or restart Home Assistant once so the model download can start. The first conversations still work; they use keyword matching until the file is on disk and loaded. Watch the log for `Local embedding model ready`.

When you change shared settings in one profile's options, they apply to all profiles. This is intentional since all profiles share the same MCP server.

## Model Compatibility Guide

Not all LLM models support tool calling (function calling) equally well. **This integration works best with frontier models** (GPT-5.2, Claude Opus 4.5, Gemini 3 Flash) **or higher-spec local models**. Smaller models may struggle with complex multi-entity queries that require synthesizing large tool result sets.  

### Understanding Tool Calling Requirements

Tool calling (function calling) requires the model to:
1. Understand the user's request
2. Decide which tool to call
3. Format the tool arguments correctly as JSON
4. Interpret the tool results
5. Generate a natural response

**Factors affecting tool calling success**:
- **Model size**: Larger models (8B+) generally handle tool calling better
- **Model architecture**: Vision-Language (VL) models behave differently than standard models
- **Inference engine**: LM Studio and Ollama optimize models differently
- **Quantization level**: Q4 vs Q8 can affect instruction following

### Instruct vs Thinking/Reasoning Models

**Instruct Models** (e.g., `qwen3-8b-instruct`):
- Fast response times
- Best for simple, single-action requests ("turn on the kitchen lights")
- May struggle with complex queries requiring multiple tool calls
- Good for basic voice commands

**Thinking/Reasoning Models** (e.g., `qwen3-8b-thinking`):
- Slower response times (more deliberate reasoning)
- **Much better at complex requests** requiring multiple tool calls and context
- Handles multi-step queries reliably ("check all rooms for open windows, then turn off lights in those rooms")
- **Recommended for Home Assistant** where queries often involve discovery + action combinations

Choose the model type that best fits your use case. Thinking/reasoning models offer better reliability with complex multi-tool queries, while instruct models provide faster responses for simple commands.

### Recommended Models

**Consistently Reliable**:
- ✅ **Qwen3 VL 32B Instruct** - Excellent tool calling
- ✅ **Qwen3 30B A3B Instruct** - Very good tool calling
- ✅ **Qwen3 8B Instruct** - Good balance, works reliably
- ✅ **Anthropic Opus 4.5** - The very best at tool calling (cloud)
- ✅ **OpenAI GPT-5.2** - Excellent tool calling, very fast (cloud)
- ✅ **Google Gemini 3 Flash** - Excellent tool calling, fast, cost-effective (cloud)

### Testing Your Model

When tool calling **doesn't work**, you'll see:
- Model claims "I turned on the lights" but nothing happens
- No `perform_action` tool calls in the logs
- Actions don't execute, only narration

When tool calling **works correctly**, you'll see in logs:
- `discover_entities` called to find devices
- `perform_action` called to control them
- "✅ Successfully executed" messages
- Devices actually change state
- Tool calls visible in Settings → Voice Assistants → (profile) → Debug

### General Guidelines

**Start with larger models** (30B) if your hardware supports it - they work consistently across platforms.

**If using smaller models** (4B-8B), test thoroughly:
- Try a simple command like "turn on the kitchen lights"
- Check logs to verify tools are being called
- Confirm the device actually changes state
- If it doesn't work, try the same model on a different platform (LM Studio vs Ollama)

**Vision-Language (VL) models** are optimized for multimodal tasks and may have different tool calling behavior than standard models.

### Dynamic Model Switching

One of MCP Assist's features is **dynamic model switching** - you can change models in the configuration UI and it takes effect immediately without restarting Home Assistant. This makes it easy to:
- Test different models
- Switch between fast (Q4) and quality (Q8) quantizations
- Try new models as they're released

## Troubleshooting

### Integration Won't Start
- Check that the MCP port isn't already in use
- Verify Home Assistant has permission to bind to the port
- Check the Home Assistant logs for specific error messages

### LM Studio Can't Connect to MCP
- Ensure the MCP server is running (check integration status)
- Verify the MCP configuration in LM Studio is correct
- Check that the URL in LM Studio matches your MCP port (default: 8090)
- Restart LM Studio after changing MCP configuration

### Ollama Connection Issues
- Verify Ollama is running: `ollama list`
- Check the URL matches where Ollama is running (default: `http://localhost:11434`)
- Ensure the model is loaded in Ollama

### Cloud Provider Connection Issues
- Verify your API key is valid with your provider
- Check you have sufficient credits/quota remaining
- Check for rate limit errors in Home Assistant logs
- Try regenerating your API key if authentication fails
- Verify your internet connection is working
- Check Home Assistant logs for specific error codes

### No Response from Assistant
- Verify your LLM has a model loaded
- Check that the model name in the integration matches exactly
- Ensure your entities are exposed to the conversation assistant
- Check Home Assistant logs for API errors
- Enable Debug Mode for more detailed logging

### Poor Response Quality
- Try a different/larger model
- Adjust the temperature setting (lower = more focused)
- Ensure the model supports tool calling/function calling
- Check that Technical Instructions are not modified

### Tools Not Working
- Verify "Control Home Assistant" is enabled
- Check that entities are exposed (Settings → Voice Assistants → Expose)
- Look for MCP server errors in logs
- Ensure Max Tool Iterations isn't set too low

## Entity Exposure

The integration only discovers entities that are exposed to the "conversation" assistant. To expose entities:

1. Go to **Settings** → **Voice Assistants** → **Expose**
2. Select entities you want the assistant to control
3. The integration will automatically discover these when needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/mike-nott/mcp-assist/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mike-nott/mcp-assist/discussions)
- **Home Assistant Community**: [Community Forum](https://community.home-assistant.io/)
