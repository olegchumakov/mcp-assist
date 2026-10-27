# Voice access control: shared contract between Wyoming Speaker ID add-on and mcp_assist_se

`mcp_assist_se` verifies this contract in `custom_components/mcp_assist_se/speaker_tag.py`.
The add-on signs the same vectors. Tests freeze time at `1700000000`.

Goal: per-person access levels for the HA voice assistant, generic (no hardcoded names/devices), usable by anyone.
- Recognized people (from the Speaker ID add-on) and "unknown voices" as one group each get a level:
  NONE (nothing allowed), SELECTED (only chosen device classes), ALL (everything).
- Device classes: user-defined named groups built by picking HA entities / devices / areas / labels in the UI.
- Each person (and "unknown") is assigned level + (for SELECTED) a set of classes.
- Enforcement is in code (mcp_assist_se), never by prompt. The LLM never sees the identity tag.

Access control is an explicit switch, off by default. While it is off, the integration does not parse tags, does not need the secret or the add-on URL, and does not call the add-on.

## Identity tag (Speaker ID add-on -> transcript text)
The Wyoming STT transcript is the only channel, so the add-on prepends a signed tag to the transcript text:

  ⟦spk1;n=<name>;c=<conf>;t=<unix_ts>;r=<nonce>;s=<sig>⟧ <transcript>

- Delimiters U+27E6 / U+27E7. Fields separated by ';', key=value.
- n: person name, percent-encoded UTF-8 (RFC 3986 unreserved kept). Empty string = unknown/guest voice.
- c: identification confidence 0..1, 2 decimals (the add-on's own score; unknown -> its best score or 0).
- t: unix seconds when the tag was created.
- r: 16 hex chars random nonce.
- s: lowercase hex HMAC-SHA256, full 64 chars, key = shared secret (UTF-8 bytes),
  message = "spk1|" + n_encoded + "|" + c + "|" + t + "|" + r + "|" + transcript_text (exact text after the tag and single space).
- Only emitted when the add-on option for the shared secret is set; otherwise the add-on keeps its existing behavior
  (current label format option) unchanged. When the secret is set, the signed tag REPLACES the old human-readable label.
- Below the add-on's recognition threshold -> n is empty (unknown). The add-on's existing thresholds/margin decide.

Percent-encoding (this add-on, and what the vectors below use): UTF-8, uppercase hex digits, as
`urllib.parse.quote(name, safe="")`. Unreserved characters `A-Z a-z 0-9 - . _ ~` are left as-is.
Confidence is clamped to `0.00`..`1.00` before it is formatted (`1.4` → `1.00`, missing → `0.00`).
The verifier should take `n`, `c`, `t` and `r` from the tag as literal strings when rebuilding the HMAC
message (do not re-encode `n`).

## Verifier (mcp_assist_se)
- Parse a leading tag; verify HMAC with constant-time compare; reject if |now - t| > 120 s or nonce seen in last 10 min.
- Always strip the tag (valid or not) before the text reaches the LLM / history. Also strip any other ⟦...⟧ occurrences.
- Invalid / missing tag on voice input -> speaker = unknown.
- Typed input (HA app/frontend, context.user_id set, no tag) -> map HA user -> person via a configurable mapping; unmapped -> unknown (or configurable).
- Optional extra: min confidence in mcp config; below it -> unknown.

When the switch is off, this section does not run.

## People list
The add-on exposes its enrolled people so mcp_assist_se's settings can offer them in a dropdown:
  GET http://<addon-host>:<port>/api/people  -> {"people":[{"name":"..."}]}
  Auth: header "X-Speaker-Key: <shared secret>" (reject otherwise). mcp config has the add-on URL; if unreachable,
  allow typing names manually.

This add-on serves that list on port **8099** (the same port as its page). The port is not published
to the host. Home Assistant Core reaches it on the internal network. The exact URL is printed in the
add-on log (`People list: http://<hostname>:8099/api/people`). Underscores in the slug become hyphens,
with a repository prefix, for example `http://local-wyoming-speaker-id:8099/api/people` for a local
add-on. A missing or wrong header, or an empty password in the add-on, answers **401**. Do not map
port 8099 to the host: that would open the add-on page without the Home Assistant login.

The integration calls this only while access control is on and a URL is saved, and only from the
options flow (the result is cached). 401 and any other failure leave the typed names in place.

## Test vectors

Shared secret (UTF-8): `test-secret`
Unix time: `1700000000`
Nonce: `0123456789abcdef`

HMAC-SHA256, lowercase hex. Message is `spk1|` + encoded name + `|` + confidence + `|` + time + `|` + nonce + `|` + transcript.
Checked with `openssl dgst -sha256 -hmac test-secret`.

### 1. Canonical (ASCII name)

```
name:       Anna
n:          Anna
c:          0.73
transcript: turn on the light
message:    spk1|Anna|0.73|1700000000|0123456789abcdef|turn on the light
s:          4e3854eaed0f699014eead3a73f1a358618a699bb372d76d6715658adad72386
tag:        ⟦spk1;n=Anna;c=0.73;t=1700000000;r=0123456789abcdef;s=4e3854eaed0f699014eead3a73f1a358618a699bb372d76d6715658adad72386⟧ turn on the light
```

### 2. Cyrillic name (uppercase percent-encoding)

```
name:       Анна
n:          %D0%90%D0%BD%D0%BD%D0%B0
c:          0.81
transcript: включи свет
message:    spk1|%D0%90%D0%BD%D0%BD%D0%B0|0.81|1700000000|0123456789abcdef|включи свет
s:          a3a00a8802e8b162bf01114baa3debc0d1b8da4df5c49382d0f0e70a90ca5318
tag:        ⟦spk1;n=%D0%90%D0%BD%D0%BD%D0%B0;c=0.81;t=1700000000;r=0123456789abcdef;s=a3a00a8802e8b162bf01114baa3debc0d1b8da4df5c49382d0f0e70a90ca5318⟧ включи свет
```

### 3. Unknown voice (empty name, best score)

```
name:       (empty)
n:          (empty)
c:          0.20
transcript: hello
message:    spk1||0.20|1700000000|0123456789abcdef|hello
s:          cfbddbb902625d1c94fa99e6bfcc2e298f624f0ab858b76fe207d0ed85628d45
tag:        ⟦spk1;n=;c=0.20;t=1700000000;r=0123456789abcdef;s=cfbddbb902625d1c94fa99e6bfcc2e298f624f0ab858b76fe207d0ed85628d45⟧ hello
```
