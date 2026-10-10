"""Unit tests for memory ranking, personality text, and embedding helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "custom_components" / "mcp_assist_se"
sys.path.insert(0, str(ROOT / "custom_components"))

# The integration package imports Home Assistant. Load only the modules
# under test by registering the package without executing __init__.py.
if "mcp_assist_se" not in sys.modules:
    import types

    package = types.ModuleType("mcp_assist_se")
    package.__path__ = [str(PACKAGE)]
    package.__package__ = "mcp_assist_se"
    sys.modules["mcp_assist_se"] = package

from mcp_assist_se.const import DEFAULT_MEMORY_AUTO_MERGE_THRESHOLD
from mcp_assist_se.embedding_api import embeddings_url, parse_openai_embeddings
from mcp_assist_se.local_embedder import (
    append_download_chunk,
    apply_prefix,
    canonical_local_model,
    discard_partial_download,
    file_is_complete,
    finalize_partial_download,
    local_model_spec,
    prepare_partial_download,
)
from mcp_assist_se.memory_rank import (
    MAX_PERSONALITY_INSTRUCTION_CHARS,
    build_search_text,
    format_forget_result,
    format_personality_block,
    format_remembered_facts,
    keyword_auto_matches,
    keyword_score,
    looks_like_personality_request,
    memories_missing_embedding,
    plan_duplicate_merges,
    public_memory,
    rank_by_cosine,
    rank_memories,
    format_similarity_decision,
    parse_store_mode,
    select_similar_candidates,
    select_forget_matches,
    stem_token,
    validate_personality_text,
)
from mcp_assist_se.memory_settings import normalize_memory_options


class MemoryRankTests(unittest.TestCase):
    def test_recall_ranking_keeps_phrase_matches_first(self) -> None:
        memories = [
            {
                "id": "old",
                "text": "The kitchen light is warm",
                "category": None,
                "created_at": "2020-01-01T00:00:00+00:00",
            },
            {
                "id": "new",
                "text": "Current lighting is bright white light",
                "category": "home",
                "created_at": "2024-01-01T00:00:00+00:00",
            },
        ]
        ranked = rank_memories(
            memories, query="turn on bright white light", category=None
        )
        self.assertEqual(ranked[0]["id"], "new")
        self.assertGreater(
            keyword_score(memories[1], "turn on bright white light"),
            keyword_score(memories[0], "turn on bright white light"),
        )

    def test_auto_keyword_skips_a_single_weak_token(self) -> None:
        memories = [
            {
                "id": "a",
                "text": "The cat is named Barsik",
                "category": None,
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ]
        self.assertEqual(
            keyword_auto_matches(memories, "please turn the lamp", limit=4),
            [],
        )
        matched = keyword_auto_matches(
            memories, "what is the cat named", limit=4
        )
        self.assertEqual(matched[0]["id"], "a")
        self.assertNotIn("embedding", matched[0])

    def test_cosine_threshold_and_model_key(self) -> None:
        memories = [
            {
                "id": "close",
                "text": "bright",
                "embedding": [1.0, 0.0],
                "embedding_model": "local:rubert",
                "created_at": "2024-01-02T00:00:00+00:00",
            },
            {
                "id": "far",
                "text": "other",
                "embedding": [0.0, 1.0],
                "embedding_model": "local:rubert",
                "created_at": "2024-01-03T00:00:00+00:00",
            },
            {
                "id": "stale",
                "text": "bright but old model",
                "embedding": [1.0, 0.0],
                "embedding_model": "local:other",
                "created_at": "2024-01-04T00:00:00+00:00",
            },
        ]
        ranked = rank_by_cosine(
            memories,
            [1.0, 0.0],
            model_key="local:rubert",
            threshold=0.5,
            limit=4,
        )
        self.assertEqual([item["id"] for item in ranked], ["close"])
        missing = memories_missing_embedding(memories, "local:rubert", limit=8)
        self.assertEqual([item["id"] for item in missing], ["stale"])

    def test_public_memory_drops_vectors(self) -> None:
        cleaned = public_memory(
            {
                "id": "1",
                "text": "fact",
                "category": None,
                "created_at": "t",
                "embedding": [0.1],
                "embedding_model": "local:x",
            }
        )
        self.assertNotIn("embedding", cleaned)


class PersonalityTests(unittest.TestCase):
    def test_style_requests_are_not_facts(self) -> None:
        self.assertTrue(looks_like_personality_request("remember to talk to me like a pirate"))
        self.assertTrue(looks_like_personality_request("говори со мной как пират"))
        self.assertTrue(looks_like_personality_request("называй меня капитан"))
        self.assertFalse(
            looks_like_personality_request(
                "the current lighting is bright white light"
            )
        )
        self.assertFalse(
            looks_like_personality_request("текущий свет — яркий белый")
        )

    def test_personality_block_always_has_routing(self) -> None:
        empty = format_personality_block([])
        self.assertIn("set_personality_instruction", empty)
        self.assertIn("No personality instructions", empty)
        filled = format_personality_block(["Talk like a pirate"])
        self.assertIn("Talk like a pirate", filled)
        self.assertIn("perform_action", filled)

    def test_instruction_length_limit(self) -> None:
        self.assertIsNone(validate_personality_text("be brief"))
        self.assertIsNotNone(validate_personality_text(""))
        self.assertIsNotNone(validate_personality_text("x" * (MAX_PERSONALITY_INSTRUCTION_CHARS + 1)))

    def test_remembered_facts_section(self) -> None:
        self.assertEqual(format_remembered_facts([]), "")
        text = format_remembered_facts(
            [{"text": "Bright white light", "category": "lighting"}]
        )
        self.assertIn("Remembered facts", text)
        self.assertIn("[lighting] Bright white light", text)


class EmbeddingHelperTests(unittest.TestCase):
    def test_prefixes(self) -> None:
        rubert = local_model_spec("sergeyzh/rubert-tiny-turbo")
        e5 = local_model_spec("multilingual-e5-small")
        self.assertEqual(apply_prefix("свет", rubert, "query"), "свет")
        self.assertEqual(apply_prefix("свет", e5, "query"), "query: свет")
        self.assertEqual(apply_prefix("query: свет", e5, "query"), "query: свет")
        self.assertEqual(apply_prefix("свет", e5, "passage"), "passage: свет")
        self.assertEqual(canonical_local_model("nope"), "sergeyzh/rubert-tiny-turbo")

    def test_search_text_includes_previous_turn(self) -> None:
        self.assertEqual(
            build_search_text("turn it on", "bright white light", include_previous=True),
            "turn it on\nbright white light",
        )
        self.assertEqual(
            build_search_text("turn it on", "bright white light", include_previous=False),
            "turn it on",
        )

    def test_openai_parser_respects_index(self) -> None:
        payload = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }
        self.assertEqual(parse_openai_embeddings(payload), [[1.0, 0.0], [0.0, 1.0]])
        self.assertIsNone(parse_openai_embeddings({"data": []}))
        self.assertTrue(embeddings_url("https://openai.bothub.chat").endswith("/v1/embeddings"))
        self.assertTrue(embeddings_url("https://api.openai.com/v1").endswith("/v1/embeddings"))
        self.assertNotIn("/v1/v1/", embeddings_url("https://api.openai.com/v1"))

    def test_options_defaults_do_not_require_old_keys(self) -> None:
        normalized = normalize_memory_options({})
        self.assertTrue(normalized["enable_personality"])
        self.assertTrue(normalized["memory_auto_retrieve"])
        self.assertEqual(normalized["local_embedding_model"], "sergeyzh/rubert-tiny-turbo")
        self.assertEqual(normalized["embedding_provider"], "local")
        self.assertEqual(normalized["memory_top_k"], 4)
        self.assertEqual(normalized["memory_candidate_threshold"], 0.65)
        self.assertEqual(normalized["memory_auto_merge_threshold"], 0.91)
        self.assertEqual(normalized["memory_forget_threshold"], 0.80)
        clamped = normalize_memory_options(
            {
                "memory_top_k": 99,
                "memory_similarity_threshold": 4,
                "local_embedding_model": "multilingual-e5-small",
                "embedding_provider": "nope",
            }
        )
        self.assertEqual(clamped["memory_top_k"], 8)
        self.assertEqual(clamped["memory_similarity_threshold"], 1.0)
        self.assertEqual(clamped["embedding_provider"], "local")
        self.assertEqual(
            clamped["local_embedding_model"], "intfloat/multilingual-e5-small"
        )


class TinyBertForwardTests(unittest.TestCase):
    def test_forward_is_normalized_cls(self) -> None:
        import numpy as np

        from mcp_assist_se.local_embedder import BertTextEmbedder, LocalModelSpec

        rng = np.random.default_rng(0)
        hidden = 8
        layers = 1
        vocab = 20
        intermediate = 16
        heads = 2
        positions = 16
        weights = {
            "embeddings.word_embeddings.weight": rng.normal(size=(vocab, hidden)).astype(np.float32),
            "embeddings.position_embeddings.weight": rng.normal(size=(positions, hidden)).astype(np.float32),
            "embeddings.token_type_embeddings.weight": rng.normal(size=(2, hidden)).astype(np.float32),
            "embeddings.LayerNorm.weight": np.ones(hidden, dtype=np.float32),
            "embeddings.LayerNorm.bias": np.zeros(hidden, dtype=np.float32),
        }
        prefix = "encoder.layer.0."
        for name, shape in (
            ("attention.self.query", (hidden, hidden)),
            ("attention.self.key", (hidden, hidden)),
            ("attention.self.value", (hidden, hidden)),
            ("attention.output.dense", (hidden, hidden)),
            ("intermediate.dense", (intermediate, hidden)),
            ("output.dense", (hidden, intermediate)),
        ):
            weights[prefix + name + ".weight"] = rng.normal(size=shape).astype(np.float32) * 0.02
            weights[prefix + name + ".bias"] = np.zeros(shape[0], dtype=np.float32)
        for name in ("attention.output.LayerNorm", "output.LayerNorm"):
            weights[prefix + name + ".weight"] = np.ones(hidden, dtype=np.float32)
            weights[prefix + name + ".bias"] = np.zeros(hidden, dtype=np.float32)

        spec = LocalModelSpec(
            model_id="tiny",
            choice_key="tiny",
            repo="tiny",
            pooling="cls",
            query_prefix="",
            passage_prefix="",
            dim=hidden,
            max_length=8,
            download_mb=1,
            ram_mb=1,
            files={},
            summary="tiny",
        )
        model = BertTextEmbedder(
            weights,
            tokenizer=None,
            spec=spec,
            config={
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "intermediate_size": intermediate,
                "layer_norm_eps": 1e-12,
                "pad_token_id": 0,
            },
        )
        input_ids = np.array([[2, 5, 7, 0]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 0]], dtype=np.float32)
        hidden_state = model._forward(input_ids, mask)
        self.assertEqual(hidden_state.shape, (1, 4, hidden))
        self.assertTrue(np.isfinite(hidden_state).all())
        import math

        sample = np.array([-1.5, 0.0, 0.8, 2.2], dtype=np.float32)
        got = model._gelu(sample)
        expected = 0.5 * sample * (1.0 + np.vectorize(math.erf)(sample / math.sqrt(2.0)))
        self.assertTrue(np.allclose(got, expected, atol=1e-6))


class ForgetAndDedupeTests(unittest.TestCase):
    def test_russian_inflection_matches_the_stored_stem(self) -> None:
        self.assertEqual(stem_token("пиратов"), "пират")
        self.assertEqual(stem_token("пиратах"), "пират")
        self.assertEqual(stem_token("пират"), "пират")
        pirate = {
            "id": "pirate",
            "text": "Ты должна разговаривать как пират.",
            "category": None,
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        door = {
            "id": "door",
            "text": "Дверь спальни закрыта.",
            "category": None,
            "created_at": "2024-01-02T00:00:00+00:00",
        }
        # The old ranker requires the query token to be a substring, so the
        # genitive "пиратов" never hits the nominative "пират".
        self.assertEqual(rank_memories([pirate], query="пиратов", category=None), [])
        for query in ("пиратов", "пиратах", "забудь всё про пиратов"):
            matched = select_forget_matches(
                [pirate, door],
                query,
                similarities=None,
                threshold=0.80,
            )
            self.assertEqual([item["id"] for item in matched], ["pirate"], query)

    def test_forget_everything_deletes_every_similar_item(self) -> None:
        memories = [
            {
                "id": "style",
                "text": "Ты должна разговаривать как пират.",
                "category": None,
                "created_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "id": "flag",
                "text": "Пиратский флаг висит в кабинете.",
                "category": None,
                "created_at": "2024-01-02T00:00:00+00:00",
            },
            {
                "id": "door",
                "text": "Дверь спальни закрыта.",
                "category": None,
                "created_at": "2024-01-03T00:00:00+00:00",
            },
        ]
        matched = select_forget_matches(
            memories,
            "забудь всё про пиратов",
            similarities={"style": 0.81, "flag": 0.84, "door": 0.67},
            threshold=0.80,
        )
        self.assertEqual({item["id"] for item in matched}, {"style", "flag"})
        one = select_forget_matches(
            memories,
            "пиратов",
            delete_all=False,
            similarities={"style": 0.81, "flag": 0.90, "door": 0.67},
            threshold=0.80,
        )
        self.assertEqual([item["id"] for item in one], ["flag"])
        self.assertIn("DELETED NOTHING", format_forget_result([], [], searched_personality=True, searched_memories=True))

    def test_similar_candidates_do_not_choose_a_replacement(self) -> None:
        items = [
            {
                "id": "name",
                "text": "Меня зовут Олег",
                "category": None,
                "created_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "id": "title",
                "text": "Олега зовут хозяином",
                "category": None,
                "created_at": "2024-02-01T00:00:00+00:00",
            },
            {
                "id": "light",
                "text": "Яркий белый свет: 100%, 2000К",
                "category": "light",
                "created_at": "2024-01-02T00:00:00+00:00",
            },
            {
                "id": "door",
                "text": "Дверь спальни закрыта.",
                "category": None,
                "created_at": "2024-03-01T00:00:00+00:00",
            },
        ]
        scores = {"name": 0.92, "title": 0.90, "light": 0.60, "door": 0.63}
        candidates = select_similar_candidates(
            items,
            "Пользователя зовут Олег.",
            scores,
            0.65,
        )
        self.assertEqual([item["id"] for item in candidates], ["name", "title"])
        self.assertNotIn("light", {item["id"] for item in candidates})
        decision = format_similarity_decision(
            "memory", "Пользователя зовут Олег.", candidates
        )
        self.assertIn("NOT STORED", decision)
        self.assertIn("mode=add", decision)
        self.assertIn("mode=update", decision)
        self.assertIn("mode=noop", decision)
        self.assertIn("хозяином", decision)
        self.assertEqual(parse_store_mode(None), None)
        self.assertEqual(parse_store_mode("UPDATE"), "update")
        self.assertEqual(parse_store_mode("nope"), "invalid")
        # A scored row under the candidate line is not promoted by stem overlap.
        hidden = select_similar_candidates(
            items,
            "Пользователя зовут Олег.",
            {"name": 0.50},
            0.65,
        )
        self.assertEqual(hidden, [])

    def test_automatic_merge_stays_above_different_facts(self) -> None:
        items = [
            {
                "id": "old-name",
                "text": "Меня зовут Олег",
                "category": None,
                "created_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "id": "new-name",
                "text": "Пользователя зовут Олег.",
                "category": None,
                "created_at": "2024-06-01T00:00:00+00:00",
            },
            {
                "id": "title",
                "text": "Олега зовут хозяином",
                "category": None,
                "created_at": "2024-07-01T00:00:00+00:00",
            },
            {
                "id": "old-light",
                "text": "Яркий белый свет: 100%, 2000К",
                "category": None,
                "created_at": "2024-01-02T00:00:00+00:00",
            },
            {
                "id": "new-light",
                "text": "Яркий белый свет для ленты в гостиной: яркость 255",
                "category": None,
                "created_at": "2024-08-01T00:00:00+00:00",
            },
        ]
        matrix = {
            "old-name": {"new-name": 0.92, "title": 0.90, "old-light": 0.60, "new-light": 0.58},
            "new-name": {"old-name": 0.92, "title": 0.90, "old-light": 0.60, "new-light": 0.58},
            "title": {"old-name": 0.90, "new-name": 0.90, "old-light": 0.55, "new-light": 0.55},
            "old-light": {"new-light": 0.82, "old-name": 0.60, "new-name": 0.60, "title": 0.55},
            "new-light": {"old-light": 0.82, "old-name": 0.58, "new-name": 0.58, "title": 0.55},
        }
        self.assertEqual(DEFAULT_MEMORY_AUTO_MERGE_THRESHOLD, 0.91)
        groups = plan_duplicate_merges(
            items, matrix, DEFAULT_MEMORY_AUTO_MERGE_THRESHOLD
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["keep"]["id"], "new-name")
        self.assertEqual([item["id"] for item in groups[0]["remove"]], ["old-name"])
        removed = {item["id"] for group in groups for item in group["remove"]}
        self.assertNotIn("title", removed)
        self.assertNotIn("old-light", removed)
        self.assertNotIn("new-light", removed)

    def test_merge_keeps_the_newest_text(self) -> None:
        items = [
            {
                "id": "old-name",
                "text": "Меня зовут Олег",
                "category": None,
                "created_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "id": "new-name",
                "text": "Пользователя зовут Олег.",
                "category": None,
                "created_at": "2024-06-01T00:00:00+00:00",
            },
            {
                "id": "door",
                "text": "Дверь спальни закрыта.",
                "category": None,
                "created_at": "2024-07-01T00:00:00+00:00",
            },
        ]
        groups = plan_duplicate_merges(
            items,
            {
                "old-name": {"new-name": 0.92, "door": 0.65},
                "new-name": {"old-name": 0.92, "door": 0.64},
                "door": {"old-name": 0.65, "new-name": 0.64},
            },
            0.80,
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["keep"]["id"], "new-name")
        self.assertEqual([item["id"] for item in groups[0]["remove"]], ["old-name"])


class PartialDownloadTests(unittest.TestCase):
    def test_chunks_are_written_off_to_the_side_then_renamed(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "model.safetensors"
            partial = destination.with_suffix(destination.suffix + ".partial")
            prepare_partial_download(partial)
            append_download_chunk(partial, b"abc")
            append_download_chunk(partial, b"de")
            finalize_partial_download(partial, destination, 5)
            self.assertTrue(file_is_complete(destination, 5))
            self.assertFalse(partial.exists())
            self.assertEqual(destination.read_bytes(), b"abcde")

    def test_wrong_size_removes_the_partial(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "model.safetensors"
            partial = destination.with_suffix(destination.suffix + ".partial")
            prepare_partial_download(partial)
            append_download_chunk(partial, b"short")
            with self.assertRaises(RuntimeError):
                finalize_partial_download(partial, destination, 100)
            discard_partial_download(partial)
            self.assertFalse(destination.exists())
            self.assertFalse(partial.exists())


if __name__ == "__main__":
    unittest.main()
