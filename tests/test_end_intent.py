"""Stop words ignore a leading speaker label and still match with punctuation."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "custom_components" / "mcp_assist_se"
sys.path.insert(0, str(ROOT / "custom_components"))

if "mcp_assist_se" not in sys.modules:
    package = types.ModuleType("mcp_assist_se")
    package.__path__ = [str(PACKAGE)]
    package.__package__ = "mcp_assist_se"
    sys.modules["mcp_assist_se"] = package

from mcp_assist_se.end_intent import (
    detect_user_ending_intent,
    strip_leading_speaker_label,
)

# Mixed on purpose: the check is the configured list, not one language.
END_WORDS = "stop, cancel, отбой, отмена, стоп, thank you, до свидания, bye, no, ok, thanks"

SIGNED = (
    "⟦spk1;n=Ada;c=0.91;t=1700000000;r=0123456789abcdef;"
    "s=ab⟧"
)


class EndIntentTests(unittest.TestCase):
    def _ends(self, text: str, agent_name: str = "") -> bool:
        return detect_user_ending_intent(text, END_WORDS, agent_name)

    def test_bare_stop_word(self):
        self.assertTrue(self._ends("отбой"))
        self.assertTrue(self._ends("Stop"))
        self.assertTrue(self._ends("  cancel  "))

    def test_labeled_stop_word(self):
        self.assertTrue(self._ends("[говорит: Олег] отбой"))
        self.assertTrue(self._ends("  [Speaker: Mary Ann]\tstop  "))
        self.assertTrue(self._ends(f"{SIGNED} отбой"))
        self.assertTrue(self._ends(f"  {SIGNED}   stop"))
        # A space inside the signed tag must not count as extra words.
        self.assertTrue(self._ends("⟦spk1; n=Ann Marie; c=0.80⟧ отмена"))

    def test_labeled_stop_phrase_with_punctuation(self):
        self.assertTrue(self._ends("[говорит: Олег] до свидания!"))
        self.assertTrue(self._ends("[speaker: Ada] thank you."))
        self.assertTrue(self._ends(f"{SIGNED} отбой."))
        self.assertTrue(self._ends("отбой!"))
        self.assertTrue(self._ends("[jx] thanks?"))

    def test_labeled_normal_phrase_is_not_an_ending(self):
        original = "[говорит: Олег] включи свет"
        self.assertFalse(self._ends(original))
        # Detection does not rewrite the transcript. The label stays available
        # for the model and for per-person access control.
        self.assertEqual(original, "[говорит: Олег] включи свет")
        self.assertEqual(strip_leading_speaker_label(original), "включи свет")
        self.assertFalse(self._ends(f"{SIGNED} turn on the kitchen light"))
        self.assertFalse(self._ends("[speaker: Ada] no turn on the light"))
        self.assertFalse(self._ends("[говорит: Олег] стоп включи свет"))

    def test_label_only_and_empty_are_not_endings(self):
        self.assertFalse(self._ends("[говорит: Олег]"))
        self.assertFalse(self._ends("   [speaker: Ada]   "))
        self.assertFalse(self._ends(SIGNED))
        self.assertFalse(self._ends(f"  {SIGNED}  "))
        # The name inside the label is a stop word. The label alone is not.
        self.assertFalse(self._ends("[говорит: стоп]"))
        self.assertFalse(self._ends(""))
        self.assertFalse(self._ends("   "))
        self.assertFalse(detect_user_ending_intent(None, END_WORDS))
        self.assertFalse(detect_user_ending_intent("stop", ""))
        self.assertFalse(detect_user_ending_intent("stop", None))

    def test_label_does_not_use_the_extra_word(self):
        # "ok please" ends because "ok" is a stop word and "please" is the one
        # other word that is allowed. The speaker label must not take that slot.
        self.assertTrue(self._ends("ok please"))
        self.assertTrue(self._ends("[speaker: Ada] ok please"))
        self.assertTrue(self._ends(f"{SIGNED} ok please"))
        self.assertTrue(self._ends("[говорит: Олег] thank you, please"))

    def test_existing_stopwords_plus_one_rules(self):
        self.assertTrue(self._ends("no thanks"))
        self.assertTrue(self._ends("no thank you"))
        self.assertTrue(self._ends("bye Jarvis", agent_name="Jarvis"))
        self.assertTrue(self._ends("[говорит: Олег] bye Jarvis", agent_name="Jarvis"))
        self.assertTrue(self._ends("ok please"))
        self.assertFalse(self._ends("no turn on light"))
        self.assertFalse(self._ends("turn on the light"))

    def test_label_later_in_the_sentence_is_kept(self):
        text = "turn on the [kitchen] light"
        self.assertEqual(strip_leading_speaker_label(text), text)
        self.assertFalse(self._ends(text))


if __name__ == "__main__":
    unittest.main()
