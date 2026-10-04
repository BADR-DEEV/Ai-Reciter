from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_qaloon_lora import WhisperCollator


class CollatorLabelTests(unittest.TestCase):
    def processor(self):
        class Tokenizer:
            eos_token_id = 9
            def pad(self, values, **kwargs):
                length = max(len(v["input_ids"]) for v in values)
                ids = [v["input_ids"] + [9] * (length - len(v["input_ids"])) for v in values]
                masks = [[1] * len(v["input_ids"]) + [0] * (length - len(v["input_ids"])) for v in values]
                return SimpleNamespace(input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks))
        class Features:
            def pad(self, values, **kwargs):
                return {"input_features": torch.zeros((len(values), 80, 3000))}
        return SimpleNamespace(tokenizer=Tokenizer(), feature_extractor=Features())

    def test_strip_bos_once_mask_padding_only_retain_terminal_eos(self):
        output = WhisperCollator(self.processor(), 10)([
            {"input_features": [], "labels": [10, 11, 12, 13, 1, 9]},
            {"input_features": [], "labels": [10, 11, 12, 13, 1, 2, 9]}])
        self.assertEqual(output["labels"][0].tolist(), [11, 12, 13, 1, 9, -100])
        self.assertEqual(output["labels"][1].tolist(), [11, 12, 13, 1, 2, 9])

    def test_wrong_bos_and_missing_eos_fail_closed(self):
        collator = WhisperCollator(self.processor(), 10)
        for labels in ([8, 1, 9], [10, 1, 2]):
            with self.assertRaises(ValueError):
                collator([{"input_features": [], "labels": labels}])


if __name__ == "__main__":
    unittest.main()
