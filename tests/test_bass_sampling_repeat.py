import json
from pathlib import Path

from stemgenrt.trainer import TrainingConfig


def test_matched_repeat_uses_the_next_data_range_and_preserves_both_recipes():
    configs = Path(__file__).resolve().parents[1] / "configs"

    def read(filename):
        return json.loads((configs / filename).read_text())

    fixed = read("bass-low-lr-repeat.json")
    hard = read("bass-hard-solo-repeat.json")
    assert hard == {**fixed, "bass_solo_sampling": "alternating-release-hard-solo-v1"}
    for repeated, original_name in ((fixed, "bass-margin-low-lr-finetune.json"),
                                    (hard, "bass-hard-solo-finetune.json")):
        original = read(original_name)
        first = original["data_start"] + original["batch_size"] * original["steps"]
        assert first == 8768000 and repeated == {**original, "data_start": first}
        parsed = TrainingConfig(**repeated).validate()
        assert parsed.data_start + parsed.steps * parsed.batch_size == 8896000
