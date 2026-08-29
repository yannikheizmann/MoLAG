from molag.training.trainer import HuggingFaceAffinityTrainer
from molag.utils.registry import Registry


def test_hugging_face_trainer_is_registered() -> None:
    assert Registry.get("TrainerBase", "HuggingFaceAffinity") is HuggingFaceAffinityTrainer
