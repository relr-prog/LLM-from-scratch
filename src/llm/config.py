from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0

@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accum_steps: int = 8
    max_steps: int = 5000
    eval_interval: int = 250
    eval_steps: int = 50
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337
    checkpoint_dir: str = "checkpoints"
