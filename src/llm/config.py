from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.block_size < 1 or self.n_layer < 1 or self.n_head < 1:
            raise ValueError("model dimensions must be positive")
        if self.vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

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
    keep_checkpoints: int = 3

    def __post_init__(self):
        if min(self.batch_size, self.grad_accum_steps, self.max_steps, self.eval_interval, self.eval_steps, self.keep_checkpoints) < 1:
            raise ValueError("training counts must be positive")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be non-negative and max lr must be positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
