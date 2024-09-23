"""
https://github.com/urchade/GLiNER
GLiNER: https://huggingface.co/papers/2311.08526
NuNER: https://huggingface.co/papers/2402.15343
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Literal

import GPUtil
import torch
from flupy import flu
from gliner import GLiNER
from loguru import logger
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

try:
    # Try to enable hf_transfer if available
    # - pip install huggingface_hub[hf_transfer])
    # - hf_transfer is a power-user that enables faster downloads from the Hub
    # https://github.com/GLiNER-project/GLiNER/issues/2907
    # https://github.com/huggingface/hf_transfer
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass

MODELS = {
    "GLiNER-S": "urchade/gliner_smallv2.1",
    "GLiNER-M": "urchade/gliner_mediumv2.1",
    "GLiNER-L": "urchade/gliner_largev2.1",
    "GLiNER-News": "EmergentMethods/gliner_medium_news-v2.1",  # long-context news fine-tune with improved zero-shot accuracy
    "GLiNER-PII": "urchade/gliner_multi_pii-v1",  # PII Detector variant
    "GLiNER-Bio": "urchade/gliner_large_bio-v0.1",  # Biomedical variant
    "GLiNER-Bird": "wjbmattingly/gliner-large-v2.1-bird",  # Bird attribution
    "NuNER-Zero": "numind/NuNER_Zero",  # * +3.1% more capable than GLiNER-large-v2.1
    "NuNER-Zero-4K": "numind/NuNER_Zero-4k",  # 4096 context window
    "NuNER-Zero-span": "numind/NuNER_Zero-span",  # NuNER Zero-span shows slightly better performance than NuNER Zero but cannot detect entities that are larger than 12 tokens.
}
"""
available models:
  - https://huggingface.co/urchade
  - https://huggingface.co/collections/numind/nunerzero-zero-shot-ner-662b59803b9b438ff56e49e2
"""

DEFAULT_MODEL = "urchade/gliner_smallv2.1"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CONFIG = {
    "num_steps": 10_000,  # N training iteration
    "train_batch_size": 2,  # batch size for training
    "eval_every": 1_000,  # evaluation/saving steps
    "save_directory": "checkpoints",  # where to save checkpoints
    "warmup_ratio": 0.1,  # warmup steps
    "device": DEVICE,
    "lr_encoder": 1e-5,  # learning rate for the backbone
    "lr_others": 5e-5,  # learning rate for other parameters
    "freeze_token_rep": False,  # freeze of not the backbone  ? Not sure what this comment means lol
    #       Parameters for set_sampling_params:
    "max_types": 25,  # maximum number of entity types during training
    "shuffle_types": True,  # if shuffle or not entity types
    "random_drop": True,  # randomly drop entity types
    "max_neg_type_ratio": 1,  # ratio of positive/negative types,
    #    1 mean 50%/50%, 2 mean 33%/66%, 3 mean 25%/75% ...
    "max_len": 384,  # maximum sentence length
}


class NERModel:
    """Named Entity Recognition model."""

    def __init__(
        self,
        name: str = "GLiNER-S",
        local_model_path: str = None,
        overwrite: bool = False,
        train_config: dict = TRAIN_CONFIG,
    ) -> None:
        """Initialize the NERModel.

        Args:
            name: The model name.
            local_model_path: The local model path.
            overwrite: Whether to overwrite the model path.
            train_config: The training config.
        """
        if name not in MODELS:
            raise ValueError(f"Invalid model name: {name}")
        # Define the model ID
        self.model_id: str = MODELS[name]

        # Create a models directory
        workdir = Path.cwd() / "models"
        workdir.mkdir(parents=True, exist_ok=True)
        if local_model_path is None:
            local_model_path = name
        else:
            local_model_path = (workdir / local_model_path).resolve()
        if Path(local_model_path).exists():
            import warnings

            warnings.warn(f"Model path already exists: {str(local_model_path)}")
            if overwrite is False:
                raise ValueError(f"Model path already exists: {str(local_model_path)}")

        # Define the local model path
        self.local_model_path: Path = local_model_path

        # Use GPU acceleration if available
        self.device: str = train_config.get("device", DEVICE)
        logger.info(f"Device: [{self.device}]")

        # Define the hyperparameters in a config variable
        self.train_config: SimpleNamespace = SimpleNamespace(**train_config)

        # Init empty for lazy loading of model weights
        self.model: GLiNER = None

    def __load_model_remote(self) -> None:
        """Actually load the model.

        Args:
          model_id: The model ID.
        """
        # available models: https://huggingface.co/urchade
        self.model = GLiNER.from_pretrained(self.model_id)

    def __load_model_local(self) -> None:
        """Load the model from a local path.

        Args:
          model_path: The model path.
        """
        try:
            local_model_path = str(Path(self.local_model_path).resolve())
            self.model = GLiNER.from_pretrained(
                local_model_path,
                local_files_only=True,
            )
        except Exception as e:
            logger.exception("Failed to load model from local path.", e)

    def load(self, mode: Literal["local", "remote", "auto"] = "auto") -> None:
        """Load the model.

        Args:
          model_id: The model ID.
        """
        if self.model is None:
            if mode == "local":
                self.__load_model_local()
            elif mode == "remote":
                self.__load_model_remote()
            elif mode == "auto":
                local_model_path = str(Path(self.local_model_path).resolve())
                if Path(local_model_path).exists():
                    self.__load_model_local()
                else:
                    self.__load_model_remote()
            else:
                raise ValueError(f"Invalid mode: {mode}")
            GPUtil.showUtilization()
            logger.info(
                f"Loaded model: [{self.model_id}] | N Params: [{self.model_param_count}] | [{self.model_size_in_mb}]"
            )
        else:
            logger.warning("Model already loaded.")

        logger.info(f"Moving model weights to: [{self.device}]")
        self.model = self.model.to(self.device)

    @property
    def model_size_in_bytes(self) -> int:
        """Returns the approximate size of the model parameters in bytes."""
        total_size = 0
        for param in self.model.parameters():
            # param.numel() returns the total number of elements in the parameter,
            # param.element_size() returns the size in bytes of an individual element.
            total_size += param.numel() * param.element_size()
        return total_size

    @property
    def model_param_count(self) -> str:
        """Returns the number of model parameters in billions."""
        return f"{sum(p.numel() for p in self.model.parameters()) / 1e9:,.2f} B"

    @property
    def model_size_in_mb(self) -> str:
        """Returns the string repr of the model parameter size in MB."""
        return f"{self.model_size_in_bytes / 1024**2:,.2f} MB"

    def train(
        self,
        train_data: list[dict[str, str]],
        eval_data: dict[str, list[Any]] = None,
    ) -> None:
        """Train the GLiNER model.

        Args:
          model: The GLiNER model.
          config: The hyperparameters.
          train_data: The training data.
          eval_data: The evaluation data.
        """
        if self.model is None:
            self.load()

        GPUtil.showUtilization()

        # Set sampling parameters from config
        self.model.set_sampling_params(
            max_types=self.train_config.max_types,
            shuffle_types=self.train_config.shuffle_types,
            random_drop=self.train_config.random_drop,
            max_neg_type_ratio=self.train_config.max_neg_type_ratio,
            max_len=self.train_config.max_len,
        )

        self.model.train()

        # Initialize data loaders
        self.train_loader = self.model.create_dataloader(
            train_data,
            batch_size=self.train_config.train_batch_size,
            shuffle=True,
        )

        # Optimizer
        self.optimizer = self.model.get_optimizer(
            self.train_config.lr_encoder,
            self.train_config.lr_others,
            self.train_config.freeze_token_rep,
        )

        pbar = tqdm(range(self.train_config.num_steps))

        if self.train_config.warmup_ratio < 1:
            num_warmup_steps = int(
                self.train_config.num_steps * self.train_config.warmup_ratio
            )
        else:
            num_warmup_steps = int(self.train_config.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.train_config.num_steps,
        )

        iter_train_loader = iter(self.train_loader)

        for step in pbar:
            try:
                x = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(self.train_loader)
                x = next(iter_train_loader)

            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(self.device)

            loss = self.model(x)  # Forward pass

            # Check if loss is nan
            if torch.isnan(loss):
                continue

            loss.backward()  # Compute gradients
            self.optimizer.step()  # Update parameters
            self.scheduler.step()  # Update learning rate schedule
            self.optimizer.zero_grad()  # Reset gradients

            description = f"step: {step} | epoch: {step // len(self.train_loader)} | loss: {loss.item():.2f}"
            pbar.set_description(description)

            if (step + 1) % self.train_config.eval_every == 0:

                self.model.eval()

                if eval_data is not None:
                    results, f1 = self.model.evaluate(
                        eval_data["samples"],
                        flat_ner=True,
                        threshold=0.5,
                        batch_size=12,
                        entity_types=eval_data["entity_types"],
                    )

                    logger.debug(f"Step={step} | F1 [{f1:.2f}]\n{results}")

                checkpoint_dir = Path(self.train_config.save_directory)
                if not checkpoint_dir.exists():
                    checkpoint_dir.mkdir(exist_ok=True, parents=True)

                self.model.save_pretrained(
                    f"{self.train_config.save_directory}/finetuned_{step}"
                )

                self.model.train()

        GPUtil.showUtilization()

        logger.success("Training complete!")

    def batch_predict(
        self,
        targets: List[str],
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.3,
        multi_label: bool = False,
        batch_size: int = 12,
    ) -> List[List[str]]:
        """Batch predict.

        Args:
          targets: The targets.
          labels: The labels.
          threshold: The threshold.
          batch_size: The batch size.

        Returns:
          The predictions.
        """

        if self.model is None:
            self.load()

        self.model.eval()
        predictions = []
        for i, batch in enumerate(tqdm(flu(targets).chunk(batch_size))):
            if i % 100 == 0:
                logger.debug(f"Predicting Batch [{i:,}]...")
            entities = self.model.batch_predict_entities(
                texts=batch,
                labels=labels,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )
            predictions.extend(entities)
        return predictions

    def save(self, file_name: str) -> None:
        """Save the model to a file.

        Args:
          file_name: The file name.
        """
        self.model.save_pretrained(file_name)

    def test(self) -> None:
        """Test the model."""
        examples = ["hello John, your reservation is at 6pm"]
        predictions = self.model.batch_predict_entities(
            examples,
            labels=["Person", "Time"],
            threshold=0.5,
        )
        logger.info(predictions)
