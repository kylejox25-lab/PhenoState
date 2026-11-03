"""Six-channel image encoders and self-supervised objectives."""

from importlib import import_module

from .mae import MAEConfig, MaskedAutoencoder
from .cnn_mae import CNNMaskedAutoencoder

# import_module supports the exact user-requested CA-MAE.py filename.
ChannelAgnosticMAE = import_module(".CA-MAE", __name__).ChannelAgnosticMAE
DiffMAE = import_module(".DiffMAE", __name__).DiffMAE

def build_model(config: MAEConfig | None = None) -> MaskedAutoencoder:
    """Shared dispatch for training, reconstruction, and embedding export."""
    config = config or MAEConfig()
    classes = {
        "mae": MaskedAutoencoder, "ca_mae": ChannelAgnosticMAE,
        "diffmae": DiffMAE, "cnn_mae": CNNMaskedAutoencoder,
    }
    return classes[config.model_type](config)


def model_from_checkpoint(state: dict) -> MaskedAutoencoder:
    """Restore the encoder/decoder from a PhenoState Lightning checkpoint."""
    if "pytorch-lightning_version" not in state or "model_config" not in state.get("hyper_parameters", {}):
        raise ValueError("Expected a PhenoState Lightning checkpoint (.ckpt)")
    model = build_model(MAEConfig(**state["hyper_parameters"]["model_config"]))
    model.load_state_dict({key.removeprefix("model."): value
                           for key, value in state["state_dict"].items() if key.startswith("model.")})
    return model

__all__ = [
    "MAEConfig", "MaskedAutoencoder", "ChannelAgnosticMAE", "DiffMAE",
    "CNNMaskedAutoencoder", "build_model", "model_from_checkpoint",
]
