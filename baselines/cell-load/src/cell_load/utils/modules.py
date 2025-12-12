from ..data_modules import PerturbationDataModule

DATA_MODULE_DICT = dict(
    PerturbationDataModule=PerturbationDataModule,
)


def get_datamodule(name, kwargs, batch_size=None, cell_sentence_len=1):
    """
    Load data/lightning modules using TOML configuration.

    Args:
        name: Name of the data module (e.g., 'PerturbationDataModule')
        kwargs: Dictionary containing 'toml_config_path' and other parameters
        batch_size: Optional batch size override
        cell_sentence_len: Optional cell sentence length override
    """
    if name not in DATA_MODULE_DICT:
        raise ValueError(f"Unknown data module '{name}'")

    if batch_size is not None:
        kwargs["batch_size"] = batch_size
        kwargs["cell_sentence_len"] = cell_sentence_len

    # Ensure toml_config_path is provided
    if "toml_config_path" not in kwargs:
        raise ValueError("toml_config_path must be provided in kwargs")

    return DATA_MODULE_DICT[name](**kwargs)
