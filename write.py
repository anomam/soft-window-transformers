# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "torch>=2.3",
#     "typer",
# ]
# ///

from typer import Typer
from enum import Enum

# ---------------------------------------------------------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------------

APP = Typer(name="swt")


class TypeModel(str, Enum):
    soft_window = "soft-window"


@APP.command()
def handwrite(
    text: str,
    bias: float = 1.0,
    model_type: TypeModel = TypeModel.soft_window,
) -> None:
    print(f"Handwriting of: {text}")


if __name__ == "__main__":
    APP()
