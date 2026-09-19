"""Compare/CLI router entrypoint."""

from loss_compare import *  # noqa: F401,F403
from loss_compare import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()

