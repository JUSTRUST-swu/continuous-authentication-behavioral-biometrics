"""API server entrypoint."""

from api_server import *  # noqa: F401,F403
from api_server import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()

