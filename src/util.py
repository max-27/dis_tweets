from pathlib import Path
import os


def get_root_path():
    return Path(__file__).parent.parent.resolve()


