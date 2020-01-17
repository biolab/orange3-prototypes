import re


def remove_ansi_control(content: str) -> str:
    """Remove ansi terminal control sequences from `content`."""
    return re.sub("\x1b[\\[0-9;]+m", "", content)
