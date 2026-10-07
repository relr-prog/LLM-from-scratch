import re
import unicodedata

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{4,}")

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = SPACE_RE.sub(" ", text)
    text = BLANK_RE.sub("\n\n\n", text)
    return text.strip()

def iter_documents(root):
    from pathlib import Path
    from .data import extract_text, TEXT_EXTENSIONS
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            text = normalize_text(extract_text(path))
            if text:
                yield path, text
