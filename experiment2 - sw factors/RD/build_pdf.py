"""
build_pdf.py
============

Compile ``R&D factors.md`` to ``R&D factors.pdf``.

No pandoc on this box, so: Markdown -> HTML (python-markdown, with GFM tables /
fenced code) -> PDF via headless Chrome's ``--print-to-pdf``.  The intermediate
HTML is written next to the markdown so the relative image paths
(``output/RD/...``) resolve unchanged when Chrome loads the file.

Run::

    python build_pdf.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
MD_PATH = HERE / "R&D factors.md"
HTML_PATH = HERE / "R&D factors.html"
PDF_PATH = HERE / "R&D factors.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; max-width: 100%;
}
h1 { font-size: 21pt; color: #14223b; border-bottom: 3px solid #14223b;
     padding-bottom: 6px; margin: 0 0 4px 0; }
h2 { font-size: 15pt; color: #14223b; border-bottom: 1px solid #c9d2e0;
     padding-bottom: 3px; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 12pt; color: #243a5e; margin-top: 16px; page-break-after: avoid; }
h1 + p em, h2 + p em { color: #555; }
p, li { orphans: 3; widows: 3; }
code { font-family: "Consolas", "SFMono-Regular", monospace; font-size: 9.2pt;
       background: #f2f4f7; padding: 1px 4px; border-radius: 3px; color: #b03a2e; }
pre code { display: block; padding: 8px 10px; color: #1a1a1a; }
blockquote { border-left: 3px solid #8fa8c8; background: #f5f8fc; margin: 10px 0;
             padding: 6px 14px; color: #33415c; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #c9d2e0; padding: 4px 7px; text-align: left;
         vertical-align: top; }
th { background: #14223b; color: #fff; font-weight: 600; }
tr:nth-child(even) td { background: #f6f8fb; }
img { max-width: 100%; height: auto; display: block; margin: 12px auto;
      page-break-inside: avoid; border: 1px solid #e2e8f0; }
hr { border: none; border-top: 1px solid #d7dee8; margin: 18px 0; }
a { color: #1f4e8c; text-decoration: none; }
strong { color: #14223b; }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>R&D Factors — Software &amp; Services</title>
<style>{css}</style></head><body>
{body}
</body></html>"""


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise SystemExit("No Chrome/Edge found for PDF rendering.")


def main() -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"],
    )
    HTML_PATH.write_text(HTML_TEMPLATE.format(css=CSS, body=body), encoding="utf-8")
    print(f"Wrote HTML -> {HTML_PATH}")

    chrome = find_chrome()
    if PDF_PATH.exists():
        PDF_PATH.unlink()
    cmd = [
        chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer",
        "--virtual-time-budget=20000",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH.as_uri(),
    ]
    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or not PDF_PATH.exists():
        sys.stderr.write(res.stdout + "\n" + res.stderr + "\n")
        raise SystemExit(f"Chrome PDF render failed (rc={res.returncode}).")
    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"Wrote PDF -> {PDF_PATH}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
