#!/usr/bin/env python3
"""Convert practice exam Markdown files to styled PDFs with Pandoc."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


STYLE_TEMPLATE = r"""
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{fvextra}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{newunicodechar}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\setlength{\headheight}{28pt}
\setlength{\footskip}{24pt}
\setlength{\emergencystretch}{3em}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\begin{tabular}{@{}l@{}}Semester 1\\<<EXAM_LABEL>>\end{tabular}}
\fancyhead[R]{\small\begin{tabular}{r@{}}Machine Learning\\CITS5508\end{tabular}}
\fancyfoot[C]{\small Page \thepage\ of \pageref{LastPage}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\titlespacing*{\section}{0pt}{0.6em}{0.35em}
\titlespacing*{\subsection}{0pt}{0.9em}{0.35em}
\titlespacing*{\subsubsection}{0pt}{0.6em}{0.25em}
\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{0.5em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{0.5em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\bfseries}{\thesubsubsection}{0.5em}{}

\setlist[itemize]{leftmargin=1.35em,itemsep=0.15em,topsep=0.2em}
\setlist[enumerate]{leftmargin=1.65em,itemsep=0.15em,topsep=0.2em}
\fvset{breaklines=true,breakanywhere=true,fontsize=\small}
\renewcommand{\arraystretch}{1.15}
\setlength{\LTpre}{0.35em}
\setlength{\LTpost}{0.35em}
\newunicodechar{ℓ}{\ensuremath{\ell}}
\newunicodechar{≫}{\ensuremath{\gg}}
"""


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def label_for(markdown_file: Path) -> str:
    if "answer key" in markdown_file.stem.lower():
        return "Practice Exam Answers"
    return "Practice Exam"


def convert_file(
    markdown_file: Path,
    output_dir: Path,
    pdf_engine: str,
) -> Path:
    output_pdf = output_dir / f"{markdown_file.stem}.pdf"
    label = latex_escape(label_for(markdown_file))
    style = STYLE_TEMPLATE.replace("<<EXAM_LABEL>>", label)

    with tempfile.TemporaryDirectory(prefix="practice_exam_pdf_") as temp_dir:
        style_file = Path(temp_dir) / "practice_exam_style.tex"
        style_file.write_text(style, encoding="utf-8")

        command = [
            "pandoc",
            str(markdown_file),
            "--standalone",
            "--from",
            "markdown+tex_math_dollars+smart",
            "--pdf-engine",
            pdf_engine,
            "--include-in-header",
            str(style_file),
            "--highlight-style",
            "tango",
            "--metadata",
            f"pagetitle={markdown_file.stem}",
            "--variable",
            "papersize=a4",
            "--variable",
            "fontsize=11pt",
            "--variable",
            "mainfont=DejaVu Serif",
            "--variable",
            "sansfont=DejaVu Sans",
            "--variable",
            "monofont=DejaVu Sans Mono",
            "--variable",
            "geometry:top=2.1cm,bottom=2.2cm,left=2.1cm,right=2.1cm",
            "--output",
            str(output_pdf),
        ]
        subprocess.run(command, check=True)

    return output_pdf


def parse_args(argv: list[str]) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert every Markdown file in the practice exam directory to PDF."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "Practice Exams",
        help="Directory containing Markdown files. Defaults to exam/Practice Exams.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated PDFs. Defaults to the input directory.",
    )
    parser.add_argument(
        "--pdf-engine",
        default="xelatex",
        help="Pandoc PDF engine to use. Defaults to xelatex.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir).expanduser().resolve()

    if shutil.which("pandoc") is None:
        print("pandoc is required but was not found on PATH.", file=sys.stderr)
        return 1
    if shutil.which(args.pdf_engine) is None:
        print(f"{args.pdf_engine} is required but was not found on PATH.", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_files = sorted(input_dir.glob("*.md"))
    if not markdown_files:
        print(f"No Markdown files found in {input_dir}", file=sys.stderr)
        return 1

    for markdown_file in markdown_files:
        output_pdf = convert_file(markdown_file, output_dir, args.pdf_engine)
        print(f"Converted {markdown_file.name} -> {output_pdf.name}")

    print(f"Converted {len(markdown_files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
