# Copyright (C) 2026 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT
"""A gallery of the tutorials, for the front page: one card per tutorial that writes a picture.

The ``tutorial-gallery`` directive makes a card for each tutorial in the table of contents that
writes an animation (``plotter.open_gif("name.gif")``) or a screenshot
(``plotter.screenshot("name.png")`` or ``plotter.show(screenshot="name.png")``). The card shows the
animation if there is one, and otherwise the last screenshot, has the tutorial's title, and links
to it. The cards are made from the tutorials' sources, as the front page is read before the
tutorials run.

The tutorials write their pictures while the book executes them, so the front page refers to copies
under ``_static/gallery``, made here once the build has finished. A card may also name its picture
without an extension, ``_static/gallery/amr.*``, which is resolved to ``amr.gif`` if a tutorial
wrote one, and otherwise to ``amr.png``.
"""

import json
import re
import shutil
from pathlib import Path
from typing import ClassVar

import yaml
from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

# Where the tutorials write their pictures, and the page with the gallery, relative to the book's root
CHAPTERS = "chapter*"
FRONT_PAGE = "index"
PATTERNS = ("*.gif", "*.png")
# For a picture named without its extension, in order of preference
PREFERENCE = (".gif", ".png")

_GIF = re.compile(r"open_gif\(\s*[\"']([^\"']+\.gif)[\"']")
_PNG = re.compile(r"screenshot\s*[(=]\s*[\"']([^\"']+\.png)[\"']")


def _toc_files(toc: dict) -> list[tuple[str, str | None]]:
    """The files of a Jupyter Book table of contents, in order, each with the file it is a section
    of, if any."""
    files: list[tuple[str, str | None]] = []

    def walk(entry, parent: str | None):
        if isinstance(entry, list):
            for item in entry:
                walk(item, parent)
            return
        if not isinstance(entry, dict):
            return
        if "file" in entry:
            files.append((entry["file"], parent))
        for key in ("parts", "chapters"):
            walk(entry.get(key, []), None)
        walk(entry.get("sections", []), entry.get("file", parent))

    walk(toc, None)
    return files


def _read(root: Path, docname: str) -> tuple[str, str]:
    """The title and the code of a page: its first top-level heading, and its paired script if it
    has one, else the code of its notebook."""
    title, code = docname, ""
    notebook, script, markdown = (
        root / f"{docname}{ext}" for ext in (".ipynb", ".py", ".md")
    )
    if notebook.exists():
        cells = json.loads(notebook.read_text())["cells"]
        text = "\n".join(
            "".join(c["source"]) for c in cells if c["cell_type"] == "markdown"
        )
        code = "\n".join(
            "".join(c["source"]) for c in cells if c["cell_type"] == "code"
        )
    elif markdown.exists():
        text = markdown.read_text()
    else:
        text = ""
    if script.exists():
        code = script.read_text()
    heading = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    if heading:
        title = heading.group(1).strip()
    return title, code


def _picture(code: str) -> str | None:
    """The picture of a tutorial's card: its animation, else its last screenshot."""
    gifs = _GIF.findall(code)
    if gifs:
        return Path(gifs[0]).name
    pngs = _PNG.findall(code)
    return Path(pngs[-1]).name if pngs else None


class TutorialGallery(SphinxDirective):
    """A grid of cards, one for each tutorial in the table of contents that writes a picture.

    With the option ``:part:``, the caption of a part of the table of contents, only the tutorials
    of that part.
    """

    has_content = False
    option_spec: ClassVar[dict] = {"part": directives.unchanged_required}

    def run(self) -> list[nodes.Node]:
        root = Path(self.env.srcdir)
        toc_path = root / getattr(self.config, "external_toc_path", "_toc.yml")
        toc = yaml.safe_load(toc_path.read_text())
        part = self.options.get("part")
        if part is not None:
            parts = [p for p in toc.get("parts", []) if p.get("caption") == part]
            if not parts:
                logger.warning(
                    f"The gallery's part {part!r} is not in {toc_path.name}",
                    location=self.get_location(),
                )
                return []
            toc = parts[0]
        files = _toc_files(toc)
        cards = []
        for docname, parent in files:
            if docname == self.env.docname:
                continue
            title, code = _read(root, docname)
            picture = _picture(code)
            if picture is None:
                continue
            # The code of a chapter is often a section titled "Implementation": name it by its chapter
            if title == "Implementation" and parent is not None:
                title = _read(root, parent)[0]
            cards.append(
                f"````{{grid-item-card}} {title}\n:link: {docname}\n:link-type: doc\n\n"
                f'```{{raw}} html\n<img src="_static/gallery/{picture}" alt="{title}" '
                f'loading="lazy" style="width: 100%">\n```\n````\n'
            )
        if not cards:
            logger.warning(
                f"The gallery of {part or 'the book'} has no tutorial with a picture",
                location=self.get_location(),
            )
            return []
        text = "`````{grid} 1 2 2 3\n:gutter: 3\n\n" + "\n".join(cards) + "`````\n"
        container = nodes.container()
        self.state.nested_parse(
            StringList(text.splitlines()), self.content_offset, container
        )
        return container.children


def _copy(root: Path, target: Path) -> None:
    """Copy every GIF and PNG of the chapters to `target`."""
    target.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for chapter in sorted(root.glob(CHAPTERS)):
        for pattern in PATTERNS:
            for picture in sorted(chapter.glob(pattern)):
                if picture.name in copied:
                    logger.warning(
                        f"Gallery pictures {copied[picture.name]} and {picture} have the same "
                        "name; the last one is used"
                    )
                copied[picture.name] = picture
                shutil.copy2(picture, target / picture.name)


def _resolve(page: Path, target: Path) -> None:
    """Point each picture of the built front page named without an extension at the animation, if
    there is one, else at the picture, and warn about pictures no tutorial wrote."""
    html = page.read_text()

    def choose(match: re.Match) -> str:
        stem = match.group(1)
        for suffix in PREFERENCE:
            if (target / f"{stem}{suffix}").exists():
                return f"_static/gallery/{stem}{suffix}"
        logger.warning(
            f"The gallery shows {stem}, but no tutorial wrote {stem}.gif or {stem}.png"
        )
        return match.group(0)

    resolved = re.sub(r"_static/gallery/([\w-]+)\.\*", choose, html)
    if resolved != html:
        page.write_text(resolved)
    for name in sorted(
        set(re.findall(r"_static/gallery/([\w.-]+\.(?:gif|png))", resolved))
    ):
        if not (target / name).exists():
            logger.warning(
                f"The gallery shows {name}, which no tutorial wrote; did it run?"
            )


def build_gallery(app: Sphinx, exception: Exception | None) -> None:
    """Copy the pictures to ``_static/gallery`` of an HTML build, and resolve those of the front
    page."""
    if exception is not None or app.builder.format != "html":
        return
    target = Path(app.outdir) / "_static" / "gallery"
    _copy(Path(app.srcdir), target)
    page = Path(app.outdir) / f"{FRONT_PAGE}.html"
    if page.exists():
        _resolve(page, target)


def setup(app: Sphinx) -> dict:
    app.add_directive("tutorial-gallery", TutorialGallery)
    app.connect("build-finished", build_gallery)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
