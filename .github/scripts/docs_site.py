# Copyright (C) 2026 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT
"""Put the web page together from the HTML of each version of the tutorial.

The web page has a folder per version: ``main`` (built against DOLFINx nightly), ``stable`` (the
newest release that was built) and one per older minor release (``v0.10``, filled by its newest
tag). The newest release's own folder (``v0.11``) only has redirects to ``stable``, so it isn't
published twice. The root of the web page, and every URL from before the web page had versions,
leads to ``stable``. The versions are listed in ``versions.json``, which the version switcher
reads.

Each version is built once, and its HTML is kept as a tarball in the ``docs-archive`` release, so
that a push to main only has to build main. Every build is tested before it is published. A tag
that failed to build or test is marked as such in the archive, and isn't built again until the
mark is deleted.

Subcommands, in the order the deploy workflow uses them:

- ``archived``: list what is in the archive.
- ``plan``: what to build and test, from the tags and what is in the archive.
- ``gate``: leave out the builds that failed or weren't tested, and mark the failures.
- ``assemble``: unpack each version, write ``versions.json``, the redirects and ``404.html``.
- ``inject``: add the version switcher to every page.

and ``latest``: the newest release, which the scheduled build of the stable book builds.

The archive keeps the tarballs the web page was last put together from, and nothing else.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / ".github" / "docs-versions.toml"
SWITCHER = REPO / ".github" / "docs" / "version-switcher.js"

ARCHIVE = "docs-archive"
STABLE = "stable"
# Release tags, e.g. v0.11.1 or v0.7.0.post0, and the archived tarball of a build
TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?$")
ASSET = re.compile(r"^docs-(main|v[^-]+)-[0-9a-f]{8}\.tar\.gz$")

# Where the version switcher is put in the web page, and how a page refers to it
SWITCHER_PATH = "_versions/version-switcher.js"
MARKER = b"<!-- tutorial-version-switcher -->"
# Hides the second search bar that sphinx-book-theme >= 1.3 shows in builds from before
# `navbar_persistent: []` was set in _config.yml (v0.11.1 and older), as a style so it doesn't flash
FIXES = b"<style>#pst-header .navbar-persistent--container{display:none}</style>"
HEAD = re.compile(rb"<head(?:\s[^>]*)?>", re.IGNORECASE)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def tags() -> dict[str, str]:
    """The tags of the repository on GitHub, and the commit each points to."""
    commits: dict[str, str] = {}
    for line in git("ls-remote", "--tags", "origin").splitlines():
        sha, ref = line.split()
        name = ref.removeprefix("refs/tags/")
        if name.endswith("^{}"):
            # The commit an annotated tag points to
            commits[name[:-3]] = sha
        else:
            commits.setdefault(name, sha)
    return commits


def version(tag: str) -> tuple[int, int, int, int] | None:
    """``(major, minor, patch, post)`` of a release tag, with ``post`` -1 if there is none."""
    match = TAG.match(tag)
    if match is None:
        return None
    major, minor, patch = (int(g) for g in match.groups()[:3])
    return (major, minor, patch, -1 if match.group(4) is None else int(match.group(4)))


def asset(name: str, sha: str) -> str:
    return f"docs-{name}-{sha[:8]}.tar.gz"


def failed(tarball: str) -> str:
    """The mark in the archive of a build that failed."""
    return tarball.removesuffix(".tar.gz") + ".failed"


def tested(tarball: str) -> str:
    """The mark of a build whose tests passed, which the test workflow uploads."""
    return tarball.removesuffix(".tar.gz") + ".tested"


def read_archived(path: Path) -> dict[str, str]:
    """What is in the archive, and when each was uploaded."""
    return dict(line.split("\t") for line in path.read_text().splitlines() if line)


def releases(config: dict, commits: dict[str, str]) -> list[dict]:
    """One slot per minor release from ``oldest`` onwards, newest first, with its newest tag."""
    oldest = tuple(int(i) for i in config["oldest"].split("."))
    newest: dict[tuple[int, int], tuple[tuple[int, ...], str]] = {}
    for name in commits:
        v = version(name)
        if v is not None and v[:2] >= oldest and v > newest.get(v[:2], ((), ""))[0]:
            newest[v[:2]] = (v, name)

    slots = []
    for (major, minor), (_, name) in sorted(newest.items(), reverse=True):
        image = config.get("images", {}).get(f"{major}.{minor}", config["image"])
        sha = commits[name]
        slots.append(
            {
                "label": f"v{major}.{minor}",
                "name": name,
                "ref": sha,
                "image": image.format(major=major, minor=minor),
                "asset": asset(name, sha),
            }
        )
    return slots


def write_outputs(**outputs: str):
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as f:
            f.writelines(f"{key}={value}\n" for key, value in outputs.items())


def list_archive(args):
    # The REST API, unlike `gh release view`, tells a missing release from any other error
    result = subprocess.run(
        ["gh", "api", f"repos/{{owner}}/{{repo}}/releases/tags/{ARCHIVE}"]
        + ["--jq", ".assets[] | [.name, .updated_at] | @tsv"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        args.out.write_text(result.stdout)
    elif "(HTTP 404)" in result.stderr:
        print(f"There is no {ARCHIVE} release yet")
        args.out.write_text("")
    else:
        raise SystemExit(f"Couldn't list the {ARCHIVE} release: {result.stderr}")


def plan(args):
    config = tomllib.loads(CONFIG.read_text())
    archived = read_archived(args.archived)
    head = git("rev-parse", "HEAD").strip()
    main = {
        "label": "main",
        "name": "main (nightly)",
        "ref": head,
        "image": config["nightly_image"],
        "asset": asset("main", head),
    }
    slots = releases(config, tags())

    # A new tag doesn't change main, so it only needs building if this commit of it isn't built
    build = [] if args.tag and main["asset"] in archived else [main]
    for slot in slots:
        if failed(slot["asset"]) in archived:
            print(
                f"::notice::{slot['name']} failed before. Delete "
                f"{failed(slot['asset'])} from the {ARCHIVE} release to try again"
            )
        elif slot["asset"] not in archived:
            build.append(slot)
    for job in build:
        print(f"Building {job['label']} ({job['name']}) in {job['image']}")
    args.out.write_text(
        json.dumps(
            {"main": main, "releases": slots, "build": [j["asset"] for j in build]},
            indent=2,
        )
    )

    # Every build is tested before it is published
    tests = [j for j in build if j is not main]
    write_outputs(
        matrix=json.dumps({"include": build}),
        build="true" if build else "false",
        main_ref=head if main in build else "",
        tests=json.dumps({"include": tests}),
        test="true" if tests else "false",
    )


def latest(args):
    config = tomllib.loads(CONFIG.read_text())
    slots = releases(config, tags())
    if not slots:
        raise SystemExit(f"There is no release tag from v{config['oldest']} on")
    newest = slots[0]
    print(f"Newest release: {newest['name']} ({newest['ref']}) in {newest['image']}")
    write_outputs(**newest)


def gate(args):
    """Leave out of ``fresh`` the builds that failed or weren't tested.

    A release whose build or tests failed has a ``.failed`` mark, uploaded by its job, which is
    archived so the release isn't built again. A build that is just missing, or wasn't tested
    (its job was cancelled, or its runner died), is built again on the next run.
    """
    site_plan = json.loads(args.plan.read_text())
    args.fresh.mkdir(parents=True, exist_ok=True)
    main = site_plan["main"]
    # main is never marked, as it is built again on every push
    (args.fresh / failed(main["asset"])).unlink(missing_ok=True)
    if args.main_tests != "success" and (args.fresh / main["asset"]).exists():
        print("::warning::The tests of main didn't pass, keeping its archived build")
        (args.fresh / main["asset"]).unlink()

    for slot in site_plan["releases"]:
        if slot["asset"] not in site_plan["build"]:
            continue
        tarball = args.fresh / slot["asset"]
        if (args.fresh / failed(slot["asset"])).exists():
            print(
                f"::warning::{slot['name']} failed to build or test, and won't be built again "
                f"until {failed(slot['asset'])} is deleted from the {ARCHIVE} release"
            )
            tarball.unlink(missing_ok=True)
        elif not tarball.exists():
            print(
                f"::warning::{slot['name']} wasn't built, it will be built on the next run"
            )
        elif not (args.fresh / tested(slot["asset"])).exists():
            print(
                f"::warning::{slot['name']} wasn't tested, it will be built on the next run"
            )
            tarball.unlink()
    for mark in args.fresh.glob("*.tested"):
        mark.unlink()


def older(slot: dict, archived: dict[str, str]) -> list[str]:
    """Archived tarballs of earlier commits of the slot, newest first: by tag, and by upload time
    for main, whose tarballs carry nothing else to order them by."""
    found = {}
    for tarball, uploaded in archived.items():
        match = ASSET.match(tarball)
        if match is None or tarball == slot["asset"]:
            continue
        if match.group(1) == slot["label"] == "main":
            found[tarball] = ((), uploaded)
        elif (v := version(match.group(1))) and f"v{v[0]}.{v[1]}" == slot["label"]:
            found[tarball] = (v, uploaded)
    return sorted(found, key=found.get, reverse=True)


def fetch(tarball: str, fresh: Path, archived: dict[str, str]) -> Path | None:
    """The tarball, built in this run or else from the archive, or None if it is in neither.

    A tarball in the archive that can't be downloaded stops the deploy, rather than being
    left out of the web page and then removed from the archive.
    """
    path = fresh / tarball
    if path.exists():
        return path
    if tarball not in archived:
        return None
    download = ["gh", "release", "download", ARCHIVE, "--pattern", tarball]
    download += ["--dir", str(fresh), "--clobber"]
    for attempt in range(3):
        if attempt:
            time.sleep(30)
        if subprocess.run(download, check=False).returncode == 0:
            return path
    raise SystemExit(f"Couldn't download {tarball} from the {ARCHIVE} release")


def redirect(page: Path, target: str):
    """A page leading to ``target`` (relative to the page), keeping the query and anchor."""
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        "<!DOCTYPE html>\n"
        '<html><head><meta charset="utf-8"><title>Redirecting</title>\n'
        f'<link rel="canonical" href="{target}">\n'
        f'<script>location.replace("{target}" + location.search + location.hash)</script>\n'
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        f'</head><body><a href="{target}">Redirecting to {target}</a></body></html>\n'
    )


def redirects(site: Path, folder: str, landing: str, skip: set[str]):
    """Redirect each page of ``landing`` from the same page in ``folder``."""
    depth = len(Path(folder).parts)
    for page in (site / landing).rglob("*.html"):
        rel = page.relative_to(site / landing)
        if rel.parts[0].startswith("_") or rel.parts[0] in skip:
            continue
        target = "../" * (depth + len(rel.parts) - 1) + f"{landing}/{rel.as_posix()}"
        redirect(site / folder / rel, target)


def not_found(site: Path, base: str, folders: list[str], moved: dict[str, str]):
    """``404.html``, sending any other file from before the web page had versions, such as a
    notebook, picture or script, and any file of a folder in ``moved``, to its new place."""
    landing = moved[""]
    site.joinpath("404.html").write_text(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Page not found</title>
<script>
(() => {{
  const base = {json.dumps(base + "/")};
  const folders = {json.dumps(folders)};
  const moved = {json.dumps(moved)};
  if (!location.pathname.startsWith(base)) return;
  const rest = location.pathname.slice(base.length);
  const [first, ...tail] = rest.split("/");
  const to = first !== "" && Object.hasOwn(moved, first) ? moved[first] + "/" + tail.join("/")
    : folders.includes(first) ? null : moved[""] + "/" + rest;
  if (to !== null) location.replace(base + to + location.search + location.hash);
}})();
</script></head>
<body><h1>Page not found</h1>
<p>See <a href="{base}/{landing}/">the FEniCSx tutorial</a>.</p></body></html>
""")


def assemble(args):
    site_plan = json.loads(args.plan.read_text())
    archived = read_archived(args.archived)
    args.site.mkdir(parents=True, exist_ok=True)
    args.fresh.mkdir(parents=True, exist_ok=True)

    # A version that failed to build or test keeps the build of an earlier commit of it, so a
    # broken new tag leaves stable where it was
    built, used = [], []
    for slot in [site_plan["main"], *site_plan["releases"]]:
        if failed(slot["asset"]) in archived:
            used.append(failed(slot["asset"]))
        for tarball in [slot["asset"], *older(slot, archived)]:
            if fetch(tarball, args.fresh, archived) is not None:
                break
        else:
            if slot is site_plan["main"]:
                raise SystemExit("main is neither built nor archived")
            print(f"::warning::No build of {slot['name']}, leaving {slot['label']} out")
            continue
        if tarball != slot["asset"]:
            print(
                f"::warning::No build of {slot['name']} at {slot['ref']}, using {tarball}"
            )
            if slot is not site_plan["main"]:
                slot = slot | {"name": ASSET.match(tarball).group(1)}
        built.append(slot | {"asset": tarball})
        used.append(tarball)
    if args.used:
        args.used.write_text("\n".join(used) + "\n")

    main, released = built[0], built[1:]
    entries = [{"name": main["name"], "version": main["label"]}]
    folders = {main["label"]: main}
    for v in released:
        entries.append({"name": v["name"], "version": v["label"]})
        folders[v["label"]] = v
    landing = main["label"]
    moved = {}
    if released:
        newest = released[0]
        landing = STABLE
        del folders[newest["label"]]
        folders[STABLE] = newest
        entries[1]["folder"] = STABLE
        stable = {"name": f"{STABLE} ({newest['name']})", "version": STABLE}
        entries.insert(0, stable | {"release": newest["label"]})
        moved[newest["label"]] = STABLE
    moved[""] = landing
    (args.site / "versions.json").write_text(json.dumps(entries, indent=2))

    for folder, v in folders.items():
        with tarfile.open(args.fresh / v["asset"]) as t:
            t.extractall(args.site / folder, filter="data")
    names = {e["version"] for e in entries}
    # Pages from before the web page had versions: <page>.html -> stable/<page>.html
    redirects(args.site, "", landing, skip=names)
    for folder in moved.keys() - {""}:
        redirects(args.site, folder, landing, skip=set())
    if args.base_path is not None:
        not_found(args.site, args.base_path.rstrip("/"), sorted(names), moved)

    for e in entries:
        print(f"{e['version']}: {e['name']}")


def inject(args):
    target = args.site / SWITCHER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SWITCHER, target)

    entries = json.loads((args.site / "versions.json").read_text())
    count = 0
    for folder in {e.get("folder", e["version"]) for e in entries}:
        for page in (args.site / folder).rglob("*.html"):
            html = page.read_bytes()
            head = HEAD.search(html)
            if MARKER in html or head is None:
                continue
            depth = len(page.relative_to(args.site).parts) - 1
            src = "../" * depth + SWITCHER_PATH
            tag = MARKER + FIXES + f'<script defer src="{src}"></script>'.encode()
            page.write_bytes(html[: head.end()] + tag + html[head.end() :])
            count += 1
    print(f"Added the version switcher to {count} pages")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(required=True)
    archived_list = {"type": Path, "default": Path("archived.txt")}
    archived = archived_list | {"help": "What is archived, as written by `archived`"}
    site_plan = {"type": Path, "default": Path("plan.json"), "help": "The plan"}

    p = commands.add_parser("archived", help="List what is in the archive")
    p.add_argument("--out", **archived_list)
    p.set_defaults(func=list_archive)

    p = commands.add_parser("plan", help="What to build, as a job matrix")
    p.add_argument("--archived", **archived)
    p.add_argument("--out", **site_plan)
    p.add_argument(
        "--tag", help="The tag the run is for: main isn't built again if it is archived"
    )
    p.set_defaults(func=plan)

    p = commands.add_parser("latest", help="The newest release")
    p.set_defaults(func=latest)

    p = commands.add_parser("gate", help="Leave out the builds that failed")
    p.add_argument("--plan", **site_plan)
    p.add_argument("--main-tests", default="", help="Result of the tests of main")
    p.add_argument("fresh", type=Path, help="Builds and marks from this run")
    p.set_defaults(func=gate)

    p = commands.add_parser("assemble", help="Put the web page together")
    p.add_argument("--plan", **site_plan)
    p.add_argument("--archived", **archived)
    p.add_argument("--used", type=Path, help="Where to write what the web page used")
    p.add_argument("--base-path", help="Path of the web page on its host, for 404.html")
    p.add_argument("fresh", type=Path, help="Tarballs built in this run")
    p.add_argument("site", type=Path, help="Where to put the web page")
    p.set_defaults(func=assemble)

    p = commands.add_parser("inject", help="Add the version switcher to every page")
    p.add_argument("site", type=Path)
    p.set_defaults(func=inject)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
