#!/usr/bin/env python3
"""Pair up the comments in two revisions, so a prose change can be read on its own.

`git diff` answers "what bytes changed". Reviewing a comment pass needs a
different question -- "what does this function say now, and what did it say
before?" -- and the ordinary diff answers it badly: a docstring rewrite shows as
one deleted block and one added block, twenty lines apart, surrounded by code
context that did not change.

This extracts every comment and docstring from both revisions, anchors each to
the thing it documents, and pairs them. The result is one before/after row per
anchor, classified as:

    new         nothing documented it before
    rewritten   it said something else
    deleted     it said something and now says nothing
    unchanged   identical text (hidden by default)

Anchors are AST paths (``function:name``, ``class:Name``, ``<module>``), not
line numbers, so inserting a docstring does not make every anchor below it look
changed. Inline comments have no such anchor and are matched by exact text
within a file, which means a reworded inline comment reads as one deletion plus
one addition rather than as a rewrite -- unavoidable without guessing.

Usage:
    python -m tools.comment_diff <base> [head]           # summary to stdout
    python -m tools.comment_diff <base> --detail         # every pair, as text
    python -m tools.comment_diff <base> --html out.html  # side-by-side report
    python -m tools.comment_diff <base> --json out.json  # for other tooling

    # only one package, only rewrites -- the interesting case for a review
    python -m tools.comment_diff eca6923 --detail --path grace_mem --only rewritten
"""
from __future__ import annotations

import argparse
import ast
import html
import io
import json
import subprocess
import sys
import tokenize
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATUSES = ("new", "rewritten", "deleted", "unchanged")


@dataclass
class Pair:
    """One anchor's comment text before and after."""

    path: str
    anchor: str
    kind: str
    status: str
    before: str = ""
    after: str = ""


@dataclass
class FileReport:
    """Every pair for one file, plus its per-status counts."""

    path: str
    pairs: list[Pair] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for p in self.pairs:
            out[p.status] += 1
        return out


def git_show(rev: str, path: str) -> str | None:
    """Read a file at a revision, or None when it did not exist there."""
    result = subprocess.run(
        ["git", "show", f"{rev}:{path}"], capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def changed_files(base: str, head: str, path_filter: str | None) -> list[str]:
    """List the .py files that differ between the two revisions."""
    cmd = ["git", "diff", "--name-only", base, head, "--", "*.py"]
    files = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.split()
    if path_filter:
        files = [f for f in files if f.startswith(path_filter)]
    return files


def extract(source: str) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Return ({anchor: (kind, docstring)}, [inline comments]) for one file.

    Docstrings are keyed by AST path so they survive line shifts. A name defined
    twice in one file (a method and a module-level function, say) collides; the
    kind prefix separates the common case, and the rest are rare enough to read
    as a single merged anchor rather than worth a scope-qualified key.
    """
    docs: dict[str, tuple[str, str]] = {}
    inline: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return docs, inline

    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    module_doc = ast.get_docstring(tree)
    if module_doc:
        docs["<module>"] = ("module", module_doc)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            kind = "class"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "method" if isinstance(parents.get(node), ast.ClassDef) else "function"
        else:
            continue
        doc = ast.get_docstring(node)
        if doc:
            docs[f"{kind}:{node.name}"] = (kind, doc)

    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                inline.append(token.string.strip())
    except (tokenize.TokenError, IndentationError):
        pass  # a file that will not tokenize still yields its docstrings

    return docs, inline


def pair_file(path: str, before_src: str, after_src: str) -> FileReport:
    """Build the before/after pairs for one file."""
    report = FileReport(path=path)
    before_docs, before_inline = extract(before_src)
    after_docs, after_inline = extract(after_src)

    for anchor in sorted(set(before_docs) | set(after_docs)):
        old = before_docs.get(anchor)
        new = after_docs.get(anchor)
        if old and new:
            status = "unchanged" if old[1].strip() == new[1].strip() else "rewritten"
        elif new:
            status = "new"
        else:
            status = "deleted"
        report.pairs.append(
            Pair(
                path=path,
                anchor=anchor,
                kind=(new or old)[0],
                status=status,
                before=old[1] if old else "",
                after=new[1] if new else "",
            )
        )

    # Inline comments carry no stable identity, so they are matched by exact
    # text. Counting occurrences rather than using a set keeps a comment that
    # legitimately appears twice from collapsing into one.
    before_counts: dict[str, int] = {}
    for comment in before_inline:
        before_counts[comment] = before_counts.get(comment, 0) + 1
    after_counts: dict[str, int] = {}
    for comment in after_inline:
        after_counts[comment] = after_counts.get(comment, 0) + 1

    for comment in sorted(set(before_counts) | set(after_counts)):
        delta = after_counts.get(comment, 0) - before_counts.get(comment, 0)
        for _ in range(abs(delta)):
            report.pairs.append(
                Pair(
                    path=path,
                    anchor="<inline>",
                    kind="inline",
                    status="new" if delta > 0 else "deleted",
                    before="" if delta > 0 else comment,
                    after=comment if delta > 0 else "",
                )
            )
    return report


def build(base: str, head: str, path_filter: str | None) -> list[FileReport]:
    """Pair every changed file between the two revisions."""
    reports = []
    for path in changed_files(base, head, path_filter):
        before = git_show(base, path)
        after = git_show(head, path)
        if before is None and after is None:
            continue
        report = pair_file(path, before or "", after or "")
        if report.pairs:
            reports.append(report)
    return reports


def print_summary(reports: list[FileReport]) -> None:
    """Print per-file counts and a total."""
    totals = {s: 0 for s in STATUSES}
    print(f"{'file':62} {'new':>5} {'rewr':>5} {'del':>5} {'same':>5}")
    print("-" * 86)
    for report in reports:
        counts = report.counts()
        for status in STATUSES:
            totals[status] += counts[status]
        if counts["new"] or counts["rewritten"] or counts["deleted"]:
            print(
                f"{report.path:62} {counts['new']:5} {counts['rewritten']:5} "
                f"{counts['deleted']:5} {counts['unchanged']:5}"
            )
    print("-" * 86)
    print(
        f"{'TOTAL':62} {totals['new']:5} {totals['rewritten']:5} "
        f"{totals['deleted']:5} {totals['unchanged']:5}"
    )


def print_detail(reports: list[FileReport], only: set[str]) -> None:
    """Print each pair as a before/after block."""
    for report in reports:
        shown = [p for p in report.pairs if p.status in only]
        if not shown:
            continue
        print(f"\n{'=' * 86}\n{report.path}\n{'=' * 86}")
        for pair in shown:
            print(f"\n[{pair.status}] {pair.kind} {pair.anchor}")
            if pair.before:
                for line in pair.before.splitlines():
                    print(f"  - {line}")
            if pair.before and pair.after:
                print("  " + "-" * 40)
            if pair.after:
                for line in pair.after.splitlines():
                    print(f"  + {line}")


def render_html(reports: list[FileReport], base: str, head: str) -> str:
    """Render a self-contained side-by-side report."""
    totals = {s: 0 for s in STATUSES}
    for report in reports:
        for status, count in report.counts().items():
            totals[status] += count

    rows = []
    for report in reports:
        shown = [p for p in report.pairs if p.status != "unchanged"]
        if not shown:
            continue
        counts = report.counts()
        rows.append(
            f'<section class="file" data-file="{html.escape(report.path)}">'
            f'<h2>{html.escape(report.path)}'
            f'<span class="badges">'
            f'<em class="new">{counts["new"]} new</em>'
            f'<em class="rewritten">{counts["rewritten"]} rewritten</em>'
            f'<em class="deleted">{counts["deleted"]} deleted</em>'
            f"</span></h2>"
        )
        for pair in shown:
            before = html.escape(pair.before) or '<span class="none">(nothing)</span>'
            after = html.escape(pair.after) or '<span class="none">(nothing)</span>'
            rows.append(
                f'<div class="pair" data-status="{pair.status}" data-kind="{pair.kind}">'
                f'<div class="anchor"><span class="tag {pair.status}">{pair.status}</span>'
                f'<span class="kind">{html.escape(pair.kind)}</span>'
                f"<code>{html.escape(pair.anchor)}</code></div>"
                f'<div class="cols">'
                f'<pre class="before">{before}</pre>'
                f'<pre class="after">{after}</pre>'
                f"</div></div>"
            )
        rows.append("</section>")

    return TEMPLATE.format(
        base=html.escape(base),
        head=html.escape(head),
        new=totals["new"],
        rewritten=totals["rewritten"],
        deleted=totals["deleted"],
        unchanged=totals["unchanged"],
        body="\n".join(rows),
    )


# Palette is the copyeditor's: insertion teal, revision indigo, deletion
# oxblood, on a warm neutral ground biased slightly toward the indigo so the
# greys read as chosen rather than defaulted. Status is encoded as a rail on the
# card edge as well as a label, so a page of 900 pairs can be scanned without
# reading any of them.
TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comment diff {base} to {head}</title>
<style>
  :root {{
    --ground:#fbfaf8; --raised:#f3f2ef; --ink:#1b1d23; --muted:#666d7a;
    --line:#e2e0dc; --new:#0d7a6f; --rew:#4a45c9; --del:#a8213f;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --ground:#131519; --raised:#1a1d23; --ink:#e6e5e2; --muted:#8a919e;
      --line:#282c34; --new:#3fbfab; --rew:#8b87f0; --del:#e8657f;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground:#131519; --raised:#1a1d23; --ink:#e6e5e2; --muted:#8a919e;
    --line:#282c34; --new:#3fbfab; --rew:#8b87f0; --del:#e8657f;
  }}
  :root[data-theme="light"] {{
    --ground:#fbfaf8; --raised:#f3f2ef; --ink:#1b1d23; --muted:#666d7a;
    --line:#e2e0dc; --new:#0d7a6f; --rew:#4a45c9; --del:#a8213f;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; padding:0 clamp(1rem,4vw,2.5rem) 5rem;
    background:var(--ground); color:var(--ink);
    font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;
    -webkit-font-smoothing:antialiased;
  }}
  code, pre {{ font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace; }}

  header {{
    position:sticky; top:0; z-index:20; background:var(--ground);
    border-bottom:1px solid var(--line); padding:1.1rem 0 .9rem;
    display:flex; flex-direction:column; gap:.75rem;
  }}
  .eyebrow {{
    font-size:11px; letter-spacing:.09em; text-transform:uppercase;
    color:var(--muted); font-weight:600;
  }}
  h1 {{ font-size:17px; font-weight:650; margin:.15rem 0 0; letter-spacing:-.01em; }}
  h1 code {{ font-size:15px; color:var(--muted); font-weight:500; }}

  .totals {{ display:flex; gap:1.6rem; flex-wrap:wrap; }}
  .stat {{ display:flex; flex-direction:column; gap:.1rem; }}
  .stat b {{ font-size:20px; font-weight:650; font-variant-numeric:tabular-nums; line-height:1; }}
  .stat span {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }}
  .stat.n b {{ color:var(--new); }}
  .stat.r b {{ color:var(--rew); }}
  .stat.d b {{ color:var(--del); }}
  .stat.u b {{ color:var(--muted); }}

  .filters {{ display:flex; gap:.4rem; flex-wrap:wrap; align-items:center; }}
  button {{
    font:inherit; font-size:13px; padding:.28rem .8rem; border:1px solid var(--line);
    background:var(--raised); color:var(--ink); border-radius:2px; cursor:pointer;
  }}
  button:hover {{ border-color:var(--muted); }}
  button:focus-visible {{ outline:2px solid var(--rew); outline-offset:2px; }}
  button[aria-pressed="true"] {{ background:var(--ink); color:var(--ground); border-color:var(--ink); }}
  input {{
    font:inherit; font-size:13px; padding:.28rem .7rem; border:1px solid var(--line);
    border-radius:2px; background:var(--raised); color:var(--ink); flex:1; min-width:14rem;
  }}
  input:focus-visible {{ outline:2px solid var(--rew); outline-offset:1px; }}

  h2 {{
    font-size:13px; font-weight:600; margin:2.2rem 0 .6rem;
    display:flex; gap:.8rem; align-items:baseline; flex-wrap:wrap;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  }}
  .badges {{ display:flex; gap:.7rem; font-family:system-ui,sans-serif; }}
  .badges em {{ font-style:normal; font-size:11px; font-variant-numeric:tabular-nums; }}
  .badges .new {{ color:var(--new); }}
  .badges .rewritten {{ color:var(--rew); }}
  .badges .deleted {{ color:var(--del); }}

  .pair {{
    border:1px solid var(--line); border-left-width:3px; border-radius:2px;
    margin-bottom:.6rem; overflow:hidden; background:var(--ground);
  }}
  .pair[data-status="new"] {{ border-left-color:var(--new); }}
  .pair[data-status="rewritten"] {{ border-left-color:var(--rew); }}
  .pair[data-status="deleted"] {{ border-left-color:var(--del); }}

  .anchor {{
    padding:.4rem .75rem; background:var(--raised); border-bottom:1px solid var(--line);
    display:flex; gap:.65rem; align-items:center; flex-wrap:wrap;
  }}
  .tag {{
    font-size:10px; text-transform:uppercase; letter-spacing:.07em; font-weight:700;
  }}
  .tag.new {{ color:var(--new); }}
  .tag.rewritten {{ color:var(--rew); }}
  .tag.deleted {{ color:var(--del); }}
  .kind {{ font-size:11px; color:var(--muted); }}
  .anchor code {{ font-size:12px; }}

  .cols {{ display:grid; grid-template-columns:1fr 1fr; }}
  .cols pre {{
    margin:0; padding:.75rem .85rem; white-space:pre-wrap; overflow-wrap:anywhere;
    font-size:12.5px; line-height:1.55; overflow-x:auto;
  }}
  .before {{ border-right:1px solid var(--line); color:var(--muted); }}
  .none {{ opacity:.4; font-style:italic; }}
  @media (max-width:860px) {{
    .cols {{ grid-template-columns:1fr; }}
    .before {{ border-right:0; border-bottom:1px solid var(--line); }}
  }}
  .hide {{ display:none; }}
  .empty {{ color:var(--muted); font-size:13px; padding:2rem 0; }}
</style>
<header>
  <div>
    <div class="eyebrow">Comment prose &middot; before and after</div>
    <h1>{base} <code>&rarr;</code> {head}</h1>
  </div>
  <div class="totals">
    <div class="stat n"><b>{new}</b><span>new</span></div>
    <div class="stat r"><b>{rewritten}</b><span>rewritten</span></div>
    <div class="stat d"><b>{deleted}</b><span>deleted</span></div>
    <div class="stat u"><b>{unchanged}</b><span>unchanged &middot; hidden</span></div>
  </div>
  <div class="filters">
    <button data-f="all" aria-pressed="true">All</button>
    <button data-f="new" aria-pressed="false">New</button>
    <button data-f="rewritten" aria-pressed="false">Rewritten</button>
    <button data-f="deleted" aria-pressed="false">Deleted</button>
    <input id="q" type="search" placeholder="Filter by path or anchor">
  </div>
</header>
<main>
{body}
<p class="empty hide" id="empty">Nothing matches that filter.</p>
</main>
<script>
  const pairs = [...document.querySelectorAll(".pair")];
  const files = [...document.querySelectorAll(".file")];
  const buttons = [...document.querySelectorAll("[data-f]")];
  const q = document.getElementById("q");
  const empty = document.getElementById("empty");
  let status = "all";

  function apply() {{
    const term = q.value.trim().toLowerCase();
    let visible = 0;
    for (const pair of pairs) {{
      const matchesStatus = status === "all" || pair.dataset.status === status;
      const haystack = (
        pair.closest(".file").dataset.file + " " +
        pair.querySelector("code").textContent
      ).toLowerCase();
      const show = matchesStatus && (!term || haystack.includes(term));
      pair.classList.toggle("hide", !show);
      if (show) visible++;
    }}
    for (const file of files) {{
      file.classList.toggle("hide", !file.querySelector(".pair:not(.hide)"));
    }}
    empty.classList.toggle("hide", visible > 0);
  }}

  buttons.forEach(button => button.addEventListener("click", () => {{
    status = button.dataset.f;
    buttons.forEach(other => other.setAttribute("aria-pressed", String(other === button)));
    apply();
  }}));
  q.addEventListener("input", apply);
</script>
"""


def main(argv: list[str] | None = None) -> int:
    """Pair the comments between two revisions and report them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", help="revision to compare against")
    parser.add_argument("head", nargs="?", default="HEAD")
    parser.add_argument("--path", help="limit to paths under this prefix")
    parser.add_argument("--detail", action="store_true", help="print every pair")
    parser.add_argument(
        "--only", action="append", choices=STATUSES,
        help="restrict --detail to these statuses (repeatable)",
    )
    parser.add_argument("--html", metavar="FILE", help="write a side-by-side report")
    parser.add_argument("--json", metavar="FILE", help="write the pairs as JSON")
    args = parser.parse_args(argv)

    reports = build(args.base, args.head, args.path)
    if not reports:
        print("no comment changes found", file=sys.stderr)
        return 1

    if args.html:
        Path(args.html).write_text(
            render_html(reports, args.base, args.head), encoding="utf-8"
        )
        print(f"wrote {args.html}")
    if args.json:
        payload = [
            {"path": r.path, "counts": r.counts(), "pairs": [asdict(p) for p in r.pairs]}
            for r in reports
        ]
        Path(args.json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.json}")

    if args.detail:
        print_detail(reports, set(args.only or ["new", "rewritten", "deleted"]))
    elif not (args.html or args.json):
        print_summary(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
