"""
prereg -- the pre-registration discipline, enforced instead of remembered.
=========================================================================

THE RULE THIS ENCODES
---------------------
From CLAUDE.md, under Research discipline:

    "Pre-register the success criterion before running the test, then report the result
     against that bar. No moving the goalposts after seeing the output."

That rule has been followed by hand, and followed well -- the crypto-slot bar was git
committed as 987f6b2 BEFORE the run, and the trial-count clause that ultimately declined it
was in the same commit. This module is that practice made mechanical, so it survives a
session that is tired, or a future Claude that has not read the doctrine.

WHAT IT ACTUALLY ENFORCES
-------------------------
* `grade()` refuses to score a result that has no registration, AND refuses one whose
  registration is not pushed. There is no path to a verdict that does not start with a
  written bar that has left this machine. The escape hatch, `unfrozen_ok=True`, does not
  buy a clean report: it stamps the output UNREGISTERED RESULT and marks the verdict line,
  so a report produced that way cannot be mistaken for -- or pasted as -- a registered one.
  An escape hatch whose output is indistinguishable from the real thing is not an escape
  hatch, it is the hole with a longer name.
* The registration records the git HEAD and whether the tree was CLEAN at the time, and
  `grade()` reports how frozen the bar actually is. FOUR states, not two: uncommitted (a
  file you can still edit after seeing the output is a note, not a pre-registration),
  committed-but-LOCAL (one `git commit --amend` rewrites it, and it dies with the disk),
  no-upstream (unanswerable, treated as local), and pushed (the registered bytes exist off
  this machine). `freeze()` performs the commit-and-push and verifies it, because the
  printed instruction it replaces was a step that could silently not happen -- which is
  exactly how the monthly snapshot task lagged its remote by a month at a time.
* `n_trials` is required, and is fed to the deflated Sharpe. The trial count is the input
  people quietly understate; asking for it up front, before there is a winner to protect,
  is when the honest number is cheapest to give.
* Criteria are stored as explicit comparisons and evaluated mechanically. Nobody gets to
  re-read a bar generously at 1am.

USE
---
    import prereg
    # BEFORE the run
    prereg.register(
        "crypto_slot_v2",
        hypothesis="A BTC slot sized inverse-vol adds risk-adjusted return to the trend sleeve",
        criteria=[("sharpe_edge", ">=", 0.10), ("maxdd_gap", "<=", 0.03),
                  ("h1_edge", ">", 0.0), ("h2_edge", ">", 0.0)],
        n_trials=16,
        notes="Declining on a lone marginal pass is pre-agreed; see Failed Improvements 5.")

    prereg.freeze("crypto_slot_v2")   # commits AND pushes the bar, then verifies it

    # AFTER the run
    print(prereg.grade("crypto_slot_v2", dict(sharpe_edge=0.20, maxdd_gap=0.00,
                                              h1_edge=0.37, h2_edge=0.01))["report"])
"""

import datetime as dt
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG_DIR = os.path.join(HERE, "reports", "prereg")

class NotFrozen(RuntimeError):
    """Raised by grade() when the bar is not pushed.

    A distinct type so a caller that genuinely means to score an unfrozen bar has to say
    so in code that greps -- `except prereg.NotFrozen` -- rather than swallowing it in a
    bare except and moving on.
    """


OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=10).stdout.strip()
    except Exception:
        return ""


def _path(name):
    return os.path.join(PREREG_DIR, f"{name}.json")


def _git_run(*args, timeout=120):
    """Like _git, but keeps the exit code and stderr.

    _git() swallows every failure into an empty string, which is right for the read-only
    queries it was written for and wrong for anything that changes state: a push that
    fails must not look like a push that returned nothing to say.
    """
    try:
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except Exception as e:  # git missing, timeout, OSError
        return subprocess.CompletedProcess(list(args), 1, "", f"{type(e).__name__}: {e}")


def _rel(name):
    return os.path.relpath(_path(name), HERE).replace(os.sep, "/")


def _blob_at(ref, rel):
    """Blob hash of `rel` as it exists in `ref`, or "" if it is not there.

    ls-tree rather than `rev-parse <ref>:<path>`: the colon form gets mangled by MSYS path
    conversion on this box (`origin/master:reports/...` came back as `origin\\master;...`),
    and a silently wrong answer here would report an unpushed bar as frozen -- failing in
    the one direction this module exists to prevent.
    """
    parts = _git("ls-tree", ref, "--", rel).split()
    return parts[2] if len(parts) >= 3 and parts[1] == "blob" else ""


def freeze_state(name):
    """How frozen is this bar? Returns (state, detail). Four states, not two:

    "absent"      -- no registration file at all.
    "uncommitted" -- editable in the working tree. A note, not a bar.
    "local"       -- committed on this machine only. Still rewritable by commit --amend,
                     reset or rebase, and gone with the disk. Not yet evidence.
    "unknown"     -- committed, but the branch has no upstream, so whether the bar left
                     this machine is unanswerable rather than answered no. Treat as local.
    "pushed"      -- the exact registered bytes exist on the remote.

    The local/pushed split is the point. The old check stopped at "is it committed", which
    accepts a commit that one `--amend` can rewrite after the result is known -- the very
    edit the module refuses to allow in the working tree.

    Reads the remote-TRACKING ref, and does no network call: grade() must work offline and
    must not stall on a hung remote. The ref is updated by this machine's own fetches and
    pushes, so it cannot report our own push as missing. It CAN lag a push made elsewhere,
    which fails toward "not frozen" -- the safe direction.
    """
    if not os.path.exists(_path(name)):
        return "absent", "no registration file"
    rel = _rel(name)
    if not _git("ls-files", "--error-unmatch", "--", rel):
        return "uncommitted", "untracked"
    if _git("status", "--porcelain", "--", rel):
        return "uncommitted", "staged or unstaged edits since the last commit"
    head_blob = _blob_at("HEAD", rel)
    if not head_blob:
        return "uncommitted", "not present in HEAD"
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if not upstream:
        return "unknown", "branch has no upstream"
    if _blob_at(upstream, rel) != head_blob:
        return "local", f"not on {upstream}"
    return "pushed", f"on {upstream} as {head_blob[:8]}"


def freeze(name, message=None):
    """Commit AND push the registration, then verify it landed. Raises if it did not.

    register() used to end by PRINTING the git commands and trusting whoever read them to
    run them. That is the same defect the monthly snapshot task had for months: the step
    everybody agrees is mandatory is the step that silently does not happen, and nothing
    in the system can tell the difference afterwards. This does it and checks it.

    Scoped on purpose: `git add -- <the one file>` and `git commit -- <the one file>`, so a
    freeze cannot sweep unrelated staged work into the bar's commit. No --force anywhere;
    if the push is rejected because the remote moved, that surfaces as a raise.
    """
    if not os.path.exists(_path(name)):
        raise FileNotFoundError(
            f"no pre-registration named {name!r} to freeze -- call register(...) first.")
    rel = _rel(name)
    state, detail = freeze_state(name)
    if state == "pushed":
        print(f"'{name}' is already frozen: {detail}")
        return state

    if state == "uncommitted":
        r = _git_run("add", "--", rel)
        if r.returncode != 0:
            raise RuntimeError(f"git add {rel} failed: {r.stderr.strip() or r.stdout.strip()}")
        r = _git_run("commit", "-m", message or f"prereg: {name}", "--", rel)
        if r.returncode != 0:
            raise RuntimeError(
                f"git commit of {rel} failed, so the bar is NOT frozen: "
                f"{r.stderr.strip() or r.stdout.strip()}")
        print(f"committed {rel}")

    r = _git_run("push")
    if r.returncode != 0:
        raise RuntimeError(
            f"the bar is committed but the PUSH FAILED, so it exists only on this machine "
            f"and can still be rewritten: {r.stderr.strip() or r.stdout.strip()}\n"
            f"Resolve by hand (do not --force), then re-run prereg.freeze({name!r}).")

    state, detail = freeze_state(name)
    if state != "pushed":
        raise RuntimeError(
            f"git push reported success but {rel} is still {state} ({detail}). Not treating "
            f"this bar as frozen; investigate before running the test.")
    print(f"FROZEN: {rel} {detail}")
    return state


def register(name, hypothesis, criteria, n_trials, notes="", overwrite=False):
    """Write the bar down before the run. Refuses to silently replace an existing one.

    `criteria` is a list of (metric, op, threshold). Every one must pass; there is no
    partial credit, because partial credit is where goalpost movement lives.
    `n_trials` is the honest count of everything tried on this question -- every variant of
    every sweep, including abandoned ones.
    """
    if not criteria:
        raise ValueError("a pre-registration with no criteria is not a pre-registration")
    for m, op, t in criteria:
        if op not in OPS:
            raise ValueError(f"unknown comparison {op!r}; use one of {sorted(OPS)}")
        float(t)
    if int(n_trials) < 1:
        raise ValueError("n_trials must be >= 1; if you genuinely tried once, say 1")

    os.makedirs(PREREG_DIR, exist_ok=True)
    p = _path(name)
    if os.path.exists(p) and not overwrite:
        raise FileExistsError(
            f"{p} already exists. Re-registering the same name after a run is exactly the "
            f"move this module exists to prevent. Pick a new name (v2, v3) so both bars "
            f"stay on the record, or pass overwrite=True and explain yourself in `notes`.")

    head = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    rec = dict(name=name, hypothesis=hypothesis,
               criteria=[[m, op, float(t)] for m, op, t in criteria],
               n_trials=int(n_trials), notes=notes,
               registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               git_head=head, tree_dirty=dirty)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")

    print(f"Pre-registered '{name}' -> {p}")
    print(f"  hypothesis: {hypothesis}")
    for m, op, t in criteria:
        print(f"  bar: {m} {op} {t}")
    print(f"  trials declared: {n_trials}")
    print("  FREEZE THIS NOW, before running the test:")
    print(f"    python -c \"import prereg; prereg.freeze('{name}')\"")
    print("  (commits AND pushes it. A bar that exists only on this disk can still be")
    print("   amended after you see the result, which is not a bar.)")
    return rec


def load(name):
    p = _path(name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"no pre-registration named {name!r}. A result cannot be graded against a bar "
            f"that was never written down -- call prereg.register(...) before the run.")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def grade(name, results, deflated_sharpe=None, unfrozen_ok=False):
    """Score results against the registered bar. Every criterion must pass.

    RAISES NotFrozen unless the registration is pushed. The bar has to exist somewhere you
    cannot quietly rewrite before this will put a verdict next to it; `freeze()` is how it
    gets there. Pass unfrozen_ok=True to score anyway -- for offline work, or a lab still
    being written -- and the report comes back stamped UNREGISTERED so it cannot be quoted
    as a registered result.

    Returns dict(passed, rows, committed, pushed, frozen, freeze_state, freeze_detail,
    n_trials, report). `report` is ASCII and printable.
    """
    rec = load(name)
    rows, all_pass = [], True
    for metric, op, thresh in rec["criteria"]:
        if metric not in results:
            rows.append((metric, op, thresh, None, False, "MISSING"))
            all_pass = False
            continue
        got = float(results[metric])
        ok = OPS[op](got, float(thresh))
        all_pass &= ok
        rows.append((metric, op, thresh, got, ok, "pass" if ok else "FAIL"))

    # Was the bar frozen before the run, or is it still editable? Asked live against git
    # rather than trusting the tree_dirty flag the record wrote about itself.
    state, state_detail = freeze_state(name)
    committed = state in ("local", "unknown", "pushed")
    pushed = state == "pushed"
    if not pushed and not unfrozen_ok:
        raise NotFrozen(
            f"the bar for {name!r} is {state} ({state_detail}), so there is nothing here to "
            f"grade against -- it can still be edited to fit the result you just saw.\n"
            f"  Freeze it:  python -c \"import prereg; prereg.freeze('{name}')\"\n"
            f"  If you mean to score an unfrozen bar anyway (offline, or a lab still being "
            f"written), pass unfrozen_ok=True. The report will say so.")

    lines = [
        f"PRE-REGISTERED RESULT -- {name}" if pushed else
        f"UNREGISTERED RESULT ({state.upper()}) -- {name}",
        f"  hypothesis: {rec['hypothesis']}",
        f"  registered: {rec['registered_utc']}  (git {rec.get('git_head') or 'n/a'}"
        f"{', tree dirty' if rec.get('tree_dirty') else ''})",
        f"  bar is {state.upper()}: {state_detail}",
        "",
        f"  {'metric':<18}{'bar':>14}{'observed':>12}   verdict",
    ]
    for metric, op, thresh, got, ok, tag in rows:
        shown = f"{got:.4f}" if got is not None else "--"
        lines.append(f"  {metric:<18}{op + ' ' + str(thresh):>14}{shown:>12}   {tag}")

    lines.append("")
    lines.append(f"  VERDICT{'' if pushed else ' (NOT A REGISTERED RESULT)'}: "
                 f"{'PASS' if all_pass else 'FAIL'} "
                 f"({sum(1 for r in rows if r[4])}/{len(rows)} criteria met)")

    if state == "uncommitted":
        lines.append("  [!] this pre-registration is NOT committed. A bar that can still be")
        lines.append("      edited after seeing the result is a note, not a pre-registration.")
        lines.append(f"      Fix: python -c \"import prereg; prereg.freeze('{name}')\"")
        lines.append("      Do that and re-run before quoting this verdict anywhere.")
    elif state == "local":
        lines.append(f"  [!] committed but NOT pushed ({state_detail}). A local commit is")
        lines.append("      still rewritable by --amend, reset or rebase after the result is")
        lines.append("      known, and it dies with the disk. Not evidence anyone can check.")
        lines.append(f"      Fix: python -c \"import prereg; prereg.freeze('{name}')\"")
    elif state == "unknown":
        lines.append("  [!] committed, but this branch has no upstream, so whether the bar")
        lines.append("      ever left this machine cannot be established. Treat as local.")

    ds = deflated_sharpe
    if ds is not None:
        lines.append(f"  deflated Sharpe at the declared {rec['n_trials']} trials: {ds:.3f}"
                     f"  {'clears 0.95' if ds > 0.95 else 'BELOW the 0.95 bar'}")
        if all_pass and ds <= 0.95:
            lines.append("  [!] passes its own bar but not the multiple-testing correction.")
            lines.append("      Failed Improvements 5 is the precedent: a lone marginal pass")
            lines.append("      ships nothing.")

    if all_pass:
        lines.append("  Reminder: a pass is a licence to look harder, not to deploy. Check")
        lines.append("  regime dependence (breaks.stability_report) before acting.")

    return dict(passed=all_pass, rows=rows, committed=committed, pushed=pushed,
                frozen=pushed, freeze_state=state, freeze_detail=state_detail,
                n_trials=rec["n_trials"], report="\n".join(lines))


def standard_battery(incumbent, candidate, n_trials, benchmark=None, labels=("incumbent",
                                                                             "candidate"),
                     periods_per_year=252, reps=1000):
    """The default evidence pack for any candidate-vs-incumbent question.

    Three things, because each catches what the others miss:
      1. confidence intervals   -- is the edge distinguishable from noise at all?
      2. multiple-testing price -- is it distinguishable once the search is counted?
      3. structural breaks      -- is it one edge, or one regime wearing an edge's clothes?

    Returns dict of numbers plus an ASCII `report`.
    """
    import stats_lib as sl
    import breaks as bk

    cmp_ = sl.compare(incumbent, candidate, n_trials=n_trials, labels=labels,
                      periods_per_year=periods_per_year, reps=reps)
    import pandas as pd
    spa = sl.spa_test(pd.Series(incumbent), pd.DataFrame({labels[1]: pd.Series(candidate)}),
                      reps=min(reps, 500))
    stab = bk.stability_report(candidate, benchmark=benchmark,
                               periods_per_year=periods_per_year)

    lines = ["STANDARD BATTERY", "", " 1) edge and uncertainty", cmp_["report"], "",
             " 2) multiple testing",
             f"  SPA consistent p = {spa['consistent']:.4f} over {n_trials} declared trials",
             f"  {spa['verdict']}", "",
             " 3) regime stability", stab["report"]]
    return dict(compare=cmp_, spa=spa, stability=stab, report="\n".join(lines))
