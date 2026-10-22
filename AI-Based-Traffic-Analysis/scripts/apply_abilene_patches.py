"""One-time, idempotent edits so the existing Mininet/Ryu code can run the Abilene topology and a replay schedule.
   python scripts/apply_abilene_patches.py        then:  export T11_TOPOLOGY=abilene12
Selecting topology: T11_TOPOLOGY=<x> makes configs/topology_<x>.yaml the topology (unset = original NSFNET)."""
from pathlib import Path

R = Path(__file__).resolve().parents[1]


def sub(rel, old, new, marker):
    p = R / rel
    with open(p, newline="") as fh:                      # keep the file's own line endings (CRLF or LF)
        s = fh.read()
    crlf = "\r\n" in s
    if crlf:
        s, old, new = s.replace("\r\n", "\n"), old.replace("\n", "\n"), new
    if marker in s:
        print("already patched:", rel)
        return
    assert s.count(old) == 1, "%s: anchor not found exactly once:\n%s" % (rel, old)
    s = s.replace(old, new)
    with open(p, "w", newline="") as fh:
        fh.write(s.replace("\n", "\r\n") if crlf else s)
    print("patched:", rel)


sub("common/config.py",
    'def load_yaml(name: str) -> dict:\n    """load_yaml(\'topology\') -> dict from configs/topology.yaml"""\n    with open(CONFIG_DIR / f"{name}.yaml", "r") as f:',
    'def cfg_file(name: str) -> Path:\n    """configs/<name>.yaml; for \'topology\' honour T11_TOPOLOGY (-> topology_<value>.yaml)."""  # T11_TOPOLOGY\n'
    '    v = os.environ.get("T11_TOPOLOGY", "")\n    return CONFIG_DIR / (f"topology_{v}.yaml" if name == "topology" and v else f"{name}.yaml")\n\n\n'
    'def load_yaml(name: str) -> dict:\n    """load_yaml(\'topology\') -> dict from configs/topology.yaml"""\n    with open(cfg_file(name), "r") as f:',
    "T11_TOPOLOGY")
sub("common/config.py", 'shutil.copy(CONFIG_DIR / f"{name}.yaml", run / f"{name}.yaml")',
    'shutil.copy(cfg_file(name), run / f"{name}.yaml")', "cfg_file(name), run")
sub("topology/portmap.py",
    '    with open(ROOT / "configs" / ("%s.yaml" % name)) as f:',
    '    import os\n    v = os.environ.get("T11_TOPOLOGY", "")\n    fn = "topology_%s.yaml" % v if (name == "topology" and v) else "%s.yaml" % name\n'
    '    with open(ROOT / "configs" / fn) as f:', "T11_TOPOLOGY")
sub("topology/mininet_topology.py",
    '    assert G.number_of_nodes() == 14 and G.number_of_edges() == 21, "NSFNET must be 14 nodes / 21 links"',
    '    if cfg.get("name", "nsfnet14") == "nsfnet14":\n        assert G.number_of_nodes() == 14 and G.number_of_edges() == 21, "NSFNET must be 14 nodes / 21 links"',
    'cfg.get("name"')
sub("experiments/run_baseline.py",
    '    ap.add_argument("--check", action="store_true", help="only build net + pingall")',
    '    ap.add_argument("--check", action="store_true", help="only build net + pingall")\n'
    '    ap.add_argument("--schedule", default=None, help="CSV flow schedule (e.g. Abilene replay) instead of the synthetic generator")',
    "--schedule")
sub("experiments/run_baseline.py",
    '        sched, pairs, w, info = build_schedule(topo, tcfg, a.load, a.profile, total, a.seed)',
    '        if a.schedule:\n            from abilene.replay import load_schedule\n            sched, pairs, w, info = load_schedule(a.schedule, topo)\n'
    '        else:\n            sched, pairs, w, info = build_schedule(topo, tcfg, a.load, a.profile, total, a.seed)',
    "load_schedule")
gi = R / ".gitignore"
t = gi.read_text() if gi.exists() else ""
if "models/checkpoints/abilene/" not in t:
    gi.write_text(t.rstrip("\n") + "\n# Abilene pipeline: raw data and trained checkpoints stay out of Git\ndata/raw/\nmodels/checkpoints/abilene/\nresults/abilene/\n")
    print("patched: .gitignore")
