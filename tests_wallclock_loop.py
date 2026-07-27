"""Exercise the wall-clock refinement loop with stubs (no GPU, no AF, no evodiff model).

The bug this is really guarding against: the loop exits at the TOP, which is
*after* the end-of-iteration re-mask, so a naive `return sample` hands back
sequences containing mask tokens. Every assertion about 'no X in sequence' below
is checking that.
"""
import os
import sys
import tempfile
import time
import types

import numpy as np
import torch

REPO = "/Users/nbhattacharya/repos/ProDifEvo-Refinement"
sys.path.insert(0, REPO)

# evodiff/generate_antibody.py imports heavy siblings at package import time;
# load the module file directly to keep this test hermetic.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "gen_ab", os.path.join(REPO, "evodiff", "generate_antibody.py"))
gen_ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_ab)

VOCAB = 21
MASK_ID = 20
SEQ_LEN = 30
CDR = list(range(10, 20))
FRAMEWORK = [i for i in range(SEQ_LEN) if i not in CDR]
REPEAT = 4


class Tok:
    mask_id = MASK_ID

    def untokenize(self, s):
        return "".join("X" if int(t) == MASK_ID else "ACDEFGHIKLMNPQRSTVWY"[int(t) % 20]
                       for t in s)


class Model:
    def __call__(self, sample, timestep):
        return torch.randn(sample.shape[0], sample.shape[1], VOCAB)


class Reward:
    """Stub reward. `per_iter_sleep` fakes iteration cost for deadline tests."""

    def __init__(self, per_iter_sleep=0.0):
        self.metrics_name = ["iptm"]
        self._timings = {"n_sequences": 0}
        self.per_iter_sleep = per_iter_sleep
        self.seen = []

    def reward_metrics(self, protein_name, mask_for_loss, S_sp, ori_pdb_file,
                       save_pdb=False, add_info=None):
        time.sleep(self.per_iter_sleep)
        self._timings["n_sequences"] += S_sp.shape[0]
        # Record whether anything we were asked to score contained a mask token.
        self.seen.append(bool((S_sp == MASK_ID).any().item()))
        n = S_sp.shape[0]
        return [[0.5] for _ in range(n)], [0.5] * n, None


def run(iteration, wallclock=None, per_iter_sleep=0.0, should_stop=None):
    folder = tempfile.mkdtemp()
    reward = Reward(per_iter_sleep)
    init = torch.randint(0, 20, (REPEAT, SEQ_LEN))
    ckpts = []
    deadline = (time.time() + wallclock) if wallclock else None
    sample, seqs = gen_ab.generate_oaardm_cdr_edit(
        Model(), Tok(), SEQ_LEN, reward,
        ori_pdb_file_path=None, batch="t",
        mask_for_loss=torch.ones((REPEAT, SEQ_LEN)),
        repeat_num=REPEAT, candidate=2, folder_path=folder,
        device=torch.device("cpu"), cdr_indices=CDR, framework_indices=FRAMEWORK,
        iteration=iteration, edit_fraction=0.3, initial_sample=init,
        deadline=deadline,
        checkpoint_cb=lambda i, s, pm, agg, mn: ckpts.append((i, list(s))),
        should_stop=should_stop,
    )
    return sample, seqs, ckpts, folder, init


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    return cond


ok = True

# 1. Iteration-capped run behaves exactly as before the change.
sample, seqs, ckpts, folder, init = run(iteration=4)
ok &= check("iteration cap: runs exactly 4 iterations", len(ckpts) == 4)
ok &= check("iteration cap: returned seqs have NO mask tokens",
            not any("X" in s for s in seqs))
ok &= check("iteration cap: returned tensor has no mask id",
            not bool((sample == MASK_ID).any().item()))
ok &= check("iteration cap: framework preserved",
            torch.equal(sample[:, FRAMEWORK], init[0, FRAMEWORK].repeat(REPEAT, 1)))
ok &= check("iteration cap: checkpoint seqs also clean",
            not any("X" in s for _, cs in ckpts for s in cs))

# 2. Deadline stops the run early and still returns clean sequences.
sample, seqs, ckpts, folder, init = run(iteration=1000, wallclock=1.2,
                                        per_iter_sleep=0.30)
ok &= check("deadline: stopped well before the 1000-iteration cap",
            0 < len(ckpts) < 1000)
ok &= check("deadline: returned seqs have NO mask tokens",
            not any("X" in s for s in seqs))
ok &= check("deadline: framework preserved",
            torch.equal(sample[:, FRAMEWORK], init[0, FRAMEWORK].repeat(REPEAT, 1)))
ok &= check("deadline: last checkpoint matches returned sequences",
            ckpts and ckpts[-1][1] == seqs)

# 3. SIGTERM-style stop flag winds up at the next boundary.
flag = {"v": False}
calls = {"n": 0}


def stopper():
    calls["n"] += 1
    if calls["n"] > 3:
        flag["v"] = True
    return flag["v"]


sample, seqs, ckpts, folder, init = run(iteration=1000, should_stop=stopper)
ok &= check("signal: stopped early", 0 < len(ckpts) < 1000)
ok &= check("signal: returned seqs have NO mask tokens",
            not any("X" in s for s in seqs))

# 4. A budget smaller than one iteration still completes iteration 0 and returns
#    valid designs, OVERRUNNING the budget. This is deliberate -- the deadline is
#    only ever checked at a boundary, so the first iteration is uninterruptible.
#    The consequence for real runs: the wall-clock budget must comfortably exceed
#    one from-scratch iteration (~5h for RERD at repeatnum=100), or the job blows
#    its budget and gets SIGKILLed with only checkpoints to show for it.
sample, seqs, ckpts, folder, init = run(iteration=1000, wallclock=0.001,
                                        per_iter_sleep=0.2)
ok &= check("tiny budget: still completes exactly one iteration", len(ckpts) == 1)
ok &= check("tiny budget: returns valid (unmasked) sequences",
            not any("X" in s for s in seqs))

# 5. timing.csv rows == iterations actually completed (drives the summary).
import csv as _csv
sample, seqs, ckpts, folder, init = run(iteration=3)
with open(os.path.join(folder, "timing.csv")) as fh:
    rows = list(_csv.reader(fh))
ok &= check("timing.csv has header + one row per completed iteration",
            len(rows) == 1 + len(ckpts))

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
