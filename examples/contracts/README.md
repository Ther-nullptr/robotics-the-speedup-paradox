# Synthetic contract example

`synthetic.manifest.json` and `synthetic.trace.jsonl` are hand-constructed protocol fixtures. They contain no model inference, robot execution, benchmark observations, or measured hardware performance.

The example has one episode and one action. Its logical timestamps place output completion at 80 ms, release at 100 ms, action start at 100 ms, and termination at 150 ms. The declared profile is synthetic. The distinct release event demonstrates that output completion alone does not grant permission to execute an action.

From the repository root:

```bash
python tools/validate_contracts.py --examples
```

Expected output:

```text
VALID synthetic-run: 8 events
```

Version 1 supports **sequential episodes in a single run**. A new episode requires the previous episode's terminal boundary. Clock domains keep the same origin across episode resets: timestamps must continue monotonically instead of restarting at zero. Started requests may report late completion/drop diagnostics for an old episode, but may not release or execute old actions.

For rejection cases, see `tests/test_contracts.py`. Tests use independent fixtures to exercise ordering, clock domains, associations, episode boundaries, missing terminals, invalid JSON and zero-success summaries. They do not run a simulator, verify an inference scheduler, or establish that reported metrics equal real experiment outcomes. More details are in `schemas/README.md`.
