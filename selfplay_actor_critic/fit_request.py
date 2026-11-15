"""Exact PG invocation identity, shared by fitting and read-only resume checks."""
from pathlib import Path
import json

from .contracts import CONTRACT, canonical_json
from .durability import behavior_content, digest_file, prediction_receipt


def fitting_request(args):
    # Import lazily: admission and selection order belong to the PG loader.
    from .pg_fit import _training_files
    settings = {key: value for key, value in vars(args).items()
                if key not in {'run', 'critic_dir', 'output', 'device',
                               'check_generation', 'request_receipt'}}
    inputs, extra = [], []
    for value in args.run:
        run = Path(value).resolve()
        files = _training_files(run, args.side, args.max_games)
        inputs.append(dict(run=str(run), receipt=prediction_receipt(run, args.side, args.critic_dir, files)))
        if args.traced_cache:
            extra.append(Path(args.traced_cache) / (run.name + '.npz'))
        if args.luck:
            extra.append(run / ('luck_' + args.luck + '.npz'))
    if args.traced and not args.traced_cache:
        extra.append(Path(args.traced))
    request = dict(schema='pg_fit_request_v2', contract=CONTRACT.to_dict(), settings=settings, inputs=inputs,
                anchor=behavior_content(args.anchor, args.side) if args.anchor else None,
                extra_inputs=[dict(path=str(p.resolve()), sha256=digest_file(p)) for p in extra])
    return json.loads(canonical_json(request))
