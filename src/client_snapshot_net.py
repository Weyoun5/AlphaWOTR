"""Native snapshot command grammar and paired rendered undo validation."""
import re
SNAPSHOT_KEYS = {'CardBuffet.3': 'moves',
 'CardBuffet.9': 'moves',
 'CardBuffet.14': 'moves',
 'Controls.0': 'draws',
 'Controls.2': 'moves',
 'Controls.3': 'moves',
 'Controls.132': 'moves',
 'Controls.271': 'moves',
 'Controls.276': 'moves',
 'Controls.281': 'moves',
 'Controls.296': 'kills',
 'Controls.304': 'kills',
 'Controls.313': 'moves',
 'Controls.323': 'flips',
 'Controls.350': 'allocates',
 'Controls.358': 'moves',
 'Controls.366': 'moves',
 'Controls.372': 'allocates',
 'Controls.452': 'draws',
 'Controls.491': 'draws',
 'Controls.500': 'draws',
 'Controls.509': 'draws',
 'Controls.518': 'draws',
 'Controls.528': 'draws',
 'Controls.534': 'moves',
 'Controls.539': 'moves',
 'Controls.557': 'draws',
 'Controls.567': 'draws',
 'Controls.573': 'moves',
 'Controls.578': 'moves',
 'Controls.587': 'draws',
 'Controls.624': 'draws',
 'Controls.630': 'draws',
 'Controls.637': 'draws',
 'Controls.644': 'kills',
 'Controls.650': 'moves',
 'Controls.655': 'moves',
 'Controls.662': 'kills',
 'Controls.670': 'kills',
 'Controls.674': 'kills',
 'Controls.680': 'kills',
 'Controls.684': 'kills',
 'Controls.726': 'moves',
 'Controls.731': 'moves',
 'Controls.736': 'moves',
 'Controls.743': 'moves',
 'Controls.750': 'moves',
 'Controls.757': 'moves',
 'Controls.762': 'moves',
 'Controls.767': 'moves',
 'Controls.779': 'moves',
 'Controls.784': 'moves',
 'Controls.790': 'moves',
 'Controls.795': 'moves',
 'Controls.802': 'moves',
 'Controls.811': 'moves',
 'Controls.851': 'moves',
 'Controls.856': 'moves',
 'Controls.861': 'moves',
 'Controls.866': 'moves',
 'Controls.873': 'moves',
 'Controls.878': 'moves',
 'Controls.888': 'moves',
 'DiceBuffet.1': 'moves',
 'HighlightLabel.81': 'moves',
 'HighlightLabel.86': 'moves',
 'HighlightLabel.91': 'moves',
 'HighlightLabel.96': 'moves',
 'HighlightLabel.101': 'moves',
 'Interpreter.48': 'moves',
 'Interpreter.50': 'draws',
 'Interpreter.51': 'kills',
 'Interpreter.52': 'flips',
 'Interpreter.54': 'allocates',
 'Interpreter.103': 'kills',
 'Interpreter.105': 'moves',
 'Interpreter.126': 'allocates',
 'Interpreter.132': 'allocates',
 'Interpreter.144': 'draws',
 'Interpreter.158': 'kills',
 'Interpreter.187': 'flips',
 'Interpreter.231': 'moves',
 'Interpreter.236': 'moves',
 'Interpreter.241': 'moves',
 'Interpreter.246': 'moves',
 'Interpreter.251': 'moves',
 'Interpreter.256': 'moves',
 'Interpreter.261': 'moves',
 'Interpreter.266': 'moves',
 'Interpreter.271': 'moves',
 'Interpreter.276': 'moves',
 'Interpreter.281': 'moves',
 'Interpreter.286': 'moves',
 'Interpreter.291': 'moves',
 'Interpreter.296': 'moves',
 'Interpreter.301': 'moves',
 'Interpreter.306': 'moves',
 'Interpreter.311': 'moves',
 'Interpreter.316': 'moves',
 'Interpreter.321': 'moves',
 'Interpreter.327': 'moves',
 'Interpreter.332': 'moves',
 'Interpreter.337': 'moves',
 'Interpreter.342': 'moves',
 'Interpreter.347': 'moves',
 'Interpreter.352': 'moves',
 'Interpreter.357': 'moves',
 'Interpreter.362': 'moves',
 'Interpreter.367': 'moves',
 'Interpreter.372': 'moves',
 'Interpreter.377': 'moves',
 'Interpreter.382': 'moves',
 'Interpreter.387': 'moves',
 'Interpreter.392': 'moves',
 'Interpreter.397': 'moves',
 'Interpreter.402': 'moves',
 'Interpreter.407': 'moves',
 'Interpreter.412': 'moves',
 'Interpreter.417': 'moves',
 'Interpreter.422': 'moves',
 'Interpreter.427': 'moves',
 'Interpreter.432': 'moves',
 'Interpreter.437': 'moves',
 'Interpreter.442': 'moves',
 'Interpreter.447': 'moves',
 'Interpreter.452': 'moves',
 'Interpreter.457': 'moves',
 'Scribe.40': 'moves',
 'Scribe.20': 'draws',
 'Scribe.50': 'moves',
 'Scribe.61': 'draws',
 'Scribe.73': 'moves',
 'Scribe.33': 'moves',
 'Scribe.14': 'moves',
 'Scribe.24': 'moves',
 'Scribe.54': 'moves',
 'Scribe.65': 'moves',
 'Scribe.45': 'moves',
 'Scribe.85': 'moves',
 'Scribe.17': 'moves',
 'Scribe.77': 'moves',
 'Scribe.28': 'moves',
 'Scribe.58': 'moves',
 'Scribe.88': 'moves',
 'Scribe.69': 'moves',
 'Scribe.141': 'moves',
 'Scribe.7': 'moves',
 'Scribe.6': 'moves'}
FIRST_KEY = re.compile(r'<~([^~]+)~>')
def snapshot_kind(line):
    if not (line.startswith('$') or line.startswith('<auto>')): return None
    match=FIRST_KEY.search(line)
    return SNAPSHOT_KEYS.get(match.group(1)) if match else None

def net_raw(lines, with_events=False):
    out=list(lines); stack=[]; redo=[]; events=[]
    for index,line in enumerate(lines):
        if not (line.startswith('$') or line.startswith('<auto>')):continue
        if '<~Controls.819~>' in line:
            targets=[]
            while stack:
                target=stack.pop();redo.append(target);out[target]='';targets.append(target+1)
                # One native Undo consumes preceding automatic snapshot labels.
                if not stack or not lines[stack[-1]].startswith('<auto>'):break
            out[index]='';events.append({'line':index+1,'op':'undo','targets':targets})
        elif '<~Controls.16~>' in line:
            targets=[]
            while redo:
                target=redo.pop();stack.append(target);out[target]=lines[target];targets.append(target+1)
                # Native Redo repeats after the just-restored automatic label.
                if not lines[target].startswith('<auto>'):break
            out[index]='';events.append({'line':index+1,'op':'redo','targets':targets})
        elif snapshot_kind(line):
            stack.append(index);redo.clear()
    return (out,events) if with_events else out


class SnapshotPairError(ValueError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(str(detail))


def rendered_snapshot_kind(record):
    op = record.get('op')
    if op in ('move', 'kill', 'flip', 'draw'):
        return op + 's'
    if op in ('alloc_eye', 'add_eye'):
        return 'allocates'
    if re.fullmatch(r'\S+ randomly discards a (?:Free|Shadow) (?:Character|Strategy) Card\.',
                    str(record.get('body') or '')):
        return 'draws'
    return None


def validate_rendered_pair(records, raw_lines):
    """Require the original native snapshot sequence and exact undo target order.

    Older Java replay rewrote <auto> labels to %auto>, disabling native
    automatic undo folding. Its internally consistent text is insufficient.
    This checks paired raw commands before any extraction can admit that text.
    Unlabelled snapshots touched by undo remain unsupported, not guessed.
    """
    raw_snapshots = []
    silent = False
    for line_number, line in enumerate(raw_lines, 1):
        command = line.startswith('$') or line.startswith('<auto>')
        if command and ' silent ' in line:
            silent = True
        if command and ' noisy ' in line:
            silent = False
        kind = snapshot_kind(line)
        empty_flip = bool(re.fullmatch(
            r'\$[^<]+<~Controls\.323~>\s*(?:[A-F0-9]{6}|null)?', line))
        if kind and not silent and not empty_flip:
            raw_snapshots.append((line_number, kind))
    rendered_snapshots, rendered_events, stack, redo = [], [], [], []
    for record in records:
        kind = rendered_snapshot_kind(record)
        if kind:
            stack.append(len(rendered_snapshots))
            rendered_snapshots.append((record.get('line'), kind))
            redo.clear()
        elif record.get('op') in ('undo', 'redo'):
            op = record['op']
            source = stack if op == 'undo' else redo
            if not source:
                raise SnapshotPairError({'detail': 'rendered_snapshot_stack_empty',
                                         'line': record.get('line'), 'operation': op})
            target = source.pop()
            (redo if op == 'undo' else stack).append(target)
            rendered_events.append((op, target))
    if [kind for _, kind in raw_snapshots] != [kind for _, kind in rendered_snapshots]:
        raise SnapshotPairError({'detail': 'raw_rendered_snapshot_sequence_mismatch',
                                 'raw_snapshots': len(raw_snapshots),
                                 'rendered_snapshots': len(rendered_snapshots)})
    raw_ordinals = {line: i for i, (line, _) in enumerate(raw_snapshots)}
    _, events = net_raw(raw_lines, with_events=True)
    raw_events = []
    for event in events:
        for line in event['targets']:
            if line not in raw_ordinals:
                raise SnapshotPairError({'detail': 'undo_unlabelled_snapshot_unproven',
                                         'raw_line': event['line'], 'target_raw_line': line})
            raw_events.append((event['op'], raw_ordinals[line]))
    if raw_events != rendered_events:
        difference = next((i for i, (a, b) in enumerate(zip(raw_events, rendered_events))
                           if a != b), min(len(raw_events), len(rendered_events)))
        raise SnapshotPairError({'detail': 'raw_rendered_undo_target_mismatch',
                                 'event_ordinal': difference,
                                 'raw_events': len(raw_events),
                                 'rendered_events': len(rendered_events)})
    return {'snapshots': len(raw_snapshots), 'undo_redo_targets': len(raw_events)}
