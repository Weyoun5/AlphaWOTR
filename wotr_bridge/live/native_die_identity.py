"""Join public native die moves and kills to their rendered gesture records."""
from collections import Counter
import re
from emitter import FP_DICE_POOL, SH_DICE_POOL, DICE_USED, FP_DEAD, SH_DEAD, HUNT_BOX
from .action_die_zones import zone, temporary, HUNT_ZONE

ZONES={FP_DICE_POOL:'fp dice',SH_DICE_POOL:'sa dice',DICE_USED:'current action',
       FP_DEAD:'fp casualties',SH_DEAD:'sa casualties',
       'Game.2037':'current fp card','Game.2036':'current sa card',HUNT_BOX:HUNT_ZONE}

# The replay executor reserves this menu form for its Fellowship handler,
# so it deliberately excludes it from _ANY_MOVE_RE. Identity joining still
# needs the public die movement that the native menu actually produces.
_HUNT_DIE_MOVE_RE = re.compile(
    r'<~Controls\.795~>([\d\s]+?)<~Controls\.796~><~([\w.]+)~>'
    r'<~Controls\.797~><~([\w.]+)~>')
_DIE_KILL_RE = re.compile(
    r'(?:<~Controls\.644~>([\d\s]+?)<~Controls\.645~>'
    r'|<~Controls\.296~>([\d\s]+?)<~Controls\.297~>)<~([\w.]+)~>')
_DIE_NAME_RE = re.compile(
    r'(?:\[(\d+)\]\s*)?(Free|Shadow) Action Die \(([ACEHMPW])\)')


def _die_owner(pid):
    import emitter
    return ('FREE_PEOPLES' if pid in emitter._FP_DIE_IDS
            else 'SHADOW' if pid in emitter._SH_DIE_IDS else None)


def kill_faces(rec):
    """Only public printed dice participate; mixed card identities stay out."""
    if rec.get('op') != 'kill' or not temporary(rec.get('zone')):
        return Counter()
    faces = Counter()
    for token in rec.get('pieces', ()):
        match = _DIE_NAME_RE.fullmatch(str(token).strip())
        if match is not None:
            owner = 'FREE_PEOPLES' if match[2] == 'Free' else 'SHADOW'
            faces[owner, match[3]] += int(match[1] or 1)
    return faces


def _kill_shape(actor, source, owners):
    return actor, source, tuple(sorted(owners.items()))

def relevant(source,target):
    return ((source in ('fp dice','sa dice') and (temporary(target) or target==HUNT_ZONE))
            or (temporary(source) and (temporary(target) or target in (
                'fp dice','sa dice','fp casualties','sa casualties',HUNT_ZONE)))
            or (source in ('fp casualties','sa casualties') and target in ('fp dice','sa dice')))

def ledger(driver):
    return driver.__dict__.setdefault('_native_die_evidence',
        {'seen':0,'tail':None,'commands':[],'records':[],
         'kill_commands':[], 'kill_records':[]})


def rebase(driver, lines, raw_count, feed_count):
    """Retire the accepted snapshot's prefix on both identity channels."""
    book = ledger(driver)
    # Keep already observed post-export commands, including a raw channel
    # that arrived before its rendered paragraph.
    seen = max(book['seen'], raw_count)
    book['seen'] = seen
    book['tail'] = lines[seen - 1] if seen else None
    book['commands'] = [row for row in book['commands'] if row['raw_index'] >= raw_count]
    book['records'] = [row for row in book['records']
                       if row[2].get('_feed_seq', 0) > feed_count]
    book['kill_commands'] = [row for row in book.get('kill_commands', ())
                             if row['raw_index'] >= raw_count]
    book['kill_records'] = [row for row in book.get('kill_records', ())
                            if row[2].get('_feed_seq', 0) > feed_count]

def observe_raw(driver, lines):
    from log_replay import LogReplayer
    import emitter
    from .driver import DriverHalt
    book=ledger(driver);lines=list(lines);seen=book['seen']
    if seen and (len(lines)<seen or lines[seen-1]!=book['tail']):
        raise DriverHalt('Native die-move identity history changed')
    new=[]
    for index,raw in enumerate(lines[seen:],seen):
        killed = _DIE_KILL_RE.search(raw)
        if killed is not None and temporary(ZONES.get(killed[3])):
            # The native Kill menu can remove cards and dice together. This
            # channel publishes only the fixed, already-public dice family.
            ids = [pid for pid in (killed[1] or killed[2]).split()
                   if _die_owner(pid) is not None]
            if ids:
                if len(ids) != len(set(ids)):
                    raise DriverHalt('Native die kill repeats a physical identity')
                actor = raw.split('<~', 1)[0].lstrip('$%').strip()
                if actor == '<auto>':
                    actor = 'auto>'
                book.setdefault('kill_commands', []).append({
                    'shape': _kill_shape(actor, ZONES[killed[3]],
                                         Counter(_die_owner(pid) for pid in ids)),
                    'pids': ids, 'raw_index': index})
            continue
        match=LogReplayer._ANY_MOVE_RE.search(raw) or _HUNT_DIE_MOVE_RE.search(raw)
        if match is None:continue
        pids=match.group(1).split();source=ZONES.get(match.group(2));target=ZONES.get(match.group(3))
        if not relevant(source,target):continue
        owner=('FREE_PEOPLES' if set(pids)<=set(emitter._FP_DIE_IDS)
               else 'SHADOW' if set(pids)<=set(emitter._SH_DIE_IDS) else None)
        if owner is None:continue  # Card/figure moves are not this channel.
        if not pids or len(pids)!=len(set(pids)):
            raise DriverHalt('Native die move repeats a physical identity')
        actor=raw.split('<~',1)[0].lstrip('$%').strip()
        if actor == '<auto>':
            actor = 'auto>'
        new.append({'shape':(actor,owner,source,target,len(pids)),
                    'pids':pids,'raw_index':index})
    book['commands'].extend(new);book['seen']=len(lines);book['tail']=lines[-1] if lines else None
    pair(driver,book)

def observe_record(driver,rec):
    faces = kill_faces(rec)
    if faces:
        owners = Counter()
        for (owner, _face), count in faces.items():
            owners[owner] += count
        shape = _kill_shape(str(rec.get('player', '')).strip(),
                            zone(rec.get('zone')), owners)
        book = ledger(driver)
        book.setdefault('kill_records', []).append((shape, faces, rec))
        pair(driver, book)
        return
    if rec.get('op')!='move':return
    source,target=zone(rec.get('from')),zone(rec.get('to'))
    if not relevant(source,target):return
    import rendered_replay as rr
    obj=str(rec.get('obj',''))
    tokens=rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
    rows=[]
    for token in tokens:
        match=re.fullmatch(r'(?:\[(\d+)\]\s*)?(Free|Shadow) Action Die \(([A-Z])\)',token.strip())
        if match is None:return
        rows.append((int(match.group(1)or 1),'FREE_PEOPLES' if match.group(2)=='Free' else 'SHADOW',match.group(3)))
    if len({row[1] for row in rows})!=1:return
    shape=(str(rec.get('player','')).strip(),rows[0][1],source,target,sum(row[0] for row in rows))
    ledger(driver)['records'].append((shape,Counter({face:sum(n for n,_,f in rows if f==face)
                                                   for _,_,face in rows}),rec))
    pair(driver,ledger(driver))

def _face_at_record(driver, pid, binding, record):
    """Public face at this gesture, including an uncommitted exact-PID change.

    Ring conversion and its paid action commit together when a die is killed.
    Until then the engine still has the old face. Only an earlier explicit
    change naming this same physical die can explain the printed new face;
    a later change must not rewrite a delayed staging record's old face.
    The retained window/prefix also gives Undo its ordinary lifecycle.
    """
    from .gesture_match import FACE_LETTER
    owner, index = binding
    face = FACE_LETTER.get(getattr(driver.es.dice[owner][index].current, 'name', None))
    window = getattr(driver.seg, 'window', None)
    records = list(getattr(driver, '_ring_gesture_records', None) or ())
    records.extend(getattr(window, 'records', ()) or ())
    boundary = record.get('_feed_seq')
    for changed in records:
        if changed is record:
            break
        sequence = changed.get('_feed_seq')
        if boundary is not None and sequence is not None and sequence >= boundary:
            continue
        if (changed.get('op') != 'die_change' or changed.get('side') != owner.name
                or changed.get('frm') != face or pid not in (pids(changed) or ())):
            continue
        if changed.get('to') in FACE_LETTER.values():
            face = changed['to']
    return face


def pair(driver,book):
    from .driver import DriverHalt, NativeDieFaceMismatch
    from .sidecar import meta_pids
    while book['commands'] and book['records']:
        command=book['commands'][0];shape,faces,rec=book['records'][0]
        if command['shape']!=shape:
            raise DriverHalt('Native die-move identity disagrees with rendered owner/source/destination/count')
        known=meta_pids(rec.get('meta'))
        if known is not None and set(known)!=set(command['pids']):
            raise DriverHalt('Native die-move identity conflicts with sidecar')
        # Recovery returns spent physical dice, whose printed faces may still
        # be the original roll after a ring-funded action. Their engine faces
        # encode that action's conversion, not a requirement for recovery.
        # The native PID/owner/source/count join above still applies.
        recovering = (shape[2], shape[3]) in (
            ('fp casualties', 'fp dice'), ('sa casualties', 'sa dice'))
        actual=[]
        for pid in command['pids']:
            binding=driver.view.die_id_map.get(pid)
            if binding is None:continue
            owner,index=binding
            if owner.name!=shape[1]:raise DriverHalt('Native die belongs to the other side')
            face=_face_at_record(driver,pid,binding,rec)
            if face is not None:actual.append(face)
        if not recovering and len(actual)==len(command['pids']) and Counter(actual)!=faces:
            raise NativeDieFaceMismatch('Native die identity has a different face from the gesture')
        rec['_native_die_pids']=list(command['pids'])
        rec['_native_die_raw_index']=command['raw_index']
        book['commands'].pop(0);book['records'].pop(0)
    commands = book.setdefault('kill_commands', [])
    records = book.setdefault('kill_records', [])
    while commands and records:
        command = commands[0]
        shape, faces, rec = records[0]
        if command['shape'] != shape:
            raise DriverHalt('Native die-kill identity disagrees with rendered owner/seat/count')
        known = meta_pids(rec.get('meta'))
        if known is not None and {pid for pid in known if _die_owner(pid) is not None} != set(command['pids']):
            raise DriverHalt('Native die-kill identity conflicts with sidecar')
        actual = Counter()
        for pid in command['pids']:
            binding = driver.view.die_id_map.get(pid)
            if binding is None:
                continue
            owner, index = binding
            if owner.name != _die_owner(pid):
                raise DriverHalt('Native die kill belongs to the other side')
            face = _face_at_record(driver, pid, binding, rec)
            if face is not None:
                actual[owner.name, face] += 1
        if sum(actual.values()) == len(command['pids']) and actual != faces:
            raise NativeDieFaceMismatch('Native die-kill identity has a different face from the gesture')
        rec['_native_die_pids'] = list(command['pids'])
        rec['_native_die_raw_index'] = command['raw_index']
        commands.pop(0)
        records.pop(0)

def pids(rec):
    from .sidecar import meta_pids
    return meta_pids(rec.get('meta')) or rec.get('_native_die_pids')
