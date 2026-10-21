"""Pin external inputs used by the Python oracle and immutable native tables."""
from pathlib import Path
import hashlib

RULE_DATA_FILES = frozenset(('card_list.csv', 'wotr_map.txt', 'board200.jpg'))


def geometry_provenance():
    import shapely
    return dict(shapely=shapely.__version__, geos=getattr(shapely, 'geos_version_string', None))


def fingerprint_rule_data(directory, expected):
    if set(expected) != RULE_DATA_FILES:
        raise RuntimeError('Rule-data manifest must pin the card list, map and board image')
    directory = Path(directory)
    actual = {}
    for name in sorted(RULE_DATA_FILES):
        try:
            digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        except OSError as error:
            raise RuntimeError('Cannot read pinned rule data: ' + str(directory / name)) from error
        if digest != expected[name]:
            raise RuntimeError('Rule data differs from pin: ' + str(directory / name))
        actual[name] = digest
    return actual


def reference_rule_data(reference):
    return fingerprint_rule_data(Path(reference['runtime']) / 'data', reference.get('data_sha256', {}))
