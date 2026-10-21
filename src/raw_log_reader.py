"""Read client log commands strictly while preserving opaque chat bytes.

The Java client writes with FileWriter's platform default charset. No single
legacy charset can be inferred from these logs. Only undecodable plain-chat
bodies may remain opaque: surrogateescape preserves their exact bytes and
their position as a boundary between commands. Action records remain UTF-8.
"""
from pathlib import Path
import re


class RawLogEncodingError(ValueError):
    reason = 'coverage_raw_log_encoding_unproven'

    def __init__(self, detail):
        self.detail = detail
        super().__init__(self.reason, detail)


_CHAT = re.compile(rb'<([!-~]+)> ([^\r\n]*)\r?\n?\Z')
_RESERVED = {b'auto', b'game', b'client', b'host', b'server', b'replay', b'system'}


def decode_raw_log(data, source=None):
    """Decode bytes without guessing a charset or dropping any input byte.

    Invalid bytes are permitted only in the body of an ASCII-framed chat
    line. Protocol-looking chat and non-ASCII speaker identities cannot
    establish that boundary and are refused. Valid UTF-8 is returned intact.
    """
    pieces = []
    offset = 0
    for line_number, line in enumerate(data.splitlines(keepends=True), 1):
        try:
            text = line.decode('utf-8')
        except UnicodeDecodeError as error:
            match = _CHAT.fullmatch(line)
            detail = 'undecodable_non_chat_record'
            if match:
                speaker, body = match.groups()
                if (b'<' in speaker or b'>' in speaker or b'~' in speaker
                        or speaker.lower() in _RESERVED):
                    detail = 'undecodable_reserved_or_malformed_chat'
                elif (b'<~' in body or re.search(rb'\b(?:controls|game|scribe|talker|interpreter)\.', body, re.I)):
                    detail = 'undecodable_protocol_looking_chat'
                elif any(byte < 32 or byte == 127 for byte in body):
                    detail = 'undecodable_chat_control_byte'
                else:
                    text = line.decode('utf-8', errors='surrogateescape')
                    pieces.append(text)
                    offset += len(line)
                    continue
            raise RawLogEncodingError({
                'detail': detail, 'source': str(source) if source is not None else None,
                'raw_line': line_number, 'byte_offset': offset + error.start,
                'invalid_bytes_hex': line[error.start:error.end].hex(),
            }) from error
        pieces.append(text)
        offset += len(line)
    return ''.join(pieces)


def read_raw_log_text(path):
    return decode_raw_log(Path(path).read_bytes(), source=path)
