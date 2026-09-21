"""Stream-validate capture coverage; optionally preserve a crash's valid prefix.

Does not execute GPU work or certify gameplay. Pixel payloads remain local.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct

EVENTS = ('unused', 'begin', 'end', 'create', 'destroy', 'upload', 'submit',
          'external', 'checkpoint', 'display', 'native_begin', 'native_end',
          'lifetime', 'visual', 'stop')
REASONS = ('closed', 'byte_limit', 'time_limit', 'io_error',
           'capture_error', 'low_address_space', 'unsupported')


def inspect(path):
    counts = Counter()
    operations = Counter()
    rejected = Counter()
    calls = {}
    valid_bytes = 16
    footer = None
    last_tick = 0
    visual_ready = 0
    with Path(path).open('rb') as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError('Missing file header')
        magic, version, frequency = struct.unpack('<IIQ', header)
        if magic != 0x52433343 or version != 2 or not 0 < frequency < 10**12:
            raise ValueError('Unsupported recording header')
        sequence = 0
        truncated = False
        while True:
            header = stream.read(40)
            if not header:
                break
            if footer is not None:
                raise ValueError('Data after footer')
            if len(header) != 40:
                truncated = True
                break
            marker, kind, size, expected, ordinal, owner, ticks = struct.unpack('<IIIIQQQ', header)
            if marker != 0x31523343 or not 0 < kind < len(EVENTS) or size > 12 * 1024 * 1024 or ordinal != sequence + 1:
                raise ValueError('Invalid event header or sequence gap')
            payload = stream.read(size)
            if len(payload) != size:
                truncated = True
                break
            # The executable verifies the per-record checksum and full payload
            # schema. This fast inspector establishes framing only; it must not
            # claim integrity validation from sizes/sequence alone.
            sequence = ordinal
            counts[EVENTS[kind]] += 1
            last_tick = ticks
            valid_bytes = stream.tell()
            if kind == 10:
                if size != 36:
                    raise ValueError('Invalid native begin')
                token, op = struct.unpack_from('<QI', payload)
                operations[op] += 1
                if len(calls) >= 8192:
                    raise ValueError('Unbounded native call nesting')
                calls[token] = op
            elif kind == 11:
                if size != 12:
                    raise ValueError('Invalid native end')
                token, result = struct.unpack('<Qi', payload)
                op = calls.pop(token, None)
                if op is None:
                    raise ValueError('Unmatched native end')
                if result <= 0:
                    rejected[op] += 1
            elif kind == 13:
                if size != 48:
                    raise ValueError('Invalid visual observation')
                visual_ready += bool(struct.unpack_from('<I', payload, 44)[0])
            elif kind == 14:
                if owner or size != 4:
                    raise ValueError('Invalid stop footer')
                reason, = struct.unpack('<I', payload)
                if reason >= len(REASONS):
                    raise ValueError('Unknown stop reason')
                footer = REASONS[reason]
    return {'scope': 'recording_framing_and_coverage_only', 'payload_integrity_checked': False,
            'events': dict(counts), 'native_operations': dict(operations),
            'native_zero_or_negative_results': dict(rejected),
            'visual_ready_observations': visual_ready, 'unfinished_native_calls': len(calls),
            'recorded_seconds': last_tick / frequency, 'valid_prefix_bytes': valid_bytes,
            'truncated_tail': truncated, 'stop_reason': footer,
            'full_gameplay_replayed': False, 'performance_accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--prefix-copy', type=Path,
                        help='Save complete framed records to a new file for strict GPU replay validation')
    args = parser.parse_args()
    report = inspect(args.recording)
    if args.prefix_copy:
        # Exclusive creation: never overwrite the original or an earlier copy.
        with args.recording.open('rb') as source, args.prefix_copy.open('xb') as destination:
            remaining = report['valid_prefix_bytes']
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError('Recording changed while copying')
                destination.write(block)
                remaining -= len(block)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
