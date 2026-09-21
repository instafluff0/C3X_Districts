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


def inspect(path, *, timeline=None, target_seconds=600):
    if not 1 <= target_seconds <= 3600:
        raise ValueError('Target duration must be between 1 and 3600 seconds')
    counts = Counter()
    event_bytes = Counter()
    seconds = {}
    operations = Counter()
    rejected = Counter()
    calls = {}
    valid_bytes = 16
    footer = None
    last_tick = 0
    first_tick = None
    display_count = 0
    previous_display = None
    maximum_display_gap = 0
    visual_ready = 0
    with Path(path).open('rb') as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError('Missing file header')
        magic, version, frequency = struct.unpack('<IIQ', header)
        if magic != 0x52433343 or version not in (2, 3) or not 0 < frequency < 10**12:
            raise ValueError('Unsupported recording header')
        sequence = 0
        truncated = False
        while True:
            offset = stream.tell()
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
            if ticks < last_tick:
                raise ValueError('Recording clock moved backwards')
            # The executable verifies the per-record checksum and full payload
            # schema. This fast inspector establishes framing only; it must not
            # claim integrity validation from sizes/sequence alone.
            sequence = ordinal
            counts[EVENTS[kind]] += 1
            event_bytes[EVENTS[kind]] += 40 + size
            if first_tick is None:
                first_tick = ticks
            # Bounded report even for a recording longer than the requested
            # interval: the final bucket aggregates everything after it.
            second = min(ticks // frequency, target_seconds)
            bucket = seconds.setdefault(second, Counter())
            bucket['events'] += 1
            bucket['bytes'] += 40 + size
            bucket[EVENTS[kind]] += 1
            last_tick = ticks
            valid_bytes = stream.tell()
            if kind == 9:
                if size != 28:
                    raise ValueError('Invalid composition display')
                surface, accepted, left, top, right, bottom = struct.unpack('<QIiiii', payload)
                if accepted not in (0, 1):
                    raise ValueError('Invalid composition display result')
                if accepted:
                    display_count += 1
                    if previous_display is not None:
                        maximum_display_gap = max(maximum_display_gap, ticks - previous_display)
                    previous_display = ticks
                    if timeline is not None:
                        timeline.write(json.dumps({'display': display_count, 'sequence': ordinal,
                            'file_offset': offset, 'ticks': ticks, 'frequency': frequency,
                            'seconds': ticks / frequency, 'stream': owner, 'surface': surface,
                            'area': [left, top, right, bottom],
                            'endpoint': 'composition_draw_accepted',
                            'physical_display_confirmed': False}) + '\n')
            elif kind == 10:
                if size != (40 if version >= 3 else 36):
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
    elapsed = (last_tick - first_tick) / frequency if first_tick is not None else 0
    rate = sum(event_bytes.values()) / elapsed if elapsed else None
    return {'scope': 'recording_framing_and_coverage_only', 'payload_integrity_checked': False,
            'version': version, 'target_seconds': target_seconds,
            'events': dict(counts), 'native_operations': dict(operations),
            'event_bytes': dict(event_bytes),
            'seconds': [{'second': n, 'aggregated_tail': n == target_seconds, **values}
                        for n, values in sorted(seconds.items())],
            'capacity': {'observed_event_bytes_per_second': rate,
                         'linear_projection_bytes': int(rate * target_seconds) if rate else None,
                         'projection_is_capacity_guarantee': False},
            'composition_displays': display_count,
            'maximum_between_display_gap_seconds': maximum_display_gap / frequency,
            'display_endpoint': 'composition_draw_accepted_not_scanout',
            'native_zero_or_negative_results': dict(rejected),
            'visual_ready_observations': visual_ready, 'unfinished_native_calls': len(calls),
            'recorded_seconds': last_tick / frequency, 'valid_prefix_bytes': valid_bytes,
            'truncated_tail': truncated, 'stop_reason': footer,
            'renderer_input_replay_ready': False,
            'missing_input_families': ['asset_config_manifest', 'initial_renderer_state',
                'authoritative_scene_and_world_updates', 'unit_lifecycle_and_action_inputs',
                'camera_requests_and_adoption', 'native_adapter_inputs_and_cpu_writes',
                'retained_ambient_inputs_and_visual_opportunities', 'presentation_outcomes'],
            'full_gameplay_replayed': False, 'performance_accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--timeline', type=Path, help='Write every accepted composition display as JSONL; not scanout evidence')
    parser.add_argument('--report', type=Path, help='Save the coverage/capacity report')
    parser.add_argument('--require-input-replay', action='store_true', help='Fail unless the recording can regenerate complete renderer inputs')
    parser.add_argument('--prefix-copy', type=Path,
                        help='Save complete framed records to a new file for strict GPU replay validation')
    args = parser.parse_args()
    if args.timeline:
        with args.timeline.open('x') as timeline:
            report = inspect(args.recording, timeline=timeline)
    else:
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
    if args.report:
        with args.report.open('x') as output:
            json.dump(report, output, indent=2)
            output.write('\n')
        print(f'Coverage report saved: {args.report}')
    else:
        print(json.dumps(report, indent=2))
    if args.require_input_replay and not report['renderer_input_replay_ready']:
        parser.exit(2, 'Recording contains external map/pose pixels, not complete renderer inputs.\n')


if __name__ == '__main__':
    main()
