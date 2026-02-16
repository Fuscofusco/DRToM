#!/usr/bin/env python3
"""
Simple LHE parser to print the four outgoing particles for a chosen event.

Usage: python3 print_outgoing.py /path/to/file.lhe [event_index]
event_index is 1-based and defaults to 1 (first event).
"""
import sys
from typing import List, Dict


def parse_event_particles(lines_iter):
    """Stream until header and particle lines for one event are captured.
    Returns (nup, particles_lines) or (None, None) at EOF.
    """
    header = None
    particles: List[str] = []
    for raw in lines_iter:
        if '<event>' in raw:
            # start collecting
            header = None
            particles = []
            continue
        if '</event>' in raw:
            if header is None:
                return None, None
            return header, particles
        if header is None:
            if raw.strip() == '':
                continue
            header = raw.strip()
        else:
            if raw.strip() == '':
                continue
            particles.append(raw.strip())
    return None, None


def parse_particle_line(cols: List[str]) -> Dict:
    # LHE particle line columns (common): id, status, m1, m2, col1, col2, px, py, pz, E, m, ...
    out = {}
    try:
        out['id'] = int(cols[0])
    except Exception:
        out['id'] = cols[0]
    try:
        out['status'] = int(cols[1])
    except Exception:
        out['status'] = cols[1]
    try:
        out['mother1'] = int(cols[2])
        out['mother2'] = int(cols[3])
    except Exception:
        out['mother1'] = cols[2] if len(cols) > 2 else None
        out['mother2'] = cols[3] if len(cols) > 3 else None
    # kinematics
    def safe_float(i):
        try:
            return float(cols[i])
        except Exception:
            return None
    out['px'] = safe_float(6)
    out['py'] = safe_float(7)
    out['pz'] = safe_float(8)
    out['E'] = safe_float(9)
    out['m'] = safe_float(10)
    out['raw'] = ' '.join(cols)
    return out


def format_outgoing_from_event(header: str, particle_lines: List[str]) -> List[str]:
    # Return a list of formatted lines for this event (no printing).
    tokens = header.split()
    try:
        nup = int(tokens[0])
    except Exception:
        nup = len(particle_lines)

    particles = []
    for i in range(min(nup, len(particle_lines))):
        cols = particle_lines[i].split()
        particles.append(parse_particle_line(cols))

    # Assumption from user: first two are incoming, next four are outgoing
    if len(particles) >= 6:
        outgoing = particles[2:6]
    else:
        outgoing = [p for idx, p in enumerate(particles) if idx >= 2 and (isinstance(p.get('status'), int) and p.get('status') > 0)]

    lines: List[str] = []
    if not outgoing:
        lines.append(f"Event nup={nup}: No outgoing particles found using heuristics.")
        return lines

    lines.append(f"Event nup={nup}, outgoing_count={len(outgoing)}")
    for i, p in enumerate(outgoing[:4], start=1):
        lines.append(
            f"Outgoing #{i} : id={p.get('id')} px={p.get('px')} py={p.get('py')} pz={p.get('pz')} E={p.get('E')} m={p.get('m')}")
        lines.append(f"  raw: {p.get('raw')}")
    lines.append("")
    return lines


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 print_outgoing.py /path/to/file.lhe [output.txt]")
        sys.exit(1)
    path = sys.argv[1]
    outpath = sys.argv[2] if len(sys.argv) > 2 else 'outgoing_particles.txt'

    current = 0
    written_events = 0
    try:
        with open(path, 'r') as fh, open(outpath, 'w') as outf:
            while True:
                header, particle_lines = parse_event_particles(fh)
                if header is None:
                    break
                current += 1
                lines = format_outgoing_from_event(header, particle_lines)
                # prepend event index to each event block
                outf.write(f"# EventIndex {current}\n")
                for L in lines:
                    outf.write(L + "\n")
                written_events += 1
    except FileNotFoundError:
        print(f"Input file not found: {path}")
        sys.exit(2)
    except Exception as e:
        print(f"Error while processing: {e}")
        sys.exit(3)

    # Minimal terminal feedback
    print(f"Wrote {written_events} events to {outpath}")


if __name__ == '__main__':
    main()
