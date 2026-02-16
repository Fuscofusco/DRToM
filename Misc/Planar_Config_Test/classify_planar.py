#!/usr/bin/env python3
"""
Classify events in outgoing_particles.txt into planar types:
  - "planar" (simple)
  - "cross"
  - "star"
  - "misc"

Usage: python3 classify_planar.py outgoing_particles.txt [output_prefix]

Produces: <output_prefix>_tally.txt and <output_prefix>_labels.txt
"""
import sys
from typing import List, Dict, Tuple
import math


def parse_outgoing_file(path: str) -> Dict[int, List[Dict[str, float]]]:
    events = {}
    with open(path, 'r') as fh:
        current_idx = None
        block_lines: List[str] = []
        for raw in fh:
            line = raw.rstrip('\n')
            if line.startswith('# EventIndex'):
                if current_idx is not None and block_lines:
                    ev = _parse_event_block(block_lines)
                    if ev:
                        events[current_idx] = ev
                parts = line.split()
                try:
                    current_idx = int(parts[-1])
                except Exception:
                    current_idx = None
                block_lines = []
                continue
            if current_idx is None:
                continue
            block_lines.append(line)
        # last
        if current_idx is not None and block_lines:
            ev = _parse_event_block(block_lines)
            if ev:
                events[current_idx] = ev
    return events


def _parse_event_block(lines: List[str]) -> List[Dict[str, float]]:
    # Expect lines like: "Event nup=6, outgoing_count=4" then pairs of "Outgoing #i : id=... px=... py=... pz=... E=... m=..."
    particles: List[Dict[str, float]] = []
    for L in lines:
        L = L.strip()
        if L.startswith('Outgoing'):
            # parse key=val tokens
            # example: Outgoing #1 : id=1 px=-568.4149 py=86.33391 pz=-2803.186 E=2861.538 m=0.0
            try:
                tokens = L.split(':', 1)[1].split()
            except Exception:
                continue
            d = {}
            for tok in tokens:
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    try:
                        d[k] = float(v)
                    except Exception:
                        try:
                            d[k] = int(v)
                        except Exception:
                            d[k] = v
            # ensure keys exist
            for k in ('px', 'py', 'pz', 'E'):
                d.setdefault(k, 0.0)
            particles.append(d)
    return particles


def approx_equal(a: float, b: float, rel: float = 0.05, abs_tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(rel * max(abs(a), abs(b), 1.0), abs_tol)


def is_simple(particles: List[Dict[str, float]]) -> bool:
    # Look for two x-like and two y-like, opposite-sign pairs, energies ~ |momentum|
    if len(particles) < 4:
        return False
    # compute pz small
    for p in particles:
        if abs(p.get('pz', 0.0)) > 1e-1 * max(1.0, abs(p.get('E', 0.0))):
            return False

    # classify each as x-like or y-like
    x_like = []
    y_like = []
    for p in particles:
        px = p.get('px', 0.0)
        py = p.get('py', 0.0)
        if abs(px) > 2 * abs(py):
            x_like.append(p)
        elif abs(py) > 2 * abs(px):
            y_like.append(p)
        else:
            return False

    if not (len(x_like) == 2 and len(y_like) == 2):
        return False

    # Check oppositeness and magnitude ~ energy
    if not (approx_equal(abs(x_like[0].get('px', 0.0)), abs(x_like[1].get('px', 0.0))) and
            x_like[0].get('px', 0.0) * x_like[1].get('px', 0.0) < 0):
        return False
    if not (approx_equal(abs(y_like[0].get('py', 0.0)), abs(y_like[1].get('py', 0.0))) and
            y_like[0].get('py', 0.0) * y_like[1].get('py', 0.0) < 0):
        return False

    # energies ~ |px| or |py|
    for p in x_like:
        if not approx_equal(abs(p.get('px', 0.0)), abs(p.get('E', 0.0))):
            return False
    for p in y_like:
        if not approx_equal(abs(p.get('py', 0.0)), abs(p.get('E', 0.0))):
            return False

    return True


def is_cross(particles: List[Dict[str, float]]) -> bool:
    # Two opposite x-axis particles equal energy; other two are opposite vectors equal energy
    if len(particles) < 4:
        return False
    # small pz
    for p in particles:
        if abs(p.get('pz', 0.0)) > 1e-1 * max(1.0, abs(p.get('E', 0.0))):
            return False

    # find any pair that are x-axis opposites (py approx 0, px opposite and |px|~E)
    x_pairs = []
    others = []
    used = set()
    for i, p in enumerate(particles):
        if abs(p.get('py', 0.0)) < 0.2 * max(1.0, abs(p.get('px', 0.0))):
            # candidate x-like
            for j, q in enumerate(particles):
                if i >= j: continue
                if abs(q.get('py', 0.0)) < 0.2 * max(1.0, abs(q.get('px', 0.0))) and p.get('px', 0.0) * q.get('px', 0.0) < 0 and approx_equal(abs(p.get('px',0.0)), abs(q.get('px',0.0))):
                    x_pairs.append((i,j))
    if not x_pairs:
        return False
    # pick first x_pair and ensure remaining two are opposite vectors with equal magnitude and opposite sign
    i,j = x_pairs[0]
    rem = [k for k in range(len(particles)) if k not in (i,j)]
    if len(rem) != 2:
        return False
    a = particles[rem[0]]
    b = particles[rem[1]]
    # check they are opposite vectors
    if not (approx_equal(a.get('E',0.0), b.get('E',0.0)) and
            approx_equal(a.get('px',0.0), -b.get('px',0.0)) and
            approx_equal(a.get('py',0.0), -b.get('py',0.0))):
        return False

    # energies consistent with sqrt(px^2+py^2)
    for p in (a,b):
        ptrans = math.hypot(p.get('px',0.0), p.get('py',0.0))
        if not approx_equal(ptrans, p.get('E',0.0)):
            return False

    return True


def is_star(particles: List[Dict[str, float]]) -> bool:
    # One opposite x pair, two particles with px ~ py in magnitude and opposite py signs
    if len(particles) < 4:
        return False
    for p in particles:
        if abs(p.get('pz', 0.0)) > 1e-1 * max(1.0, abs(p.get('E', 0.0))):
            return False

    # find x-like opposite pair (py small, px opposite)
    x_pair = None
    for i, p in enumerate(particles):
        for j, q in enumerate(particles):
            if i >= j: continue
            if abs(p.get('py',0.0)) < 0.2 * max(1.0, abs(p.get('px',0.0))) and abs(q.get('py',0.0)) < 0.2 * max(1.0, abs(q.get('px',0.0))) and p.get('px',0.0) * q.get('px',0.0) < 0 and approx_equal(abs(p.get('px',0.0)), abs(q.get('px',0.0))):
                x_pair = (i,j)
                break
        if x_pair:
            break
    if x_pair is None:
        return False

    rem = [k for k in range(len(particles)) if k not in x_pair]
    if len(rem) != 2:
        return False
    p2 = particles[rem[0]]
    p4 = particles[rem[1]]
    # check px ~ py in magnitude for both
    if not (approx_equal(abs(p2.get('px',0.0)), abs(p2.get('py',0.0))) and approx_equal(abs(p4.get('px',0.0)), abs(p4.get('py',0.0)))):
        return False
    # check py signs opposite
    if not (p2.get('py',0.0) * p4.get('py',0.0) < 0):
        return False

    # energy relation E2 = sqrt(2) * p2y
    if not approx_equal(p2.get('E',0.0), math.sqrt(2.0) * abs(p2.get('py',0.0))):
        return False
    if not approx_equal(p4.get('E',0.0), math.sqrt(2.0) * abs(p4.get('py',0.0))):
        return False

    return True


def classify_all(events: Dict[int, List[Dict[str, float]]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    tally = {'planar': 0, 'cross': 0, 'star': 0, 'misc': 0}
    labels: Dict[int, str] = {}
    for idx, parts in events.items():
        label = 'misc'
        if is_simple(parts):
            label = 'planar'
        elif is_cross(parts):
            label = 'cross'
        elif is_star(parts):
            label = 'star'
        tally[label] += 1
        labels[idx] = label
    return tally, labels


def compute_pt_labels_and_angles(events: Dict[int, List[Dict[str, float]]]) -> Dict[int, Tuple[float, float, float]]:
    """
    For each event compute pT for each particle, label by pT (leading, subleading, third, fourth)
    and return three angles (degrees) between leading and subleading, leading and third,
    leading and fourth. If fewer than 4 particles, missing angles are written as NaN.
    """
    out = {}
    for idx, parts in events.items():
        # compute pT and azimuth for each particle
        pts = []
        for p in parts:
            px = p.get('px', 0.0)
            py = p.get('py', 0.0)
            pt = math.hypot(px, py)
            phi = math.atan2(py, px)
            pts.append({'pt': pt, 'phi': phi})

        # sort by pt descending
        pts_sorted = sorted(pts, key=lambda x: x['pt'], reverse=True)



        if len(pts_sorted) >= 2:
            lead = pts_sorted[0]
            sub = pts_sorted[1] if len(pts_sorted) >= 2 else None
            third = pts_sorted[2] if len(pts_sorted) >= 3 else None
            fourth = pts_sorted[3] if len(pts_sorted) >= 4 else None
            a1 = angdiff_deg_ccw_from_lead(lead['phi'], sub['phi']) if sub is not None else float('nan')
            a2 = angdiff_deg_ccw_from_lead(lead['phi'], third['phi']) if third is not None else float('nan')
            a3 = angdiff_deg_ccw_from_lead(lead['phi'], fourth['phi']) if fourth is not None else float('nan')
        else:
            a1 = a2 = a3 = float('nan')

        out[idx] = (a1, a2, a3)

    return out


# helper to compute minimal angle difference in degrees
def angdiff_deg_ccw_from_lead(lead_phi, other_phi):
    """Return CCW delta (degrees) from lead_phi to other_phi in [0,360)."""
    if lead_phi is None or other_phi is None:
        return float('nan')
    d = other_phi - lead_phi
    # convert to degrees
    ddeg = d * 180.0 / math.pi
    # normalize to [0,360)
    while ddeg < 0.0:
        ddeg += 360.0
    while ddeg >= 360.0:
        ddeg -= 360.0
    return ddeg



def compute_orientations_and_summary(events: Dict[int, List[Dict[str, float]]], prefix: str, tol_deg: float = 10.0):
    """
    Produce per-event orientation info and an overall summary describing common orientations.

    Writes two files:
      - <prefix>_orientations.txt : per-event lines with event index, then for each particle (pt, phi_deg),
        and hemisphere counts relative to leading.
      - <prefix>_orientation_summary.txt : aggregated counts describing common patterns.

    The summary includes counts for: mostly collinear with lead, back-to-back, ~90deg-separated (planar),
    and hemisphere patterns (how many others share leading's half-plane).
    """
    orientations_path = f'{prefix}_orientations.txt'
    summary_path = f'{prefix}_orientation_summary.txt'

    # accumulators
    total_events = 0
    counts = {
        'lead_collinear_others_all_within_tol': 0,
        'lead_back_to_back_others_all_within_tol': 0,
        'planar_approx_90': 0,
        'hemisphere_3_same_as_lead': 0,
        'hemisphere_2_same_as_lead': 0,
        'hemisphere_1_same_as_lead': 0,
        'hemisphere_0_same_as_lead': 0,
    }

    def wrap_deg(phi):
        # convert rad->deg in [0,360)
        d = phi * 180.0 / math.pi
        if d < 0:
            d += 360.0
        return d

    def angdiff_deg_signed(a, b):
        d = a - b
        while d <= -180.0:
            d += 360.0
        while d > 180.0:
            d -= 360.0
        return d

    with open(orientations_path, 'w') as outf:
        for idx, parts in events.items():
            total_events += 1
            items = []
            for p in parts:
                px = p.get('px', 0.0)
                py = p.get('py', 0.0)
                pt = math.hypot(px, py)
                phi = math.atan2(py, px)
                items.append({'pt': pt, 'phi': phi})

            if not items:
                continue
            items_sorted = sorted(items, key=lambda x: x['pt'], reverse=True)
            # list of (pt, phi_deg) for writing
            s_list = [f"({it['pt']:.4f},{wrap_deg(it['phi']):.2f})" for it in items_sorted]

            lead = items_sorted[0]
            # count others in same half-plane as lead (|Δφ| < 90)
            same_half = 0
            for other in items_sorted[1:]:
                d_signed = abs(angdiff_deg_signed(wrap_deg(other['phi']), wrap_deg(lead['phi'])))
                if d_signed < 90.0:
                    same_half += 1

            # hemisphere tally
            if same_half == 3:
                counts['hemisphere_3_same_as_lead'] += 1
            elif same_half == 2:
                counts['hemisphere_2_same_as_lead'] += 1
            elif same_half == 1:
                counts['hemisphere_1_same_as_lead'] += 1
            elif same_half == 0:
                counts['hemisphere_0_same_as_lead'] += 1

            # check if all others nearly collinear with lead (within tol_deg)
            all_collinear = True
            all_back = True
            phis_deg = [wrap_deg(it['phi']) for it in items_sorted]
            lead_deg = phis_deg[0]
            for ddeg in phis_deg[1:]:
                diff = abs(angdiff_deg_signed(ddeg, lead_deg))
                if diff > tol_deg:
                    all_collinear = False
                if abs(abs(diff) - 180.0) > tol_deg:
                    all_back = False

            if all_collinear:
                counts['lead_collinear_others_all_within_tol'] += 1
            if all_back:
                counts['lead_back_to_back_others_all_within_tol'] += 1

            # check approx planar 90 deg: need at least 4 particles and the 4 sorted by phi have gaps ~90
            if len(phis_deg) >= 4:
                ph_sorted = sorted(phis_deg[:4])
                gaps = [(ph_sorted[(i+1)%4] - ph_sorted[i]) % 360.0 for i in range(4)]
                # normalize gaps to [-180,180]
                gaps = [g if g <= 180.0 else 360.0 - g for g in gaps]
                ok = all(abs(g - 90.0) <= tol_deg for g in gaps)
                if ok:
                    counts['planar_approx_90'] += 1

            outf.write(f"{idx} ")
            outf.write(' '.join(s_list))
            outf.write(f" same_half={same_half}\n")

    # write summary
    with open(summary_path, 'w') as sf:
        sf.write(f"total_events {total_events}\n")
        for k, v in counts.items():
            sf.write(f"{k} {v}\n")

    return orientations_path, summary_path


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 classify_planar.py outgoing_particles.txt [output_prefix]')
        sys.exit(1)
    path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else 'planar'
    events = parse_outgoing_file(path)
    tally, labels = classify_all(events)
    tally_path = f'{prefix}_tally.txt'
    labels_path = f'{prefix}_labels.txt'
    with open(tally_path, 'w') as tf:
        for k, v in tally.items():
            tf.write(f'{k}: {v}\n')
    with open(labels_path, 'w') as lf:
        for idx in sorted(labels.keys()):
            lf.write(f'{idx} {labels[idx]}\n')

    # compute pT labels and angles and write per-event three-angle output
    angles = compute_pt_labels_and_angles(events)
    def wrap_deg(phi):
        d = phi * 180.0 / math.pi
        if d < 0:
            d += 360.0
        return d

    def write_readable_angles(events_dict, angles_dict, out_prefix):
        path = f'{out_prefix}_angles.txt'
        with open(path, 'w') as af:
            for idx in sorted(events_dict.keys()):
                parts = events_dict[idx]
                # compute pT and phi for each parton
                items = []
                for i, p in enumerate(parts, start=1):
                    px = p.get('px', 0.0)
                    py = p.get('py', 0.0)
                    pt = math.hypot(px, py)
                    phi = math.atan2(py, px)
                    items.append({'index': i, 'pt': pt, 'phi': phi})
                items_sorted = sorted(items, key=lambda x: x['pt'], reverse=True)

                af.write(f"Event {idx}:\n")
                # write pT magnitudes
                pt_strs = [f"p{j+1}= {it['pt']:.4f}" for j, it in enumerate(items_sorted)]
                af.write('  Partons (by pT): ' + ', '.join(pt_strs) + '\n')

                # angles
                a1, a2, a3 = angles_dict.get(idx, (float('nan'), float('nan'), float('nan')))
                af.write(f"  lj->sl: {a1:.3f} deg, lj->3rd: {a2:.3f} deg, lj->4th: {a3:.3f} deg\n")
                # also write lead phi for reference
                if items_sorted:
                    af.write(f"  lead_phi_deg: {wrap_deg(items_sorted[0]['phi']):.2f}\n")
                af.write('\n')
        return path

    angles_path = write_readable_angles(events, angles, prefix)

    # compute orientations and summary
    orient_path, summary_path = compute_orientations_and_summary(events, prefix)
    print(f'Wrote angles -> {angles_path}, orientations -> {orient_path}, summary -> {summary_path}')

    print(f'Wrote tally -> {tally_path} and labels -> {labels_path} (events processed: {len(events)})')


if __name__ == '__main__':
    main()

# python3 DRToM/Misc/Planar_Config_Test/classify_planar.py outgoing_particles.txt [output_prefix]
