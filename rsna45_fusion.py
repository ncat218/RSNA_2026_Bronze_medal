"""CPU-only, predeclared final fusion experiments. Embedded verbatim in notebooks.

No labels, reports, training, image transforms, or hyperparameter search here.
Predictions with incomplete model coverage retain the exact successful40B output.
"""
import numpy as np
import pandas as pd

WEIGHTS_33 = {'Medial OA': .5, 'Lateral OA': .2, 'PF OA': .6,
              'Synovitis': .8, "Baker's": .35}
LOGIT_EPSILON = 1e-5


def diagnostic_json_safe(value):
    """Normalize diagnostic NaN to JSON null; NEVER call this on predictions."""
    if isinstance(value, dict):
        return {str(k): diagnostic_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return diagnostic_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return diagnostic_json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def aligned_prediction(frame, ids, labels, name, valid=None):
    if list(frame.columns) != ['StudyInstanceUID'] + labels:
        raise RuntimeError(name + ' column identity mismatch')
    if not frame.StudyInstanceUID.is_unique or set(frame.StudyInstanceUID) != set(ids):
        raise RuntimeError(name + ' UID coverage mismatch')
    values = frame.set_index('StudyInstanceUID').loc[ids, labels].to_numpy(float)
    subset = values if valid is None else values[valid]
    if not np.isfinite(subset).all() or (subset < 0).any() or (subset > 1).any():
        raise RuntimeError(name + ' invalid scored probabilities')
    return values


def percentile_on(values, valid):
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if valid.any():
        result[valid] = pd.DataFrame(values[valid]).rank(method='average', pct=True).to_numpy()
    return result


def final_fusion_probe(tag, anchor, s6, raw384, raw224, raw6, ids, labels,
                       failed_s6, coverage, receipt, baseline, expected_prefix_sha):
    if tag not in ('45A', '45B', '45C', '45D', '45E'):
        raise RuntimeError('Unknown predeclared experiment')
    if not ids or len(set(ids)) != len(ids):
        raise RuntimeError('Invalid canonical test IDs')
    if coverage.get('schema') != '45_coverage_v1':
        raise RuntimeError('Missing coverage schema')
    if coverage.get('original_anchor_script_sha256') != expected_prefix_sha:
        raise RuntimeError('Coverage adapter source identity mismatch')
    if coverage.get('uids') != ids or coverage.get('labels') != labels:
        raise RuntimeError('Coverage UID/order/label mismatch')
    if coverage.get('a224') != .35 or coverage.get('default_raptor_weight') != .5:
        raise RuntimeError('Unexpected anchor weights')
    if coverage.get('raptor_per_class') != WEIGHTS_33:
        raise RuntimeError('Unexpected anchor class overrides')
    masks = []
    for key, receipt_key in [('failed384', 'raptor384_failed'),
                             ('failed224', 'raptor224_failed'), ('failed6', 'ours_failed')]:
        seq = coverage.get(key)
        if not isinstance(seq, list) or len(seq) != len(ids) or any(type(x) is not bool for x in seq):
            raise RuntimeError('Invalid boolean coverage: ' + key)
        mask = np.array(seq, dtype=bool)
        if receipt.get(receipt_key) != int(mask.sum()):
            raise RuntimeError('Coverage/anchor receipt disagreement: ' + key)
        masks.append(mask)
    f384, f224, f6 = masks
    if f384.sum() > .03 * len(ids) or f224.sum() > .03 * len(ids) or f6.sum() > .5 * len(ids):
        raise RuntimeError('Inherited anchor failure limit exceeded')
    if not isinstance(failed_s6, dict) or set(failed_s6) - set(ids):
        raise RuntimeError('Invalid S6 failure coverage')
    fs = np.array([u in failed_s6 for u in ids])
    a = aligned_prediction(anchor, ids, labels, 'anchor33')
    b = aligned_prediction(s6, ids, labels, 'S6', ~fs)
    if np.isfinite(b[fs]).any():
        raise RuntimeError('Failed S6 row unexpectedly has predictions')
    p384 = aligned_prediction(raw384, ids, labels, 'Raptor384')
    p224 = aligned_prediction(raw224, ids, labels, 'Raptor224')
    p6 = aligned_prediction(raw6, ids, labels, 'R6')
    if not np.array_equal((p384 == .5).all(axis=1), f384):
        raise RuntimeError('Raptor384 sentinel/coverage mismatch')
    if not np.array_equal((p224 == .5).all(axis=1), f224):
        raise RuntimeError('Raptor224 sentinel/coverage mismatch')
    base = aligned_prediction(baseline, ids, labels, 'baseline40B')
    expected_base = a.copy()
    expected_base[~fs] = .9 * a[~fs] + .1 * b[~fs]
    np.testing.assert_allclose(base, expected_base.astype(np.float32), rtol=0, atol=1e-7,
                               err_msg='Not exact successful40B baseline')
    w = np.array([WEIGHTS_33.get(c, .5) for c in labels], dtype=float)
    # Keep each model's OWN valid rank population, not the complete-case intersection.
    r384, r224, r6 = (percentile_on(p, ~f) for p, f in zip((p384, p224, p6), masks))
    rr = np.where(f384[:, None], r224,
                  np.where(f224[:, None], r384, .65 * r384 + .35 * r224))
    reproduced = w * rr + (1 - w) * r6
    reproduced[f384 & f224] = r6[f384 & f224]
    reproduced[f6] = rr[f6]
    reproduced[f384 & f224 & f6] = .5
    anchor_delta = float(np.max(np.abs(reproduced.astype(np.float32) - a)))
    if anchor_delta > 2e-6:
        raise RuntimeError('Raw outputs do not reconstruct exact33: ' + str(anchor_delta))
    eligible = ~(f384 | f224 | f6 | fs)
    if not eligible.any():
        raise RuntimeError('No fully scored row available for the declared experiment')
    # Only compute candidate rows on complete coverage; never average imputed/NaN rows.
    e = eligible
    if tag == '45A':
        candidate = .9 * percentile_on(a, np.ones(len(ids), dtype=bool))[e] + .1 * b[e]
    elif tag == '45B':
        candidate = .9 * (w * (.65 * p384[e] + .35 * p224[e]) + (1 - w) * p6[e]) + .1 * b[e]
    elif tag == '45C':
        def logit(p):
            q = np.clip(p, LOGIT_EPSILON, 1 - LOGIT_EPSILON)
            return np.log(q) - np.log1p(-q)
        z = .9 * (w * (.65 * logit(p384[e]) + .35 * logit(p224[e])) +
                  (1 - w) * logit(p6[e])) + .1 * logit(b[e])
        candidate = 1 / (1 + np.exp(-z))
    elif tag == '45D':
        candidate = .9 * (.5 * (.65 * r384[e] + .35 * r224[e]) + .5 * r6[e]) + .1 * b[e]
    else:
        candidate = .9 * (w * r384[e] + (1 - w) * r6[e]) + .1 * b[e]
    if not np.isfinite(candidate).all() or (candidate < 0).any() or (candidate > 1).any():
        raise RuntimeError('Invalid experiment predictions')
    result = base.copy()
    result[e] = candidate
    frame = pd.DataFrame(result.astype(np.float32), columns=labels)
    frame.insert(0, 'StudyInstanceUID', ids)
    delta = result - base
    audit = {'experiment': tag, 'rows': len(ids), 'candidate_rows': int(e.sum()),
             'baseline40B_fallback_rows': int((~e).sum()),
             'baseline40B_fallback_uids': [u for u, ok in zip(ids, e) if not ok],
             'raw_to_anchor33_max_abs_delta': anchor_delta,
             'max_abs_delta_from40B': float(np.max(np.abs(delta))),
             'mean_abs_delta_from40B': float(np.mean(np.abs(delta))),
             'per_target_mean_abs_delta': dict(zip(labels, np.abs(delta).mean(axis=0).tolist())),
             'logit_epsilon': LOGIT_EPSILON if tag == '45C' else None,
             'rank_population': 'each component own scored UIDs;45A anchor rank uses all canonical UIDs',
             'independent_validation': False, 'new_model_forwards': 0,
             'incomplete_coverage_policy': 'exact successful40B prediction for that row'}
    return frame, audit
