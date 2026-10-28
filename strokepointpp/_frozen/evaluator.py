"""Original macro12 candidate recall and sampled-point/native-presence metrics.

No import-time reads. Native candidate targets and full-mask presence are
evaluator-only arguments, not action-policy features. No training implementation
may substitute histogram credit, AUROC or sampled label absence for these.
"""
import numpy as np

LABELS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15])
MACRO = LABELS[:12]


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def vote(values, anchor):
    values = np.asarray(values)
    require(np.isin(values, np.r_[0, LABELS]).all() and anchor in LABELS, 'Invalid vote labels')
    if not len(values) or not np.any(values > 0):
        return int(anchor)
    counts = np.array([(values == c).sum() for c in LABELS])
    tied = LABELS[counts == counts.max()]
    return int(anchor if anchor in tied else tied.min())


def identity_distribution(records, patients_by_fold, weights, modalities=('CT', 'MR')):
    """Same full-candidate denominator as archived LV/SA, including empty cases.

    Each record contains patient,fold,mod,case_ids,truth,prediction for ONE arm.
    Patient multiplicities are shared across modalities, never separate scan n.
    A missing required class stays NaN through the macro; no nanmean/zero-fill.
    """
    require(len(modalities) > 0 and len(set(modalities)) == len(modalities), 'Modality schema')
    lookup = {}
    for r in records:
        key = (str(r['fold']), str(r['patient']), r['mod'])
        require(key not in lookup, 'Duplicate patient/modality record')
        require(len(r['case_ids']) == len(set(r['case_ids'])) == len(r['truth']) == len(r['prediction']),
                'Full candidate row alignment')
        require(np.isin(r['truth'], LABELS).all() and np.isin(r['prediction'], np.r_[0, LABELS]).all(),
                'Unknown candidate class')
        lookup[key] = r
    expected_keys = {(str(f), str(p), m) for f, ids in patients_by_fold.items() for p in ids for m in modalities}
    require(set(lookup) == expected_keys, 'Population changed, missing or extra patient/modality')
    units, supports = [], []
    n_draws = None
    for fold, ids in patients_by_fold.items():
        require(len(ids) == len(set(ids)), 'Patient list duplicates')
        w = np.asarray(weights[str(fold)], float)
        require(w.ndim == 2 and w.shape[1] == len(ids) and np.isfinite(w).all() and (w >= 0).all(), 'Invalid cluster weights')
        require((w.sum(1) > 0).all() and np.all(w[0] == 1), 'First row must be original patient population')
        require(n_draws is None or n_draws == len(w), 'Unpaired bootstrap count')
        n_draws = len(w)
        for mod in modalities:
            numerator, denominator = [], []
            for pid in ids:
                r = lookup[(str(fold), str(pid), mod)]
                y, p = np.asarray(r['truth']), np.asarray(r['prediction'])
                numerator.append([int(((y == c) & (p == c)).sum()) for c in MACRO])
                denominator.append([int((y == c).sum()) for c in MACRO])
            nn, dd = w @ np.asarray(numerator), w @ np.asarray(denominator)
            recalls = np.divide(nn, dd, out=np.full(dd.shape, np.nan), where=dd > 0)
            units.append(recalls.mean(1))
            supports.append({'fold': str(fold), 'mod': mod, 'full_native_candidate_counts': dd[0].astype(int).tolist(),
                             'undefined_resamples_by_class': (dd[1:] == 0).sum(0).tolist()})
    return np.mean(units, axis=0), supports


def summarize_distribution(values, alpha=.05):
    """Report conditional quantiles honestly; any invalid draw makes CI unresolved."""
    v = np.asarray(values, float)
    require(v.ndim == 1 and len(v) >= 2, 'Observed value plus resamples required')
    valid = np.isfinite(v[1:])
    conditional = np.quantile(v[1:][valid], [alpha / 2, 1 - alpha / 2]).tolist() if valid.any() else None
    complete = bool(np.isfinite(v[0]) and valid.all())
    return {'value': float(v[0]) if np.isfinite(v[0]) else None,
            'CI': conditional if complete else None,
            'conditional_finite_only_CI': conditional,
            'valid_resamples': int(valid.sum()), 'total_resamples': len(v) - 1,
            'status': 'DEFINED' if complete else 'UNDETERMINED_UNDEFINED_CLASS_DENOMINATORS'}


def fit_action_gain(native_target, before, after, full_fit_denominators, stratum_weight):
    """Offline fit target, not an inference input. Denominators use ALL candidates.

    Summing all fixed action contributions equals the macro recall contrast if
    actions affect distinct candidate identities. This is not a new interaction.
    """
    d = np.asarray(full_fit_denominators, float)
    require(d.shape == (12,) and np.isfinite(d).all() and (d > 0).all(), 'Required full-fit class missing')
    require(native_target in LABELS and before in np.r_[0, LABELS] and after in np.r_[0, LABELS], 'Unknown identity')
    require(0 < stratum_weight <= 1, 'Invalid fixed stratum weight')
    if native_target == 15:
        return 0.0
    return stratum_weight * (int(after == native_target) - int(before == native_target)) / (12 * d[int(native_target) - 1])


def point_metrics(truth, prediction, reference, native_present):
    y, p, b = map(np.asarray, (truth, prediction, reference))
    require(y.ndim == 1 and y.shape == p.shape == b.shape and len(y) > 0, 'Point schema mismatch')
    require(np.isin(y, LABELS).all() and np.isin(p, np.r_[0, LABELS]).all() and np.isin(b, np.r_[0, LABELS]).all(), 'Invalid point class')
    require(set(np.unique(y)) <= set(native_present) <= set(LABELS), 'Native-presence contradiction')
    gt = np.array([(y == c).sum() for c in LABELS])
    pr = np.array([(p == c).sum() for c in LABELS])
    tp = np.array([((y == c) & (p == c)).sum() for c in LABELS])
    denominator = gt + pr
    absent = ~np.isin(LABELS, native_present)
    return {'Dice': float((2 * tp[denominator > 0] / denominator[denominator > 0]).mean()),
            'native_absent_FPR': float((pr[absent] > 0).mean()) if absent.any() else None,
            'native_absent_class_count': int(absent.sum()),
            'TP': tp.tolist(), 'GT': gt.tolist(), 'predicted': pr.tolist(),
            'recall': [float(a / n) if n else None for a, n in zip(tp, gt)],
            'new_errors': int(((b == y) & (p != y)).sum()),
            'any_new_error': bool(((b == y) & (p != y)).any()),
            'corrected': int(((b != y) & (p == y)).sum()),
            'unsampled_native_positive': [int(c) for c in native_present if not (y == c).any()]}


def synthetic_checks():
    # Independent arithmetic fixtures, no local patient arrays.
    weights = {'0': np.ones((4, 2)), '1': np.ones((4, 2))}
    groups = {'0': ['SYN:A', 'SYN:B'], '1': ['SYN:C', 'SYN:D']}
    rows = []
    for f, ids in groups.items():
        for pid in ids:
            for mod in ('CT', 'MR'):
                rows.append(dict(fold=int(f), patient=pid, mod=mod, case_ids=list(range(13)),
                                 truth=LABELS.tolist(), prediction=LABELS.tolist()))
    out, support = identity_distribution(rows, groups, weights)
    checks = {'perfect_identity_one': np.array_equal(out, np.ones(4)),
              'all_native_counts_retained': all(r['full_native_candidate_counts'] == [2] * 12 for r in support),
              'empty_vote_anchor': vote([], 8) == 8,
              'tie_anchor': vote([8, 9], 9) == 9,
              'tie_canonical': vote([8, 9], 1) == 8}
    altered = [dict(r) for r in rows]
    altered[0]['prediction'] = list(range(1, 13)) + [1]
    no15, _ = identity_distribution(altered, groups, weights)
    checks['label15_excluded_only_legacy_endpoint'] = np.array_equal(no15, out)
    altered[0]['prediction'] = [2] + LABELS[1:].tolist()
    bad, _ = identity_distribution(altered, groups, weights)
    checks['one_error_exact_denominator'] = abs(out[0] - bad[0] - 1 / 96) < 1e-12
    g = fit_action_gain(1, 2, 1, [2] * 12, .25)
    checks['fit_gain_additive_exact'] = abs(g - 1 / 96) < 1e-12
    pm = point_metrics(np.array([8, 9, 1]), np.array([9, 9, 0]), np.array([8, 1, 1]), [1, 8, 9, 15])
    checks['new_errors_distinct_event'] = pm['new_errors'] == 2 and pm['any_new_error']
    checks['unknown0_is_FN_not_new_native_class'] = pm['Dice'] == 2 / 9
    checks['unsampled_present_not_absent'] = pm['unsampled_native_positive'] == [15]
    checks['invalid_bootstrap_not_hidden'] = summarize_distribution([1, 1, np.nan])['CI'] is None
    checks['conditional_interval_explicit_only'] = summarize_distribution([1, 1, np.nan])['conditional_finite_only_CI'] == [1, 1]
    return {k: bool(v) for k, v in checks.items()}


if __name__ == '__main__':
    import json
    checks = synthetic_checks()
    print(json.dumps({'qualified': all(checks.values()), 'checks': checks}, indent=2))
    assert all(checks.values())
