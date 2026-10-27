"""Full-song metrics used for the frozen v0.7-weight benchmark.

Separate from the activity-window development metric in ``evaluation``.
Inputs must already be aligned; no gain fitting, resampling or delay search.
"""
import numpy as np
SOURCES=("drums", "bass", "vocals", "other")

def score(reference,estimate, *, compute_museval=True):
    """Literal museval BSS Eval v4 and MDX/Demucs whole-song SDR, float64."""
    if reference.shape!=estimate.shape or reference.ndim!=3 or reference.shape[0]!=4 or reference.shape[2]!=2:
        raise ValueError('Require matching [4, samples, 2] arrays')
    if reference.shape[1] == 0 or not np.isrealobj(reference) or not np.isrealobj(estimate):
        raise ValueError("Require nonempty real audio")
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():raise ValueError('Nonfinite audio')
    ref=np.asarray(reference,dtype=np.float64);est=np.asarray(estimate,dtype=np.float64)
    # Float64 accumulation, with the published MDX delta in both energies.
    energy=np.einsum('stc,stc->s',ref,ref)
    error=ref-est
    error_energy=np.einsum('stc,stc->s',error,error)
    del error
    global_sdr=10*np.log10((energy+1e-7)/(error_energy+1e-7))
    result = {"global_sdr_db": dict(zip(SOURCES, global_sdr.tolist())),
              "reference_energy": dict(zip(SOURCES, energy.tolist())),
              "error_energy": dict(zip(SOURCES, error_energy.tolist()))}
    if not compute_museval:
        return result
    import museval
    # Preserve literal library behavior for wholly silent sources, without dropping a song.
    undefined=None
    try:
        metrics=museval.metrics.bss_eval(ref,est,window=44100,hop=44100,compute_permutation=False,
            filters_len=512,framewise_filters=False,bsseval_sources_version=False)
    except ValueError as exc:
        if 'all 0s' not in str(exc):raise
        undefined=str(exc)
        count=museval.metrics.Framing(44100,44100,ref.shape[1]).nwin
        metrics=[np.full((4,count),np.nan) for _ in range(4)]
    return {'global_sdr_db':dict(zip(SOURCES,global_sdr.tolist())),
        'reference_energy':dict(zip(SOURCES,energy.tolist())),
        'error_energy':dict(zip(SOURCES,error_energy.tolist())),
        'museval_undefined_reason':undefined,
        'museval':{key:{s:values[i].tolist() for i,s in enumerate(SOURCES)}
                   for key,values in zip(('SDR','ISR','SIR','SAR'),metrics[:4])}}


def json_metrics(value):
    """Preserve undefined frames as null and signed infinities as strings."""
    if isinstance(value, dict):
        return {key: json_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_metrics(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def aggregate(records):
    """Mean songs for whole-song SDR; median frames then songs for museval."""
    if not records:
        raise ValueError('Require at least one complete song')
    if len({row['track'] for row in records}) != len(records):
        raise ValueError('Duplicate song identity')
    result = {'songs': len(records), 'sources': {}}
    has_museval = ['museval' in row['scores'] for row in records]
    if any(has_museval) != all(has_museval):
        raise ValueError('Cannot mix metric protocols')
    for source in SOURCES:
        whole = [row['scores']['global_sdr_db'][source] for row in records]
        if not np.isfinite(whole).all():
            raise ValueError('Whole-song scores must be finite')
        entry = {'global_mean_db': float(np.mean(whole))}
        if all(has_museval):
            medians = []
            total = defined = 0
            for row in records:
                values = np.asarray([np.nan if value is None else float(value)
                                     for value in row['scores']['museval']['SDR'][source]])
                valid = values[~np.isnan(values)]
                total += len(values)
                defined += len(valid)
                medians.append(float(np.median(valid)) if len(valid) else np.nan)
            valid = np.asarray(medians)[~np.isnan(medians)]
            entry.update(museval_median_db=float(np.median(valid)) if len(valid) else None,
                         museval_defined_songs=len(valid), museval_total_songs=len(records),
                         museval_defined_frames=defined, museval_total_frames=total)
        result['sources'][source] = entry
    result['overall'] = {'global_mean_db': float(np.mean(
        [result['sources'][source]['global_mean_db'] for source in SOURCES]))}
    if all(has_museval):
        medians = [result['sources'][source]['museval_median_db'] for source in SOURCES]
        result['overall']['museval_median_db'] = (
            None if any(v is None for v in medians) else float(np.mean(medians)))
    return json_metrics(result)
