"""Final fusion extracted verbatim from notebook 40B; inference stays in the notebook."""

def fuse_anchor_s6(anchor_frame,s6_frame,ids,labels,failed):
    import numpy as np
    import pandas as pd
    for name,frame in [('anchor33',anchor_frame),('S6',s6_frame)]:
        if list(frame.columns)!=['StudyInstanceUID']+labels:
            raise RuntimeError(name+' column identity mismatch')
        if not frame.StudyInstanceUID.is_unique or set(frame.StudyInstanceUID)!=set(ids):
            raise RuntimeError(name+' UID coverage mismatch')
    a=anchor_frame.set_index('StudyInstanceUID').loc[ids,labels].to_numpy(float)
    b=s6_frame.set_index('StudyInstanceUID').loc[ids,labels].to_numpy(float)
    if set(failed)-set(ids):raise RuntimeError('Unknown S6 failed UID')
    ok=np.array([u not in failed for u in ids])
    if not ok.any():raise RuntimeError('No S6 study scored')
    if not np.isfinite(a).all() or (a<0).any() or (a>1).any():raise RuntimeError('Invalid anchor33 output')
    if not np.isfinite(b[ok]).all() or (b[ok]<0).any() or (b[ok]>1).any():raise RuntimeError('Invalid scored S6 output')
    if np.isfinite(b[~ok]).any():raise RuntimeError('S6 failed rows unexpectedly contain predictions')
    # Preserve anchor rank blend; intentionally use raw S6 confidence for this probe only.
    out=a.copy()
    ranks=b[ok]  # hypothesis: preserve S6 confidence; calibration mismatch is a risk
    out[ok]=0.90*a[ok]+0.10*ranks
    frame=pd.DataFrame(out.astype(np.float32),columns=labels)
    frame.insert(0,'StudyInstanceUID',ids)
    return frame
