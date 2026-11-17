# SatellaRhythm-Edge v0.3 — experimental ONNX single-lead ECG inference.
# Not for diagnosis, screening, treatment, triage or health reassurance.
# Requires numpy, scipy, onnxruntime. Usage: from infer_onnx import RhythmEdge
import json
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly
import onnxruntime as ort

class RhythmEdge:
    def __init__(self, folder):
        folder=Path(folder)
        self.cfg=json.loads((folder/'config_research.json').read_text(encoding='utf-8'))
        self.session=ort.InferenceSession(str(folder/'satellarhythm_edge_v03.onnx'),providers=['CPUExecutionProvider'])
        self.fs=int(self.cfg['sampling_rate_hz']);self.win=int(self.cfg['window_samples'])
        self.thr=self.cfg['thresholds']
    def _window(self,x,start):
        y=np.asarray(x[start:start+self.win],dtype=np.float32)
        if y.size<self.win:y=np.pad(y,(0,self.win-y.size),mode='reflect')
        y=y-np.median(y)
        scale=max(float(np.percentile(np.abs(y),90)),1e-6)
        return np.clip(y/scale,-6,6).astype(np.float32)
    @staticmethod
    def _softmax(a):
        a=a-a.max(axis=-1,keepdims=True)
        e=np.exp(a);return e/e.sum(axis=-1,keepdims=True)
    def predict(self,ecg,sample_rate_hz=300,n_windows=5):
        fs=float(sample_rate_hz)
        if fs<=0 or fs>2000:return {'status':'UNSUPPORTED_SAMPLE_RATE','research_only':True}
        try:x=np.asarray(ecg,dtype=np.float32).reshape(-1)
        except Exception:return {'status':'INVALID_INPUT','research_only':True}
        if not np.isfinite(x).all() or len(x)<int(8.5*fs) or len(x)>int(120*fs) or np.std(x)<1e-5:
            return {'status':'INVALID_OR_INSUFFICIENT_SIGNAL','research_only':True}
        if fs!=self.fs:
            rat=Fraction(self.fs,1)/Fraction(str(fs)).limit_denominator(10000)
            x=resample_poly(x,rat.numerator,rat.denominator).astype(np.float32)
        starts=np.linspace(0,max(0,len(x)-self.win),int(n_windows),dtype=int)
        inp=np.stack([self._window(x,int(s)) for s in starts])[:,None,:]
        rl,ql=self.session.run(None,{'ecg':inp})
        rp=self._softmax(rl).mean(0);qs=self._softmax(ql)
        risk=.7*float(qs[:,1].mean())+.3*float(qs[:,1].max())
        label=int(rp.argmax());low_conf=float(rp.max())<self.thr['min_rhythm_confidence']
        if label==0 and float(rp[0])<self.thr['min_normal_confidence']:low_conf=True
        if risk>=self.thr['max_quality_risk'] or low_conf:label=3
        names=self.cfg['rhythm_classes']+['REVIEW_UNINTERPRETABLE']
        return {'status':'EXPERIMENTAL_RESEARCH_ONLY','prediction':names[label],
            'rhythm_probabilities':dict(zip(self.cfg['rhythm_classes'],rp.astype(float).round(4).tolist())),
            'quality_risk':round(risk,4),'not_for_diagnosis_or_reassurance':True}
