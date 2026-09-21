import time
class Metrics:
    def __init__(self): self.rows=[]; self.latencies=[]
    def add(self,row,latency): self.rows.append(row); self.latencies.append(latency)
    def summary(self):
        keys=self.rows[0].keys() if self.rows else []
        out={k:sum(r[k] for r in self.rows)/len(self.rows) for k in keys}; out["inference_latency_ms"]=sum(self.latencies)/len(self.latencies) if self.latencies else 0.; return out

