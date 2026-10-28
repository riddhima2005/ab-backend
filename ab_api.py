from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy import stats
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class ABData(BaseModel):
    control_conversions: int
    control_visitors: int
    treatment_conversions: int
    treatment_visitors: int

@app.post("/ab-test")
def ab_test(d: ABData):
    pc = d.control_conversions / d.control_visitors
    pt = d.treatment_conversions / d.treatment_visitors
    pooled = (d.control_conversions + d.treatment_conversions) / (d.control_visitors + d.treatment_visitors)
    se_test = np.sqrt(pooled * (1 - pooled) * (1 / d.control_visitors + 1 / d.treatment_visitors))
    z = (pt - pc) / se_test
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    diff = pt - pc
    se_diff = np.sqrt(pt * (1 - pt) / d.treatment_visitors + pc * (1 - pc) / d.control_visitors)
    low, high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    if p < 0.05:
        verdict = "The change made a real difference. It is unlikely to be luck."
    else:
        verdict = "No clear difference. The gap could easily be luck."

    return {
        "control_rate": round(pc, 4),
        "treatment_rate": round(pt, 4),
        "difference": round(diff, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "p_value": round(p, 4),
        "conclusion": verdict,
    }








