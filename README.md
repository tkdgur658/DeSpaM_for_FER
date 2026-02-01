# DeSpaM
DeSpaM: Decomposed Spatial Modulator for efficient facial expression recognition (FER).

DeSpaM is a lightweight decomposed spatial modulator that generates a single-channel spatial gating map on the full $H \times W$ grid (without global pooling) to improve the accuracy-latency tradeoff in real-time facial expression recognition.

# DeSpaM: Decomposed Spatial Modulator for Efficient FER

![DeSpaM Module Overview](DeSpaM.png)

**DeSpaM** is a lightweight decomposed spatial modulator that generates a single-channel spatial gating map on the full H×W grid (without global pooling) to improve the accuracy–latency tradeoff in real-time facial expression recognition.


## Comparison with Existing Modulators

![Modulator comparison](despam_comparison.png)

DeSpaM avoids spatial-resolution collapse and heavy multi-branch attention by decomposing spatial modulation into context aggregation and channel compression.
