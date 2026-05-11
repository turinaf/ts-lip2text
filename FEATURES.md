# Lip Movement Feature

## Overview

We extract a **5-dimensional multivariate time series** from lip landmarks detected via MediaPipe Face Mesh (478 landmarks per frame). Each frame produces a feature vector that captures the lip configuration at that instant. The temporal evolution of these features encodes the dynamics of speech.

All spatial features are **normalized by inter-ocular distance** (distance between outer eye corners, landmarks 33 and 263) to make them invariant to head size, distance from camera, and resolution.

## Landmark Reference

```
MediaPipe Face Mesh — Lip Landmarks Used:

Outer Lip (20 points): 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                        185, 40, 39, 37, 0, 267, 269, 270, 409

Inner Lip (20 points): 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                        191, 80, 81, 82, 13, 312, 311, 310, 415

Key Anatomical Points:
  0   — Top center (outer lip, upper vermilion border)
  17  — Bottom center (outer lip, lower vermilion border)
  61  — Left corner of mouth
  291 — Right corner of mouth
  13  — Top center (inner lip)
  14  — Bottom center (inner lip)
```

## Normalization

$$d_{\text{eye}} = \| \mathbf{p}_{33} - \mathbf{p}_{263} \|$$

All distance-based features are divided by $d_{\text{eye}}$; area-based features are divided by $d_{\text{eye}}^2$.

---

## Features

### 1. Vertical Aperture

$$f_1 = \frac{\| \mathbf{p}_{13} - \mathbf{p}_{14} \|}{d_{\text{eye}}}$$

The Euclidean distance between the top (landmark 13) and bottom (landmark 14) of the **inner** lip contour, normalized.

**What it captures:** The actual oral opening — the gap between the inner edges of the lips. This is more speech-relevant than the outer lip distance because it reflects the true airway/vocal tract opening. Near zero when the mouth is closed (e.g., during /m/, /b/, /p/) and large during open vowels (e.g., "ah" in "5").

### 2. Horizontal Spread

$$f_2 = \frac{\| \mathbf{p}_{61} - \mathbf{p}_{291} \|}{d_{\text{eye}}}$$

The Euclidean distance between the left (landmark 61) and right (landmark 291) mouth corners, normalized.

**What it captures:** How wide the lips stretch. High for spread vowels like /i:/ (as in "three"), low for rounded vowels like /u:/ (as in "two") or pursed lips.

### 3. Inner Lip Area

$$f_3 = \frac{A_{\text{inner}}}{d_{\text{eye}}^2}$$

where $A_{\text{inner}}$ is computed via the **Shoelace formula** over the 20 ordered inner lip landmarks:

$$A = \frac{1}{2} \left| \sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i) \right|$$

**What it captures:** The area of the actual oral opening. Directly related to the visible oral cavity. More discriminative than outer lip area for distinguishing speech sounds because it ignores lip thickness variations and focuses on the true opening.

### 4. Compactness (Circularity)

$$f_4 = \frac{4\pi \cdot A_{\text{outer}}}{P_{\text{outer}}^2}$$

where $P_{\text{outer}}$ is the perimeter of the outer lip polygon (sum of edge lengths). This is the **isoperimetric quotient**; it equals 1.0 for a perfect circle and approaches 0 for extremely elongated shapes.

**What it captures:** Whether the lip opening is **round** vs **elongated**. High compactness for rounded vowels like /o/ (as in "0", "4"); low compactness for spread shapes like /i:/ (as in "3"). This feature is independent of scale — it only measures shape.

### 5. Lip Speed

$$f_5 = \sqrt{\left(\frac{d f_1}{dt}\right)^2 + \left(\frac{d f_2}{dt}\right)^2}$$

The magnitude of the 2D velocity vector formed by the vertical aperture velocity and horizontal spread velocity (both computed via `np.gradient()`, central differences).

**What it captures:** The **overall speed of lip movement** regardless of direction. High values indicate rapid articulatory transitions — plosive consonants (like /b/ in "8") produce sharp spikes, while sustained vowels show near-zero speed. By combining vertical and horizontal velocities into a single magnitude, this avoids redundancy while preserving the dynamic information. It captures how "active" the lips are at each moment without needing to distinguish the direction of movement.


## Why Model as Time Series?

Speech is an inherently **temporal** process. Each digit consists of a sequence of phonemes, each requiring a different mouth configuration. The utterance of a digit is not a single shape — it is a **trajectory** through lip-shape space.

For example, saying "5" (/faɪv/) involves:
1. Lower lip tucks under upper teeth (/f/) → low inner aperture, moderate spread
2. Mouth opens wide (/aɪ/) → high vertical aperture, increasing area
3. Lower lip tucks again (/v/) → rapid closing

This trajectory is distinct from "9" (/naɪn/), even though both contain the /aɪ/ diphthong, because the start and end configurations differ (nasal /n/ vs labiodental /f,v/).

A single-frame snapshot cannot distinguish digits that share similar mouth shapes at some point during articulation. The **temporal pattern** — the sequence of shapes and the speed of transitions — is what makes each digit unique.


# SEq2seq
MT
encoder-decoder
Instance normalization = normalize the whole input to zero mean and 1 std deviation

Tokenize ts: patchTSt paper
ViT - 



- **Normalization (general):** Distances scale as length (L) and areas as L^2. The inter-ocular distance $d_{\text{eye}}$ is a linear scale estimate, so divide:
  - distances by $d_{\text{eye}}$ to get scale-invariant, unitless values (e.g., `f1`, `f2`);
  - areas by $d_{\text{eye}}^2$ to remove quadratic scaling (e.g., `f3`).  
  This is pure dimensional analysis: dividing an area by a length gives wrong units, so you must divide by length squared for a dimensionless area measure.

- **`f_1` Vertical aperture and `f_2` Horizontal spread:** Both are Euclidean distances ($\|p_a-p_b\|$) normalized by $d_{\text{eye}}$. Rationale:
  - Euclidean distance is the natural metric for landmark separation.
  - Normalizing by $d_{\text{eye}}$ removes subject/camera scale differences so features reflect shape not size.
  - Using inner-lip points for aperture targets the actual oral opening (physiologically relevant); mouth-corner points for spread capture lateral stretch.

- **`f_3` Inner lip area (Shoelace) and why squared normalization:** Area for a polygon via the Shoelace formula is in units L^2. To make it comparable across subjects/recordings you compute
  $$f_3 = \frac{A_{\text{inner}}}{d_{\text{eye}}^2}.$$
  The squared denominator is necessary because area scales with the square of linear dimensions; this produces a dimensionless, scale-invariant area feature.

- **`f_4` Compactness / circularity:**
  - Formula: $f_4 = \dfrac{4\pi\,A_{\text{outer}}}{P_{\text{outer}}^2}$.
  - Why $P^2$ in denominator: dimensions—area has units L^2 and perimeter squared is (L)^2, so $A/P^2$ is dimensionless. That is required for a pure shape descriptor independent of scale.
  - Why multiply by $4\pi$: it normalizes the quantity so a perfect circle yields exactly 1. Proof: circle area $A=\pi r^2$, perimeter $P=2\pi r$ → substitute gives $(4\pi \cdot \pi r^2)/(4\pi^2 r^2)=1$. Thus the factor sets the upper bound and makes interpretation easier (1 = circle, closer to 0 = elongated/irregular).
  - Interpretation: it is the classical isoperimetric quotient — a compactness measure robustly distinguishing round vs elongated shapes while being rotation- and scale-invariant. (Practical note: perimeter estimates are sensitive to small boundary jitter; smoothing or using polygonal perimeter helps.)

- **`f_5` Lip speed (sqrt of sum squares of derivatives):**
  - Formula: $f_5 = \sqrt{\left(\frac{d f_1}{dt}\right)^2 + \left(\frac{d f_2}{dt}\right)^2}$.
  - Why square and sqrt: this is the Euclidean (L2) norm of the velocity vector $v=(\dot f_1,\dot f_2)$. Squaring then summing yields the squared magnitude; the square root returns the actual magnitude (speed). That gives a positive scalar representing instantaneous lip motion magnitude regardless of direction.
  - Why operate on `f1`,`f2` (normalized features): derivatives of normalized measures remain scale-invariant; if you differentiated raw pixel coords you'd conflate camera distance changes with motion.
  - Implementation note: finite differences (e.g., `np.gradient`) approximate $\dot f$, and if you need physical units convert per-frame to per-second by dividing by frame period.

- **Practical and interpretability notes (short):**
  - Dimensionless features (from the normalization rules above) allow consistent weighting and learning across subjects and recording conditions.
  - Compactness is sensitive to noise in perimeter; area is more robust but ignores boundary complexity—use both for complementary information.
  - `f_5` (L2 speed) compresses 2D motion into one scalar; if direction matters, keep components separately or use angular measures.
  - If frames-per-second varies, rescale derivatives to seconds for comparability.

