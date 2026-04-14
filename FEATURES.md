# Lip Movement Feature Documentation

## Overview

We extract a **9-dimensional multivariate time series** from lip landmarks detected via MediaPipe Face Mesh (478 landmarks per frame). Each frame produces a feature vector that captures the lip configuration at that instant. The temporal evolution of these features encodes the dynamics of speech.

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

### 1. Vertical Aperture (outer)

$$f_1 = \frac{\| \mathbf{p}_0 - \mathbf{p}_{17} \|}{d_{\text{eye}}}$$

The Euclidean distance between the top (landmark 0) and bottom (landmark 17) of the **outer** lip contour, normalized.

**What it captures:** How open the mouth is vertically. Increases during open vowels (e.g., "ah" in "5"), decreases during bilabial closures (e.g., "b" in "8").

### 2. Horizontal Spread

$$f_2 = \frac{\| \mathbf{p}_{61} - \mathbf{p}_{291} \|}{d_{\text{eye}}}$$

The Euclidean distance between the left (landmark 61) and right (landmark 291) mouth corners, normalized.

**What it captures:** How wide the lips stretch. High for spread vowels like /i:/ (as in "three"), low for rounded vowels like /u:/ (as in "two") or pursed lips.

### 3. Inner Vertical Aperture

$$f_3 = \frac{\| \mathbf{p}_{13} - \mathbf{p}_{14} \|}{d_{\text{eye}}}$$

The Euclidean distance between the top (landmark 13) and bottom (landmark 14) of the **inner** lip contour, normalized.

**What it captures:** The actual oral opening — the gap between the inner edges of the lips. This is more speech-relevant than outer vertical aperture because it reflects the true airway/vocal tract opening. It is near zero when the mouth is closed (e.g., during /m/, /b/, /p/) and large during open vowels.

### 4. Outer Lip Area

$$f_4 = \frac{A_{\text{outer}}}{d_{\text{eye}}^2}$$

where $A_{\text{outer}}$ is computed via the **Shoelace formula** over the 20 ordered outer lip landmarks:

$$A = \frac{1}{2} \left| \sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i) \right|$$

**What it captures:** The total area enclosed by the outer lip contour. This is a holistic measure that jointly encodes both vertical opening and horizontal spread. A large area means the mouth is both open and wide; a small area means lips are closed or pursed.

### 5. Inner Lip Area

$$f_5 = \frac{A_{\text{inner}}}{d_{\text{eye}}^2}$$

Same Shoelace formula applied to the 20 inner lip landmarks.

**What it captures:** The area of the actual oral opening. Directly related to the visible oral cavity. More discriminative than outer area for distinguishing speech sounds because it ignores lip thickness variations.

### 6. Compactness (Circularity)

$$f_6 = \frac{4\pi \cdot A_{\text{outer}}}{P_{\text{outer}}^2}$$

where $P_{\text{outer}}$ is the perimeter of the outer lip polygon (sum of edge lengths). This is the **isoperimetric quotient**; it equals 1.0 for a perfect circle and approaches 0 for extremely elongated shapes.

**What it captures:** Whether the lip opening is **round** vs **elongated**. High compactness for rounded vowels like /o/ (as in "0", "4"); low compactness for spread shapes like /i:/ (as in "3"). This feature is independent of scale — it only measures shape.

### 7. Corner Angle

$$f_7 = \angle(\mathbf{p}_{61},\ \mathbf{p}_0,\ \mathbf{p}_{291})$$

The angle (in degrees) at the top-center of the outer lip (landmark 0), formed by the vectors to the left corner (landmark 61) and right corner (landmark 291).

$$\theta = \arccos\left(\frac{(\mathbf{p}_{61} - \mathbf{p}_0) \cdot (\mathbf{p}_{291} - \mathbf{p}_0)}{\|\mathbf{p}_{61} - \mathbf{p}_0\| \cdot \|\mathbf{p}_{291} - \mathbf{p}_0\|}\right)$$

**What it captures:** The angular spread of the lip corners as seen from the upper lip center. A wide angle (close to 180°) means the corners are far apart and nearly level with the top — indicating a spread/smile configuration. A narrow angle means the corners are pulled down or inward — indicating pursing or a more closed configuration.

### 8. Vertical Velocity

$$f_8 = \frac{d}{dt} f_1 \approx \nabla f_1$$

Computed via `np.gradient()` (central differences) of the vertical aperture time series.

**What it captures:** The **rate of mouth opening/closing**. Positive values indicate the mouth is opening; negative values indicate closing. This captures articulatory dynamics — plosive consonants (like /b/ in "8") show rapid open→close transitions, while sustained vowels show near-zero velocity. Critical for distinguishing digits with similar static shapes but different temporal dynamics.

### 9. Horizontal Velocity

$$f_9 = \frac{d}{dt} f_2 \approx \nabla f_2$$

Central differences of the horizontal spread time series.

**What it captures:** The **rate of lip spreading/narrowing**. Captures transitions like the spread-to-round movement in "5" (/faɪv/) or the spread onset in "3" (/θriː/).

---

## Why Not Aspect Ratio?

The original implementation included **Aspect Ratio** $= f_1 / f_2$ (vertical aperture ÷ horizontal spread). This was removed because:

1. **High correlation with Vertical Aperture:** Since horizontal spread varies much less than vertical aperture during speech (the mouth corners stay relatively stable), the aspect ratio is dominated by the numerator and tracks vertical aperture almost identically.
2. **Redundancy:** The information about the relationship between vertical and horizontal dimensions is already captured by having both $f_1$ and $f_2$ as separate features — any downstream model can learn the ratio implicitly.
3. **Compactness is better:** Feature 6 (compactness) captures the shape distinction (round vs elongated) more robustly via the isoperimetric quotient, which is scale-invariant and uses all contour points rather than just 4.

---

## Why Model as Time Series?

Speech is an inherently **temporal** process. Each digit consists of a sequence of phonemes, each requiring a different mouth configuration. The utterance of a digit is not a single shape — it is a **trajectory** through lip-shape space.

For example, saying "5" (/faɪv/) involves:
1. Lower lip tucks under upper teeth (/f/) → low inner aperture, moderate spread
2. Mouth opens wide (/aɪ/) → high vertical aperture, increasing area
3. Lower lip tucks again (/v/) → rapid closing

This trajectory is distinct from "9" (/naɪn/), even though both contain the /aɪ/ diphthong, because the start and end configurations differ (nasal /n/ vs labiodental /f,v/).

A single-frame snapshot cannot distinguish digits that share similar mouth shapes at some point during articulation. The **temporal pattern** — the sequence of shapes and the speed of transitions — is what makes each digit unique.
