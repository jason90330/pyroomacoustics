# Gunfire Localization Dataset Generator — Technical Architecture & Implementation Documentation

This document provides a comprehensive, high-fidelity guide to the technical design, architectural patterns, and mathematical models underpinning the game audio engine emulator and synthetic dataset generator in `generate_dataset.py`.

---

## 1. System Overview & Objectives
The `generate_dataset.py` system acts as a high-fidelity game audio engine emulator. Its primary objective is to generate synthetic binaural spatial audio datasets representing gunfire sounds in various acoustic spaces (presets). These datasets are crucial for training and testing machine learning models designed for **FPS game gunfire localization** (3D direction-of-arrival estimation).

Key engineering goals achieved in this implementation:
*   **Physical Realism**: Utilizes a geometric Image Source Method (ISM) via `pyroomacoustics` to model reflections up to high orders.
*   **Binaural HRTF Rendering**: Integrates human head-related transfer functions (HRTF) from the KEMAR database (SOFA format) to reproduce realistic Interaural Time Differences (ITD) and Interaural Level Differences (ILD).
*   **Acoustic Versatility**: Provides 12 consolidated architectural presets representing indoor rooms, open fields, street canyons, corridors, and courtyards.
*   **Dynamic Customization**: Supports real-time material overrides and random obstacle generation (with occlusion/diffraction effects) resembling game level geometries.

---

## 2. Coordinate Systems & Conventions

```
              North (+Y) [90° std / 0° front]
                            ^
                            |  Gunshot Source
                            |     o (dist=d)
                            |    /
                            |   / azimuth (clockwise)
                            |  / 
                            | /
  West (-X) [180° std] <----+----> East (+X) [0° std] (Listener faces +X)
                            |
                            |
                            v
              South (-Y) [270° std]
```

### 2.1 Standard vs. Clockwise Systems
*   **Listener Orientation**: The listener is positioned at `listener_pos` and faces along the **East (+X)** direction.
*   **Clockwise Game Convention (User Input)**:
    *   `0°` is **Front** (facing East, +X axis).
    *   `90°` is **Right** (facing South, -Y axis).
    *   `180°` is **Back** (facing West, -X axis).
    *   `270°` is **Left** (facing North, +Y axis).
*   **Counter-Clockwise Standard System (pyroomacoustics/SOFA)**:
    *   `0°` is **East (+X)**.
    *   `90°` is **North (+Y)**.
    *   `180°` is **West (-X)**.
    *   `270°` is **South (-Y)**.
*   **Azimuth Translation Formula**:
    $$\theta_{\text{standard}} = (360 - \theta_{\text{user}}) \pmod{360}$$
    $$\text{dx} = d \cdot \cos(\theta_{\text{standard}}), \quad \text{dy} = d \cdot \sin(\theta_{\text{standard}})$$

### 2.2 Z-Axis Elevation heights
*   **Listener Height**: Default standing ear-height is fixed at `1.8m`.
*   **Gunshot Source Height**: Default weapon muzzle height is set to `1.5m`.

---

## 3. The 12 Consolidated Acoustic Presets

The emulator consolidates physical environments into 12 highly tuned, geometrically distinct presets:

| Preset Name | Dimensions (W x L x H) | Max Reflection Order | Ceiling Material | Floor Material | Walls (E/W/N/S) | Description |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| `open_field` | Calculated (Adaptive) | `0` | Anechoic | Anechoic | Anechoic | Flat, reflectionless open field to isolate the clean gunshot sound. |
| `indoor_room` | $6.0 \times 8.0 \times 2.8$ m | `5` | Plasterboard | Wood | Plasterboard | Standard small residential room / site building. |
| `indoor_wood_room` | $10.0 \times 12.0 \times 3.0$ m | `6` | Plasterboard | Wood | Wood | Medium-sized wooden room with prominent warm mid-range reflections. |
| `indoor_hall` | $20.0 \times 30.0 \times 6.0$ m | `8` | Plasterboard | Tiles | Brick | Large exhibition hall or concrete lobby with long reverberation. |
| `indoor_warehouse` | $30.0 \times 40.0 \times 10.0$ m | `10` | Metal | Concrete | Metal | Massive industrial warehouse with loud, metallic, long-decay reflections. |
| `outdoor_street` | $10.0 \times 100.0 \times 15.0$ m | `4` | Anechoic | Concrete | Concrete | Street canyon (walls on North/South, open sky on ceiling/sides). |
| `outdoor_grass` | Calculated (Adaptive) | `2` | Anechoic | Grass (0.35 abs) | Grass (0.35 abs) | Forest or park with grass ground and scattering tree boundaries. |
| `outdoor_desert` | Calculated (Adaptive) | `2` | Anechoic | Sand (0.50 abs) | Sand (0.50 abs) | Desert dunes or sand pits; highly damp and dry acoustics. |
| `outdoor_snow` | Calculated (Adaptive) | `2` | Anechoic | Snow (0.15 abs) | Snow (0.15 abs) | Cold snowy fields; reflective snow surface floor. |
| `corridor` | $2.5 \times 30.0 \times 2.8$ m | `12` | Plasterboard | Linoleum | Plasterboard | Narrow, claustrophobic hallway; high-order reflections along the length. |
| `tunnel` | $5.0 \times 60.0 \times 4.0$ m | `15` | Concrete | Concrete | Concrete | Long concrete pipe or underground tunnel with heavy, dense echo. |
| `courtyard` | Calculated (Adaptive) | `4` | Anechoic | Concrete | Brick | Semi-open stone courtyard; stone walls with open sky ceiling. |

---

## 4. Dynamic Material Mapping Engine

### 4.1 Game Material Registry
To simulate maps of varying materials (e.g., metal containers vs. stone columns), human-readable string tokens are mapped to physical multi-octave absorption coefficients from the `pyroomacoustics` database or directly to flat absorption values:

```python
GAME_MATERIALS = {
    "brick": "brickwork",
    "concrete": "unpainted_concrete",
    "rough_concrete": "rough_concrete",
    "stone": "limestone_wall",
    "wood": "wood_16mm",
    "wooden_lining": "wooden_lining",
    "metal": 0.10,          # highly reflective metal sheets
    "carpet": "carpet_cotton",
    "linoleum": "linoleum_on_concrete",
    "tiles": "ceramic_tiles",
    "marble": "marble_floor",
    "glass": "glass_window",
    "plaster": "ceiling_plasterboard",
    "plasterboard": "plasterboard",
    "grass": 0.35,          # Grass absorption
    "sand": 0.50,           # Sand absorption
    "snow": 0.15,           # Snow absorption
    "water": 0.05,          # Water surface
    "anechoic": "anechoic"  # Absorbs 100% sound (open sky)
}
```

### 4.2 Material Override Flow
1.  **CLI Arguments**: The user can override default preset materials using `--wall-material`, `--floor-material`, and `--ceiling-material`.
2.  **Lookup & Extraction**: Valid overrides are cross-referenced with `GAME_MATERIALS`.
3.  **Instantiation**: `pra.make_materials()` converts these definitions into `Material` objects containing target absorption/scattering coefficients across frequency bands.

---

## 5. Low-Frequency Crossover & Bass Preservation

### 5.1 The Phase Cancellation Problem
Head-Related Transfer Functions (HRTFs) are highly spatialized filters. However, at low frequencies (below ~400 Hz), the human head is acoustically transparent. Applying high-frequency HRTF interpolation to low frequencies can introduce unnatural comb-filtering, ringing, or severe phase cancellations, completely gutting the powerful bass/thump of a gunshot.

### 5.2 Crossover Filter Pipeline
To preserve the realistic low-end energy of muzzle blasts, a Butterworth crossover filter splits the simulated audio into two bands at `400 Hz`:
*   **Low-Pass Branch (< 400 Hz)**: Processed using a spatial-free, omnidirectional room impulse response (RIR) to preserve natural bass cues.
*   **High-Pass Branch (> 400 Hz)**: Processed using the full, spatialized HRTF convolved RIR to provide high-fidelity localization cues (ITD/ILD).
*   **Recombination**: Both branches are filtered using matched 2nd-order Butterworth filters (to prevent phase misalignment at the crossover frequency) and summed back together:

```
                          +---> HPF (>400Hz) ---> Convolve with HRTF ---+
                          |                                             |
Input Mono Gunshot Sound -+                                             +---> Summed Stereo Output
                          |                                             |
                          +---> LPF (<400Hz) ---> Convolve with Omni ----+
```

---

## 6. HRTF Directivity & SOFA Interpolation

### 6.1 SOFA Head Orientation
We load the `mit_kemar_normal_pinna` SOFA dataset. The KEMAR head faces along the $+X$ direction by default. A 3D rotation object (`Rotation3D`) aligns the listener's head with the virtual coordinate frame:
```python
head_orientation = Rotation3D([0.0, 0.0], "yz", degrees=True)
```
This maps the left ear to the $-Y$ direction and the right ear to the $+Y$ direction.

### 6.2 Nearest-Neighbor vs. Spherical Harmonics
By default, the script sets `--interp-order` to `None`, which utilizes **nearest-neighbor interpolation** for spatial SOFA sampling. This design decision is crucial: high-order spherical harmonics interpolation can introduce frequency response blurring and ringing artifacts, whereas nearest-neighbor preserves the raw, sharp frequency notches necessary for vertical localization.

---

## 7. Random Obstacle Generation Engine

FPS games contain obstacles (boxes, crates, pillars, or inner walls) that block the direct line-of-sight between a shooter and the listener. The obstacle generator simulates this shadowing effect:

```
        +-----------------------------------+
        |            East Wall              |
        |                                   |
        |          Obstacle Wall            |
        |           /=======\               |
        |          /         \              |
        |   Source o          x Listener    |
        |                                   |
        |            West Wall              |
        +-----------------------------------+
```

### 7.1 Mathematical Placement Algorithm
When `--obstacle` is requested:
1.  **Seed Lock**: We instantiate a local random generator `rng = np.random.RandomState(args.obstacle_seed)`. Because the seed is fixed for the duration of a run, the obstacle remains **completely static** across all 36 simulated azimuth angles.
2.  **Coordinate Selection**: We pick a random center point $(c_x, c_y)$ inside the inner 20%-80% region of the room, a random rotation angle $\theta$, and a random length $L$ scaled to the room dimensions.
3.  **Boundary Clipping**: The wall endpoints $p_1, p_2$ are calculated and clipped to maintain at least a `0.5m` safety margin from the outer walls.
4.  **Listener Collision Avoidance**: We calculate the minimum perpendicular 2D distance from the listener position to the wall segment. If it falls below `0.8m`, the wall is discarded, and a new candidate is generated:
    $$t = \text{clip}\left(\frac{\mathbf{ap} \cdot \mathbf{ab}}{\|\mathbf{ab}\|^2}, 0, 1\right), \quad \text{proj} = \mathbf{a} + t \cdot \mathbf{ab}$$
    $$\text{distance} = \|\mathbf{p} - \text{proj}\|$$

### 7.2 Source Collision Nudging
Because the gunshot source rotates on a circle of radius `distance`, it might intersect the generated obstacle. To prevent simulation errors, if the source-to-wall distance falls below `0.3m`, the source position is automatically nudged perpendicular to the obstacle wall to maintain the safety margin:
```python
wall_vec = w2 - w1
perp_vec = np.array([-wall_vec[1], wall_vec[0]]) / np.linalg.norm(wall_vec)
side = np.sign(np.dot(source_pos[:2] - w1, perp_vec))
source_pos[:2] += side * perp_vec * (0.3 - dist_source)
```

### 7.3 Band Count Matching
If the outer walls are flat/single-band (`len(outer_walls[0].absorption) == 1`, such as in the `open_field` preset), the multi-band coefficients of the obstacle material are averaged to create a matching flat/single-band `pra.Material` object. Otherwise, the obstacle material is resampled to match the room's multi-band `octave_bands` definition.

### 7.4 General Room Transition
If an obstacle is generated, the simulation dynamically switches from a `ShoeBox` to a general polygonal `pra.Room` using the 6 boundary walls and the 1 obstacle wall, enabling C++ level ray obstruction checks.

---

## 8. Spatial Visualization System

The `--visualize` CLI option provides a 3D visual confirmation of the room layout, obstacle placement, listener position, and source location.

### 8.1 Legend Mapping & Plot Features
*   **Listener (Mics)**: Marked as a black `x` at the listener coordinates.
*   **Sound Source (Gun)**: Marked as a deep blue `o` (representing the gunshot origin).
*   **Image Sources (Reflections)**: Marked as light-blue `o` points outside the room boundaries.
*   **Top-Right Annotation Box**: Dynamically displays the current preset and active material names for the walls, floor, and ceiling.
*   **Legend**: Placed in the top-left corner to clarify all symbols.

### 8.2 Bypassing the Directivity Bug
In `pyroomacoustics`, calling `room.plot()` with `plot_directivity=True` (default) triggers a lookup for `plot_response()` on the microphone array. Because our microphones use SOFA-based `MeasuredDirectivity` objects, this lookup raises an `AttributeError`. To resolve this, we call:
```python
fig, ax = room.plot(plot_directivity=False)
```
This successfully bypasses the bug, rendering the room geometry and markers flawlessly.

### 8.3 Early Exit Optimization
To save time and prevent generating unwanted data during geometry verification, if `--visualize` is provided, the script will render the 3D plot for the first angle, print a completion notice, and immediately exit the process (`sys.exit(0)`) without simulating the other 35 angles or writing any `.wav` files to disk.

---

## 9. Spatial WAV Analysis & Visualization Tool

A dedicated analysis script is provided in `utils/visualize_wav.py` to inspect the generated spatial audio signals and evaluate the exact physical cues (ITD, ILD, and spectral notches) used by human ears (and ML models) for gunfire direction-of-arrival (DOA) estimation.

### 9.1 Visual Analysis Layout (3x2 Subplots)
1. **Waveforms (Left / Right)**: Display the raw time-domain signals (in milliseconds) for both ears to inspect relative onset delay and amplitude decay.
2. **Spectrograms (Left / Right)**: Calculate the log-dB short-time Fourier transform (STFT) for each ear. This exposes the sound energy over time and frequency, making spectral notches (from the pinna reflection) and high-frequency roll-off (from the head shadow) visible.
3. **Cross-Correlation (ITD Analysis)**: Computes the fast cross-correlation between left and right channels using FFT (`scipy.signal.correlate(..., method='fft')`).
   - The lag window is masked to $\pm 1.2\text{ ms}$ (encompassing the maximum human head size).
   - The red dashed line marks the peak correlation lag, showing the exact Interaural Time Difference (ITD) in milliseconds.
4. **Interaural Level Difference (ILD)**: Plots the frequency-dependent level difference (Left minus Right PSD in dB) across the entire frequency range.

### 9.2 Integrated Frequency Statistics
To resolve front/back ambiguity (the "cone of confusion") and quantify acoustic coloring, the tool computes and overlays statistics boxes directly onto the top-right corner of their respective Spectrogram plots:
* **Peak Frequency**: The frequency component containing the maximum spectral energy.
* **Spectral Centroid**: The weighted average frequency of the spectrum, representing the brightness of the tone.
* **High-Frequency (HF) Ratio ($>5\text{ kHz}$)**: The percentage of the signal's total energy that lies above $5\text{ kHz}$. A lower HF ratio in one ear indicates physical shadowing by the head or the pinna (indicating the source is located to the rear or opposite side).

### 9.3 Usage & CLI Options
```bash
python3 ./utils/visualize_wav.py --input <path_to_wav> [options]
```
* `--input` / `-i`: Path to the target stereo WAV file.
* `--output-plot` / `-o`: Output path for the analysis image (defaults to saving next to the WAV file as `*_analysis.png`).
* `--no-show`: Skips opening the matplotlib GUI window and immediately saves the plot.

---

## 10. CLI Arguments Reference

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-h`, `--help` | - | - | Shows the help menu and exits. |
| `--input-wav` | `str` | `None` | Path to a mono gunshot WAV file. If unspecified, a high-fidelity synthetic gunshot is generated. |
| `--output-dir` | `str` | `None` | Directory where output `.wav` files will be saved. |
| `--preset` | `str` | `indoor_hall` | Choose from the 12 unified presets (e.g. `open_field`, `indoor_room`, `tunnel`). |
| `--distance` | `float` | `2.0` | Range from listener to gunshot source in meters. |
| `--fs` | `int` | `44100` | Target sampling rate in Hz. |
| `--hrtf-name` | `str` | `mit_kemar_normal_pinna` | SOFA HRTF filename. |
| `--interp-order` | `int` | `None` | Spherical harmonics interpolation order (set to `None` for nearest-neighbor). |
| `--bgm-wav` | `str` | `None` | Path to optional background music/ambient noise to mix in. |
| `--bgm-snr` | `float` | `20.0` | SNR level in dB when mixing background music. |
| `--crossover-freq` | `float` | `400.0` | Crossover split frequency. Set to `0` to disable. |
| `--wall-material` | `str` | `None` | Overrides the material of all four walls. |
| `--floor-material` | `str` | `None` | Overrides the floor material. |
| `--ceiling-material`| `str` | `None` | Overrides the ceiling material. |
| `--visualize` | - | `False` | Visualizes the 3D space for the first angle and exits immediately. |
| `--obstacle` | - | `False` | Generates a random static obstacle partition inside the room. |
| `--obstacle-seed` | `int` | `42` | Seed for deterministic random obstacle coordinates. |
