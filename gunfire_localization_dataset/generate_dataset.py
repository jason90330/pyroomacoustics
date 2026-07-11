#!/usr/bin/env python3
"""
Gunfire Localization Audio Dataset Generator
Creates a binaural spatial audio dataset using pyroomacoustics and KEMAR HRTF.
The dataset simulates a gunshot source placed at 10-degree increments around a listener.
Coordinate conventions:
- Front: 0 degrees
- Right: 90 degrees
- Back: 180 degrees
- Left: 270 degrees
(Clockwise system)
"""

import os
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import pyroomacoustics as pra
from pyroomacoustics.directivities import MeasuredDirectivityFile, Rotation3D

# Game-friendly materials mapped to pyroomacoustics database keys or absorption floats.
# Allows the user to configure acoustic spaces dynamically like a game audio engine.
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
    "grass": 0.35,         # grass absorption
    "sand": 0.50,          # dry sand absorption
    "snow": 0.15,          # snow absorption
    "water": 0.05,         # water surface
    "anechoic": "anechoic" # open sky/anechoic
}

def generate_synthetic_gunshot(fs=44100):
    """
    Synthesize a realistic gunshot sound:
    - An initial sharp pressure spike (impulse).
    - A high-frequency crack (high-pass filtered noise with fast decay).
    - A low-frequency muzzle blast (low-pass filtered noise with medium decay).
    """
    duration = 0.5  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # White noise base
    noise = np.random.normal(0, 1, len(t))
    
    # Muzzle blast (low-frequency explosion)
    b_lp, a_lp = butter(2, 350.0 / (fs / 2.0), btype='low')
    blast_noise = lfilter(b_lp, a_lp, noise)
    blast_decay = np.exp(-t / 0.05)  # 50 ms time constant
    muzzle_blast = blast_noise * blast_decay
    
    # Supersonic crack (high-frequency snap)
    b_hp, a_hp = butter(2, 2000.0 / (fs / 2.0), btype='high')
    crack_noise = lfilter(b_hp, a_hp, noise)
    crack_decay = np.exp(-t / 0.006)  # 6 ms time constant
    muzzle_crack = crack_noise * crack_decay
    
    # Combine components
    signal = 0.4 * muzzle_blast + 0.6 * muzzle_crack
    
    # Add sharp initial impulse
    signal[0] += 1.0
    
    # Normalize to peak amplitude of 1.0
    signal /= np.max(np.abs(signal))
    return signal

def run_simulation(hrtf_path, source_signal, fs, user_angle_deg, distance=2.0, preset='indoor_hall', interp_order=None, crossover_freq=400.0, wall_material=None, floor_material=None, ceiling_material=None, visualize=False, obstacle=False, obstacle_seed=42):
    """
    Simulates a single sound event at a specific clockwise user angle.
    """
    # 1. Map clockwise user angle to standard counter-clockwise azimuth angle (in degrees)
    # Front (0) -> 0
    # Right (90) -> 270
    # Back (180) -> 180
    # Left (270) -> 90
    azimuth_std = (360 - user_angle_deg) % 360
    azimuth_rad = np.radians(azimuth_std)
    
    # 2. Define listener (head) and source configurations based on game presets
    # Eye/ear heights and gun muzzle heights are set realistically
    listener_height = 1.8  # Default standing height
    source_height = 1.7    # Default standing muzzle height
    
    # Initialize default material variables for the surface sides
    east_mat = "brickwork"
    west_mat = "brickwork"
    north_mat = "brickwork"
    south_mat = "brickwork"
    ceiling_mat = "ceiling_plasterboard"
    floor_mat = "concrete_floor"
    
    if preset == 'open_field':
        # Completely open reflection-free field (Anechoic)
        width = 2 * distance + 20.0
        length = 2 * distance + 20.0
        height = 20.0
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, length / 2.0, listener_height])
        east_mat = "anechoic"
        west_mat = "anechoic"
        north_mat = "anechoic"
        south_mat = "anechoic"
        ceiling_mat = "anechoic"
        floor_mat = "anechoic"
        max_order = 0
    elif preset == 'indoor_room':
        # Small/Medium residential room (wooden floors, plasterboard ceiling, brick walls)
        width = max(8.0, 2 * distance + 2.0)
        length = max(8.0, 2 * distance + 2.0)
        height = 3.0
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, length / 2.0, listener_height])
        east_mat = "brickwork"
        west_mat = "brickwork"
        north_mat = "brickwork"
        south_mat = "brickwork"
        ceiling_mat = "ceiling_plasterboard"
        floor_mat = "wood_16mm"
        max_order = 10
    elif preset == 'indoor_wood_room':
        # Medium wood-panelled indoor room (e.g. partition boards, wooden linings)
        width = max(12.0, 2 * distance + 2.0)
        length = max(12.0, 2 * distance + 2.0)
        height = 5.0
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, length / 2.0, listener_height])
        east_mat = "wooden_lining"
        west_mat = "wooden_lining"
        north_mat = "wooden_lining"
        south_mat = "wooden_lining"
        ceiling_mat = "ceiling_plasterboard"
        floor_mat = "wood_16mm"
        max_order = 10
    elif preset == 'indoor_hall':
        # Large concrete/masonry hall (highly reflective brick/stone walls, concrete floor)
        width = max(15.0, 2 * distance + 2.0)
        length = max(15.0, 2 * distance + 2.0)
        height = 6.0
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, length / 2.0, listener_height])
        east_mat = "brickwork"
        west_mat = "brickwork"
        north_mat = "brickwork"
        south_mat = "brickwork"
        ceiling_mat = "ceiling_plasterboard"
        floor_mat = "concrete_floor"
        max_order = 10
    elif preset == 'indoor_warehouse':
        # Large metal/concrete warehouse structure (highly reflective, metallic ceiling and walls)
        width = max(30.0, 2 * distance + 2.0)
        length = max(20.0, 2 * distance + 2.0)
        height = 8.0
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, length / 2.0, listener_height])
        east_mat = 0.1
        west_mat = 0.1
        north_mat = 0.1
        south_mat = 0.1
        ceiling_mat = 0.1
        floor_mat = "concrete_floor"
        max_order = 8
    elif preset == 'outdoor_street':
        # Street canyon flanked by buildings on the east/west sides, open on north/south
        width = max(10.0, 2 * distance + 2.0)
        length = max(20.0, distance + 10.0)
        height = 12.0
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, 5.0, listener_height])
        east_mat = "brickwork"
        west_mat = "brickwork"
        north_mat = "anechoic"
        south_mat = "anechoic"
        ceiling_mat = "anechoic"
        floor_mat = "concrete_floor"
        max_order = 4
    elif preset == 'outdoor_grass':
        # Open field or forest terrain surrounded by distant trees/foliage
        width = 2 * distance + 20.0
        length = 2 * distance + 20.0
        height = 15.0
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, length / 2.0, listener_height])
        east_mat = 0.92
        west_mat = 0.92
        north_mat = 0.92
        south_mat = 0.92
        ceiling_mat = "anechoic"
        floor_mat = 0.35
        max_order = 3
    elif preset == 'outdoor_desert':
        # Open sandy desert terrain with distant stone walls/dunes
        width = max(15.0, 2 * distance + 2.0)
        length = max(15.0, 2 * distance + 2.0)
        height = 12.0
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, length / 2.0, listener_height])
        east_mat = "brickwork"
        west_mat = "brickwork"
        north_mat = "brickwork"
        south_mat = "brickwork"
        ceiling_mat = "anechoic"
        floor_mat = 0.45
        max_order = 3
    elif preset == 'outdoor_snow':
        # Cold snowy landscape surrounded by distant quiet cliffs
        width = 2 * distance + 15.0
        length = 2 * distance + 15.0
        height = 15.0
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, length / 2.0, listener_height])
        east_mat = 0.95
        west_mat = 0.95
        north_mat = 0.95
        south_mat = 0.95
        ceiling_mat = "anechoic"
        floor_mat = 0.15
        max_order = 2
    elif preset == 'corridor':
        # Narrow hallway (e.g. 3m wide, reflective walls, plaster ceiling)
        width = 3.0
        length = max(20.0, distance + 5.0)
        height = 3.5
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, 5.0, listener_height])
        east_mat = "brickwork"
        west_mat = "brickwork"
        north_mat = "brickwork"
        south_mat = "brickwork"
        ceiling_mat = "ceiling_plasterboard"
        floor_mat = "concrete_floor"
        max_order = 8
    elif preset == 'tunnel':
        # Highly reflective masonry tunnel/sewer
        width = 4.0
        length = max(15.0, distance + 5.0)
        height = 3.0
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, 5.0, listener_height])
        east_mat = "brickwork"
        west_mat = "brickwork"
        north_mat = "brickwork"
        south_mat = "brickwork"
        ceiling_mat = "brickwork"
        floor_mat = "marble_floor"
        max_order = 10
    elif preset == 'courtyard':
        # Semi-open stone courtyard or plaza surrounded by stone buildings
        width = max(20.0, 2 * distance + 2.0)
        length = max(20.0, 2 * distance + 2.0)
        height = 10.0
        room_dim = [width, length, height]
        listener_pos = np.array([width / 2.0, length / 2.0, listener_height])
        east_mat = "brickwork"
        west_mat = "brickwork"
        north_mat = "brickwork"
        south_mat = "brickwork"
        ceiling_mat = "anechoic"
        floor_mat = "concrete_floor"
        max_order = 4
    else:
        raise ValueError(f"Unknown preset: {preset}")

    # Apply material overrides dynamically if specified
    if wall_material is not None:
        east_mat = GAME_MATERIALS[wall_material]
        west_mat = GAME_MATERIALS[wall_material]
        north_mat = GAME_MATERIALS[wall_material]
        south_mat = GAME_MATERIALS[wall_material]
    if floor_material is not None:
        floor_mat = GAME_MATERIALS[floor_material]
    if ceiling_material is not None:
        ceiling_mat = GAME_MATERIALS[ceiling_material]

    materials = pra.make_materials(
        east=east_mat,
        west=west_mat,
        north=north_mat,
        south=south_mat,
        ceiling=ceiling_mat,
        floor=floor_mat,
    )

    # 3. Load HRTF (measured directivity patterns for left and right ears)
    # KEMAR head faces +x (identity rotation: azimuth=0, colatitude=90)
    head_orientation = Rotation3D([0.0, 0.0], "yz", degrees=True)
    
    # MeasuredDirectivityFile handles the interpolation of HRIRs from the SOFA dataset
    hrtf = MeasuredDirectivityFile(
        path=hrtf_path,
        fs=fs,
        interp_order=interp_order,
        interp_n_points=1000,
    )
    
    dir_left = hrtf.get_mic_directivity("left", orientation=head_orientation)
    dir_right = hrtf.get_mic_directivity("right", orientation=head_orientation)
    
    # 4. Calculate source position based on distance and azimuth
    if distance == 0:
        # Self-firing event: the gun is held in front of the listener's head.
        # It remains at a fixed relative position (facing forward / +x direction).
        dx = 0.5
        dy = 0.0
        dz = -0.1
    else:
        dx = distance * np.cos(azimuth_rad)
        dy = distance * np.sin(azimuth_rad)
        dz = source_height - listener_height
        
    source_pos = listener_pos + np.array([dx, dy, dz])
    
    # Ensure listener and source are inside the room boundaries (with a small safety margin)
    margin = 0.1
    for i in range(3):
        listener_pos[i] = np.clip(listener_pos[i], margin, room_dim[i] - margin)
        source_pos[i] = np.clip(source_pos[i], margin, room_dim[i] - margin)
    
    # 5. Build room simulation
    if obstacle:
        # Create a temporary ShoeBox room to get the outer walls
        if materials is None:
            room_materials = pra.make_materials(
                east=0.0, west=0.0, north=0.0, south=0.0, ceiling=0.0, floor=0.0
            )
        else:
            room_materials = materials
            
        temp_room = pra.ShoeBox(
            room_dim,
            fs=fs,
            max_order=max_order,
            materials=room_materials,
            air_absorption=True
        )
        outer_walls = list(temp_room.walls)
        
        # Helper to compute distance from point to 2D line segment
        def dist_point_to_segment_2d(p, a, b):
            ab = b - a
            ap = p - a
            ab_len_sq = np.sum(ab**2)
            if ab_len_sq == 0:
                return np.linalg.norm(ap)
            t = np.clip(np.sum(ap * ab) / ab_len_sq, 0.0, 1.0)
            projection = a + t * ab
            return np.linalg.norm(p - projection)
            
        # Seed the local random generator for deterministic obstacle placement
        rng = np.random.RandomState(obstacle_seed)
        
        w1, w2 = None, None
        for attempt in range(200):
            cx = rng.uniform(0.2 * width, 0.8 * width)
            cy = rng.uniform(0.2 * length, 0.8 * length)
            theta = rng.uniform(0.0, 2.0 * np.pi)
            
            min_dim = min(width, length)
            max_len = max(1.5, min_dim * 0.5)
            L = rng.uniform(1.5, max_len)
            
            p1 = np.array([cx - (L/2) * np.cos(theta), cy - (L/2) * np.sin(theta)])
            p2 = np.array([cx + (L/2) * np.cos(theta), cy + (L/2) * np.sin(theta)])
            
            # Clip to room boundaries with 0.5m margin
            p1[0] = np.clip(p1[0], 0.5, width - 0.5)
            p1[1] = np.clip(p1[1], 0.5, length - 0.5)
            p2[0] = np.clip(p2[0], 0.5, width - 0.5)
            p2[1] = np.clip(p2[1], 0.5, length - 0.5)
            
            if np.linalg.norm(p2 - p1) < 1.0:
                continue
                
            # Check safety distance to listener (must be > 0.8m)
            dist_listener = dist_point_to_segment_2d(listener_pos[:2], p1, p2)
            if dist_listener > 0.8:
                w1, w2 = p1, p2
                break
                
        if w1 is not None and w2 is not None:
            # Create the 3D corners for the vertical obstacle wall (from z=0 to z=height)
            corners = np.array([
                [w1[0], w1[1], 0.0],
                [w2[0], w2[1], 0.0],
                [w2[0], w2[1], height],
                [w1[0], w1[1], height]
            ]).T
            
            # Obstacle material (concrete by default, or wall_material if provided)
            obs_mat_name = wall_material if wall_material is not None else "concrete"
            obs_mat_coeffs = GAME_MATERIALS[obs_mat_name]
            obs_materials = pra.make_materials(obs=obs_mat_coeffs)
            obs_material_obj = obs_materials["obs"]
            
            # Match the number of bands of the outer walls
            if len(outer_walls[0].absorption) == 1:
                abs_val = float(np.mean(obs_material_obj.absorption_coeffs))
                scat_val = float(np.mean(obs_material_obj.scattering_coeffs))
                obs_material_obj = pra.Material(abs_val, scat_val)
            else:
                obs_material_obj.resample(temp_room.octave_bands)
            
            obstacle_wall = pra.Wall(
                corners,
                obs_material_obj.absorption_coeffs,
                obs_material_obj.scattering_coeffs,
                name="obstacle"
            )
            
            # Combine all walls
            all_walls = outer_walls + [obstacle_wall]
            
            # Build general Room
            room = pra.Room(
                all_walls,
                fs=fs,
                max_order=max_order,
                air_absorption=True
            )
            
            # Check if source is too close to the obstacle wall
            dist_source = dist_point_to_segment_2d(source_pos[:2], w1, w2)
            if dist_source < 0.3:
                # Nudge the source perpendicular to the wall
                wall_vec = w2 - w1
                perp_vec = np.array([-wall_vec[1], wall_vec[0]])
                perp_vec = perp_vec / np.linalg.norm(perp_vec)
                source_vec = source_pos[:2] - w1
                side = np.sign(np.dot(source_vec, perp_vec))
                if side == 0:
                    side = 1.0
                source_pos[0] += side * perp_vec[0] * (0.3 - dist_source)
                source_pos[1] += side * perp_vec[1] * (0.3 - dist_source)
                # Keep it inside the room boundaries
                source_pos[0] = np.clip(source_pos[0], margin, room_dim[0] - margin)
                source_pos[1] = np.clip(source_pos[1], margin, room_dim[1] - margin)
        else:
            # Fallback to standard ShoeBox if no safe wall was found
            if materials is None:
                room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, air_absorption=True)
            else:
                room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, materials=materials, air_absorption=True)
    else:
        if materials is None:
            room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, air_absorption=True)
        else:
            room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, materials=materials, air_absorption=True)
        
    # 6. Add source and binaural microphone array
    room.add_source(source_pos, signal=source_signal)
    
    # Both microphones are at the listener center, convolved with ear directivities
    room.add_microphone(listener_pos, directivity=dir_left)
    room.add_microphone(listener_pos, directivity=dir_right)
    
    # Optionally add an omnidirectional microphone at the head center for low-frequency crossover
    if crossover_freq > 0:
        room.add_microphone(listener_pos)
        
    # 7. Simulate propagation
    room.simulate()
    
    if visualize:
        print("Visualizing the room layout. Close the matplotlib window to proceed...")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        
        fig, ax = room.plot(plot_directivity=False)
        
        cmap = plt.get_cmap("YlGnBu")
        legend_elements = [
            Line2D([0], [0], marker='x', color='k', label='Listener (Mics)', linestyle='None', markersize=10),
            Line2D([0], [0], marker='o', markerfacecolor=cmap(1.0), markeredgecolor=cmap(1.0), label='Sound Source (Gun)', linestyle='None', markersize=10)
        ]
        if len(room.sources) > 0 and hasattr(room.sources[0], 'images') and room.sources[0].images.shape[1] > 1:
            legend_elements.append(
                Line2D([0], [0], marker='o', markerfacecolor=cmap(0.5), markeredgecolor=cmap(0.5), label='Image Sources (Reflections)', linestyle='None', markersize=8)
            )
            
        ax.legend(handles=legend_elements, loc='upper left')
        
        def fmt_mat(m):
            if isinstance(m, (float, int)):
                return f"abs={m:.2f}"
            return str(m)
            
        material_text = (
            f"Preset: {preset}\n"
            f"Materials:\n"
            f"  Walls (E/W/N/S): {fmt_mat(east_mat)} / {fmt_mat(west_mat)} / {fmt_mat(north_mat)} / {fmt_mat(south_mat)}\n"
            f"  Floor: {fmt_mat(floor_mat)}\n"
            f"  Ceiling: {fmt_mat(ceiling_mat)}"
        )
        
        ax.text2D(0.95, 0.95, material_text, transform=ax.transAxes,
                  horizontalalignment='right', verticalalignment='top',
                  fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
        
        plt.show()
    
    # 8. Extract microphone signals
    signals = room.mic_array.signals  # shape: (2, samples) or (3, samples)
    
    # Apply low-frequency crossover if active
    if crossover_freq > 0:
        import scipy.signal as signal
        nyq = 0.5 * fs
        if crossover_freq >= nyq:
            print(f"Warning: Crossover frequency {crossover_freq} Hz is above Nyquist frequency. Crossover disabled.")
            signals = signals[0:2]
        else:
            normal_cutoff = crossover_freq / nyq
            # Design 2nd-order Butterworth (filtfilt makes it 4th-order zero-phase)
            b_lp, a_lp = signal.butter(2, normal_cutoff, btype='low')
            b_hp, a_hp = signal.butter(2, normal_cutoff, btype='high')
            
            left_spatial = signals[0]
            right_spatial = signals[1]
            omni = signals[2]
            
            # Low-pass the omni signal (bass is identical/diffracted equally to both ears)
            low_pass = signal.filtfilt(b_lp, a_lp, omni)
            
            # High-pass the spatial signals (preserves directional ITD/ILD cues)
            high_left = signal.filtfilt(b_hp, a_hp, left_spatial)
            high_right = signal.filtfilt(b_hp, a_hp, right_spatial)
            
            # Combine to form high-fidelity binaural output
            signals = np.stack([low_pass + high_left, low_pass + high_right], axis=0)
            
    # Normalize output audio to avoid clipping
    max_val = np.max(np.abs(signals))
    if max_val > 0:
        signals = signals * (0.95 / max_val)
        
    return signals
 
def main():
    parser = argparse.ArgumentParser(description="Generate binaural gunfire dataset at 10-degree increments.")
    parser.add_argument(
        "--input-wav",
        type=str,
        default=None,
        help="Path to an input mono WAV file containing a real gunshot sound. If not specified, a synthetic gunshot is generated."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save the generated dataset audio files."
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=[
            "open_field",
            "indoor_room",
            "indoor_wood_room",
            "indoor_hall",
            "indoor_warehouse",
            "outdoor_street",
            "outdoor_grass",
            "outdoor_desert",
            "outdoor_snow",
            "corridor",
            "tunnel",
            "courtyard"
        ],
        default="indoor_hall",
        help="Acoustic environment preset."
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=2.0,
        help="Distance from the listener to the gunshot source in meters (default: 2.0)."
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=44100,
        help="Sampling rate of the output audio files in Hz (default: 44100)."
    )
    parser.add_argument(
        "--hrtf-name",
        type=str,
        default="mit_kemar_normal_pinna",
        help="Name of the SOFA HRTF file to use (default: mit_kemar_normal_pinna)."
    )
    parser.add_argument(
        "--interp-order",
        type=int,
        default=None,
        help="Order of spherical harmonics for HRTF interpolation (default: None, using nearest-neighbor to prevent frequency-response blurring/ringing)."
    )
    parser.add_argument(
        "--bgm-wav",
        type=str,
        default=None,
        help="Path to an optional WAV file containing background music or ambient noise to mix into the generated output."
    )
    parser.add_argument(
        "--bgm-snr",
        type=float,
        default=20.0,
        help="Signal-to-Noise Ratio (SNR) in dB for mixing the gunshot with the background music (default: 20.0 dB)."
    )
    parser.add_argument(
        "--crossover-freq",
        type=float,
        default=400.0,
        help="Crossover frequency in Hz to preserve low-frequency bass. Below this, the omnidirectional room response is used instead of the HRTF. Set to 0 to disable (default: 400.0 Hz)."
    )
    parser.add_argument(
        "--wall-material",
        type=str,
        default=None,
        choices=list(GAME_MATERIALS.keys()),
        help="Override wall material for the simulation."
    )
    parser.add_argument(
        "--floor-material",
        type=str,
        default=None,
        choices=list(GAME_MATERIALS.keys()),
        help="Override floor material for the simulation."
    )
    parser.add_argument(
        "--ceiling-material",
        type=str,
        default=None,
        choices=list(GAME_MATERIALS.keys()),
        help="Override ceiling material for the simulation."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the simulated room layout using matplotlib (plots only the first angle)."
    )
    parser.add_argument(
        "--obstacle",
        action="store_true",
        help="Randomly generate a vertical obstacle wall inside the room."
    )
    parser.add_argument(
        "--obstacle-seed",
        type=int,
        default=42,
        help="Random seed for obstacle generation (default: 42)."
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine the source signal
    if args.input_wav:
        print(f"Loading input gunshot sound from {args.input_wav}...")
        fs_in, source_signal = wavfile.read(args.input_wav)
        
        # Ensure signal is mono
        if len(source_signal.shape) > 1:
            print("Warning: Input audio is stereo, mixing down to mono...")
            source_signal = np.mean(source_signal, axis=1)
            
        # Convert integer to float
        if source_signal.dtype.kind in 'iu':
            source_signal = source_signal.astype(np.float32) / np.iinfo(source_signal.dtype).max
            
        # Resample if sample rate doesn't match target
        if fs_in != args.fs:
            print(f"Resampling source from {fs_in}Hz to {args.fs}Hz...")
            # We can use pyroomacoustics's internal resample helper
            from pyroomacoustics.utilities import resample
            source_signal = resample(source_signal, fs_in, args.fs)
    else:
        print("Generating synthetic gunshot sound...")
        source_signal = generate_synthetic_gunshot(args.fs)
        
    # Load background music / ambient noise if provided
    bgm_signal = None
    if args.bgm_wav:
        print(f"Loading background music from {args.bgm_wav}...")
        fs_bgm, bgm_raw = wavfile.read(args.bgm_wav)
        
        # Convert integer to float
        if bgm_raw.dtype.kind in 'iu':
            bgm_raw = bgm_raw.astype(np.float32) / np.iinfo(bgm_raw.dtype).max
            
        from pyroomacoustics.utilities import resample
        # Handle mono/stereo and resample each channel
        if len(bgm_raw.shape) > 1:
            bgm_left = bgm_raw[:, 0]
            bgm_right = bgm_raw[:, 1]
            if fs_bgm != args.fs:
                print(f"Resampling background music from {fs_bgm}Hz to {args.fs}Hz...")
                bgm_left = resample(bgm_left, fs_bgm, args.fs)
                bgm_right = resample(bgm_right, fs_bgm, args.fs)
            bgm_signal = np.stack([bgm_left, bgm_right], axis=1)
        else:
            if fs_bgm != args.fs:
                print(f"Resampling background music from {fs_bgm}Hz to {args.fs}Hz...")
                bgm_raw = resample(bgm_raw, fs_bgm, args.fs)
            bgm_signal = np.stack([bgm_raw, bgm_raw], axis=1)
        
    print(f"Starting dataset generation (preset: {args.preset}, distance: {args.distance}m)...")
    
    # Generate every 10 degrees (0, 10, 20, ..., 350)
    angles = list(range(0, 360, 10))
    for i, angle in enumerate(angles):
        print(f"Simulating azimuth {angle:03d} degrees...")
        signals = run_simulation(
            hrtf_path=args.hrtf_name,
            source_signal=source_signal,
            fs=args.fs,
            user_angle_deg=angle,
            distance=args.distance,
            preset=args.preset,
            interp_order=args.interp_order,
            crossover_freq=args.crossover_freq,
            wall_material=args.wall_material,
            floor_material=args.floor_material,
            ceiling_material=args.ceiling_material,
            visualize=(args.visualize and i == 0),
            obstacle=args.obstacle,
            obstacle_seed=args.obstacle_seed
        )
        
        if args.visualize:
            print("Visualization complete. Exiting without generating dataset WAV files.")
            import sys
            sys.exit(0)
        
        # signals is shape (2, N) where row 0 is left ear, row 1 is right ear.
        # Format as stereo (N, 2)
        stereo_signal = signals.T
        
        # Mix background music / ambient noise if provided
        if bgm_signal is not None:
            L_gun = len(stereo_signal)
            L_bgm = len(bgm_signal)
            if L_bgm >= L_gun:
                # Seed for reproducible random slice per angle
                np.random.seed(angle)
                start_idx = np.random.randint(0, L_bgm - L_gun + 1)
                bgm_slice = bgm_signal[start_idx : start_idx + L_gun, :]
            else:
                # Loop BGM to match length
                repeats = int(np.ceil(L_gun / L_bgm))
                bgm_looped = np.tile(bgm_signal, (repeats, 1))
                bgm_slice = bgm_looped[:L_gun, :]
                
            # Determine active gunshot region (amplitude > 5% of peak) to calculate SNR
            gun_abs = np.max(np.abs(stereo_signal), axis=1)
            peak_val = np.max(gun_abs)
            active_mask = gun_abs > (0.05 * peak_val)
            if np.sum(active_mask) > 0:
                active_gunshot = stereo_signal[active_mask]
            else:
                active_gunshot = stereo_signal
                
            rms_signal = np.sqrt(np.mean(active_gunshot**2))
            rms_noise = np.sqrt(np.mean(bgm_slice**2))
            
            if rms_noise > 0 and rms_signal > 0:
                # SNR = 20 * log10(rms_signal / target_rms_noise) => target_rms_noise = rms_signal * 10^(-SNR/20)
                target_rms_noise = rms_signal * (10 ** (-args.bgm_snr / 20.0))
                scale_factor = target_rms_noise / rms_noise
                bgm_mixed = bgm_slice * scale_factor
            else:
                bgm_mixed = bgm_slice * 0.0
                
            stereo_signal = stereo_signal + bgm_mixed
            
            # Prevent clipping
            peak_mixed = np.max(np.abs(stereo_signal))
            if peak_mixed > 1.0:
                stereo_signal = stereo_signal * (0.98 / peak_mixed)
        
        # Convert to 16-bit PCM integer
        pcm_signal = (stereo_signal * 32767).astype(np.int16)
        
        # Output filename
        filename = f"gunshot_az_{angle:03d}.wav"
        output_filepath = os.path.join(args.output_dir, filename)
        
        wavfile.write(output_filepath, args.fs, pcm_signal)
        
    print(f"\nSuccess! Dataset generated successfully. Output saved to: {os.path.abspath(args.output_dir)}")
    print(f"Total files generated: {len(angles)}")

if __name__ == "__main__":
    main()
