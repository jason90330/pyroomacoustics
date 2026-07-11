#!/usr/bin/env python3
"""
WAV Analysis and Spatial Audio Visualizer
Plots the time-domain waveform, spectrograms, cross-correlation (ITD), 
and frequency-dependent level differences (ILD) for spatialized audio files.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, correlate

def analyze_and_plot(wav_path, output_plot=None, show_plot=True):
    print(f"Loading WAV file: {wav_path}...")
    fs, data = wavfile.read(wav_path)
    
    # Convert integer PCM to float32
    if data.dtype.kind in 'iu':
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
        
    duration = len(data) / fs
    time = np.linspace(0, duration, len(data))
    
    # Handle Mono vs Stereo
    if len(data.shape) == 1:
        print("Input is Mono. Converting to pseudo-stereo for analysis...")
        left = data
        right = data
        is_stereo = False
    else:
        left = data[:, 0]
        right = data[:, 1]
        is_stereo = True

    # Initialize subplots (3 rows, 2 columns)
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Spatial Audio Analysis: {os.path.basename(wav_path)}", fontsize=16, fontweight='bold', y=0.98)
    
    # -------------------------------------------------------------
    # 1. TIME DOMAIN WAVEFORM
    # -------------------------------------------------------------
    axs[0, 0].plot(time * 1000, left, color='#1f77b4', alpha=0.8, label='Left Channel')
    axs[0, 0].set_title("Waveform - Left Channel", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Time (ms)")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].set_xlim(0, duration * 1000)
    
    axs[0, 1].plot(time * 1000, right, color='#ff7f0e', alpha=0.8, label='Right Channel')
    axs[0, 1].set_title("Waveform - Right Channel", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Time (ms)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].set_xlim(0, duration * 1000)

    # -------------------------------------------------------------
    # 2. SPECTROGRAMS
    # -------------------------------------------------------------
    nperseg = min(256, len(left))
    f_l, t_l, Sxx_l = spectrogram(left, fs, nperseg=nperseg)
    f_r, t_r, Sxx_r = spectrogram(right, fs, nperseg=nperseg)
    
    # Avoid log(0)
    Sxx_l_db = 10 * np.log10(Sxx_l + 1e-10)
    Sxx_r_db = 10 * np.log10(Sxx_r + 1e-10)
    
    # Left Spectrogram
    im_l = axs[1, 0].pcolormesh(t_l * 1000, f_l / 1000, Sxx_l_db, cmap='viridis', shading='gouraud', vmin=-80, vmax=0)
    axs[1, 0].set_title("Spectrogram - Left Channel", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Time (ms)")
    axs[1, 0].set_ylabel("Frequency (kHz)")
    axs[1, 0].set_ylim(0, fs / 2 / 1000)
    fig.colorbar(im_l, ax=axs[1, 0], format="%+2.0f dB")
    
    # Right Spectrogram
    im_r = axs[1, 1].pcolormesh(t_r * 1000, f_r / 1000, Sxx_r_db, cmap='viridis', shading='gouraud', vmin=-80, vmax=0)
    axs[1, 1].set_title("Spectrogram - Right Channel", fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel("Time (ms)")
    axs[1, 1].set_ylabel("Frequency (kHz)")
    axs[1, 1].set_ylim(0, fs / 2 / 1000)
    fig.colorbar(im_r, ax=axs[1, 1], format="%+2.0f dB")

    # -------------------------------------------------------------
    # 3. BINAURAL DIFFERENCE METRICS
    # -------------------------------------------------------------
    if is_stereo:
        # A. Interaural Time Difference (ITD) via Cross-Correlation
        # Compute normalized cross-correlation using fast FFT method
        corr = correlate(left - np.mean(left), right - np.mean(right), mode='same', method='fft')
        l_corr = len(corr)
        lags = np.arange(-l_corr // 2, l_corr // 2)
        lag_times_ms = lags * 1000.0 / fs
        
        # Max lag corresponding to typical head size (+- 1ms lag maximum)
        max_lag_ms = 1.2
        mask = (lag_times_ms >= -max_lag_ms) & (lag_times_ms <= max_lag_ms)
        
        # Plot Cross-Correlation
        axs[2, 0].plot(lag_times_ms[mask], corr[mask], color='#2ca02c', linewidth=2)
        
        # Find peak correlation
        peak_idx = np.argmax(corr[mask])
        itd_ms = lag_times_ms[mask][peak_idx]
        axs[2, 0].axvline(itd_ms, color='red', linestyle='--', alpha=0.8, 
                          label=f"Peak ITD: {itd_ms:+.3f} ms")
        
        axs[2, 0].set_title("Cross-Correlation (ITD Analysis)", fontsize=12, fontweight='bold')
        axs[2, 0].set_xlabel("Lag (ms)")
        axs[2, 0].set_ylabel("Correlation Coefficient")
        axs[2, 0].grid(True, linestyle='--', alpha=0.6)
        axs[2, 0].legend(loc='upper right')
        
        # B. Interaural Level Difference (ILD)
        # Compute average PSD difference across frequencies
        left_psd = np.mean(Sxx_l, axis=1)
        right_psd = np.mean(Sxx_r, axis=1)
        ild_db = 10 * np.log10((left_psd + 1e-10) / (right_psd + 1e-10))
        
        axs[2, 1].plot(f_l / 1000, ild_db, color='#9467bd', linewidth=2)
        axs[2, 1].axhline(0, color='black', linestyle=':', alpha=0.5)
        axs[2, 1].set_title("Interaural Level Difference (ILD)", fontsize=12, fontweight='bold')
        axs[2, 1].set_xlabel("Frequency (kHz)")
        axs[2, 1].set_ylabel("Level Difference (L - R) dB")
        axs[2, 1].grid(True, linestyle='--', alpha=0.6)
        axs[2, 1].set_xlim(0, fs / 2 / 1000)
        
        # Compute frequency statistics
        def compute_freq_stats(signal, fs):
            N = len(signal)
            fft_vals = np.fft.rfft(signal)
            fft_freqs = np.fft.rfftfreq(N, 1.0 / fs)
            psd = np.abs(fft_vals)**2
            
            peak_idx = np.argmax(psd)
            peak_freq = fft_freqs[peak_idx]
            
            sum_psd = np.sum(psd)
            centroid = np.sum(fft_freqs * psd) / sum_psd if sum_psd > 0 else 0.0
            
            hf_mask = fft_freqs >= 5000.0
            hf_energy = np.sum(psd[hf_mask])
            hf_ratio = (hf_energy / sum_psd) * 100.0 if sum_psd > 0 else 0.0
            
            return peak_freq, centroid, hf_ratio
            
        peak_f_l, centroid_l, hf_ratio_l = compute_freq_stats(left, fs)
        peak_f_r, centroid_r, hf_ratio_r = compute_freq_stats(right, fs)
        
        stats_text = (
            f"Frequency Statistics:\n"
            f"Left:\n"
            f"  Peak: {peak_f_l/1000:.2f} kHz\n"
            f"  Centroid: {centroid_l/1000:.2f} kHz\n"
            f"  HF Ratio: {hf_ratio_l:.1f}%\n"
            f"Right:\n"
            f"  Peak: {peak_f_r/1000:.2f} kHz\n"
            f"  Centroid: {centroid_r/1000:.2f} kHz\n"
            f"  HF Ratio: {hf_ratio_r:.1f}%"
        )
        props = dict(boxstyle='round', facecolor='#fcfcfc', edgecolor='#cccccc', alpha=0.9)
        axs[2, 1].text(0.97, 0.95, stats_text, transform=axs[2, 1].transAxes, fontsize=9,
                       fontfamily='monospace', verticalalignment='top', horizontalalignment='right', bbox=props)
    else:
        # Placeholder for mono analysis
        axs[2, 0].text(0.5, 0.5, "ITD Analysis requires a Stereo file", 
                      ha='center', va='center', fontsize=12, color='gray')
        axs[2, 0].set_title("Cross-Correlation (ITD Analysis)", fontsize=12, fontweight='bold')
        axs[2, 1].text(0.5, 0.5, "ILD Analysis requires a Stereo file", 
                      ha='center', va='center', fontsize=12, color='gray')
        axs[2, 1].set_title("Interaural Level Difference (ILD)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    # Save the output plot
    if output_plot is not None:
        print(f"Saving plot to: {output_plot}...")
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        
    if show_plot:
        print("Rendering interactive plot window...")
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="WAV Waveform, Spectrogram, ITD, and ILD Analyzer."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input WAV audio file to analyze."
    )
    parser.add_argument(
        "--output-plot", "-o",
        type=str,
        default=None,
        help="Path to save the generated analysis plot (optional)."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot interactively using matplotlib."
    )
    args = parser.parse_args()
    
    # If no output path is provided, default to placing it next to the input wav file
    output_plot = args.output_plot
    if output_plot is None:
        base, _ = os.path.splitext(args.input)
        output_plot = f"{base}_analysis.png"

    analyze_and_plot(
        wav_path=args.input,
        output_plot=output_plot,
        show_plot=not args.no_show
    )

if __name__ == "__main__":
    main()
