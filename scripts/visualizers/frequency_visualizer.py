# =============================================================================
# FREQUENCY_VISUALIZER
# =============================================================================
# Purpose:
#   Provides interactive and batch visualisation of spectral results for all
#   grid points. Tab 1 (Point Data): for any user-selected raw CSV grid-point
#   file, it plots the PSD of the transverse velocity fluctuation v' (computed
#   via Welch's method, displayed on a semi-logarithmic scale) with the
#   dominant peak annotated; a batch mode saves individual PSD figures for all
#   grid points using a memory-safe object-oriented Matplotlib API (zero pyplot
#   state leakage). A separate function generates a superposed PSD plot for
#   all points along the closest-to-centerline row, with staggered offsets and
#   global mean frequency marked, plus companion St-vs-X/B and f_dom-vs-X/B
#   centerline profile figures.
#
# Inputs:
#   - Raw per-point velocity time-series CSV files:
#       experiments/<case>/Processed_CSVs/Raw_Data/*.csv
#   - Per-point PSD data:
#       experiments/<case>/Processed_CSVs/FFT/<grid>_FFT.csv
#   - Centerline frequency profile (for St / f_dom annotations and profiles):
#       experiments/<case>/1D_Profiles_Results/CSV/Centerline_Freq_Profile.csv
#   - Global summary:
#       experiments/<case>/Frequency_Results/CSV/Global_Frequency_Summary.csv
#
# Outputs:
#   - Per-point PSD figures (PNG and PDF):
#       experiments/<case>/Frequency_Results/FFT_Figure/
#   - Centerline superposed PSD figure:
#       experiments/<case>/Frequency_Results/FFT_Figure/Centerline_Superposed_PSD.png/.pdf
#   - Centerline St and f_dom profile figures (PNG and PDF):
#       experiments/<case>/Frequency_Results/PNG|PDF/Centerline_St_Profile.*
#       experiments/<case>/Frequency_Results/PNG|PDF/Centerline_Freq_Profile.*
#
# Dependencies:
#   - frequency_processor (produces PSD CSVs and the centerline profile CSV)
#   - Imported by pipeline.py (Step 6) and usage.py
#
# Usage:
#   - Standalone: python frequency_visualizer.py  (opens Tkinter GUI)
#   - Via pipeline: called programmatically by pipeline.py (Step 6)
#   - Via hub:      launched from usage.py as "Time & Frequency Visualizer"
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy import signal
import threading
import gc

# --- Professional Plotting & LaTeX Configuration ---
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", 
    "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 300, 
})

class FrequencyVisualizerApp:
    def __init__(self, root, default_folder=None):
        self.root = root
        self.root.title("Time & Frequency Visualizer (FFT) - Zero Leak Mode")
        self.root.geometry("1150x850")
        self.folder_path = tk.StringVar(value=default_folder if default_folder else "")
        self.time_files = []
        self.is_processing = False
        
        # --- Cache Systems ---
        self.t1_types = ['psd']
        self.t1_frames = {pt: None for pt in self.t1_types}
        self.t1_figs = {pt: None for pt in self.t1_types}
        
        self.setup_ui()
        if self.folder_path.get(): self.scan_folder()

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Main Data Folder")
        if folder:
            if self.folder_path.get() != folder:
                self.folder_path.set(folder)
                self.scan_folder()

    def scan_folder(self):
        raw_folder = os.path.join(self.folder_path.get(), "Processed_CSVs", "Raw_Data")
        if not os.path.exists(raw_folder):
            raw_folder = os.path.join(self.folder_path.get(), "Processed_CSVs")
            
        if os.path.exists(raw_folder):
            self.time_files = glob.glob(os.path.join(raw_folder, "*.csv"))
            file_names = [os.path.basename(f) for f in self.time_files]
            if file_names:
                self.combo_point['values'] = file_names
                self.combo_point.current(0)
                self.clear_t1_cache()

    # --- HELPERS ---
    def save_figure_files(self, fig, save_dir_base, filename_base):
        png_dir = os.path.join(save_dir_base, "PNG")
        pdf_dir = os.path.join(save_dir_base, "PDF")
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)
        
        fig.savefig(os.path.join(png_dir, f"{filename_base}.png"), dpi=300, bbox_inches='tight')
        
        original_titles = []
        for ax in fig.axes:
            original_titles.append(ax.get_title())
            ax.set_title("")
            
        fig.savefig(os.path.join(pdf_dir, f"{filename_base}.pdf"), format='pdf', bbox_inches='tight')
        
        for ax, title in zip(fig.axes, original_titles):
            ax.set_title(title)

    def _get_peak_threshold(self):
        if not self.folder_path.get(): return 75.0
        folder_name = os.path.basename(os.path.normpath(self.folder_path.get()))
        return 400.0 if 'Cyl12' in folder_name else 75.0

    def _get_global_means(self):
        if not self.folder_path.get(): return None, None
        summary_path = os.path.join(self.folder_path.get(), "Frequency_Results", "CSV", "Global_Frequency_Summary.csv")
        if os.path.exists(summary_path):
            try:
                df = pd.read_csv(summary_path)
                return df['Global_Mean_Freq_Hz'].iloc[0], df['Global_Mean_St'].iloc[0]
            except Exception: pass
        return None, None

    # ==========================================
    # TAB 1: POINT FFT (PSD)
    # ==========================================
    def on_point_selected(self, event=None):
        self.clear_t1_cache()

    def clear_t1_cache(self):
        for pt in self.t1_types:
            if self.t1_frames[pt]: self.t1_frames[pt].destroy()
            if self.t1_figs[pt]: plt.close(self.t1_figs[pt])
            self.t1_frames[pt] = None
            self.t1_figs[pt] = None
        self.hide_all_t1()
        self.t1_placeholder.pack(expand=True)

    def hide_all_t1(self):
        self.t1_placeholder.pack_forget()
        for frame in self.t1_frames.values():
            if frame: frame.pack_forget()

    def generate_t1_plot(self, plot_type):
        point_file = self.combo_point.get()
        if not point_file: return False
        
        target_path = next((f for f in self.time_files if os.path.basename(f) == point_file), None)
        if not target_path: return False

        try:
            df = pd.read_csv(target_path)
            time = df['Time (s)'].values
            v_vel = df['v (m/s)'].values
            fs = 1.0 / (time[1] - time[0])
            v_fluct = v_vel - np.mean(v_vel)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if plot_type == 'psd':
                center_files = []
                closest_y = None
                
                for f in self.time_files:
                    try:
                        temp_df = pd.read_csv(f, nrows=1)
                        if 'Y/B' in temp_df.columns:
                            y_val = temp_df['Y/B'].iloc[0]
                            if closest_y is None or abs(y_val - 0) < abs(closest_y - 0):
                                closest_y = y_val
                    except Exception: pass
                            
                if closest_y is not None:
                    for f in self.time_files:
                        try:
                            temp_df = pd.read_csv(f, nrows=1)
                            if 'Y/B' in temp_df.columns and abs(temp_df['Y/B'].iloc[0] - closest_y) < 1e-4:
                                center_files.append((temp_df['X/B'].iloc[0], f))
                        except Exception: pass
                        
                if not center_files:
                    center_files = [(0, target_path)]
                    
                center_files.sort(key=lambda x: x[0])
                colors = plt.cm.viridis(np.linspace(0, 1, len(center_files)))
                # Task 15C — load per-station St for label annotation
                st_map_interactive = self._load_centerline_st_map()

                for idx, (x_val, filepath) in enumerate(center_files):
                    df_c = pd.read_csv(filepath)
                    time_c = df_c['Time (s)'].values
                    v_fluct_c = df_c['v (m/s)'].values - np.mean(df_c['v (m/s)'].values)
                    fs_c = 1.0 / (time_c[1] - time_c[0])

                    freqs, psd = signal.welch(v_fluct_c, fs=fs_c, nperseg=min(len(v_fluct_c), 2048))

                    # Build label with optional St annotation
                    if st_map_interactive:
                        closest_key = min(st_map_interactive.keys(), key=lambda k: abs(k - x_val))
                        st_val_i = st_map_interactive[closest_key] if abs(closest_key - x_val) < 0.05 else None
                    else:
                        st_val_i = None
                    if st_val_i is not None and st_val_i > 0:
                        lbl = rf'$X/B = {x_val:.2f}$  ($St = {st_val_i:.2f}$)'
                    else:
                        lbl = rf'$X/B = {x_val:.2f}$'

                    offset = 10**(idx * 0.5) if len(center_files) > 1 else 1.0
                    ax.semilogy(freqs, psd * offset, color=colors[idx], lw=1.2, alpha=0.8, label=lbl)
                
                mean_freq, mean_st = self._get_global_means()
                if mean_freq and mean_freq > 0:
                    ax.axvline(mean_freq, color='red', linestyle='--', lw=1.5, alpha=0.7)
                    # Text box at top left of the line
                    ax.text(mean_freq, 0.95, f"{mean_freq:.1f} Hz\n(St: {mean_st:.2f})",
                            transform=ax.get_xaxis_transform(), color='red', fontsize=9,
                            ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

                ax.set_title(rf'Centerline PSD ($Y/B \approx {closest_y if closest_y is not None else 0:.2f}$)')
                ax.set_xlabel(r'$f$ [Hz]')
                ax.set_ylabel(r'PSD [m$^2$/s$^2$/Hz] (staggered)')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(left=0, right=1000)
                ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
            
                filename = "Centerline_Superposed_PSD"
                subfolder = "FFT_Figure"

            save_dir = os.path.join(self.folder_path.get(), "Frequency_Results", subfolder)
            self.save_figure_files(fig, save_dir, filename)

            frame = tk.Frame(self.t1_canvas_frame, bg="white")
            self.t1_frames[plot_type] = frame
            self.t1_figs[plot_type] = fig

            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            NavigationToolbar2Tk(canvas, frame).update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            return True

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate {plot_type}: {e}")
            return False

    def trigger_t1_plot(self, plot_type):
        if not self.folder_path.get(): return
        if self.t1_frames[plot_type] is None:
            if not self.generate_t1_plot(plot_type): return
        
        self.hide_all_t1()
        self.t1_frames[plot_type].pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def save_current_t1(self):
        if not self.folder_path.get() or not self.combo_point.get(): return
        
        self.hide_all_t1()
        self.t1_placeholder.config(text=f"[ Saving figures for {self.combo_point.get()}... ]")
        self.t1_placeholder.pack(expand=True)
        self.root.update_idletasks()
        
        if self.t1_frames['psd'] is None: self.generate_t1_plot('psd')

        self.t1_placeholder.config(text="[ Success! Current point figures saved. ]\n\nClick the button above to view it.")

    def save_all_t1_points(self):
        if not self.folder_path.get() or not self.time_files: return
        if self.is_processing: return
        
        self.is_processing = True
        self.hide_all_t1()
        self.t1_placeholder.pack(expand=True)
        
        threading.Thread(target=self._run_sequential_batch, daemon=True).start()

    def _run_sequential_batch(self):
        total = len(self.time_files)
        save_dir_base = os.path.join(self.folder_path.get(), "Frequency_Results")
        peak_threshold = self._get_peak_threshold()
        
        fft_dir = os.path.join(save_dir_base, "FFT_Figure")
        os.makedirs(os.path.join(fft_dir, "PNG"), exist_ok=True)
        os.makedirs(os.path.join(fft_dir, "PDF"), exist_ok=True)

        self.root.after(0, lambda: self.t1_placeholder.config(text=f"[ Starting Batch Process... ]\n\nProcessing {total} files using Zero-Leak Object API."))

        for i, target_path in enumerate(self.time_files):
            point_file = os.path.basename(target_path)
            
            self.root.after(0, lambda c=i+1, t=total, f=point_file: self.t1_placeholder.config(
                text=f"[ Processing {c} / {t} ]\n\nGenerating figures for:\n{f}\n\nRAM usage will remain perfectly flat."
            ))

            try:
                df = pd.read_csv(target_path)
                time = df['Time (s)'].values
                v_vel = df['v (m/s)'].values
                fs = 1.0 / (time[1] - time[0])
                v_fluct = v_vel - np.mean(v_vel)

                # --- Retrieve spatial coordinates for this point ---
                xb = df['X/B'].iloc[0] if 'X/B' in df.columns else float('nan')
                yb = df['Y/B'].iloc[0] if 'Y/B' in df.columns else float('nan')

                # --- 1. PSD (Object Oriented API - No Pyplot) ---
                fig_psd = Figure(figsize=(8, 6))
                canvas_psd = FigureCanvasAgg(fig_psd)
                ax_psd = fig_psd.add_subplot(111)

                freqs, psd = signal.welch(v_fluct, fs=fs, nperseg=min(len(v_fluct), 2048))
                ax_psd.semilogy(freqs, psd, color='#6f42c1', lw=1.5)

                # Mark the dominant peak ABOVE the case noise-floor threshold
                # (peak_threshold_hz: 400 Hz Cyl12 / 75 Hz free jet) — identical to
                # the detection in frequency_processor, so per-point markers and the
                # global mean agree.
                valid_idx = np.where(freqs > peak_threshold)[0]
                if len(valid_idx) > 0:
                    dom_idx = valid_idx[np.argmax(psd[valid_idx])]
                    ax_psd.plot(freqs[dom_idx], psd[dom_idx], 'ro', ms=8)

                # NOTE: Intentionally removed the global mean line here per instructions.

                # Title and filename use actual spatial coordinates (Task 11)
                if not (np.isnan(xb) or np.isnan(yb)):
                    ax_psd.set_title(rf'PSD of $v^\prime$ at $X/B = {xb:.2f}$, $Y/B = {yb:.2f}$')
                    name_psd = f"PSD_XB{xb:.2f}_YB{yb:.2f}"
                else:
                    ax_psd.set_title(r'PSD of $v^\prime$')
                    name_psd = point_file.replace('.csv', '_PSD')
                ax_psd.set_xlabel(r'$f$ [Hz]')
                ax_psd.set_ylabel(r'PSD [m$^2$/s$^2$/Hz]')
                ax_psd.grid(True, alpha=0.3)
                ax_psd.set_xlim(left=0, right=min(fs/2, 2000))

                self.save_figure_files(fig_psd, fft_dir, name_psd)
                
                fig_psd.clear()
                del fig_psd, canvas_psd, ax_psd

                gc.collect()

            except Exception as e:
                print(f"Error on {point_file}: {e}")

        self.is_processing = False
        self.root.after(0, lambda: self.t1_placeholder.config(
            text=f"[ Success! All {total} points exported safely. ]\n\nCheck the 'Frequency_Results' folder."
        ))

    # ==========================================
    # CENTERLINE PROFILES & SUPERPOSED PSD
    # ==========================================
    def _load_centerline_st_map(self):
        """Return a dict {xb_float: st_float} from Centerline_Freq_Profile.csv, or {}."""
        if not self.folder_path.get():
            return {}
        csv_path = os.path.join(
            self.folder_path.get(),
            "1D_Profiles_Results", "CSV", "Centerline_Freq_Profile.csv"
        )
        if not os.path.exists(csv_path):
            return {}
        try:
            df = pd.read_csv(csv_path)
            if 'X/B' not in df.columns or 'Strouhal_Number' not in df.columns:
                return {}
            return dict(zip(df['X/B'].values, df['Strouhal_Number'].values))
        except Exception:
            return {}

    def save_centerline_superposed_psd(self):
        if not self.folder_path.get() or not self.time_files: return

        try:
            self.t1_placeholder.config(text="[ Generating Centerline Superposed PSD... ]")
            self.root.update_idletasks()

            peak_threshold = self._get_peak_threshold()
            mean_freq, mean_st = self._get_global_means()
            # Task 15C — load per-station St values for label annotation
            st_map = self._load_centerline_st_map()
            center_files = []
            closest_y = None

            for f in self.time_files:
                try:
                    temp_df = pd.read_csv(f, nrows=1)
                    if 'Y/B' in temp_df.columns:
                        y_val = temp_df['Y/B'].iloc[0]
                        if closest_y is None or abs(y_val - 0) < abs(closest_y - 0):
                            closest_y = y_val
                except Exception: pass

            if closest_y is not None:
                for f in self.time_files:
                    try:
                        temp_df = pd.read_csv(f, nrows=1)
                        if 'Y/B' in temp_df.columns and abs(temp_df['Y/B'].iloc[0] - closest_y) < 1e-4:
                            center_files.append((temp_df['X/B'].iloc[0], f))
                    except Exception: pass

            if not center_files: return

            center_files.sort(key=lambda x: x[0])

            fig = Figure(figsize=(10, 6))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            colors = plt.cm.viridis(np.linspace(0, 1, len(center_files)))

            for idx, (x_val, filepath) in enumerate(center_files):
                df_c = pd.read_csv(filepath)
                time_c = df_c['Time (s)'].values
                v_fluct_c = df_c['v (m/s)'].values - np.mean(df_c['v (m/s)'].values)
                fs_c = 1.0 / (time_c[1] - time_c[0])

                freqs, psd = signal.welch(v_fluct_c, fs=fs_c, nperseg=min(len(v_fluct_c), 2048))

                # Task 15C — find closest X/B key in st_map and append St to label
                if st_map:
                    closest_key = min(st_map.keys(), key=lambda k: abs(k - x_val))
                    st_val = st_map[closest_key] if abs(closest_key - x_val) < 0.05 else None
                else:
                    st_val = None
                if st_val is not None and st_val > 0:
                    label = rf'$X/B = {x_val:.2f}$  ($St = {st_val:.2f}$)'
                else:
                    label = rf'$X/B = {x_val:.2f}$'

                ax.semilogy(freqs, psd, color=colors[idx], lw=1.2, alpha=0.8, label=label)

                valid_idx = np.where(freqs > peak_threshold)[0]
                if len(valid_idx) > 0:
                    peak_idx = valid_idx[np.argmax(psd[valid_idx])]
                    ax.plot(freqs[peak_idx], psd[peak_idx], 'o', color=colors[idx], ms=6, markeredgecolor='black', markeredgewidth=0.5)

            if mean_freq and mean_freq > 0:
                ax.axvline(mean_freq, color='red', linestyle='--', lw=1.5, alpha=0.7)
                ax.text(mean_freq, 0.95, f"{mean_freq:.1f} Hz\n(St: {mean_st:.2f})",
                        transform=ax.get_xaxis_transform(), color='red', fontsize=9,
                        ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

            ax.set_title(rf'Centerline PSD ($Y/B \approx {closest_y:.2f}$)')
            ax.set_xlabel(r'$f$ [Hz]')
            ax.set_ylabel(r'PSD [m$^2$/s$^2$/Hz]')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0, right=1000)
            ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)

            save_dir = os.path.join(self.folder_path.get(), "Frequency_Results", "FFT_Figure")
            self.save_figure_files(fig, save_dir, "Centerline_Superposed_PSD")

            fig.clear()
            del fig, canvas, ax
            gc.collect()

            # Task 15B — companion St vs X/B profile plot
            self.save_centerline_st_profile()
            # Task 10 — companion f_dom vs X/B profile plot
            self.save_centerline_freq_profile()

        except Exception as e:
            print(f"Error superposing PSD: {e}")

    def save_centerline_st_profile(self):
        """Task 15B — Plot St vs X/B along the centerline (Y/B ≈ 0) and save."""
        if not self.folder_path.get():
            return

        csv_path = os.path.join(
            self.folder_path.get(),
            "1D_Profiles_Results", "CSV", "Centerline_Freq_Profile.csv"
        )

        if not os.path.exists(csv_path):
            messagebox.showinfo(
                "Not available",
                "Centerline_Freq_Profile.csv not found.\n\n"
                "Run the Frequency Processor first to generate it."
            )
            return

        try:
            df = pd.read_csv(csv_path)

            if 'X/B' not in df.columns or 'Strouhal_Number' not in df.columns:
                messagebox.showerror(
                    "Error",
                    "Centerline_Freq_Profile.csv is missing 'X/B' or 'Strouhal_Number' columns."
                )
                return

            df = df.sort_values('X/B')
            xb_vals = df['X/B'].values
            st_vals = df['Strouhal_Number'].values

            fig = Figure(figsize=(8, 5))
            canvas_obj = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            ax.plot(xb_vals, st_vals, 'o-', color='#6f42c1', lw=1.8, ms=5,
                    markeredgecolor='black', markeredgewidth=0.5)

            # Annotate mean St if at least one non-zero value exists
            valid_st = st_vals[st_vals > 0]
            if len(valid_st) > 0:
                mean_st = np.mean(valid_st)
                ax.axhline(mean_st, color='red', linestyle='--', lw=1.2, alpha=0.8,
                           label=rf'$\overline{{St}} = {mean_st:.3f}$')
                ax.legend(fontsize=10)

            ax.set_xlabel(r'$X/B$ [-]')
            ax.set_ylabel(r'$St$ [-]')
            ax.set_title(r'Strouhal Number $St$ along Centerline ($Y/B \approx 0$)')
            ax.grid(True, alpha=0.3)

            save_dir = os.path.join(self.folder_path.get(), "Frequency_Results", "PNG")
            pdf_dir  = os.path.join(self.folder_path.get(), "Frequency_Results", "PDF")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(pdf_dir, exist_ok=True)

            fig.savefig(os.path.join(save_dir, "Centerline_St_Profile.png"),
                        dpi=300, bbox_inches='tight')

            # PDF version: strip title
            ax.set_title("")
            fig.savefig(os.path.join(pdf_dir, "Centerline_St_Profile.pdf"),
                        format='pdf', bbox_inches='tight')
            ax.set_title(r'Strouhal Number $St$ along Centerline ($Y/B \approx 0$)')

            fig.clear()
            del fig, canvas_obj, ax
            gc.collect()

            messagebox.showinfo("Saved", "Centerline_St_Profile.png/.pdf saved to Frequency_Results/.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate Centerline St Profile: {e}")

    def save_centerline_freq_profile(self):
        """Task 10 — Plot dominant frequency f_dom vs X/B along the centerline (Y/B ≈ 0) and save."""
        if not self.folder_path.get():
            return

        csv_path = os.path.join(
            self.folder_path.get(),
            "1D_Profiles_Results", "CSV", "Centerline_Freq_Profile.csv"
        )

        if not os.path.exists(csv_path):
            messagebox.showinfo(
                "Not available",
                "Centerline_Freq_Profile.csv not found.\n\n"
                "Run the Frequency Processor first to generate it."
            )
            return

        try:
            df = pd.read_csv(csv_path)

            # Column name check — processor writes 'Mean_Resonance_Hz'; fall back
            # to 'Dominant_Freq_Hz' if this CSV was produced by an older version.
            if 'Mean_Resonance_Hz' in df.columns:
                freq_col = 'Mean_Resonance_Hz'
            elif 'Dominant_Freq_Hz' in df.columns:
                freq_col = 'Dominant_Freq_Hz'
            else:
                messagebox.showerror(
                    "Error",
                    "Centerline_Freq_Profile.csv has no recognised frequency column\n"
                    "('Mean_Resonance_Hz' or 'Dominant_Freq_Hz')."
                )
                return

            if 'X/B' not in df.columns:
                messagebox.showerror(
                    "Error",
                    "Centerline_Freq_Profile.csv is missing the 'X/B' column."
                )
                return

            df = df.sort_values('X/B')
            xb_vals  = df['X/B'].values
            freq_vals = df[freq_col].values

            fig = Figure(figsize=(8, 5))
            canvas_obj = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            ax.plot(xb_vals, freq_vals, 'o-', color='#e07b00', lw=1.8, ms=5,
                    markeredgecolor='black', markeredgewidth=0.5)

            # Annotate mean frequency if at least one non-zero value exists
            valid_f = freq_vals[freq_vals > 10.0]
            if len(valid_f) > 0:
                mean_f = np.mean(valid_f)
                ax.axhline(mean_f, color='red', linestyle='--', lw=1.2, alpha=0.8,
                           label=rf'$\overline{{f_{{dom}}}} = {mean_f:.1f}$ Hz')
                ax.legend(fontsize=10)

            ax.set_xlabel(r'$X/B$ [-]')
            ax.set_ylabel(r'$f_{dom}$ [Hz]')
            ax.set_title(r'Dominant Frequency $f_{dom}$ along Centerline ($Y/B \approx 0$)')
            ax.grid(True, alpha=0.3)

            save_dir = os.path.join(self.folder_path.get(), "Frequency_Results", "PNG")
            pdf_dir  = os.path.join(self.folder_path.get(), "Frequency_Results", "PDF")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(pdf_dir, exist_ok=True)

            fig.savefig(os.path.join(save_dir, "Centerline_Freq_Profile.png"),
                        dpi=300, bbox_inches='tight')

            # PDF version: strip title
            ax.set_title("")
            fig.savefig(os.path.join(pdf_dir, "Centerline_Freq_Profile.pdf"),
                        format='pdf', bbox_inches='tight')
            ax.set_title(r'Dominant Frequency $f_{dom}$ along Centerline ($Y/B \approx 0$)')

            fig.clear()
            del fig, canvas_obj, ax
            gc.collect()

            messagebox.showinfo("Saved", "Centerline_Freq_Profile.png/.pdf saved to Frequency_Results/.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate Centerline Freq Profile: {e}")

    def setup_ui(self):
        ctrl = tk.Frame(self.root, bg="#e0e0e0", bd=2, relief="groove")
        ctrl.pack(fill="x", padx=10, pady=10)
        tk.Label(ctrl, text="Global Workspace:", bg="#e0e0e0", font=("Arial", 10, "bold")).pack(side="left", padx=10, pady=10)
        tk.Entry(ctrl, textvariable=self.folder_path, state="readonly", width=50).pack(side="left", padx=5)
        tk.Button(ctrl, text="Browse...", command=self.select_folder).pack(side="left", padx=5)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=5)

        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="  1. Time-Frequency (Point Data)  ")
        
        t1_left = tk.Frame(tab1, width=220, bg="#f4f4f4")
        t1_left.pack(side="left", fill="y", padx=10, pady=10)
        
        tk.Label(t1_left, text="Select Grid Point File:", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(pady=(10,5))
        self.combo_point = ttk.Combobox(t1_left, state="readonly", width=25)
        self.combo_point.pack(fill="x", padx=5, pady=(0,15))
        self.combo_point.bind('<<ComboboxSelected>>', self.on_point_selected)

        tk.Button(t1_left, text="Plot PSD (FFT)", command=lambda: self.trigger_t1_plot('psd'), bg="#6f42c1", fg="white", height=2).pack(fill="x", pady=3)
        
        tk.Frame(t1_left, height=2, bg="#ccc").pack(fill="x", pady=15)
        
        tk.Button(t1_left, text="Save Current Point", command=self.save_current_t1, bg="#343a40", fg="white", height=2, font=("Arial", 10, "bold")).pack(fill="x", pady=3)
        tk.Button(t1_left, text="Save ALL Points (Safe Mode)", command=self.save_all_t1_points, bg="#dc3545", fg="white", height=2, font=("Arial", 10, "bold")).pack(fill="x", pady=3)

        tk.Frame(t1_left, height=2, bg="#ccc").pack(fill="x", pady=10)
        tk.Label(t1_left, text="Centerline Profiles", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(pady=(0, 5))
        tk.Button(t1_left, text="St vs X/B (Centerline)", command=self.save_centerline_st_profile, bg="#007acc", fg="white", height=2).pack(fill="x", pady=3)
        tk.Button(t1_left, text="f_dom vs X/B (Centerline)", command=self.save_centerline_freq_profile, bg="#e07b00", fg="white", height=2).pack(fill="x", pady=3)

        self.t1_canvas_frame = tk.Frame(tab1, bg="white", relief="sunken", bd=2)
        self.t1_canvas_frame.pack(side="right", fill="both", expand=True, padx=(0, 10), pady=10)
        self.t1_placeholder = tk.Label(self.t1_canvas_frame, text="[ Point Data Render Area ]\n\nGenerated graphs will appear here.", bg="white", fg="#888", font=("Arial", 14))
        self.t1_placeholder.pack(expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = FrequencyVisualizerApp(root)
    root.mainloop()