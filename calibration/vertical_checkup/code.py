import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
# Fetch the directory where THIS script is saved, not where the terminal is open
DATA_FOLDER = os.path.dirname(os.path.abspath(__file__)) 
EXCEL_OUTPUT = "Nozzle_Flow_Data.xlsx"
PLOT_OUTPUT = "Nozzle_Flow_Profile.png"
SPATIAL_STEP_MM = 2  # Measurement taken every 2mm

# ==========================================
# PARSER FUNCTION
# ==========================================
def parse_asa_file(filepath):
    """Reads a Cobra Probe .asA file and extracts key flow parameters."""
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()

    data = {}
    
    # Extract position from filename (e.g., "0001 (Ve...)" -> index 1 -> 2mm)
    filename = os.path.basename(filepath)
    match_index = re.search(r'^(\d{4})', filename)
    if match_index:
        index = int(match_index.group(1))
        data['Position (mm)'] = index * SPATIAL_STEP_MM
    else:
        return None # Skip files that don't match the 00XX naming convention

    # Extract Mean flow speed, pitch, yaw, and Pstatic
    flow_match = re.search(r'Mean flow speed, pitch angle, yaw angle and Pstatic[^\n]*\n\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)', content)
    if flow_match:
        data['Mean Flow Speed (m/s)'] = float(flow_match.group(1))
        data['Pitch Angle (°)'] = float(flow_match.group(2))
        data['Yaw Angle (°)'] = float(flow_match.group(3))
        data['Pstatic (Pa)'] = float(flow_match.group(4))

    # Extract Mean U, V and W
    uvw_match = re.search(r'Mean U, V and W[^\n]*\n\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)', content)
    if uvw_match:
        data['U (m/s)'] = float(uvw_match.group(1))
        data['V (m/s)'] = float(uvw_match.group(2))
        data['W (m/s)'] = float(uvw_match.group(3))

    # Extract Turbulence intensities
    turb_match = re.search(r'Turbulence intensities - overall, Iuu, Ivv, Iww[^\n]*\n\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)', content)
    if turb_match:
        data['Turbulence Overall (%)'] = float(turb_match.group(1))
        data['Iuu (%)'] = float(turb_match.group(2))
        data['Ivv (%)'] = float(turb_match.group(3))
        data['Iww (%)'] = float(turb_match.group(4))

    return data

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # Look for .asA files in the script's directory
    search_pattern = os.path.join(DATA_FOLDER, "*.asA")
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        print(f"No .asA files found in the directory: {DATA_FOLDER}")
        return

    print(f"Found {len(file_list)} files in {DATA_FOLDER}. Parsing data...")
    
    parsed_data = []
    for filepath in file_list:
        file_data = parse_asa_file(filepath)
        if file_data:
            parsed_data.append(file_data)
            
    # Create DataFrame and sort by spatial position
    df = pd.DataFrame(parsed_data)
    df = df.sort_values(by='Position (mm)').reset_index(drop=True)
    
    # Export to Excel
    df.to_excel(EXCEL_OUTPUT, index=False)
    print(f"Data successfully exported to {EXCEL_OUTPUT}")

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle('Nozzle Exit Flow Profile (30mm Slice)', fontsize=16, fontweight='bold')

    # Plot 1: All Velocity Components
    axs[0, 0].plot(df['Position (mm)'], df['Mean Flow Speed (m/s)'], marker='o', label='Mean Speed', color='black', linewidth=2)
    axs[0, 0].plot(df['Position (mm)'], df['U (m/s)'], marker='s', label='U (Streamwise)', linestyle='--')
    axs[0, 0].plot(df['Position (mm)'], df['V (m/s)'], marker='^', label='V (Vertical)', linestyle='--')
    axs[0, 0].plot(df['Position (mm)'], df['W (m/s)'], marker='v', label='W (Lateral)', linestyle='--')
    axs[0, 0].set_ylabel('Velocity (m/s)')
    axs[0, 0].set_title('Velocity Components')
    axs[0, 0].grid(True, linestyle=':', alpha=0.7)
    axs[0, 0].legend()

    # Plot 2: Flow Angles
    axs[0, 1].plot(df['Position (mm)'], df['Pitch Angle (°)'], marker='^', label='Pitch Angle', color='red')
    axs[0, 1].plot(df['Position (mm)'], df['Yaw Angle (°)'], marker='v', label='Yaw Angle', color='blue')
    axs[0, 1].set_ylabel('Angle (°)')
    axs[0, 1].set_title('Flow Angularity')
    axs[0, 1].grid(True, linestyle=':', alpha=0.7)
    axs[0, 1].legend()

    # Plot 3: All Turbulence Intensities
    axs[1, 0].plot(df['Position (mm)'], df['Turbulence Overall (%)'], marker='o', color='purple', label='Overall', linewidth=2)
    axs[1, 0].plot(df['Position (mm)'], df['Iuu (%)'], marker='s', label='Iuu', linestyle=':')
    axs[1, 0].plot(df['Position (mm)'], df['Ivv (%)'], marker='^', label='Ivv', linestyle=':')
    axs[1, 0].plot(df['Position (mm)'], df['Iww (%)'], marker='v', label='Iww', linestyle=':')
    axs[1, 0].set_xlabel('Position across nozzle (mm)')
    axs[1, 0].set_ylabel('Turbulence Intensity (%)')
    axs[1, 0].set_title('Turbulence Profiles')
    axs[1, 0].grid(True, linestyle=':', alpha=0.7)
    axs[1, 0].legend()

    # Plot 4: Static Pressure
    axs[1, 1].plot(df['Position (mm)'], df['Pstatic (Pa)'], marker='d', color='darkred', label='Static Pressure')
    axs[1, 1].set_xlabel('Position across nozzle (mm)')
    axs[1, 1].set_ylabel('Pressure (Pa)')
    axs[1, 1].set_title('Static Pressure Profile')
    axs[1, 1].grid(True, linestyle=':', alpha=0.7)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Adjust to fit the main title
    
    # Save the figure
    plt.savefig(PLOT_OUTPUT, dpi=300)
    print(f"Visualizations saved to {PLOT_OUTPUT}")
    
    # Show the plot briefly if running interactively
    plt.show()

if __name__ == "__main__":
    main()