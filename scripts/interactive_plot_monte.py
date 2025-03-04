import numpy as np
import plotly.graph_objects as go
from B8_project import file_reading
from B8_project.crystal import UnitCell
from B8_project.diffraction_monte_carlo import DiffractionMonteCarlo

# Generate synthetic data (replace with your actual data)
LATTICE_FILE = "data/GaAs_lattice.csv"
BASIS_FILE = "data/GaAs_basis.csv"

lattice = file_reading.read_lattice(LATTICE_FILE)
basis = file_reading.read_basis(BASIS_FILE)

unit_cell = UnitCell.new_unit_cell(basis, lattice)
diff = DiffractionMonteCarlo(unit_cell,
                             1.23,
                             min_angle_deg=18,
                             max_angle_deg=60)

all_nd_form_factors = file_reading.read_neutron_scattering_lengths(
    "data/neutron_scattering_lengths.csv")
nd_form_factors = {}
for atom in diff.unit_cell.atoms:
    nd_form_factors[atom.atomic_number] = all_nd_form_factors[atom.atomic_number]
nd_form_factors[49] = all_nd_form_factors[49]

all_xray_form_factors = file_reading.read_xray_form_factors(
    "data/x_ray_form_factors.csv")
xrd_form_factors = {}
for atom in diff.unit_cell.atoms:
    xrd_form_factors[atom.atomic_number] = all_xray_form_factors[atom.atomic_number]
xrd_form_factors[49] = all_xray_form_factors[49]

# diffraction.set_pdf(pdf)

two_thetas, intensities, stream = (
    diff.calculate_diffraction_pattern_ideal_crystal(
        xrd_form_factors,
        target_accepted_trials=3_000_000,
        unit_cell_reps=(10, 10, 10),
        trials_per_batch=1000,
        angle_bins=200,
        weighted=False))
np.savetxt('two_thetas.txt', two_thetas)
np.savetxt('intensities.txt', intensities)
np.savetxt('stream.txt', stream)

# Create a DataFrame for easier filtering
import pandas as pd

x, y, z, f = stream[:, 0], stream[:, 1], stream[:, 2], stream[:, 3]
r = np.linalg.norm(stream[:, 0:3], axis=1)
r = np.degrees(np.arcsin(r / 2 / diff.k()) * 2) # convert to scattering angle
df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'r': r, 'f': f})

r_min, r_max = r.min(), r.max()
r_steps = np.linspace(r_min, r_max, num=20)

steps = []
for i in range(len(r_steps) - 1):
    r_lower, r_upper = r_steps[i], r_steps[i + 1]
    mask = (df['r'] >= r_lower) & (df['r'] < r_upper)

    scatter = go.Scatter3d(
        x=df.loc[mask, 'x'], y=df.loc[mask, 'y'], z=df.loc[mask, 'z'],
        mode='markers',
        marker=dict(size=5, color=df.loc[mask, 'f'], colorscale='YlOrRd', opacity=0.8),
        name=f"r: {r_lower:.2f} - {r_upper:.2f}"
    )

    steps.append(dict(
        method='update',
        args=[{'visible': [False] * len(r_steps)},
              {'title': f"Shell: r = {r_lower:.2f} - {r_upper:.2f}"}],
        label=f"{r_lower:.2f} - {r_upper:.2f}"
    ))
    steps[-1]['args'][0]['visible'][i] = True  # Show only this step

# Initial figure setup
fig = go.Figure()

# Add all traces but make only the first one visible
COLOR_POW = 0.25
c_min = np.min(np.power(f, COLOR_POW))
c_max = np.max(np.power(f, COLOR_POW))
print(c_min, c_max)
for i, step in enumerate(steps):
    visible = [False] * len(r_steps)
    visible[i] = True
    scatter = go.Scatter3d(
        x=df.loc[(df['r'] >= r_steps[i]) & (df['r'] < r_steps[i + 1]), 'x'],
        y=df.loc[(df['r'] >= r_steps[i]) & (df['r'] < r_steps[i + 1]), 'y'],
        z=df.loc[(df['r'] >= r_steps[i]) & (df['r'] < r_steps[i + 1]), 'z'],
        mode='markers',
        marker=dict(size=5,
                    cmin=c_min,
                    cmax=c_max,
                    color=np.power(
                        df.loc[(df['r'] >= r_steps[i]) & (df['r'] < r_steps[i + 1]), 'f'],
                        COLOR_POW),
                    colorscale='YlOrRd',
                    colorbar=dict(
                        title="Intensity",
                        tickvals=[397.635, 316.228, 251.487, 177.828],
                        ticktext=["25B", "10B", "4B", "1B"],
                    ),
                    opacity=0.4),
        name=f"r: {r_steps[i]:.2f} - {r_steps[i + 1]:.2f}",
        visible=(i == 0)  # Only first trace is visible initially
    )
    fig.add_trace(scatter)

# Add slider
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Scattering magnitude: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=9, range=[-4.5,4.5],),
        yaxis = dict(nticks=9, range=[-4.5,4.5],),
        zaxis = dict(nticks=9, range=[-4.5,4.5],),),
    sliders=sliders
)
# fig.update_layout(
#     updatemenus=[dict(
#         type='buttons',
#         showactive=True,
#         buttons=steps,
#         direction="down",
#         x=0.17,
#         xanchor="left",
#         y=1.2,
#         yanchor="top"
#     )],
#     title="Interactive 3D Scatter Plot of Spherical Data",
#     margin=dict(l=0, r=0, b=0, t=30),
#     scene=dict(
#         xaxis_title="X",
#         yaxis_title="Y",
#         zaxis_title="Z"
#     )
# )

fig.show()