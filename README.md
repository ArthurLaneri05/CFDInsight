# CFDInsight
<p align="left">
  <img src="Assets/logo.png" alt="CFDInsight logo" width="400" height="400">
</p>
  
by [Arthur Laneri](https://www.linkedin.com/in/arthur-laneri).

<p align="left">
  <!-- License -->
  <a href="LICENSE">
    <img alt="License: AGPL v3" src="https://img.shields.io/badge/License-AGPL_v3-green.svg">
  </a>
</p>
<hl>


**CFDInsight is a Python application that fully automates common CFD post-processing pipelines for [OpenFOAM](https://www.openfoam.com/), using [ParaView](https://www.paraview.org/) commands within a **Linux** environment.**

It’s designed for **fire-and-forget** runs: you define everything once in `Preferences.py`, then launch a <ins>single terminal command</ins> to produce plots, images, .txt files and CSVs — consistently and reproducibly, 
taking advantage of ParaView *parallel rendering* capabilities to speed up execution.


# What can I do with it?
Below is a brief tour of the features, illustrated with images from the *Propeller* example.

## Coefficients
Parses and aggregates forces, torques, residuals histories; exports **plots** and `averages.txt` file.

<br>

<div>
  <pre><code>Propeller [RotorPatch]:
Origin = [0 0 0]m 
D = 0.400 m 
A = 0.1257 m^2 
omega = 10.000 rps
Reference frame: x=[0. 1. 0.]; y=[ 0.  0. -1.]; z=[-1.  0.  0.]

J   = 0.5
mu  = 0.0
Ct  = 0.533     (rms=0.0)
Cx  = -0.002    (rms=0.001)
Cy  = 0.003     (rms=0.001)
Cq  = 0.204     (rms=0.0)
Cp  = 1.283     (rms=0.0)
Eta = 0.208     (rms=0.0)</code></pre>
  <p><em>Figure 1 — Snippet of <code>averages.txt</code>.</em></p>
</div>

<br>

<p align="left"> 
  <img src="https://github.com/user-attachments/assets/6d25f0fb-8990-42a4-8551-844003462398" alt="Coefficients plot for patch Hub" width="720" /> <br /> 
  <em>Figure 2 — Coefficients plot for patch <code>Hub</code>.</em> </p>

<br>

## Slices 
As many **Planar** and **cylindrical** slices of your volume mesh as you like, with total freedom on the fields displayed. Easy to setup and with absolute control. 
*Ghost meshes* — semi-transparent STL/OBJ overlays — can be loaded in the view to provide geometric context (e.g., highlighting an actuator disk).

<br>

<p align="left"> 
  <img src="https://github.com/user-attachments/assets/90f30dac-f0c9-42fc-809d-cde89ea52415" alt="Slice of the volume mesh along the X axis" width="720" /> <br /> 
  <em>Figure 3 — Slice of the volume mesh along the X axis.</em> </p>

<br>

## Surfaces
High-quality surface renders, sharing the same, powerful [**Slices**](#slices) workflow. Example below with a ghost mesh loaded.

<br>

<p align="left"> 
  <img src="https://github.com/user-attachments/assets/a668e942-013e-49d8-8221-1ea8e197cb6b" alt="Pseudo-isometric view of the propeller" width="720" /> <br /> 
  <em>Figure 4 — Pseudo-isometric view of the propeller.</em> </p>

<br>
 
## IntersectionPlots
Define as many **Planar** and **cylindrical** surfaces to slice your surface mesh, and extract information about the fields you want at the intersection.
Both raw and refined datasets are exported as CSV files. Results are also visualized through **plots**.

<br>

<p align="left"> 
  <img src="https://github.com/user-attachments/assets/12d689d8-fd4e-4a54-a5aa-a30d0cc39850" alt="Intersection Plot" width="720" /> <br /> 
  <em>Figure 5 — Plot of intersection between the propeller and a coaxial cylinder of radius 0.1m.</em> </p>

<br>
 
## DeltaSurfaces
Image-based field differencing between a baseline and a target run, with contours and calibrated colorbars.

<br>

<p align="left"> 
  <img src="https://github.com/user-attachments/assets/08930a02-1d6e-4356-915d-8fa7381e000c" alt="DeltaSurface" width="720" /> <br /> 
  <em>Figure 6 — Δ-Cp between a propeller at 600rpm and a baseline propeller rotating at 1200rpm.</em> </p>
  
<hl>

## Why it’s handy
- **Zero-click runs.** Configure once in Preferences.py, then CFDInsight takes it from there.

- **OpenFOAM-aware.** Automatically parses your case’s _"initialConditions"_ to populate post-processing parameters (e.g., inserts the correct freestream 
velocity in the Cp expression) and can overlay key values on outputs for context.

- **High resolution and runnable in parallel.** Uses ParaView’s _pvserver_ (and benefits from MPI parallelism) for scalable rendering and data extraction.

- **Reproducible and easily comparable outputs.** Using the same _Preferences.py_ across different revisions of a component ensures consistent post-processing and effortless comparisons—especially when combined with the powerful **DeltaSurfaces** workflow, which directly visualizes surface-field differences relative to a chosen **baseline** case.


## Installation
- Download [ParaView 6.0](https://www.paraview.org/download/) (compatibility with previous versions is NOT guaranteed).

- Place the _CFDInsight_ package root whenever you like.

- Create the `CFDInsight` **bash function**.
  In a random terminal, type:
  ```
  bash
  echo 'CFDInsight() { pvpython "/path/to/CFDInsight/Main.py" "$@"; }' >> ~/.bashrc
  ```

- Set up **pvserver** and **pvpython** availability inside PATH:
  ```
  bash
  echo 'export PATH="/path/to/paraview/bin:$PATH"' >> ~/.bashrc
  ```

- Add `CFDInsight` to your PYTHONPATH so IDEs can show docstrings on hover:
  ```
  bash
  echo 'export PYTHONPATH="/path/to/CFDInsight:$PYTHONPATH"' >> ~/.bashrc
  source ~/.bashrc
  ```

## Tutorials, Examples, and Documentation
To get started, check the tutorials present in the `Examples` folder consisting of **ready-to-run** CFD examples (runnable in a matter of minutes) accompanied by thoroughly commented Post-Processing pipeline setup.
After that, I guarantee you will be at easy with all the functionalities of _CFDInsight_.


# Legal Notes
CFDInsight is an independent open-source project and is not affiliated with or endorsed by Ansys, Inc.  
ANSYS® and EnSight® are registered trademarks of Ansys, Inc. and/or its subsidiaries.  
All other product names, logos, and brands are property of their respective owners.

