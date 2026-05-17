# Development roadmap

## Stage 1: single-phase pressure diffusion

Implemented. Use it to validate grid, transmissibility, pressure diffusion, BHP wells, source terms, and Newton solving.

## Stage 2: two-phase water-oil model

Implemented. Use it to validate saturation transport, upwinding, Corey relative permeability, water injection, production, and recovery factor.

## Stage 3: dead-oil model

Next step. Add more realistic pressure-dependent Bo, Bw, muo, muw, rock compressibility, and realistic well controls.

## Stage 4: live-oil black-oil model

Add solution gas ratio `Rs(p)`, bubble-point pressure, and phase-state switching between undersaturated and saturated oil.

## Stage 5: full three-phase black-oil model

Add gas conservation, gas saturation, gas PVT, three-phase relative permeability, capillary pressure, and gravity.

## Stage 6: production-quality simulator

Add adaptive timesteps, sparse Jacobians, linear-solver options, field input files, restart files, VTK output, and rigorous validation cases.
