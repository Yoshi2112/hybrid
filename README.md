# hybrid

1D plasma particle-in-cell (PIC) EM-solver designed to grow waves from a given particle distribution

## Features
- Multi-ion and multi-species capable
- Uniform or parabolic background field geometry available
- Parallelized particle functions for multi-threaded CPUs
- Periodic or open (experimental) boundary conditions available for the fields
- Periodic, open flux, reflective, or reinitialized distribution boundary conditions for the particles
- Analysis script available, but it's a bit of a mess

This version of the hybrid simulation is based largely off the review chapter Space Plasma Simulation (2003) which was originally developed as part of research into ULF waves and particles in the Earth's foreshock region by Winske et al. (1984). This code has been modified according to techniques discovered in other literature, such as
- Quiet start (Bridsall and Langdon, 1985)
- Triangular shaped cloud (TSC) particle geometry from Matthews et al. (1994)
- Absorbing boundary conditions and parabolic background field as per Shoji et al. (2009)
- Various influences from Winske and Omidi (1993), Lipatov (2005), Katoh and Omura (2006)
- Archived versions of the code have functions for included resistivity, bit-reversed radix intialization, oblique background fields, and externally driven (tranverse) waves



## Installation
Clone the repository to your local machine. Strictly, only the /simulation_codes/ folder is needed.

## Setting up a run
The parameters that determine the characteristics of a run are governed by two input files found in the simulation_codes/run_inputs/ directory:

_run_params.run, containing
- `DRIVE` : The main drive path to save runs to (e.g. `D:/` or ```/home/c_student/```)
- `SAVE_PATH` : The path on `DRIVE` containing the main run folder for this series of runs
- `RUN` : Number determining which run of a series this will be. Setting to `-` autosets sequentially (care must be taken if several run folders exist that aren't sequential)
- `SAVE_PARTICLE_FLAG` : 1 to save particles, 0 otherwise. Useful to disable to save space if only wave effects are of interest
- `SAVE_FIELD_FLAG` : As above, but for fields. Most commonly disabled for debugging purposes.
- `SEED` : Random seed
- `HOMOGENEOUS_B0_FLAG` : 1 for uniform field, 0 for parabolic
- `PARTICLE_PERIODIC_FLAG` : Exiting particles are reinitialized at the opposite boundary with no change to their velocity
- `PARTICLE_REFLECT_FLAG` : Particles reflect off boundaries, conserving energy
- `PARTICLE_REINIT_FLAG` : Exiting particles are reinitialized using a random flux distribution at each boundary
- `FIELD_PERIODIC` : Fields solve on a periodic grid with ghost cells at each end. Absorbing (open) boundaries otherwise
- `NOWAVES_FLAG` : Disables wave solver. Useful for debugging boundary conditions or observing particle motions in the background field
- `SOURCE_SMOOTHING_FLAG` : Applies 3-point gaussian smoothing to the source term (current and charge density) arrays. Reduces high frequency noise prevalent with low particle counts
- `QUIET_START_FLAG` : Enables the _quiet start_ which initializes particles in pairs with opposing perpendicular velocities. Reduces numerical noise in the current density array which can cause waves to grow unphysically.
- `E_DAMPING_FLAG` : For open boundaries, apply damping on the Hall Term of the E-field solution, in addition to damping the B-field.
- `DAMPING_MULTIPLIER_RD` : For open boundary conditions, this factor is multiplied by the Shoji/Omura damping array. Fudge factor because their parameters caused too much damping (and thus reflection)
- `DAMPING_ENCROACH_FRAC` : Specifies how much of the solution domain includes damping on each side (as a fraction of total solution domain). Independent of ND (but changing ND might alter rD and thus total damping)
- `NX` : Number of cells containing particles and constituting the 'real' region of the simulation (i.e. solution space)
- `ND` : Number of damping cells on each side of the solution space, generally set as same fraction of NX (e.g. NX/2)
- `SIM_TIME` : Maximum simulation time in multiples of the inverse gyrofrequency. Generally 1000-2000 is sufficient
- `DXM` : Cell size in multiples of the ion inertial length c/wpi. Default is 1.0, but can be smaller if higher frequency waves need to be resolved
- `IE` : Electron treatment. 1 for adiabatic (default) or 0 for isothermal (contant Te)
- `GYROPERIOD_RESOLUTION` : Maximum initial timestep in units of inverse gyrofrequency. Generally limited to 0.02 for Predictor-Corrector
- `FREQUENCY_RESOLUTION` : Deprecated, but originally used as a timestep limiter on other cyclic quantities in the simulation 
- `PARTICLE_DUMP_FREQ` : Number of inverse gyrofrequencies between each particle dump. Less often is better, since particles are huge files
- `FIELD_DUMP_FREQ` : Number of inverse gyrofrequencies between each field dump. Minimum of 0.5 required to resolve frequencies up to the proton gyrofrequency, 0.25 is a good default

Anything under this line gets added to the run save file as a comment. Note for the particle boundary conditions that only one can be active at a time or an error is raised.

_plasma_params.plasma, containing ion species to be used in the run, as well as general magnetic field parameters. Each column is a new species. Parameters are:
- `LABEL` : The name of the species. Latex $$ signs may be used to indicate charge or subscript info. Used for plotting and ID.
- `COLOUR` : Species colour, used when outputting plots
- `TEMP_FLAG` : Temperature flag defining species as **hot** (1) ring current or **cold** (0) background populations
- `DIST_FLAG` : Deprecated, but previously used to indicate the spatial distribution of the species at init.
- `NSP_PPC` : Number of particles per cell for this species. Most crucial parameter for balancing runtime vs. accuracy. Cold species only need hundreds, hot species can use tens of thousands per cell.
- `MASS_PROTON` : Ion mass in proton mass units. Generally 1, 4, or 16 for hydrogen, helium, and oxygen, respectively
- `CHARGE_ELEM` : Ion charge in elementary charge units. Generally ions are singly charged (1.)
- `DRIFT_VA` : Bulk drift velocity along the simulation dimension in multiples of the Alfven speed. Only valid for periodic boundaries.
- `DENSITY_CM3` : Species density in particles per cubic centimeter. Determines the macroparticle weighting in real units of each species.
- `ANISOTROPY` : Temperature anisotropy of this species as A = T_perp/T_parallel - 1
- `ENERGY_PERP` : Perpendicular (to the field) component of the population energy. Determines perpendicular velocity distribution.
- `ELEC_ENERGY` : Electron energy. Determines (initial) electron fluid temperature
- `BETA_FLAG` : Flag to define energies in terms of their plasma beta (prevalent in literature) or in eV (for data comparison)
- `L` : McIllwain L shell approximation for the parabolic field. Determines magnetic field gradient if applicable.
- `B_EQ` : Sets value of constant uniform magnetic field in nT, or as an override for the equatorial value in the non-uniform case. Set to `-` to calculate based on L value.
- `B_XMAX` : Manually set non-uniform magnetic field intensity at simulation boundary. Overrides the 'a' value calculated from L.

Note that these files are simply the defaults and other.run and .plasma files can be called either by command line or through internal flags (such as in the _Fu_test.plasma case). This is useful for batch running or for comparison to literature values.

## Running the code
With a powershell, terminal, or cmd window open in the /PREDCORR_1D_PARALLEL/ directory, a run can be initialized by
```sh
python main1D.py
```

### Flags
Generally only used when creating batch files for RCG. File paths are relative to /run_inputs/
-r or --runfile to override the .run file called
-p or --plasmafile to override the .plasma file called
-n or --run_num to override the run number assigned

**EXAMPLES**
```sh
python main1D.py -r multispecies_SI_test_cold/run_params_PU1024.run
```

```sh
python main1D.py -p multispecies_SI_test_cold/plasma_params_H_He.plasma -n 2
```

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
