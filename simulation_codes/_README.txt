I'm terrible at documenting code, so here are notes. Generally, ABCs still need testing, but it's been difficult trying to get waves in the sim without generating them from particles.

PREDCORR_1D_INJECT_OPT
1D Predictor/Corrector hybrid code. Originally meant to test injection, later optimized to run 4x faster (thanks vectorized Boris method). Used as a testbed to try a bunch of "Open" boundary conditions. Particles are hard, yo.

PREDCORR_1D_OPEN_NEW
1D Predictor/Corrector hybrid code. Attempted open boundary conditions with ABCs and an "open flux" boundary based on Daughton's 2006 paper (measure moments at edges, inject based on that distro function). However, I can't quite work out how to go from the maths of a distro to knowing how many particles (and at what velocities) to inject per timestep. Heavily modified with an "injection" routine

PREDCORR_1D_KLIMAS
1D Predictor/Corrector hybrid code. Open boundary conditions applied by using the zero derivative condition from Klimas' 2008 paper, as well as some modifications to include a bit-reversed radix-N method for better quiet starts. This version still doesn't quite match the INJECT_OPT code with periodic conditions, so there may still be some issues here I haven't account for. Needs testing, but it seems roughly fine and stable. The particle boundaries are unstable under certain conditions however, and need further testing. There's also a code in here that loads particles with different temperatures based on position, in an attempt to localise growth to the centre. This hasn't worked yet however.

PREDCORR_1D_PULSE
Basically the KLIMAS code but modified to include some sort of analytic wave generation at the equator, for testing purposes. Just a junk code to see if it even works. Need the generation in order to properly test and define the ABCs (with a uniform cold plasma). All recent reflection and flux tests were carried out in this code, and it is the latest version that will be used.
NOTE: THERE WAS AN ISSUE IN THE DAMPING SOURCE TERM FOUND AND FIXED : J COMPONENTS WEREN'T COPIED FOR THE "LAST VALUE" CONDITION (OPTIONS SINCE DELETED). IF OLDER CODES USED, CHECK THAT THIS HAS BEEN FIXED IN THEM.

PREDCORR_1D_NEWFLUX
So the Klimas BC's seem to die once a wave hits them. I don't think they were designed for such a dynamic interior (more of a steady-state sort of thing). Seems the only way to get around it is to actually inject particles by retrieving some sort of flux, and then injecting particles based on this flux. Will need to try either measuring outgoing and equalling that (easy method) or having some set flux per timestep as per the original distribution (harder method, as it requires calculating that incoming flux). PROTIP: The outgoing flux should be equal to the incoming flux with either method for steady state. Good check.

PREDCORR_1D_CARLO
Version with monte-carlo injection at edges. Used INJECT_OPT code as source since the PULSE code seemed to have some issues (as at this writing). Though, this is mostly just a testbed for the new injection mechanism - have to sample new vx from flux distribution vx*f(vx), not just from half maxwellian. Should have versions to reinitialise leaving particles, as well as a version to inject at a constant-ish rate based on calculation at t=0 (with some accounting for non-integer bits). For runs with no waves, constant vs. matching flux conditions should be practically identical. Also going to remove 3D positions because I probably won't be able to run this for long enough to matter.