## PYTHON MODULES ##
from timeit import default_timer as timer
import numpy as np

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources

from simulation_parameters_1D import generate_data, generate_plots

## OUTPUT MODULES ##
import plot_and_save as pas

def main_simulation_loop():
    print 'Initializing parameters...'

    part                    = init.initialize_particles()
    B, E, Ji, dns, W, Wb    = init.initialize_fields()
    DT, maxtime, framegrab  = aux.set_timestep(part)

    for qq in range(maxtime):
        if qq == 0:
            print 'Simulation starting...'
            W           = sources.assign_weighting(part[0, :], part[1, :], 1)               # Assign initial (E) weighting to particles
            dns         = sources.collect_density(part[1, :], W, part[2, :])                # Collect initial density
            Ji          = sources.collect_current(part, W)                                  # Collect initial current

            B[:, 0:3] = fields.push_B(B[:, 0:3], E[:, 0:3], 0)                              # Initialize magnetic field (should be second?)
            E[:, 0:3] = fields.push_E(B[:, 0:3], Ji, dns, 0)                                # Initialize electric field

            part = particles.boris_velocity_update(part, B[:, 0:3], E[:, 0:3], -0.5*DT, W)        # Retard velocity to N - 1/2 to prevent numerical instability
        else:
            # N + 1/2
            part      = particles.boris_velocity_update(part, B[:, 0:3], E[:, 0:3], DT, W)        # Advance Velocity to N + 1/2
            part, W   = particles.position_update(part, DT)                                 # Advance Position to N + 1
            B[:, 0:3] = fields.push_B(B[:, 0:3], E[:, 0:3], DT)                             # Advance Magnetic Field to N + 1/2

            dns       = 0.5 * (dns + sources.collect_density(part[1, :], W, part[2, :]))    # Collect ion density at N + 1/2 : Collect N + 1 and average with N
            Ji        = sources.collect_current(part, W)                                    # Collect ion flow at N + 1/2
            E[:, 6:9] = E[:, 0:3]                                                           # Store Electric Field at N because PC, yo
            E[:, 0:3] = fields.push_E(B[:, 0:3], Ji, dns, DT)                               # Advance Electric Field to N + 1/2   ii = even numbers

            # ----- Predictor-Corrector Method ----- #
            # Predict values of fields at N + 1
            B[:, 3:6] = B[:, 0:3]                                                           # Store last "real" magnetic field (N + 1/2)
            E[:, 3:6] = E[:, 0:3]                                                           # Store last "real" electric field (N + 1/2)
            E[:, 0:3] = -E[:, 6:9] + 2*E[:, 0:3]                                            # Predict Electric Field at N + 1
            B[:, 0:3] = fields.push_B(B[:, 0:3], E[:, 0:3], DT)                             # Predict Magnetic Field at N + 1 (Faraday, based on E(N + 1))

            # Extrapolate Source terms and fields at N + 3/2
            old_part = np.copy(part)                                                        # Back up particle attributes at N + 1
            dns_old  = np.copy(dns)                                                         # Store last "real" densities (in an E-field position, I know....)

            part      = particles.boris_velocity_update(part, B[:, 0:3], E[:, 0:3], DT, W)        # Advance particle velocities to N + 3/2
            part, W   = particles.position_update(part, DT)                                 # Push particles to positions at N + 2
            dns       = 0.5 * (dns + sources.collect_density(part[1, :], W, part[2, :]))    # Collect ion density as average of N + 1, N + 2
            Ji        = sources.collect_current(part, W)                                    # Collect ion flow at N + 3/2
            B[:, 0:3] = fields.push_B(B[:, 0:3], E[:, 0:3], DT)                             # Push Magnetic Field again to N + 3/2 (Use same E(N + 1)
            E[:, 0:3] = fields.push_E(B[:, 0:3], Ji, dns, DT)                               # Push Electric Field to N + 3/2   ii = odd numbers

            # Correct Fields
            E[:, 0:3] = 0.5 * (E[:, 3:6] + E[:, 0:3])                                       # Electric Field interpolation
            B[:, 0:3] = fields.push_B(B[:, 3:6], E[:, 0:3], DT)                             # Push B using new E and old B

            # Reset Particle Array to last real value
            part = old_part                                                                 # The stored densities at N + 1/2 before the PC method took place (previously held PC at N + 3/2)
            dns  = dns_old

        if qq%framegrab == 0:                                                               # At a specified interval
            if generate_plots == 1:
                pas.create_figure_and_save(part, E, B, dns, qq, DT, framegrab)              # Generate and save plots, if flagged
            if generate_data == 1:
                pas.save_data(DT, framegrab, qq, part, Ji, E, B, dns)                       # Save data, if flagged

        print 'Step {} of {} complete'.format(qq, maxtime)
    return

if __name__ == '__main__':
    start_time = timer()
    main_simulation_loop()

    print "Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2))  # Time taken to run simulation