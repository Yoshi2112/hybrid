## PYTHON MODULES ##
from timeit import default_timer as timer
import numpy as np
import pdb
import sys

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources

from simulation_parameters_1D import generate_data, generate_plots, dx

## OUTPUT MODULES ##
import plot_and_save as pas

def main_simulation_loop():
    print 'Initializing parameters...'
    part                         = init.initialize_particles()
    B, E, Ji, dns, W, Wb         = init.initialize_fields()

    DT, maxtime, data_dump_iter, plot_dump_iter = aux.set_timestep(part)

    if generate_data == 1:
        pas.store_run_parameters(DT, data_dump_iter)

    print 'Loading initial state...'
    W           = sources.assign_weighting(part[0, :], part[1, :], 1)               # Assign initial (E) weighting to particles
    dns         = sources.collect_density(part[1, :], W, part[2, :])                # Collect initial density
    Ji          = sources.collect_current(part, W)                                  # Collect initial current

    B[:, 0:3]   = fields.push_B(B[:, 0:3], E[:, 0:3], 0)                            # Initialize magnetic field (should be second?)
    E[:, 0:3]   = fields.push_E(B[:, 0:3], Ji, dns, 0)                              # Initialize electric field

    part, W   = particles.position_update(part, 0.5*DT)                             # Retard velocity to N - 1/2 to prevent numerical instability

    for qq in range(maxtime):
        # N + 1/2
        part      = particles.boris_velocity_update(part, B[:, 0:3], E[:, 0:3], DT, W)  # Advance Velocity to N + 1/2
        part, W   = particles.position_update(part, DT)                                 # Advance Position to N + 1
        B[:, 0:3] = fields.push_B(B[:, 0:3], E[:, 0:3], DT)                             # Advance Magnetic Field to N + 1/2

        dns       = 0.5 * (dns + sources.collect_density(part[1, :], W, part[2, :]))    # Collect ion density at N + 1/2 : Collect N + 1 and average with N
        Ji        = sources.collect_current(part, W)                                    # Collect ion flow at N + 1/2
        E[:, 6:9] = E[:, 0:3]                                                           # Store Electric Field at N because PC, yo
        E[:, 0:3] = fields.push_E(B[:, 0:3], Ji, dns, DT)                               # Advance Electric Field to N + 1/2   ii = even numbers

        if qq%data_dump_iter == 0 and generate_data == 1:                                   # Save data, if flagged
            pas.save_data(DT, data_dump_iter, qq, part, Ji, E, B, dns)

        if qq%plot_dump_iter == 0 and generate_plots == 1:                                  # Generate and save plots, if flagged
            pas.create_figure_and_save(part, E, B, dns, qq, DT, plot_dump_iter)

        print 'Timestep {} of {} complete'.format(qq, maxtime)
    return

if __name__ == '__main__':
    start_time = timer()
    main_simulation_loop()

    print "Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2))  # Time taken to run simulation