## PYTHON MODULES ##
from timeit import default_timer as timer
import numpy as np

## HYBRID MODULES ##
import hybrid_init as init
import hybrid_auxilliary as aux
import hybrid_particles as particles
import hybrid_fields as fields
import hybrid_sources as sources

from const import generate_data, generate_plots

## OUTPUT MODULES ##
import plot_and_save as pas  


if __name__ == '__main__':                         # Main program start
    start_time     = timer()                       # Start Timer

    print 'Initializing parameters...'
    part                    = init.initialize_particles()
    B, E, Ji, dns, W, Wb    = init.initialize_fields()

    DT, maxtime, framegrab  = aux.set_timestep(part)
    maxtime = 1000
    for qq in range(maxtime):
        if qq == 0:
            print 'Simulation starting...'
            W           = sources.assign_weighting(part[0, :], part[6, :], 1)               # Assign initial (E) weighting to particles
            dns         = sources.collect_density(part[6, :], W, part[2, :])                # Collect initial density   
            Ji          = sources.collect_current(part, W)                                  # Collect initial current
            initial_cell_density      = dns                                                 # Store initial particle density: For normalization

            B[:, 0:3] = fields.push_B(B[:, 0:3], E[:, 0:3], 0)                              # Initialize magnetic field (should be second?)
            E[:, 0:3] = fields.push_E(B[:, 0:3], Ji, dns, 0)                                # Initialize electric field
            
            part = particles.velocity_update(part, B[:, 0:3], E[:, 0:3], -0.5*DT, W)        # Retard velocity to N - 1/2 to prevent numerical instability
                    
        else:
            # N + 1/2
            part      = particles.velocity_update(part, B[:, 0:3], E[:, 0:3], DT, W)        # Advance Velocity to N + 1/2
            part, W   = particles.position_update(part, DT)                                 # Advance Position to N + 1
            B[:, 0:3] = fields.push_B(B[:, 0:3], E[:, 0:3], DT)                             # Advance Magnetic Field to N + 1/2
            
            dns       = 0.5 * (dns + sources.collect_density(part[6, :], W, part[2, :]))    # Collect ion density at N + 1/2 : Collect N + 1 and average with N                                             
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
            
            part = particles.velocity_update(part, B[:, 0:3], E[:, 0:3], DT, W)             # Advance particle velocities to N + 3/2
            part, W = particles.position_update(part, DT)                                   # Push particles to positions at N + 2
            dns  = 0.5 * (dns + sources.collect_density(part[6, :], W, part[2, :]))         # Collect ion density as average of N + 1, N + 2
            Ji   = sources.collect_current(part, W)                                         # Collect ion flow at N + 3/2
            B[:, 0:3] = fields.push_B(B[:, 0:3], E[:, 0:3], DT)                             # Push Magnetic Field again to N + 3/2 (Use same E(N + 1)
            E[:, 0:3] = fields.push_E(B[:, 0:3], Ji, dns, DT)                               # Push Electric Field to N + 3/2   ii = odd numbers
            
            # Correct Fields
            E[:, 0:3] = 0.5 * (E[:, 3:6] + E[:, 0:3])                                       # Electric Field interpolation
            B[:, 0:3] = fields.push_B(B[:, 3:6], E[:, 0:3], DT)                             # Push B using new E and old B
            
            # Reset Particle Array to last real value
            part = old_part                                                                 # The stored densities at N + 1/2 before the PC method took place (previously held PC at N + 3/2)
            dns  = dns_old     
                 
        print 'Iteration {} complete'.format(qq)
        
        if qq%framegrab == 0:                                                               # At a specified interval
            if generate_plots == 1:
                pas.create_figure_and_save(part, E, B, dns, qq, DT, framegrab)              # Generate and save plots, if flagged
            if generate_data == 1:                                                          # Generate and save data, if flagged
                pas.save_data(DT, framegrab, qq, part, Ji, E, B, dns)
                
    print "Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2))  # Time taken to run simulation
    print "Time per iteration: {0:.4f} seconds".format(round((timer() - start_time) / (qq + 1), 4))

