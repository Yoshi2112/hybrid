## PYTHON MODULES ##
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
#import plot_and_save as pas

#from simulation_parameters_1D import generate_data, generate_plots, dx


def main_simulation_loop():
    print 'Initializing parameters...'
    part                         = init.initialize_particles()
    B, E, J, dns_int             = init.initialize_fields()

    DT, maxtime, data_dump_iter, plot_dump_iter = aux.set_timestep(part)

# =============================================================================
#     if generate_data == 1:
#         pas.store_run_parameters(DT, data_dump_iter)
# =============================================================================

    print 'Loading initial state...'
    part, dns_int, dns_half, J_plus, J_minus, G, L   = sources.collect_moments(part, 0.5*DT, init=True)

    for qq in range(maxtime):
        B = fields.cyclic_leapfrog(B, dns_int, J_minus, DT)
        E = fields.calculate_E(B, J_minus, dns_half)
        J = sources.push_current(J_plus, E, B, L, G, DT)
        E = fields.calculate_E(B, J, dns_half)
        
        part = particles.boris_velocity_update(part, B, E, DT)
        
        part, dns_int, J_plus, J_minus, G, L = sources.collect_moments(part, DT)
        
        dns_int = 0.5 * (dns_int + dns_half)
        J       = 0.5 * (J_plus  +  J_minus)
        B       = fields.cyclic_leapfrog(B, dns_int, J, DT)
        
# =============================================================================
#         if qq%data_dump_iter == 0 and generate_data == 1:                                   # Save data, if flagged
#             pas.save_data(DT, data_dump_iter, qq, part, Ji, E, B, dns)
# 
#         if qq%plot_dump_iter == 0 and generate_plots == 1:                                  # Generate and save plots, if flagged
#             pas.create_figure_and_save(part, E, B, dns, qq, DT, plot_dump_iter)
# =============================================================================

        print 'Timestep {} of {} complete'.format(qq, maxtime)
    return

if __name__ == '__main__':
    start_time = timer()
    main_simulation_loop()

    print "Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2))  # Time taken to run simulation