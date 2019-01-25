## PYTHON MODULES ##
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
import plot_and_save as pas
import diagnostics   as diag

from simulation_parameters_1D import generate_data, generate_plots, NX, N


if __name__ == '__main__':
    start_time = timer()

    part = init.initialize_particles()
    B, E = init.initialize_magnetic_field()

    DT, maxtime, data_dump_iter, plot_dump_iter = aux.set_timestep(part)

    if generate_data == 1:
        pas.store_run_parameters(DT, data_dump_iter)

    print 'Loading initial state...\n'
    part, dns_int, dns_half, J_plus, J_minus, G, L   = sources.init_collect_moments(part, 0.5*DT)
    
    #diag.check_cell_velocity_distribution(part, NX/2, 0)
    #diag.check_position_distribution(part, 0)
    #diag.check_velocity_distribution(part, 0)
    
    qq = 0
    maxtime = 0
    while qq < maxtime:
        qq, DT, maxtime, data_dump_iter, plot_dump_iter, change_flag = aux.check_timestep(qq, DT, part, B, E, dns_int, maxtime, data_dump_iter, plot_dump_iter)
        
        if change_flag == 1:
            print 'Timestep halved. DT = {}'.format(DT)
        
        B = fields.cyclic_leapfrog(B, dns_int, J_minus, DT)
        E = fields.calculate_E(B, J_minus, dns_half)
        J = sources.push_current(J_plus, E, B, L, G, DT)
        E = fields.calculate_E(B, J, dns_half)
        
        part = particles.velocity_update(part, B, E, DT)
        
        part, dns_int, J_plus, J_minus, G, L = sources.collect_moments(part, DT)
        
        dns_int = 0.5 * (dns_int + dns_half)
        J       = 0.5 * (J_plus  +  J_minus)
        B       = fields.cyclic_leapfrog(B, dns_int, J, DT)
        
        if qq%data_dump_iter == 0 and generate_data == 1:                                   # Save data, if flagged
            pas.save_data(DT, data_dump_iter, qq, part, J, E, B, dns_int)

        if qq%plot_dump_iter == 0 and generate_plots == 1:                                  # Generate and save plots, if flagged
            pas.create_figure_and_save(part, J, B, dns_int, qq, DT, plot_dump_iter)

        if (qq + 1)%10 == 0:
            print 'Timestep {} of {} complete'.format(qq + 1, maxtime)

        qq += 1

    print "Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2))  # Time taken to run simulation