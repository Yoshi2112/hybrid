## PYTHON MODULES ##
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
import plot_and_save as pas

from simulation_parameters_1D import generate_data, generate_plots

if __name__ == '__main__':
    start_time = timer()
    
    pos, vel, Ie, W_elec, idx                    = init.initialize_particles()
    B, E_int                                     = init.initialize_fields()
    DT, max_inc, data_iter, plot_iter, subcycles = aux.set_timestep(vel)

    q_dens, Ji    = sources.collect_moments(vel, Ie, W_elec, idx)
    E_int, Ve, Te = fields.calculate_E(B, Ji, q_dens)
    vel           = particles.velocity_update(pos, vel, Ie, W_elec, idx, B, E_int, -0.5*DT)
    
    qq      = 0
    while qq < max_inc:
        # Check timestep
        pos, vel, qq, DT, max_inc, data_iter, plot_iter, subcycles \
        = aux.check_timestep(qq, DT, pos, vel, B, E_int, q_dens, Ie, W_elec, max_inc, data_iter, plot_iter, subcycles, idx)
        
        # Main loop
        pos, vel, Ie, W_elec, q_dens_adv, Ji = particles.advance_particles_and_moments(pos, vel, Ie, W_elec, idx, B, E_int, DT)
        q_dens                               = 0.5 * (q_dens + q_dens_adv)
        B, E_half, Ve, Te                    = fields.cyclic_leapfrog(B, q_dens, Ji, DT, subcycles)
        q_dens                               = q_dens_adv.copy()
        
        # Predictor-Corrector: Advance fields to start of next timestep
        E_int = fields.predictor_corrector(B, E_int, E_half, pos, vel, q_dens_adv, Ie, W_elec, idx, DT, subcycles)
        
        if qq%data_iter == 0 and generate_data == 1:
            pas.save_data(DT, data_iter, qq, pos, vel, Ji, E_int, B, Ve, Te, q_dens)

        if qq%plot_iter == 0 and generate_plots == 1:
            pas.create_figure_and_save(pos, vel, E_int, B, q_dens, qq, DT, plot_iter)

        if (qq + 1)%25 == 0:
            print('Timestep {} of {} complete'.format(qq + 1, max_inc))

        qq += 1

    print("Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2)))  # Time taken to run simulation