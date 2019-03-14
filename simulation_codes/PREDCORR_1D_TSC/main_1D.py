## PYTHON MODULES ##
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
import plot_and_save as pas

from simulation_parameters_1D import generate_data, generate_plots, NX

def main_loop():
    return


if __name__ == '__main__':
    start_time = timer()
    
    pos, vel, Ie, W_elec, idx           = init.initialize_particles()
    B, E_int                            = init.initialize_fields()
    
    
    DT, max_inc, data_iter, plot_iter   = aux.set_timestep(vel)
    print 'Timestep: %.4fs, %d iterations total' % (DT, max_inc)
    if generate_data == 1:
        pas.store_run_parameters(DT, data_iter)
    
    q_dens, Ji    = sources.collect_moments(vel, Ie, W_elec, idx)
    
    print 'Initial source term check:'
    print 'Average cell density: {}cc'.format(q_dens[1: NX + 1].mean())
    print 'Average cell current: {}A/m'.format(Ji[1: NX + 1].mean())
    
    E_int, Ve, Te = fields.calculate_E(B, Ji, q_dens)
    vel           = particles.velocity_update(pos, vel, Ie, W_elec, idx, B, E_int, -0.5*DT)

    qq      = 0
    max_inc  = 1
    while qq < max_inc:
        # Check timestep
        pos, vel, qq, DT, max_inc, data_iter, plot_iter, ch_flag \
        = aux.check_timestep(qq, DT, pos, vel, B, E_int, q_dens, Ie, W_elec, max_inc, data_iter, plot_iter, idx)
        
        if ch_flag == 1:
            print 'Timestep halved. Syncing particle velocity with DT = {}'.format(DT)
        elif ch_flag == 2:
            print 'Timestep Doubled. Syncing particle velocity with DT = {}'.format(DT)
        
        # Main loop
        pos, vel, Ie, W_elec, q_dens_adv, Ji = particles.advance_particles_and_moments(pos, vel, Ie, W_elec, idx, B, E_int, DT)
        q_dens                               = 0.5 * (q_dens + q_dens_adv)
        B                                    = fields.push_B(B, E_int, DT)
        E_half, Ve, Te                       = fields.calculate_E(B, Ji, q_dens)
        q_dens                               = q_dens_adv.copy()
        
        # Predictor-Corrector: Advance fields to start of next timestep
        E_int, B = fields.predictor_corrector(B, E_int, E_half, pos, vel, q_dens_adv, Ie, W_elec, idx, DT)
        
        if generate_data == 1:
            if qq%data_iter == 0:
                pas.save_data(DT, data_iter, qq, pos, vel, Ji, E_int, B, Ve, Te, q_dens)

        if generate_plots == 1:
            if qq%plot_iter == 0:
                pas.create_figure_and_save(pos, vel, E_int, B, q_dens, qq, DT, plot_iter)

        if (qq + 1)%25 == 0:
            print 'Timestep {} of {} complete'.format(qq + 1, max_inc)

        qq += 1
    print "Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2))  # Time taken to run simulation