## PYTHON MODULES ##
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D_multithreaded  as particles
import fields_1D     as fields
import plot_and_save as pas
import pdb
from simulation_parameters_1D import generate_data, generate_plots


if __name__ == '__main__':
    start_time = timer()

    pos, vel, idx                       = init.initialize_particles()
    B, E_int                            = init.initialize_fields()
    DT, max_inc, data_iter, plot_iter   = aux.set_timestep(vel)

    q_dens, Ji           = particles.advance_particles_and_moments(pos, vel, idx, B, E_int, 0)
    E_int, Ve, Te        = fields.calculate_E(B, Ji, q_dens)

    particles.sync_velocities(pos, vel, idx, B, E_int, -0.5*DT)

    qq      = 0
    while qq < max_inc:

        # TIMESTEP CHECK
        qq, DT, max_inc, data_iter, plot_iter \
        = aux.check_timestep(qq, DT, pos, vel, B, E_int, q_dens, max_inc, data_iter, plot_iter, idx)



        # MAIN LOOP
        q_dens_adv, Ji = particles.advance_particles_and_moments(pos, vel, idx, B, E_int, DT)

        q_dens                   = 0.5 * (q_dens + q_dens_adv)
        B                        = fields.push_B(B, E_int, DT)
        E_half, Ve, Te           = fields.calculate_E(B, Ji, q_dens)
        q_dens                   = q_dens_adv.copy()

        E_int, B                 = fields.predictor_corrector(B, E_int, E_half, pos, vel, q_dens, idx, DT)



        # OUTPUT
        if qq%data_iter == 0 and generate_data == 1:
            pas.save_data(DT, data_iter, qq, pos, vel, Ji, E_int, B, Ve, Te, q_dens)

        if qq%plot_iter == 0 and generate_plots == 1:
            pas.create_figure_and_save(pos, vel, E_int, B, q_dens, qq, DT, plot_iter)

        if (qq + 1)%25 == 0:
            print 'Timestep {} of {} complete'.format(qq + 1, max_inc)

        qq += 1







    print "Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2))  # Time taken to run simulation