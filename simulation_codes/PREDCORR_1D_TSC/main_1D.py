## PYTHON MODULES ##
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
import save_routines as save

from simulation_parameters_1D import save_particles, save_fields


if __name__ == '__main__':
    start_time = timer()
    
    pos, vel, Ie, W_elec, idx           = init.initialize_particles()
    B, E_int                            = init.initialize_fields()
    
    DT, max_inc, part_save_iter, field_save_iter = aux.set_timestep(vel)
    print('Timestep: %.4fs, %d iterations total\n' % (DT, max_inc))
    
    
    q_dens, Ji    = sources.collect_moments(vel, Ie, W_elec, idx)    
    E_int, Ve, Te = fields.calculate_E(B, Ji, q_dens)    
    vel           = particles.velocity_update(pos, vel, Ie, W_elec, idx, B, E_int, -0.5*DT)

    qq      = 0
    print('Starting main loop...')
    while qq < max_inc:
        ############################
        ##### EXAMINE TIMESTEP #####
        ############################
        vel, qq, DT, max_inc, part_save_iter, field_save_iter, ch_flag \
        = aux.check_timestep(qq, DT, pos, vel, B, E_int, q_dens, Ie, W_elec, max_inc, part_save_iter, field_save_iter, idx)
        
        if ch_flag == 1:
            print('Timestep halved. Syncing particle velocity with DT = {}'.format(DT))
        elif ch_flag == 2:
            print('Timestep Doubled. Syncing particle velocity with DT = {}'.format(DT))
        
        #######################
        ###### MAIN LOOP ######
        #######################
        pos, vel, Ie, W_elec, q_dens_adv, Ji = particles.advance_particles_and_moments(pos, vel, Ie, W_elec, idx, B, E_int, DT)
        q_dens                               = 0.5 * (q_dens + q_dens_adv)
        B                                    = fields.push_B(B, E_int, DT)
        E_half, Ve, Te                       = fields.calculate_E(B, Ji, q_dens)
        q_dens                               = q_dens_adv.copy()
        
        E_int, B = fields.predictor_corrector(B, E_int, E_half, pos, vel, q_dens_adv, Ie, W_elec, idx, DT)

        ########################
        ##### OUTPUT DATA  #####
        ########################
        if qq%part_save_iter == 0 and save_particles == 1:
            save.save_particle_data(DT, part_save_iter, qq, pos, vel)

        if qq%field_save_iter == 0 and save_fields == 1:
            save.save_field_data(DT, field_save_iter, qq, q_dens, E_int, B, Ve, Te, q_dens)
            
        if (qq + 1)%25 == 0:
            print('Timestep {} of {} complete'.format(qq + 1, max_inc))

        qq += 1
    print("Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2)))