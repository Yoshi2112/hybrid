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
    
    # Collect initial moments and save initial state
    q_dens, Ji    = sources.collect_moments(vel, Ie, W_elec, idx)    
    E_int, Ve, Te = fields.calculate_E(B, Ji, q_dens)   
    
    if save_particles == 1:
        save.save_particle_data(DT, part_save_iter, 0, pos, vel)
        
    if save_fields == 1:
        save.save_field_data(DT, field_save_iter, 0, Ji, E_int, B, Ve, Te, q_dens)
        
    vel           = particles.velocity_update(pos, vel, Ie, W_elec, idx, B, E_int, -0.5*DT)
    
    qq      = 1
    print('Starting main loop...')
    while qq < max_inc:
        ############################
        ##### EXAMINE TIMESTEP #####
        ############################
        vel, qq, DT, max_inc, part_save_iter, field_save_iter \
        = aux.check_timestep(qq, DT, pos, vel, B, E_int, q_dens, Ie, W_elec, max_inc, part_save_iter, field_save_iter, idx)

        #######################
        ###### MAIN LOOP ######
        #######################
        pos, vel, Ie, W_elec, q_dens_adv, Ji = particles.advance_particles_and_moments(pos, vel, Ie, W_elec, idx, B, E_int, DT)
        q_dens                               = 0.5 * (q_dens + q_dens_adv)
        
        B                                    = fields.push_B(B, E_int, DT)
        
        E_half, Ve, Te                       = fields.calculate_E(B, Ji, q_dens)
        
        q_dens                               = q_dens_adv.copy()

        E_int, B = fields.predictor_corrector(B, E_int, E_half, pos, vel, q_dens_adv, Ie, W_elec, idx, DT)
        
# =============================================================================
#         main_folder = 'F:\\runs\\test_optimization2\\raw_dump\\run_0\\'
#         np.savetxt(main_folder + 'E.txt', E_int)
#         np.savetxt(main_folder + 'B.txt', B)
#         np.savetxt(main_folder + 'Ji.txt', Ji)
#         np.savetxt(main_folder + 'q_dens.txt', q_dens)
#         np.savetxt(main_folder + 'Ve.txt', Ve)
#         np.savetxt(main_folder + 'Te.txt', Te)
#         
#         np.savetxt(main_folder + 'pos.txt', pos)
#         np.savetxt(main_folder + 'vel_x.txt', vel[0, :])
#         np.savetxt(main_folder + 'vel_y.txt', vel[1, :])
#         np.savetxt(main_folder + 'vel_z.txt', vel[2, :])
#         pdb.set_trace()
# =============================================================================
        
        
        ########################
        ##### OUTPUT DATA  #####
        ########################
        if qq%part_save_iter == 0 and save_particles == 1:
            save.save_particle_data(DT, part_save_iter, qq, pos, vel)

        if qq%field_save_iter == 0 and save_fields == 1:
            save.save_field_data(DT, field_save_iter, qq, Ji, E_int, B, Ve, Te, q_dens)
            
        if qq%25 == 0:
            print('Timestep {} of {} complete'.format(qq, max_inc))

        qq += 1
    print("Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2)))