## PYTHON MODULES ##
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
import save_routines as save

import diagnostics   as diag

from simulation_parameters_1D import save_particles, save_fields
import pdb

if __name__ == '__main__':
    start_time = timer()
    
    # Initialize simulation: Allocate memory and set time parameters
    pos, vel, Ie, W_elec, Ib, W_mag, idx                = init.initialize_particles()
    B, E_int, E_half, Ve, Te, damping_array             = init.initialize_fields()
    q_dens, q_dens_adv, Ji, ni, nu                      = init.initialize_source_arrays()
    old_particles, old_fields, temp3De, temp3Db, temp1D = init.initialize_tertiary_arrays()
    
    # Collect initial moments and save initial state
    sources.collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu, temp1D) 
    fields.calculate_E(B, Ji, q_dens, E_int, Ve, Te, temp3De, temp3Db, temp1D)
    
    DT, max_inc, part_save_iter, field_save_iter        = init.set_timestep(vel, E_int)
    
    if save_particles == 1:
        save.save_particle_data(0, DT, part_save_iter, 0, pos, vel)
        
    if save_fields == 1:
        save.save_field_data(0, DT, field_save_iter, 0, Ji, E_int, B, Ve, Te, q_dens)
    
    #diag.check_source_term_boundaries(q_dens, Ji)
    #diag.save_diagnostic_plots(0, pos, vel, B, E_int, q_dens, Ji, 0, DT)
    
    # Retard velocity
    particles.assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)
    particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E_int, -0.5*DT)
    
    qq       = 1;    sim_time = DT
    print('Starting main loop...')
    while qq < max_inc:
        qq, DT, max_inc, part_save_iter, field_save_iter =               \
        aux.main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag,              \
              B, E_int, E_half, q_dens, q_dens_adv, Ji, ni, nu,          \
              Ve, Te, temp3De, temp3Db, temp1D, old_particles, old_fields,\
              damping_array, qq, DT, max_inc, part_save_iter, field_save_iter)

        if qq%part_save_iter == 0 and save_particles == 1:
            save.save_particle_data(sim_time, DT, part_save_iter, qq, pos, vel)
            
        if qq%field_save_iter == 0 and save_fields == 1:
            save.save_field_data(sim_time, DT, field_save_iter, qq, Ji, E_int, B, Ve, Te, q_dens)
        
        if qq%20 == 0:
            diag.save_diagnostic_plots(qq, pos, vel, B, E_int, q_dens, Ji, sim_time, DT)
            
        print('Timestep {} of {} complete'.format(qq, max_inc))
        qq       += 1
        sim_time += DT
        
    print("Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2)))