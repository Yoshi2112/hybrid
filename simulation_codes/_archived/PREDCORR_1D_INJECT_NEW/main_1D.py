## PYTHON MODULES ##
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
import save_routines as save

from simulation_parameters_1D import save_particles, save_fields, te0_equil

import pdb

if __name__ == '__main__':
    start_time = timer()
    
    # Initialize simulation: Allocate memory and set time parameters
    pos, vel, Ie, W_elec, Ib, W_mag, idx                = init.initialize_particles()
    B, E_int, E_half, Ve, Te, Te0                       = init.initialize_fields()
    q_dens, q_dens_adv, Ji, ni, nu                      = init.initialize_source_arrays()
    old_particles, old_fields, temp3De, temp3Db, temp1D = init.initialize_tertiary_arrays()
    
    # Collect initial moments and save initial state
    sources.collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu, temp1D) 
    
    if te0_equil == True:
        init.set_equilibrium_te0(q_dens, Te0)
    
    fields.calculate_E(B, Ji, q_dens, E_int, Ve, Te, Te0, temp3De, temp3Db, temp1D)
    
# =============================================================================
#     if False:
#         import matplotlib.pyplot as plt
#         import sys
#         
#         fig, axes = plt.subplots(3, figsize=(15, 10), sharex=True)
#         axes[0].set_title('Initial E field with zero derivative at ND-NX interface')
#                 
#         axes[0].plot(E_int[:, 0])
#         axes[0].set_ylabel('Ex (mV/m)')
#         axes[1].plot(E_int[:, 1])
#         axes[1].set_ylabel('Ey (mV/m)')
#         axes[2].plot(E_int[:, 2])
#         axes[2].set_ylabel('Ez (mV/m)')
#         
#         axes[2].set_xlabel('Cell #')
#         sys.exit()
# =============================================================================
    
    DT, max_inc, part_save_iter, field_save_iter, damping_array = init.set_timestep(vel, E_int, Te0)
    
    if save_particles == 1:
        save.save_particle_data(0, DT, part_save_iter, 0, pos, vel, idx)
        
    if save_fields == 1:
        save.save_field_data(0, DT, field_save_iter, 0, Ji, E_int, B, Ve, Te, q_dens, damping_array)
    
    # Retard velocity
    
    print('Retarding velocity...')
    particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E_int, -0.5*DT)

    qq       = 1;    sim_time = DT
    print('Starting main loop...')
    while qq < max_inc:
        qq, DT, max_inc, part_save_iter, field_save_iter =               \
        aux.main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag,              \
              B, E_int, E_half, q_dens, q_dens_adv, Ji, ni, nu,          \
              Ve, Te, Te0, temp3De, temp3Db, temp1D, old_particles, old_fields,\
              damping_array, qq, DT, max_inc, part_save_iter, field_save_iter)

        if qq%part_save_iter == 0 and save_particles == 1:
            save.save_particle_data(sim_time, DT, part_save_iter, qq, pos,
                                    vel, idx)
            
        if qq%field_save_iter == 0 and save_fields == 1:
            save.save_field_data(sim_time, DT, field_save_iter, qq, Ji, E_int,
                                 B, Ve, Te, q_dens, damping_array)
        
        if qq%100 == 0:            
            running_time = int(timer() - start_time)
            hrs          = running_time // 3600
            rem          = running_time %  3600
            
            mins         = rem // 60
            sec          = rem %  60
            print('Step {} of {} :: Current runtime {:02}:{:02}:{:02}'.format(qq, max_inc, hrs, mins, sec))
            
        qq       += 1
        sim_time += DT
    
    runtime = round(timer() - start_time,2)
    
    if save_fields == 1 or save_particles == 1:
        save.add_runtime_to_header(runtime)
    print("Time to execute program: {0:.2f} seconds".format(runtime))