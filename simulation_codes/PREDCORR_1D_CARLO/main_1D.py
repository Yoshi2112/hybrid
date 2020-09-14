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


if __name__ == '__main__':
    start_time = timer()
    
    # Initialize simulation: Allocate memory and set time parameters
    print('Initializing arrays...')
    pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp,temp_N = init.initialize_particles()
    B, E_int, E_half, Ve, Te, Te0                       = init.initialize_fields()
    q_dens, q_dens_adv, Ji, ni, nu                      = init.initialize_source_arrays()
    old_particles, old_fields, temp3De, temp3Db, temp1D,\
                                v_prime, S, T, mp_flux  = init.initialize_tertiary_arrays()
    
    # Collect initial moments and save initial state
    sources.collect_moments(vel, Ie, W_elec, idx, q_dens, Ji, ni, nu) 
    
    if te0_equil == 1:
        init.set_equilibrium_te0(q_dens, Te0)
    
    DT, max_inc, part_save_iter, field_save_iter, B_damping_array, E_damping_array\
        = init.set_timestep(vel, Te0)
        
    fields.calculate_E(B, Ji, q_dens, E_int, Ve, Te, Te0, temp3De, temp3Db, temp1D, E_damping_array, 0, DT, 0)
    
    if save_particles == 1:
        save.save_particle_data(0, DT, part_save_iter, 0, pos, vel, idx)
        
    if save_fields == 1:
        save.save_field_data(0, DT, field_save_iter, 0, Ji, E_int,\
                             B, Ve, Te, q_dens, B_damping_array, E_damping_array)

    # Retard velocity
    print('Retarding velocity...')
    particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E_int, v_prime, S, T, temp_N, -0.5*DT)

    qq       = 1;    sim_time = DT
    print('Starting main loop...')
    while qq < max_inc:
        qq, DT, max_inc, part_save_iter, field_save_iter =                                \
        aux.main_loop(pos, vel, idx, Ie, W_elec, Ib, W_mag, Ep, Bp, v_prime, S, T, temp_N,\
              B, E_int, E_half, q_dens, q_dens_adv, Ji, ni, nu, mp_flux,                  \
              Ve, Te, Te0, temp3De, temp3Db, temp1D, old_particles, old_fields,           \
              B_damping_array, E_damping_array, qq, DT, max_inc, part_save_iter, field_save_iter)

        if qq%part_save_iter == 0 and save_particles == 1:
            save.save_particle_data(sim_time, DT, part_save_iter, qq, pos,
                                    vel, idx)
            
        if qq%field_save_iter == 0 and save_fields == 1:
            save.save_field_data(sim_time, DT, field_save_iter, qq, Ji, E_int,
                                 B, Ve, Te, q_dens, B_damping_array, E_damping_array)
        
        if qq%50 == 0:            
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
