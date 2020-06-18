## PYTHON MODULES ##
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
import save_routines as save

import diagnostics as diag

from simulation_parameters_1D import save_particles, save_fields, NC, Nj
import sys

# TODO:
# -- Check initial moments (i.e. verify uniform and expected charge density, zero transverse current density)
# -- Vectorise/Optimize particle loss/injection
# -- Test run :: Does it compile and execute?

if __name__ == '__main__':
    start_time = timer()
    
    # Initialize simulation: Allocate memory and set time parameters
    pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp,temp_N = init.initialize_particles()
    B, E_int, E_half, Ve, Te, Te0                       = init.initialize_fields()
    q_dens, q_dens_adv, Ji, ni, nu, Pi                  = init.initialize_source_arrays()
    old_particles, old_fields, old_moments, flux_rem, \
             temp3De, temp3Db, temp1D, v_prime, S, T    = init.initialize_tertiary_arrays()
     
    # Collect initial moments and save initial state
    sources.collect_velocity_moments(pos, vel, Ie, W_elec, idx, nu, Ji, Pi) 
    sources.collect_position_moment(pos, Ie, W_elec, idx, q_dens, ni)

    DT, max_inc, part_save_iter, field_save_iter, B_damping_array, E_damping_array\
        = init.set_timestep(vel, Te0)
    
    fields.calculate_E(B, Ji, q_dens, E_int, Ve, Te, Te0, temp3De, temp3Db, temp1D, E_damping_array)
    
    if save_particles == 1:
        save.save_particle_data(0, DT, part_save_iter, 0, pos, vel, idx)
        
    if save_fields == 1:
        save.save_field_data(0, DT, field_save_iter, 0, Ji, E_int,\
                             B, Ve, Te, q_dens, B_damping_array, E_damping_array)

    # Retard velocity
    print('Retarding velocity...')
    particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E_int, v_prime, S, T, temp_N, -0.5*DT)
    print('<< Initialization Complete >>\n')
    if False:
        qq       = 1;    sim_time = DT
        print('Starting main loop...')
        while qq < max_inc:
            ###########################
            ####### MAIN LOOP #########
            ###########################
            
            # Check timestep
            qq, DT, max_inc, part_save_iter, field_save_iter, damping_array \
            = aux.check_timestep(pos, vel, B, E_int, q_dens, Ie, W_elec, Ib, W_mag, temp3De, Ep, Bp, v_prime, S, T,temp_N,\
                             qq, DT, max_inc, part_save_iter, field_save_iter, idx, B_damping_array)
            
            # Move particles, collect moments, delete or inject new particles
            particles.advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, v_prime, S, T,temp_N,\
                                                    B, E_int, DT, q_dens_adv, Ji, ni, nu, Pi, flux_rem)
            
            # Average N, N + 1 densities (q_dens at N + 1/2)
            q_dens *= 0.5
            q_dens += 0.5 * q_dens_adv
            
            # Push B from N to N + 1/2
            fields.push_B(B, E_int, temp3Db, DT, qq, B_damping_array, half_flag=1)
            
            # Calculate E at N + 1/2
            fields.calculate_E(B, Ji, q_dens, E_half, Ve, Te, Te0, temp3De, temp3Db, temp1D, E_damping_array)
        
            ###################################
            ### PREDICTOR CORRECTOR SECTION ###
            ###################################
        
            # Store old values
            old_particles[0:3 , :] = pos
            old_particles[3:6 , :] = vel
            old_particles[6   , :] = Ie
            old_particles[7   , :] = W_elec
            old_particles[8   , :] = idx
            
            old_fields[:,   0:3]  = B
            old_fields[:NC, 3:6]  = Ji
            old_fields[:NC, 6:9]  = Ve
            old_fields[:NC,   9]  = Te
            
            # Note: This could be shortened to potentially increase speed later, if desired.
            # But probably wouldn't do much compared to particle quantities.
            for jj in range(Nj):
                old_moments[:,    0 , jj]  = ni[:, jj]
                old_moments[:,  1:4 , jj]  = nu[:, jj, :]
                old_moments[:,  4:7 , jj]  = Pi[:, jj, 0, :]
                old_moments[:,  7:10, jj]  = Pi[:, jj, 1, :]
                old_moments[:, 10:13, jj]  = Pi[:, jj, 2, :]
            
            # Predict fields
            E_int *= -1.0
            E_int +=  2.0 * E_half
            
            fields.push_B(B, E_int, temp3Db, DT, qq, B_damping_array, half_flag=0)
        
            # Advance particles to obtain source terms at N + 3/2
            particles.advance_particles_and_moments(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, v_prime, S, T,temp_N,\
                                                    B, E_int, DT, q_dens, Ji, ni, nu, Pi, flux_rem, pc=1)
            
            q_dens *= 0.5;    q_dens += 0.5 * q_dens_adv
            
            # Compute predicted fields at N + 3/2
            fields.push_B(B, E_int, temp3Db, DT, qq + 1, B_damping_array, half_flag=1)
            fields.calculate_E(B, Ji, q_dens, E_int, Ve, Te, Te0, temp3De, temp3Db, temp1D, E_damping_array)
            
            # Determine corrected fields at N + 1 
            E_int *= 0.5;    E_int += 0.5 * E_half
        
            # Restore old values: [:] allows reference to same memory (instead of creating new, local instance)
            pos[:]    = old_particles[0:3 , :]
            vel[:]    = old_particles[3:6 , :]
            Ie[:]     = old_particles[6   , :]
            W_elec[:] = old_particles[7   , :]
            idx[:]    = old_particles[8   , :]
            
            B[:]      = old_fields[:,   0:3]
            Ji[:]     = old_fields[:NC, 3:6]
            Ve[:]     = old_fields[:NC, 6:9]
            Te[:]     = old_fields[:NC,   9]
            
            for jj in range(Nj):
                ni[:, jj]       = old_moments[:,    0 , jj]
                nu[:, jj, :]    = old_moments[:,  1:4 , jj]  
                Pi[:, jj, 0, :] = old_moments[:,  4:7 , jj]  
                Pi[:, jj, 1, :] = old_moments[:,  7:10, jj]  
                Pi[:, jj, 2, :] = old_moments[:, 10:13, jj]  
            
            fields.push_B(B, E_int, temp3Db, DT, qq, B_damping_array, half_flag=0)   # Advance the original B
        
            q_dens[:] = q_dens_adv
    
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
