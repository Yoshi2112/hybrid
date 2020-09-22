## PYTHON MODULES ##
from timeit import default_timer as timer
import numpy as np
import pdb, sys

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
import save_routines as save

from simulation_parameters_1D import adaptive_timestep, save_particles, save_fields, Bc, size, N, Nj


if __name__ == '__main__':
    start_time = timer()
    
    # Initialize arrays and initial conditions
    Ie       = np.zeros(N,      dtype=np.uint16)
    Ib       = np.zeros(N,      dtype=np.uint16)
    W_elec   = np.zeros((3, N), dtype=np.float64)
    W_mag    = np.zeros((3, N), dtype=np.float64)
    
    Bp       = np.zeros((3, N), dtype=np.float64)
    Ep       = np.zeros((3, N), dtype=np.float64)
    temp_N   = np.zeros((N),    dtype=np.float64)
    
    v_prime  = np.zeros((3, N), dtype=np.float64)
    S        = np.zeros((3, N), dtype=np.float64)
    T        = np.zeros((3, N), dtype=np.float64)
    
    ni       = np.zeros((size, Nj), dtype=np.float64)
    ni_init  = np.zeros((size, Nj), dtype=np.float64)
    
    nu_init  = np.zeros((size, Nj, 3), dtype=np.float64)
    nu_plus  = np.zeros((size, Nj, 3), dtype=np.float64)
    nu_minus = np.zeros((size, Nj, 3), dtype=np.float64)
    
    rho_half = np.zeros(size, dtype=np.float64)
    rho_int  = np.zeros(size, dtype=np.float64)
    
    J        = np.zeros((size, 3), dtype=np.float64)
    J_plus   = np.zeros((size, 3), dtype=np.float64)
    J_minus  = np.zeros((size, 3), dtype=np.float64)
    L        = np.zeros( size,     dtype=np.float64)
    G        = np.zeros((size, 3), dtype=np.float64)
    
    B        = np.zeros((size, 3), dtype=np.float64)
    B2       = np.zeros((size, 3), dtype=np.float64)
    E        = np.zeros((size, 3), dtype=np.float64)
    Ve       = np.zeros((size, 3), dtype=np.float64)
    Te       = np.zeros((size   ), dtype=np.float64)

    temp3d   = np.zeros((size, 3), dtype=np.float64)

    if False:
        pos, idx = init.uniform_distribution()
        vel      = init.gaussian_distribution()
    else:
        pos, vel, idx = init.quiet_start()
    
    B[:, 0]  = Bc[0]      # Set Bx initial
    B[:, 1]  = Bc[1]      # Set By initial
    B[:, 2]  = Bc[2]      # Set Bz initial
    
    particles.assign_weighting_TSC(pos, Ie, W_elec)

    DT, max_inc, part_save_iter, field_save_iter, subcycles = aux.set_timestep(vel)

    print('Loading initial state...\n')
    sources.init_collect_moments(pos, vel, Ie, W_elec, idx, ni_init, nu_init, ni, nu_plus, 
                         rho_int, rho_half, J, J_plus, L, G, 0.5*DT)

    # Put init into qq = 0 and save as usual, qq = 1 will be at t = dt
    qq      = 0
    print('Starting loop...')
    while qq < max_inc:
        ############################
        ##### EXAMINE TIMESTEP #####
        ############################
        if adaptive_timestep == 1:
            qq, DT, max_inc, part_save_iter, field_save_iter, change_flag, subcycles =\
                aux.check_timestep(qq, DT, pos, vel, Ie, W_elec, B, E, rho_int, max_inc, part_save_iter, field_save_iter, subcycles)
    
            # Collect new moments and desync position and velocity
            if change_flag == 1:
                sources.init_collect_moments(pos, vel, Ie, W_elec, idx, ni_init, nu_init, ni, nu_plus, 
                         rho_int, rho_half, J, J_plus, L, G, 0.5*DT)
        
        # Debug: Test if everything is the same if J is replaced with J_minus at each loop.
        # Yes it is after loop 0 and loop 1 up until collect_moments()
        # Disable this at some point and see if it improves (or even changes) anything.
# =============================================================================
#         if qq > 0:
#             J[:, :] = J_minus[:, :]
# =============================================================================
        
        #######################
        ###### MAIN LOOP ######
        #######################
        fields.cyclic_leapfrog(B, B2, rho_int, J, temp3d, DT, subcycles)
        E, Ve, Te = fields.calculate_E(B, J, rho_half)

        sources.push_current(J_plus, J, E, B, L, G, DT)
        E, Ve, Te = fields.calculate_E(B, J, rho_half)
        
        particles.velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T, temp_N, DT)

        # Store pc(1/2) here while pc(3/2) is collected
        rho_int[:]  = rho_half[:] 
        sources.collect_moments(pos, vel, Ie, W_elec, idx, ni, nu_plus, nu_minus, 
                                             rho_half, J_minus, J_plus, L, G, DT)
        
        rho_int += rho_half
        rho_int /= 2.0
        J        = 0.5 * (J_plus  +  J_minus)

        fields.cyclic_leapfrog(B, B2, rho_int, J, temp3d, DT, subcycles)
        E, Ve, Te   = fields.calculate_E(B, J, rho_int)                                     # This one's just for output

        ########################
        ##### OUTPUT DATA  #####
        ########################
        if qq%part_save_iter == 0 and save_particles == 1:                                   # Save data, if flagged
            save.save_particle_data(DT, part_save_iter, qq, pos, vel)

        if qq%field_save_iter == 0 and save_fields == 1:                                   # Save data, if flagged
            save.save_field_data(DT, field_save_iter, qq, J, E, B, Ve, Te, rho_int)
        
        if qq%50 == 0:
            running_time = int(timer() - start_time)
            hrs          = running_time // 3600
            rem          = running_time %  3600
            
            mins         = rem // 60
            sec          = rem %  60
            
            print('Step {} of {} :: Current runtime {:02}:{:02}:{:02}'.format(qq, max_inc, hrs, mins, sec))

# =============================================================================
#         if qq == 100:
#             aux.dump_to_file(pos, vel, E, Ve, Te, B, J, J_minus, J_plus, rho_int, rho_half, qq, suff='')
#             sys.exit()
# =============================================================================
        
        qq += 1
        
    # Resync particle positions for end step? Not really necessary.
    end_time = int(timer() - start_time)
    hrs          = running_time // 3600
    rem          = running_time %  3600
    
    mins         = rem // 60
    sec          = rem %  60
    print('Time to execute program :: {:02}:{:02}:{:02}'.format(hrs, mins, sec))  # Time taken to run simulation
