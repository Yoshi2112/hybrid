## PYTHON MODULES ##
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
import plot_and_save as pas

#import diagnostics   as diag

from simulation_parameters_1D import adaptive_timestep, save_particles, save_fields


if __name__ == '__main__':
    start_time = timer()

    pos, vel, Ie, W_elec, idx = init.initialize_particles()
    B, E                      = init.initialize_fields()

    DT, max_inc, part_save_iter, field_save_iter, subcycles = aux.set_timestep(vel)

    print('Loading initial state...\n')
    pos, Ie, W_elec, dns_int, dns_half, J_plus, J_minus, G, L   = sources.init_collect_moments(pos, vel, Ie, W_elec, idx, 0.5*DT)

    qq      = 0
    print('Starting loop...')
    while qq < max_inc:
        ############################
        ##### EXAMINE TIMESTEP #####
        ############################
        if adaptive_timestep == 1:
            pos, qq, DT, max_inc, part_save_iter, field_save_iter, change_flag, subcycles = aux.check_timestep(qq, DT, pos, vel, B, E, dns_int, max_inc, part_save_iter, field_save_iter, subcycles)
    
            if change_flag == 1:
                print('Timestep halved. Syncing particle velocity/position with DT = {}'.format(DT))
                pos, Ie, W_elec, dns_int, dns_half, J_plus, J_minus, G, L   = sources.init_collect_moments(pos, vel, Ie, W_elec, idx, 0.5*DT)
            elif change_flag == 2:
                print('Timestep doubled. Syncing particle velocity/position with DT = {}'.format(DT))
                pos, Ie, W_elec, dns_int, dns_half, J_plus, J_minus, G, L   = sources.init_collect_moments(pos, vel, Ie, W_elec, idx, 0.5*DT)
        
        
        #######################
        ###### MAIN LOOP ######
        #######################
        B         = fields.cyclic_leapfrog(B, dns_int, J_minus, DT, subcycles)
        E, Ve, Te = fields.calculate_E(B, J_minus, dns_half)
        J         = sources.push_current(J_plus, E, B, L, G, DT)
        E, Ve, Te = fields.calculate_E(B, J, dns_half)

        vel = particles.velocity_update(pos, vel, Ie, W_elec, idx, B, E, J, DT)

        # Store pc(1/2) here while pc(3/2) is collected
        dns_int = dns_half          
        pos, Ie, W_elec, dns_half, J_plus, J_minus, G, L = sources.collect_moments(pos, vel, Ie, W_elec, idx, DT)
        
        dns_int = 0.5 * (dns_int + dns_half)
        J       = 0.5 * (J_plus  +  J_minus)
        
        B           = fields.cyclic_leapfrog(B, dns_int, J, DT, subcycles)
        E, Ve, Te   = fields.calculate_E(B, J, dns_int)                                     # This one's just for output


        ########################
        ##### OUTPUT DATA  #####
        ########################
        if qq%part_save_iter == 0 and save_particles == 1:                                   # Save data, if flagged
            pas.save_particle_data(DT, part_save_iter, qq, pos, vel)

        if qq%field_save_iter == 0 and save_fields == 1:                                   # Save data, if flagged
            pas.save_field_data(DT, field_save_iter, qq, J, E, B, Ve, Te, dns_int)

        if (qq + 1)%25 == 0:
            print('Timestep {} of {} complete'.format(qq + 1, max_inc))

        qq += 1

    print("Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2)))  # Time taken to run simulation