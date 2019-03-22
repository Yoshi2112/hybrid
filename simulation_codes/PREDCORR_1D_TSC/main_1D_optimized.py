## PYTHON MODULES ##
import numba as nb
from timeit import default_timer as timer

## HYBRID MODULES ##
import init_1D       as init
import auxilliary_1D as aux
import particles_1D  as particles
import fields_1D     as fields
import sources_1D    as sources
import save_routines as save

from simulation_parameters_1D import generate_data, NX


@nb.njit()
def initialize():
    pos, vel, Ie, W_elec, idx   = init.initialize_particles()
    B, E_int                    = init.initialize_fields()
    DT, max_inc, data_iter      = aux.set_timestep(vel)
    
    if data_iter == 0:
        data_iter = max_inc
    
    q_dens, Ji                  = sources.collect_moments(vel, Ie, W_elec, idx)
    
    E_int, Ve, Te               = fields.calculate_E(B, Ji, q_dens)
    vel                         = particles.velocity_update(pos, vel, Ie, W_elec, idx, B, E_int, -0.5*DT)
    return pos, vel, Ie, W_elec, idx, B, E_int, q_dens, Ji, Ve, Te, DT, max_inc, data_iter 


@nb.njit()
def numerical_loop(real_time, pos, vel, Ie, W_elec, idx, B, E_int, q_dens, Ji, Ve, Te, DT, max_inc, data_iter, ch_flag):
    '''
    Does the number crunching for a short snippet. Logs number of time variable changes in ch_flag as powers of 2
    (-1 = half, 2 = 4 times slower)
    
    Array values are mutable: Don't have to be returned. Only integer values
    '''
    
    qq = 0
    while qq < data_iter:
        # Check timestep used for this iteration
        vel, qq, DT, max_inc, data_iter, ch_flag = aux.check_timestep(qq, DT, pos, vel, B, E_int, q_dens, Ie, W_elec, max_inc, data_iter, idx)
        
        # Add timestep to counter
        real_time += DT
        
        # Main loop
        pos, vel, Ie, W_elec, q_dens_adv, Ji = particles.advance_particles_and_moments(pos, vel, Ie, W_elec, idx, B, E_int, DT)
        q_dens                               = 0.5 * (q_dens + q_dens_adv)
        B                                    = fields.push_B(B, E_int, DT)
        E_half, Ve, Te                       = fields.calculate_E(B, Ji, q_dens)
        q_dens                               = q_dens_adv.copy()
        
        # Predictor-Corrector: Advance fields to start of next timestep
        E_int, B = fields.predictor_corrector(B, E_int, E_half, pos, vel, q_dens_adv, Ie, W_elec, idx, DT)
        
        # Increment loop variable
        qq += 1
    return DT, ch_flag, max_inc, data_iter, real_time


def main():
    pos, vel, Ie, W_elec, idx, B, E_int, q_dens, Ji, Ve, Te, DT, max_inc, data_iter  = initialize()
    
    if generate_data == 1:
        save.store_run_parameters(DT, data_iter)
        
    print('Timestep: %.4fs, %d iterations total' % (DT, max_inc))
    print('Initial source term check:')
    print('Average cell density: {}cc'.format(q_dens[1: NX + 1].mean()))
    print('Average cell current: {}A/m'.format(Ji[1: NX + 1].mean()))

    xx       = 0; ch_flag  = 0; real_time = 0.
    max_inc = 50; data_iter = 25
    while xx < max_inc:
        xx += data_iter     # Iterate now before data_iter changed in numerical_loop 
                            # If dt has changed, it will still have 'done' this many iterations

        DT, ch_flag, max_inc, data_iter, real_time = numerical_loop(real_time, pos, vel, Ie, W_elec, idx, B, E_int, q_dens, Ji, Ve, Te, DT, max_inc, data_iter, ch_flag)
                
        if ch_flag != 0:          
            print('Timestep changed by factor of {}. New DT = {}'.format(2 ** (ch_flag), DT))
            
        if generate_data == 1:
            if xx%data_iter == 0:
                save.save_data(real_time, DT, data_iter, xx, pos, vel, Ji, E_int, B, Ve, Te, q_dens)

        print('Timestep {} of {} complete'.format(xx, max_inc))
    return


if __name__ == '__main__':
    start_time = timer()
    main()
    print("Time to execute program: {0:.2f} seconds".format(round(timer() - start_time,2)))  # Time taken to run simulation