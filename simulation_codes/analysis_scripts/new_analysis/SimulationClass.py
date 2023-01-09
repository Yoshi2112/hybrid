# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:52:11 2022

@author: Yoshi
"""
import os, sys, pickle, shutil
import numpy as np   
import numba as nb

'''
Refactored version of the analysis software used for the hybrid simulation.
Main difference is changing the focus from using global variables to using 
instances of a Simulation class which contain all the variables for that run.

Most changes will be based on passing simulation instances to functions that
will then plot the desired output. Use this as an opportunity to create modules
that do certain specific plot types, now that we don't have to worry about 
the global namespace. This way, code is much cleaner and I can pass Simulation
instances to external scripts, rather than having them all in the one huge file.

TODO:
    -- Break huge class up into smaller classes that have specific jobs, e.g.
    -- ParameterLoader clor header/runfile stuff
    -- FieldLoader and Particass fleLoader, or just DataLoader
    -- AnalysisToolkit or HybridAnalysis, for calculating things like energies, helicities, etc.
    
Maybe even make a SummaryPlot or StandardAnalysisOutput class that calls some
plot routines from another script (after they're rewritten to accept an instance)
'''
# Constants
UNIT_CHARGE        = 1.602e-19               # Elementary charge (C)
PROTON_MASS        = 1.673e-27               # Mass of proton (kg)
ELECTRON_MASS      = 9.109e-31               # Mass of electron (kg)
ELEC_PERMITTIVITY  = 8.854e-12               # Electric permittivity of free space (SI units)
MAGN_PERMEABILITY  = (4e-7) * np.pi          # Magnetic permeability of Free Space (SI units)
BOLTZMANN_CONSTANT = 1.381e-23               # Boltmann's constant in J/K
LIGHT_SPEED        = 3e8


class HybridSimulationRun:
    '''
    Represents a single run of the hybrid code. Contains parameters, paths, 
    pointers to files, and methods to load and parse simulation run data
    
    Store parameters as attributes
    '''
    home_dir = 'E:/runs/'   # Contains drive information. Is this the correct way to use this?
                            # E.g. class attribute vs. instance attribute?
    
    def __init__(self, series_name, run_num, extract_arrays=True, overwrite_summary=False):
        self.series_name = series_name
        self.run_num = run_num
        self.base_dir = self.home_dir + series_name + '/'   # Main series directory, containing runs
        self.run_dir  = self.base_dir + f'/run_{run_num}/'  # Main run directory
        
        # Check existence of series and run number
        if not os.path.exists(self.base_dir): raise OSError(f'Simulation run does not exist : {self.base_dir}')
        if not os.path.exists(self.run_dir ): raise OSError(f'Run {run_num} does not exist for series {self.series_name}')
        
        print(f'Loading HybridRun {self.series_name}[{self.run_num}]')
        self.create_directories()
        self.load_run_params()
        self.load_species_params()
        self.define_variables()
        
        self.create_output_summary(overwrite_summary)
        self.extract_fields()
        self.init_set_mmaps()
        
    def __str__(self): return f'Hybrid simulation :: Series {self.series_name} :: Run {self.run_num}'
    
    def create_directories(self):
        '''
        Create directory trees when run is loaded.
        '''                
        self.data_dir = self.run_dir + 'data/'                # Directory containing .npz output files for the simulation run
        self.anal_dir = self.run_dir + 'analysis/'            # Output directory for all this analysis (each will probably have a subfolder)
        self.temp_dir = self.run_dir + 'extracted/'           # Saving things like matrices so we only have to do them once

        self.field_dir    = self.data_dir + '/fields/'
        self.particle_dir = self.data_dir + '/particles/'
        self.equil_dir    = self.data_dir + '/equil_particles/'

        # Make Output folders if they don't exist
        for this_dir in [self.anal_dir, self.temp_dir]:
            if os.path.exists(this_dir) == False:
                print(f'Creating directory {this_dir}')
                os.makedirs(this_dir)
        return
    
    def load_species_params(self):
        '''
        Load header file and store parameters in instance attributes
        Should these be instances of a Species class?
        '''
        print('Loading species parameters...')
        p_data = np.load(self.data_dir+'particle_parameters.npz', allow_pickle=True)

        self.species_lbl= p_data['species_lbl']
        self.temp_color = p_data['temp_color']
        self.temp_type  = p_data['temp_type']
        self.dist_type  = p_data['dist_type']
        
        self.mass       = p_data['mass']
        self.charge     = p_data['charge']
        self.drift_v    = p_data['drift_v']
        self.nsp_ppc    = p_data['nsp_ppc']
        self.density    = p_data['density']
        self.N_species  = p_data['N_species']
        self.nsp_ppc    = p_data['nsp_ppc']
        
        self.Bc         = p_data['Bc']
        
        self.idx_start0 = p_data['idx_start']
        self.idx_end0   = p_data['idx_end']

        self.Tpara      = p_data['Tpar']
        self.Tperp      = p_data['Tperp'] if 'Tperp' in p_data.keys() else p_data['Tper']
        self.anisotropy = self.Tperp / self.Tpara - 1.
        
        self.vth_par     = p_data['vth_par']
        self.vth_perp    = p_data['vth_perp']

        try:
            self.Te0_arr = p_data['Te0']
        except:
            self.Te0_arr = np.ones(self.NC, dtype=np.float64) * self.Te0

        self.n_contr    = self.density / self.nsp_ppc  
        self.species_present = [False, False, False]                # Test for the presence of singly charged H, He, O
        for ii in range(self.Nj):
            if 'H^+' in self.species_lbl[ii]:
                self.species_present[0] = True
            elif 'He^+' in self.species_lbl[ii]:
                self.species_present[1] = True
            elif 'O^+'  in self.species_lbl[ii]:
                self.species_present[2] = True
                
        self.rho_mass   = (self.mass*self.density).sum()            # Mass density
        return
    
    
    def load_run_params(self):
        print('Loading run parameters...')
        f      = open(self.data_dir+'simulation_parameters.pckl', 'rb')
        obj    = pickle.load(f)
        f.close()
        
        self.seed           = obj['seed']
        self.Nj             = obj['Nj']
        self.dt_sim         = obj['dt']
        self.NX             = obj['NX']
        self.ND             = obj['ND']
        self.NC             = obj['NC']
        self.N              = obj['N']
        self.dxm            = obj['dxm']
        self.dx             = obj['dx']
        self.ne             = obj['ne']
        self.Te0            = obj['Te0']
        self.ie             = obj['ie']
        self.xmin           = obj['xmin']
        self.xmax           = obj['xmax']
        self.B_xmax         = obj['B_xmax']
        self.B_eq           = obj['B_eq']
        self.a              = obj['a']
        self.L              = obj['L']
        self.loss_cone      = obj['loss_cone']
        self.rc_hwidth      = obj['rc_hwidth']
        self.theta_xmax     = obj['theta_xmax']
        self.orbit_res      = obj['orbit_res']
        self.freq_res       = obj['freq_res']
        self.run_desc       = obj['run_desc']
        self.method_type    = obj['method_type'] 
        self.particle_shape = obj['particle_shape']
        self.max_inc        = obj['max_inc']
        if 'max_wcinv' in obj.keys():
            self.max_wcinv     = obj['max_wcinv']
        else:
            self.max_wcinv     = obj['max_rev'] / (2*np.pi)
        
        if 'damping_multiplier' in obj.keys():
            self.damping_multiplier = obj['damping_multiplier']
        else:
            self.damping_multiplier = None

        self.part_save_iter    = obj['part_save_iter']
        self.field_save_iter   = obj['field_save_iter']
        self.dt_field          = self.dt_sim * self.field_save_iter
        self.dt_particle       = self.dt_sim * self.part_save_iter
        self.subcycles         = obj['subcycles']
        
        self.particle_periodic = obj['particle_periodic']
        self.particle_reflect  = obj['particle_reflect']
        self.particle_reinit   = obj['particle_reinit']

        if self.particle_reinit + self.particle_reflect + self.particle_periodic == 0:
            self.particle_open = 1
        else:
            self.particle_open = 0
            
        self.disable_waves    = obj['disable_waves']
        self.source_smoothing = obj['source_smoothing']
        self.E_damping        = obj['E_damping']
        self.quiet_start      = obj['quiet_start']
        self.homogenous       = obj['homogeneous']
        self.field_periodic   = obj['field_periodic']
        
        if obj['run_time'] is None:
            self.run_time_str = 'Incomplete'
        else:
            run_time = obj['run_time']
            hrs      = int(run_time // 3600)
            rem      = run_time %  3600
            mins     = int(rem // 60)
            sec      = round(rem %  60, 2)
            self.run_time_str = '{:02}:{:02}:{:02}'.format(hrs, mins, sec)
        
        if obj['loop_time'] is None:
            self.loop_time = 'N/A'
        else:
            self.loop_time = round(obj['loop_time'], 3)
        
        try:
            # Test if scalar
            print(obj['Te0'][0])
        except:
            # If it is, make it a vector
            self.Te0 = np.ones(self.NC, dtype=float) * self.Te0

        self.driven_freq   = obj['driven_freq']
        self.driven_ampl   = obj['driven_ampl']
        self.pulse_offset  = obj['pulse_offset']
        self.pulse_offset  = obj['pulse_offset']
        self.pulse_width   = obj['pulse_width']
        self.driven_k      = obj['driven_k']
        self.driver_status = obj['pol_wave'] 
        self.num_threads   = obj['num_threads']

        # Set spatial boundaries and gridpoints
        self.x0B, self.x1B = self.ND, self.ND + self.NX + 1
        self.x0E, self.x1E = self.ND, self.ND + self.NX
        
        self.B_nodes = (np.arange(self.NC + 1) - self.NC // 2)       * self.dx
        self.E_nodes = (np.arange(self.NC)     - self.NC // 2 + 0.5) * self.dx  
        
        # Load number of field and particle files
        self.num_field_steps = len(os.listdir(self.field_dir))
        self.num_particle_steps = len(os.listdir(self.particle_dir))
        return
    
    
    def define_variables(self):
        '''
        Initialize other derived variables such as characteristic frequencies
        and velocities
        '''
        print('Defining working variables...')
        self.wpp        = np.sqrt(self.ne * UNIT_CHARGE ** 2 / (PROTON_MASS * ELEC_PERMITTIVITY))               # Proton plasma frequency
        self.wpi        = np.sqrt((self.density * self.charge ** 2 / (self.mass * ELEC_PERMITTIVITY)).sum())    # Ion plasma frequency
        self.gyfreq     = UNIT_CHARGE * self.B_eq   / PROTON_MASS                             # Proton gyrofrequency (rad/s) (compatibility)
        self.gyfreq_eq  = UNIT_CHARGE * self.B_eq   / PROTON_MASS                             # Proton gyrofrequency (rad/s) (equator)
        self.gyfreq_xmax= UNIT_CHARGE * self.B_xmax / PROTON_MASS                             # Proton gyrofrequency (rad/s) (boundary)
        self.gyperiod   = (PROTON_MASS * 2 * np.pi) / (UNIT_CHARGE * self.B_eq)               # Proton gyroperiod (s)
        self.va_proton  = self.B_eq / np.sqrt(MAGN_PERMEABILITY*self.ne*PROTON_MASS)          # Alfven speed: Assuming pure proton plasma
        self.va         = self.B_eq / np.sqrt(MAGN_PERMEABILITY*self.rho_mass)                # Alfven speed: Accounting for actual mass density
        return
    
    
    def delete_analysis_folders(self):
        '''
        Deletes all but the raw data, in case something was analysed incorrectly
        '''
        print('Deleting analysis and temp folders.')
        # Delete directory and contents
        for directory in [self.anal_dir, self.temp_dir]:
            if os.path.exists(directory) == True:
                shutil.rmtree(directory)
        
        # Delete summary file
        param_file = self.run_dir + 'simulation_parameter_file.txt'
        if os.path.exists(param_file) == True:
            os.remove(param_file)
        return
    
    
    def create_output_summary(self, overwrite_summary):
        '''
        Saves a .txt file containing details of the instanced run in the run
        home directory.
        '''
        print('Creating output summary...')
        output_file = self.run_dir + 'simulation_parameter_file.txt'

        if self.particle_open == 1:
            self.particle_boundary = 'Open'
        elif self.particle_reinit == 1:
            self.particle_boundary = 'Reinitialize'
        elif self.particle_reflect == 1:
            self.particle_boundary = 'Reflection'
        elif self.particle_periodic == 1:
            self.particle_boundary = 'Periodic'
        else:
            self.particle_boundary = '-'

        if self.ie == 0:
            self.electron_treatment = 'Isothermal'
        elif self.ie == 1:
            self.electron_treatment = 'Adiabatic'
        else:
            self.electron_treatment = 'Other'

        echarge  = self.charge / UNIT_CHARGE
        pmass    = self.mass   / PROTON_MASS
        va_drift = self.drift_v / self.va

        if os.path.exists(output_file) == True and overwrite_summary == False:
            pass
        else:
            with open(output_file, 'w') as f:
                print('HYBRID SIMULATION :: PARAMETER FILE', file=f)
                print('', file=f)
                print('Series[run]   :: {}[{}]'.format(self.series_name, self.run_num), file=f)
                print('Series Desc.  :: {}'.format(self.run_desc), file=f)
                print('Hybrid Type   :: {}'.format(self.method_type), file=f)
                print('Random Seed   :: {}'.format(self.seed), file=f)
                print('Final runtime :: {}'.format(self.run_time_str), file=f)
                print('Av. loop time :: {}'.format(self.loop_time), file=f)
                print('N_loops_start :: {}'.format(self.max_inc), file=f)
                print('', file=f)
                print('Flags', file=f)
                print('Disable Wave Growth:: {}'.format(self.disable_waves), file=f)
                print('Source Smoothing   :: {}'.format(self.source_smoothing), file=f)
                print('E-field Damping    :: {}'.format(self.E_damping), file=f)
                print('Quiet Start        :: {}'.format(self.quiet_start), file=f)
                print('Homogenous B0      :: {}'.format(self.homogenous), file=f)
                print('Field Periodic BCs :: {}'.format(self.field_periodic), file=f)
                print('', file=f)
                print('Temporal Parameters', file=f)
                print('Maximum Sim. Time  :: {}     wcinv'.format(self.max_wcinv), file=f)
                print('Maximum Sim. Time  :: {}     seconds'.format(round(self.max_wcinv/self.gyfreq, 1)), file=f)
                print('Simulation cadence :: {:.5f} seconds'.format(self.dt_sim), file=f)
                print('Particle Dump Time :: {:.5f} seconds'.format(self.dt_particle), file=f)
                print('Field Dump Time    :: {:.5f} seconds'.format(self.dt_field), file=f)
                print('Frequency Resol.   :: {:.5f} gyroperiods'.format(self.freq_res), file=f)
                print('Gyro-orbit Resol.  :: {:.5f} gyroperiods'.format(self.orbit_res), file=f)
                print('Subcycles init.    :: {} '.format(self.subcycles), file=f)
                print('', file=f)
                print('Simulation Parameters', file=f)
                print('# Spatial Cells    :: {}'.format(self.NX), file=f)
                print('# Damping Cells    :: {}'.format(self.ND), file=f)
                print('# Cells Total      :: {}'.format(self.NC), file=f)
                print('va/pcyc per dx     :: {}'.format(self.dxm), file=f)
                print('Cell width         :: {:.1f} km'.format(self.dx*1e-3), file=f)
                print('Simulation Min     :: {:.1f} km'.format(self.xmin*1e-3), file=f)
                print('Simulation Max     :: {:.1f} km'.format(self.xmax*1e-3), file=f)
                if self.damping_multiplier is not None:
                    print('Damping Multipl.   :: {:.2f}'.format(self.damping_multiplier), file=f)
                else:
                    print('Damping Multipl.   ::', file=f)
                print('', file=f)
                print('Equatorial B0       :: {:.2f} nT'.format(self.B_eq*1e9), file=f)
                print('Boundary   B0       :: {:.2f} nT'.format(self.B_xmax*1e9), file=f)
                print('max MLAT            :: {:.2f} deg'.format(self.theta_xmax * 180. / np.pi), file=f)
                print('McIlwain L value    :: {:.2f}'.format(self.L), file=f)
                print('Parabolic s.f. (a)  :: {}'.format(self.a), file=f)
                print('', file=f)
                print('Electron Density    :: {} /cc'.format(self.ne*1e-6), file=f)
                print('Electron Treatment  :: {}'.format(self.electron_treatment), file=f)

                print('', file=f)
                print('Particle Parameters', file=f)
                print('Number of Species   :: {}'.format(self.Nj), file=f)
                print('Number of Particles :: {}'.format(self.N), file=f)
                print('Species Per Cell    :: {}'.format(self.nsp_ppc), file=f)
                print('Species Particles # :: {}'.format(self.N_species), file=f)
                print('Particle Shape Func :: {}'.format(self.particle_shape), file=f)
                print('Particle Bound. Cond:: {}'.format(self.particle_boundary), file=f)
                print('', file=f)
                
                
                print('Ion Composition', file=f)
                
                ccdens   = self.density*1e-6
                if self.vth_par is not None:
                    va_para = self.vth_par  / self.va
                    va_perp = self.vth_perp / self.va
                else:
                    va_para = np.zeros(echarge.shape)
                    va_perp = np.zeros(echarge.shape)
                
                species_str = temp_str = cdens_str = charge_str = va_perp_str = \
                va_para_str = mass_str = drift_str = contr_str = ''
                for ii in range(self.Nj):
                    species_str += '{:>13}'.format(self.species_lbl[ii])
                    temp_str    += '{:>13d}'.format(self.temp_type[ii])
                    cdens_str   += '{:>13.3f}'.format(ccdens[ii])
                    charge_str  += '{:>13.1f}'.format(echarge[ii])
                    mass_str    += '{:>13.1f}'.format(pmass[ii])
                    drift_str   += '{:>13.1f}'.format(va_drift[ii])
                    va_perp_str += '{:>13.2f}'.format(va_perp[ii])
                    va_para_str += '{:>13.2f}'.format(va_para[ii])
                    contr_str   += '{:>13.1f}'.format(self.n_contr[ii])
        
                print('Species Name    :: {}'.format(species_str), file=f)
                print('Species Type    :: {}'.format(temp_str), file=f)
                print('Species Dens    :: {}  /cc'.format(cdens_str), file=f)
                print('Species Charge  :: {}  elementary units'.format(charge_str), file=f)
                print('Species Mass    :: {}  proton masses'.format(mass_str), file=f)
                print('Drift Velocity  :: {}  vA'.format(drift_str), file=f)
                print('V_thermal Perp  :: {}  vA'.format(va_perp_str), file=f)
                print('V_thermal Para  :: {}  vA'.format(va_para_str), file=f)
                print('MParticle s.f   :: {}  real particles/macroparticle'.format(contr_str), file=f)
        return
    
    
    def extract_fields(self):
        '''
        Condense field files into single contiguous file for each field
        Note: Might change once the HDF5 stuff gets sorted
        
        TODO: Delete raw field files, or at least have an option for it
        '''
        # Check if field files exist:
        if len(os.listdir(self.field_dir)) == 0:
            print('No field files found, skipping extraction.')
            return
        
        # Check that all components are extracted
        comps_missing = 0
        for component in ['bx', 'by', 'bz', 'ex', 'ey', 'ez']:
            check_path = self.temp_dir + component + '_array.npy'
            if os.path.isfile(check_path) == False:
                comps_missing += 1
        
        if comps_missing == 0:
            print('Field components already extracted.')
            return
        else:
            num_field_steps = len(os.listdir(self.field_dir)) 
            
            # Load to specify arrays
            zB, zB_cent, zE, zVe, zTe, zJ, zq_dns, zsim_time, zdamp = self.load_fields(0)

            bx_arr, by_arr, bz_arr, damping_array = [np.zeros((num_field_steps, zB.shape[0])) for _ in range(4)]
            
            if zB_cent is not None:
                bxc_arr, byc_arr, bzc_arr = [np.zeros((num_field_steps, zB_cent.shape[0])) for _ in range(3)]
            
            ex_arr,ey_arr,ez_arr,vex_arr,jx_arr,vey_arr,jy_arr,vez_arr,jz_arr,te_arr,qdns_arr\
            = [np.zeros((num_field_steps, zE.shape[0])) for _ in range(11)]
        
            field_sim_time = np.zeros(num_field_steps)
        
            print('Extracting fields...')
            for ii in range(num_field_steps):
                sys.stdout.write('\rExtracting field timestep {}'.format(ii))
                sys.stdout.flush()
                
                B, B_cent, E, Ve, Te, J, q_dns, sim_time, damp = self.load_fields(ii)

                bx_arr[ii, :] = B[:, 0]
                by_arr[ii, :] = B[:, 1]
                bz_arr[ii, :] = B[:, 2]
                
                if B_cent is not None:
                    bxc_arr[ii, :] = B_cent[:, 0]
                    byc_arr[ii, :] = B_cent[:, 1]
                    bzc_arr[ii, :] = B_cent[:, 2]
                
                ex_arr[ii, :] = E[:, 0]
                ey_arr[ii, :] = E[:, 1]
                ez_arr[ii, :] = E[:, 2]

                jx_arr[ii, :] = J[:, 0]
                jy_arr[ii, :] = J[:, 1]
                jz_arr[ii, :] = J[:, 2]
                
                vex_arr[ii, :] = Ve[:, 0]
                vey_arr[ii, :] = Ve[:, 1]
                vez_arr[ii, :] = Ve[:, 2]
                
                te_arr[  ii, :]      = Te
                qdns_arr[ii, :]      = q_dns
                field_sim_time[ii]   = sim_time
                damping_array[ii, :] = damp
            print('\nExtraction Complete.')
            
            np.save(self.temp_dir + 'bx' +'_array.npy', bx_arr)
            np.save(self.temp_dir + 'by' +'_array.npy', by_arr)
            np.save(self.temp_dir + 'bz' +'_array.npy', bz_arr)
            
            if B_cent is not None:
                np.save(self.temp_dir + 'bxc' +'_array.npy', bxc_arr)
                np.save(self.temp_dir + 'byc' +'_array.npy', byc_arr)
                np.save(self.temp_dir + 'bzc' +'_array.npy', bzc_arr)
            
            np.save(self.temp_dir + 'ex' +'_array.npy', ex_arr)
            np.save(self.temp_dir + 'ey' +'_array.npy', ey_arr)
            np.save(self.temp_dir + 'ez' +'_array.npy', ez_arr)
            
            np.save(self.temp_dir + 'jx' +'_array.npy', jx_arr)
            np.save(self.temp_dir + 'jy' +'_array.npy', jy_arr)
            np.save(self.temp_dir + 'jz' +'_array.npy', jz_arr)
            
            np.save(self.temp_dir + 'vex' +'_array.npy', vex_arr)
            np.save(self.temp_dir + 'vey' +'_array.npy', vey_arr)
            np.save(self.temp_dir + 'vez' +'_array.npy', vez_arr)
            
            np.save(self.temp_dir + 'te'    +'_array.npy', te_arr)
            np.save(self.temp_dir + 'qdens' +'_array.npy', qdns_arr)
            np.save(self.temp_dir + 'damping_array' +'_array.npy', damping_array)
            np.save(self.temp_dir + 'field_sim_time' +'_array.npy', field_sim_time)
            
            print('Field component arrays saved in {}'.format(self.temp_dir))
        return
    
    
    def load_fields(self, ii):
        data = np.load(self.field_dir+ f'data{ii:05d}.npz')

        B  = data['B']
        E  = data['E']
        Ve = data['Ve']
        Te = data['Te']
        q_dns    = data['dns']
        sim_time = data['sim_time']

        J             = data['Ji']            if 'Ji' in data.keys()            else data['J']
        damping_array = data['damping_array'] if 'damping_array' in data.keys() else None
        B_cent        = data['B_cent']        if 'B_cent' in data.keys()        else None
        return B, B_cent, E, Ve, Te, J, q_dns, sim_time, damping_array
    
    
    def load_particles(self, ii, load_equilibrium=False):  
        '''
        Sort kw for arranging particles by species index since they get jumbled.
        
        Still need to test the jumble fixer more in depth (i.e. multispecies etc.)
        
        Also need to group disabled particles with their active counterparts... but then
        there's a problem with having all 'spare' particles having index -128 for some
        runs - they're not 'deactivated cold particles', they were never active in the
        first place
        
        SOLUTION: For subsequent runs, use idx = -1 for 'spare' particles, since their
        index will be reset when they're turned on.
        
        ACTUALLY, the current iteration of the code doesn't use spare particles. idx_start/end
        should still be valid, maybe put a flag in for it under certain circumstances (For speed)
        
        Flag will be anything that involves spare particles. Load it later (nothing like
        that exists yet).
        '''
        pdir = self.equil_dir if load_equilibrium == True else self.particle_dir
        data = np.load(pdir + f'data{ii:05d}.npz', mmap_mode='c')
        
        pos       = data['pos']
        vel       = data['vel']
        sim_time  = data['sim_time']
        idx       = data['idx']

        # Open or reinitialized boundaries have to be sorted
        # TODO : Maybe overwrite these files later on? Or sort the particle files
        #      :   on save within the code itself (since particles aren't saved often)
        #      : OR maybe run the particle_unscrambler() the first time load_particles is called
        #      : that way, subsequent times don't have to
        if self.particle_open == True or self.particle_reinit == True:
            order = np.argsort(idx)
            idx   = idx[order]
            pos   = pos[order]
            vel   = vel[:, order]
        
            # start, end indices are variable for open boundaries
            # Could skip this part for reinit
            idx_start = np.zeros(self.Nj, dtype=int)
            idx_end   = np.zeros(self.Nj, dtype=int)
            
            # Get first index of each species. If None found,  
            acc = 0
            for jj in range(self.Nj):
                found_st = 0; found_en = 0
                for ii in range(acc, idx.shape[0]):
                    if idx[ii] >= 0:
                        
                        # Find start point (Store, and keep looping through particles)
                        if idx[ii] == jj and found_st == 0:
                            idx_start[jj] = ii
                            found_st = 1
                            
                        # Find end point (Find first value that doesn't match, if start is found
                        elif idx[ii] != jj and found_st == 1:
                            idx_end[jj] = ii; found_en = 1; acc = ii
                            break
                        
                # This should only happen with last species in array
                if found_st == 1 and found_en == 0:
                    idx_end[jj] = idx.shape[0]
        else:
            idx_start = self.idx_start0
            idx_end   = self.idx_end0
        return pos, vel, idx, sim_time, idx_start, idx_end


    def _set_mmap(self, component):
        '''
        Sets memory map pointer for field array specified by component
        Arrays stored in vectors (time, space)
        '''
        arr_path   = self.temp_dir + component.lower() + '_array.npy'
        if os.path.exists(arr_path) == True:
            arr = np.load(arr_path, mmap_mode='c')
        else:
            print('File {} does not exist.'.format(component.lower() + '_array.npy'))
            arr = None
        return arr
    
    
    def init_set_mmaps(self):
        '''
        Sets attributes for field memmaps, called in __init__()
        '''
        for arr_name in ['bx', 'by', 'bz', 'bxc', 'byc', 'bzc', 'ex', 'ey', 'ez', 'jx', 'jy', 'jz', 'vex', 'vey', 'vez', 'te', 'qdens',
                    'damping_array', 'field_sim_time']:
            arr_mmap = self._set_mmap(arr_name)
            setattr(self, arr_name, arr_mmap)
        self._get_particle_times()
            
        #ftime_sec = self.dt_field * np.arange(bx.shape[0])  
        #self.ftime_gyperiod = self.field_sim_time / self.gyperiod
        #self.ftime_gyfreq = self.field_sim_time * self.gyfreq 
        return
    
    
    def _interpolateFields2ParticleTimes(self):
        '''
        For each particle timestep, interpolate field values. Arrays are (time, space)
        '''        
        pbx, pby, pbz = [np.zeros((self.num_particle_steps, self.NC + 1)) for _ in range(3)]
        
        for ii in range(self.NC + 1):
            pbx[:, ii] = np.interp(self.particle_sim_time, self.field_sim_time, self.bx[:, ii])
            pby[:, ii] = np.interp(self.particle_sim_time, self.field_sim_time, self.by[:, ii])
            pbz[:, ii] = np.interp(self.particle_sim_time, self.field_sim_time, self.bz[:, ii])
        
        pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens = [np.zeros((self.num_particle_steps, self.NC)) for _ in range(11)]
        
        for ii in range(self.NC):
            pex[:, ii]    = np.interp(self.particle_sim_time, self.field_sim_time, self.ex[:, ii])
            pey[:, ii]    = np.interp(self.particle_sim_time, self.field_sim_time, self.ey[:, ii])
            pez[:, ii]    = np.interp(self.particle_sim_time, self.field_sim_time, self.ez[:, ii])
            pvex[:, ii]   = np.interp(self.particle_sim_time, self.field_sim_time, self.vex[:, ii])
            pvey[:, ii]   = np.interp(self.particle_sim_time, self.field_sim_time, self.vey[:, ii])
            pvez[:, ii]   = np.interp(self.particle_sim_time, self.field_sim_time, self.vez[:, ii])
            pte[:, ii]    = np.interp(self.particle_sim_time, self.field_sim_time, self.te[:, ii])
            pjx[:, ii]    = np.interp(self.particle_sim_time, self.field_sim_time, self.jx[:, ii])
            pjy[:, ii]    = np.interp(self.particle_sim_time, self.field_sim_time, self.jy[:, ii])
            pjz[:, ii]    = np.interp(self.particle_sim_time, self.field_sim_time, self.jz[:, ii])
            pqdens[:, ii] = np.interp(self.particle_sim_time, self.field_sim_time, self.qdens[:, ii])

        return pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens
    
    
    def _get_particle_times(self):
        '''
        Extract particle times in seconds from the particle files, if they exist. Save as a separate array for easy access.
        '''
        ptime_file = self.temp_dir + 'particle_sim_time' +'_array.npy'
        if not os.path.exists(ptime_file): 
            print('Extracting particle times...')
            particle_times  = np.zeros((self.num_particle_steps))
            for ii in range(self.num_particle_steps):
                print('Loading particle file {}'.format(ii))
                data = np.load(self.particle_dir + f'data{ii:05d}.npz', mmap_mode='r')
                particle_times[ii] = data['sim_time']
            np.save(ptime_file, particle_times)
        else:
            print('Loading particle times from file...')
            particle_times = np.load(ptime_file)
        self.particle_sim_time = particle_times
        return
    
    
    def get_energies(self): 
        '''
        Computes and saves field and particle energies at each field/particle timestep.
        
        TO DO: Use particle-interpolated fields rather than raw field files.
        OR    Use these interpolated fields ONLY for the total energy, would be slower.
        
        Also TODO:
            -- Calculate energies and save as file. Then delete array and use mmap instead? 
                Is this actually more memory efficient or am I just wasting time?
                
        Note: Electric field energy not calculated because reasons. Something to do about it
            already being included in the electron temperature?
        '''
        energy_file = self.temp_dir + 'energies.npz'
        if os.path.exists(energy_file) == False: 
            print('Calculating energies...')                       
            xst = self.ND; xen = self.ND + self.NX
            
            mag_energy      = (0.5 / MAGN_PERMEABILITY) * (self.bx[:, xst:xen]**2 + self.by[:, xst:xen]**2 + self.bz[:, xst:xen]**2 ).sum(axis=1) * self.dx
            electron_energy = 1.5 * (BOLTZMANN_CONSTANT * self.te[:, xst:xen] * self.qdens[:, xst:xen] / UNIT_CHARGE).sum(axis=1) * self.dx
            
            particle_energy = np.zeros((self.num_particle_steps, self.Nj, 2))
            for ii in range(self.num_particle_steps):
                print('Loading particle file {}'.format(ii))
                pos, vel, idx, ptimes, idx_start, idx_end = self.load_particles(ii)
                for jj in range(self.Nj):
                    '''
                    Only works properly for theta = 0 : Fix later
                    '''
                    vpp2 = vel[0, idx_start[jj]:idx_end[jj]] ** 2
                    vpx2 = vel[1, idx_start[jj]:idx_end[jj]] ** 2 + vel[2, idx_start[jj]:idx_end[jj]] ** 2
            
                    particle_energy[ii, jj, 0] = 0.5 * self.mass[jj] * vpp2.sum() * self.n_contr[jj] * self.NX * self.dx
                    particle_energy[ii, jj, 1] = 0.5 * self.mass[jj] * vpx2.sum() * self.n_contr[jj] * self.NX * self.dx
            
            # Calculate total energy
            pbx, pby, pbz, pex, pey, pez, pvex, pvey, pvez, pte, pjx, pjy, pjz, pqdens = \
            self._interpolateFields2ParticleTimes()
            
            pmag_energy      = (0.5 / MAGN_PERMEABILITY) * (np.square(pbx) + np.square(pby) + np.square(pbz)).sum(axis=1) * self.NX * self.dx
            pelectron_energy = 1.5 * (BOLTZMANN_CONSTANT * pte * pqdens / UNIT_CHARGE).sum(axis=1) * self.NX * self.dx

            total_energy = np.zeros(self.num_particle_steps)   # Placeholder until I interpolate this
            for ii in range(self.num_particle_steps):
                total_energy[ii] = pmag_energy[ii] + pelectron_energy[ii]
                for jj in range(self.Nj):
                    total_energy[ii] += particle_energy[ii, jj, :].sum()
            
            print('Saving energies to file...')
            np.savez(energy_file, mag_energy      = mag_energy,
                                  electron_energy = electron_energy,
                                  particle_energy = particle_energy,
                                  total_energy    = total_energy)
        else:
            print('Loading energies from file...')
            energies        = np.load(energy_file) 
            particle_times  = energies['particle_times']
            mag_energy      = energies['mag_energy']
            electron_energy = energies['electron_energy']
            particle_energy = energies['particle_energy']
            total_energy    = energies['total_energy']
        return mag_energy, electron_energy, particle_energy, total_energy