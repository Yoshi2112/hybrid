# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:03:47 2019

@author: Yoshi
"""
import matplotlib.pyplot as plt
import numpy as np

import analysis_config  as cf
import analysis_backend as bk

import lmfit as lmf
from scipy.optimize         import curve_fit


def get_max_frequency(arr, plot=False):
    '''
    Calculates strongest frequency within a given field component across
    the simulation space. Returns frequency and power axes and index of 
    maximum frequency in axis.
    '''
    npts      = arr.shape[0]
    fft_freqs = np.fft.fftfreq(npts, d=cf.dt_field)
    fft_freqs = fft_freqs[fft_freqs >= 0]
    
    # For each gridpoint, take temporal FFT
    fft_matrix  = np.zeros((npts, cf.NX), dtype='complex128')
    for ii in range(cf.NX):
        fft_matrix[:, ii] = np.fft.fft(arr[:, ii] - arr[:, ii].mean())

    # Convert FFT output to power and normalize
    fft_pwr   = (fft_matrix[:fft_freqs.shape[0], :] * np.conj(fft_matrix[:fft_freqs.shape[0], :])).real
    fft_pwr  *= 4. / (npts ** 2)
    fft_pwr   = fft_pwr.sum(axis=1)

    max_idx = np.where(fft_pwr == fft_pwr.max())[0][0]
    print('Maximum frequency at {}Hz\n'.format(fft_freqs[max_idx]))
    
    if plot == True:
        plt.figure()
        plt.plot(fft_freqs, fft_pwr)
        plt.scatter(fft_freqs[max_idx], fft_pwr[max_idx], c='r')
        plt.title('Frequencies across simulation domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (nT^2 / Hz)')
        plt.legend()
        plt.show()
    return fft_freqs, fft_pwr, max_idx


def growing_sine(pars, t, data=None):
    vals   = pars.valuesdict()
    amp    = vals['amp']
    freq   = vals['freq']
    growth = vals['growth']

    model = amp * np.exp(1j*2*np.pi*freq*t).imag * np.exp(growth*t)
    
    if data is None:
        return model
    else:
        return model - data


def get_growth_rates(do_plot=None):
    '''
    Extract the magnetic linear wave growth rate from:
        -- Fitting an exponential to the magnetic energy
        -- Fitting a growing sine wave to the field components at each cell
    
    The linear regime is calculated as all times before the maximum energy derivative,
    i.e. the growth is assumed exponential until the rate of energy transfer decreases.
    
    One could also take the min/max (i.e. abs) of the field through time and 
    fit an exponential to that, but it should be roughly equivalent to the energy fit.
    
    INPUT:
        -- do_plot : 'show', 'save' or 'None'. 'save' will also output growth rates to a text file.
    '''
    by  = cf.get_array('By') * 1e9
    bz  = cf.get_array('Bz') * 1e9
    
    linear_cutoff, gr_rate_energy   = fit_magnetic_energy(by, bz, plot=do_plot)
    freqs, power, max_idx           = get_max_frequency(by,       plot=do_plot)
    
    growth_rate_kt(by, linear_cutoff, freqs[max_idx])
    
    by_wamps, by_wfreqs, by_gr_rate = fit_field_component(by, freqs[max_idx], 'By', linear_cutoff, plot=do_plot)
    bz_wamps, bz_wfreqs, bz_gr_rate = fit_field_component(bz, freqs[max_idx], 'Bz', linear_cutoff, plot=do_plot)
    
    if do_plot == 'save':
        txt_path  = cf.anal_dir + 'growth_rates.txt'
        text_file = open(txt_path, 'w')
    else:
        text_file = None
    
    print('Energy growth rate: {}'.format(gr_rate_energy), file=text_file)
    print('By av. growth rate: {}'.format(by_gr_rate.mean()), file=text_file)
    print('Bz av. growth rate: {}'.format(bz_gr_rate.mean()), file=text_file)
    print('By min growth rate: {}'.format(by_gr_rate.min()), file=text_file)
    print('Bz min growth rate: {}'.format(bz_gr_rate.min()), file=text_file)
    print('By max growth rate: {}'.format(by_gr_rate.max()), file=text_file)
    print('Bz max growth rate: {}'.format(bz_gr_rate.max()), file=text_file)
    return


def fit_field_component(arr, fi, component, cut_idx=None, plot=False, plot_cell=64):
    '''
    Calculates and returns parameters for growing sine wave function for each
    gridpoint up to the linear cutoff time.
    '''
    print('Fitting field component')
    time_fit  = cf.time_seconds_field[:cut_idx]
    gyfreq_hz = cf.gyfreq/(2*np.pi)    
    
    growth_rates = np.zeros(cf.NX)
    frequencies  = np.zeros(cf.NX)
    amplitudes   = np.zeros(cf.NX)
    
    fit_params = lmf.Parameters()
    fit_params.add('amp'   , value=1.0            , vary=True, min=-0.5*cf.B0*1e9 , max=0.5*cf.B0*1e9)
    fit_params.add('freq'  , value=fi             , vary=True, min=-gyfreq_hz     , max=gyfreq_hz    )
    fit_params.add('growth', value=0.001*gyfreq_hz, vary=True, min=0.0            , max=0.1*gyfreq_hz)
    
    for cell_num in range(cf.NX):
        data_to_fit  = arr[:cut_idx, cell_num]
        
        fit_output      = lmf.minimize(growing_sine, fit_params, args=(time_fit,), kws={'data': data_to_fit},
                                   method='leastsq')
        
        fit_function    = growing_sine(fit_output.params, time_fit)
    
        fit_dict        = fit_output.params.valuesdict()
        
        growth_rates[cell_num] = fit_dict['growth']
        frequencies[ cell_num] = fit_dict['freq']
        amplitudes[  cell_num] = fit_dict['amp']
    
        if plot != None and cell_num == plot_cell:
            plt.figure()
            plt.plot(time_fit, data_to_fit,  label='Magnetic field')
            plt.plot(time_fit, fit_function, label='Fit')
            plt.figtext(0.135, 0.73, r'$f = %.3fHz$' % (frequencies[cell_num] / (2 * np.pi)))
            plt.figtext(0.135, 0.69, r'$\gamma = %.3fs^{-1}$' % (growth_rates[cell_num] / (2 * np.pi)))
            plt.figtext(0.135, 0.65, r'$A_0 = %.3fnT$' % (amplitudes[cell_num] ))
            plt.title('{} cell {}'.format(component, plot_cell))
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (nT)')
            plt.legend()
            print(lmf.fit_report(fit_output))
            
            if plot == 'save':
                save_path = cf.anal_dir + '{}_envfit_{}.png'.format(component, plot_cell)
                plt.savefig(save_path)
                plt.close('all')
            elif plot == 'show':
                plt.show()
            else:
                pass
            
    return amplitudes, frequencies, growth_rates


def residual_exp(pars, t, data=None):
    vals   = pars.valuesdict()
    amp    = vals['amp']
    growth = vals['growth']

    model  = amp * np.exp(growth*t)
    
    if data is None:
        return model
    else:
        return model - data
    

def fit_magnetic_energy(by, bz, plot=False):
    '''
    Calculates an exponential growth rate based on transverse magnetic field
    energy.
    '''
    mu0 = (4e-7) * np.pi             # Magnetic Permeability of Free Space (SI units)
    
    print('Fitting magnetic energy')
    bt  = np.sqrt(by ** 2 + bz ** 2) * 1e-9
    U_B = 0.5 * np.square(bt).sum(axis=1) * cf.NX * cf.dx / mu0
    dU  = bk.get_derivative(U_B)
    
    linear_cutoff = np.where(dU == dU.max())[0][0]
    
    time_fit = cf.time_seconds_field[:linear_cutoff]

    fit_params = lmf.Parameters()
    fit_params.add('amp'   , value=1.0            , min=None , max=None)
    fit_params.add('growth', value=0.001*cf.gyfreq, min=0.0  , max=None)
    
    fit_output      = lmf.minimize(residual_exp, fit_params, args=(time_fit,), kws={'data': U_B[:linear_cutoff]},
                               method='leastsq')
    fit_function    = residual_exp(fit_output.params, time_fit)

    fit_dict        = fit_output.params.valuesdict()

    if plot != None:
        plt.ioff()
        plt.figure()
        plt.plot(cf.time_seconds_field[:linear_cutoff], U_B[:linear_cutoff], color='green', marker='o', label='Energy')
        plt.plot(cf.time_seconds_field[:linear_cutoff], fit_function, color='b', label='Exp. fit')
        plt.figtext(0.135, 0.725, r'$\gamma = %.3fs^{-1}$' % (fit_dict['growth'] / (2 * np.pi)))
        plt.title('Transverse magnetic field energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.legend()
        
# =============================================================================
#         plt.figure()
#         plt.plot(time_seconds[:linear_cutoff], dU[:linear_cutoff])
# =============================================================================
        
        if plot == 'save':
            save_path = cf.anal_dir + 'magnetic_energy_expfit.png'
            plt.savefig(save_path)
            plt.close('all')
        elif plot == 'show':
            plt.show()
        else:
            pass
        
    return linear_cutoff, fit_dict['growth']


def exponential_sine(t, amp, freq, growth, phase):
    return amp * np.sin(2*np.pi*freq*t + phase) * np.exp(growth*t)


def growth_rate_kt(arr, cut_idx, fi, saveas='kt_growth'):
    plt.ioff()

    time_fit  = cf.time_seconds_field[:cut_idx]
    k         = np.fft.fftfreq(cf.NX, cf.dx)
    k         = k[k>=0]

    # Take spatial FFT at each time
    mode_matrix  = np.zeros(arr.shape, dtype='complex128')
    for ii in range(arr.shape[0]):
        mode_matrix[ii, :] = np.fft.fft(arr[ii, :] - arr[ii, :].mean())

    # Cut off imaginary bits
    mode_matrix = 2*mode_matrix[:, :k.shape[0]]
    
    gmodel = lmf.Model(exponential_sine, nan_policy='propagate')
    
    gmodel.set_param_hint('amp',    value=1.0, min=0.0,     max=abs(mode_matrix).max())
    gmodel.set_param_hint('freq',   value=fi, min=-2*fi,    max=2*fi)
    gmodel.set_param_hint('growth', value=0.05, min=0.0,    max=0.5*fi)
    gmodel.set_param_hint('phase',  value=0.0, vary=False)
    
    for mode_num in [1]:#range(1, k.shape[0]):
        data_to_fit = mode_matrix[:cut_idx, mode_num].real
    
        result      = gmodel.fit(data_to_fit, t=time_fit, method='leastsq')

        plt.plot(time_fit, data_to_fit, 'ko', label='data')
        plt.plot(time_fit, result.best_fit, 'r-', label='lmfit')

        popt, pcov = curve_fit(exponential_sine, time_fit, data_to_fit, maxfev=1000000000)
        plt.plot(time_fit, exponential_sine(time_fit, *popt), label='curve_fit')
        plt.legend()
        print(popt)
# =============================================================================
#         fit_output      = minimize(exponential_sine, fit_params, args=(time_fit,), kws={'data': data_to_fit},
#                                    method='leastsq')
#         
#         fit_function    = exponential_sine(fit_output.params, time_fit)
# 
#         fit_dict        = fit_output.params.valuesdict()
#         
#         growth_rates[mode_num] = fit_dict['growth']
#         frequencies[ mode_num] = fit_dict['freq']
#         amplitudes[  mode_num] = fit_dict['amp']
#     
#         plt.plot(time_fit, data_to_fit)
#         plt.plot(time_fit, fit_function)
# =============================================================================

    plt.show()

    return

    
def get_linear_growth(plot=False):
    '''
    Calculates an exponential growth rate based on transverse magnetic field
    energy.
    '''
    import pdb
    by         = cf.get_array('By') * 1e9
    bz         = cf.get_array('Bz') * 1e9
    
    mu0 = (4e-7) * np.pi             # Magnetic Permeability of Free Space (SI units)
    
    print('Fitting magnetic energy')
    bt  = np.sqrt(by ** 2 + bz ** 2)
    U_B = np.square(bt[:, 0])#.sum(axis=1)# * cf.NX * cf.dx / mu0 * 0.5
    #dU  = bk.get_derivative(U_B)
    
    #linear_cutoff = np.where(dU == dU.max())[0][0]
    
    #time_fit = cf.time_seconds_field[:linear_cutoff]
    plt.plot(cf.time_radperiods_field, by, marker='o')
    plt.xlim(0, 200)
# =============================================================================
#     fit_params = lmf.Parameters()
#     fit_params.add('amp'   , value=1.0            , min=None , max=None)
#     fit_params.add('growth', value=0.001*cf.gyfreq, min=0.0  , max=None)
#     
#     fit_output      = lmf.minimize(residual_exp, fit_params, args=(time_fit,), kws={'data': U_B[:linear_cutoff]},
#                                method='leastsq')
#     fit_function    = residual_exp(fit_output.params, time_fit)
# 
#     fit_dict        = fit_output.params.valuesdict()
# =============================================================================

# =============================================================================
#     if plot == True:
#         plt.ioff()
#         plt.figure()
#         plt.plot(cf.time_seconds_field[:linear_cutoff], U_B[:linear_cutoff], color='green', marker='o', label='Energy')
#         plt.plot(cf.time_seconds_field[:linear_cutoff], fit_function, color='b', label='Exp. fit')
#         plt.figtext(0.135, 0.725, r'$\gamma = %.3fs^{-1}$' % (fit_dict['growth'] / (2 * np.pi)))
#         plt.title('Transverse magnetic field energy')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Energy (J)')
#         plt.legend()
#         
# # =============================================================================
# #         plt.figure()
# #         plt.plot(time_seconds[:linear_cutoff], dU[:linear_cutoff])
# # =============================================================================
#         
#         if plot == 'save':
#             save_path = cf.anal_dir + 'magnetic_energy_expfit.png'
#             plt.savefig(save_path)
#             plt.close('all')
#         elif plot == 'show':
#             plt.show()
#         else:
#             pass
# =============================================================================
    return


def straight_line_fit(save=True, normalized_output=False):
    '''
    To do: Check units. Get growth rate from amplitude only? How to do across space
    -- Is wave power averaged/summed across space analogous to a single point? Or do I have to do single point?
    -- How to calculate growth rate of energy summed across space?
    -- Start with the simple and go from there. Saturation amplitudes have to match too?
    -- How do the Winske saturation amplitudes look? It might just be the Fu thing being fucky.
    '''
    print('Calculating growth rate...')
    ftime, by  = cf.get_array('By')
    ftime, bz  = cf.get_array('Bz')
    bt         = np.sqrt(by ** 2 + bz ** 2)
    qp         = 1.602e-19
    mp         = 1.673e-27
    mu0        = (4e-7) * np.pi
    dB         = np.square(bt[:, :]).sum(axis=1)
    U_B        = dB * cf.NX * cf.dx / mu0 * 0.5
    pcyc       = qp * cf.B_eq / mp
    
    max_idx = np.argmax(U_B)
    st      = int(0.3 * max_idx)
    en      = int(0.7 * max_idx)
            
    # Rise/Run method (super basic, up to max_B)
    rise          = np.log(U_B[en]) - np.log(U_B[st]) 
    run           = ftime[en] - ftime[st]
    growth_rate   = rise/run
    normalized_gr = growth_rate / pcyc
    
    # Plot to check
    fig, ax = plt.subplots(figsize=(15, 10))
    
    ax.semilogy(ftime, U_B)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$U_B$')
    ax.set_xlim(0, ftime[-1])
    ax.set_ylim(None, None)
    
    # Mark growth rate indicators
    ax.scatter(ftime[max_idx], U_B[max_idx], c='r', s=20, marker='x')
    ax.scatter(ftime[st]     , U_B[st], c='r', s=20, marker='o')
    ax.scatter(ftime[en]     , U_B[en], c='r', s=20, marker='o')
    ax.semilogy([ftime[st], ftime[en]], [U_B[st], U_B[en]], c='r', ls='--', lw=2.0)
    
    if normalized_output == True:
        return normalized_gr
    else:
        return growth_rate
