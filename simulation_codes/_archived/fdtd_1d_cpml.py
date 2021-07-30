# -*- coding: utf-8 -*-
#
#     1-D FDTD cut down from 2-D version by Jamesina Simpson
#
# C Waters, Apr 2020
# Eqns from Taflove and Hagness; Computational EM
# hard source
# for epsr >= 1 and sig >= 0.0
#
# added Fourier Transform calc in time loop
#
# Converted to Python
# J Williams, Sep 2020
#
import numpy as np

nmax = 450                 # total number of time steps
XSz  = 1000
YSz  = 300

# Constants
c    = 2.99792458e8        # Speed of light in free space
muz  = 4.0e-7*np.pi        # Permeability of free space
epsz = 1.0/(c*c*muz)       # Permittivity of free space
etaz = np.sqrt(muz/epsz)   # Impedance of free space

# Material properties - set independent of z
sig  = 0.0                 # Conductivity?
epsr = 2.0                 # relative permittivity
mur  = 1.0                 # relative permeability

freq  = 5.0e+9             # center frequency of source excitation
lamb  = (c/epsr)/freq
omega = 2.0*np.pi*freq

# Grid parameters
nz      = 201                       # number of grid cells in z-direction in non PML section - must be odd
nzPML_1 = int(nz/10)                # number of PML layers in LH
nzPML_2 = nzPML_1                   # RH PML
kmax    = (nzPML_1 + nz + nzPML_2)  # tot points for PML + bulk grid, must be odd

# Location of source
src_loc = kmax//2

dz      = lamb/15.0       # grid -> number of points per wavelength
dt      = 0.8*dz/c

print('freq=',freq/1e6,' MHz')
print('wavelength=',lamb,' m')
print('dz=',dz,' m')
print('dt=',dt,' s')

#  Wave excitation
rtau     = 160.0e-12    # increase for freq inc
tau      = rtau/dt
delay    = 3.0*tau
src_mult = 1.5

# PML vars
kappa_z_max = 10.0
ma          = 1
m           = 3
sig_z_max   = 0.75*(0.8*(m+1)/(etaz*dz*(mur*epsr)**0.5))    # eqn 7.67 see Fig 7.4 for 0.75 (sig_max/sig_opt)
alpha_z_max = 0.1                                          # see Fig 7.4

# Allocate arrays
psi_Exz_1 = np.zeros(nzPML_1)
psi_Exz_2 = np.zeros(nzPML_2)

psi_Hyz_1 = np.zeros(nzPML_1-1)
psi_Hyz_2 = np.zeros(nzPML_2-1)

den_ez    = np.zeros(kmax-1)
den_hz    = np.zeros(kmax-1)

sige_z_PML_1   = np.zeros(nzPML_1)
alphae_z_PML_1 = np.zeros(nzPML_1)
kappae_z_PML_1 = np.zeros(nzPML_1)
be_z_1         = np.zeros(nzPML_1)
ce_z_1         = np.zeros(nzPML_1)

# IDL    loops: Exclude last value
# Python loops: Include last value
# Both start at 0?
for kk in range(nzPML_1):
    # idx based scaling
    sige_z_PML_1[kk]   = sig_z_max*((nzPML_1-(kk+1))/(nzPML_1-1.0))**m                  # eqn 7.115a, 7.60a,b
    alphae_z_PML_1[kk] = alpha_z_max*(kk/(nzPML_1-1.0))**ma                             # eqn 7.115c, 7.79
    kappae_z_PML_1[kk] = 1.0 + (kappa_z_max-1.0)*((nzPML_1-(kk+1))/(nzPML_1 - 1.0))**m  # eqn 7.115b
    
    # conv in eqn 7.98
    be_z_1[kk]         = np.exp(-(sige_z_PML_1[kk]/kappae_z_PML_1[kk] + alphae_z_PML_1[kk])*dt/epsz) # eqn 7.102, 7.114a (mistake in book)
    denom              = kappae_z_PML_1[kk]*(sige_z_PML_1[kk] + kappae_z_PML_1[kk]*alphae_z_PML_1[kk])
    if denom == 0.0:
        ce_z_1[kk] = 0.0
    else:
        ce_z_1[kk] = sige_z_PML_1[kk]*(be_z_1[kk]-1.0)/denom                            # eqn 7.99, 7.114b

sigh_z_PML_1   = np.zeros(nzPML_1-1)
alphah_z_PML_1 = np.zeros(nzPML_1-1)
kappah_z_PML_1 = np.zeros(nzPML_1-1)
bh_z_1         = np.zeros(nzPML_1-1)
ch_z_1         = np.zeros(nzPML_1-1)

for kk in range(nzPML_1-1):
    sigh_z_PML_1[kk]   = sig_z_max*((nzPML_1-(kk+1)-0.5)/(nzPML_1-1.0))**m                      # eqn 7.119a
    alphah_z_PML_1[kk] = alpha_z_max*(((kk+1)-0.5)/(nzPML_1-1.0))**ma                             # eqn 7.119c
    kappah_z_PML_1[kk] = 1.0+(kappa_z_max-1.0)*((nzPML_1-(kk+1)-0.5)/(nzPML_1 - 1.0))**m        # eqn 7.119b
    bh_z_1[kk]         = np.exp(-(sigh_z_PML_1[kk]/kappah_z_PML_1[kk] + alphah_z_PML_1[kk])*dt/epsz)  # eqn 7.118a (mistake in book), from eqn 7.102
    denom              = kappah_z_PML_1[kk]*(sigh_z_PML_1[kk] + kappah_z_PML_1[kk]*alphah_z_PML_1[kk])
    ch_z_1[kk]         = sigh_z_PML_1[kk]*(bh_z_1[kk]-1.0)/denom                                # eqn 7.118b


sige_z_PML_2   = np.zeros(nzPML_2)
alphae_z_PML_2 = np.zeros(nzPML_2)
kappae_z_PML_2 = np.zeros(nzPML_2)
be_z_2         = np.zeros(nzPML_2)
ce_z_2         = np.zeros(nzPML_2)

for kk in range(nzPML_2):
    sige_z_PML_2[kk]   = sig_z_max*((nzPML_2-(kk+1))/(nzPML_2-1.0))**m
    alphae_z_PML_2[kk] = alpha_z_max*(kk/(nzPML_2-1.0))**ma
    kappae_z_PML_2[kk] = 1.0+(kappa_z_max-1.0)*((nzPML_2-(kk+1))/(nzPML_2-1.0))**m
    be_z_2[kk]         = np.exp(-(sige_z_PML_2[kk]/kappae_z_PML_2[kk] + alphae_z_PML_2[kk])*dt/epsz)

    denom = kappae_z_PML_2[kk]*(sige_z_PML_2[kk] + kappae_z_PML_2[kk]*alphae_z_PML_2[kk])
    if denom == 0.0:
        ce_z_2[kk] = 0.0
    else:
        ce_z_2[kk] = sige_z_PML_2[kk]*(be_z_2[kk]-1.0)/denom                       # eqn 7.99, 7.114b


sigh_z_PML_2   = np.zeros(nzPML_2-1)
alphah_z_PML_2 = np.zeros(nzPML_2-1)
kappah_z_PML_2 = np.zeros(nzPML_2-1)
bh_z_2         = np.zeros(nzPML_2-1)
ch_z_2         = np.zeros(nzPML_2-1)

for kk in range(nzPML_2-1):
    sigh_z_PML_2[kk]   = sig_z_max*((nzPML_2-(kk+1)-0.5)/(nzPML_2-1.0))**m
    alphah_z_PML_2[kk] = alpha_z_max*(((kk+1)-0.5)/(nzPML_2-1.0))**ma
    kappah_z_PML_2[kk] = 1.0+(kappa_z_max-1.0)*((nzPML_2-(kk+1)-0.5)/(nzPML_2-1.0))**m
    bh_z_2[kk]         = np.exp(-(sigh_z_PML_2[kk]/kappah_z_PML_2[kk] + alphah_z_PML_2[kk])*dt/epsz)
    denom              = kappah_z_PML_2[kk]*(sigh_z_PML_2[kk] + kappah_z_PML_2[kk]*alphah_z_PML_2[kk])
    ch_z_2[kk]         = sigh_z_PML_2[kk]*(bh_z_2[kk]-1.0)/denom

# update coeffs
da = 1.0
db = (dt/(muz*mur))
# if sig_star ne 0 then
#  da = (1.0-sig_star*dt/(2.0*muz*mur)) / ((1.0+sig_star*dt/(2.0*muz*mur)))   # eqn 7.109a
#  db = dt/(muz*mur) / ((1.0+sig_star*dt/(2.0*muz*mur)))                      # eqn 7.109b
#
# ca = dblarr(kmax-1)   ; comment out to make sig(z) const
# cb = dblarr(kmax-1)

eaf = dt*sig/(2.0*epsz*epsr)
ca  = (1.0-eaf)/(1.0+eaf)              # Eqn 7.107a, this is a constant if the permittivity is const in the soln space
cb  = (dt/(epsz*epsr))/(1.0+eaf)       # Eqn 7.107b

# denominator terms
for kk in range(nzPML_1-1):
    den_hz[kk] = 1.0/(kappah_z_PML_1[kk]*dz)

for kk in range(nzPML_1-1, kmax-nzPML_2):
    den_hz[kk] = 1.0/dz

k2 = nzPML_2-2
for kk in range(kmax-nzPML_2, kmax-1):
    den_hz[kk] = 1.0/(kappah_z_PML_2[k2]*dz)
    k2         = k2 - 1

for kk in range(nzPML_1-1):
    den_ez[kk] = 1.0/(kappae_z_PML_1[kk]*dz)

for kk in range(nzPML_1-1, kmax-nzPML_2):
    den_ez[kk] = 1.0/dz

k2 = nzPML_2-2
for kk in range(kmax-nzPML_2, kmax-1):
    den_ez[kk] = 1.0/(kappae_z_PML_2[k2]*dz)
    k2         = k2 - 1

# =============================================================================
# # Plot things
# Set_Plot,'win'
# Device,decomposed=0
# loadCT,0,/sil
# window,0,xsize=XSz,ysize=YSz,title='Ex'
# =============================================================================

# soln field arrays
ex = np.zeros(kmax)       # bulk + 2*PML
hy = np.zeros(kmax-1)

# FT arrays
maxF     = 1.0/(2.0*dt)
del_f    = 1.0/(nmax*dt)
nfreq    = int(maxF/del_f)
f_ax     = np.arange(nfreq)*del_f
K_a      = np.zeros(nfreq, dtype=np.complex128)
imag_u   = 0.0+1.0j
K_a      = np.exp(-imag_u*2.0*np.pi*f_ax*dt)
Ex_spec  = np.zeros(nfreq, dtype=np.complex128)
spec_pos = int(kmax/4)

# *** *** *** *** ***  BEGIN TIME-STEPPING LOOP *** *** *** *** *** ***
for n in range(nmax):

    #  Update EX
    for kk in range(kmax-1):
        ex[kk] = ca*ex[kk] + cb*(hy[kk-1]-hy[kk])*den_ez[kk]     # eqn 7.106 (psi=0 in the bulk)


    # PML z dir
    for kk in range(nzPML_1):
        psi_Exz_1[kk] = be_z_1[kk]*psi_Exz_1[kk] + ce_z_1[kk]*(hy[kk-1]-hy[kk])/dz  # eqn 7.100, 7.105b
        ex[kk]        = ex[kk] + cb*psi_Exz_1[kk]


    kk = nzPML_2-1
    for k in range(kmax - nzPML_2, kmax-1):
        psi_Exz_2[kk] = be_z_2[kk]*psi_Exz_2[kk] + ce_z_2[kk]*(hy[k-1]-hy[k])/dz
        ex[k]         = ex[k] + cb*psi_Exz_2[kk]
        kk = kk-1      # move thru (forward) RH PML


    # FT of Ex at some location
    for nf in range(nfreq):
        Ex_spec[nf] = Ex_spec[nf] + (K_a[nf]**n)*ex[spec_pos]


    # Update Hy
    for kk in range(kmax - 1):
        hy[kk] = da*hy[kk] + db*(ex[kk]-ex[kk + 1])*den_hz[kk]   # eqn 7.108 (with psi=0), note -ve sign


    # PML z dir
    for kk in range(nzPML_1 - 1):
        psi_Hyz_1[kk] = bh_z_1[kk]*psi_Hyz_1[kk] + ch_z_1[kk]*(ex[kk]-ex[kk + 1])/dz  # eqn 7.110a
        hy[kk]        = hy[kk] + db*psi_Hyz_1[kk]
    

    kk = nzPML_2 - 2
    for k in range(kmax - nzPML_2, kmax - 1):
        psi_Hyz_2[kk] = bh_z_2[kk]*psi_Hyz_2[kk] + ch_z_2[kk]*(ex[k]-ex[k + 1])/dz
        hy[k]         = hy[k] + db*psi_Hyz_2[kk]
        kk = kk - 1
    
    # add source
    gaus        = np.exp(-((float(n) - delay)**2/tau**2))
    ex[src_loc] = src_mult*np.sin(omega*(float(n)-delay)*dt)*gaus

# =============================================================================
#     # Plot things
#     plot,ex,yrange=[-1.2,1.2],psym=2,title='T='+strtrim(string(n),2)
#     oplot,[nzPML_1,nzPML_1],[1.,-1]
#     oplot,[kmax-nzPML_2,kmax-nzPML_2],[1.,-1]
# 
#     wait,0.01
# =============================================================================
# *** *** *** *** *** END TIME STEP LOOP *** *** *** *** ***

Ex_spec[:] = Ex_spec[:]*dt # Does the star in the () mean all values? was previously Ex_spec(*)

# =============================================================================
# window,1,title='FT'
# plot,f_ax,abs(ex_spec),xtitle='Hz'
# =============================================================================

print('Finished')