import numpy as np
import matplotlib.pyplot as plt
import pdb

def smooth():
    return

#@nb.njit
def parmov(by, bz, ex, ey, ez):
    # Zero particle moments
    dns = np.zeros((nx + 2, nsp))               # Total macroparticle number density, per cell
    vxs = np.zeros((nx + 2, nsp))
    vys = np.zeros((nx + 2, nsp))
    vzs = np.zeros((nx + 2, nsp))
    
    acc1 = 0; acc2 = 0
    for kk in range(nsp):
        acc2  += nspec[kk]
        isp    = kk                       # Species identifier
        dta    = dt / wspec[isp]          # dt / m
        dta2   = .5 * dta                 # dt / 2
       
        for ii in range(acc1, acc2):
            i     = int(x[ii] / hx + 0.5)                                           # Leftmost node, I
            fx    = x[ii] / hx + 0.5 - i                                            # Weight at I + 1
            fxc   = 1. - fx                                                         # Weight at I

            exa   = ex[i] * fxc + ex[i + 1] * fx                                    # Ex at particle
            eya   = ey[i] * fxc + ey[i + 1] * fx + foy[i] * fxc + foy[i + 1] * fx   # Ey at particle
            eza   = ez[i] * fxc + ez[i + 1] * fx + foz[i] * fxc + foz[i + 1] * fx   # Ez at particle
            bya   = by[i] * fxc + by[i + 1] * fx                                    # By at particle
            bza   = bz[i] * fxc + bz[i + 1] * fx                                    # Bz at particle

            f     = 1. - dta * dta2 * (bxc**2 + bya**2 + bza**2)
            g     = dta2 * (vx[ii]*bxc + vy[ii]*bya + vz[ii]*bza)

            vxa   = vx[ii] + dta2 * exa
            vya   = vy[ii] + dta2 * eya
            vza   = vz[ii] + dta2 * eza

            vx[ii] = f * vx[ii] + dta * (exa + vya*bza - vza*bya + g*bxc)
            vy[ii] = f * vy[ii] + dta * (eya + vza*bxc - vxa*bza + g*bya)
            vz[ii] = f * vz[ii] + dta * (eza + vxa*bya - vya*bxc + g*bza)

            x[ii] += vx[ii] * dt

            # check to see particle is still in simulation region
            if x[ii] >= xmax:
                x[ii] -= xmax
            if x[ii] < 0:
                x[ii] += xmax

            # push extra half time step to collect density and current
            i     = int(x[ii] / hx + 0.5)
            fx    = x[ii] / hx + 0.5 - i
            fxc   = 1. - fx

            exa = ex[i]*fxc + ex[i + 1]*fx
            eya = ey[i]*fxc + ey[i + 1]*fx + foy[i]*fxc + foy[i + 1]*fx
            eza = ez[i]*fxc + ez[i + 1]*fx + foz[i]*fxc + foz[i + 1]*fx
            bya = by[i]*fxc + by[i + 1]*fx
            bza = bz[i]*fxc + bz[i + 1]*fx

            vxa = vx[ii] + dta2 * (exa + vy[ii]*bza - vz[ii]*bya)
            vya = vy[ii] + dta2 * (eya + vz[ii]*bxc - vx[ii]*bza)
            vza = vz[ii] + dta2 * (eza + vx[ii]*bya - vy[ii]*bxc)
            
            vxs[i, isp]    += vxa*fxc
            vxs[i + 1,isp] += vxa*fx
            vys[i, isp]    += vya*fxc
            vys[i + 1,isp] += vya*fx
            vzs[i, isp]    += vza*fxc
            vzs[i + 1,isp] += vza*fx
            dns[i, isp]    += fxc
            dns[i + 1,isp] += fx

            acc1 = acc2
    return dns, vxs, vys, vzs


#@nb.njit
def trans(by, bz, ex, ey, ez):
    # Zero source arrays
    den = np.zeros(nx + 2)                              # Normalized total ion densities for each cell
    vix = np.zeros(nx + 2)                              # x, y, z normalized moments for each cell
    viy = np.zeros(nx + 2)
    viz = np.zeros(nx + 2)

    dns, vxs, vys, vzs = parmov(by, bz, ex, ey, ez)     # Move particles and do particle diagnostics
    
    for k in range(nsp):                                # Put ghost cell contributions into real domain
        dns[1, k]   += dns[nx + 1, k]
        vxs[1, k]   += vxs[nx + 1, k]
        vys[1, k]   += vys[nx + 1, k]
        vzs[1, k]   += vzs[nx + 1, k]
        dns[nx, k]  += dns[0, k]
        vxs[nx, k]  += vxs[0, k]
        vys[nx, k]  += vys[0, k]
        vzs[nx, k]  += vzs[0, k]

    for k in range(nsp):                                # Normalize species densities and velocities
        for i in range(1, nx + 1):
            vxs[i, k] /= dns[i, k]
            vys[i, k] /= dns[i, k]
            vzs[i, k] /= dns[i, k]
            dns[i, k] *= dfac[k]

    # set ghost cell values
    for k in range(nsp):
        dns[0, k]      = dns[nx, k]
        vxs[0, k]      = vxs[nx, k]
        vys[0, k]      = vys[nx, k]
        vzs[0, k]      = vzs[nx, k]
        dns[nx + 1, k] = dns[1, k]
        vxs[nx + 1, k] = vxs[1, k]
        vys[nx + 1, k] = vys[1, k]
        vzs[nx + 1, k] = vzs[1, k]

    # calculate total ion density and velocities
    for i in range(nx + 2):
        for k in range(nsp):
            den[i] += dns[i, k] * frac[k]

            if den[i] < 0.05:
                den[i] = 0.05

    for k in range(nsp):
        for i in range(nx + 2):
            vix[i] += dns[i, k] * frac[k] * vxs[i,k] / den[i]
            viy[i] += dns[i, k] * frac[k] * vys[i,k] / den[i]
            viz[i] += dns[i, k] * frac[k] * vzs[i,k] / den[i]

#  can smooth source terms here if desired
#      call smooth(den,nx2)
#      call smooth(vix,nx2)
#      call smooth(viy,nx2)
#      call smooth(viz,nx2)
    return den, vix, viy, viz


def field(den, vix, viy, viz):
    hxi2 = .5 * hxi
    #dtx  = dt * hxi2
    hxs  = hx * hx
    #dxt  = hxs / dt
    ec11 = 1.
    ec12 = 0.
    ec21 = 0.
    ec22 = 1.
    fc1  = 0.
    fc2  = 0.
    gc11 = 0.
    gc12 = 0.
    gc21 = 0.
    gc22 = 0.

    # Initialize temp arrays
    e11  = np.zeros(nx + 2)
    e12  = np.zeros(nx + 2)
    e21  = np.zeros(nx + 2)
    e22  = np.zeros(nx + 2)
    f1   = np.zeros(nx + 2)
    f2   = np.zeros(nx + 2)
    g11  = np.zeros(nx + 2)
    g12  = np.zeros(nx + 2)
    g21  = np.zeros(nx + 2)
    g22  = np.zeros(nx + 2)
    h11  = np.zeros(nx + 2)
    h12  = np.zeros(nx + 2)
    h21  = np.zeros(nx + 2)
    h22  = np.zeros(nx + 2)

    # Set up A B C D tridiagonal arrays 
    for i in range(1, nx):
        df   = den[i] * resis / bxc                                 # Eq 5.43

        a12  = -den[i]*hx*vix[i] / (2. * bxc * (1. + df ** 2))      # Eq 5.50                            
        a11  = 1. - df*a12                                          # Eq 5.49
        a21  = -a12                                                 
        a22  =  a11
        
        b12  = (den[i] * hx ** 2) / (bxc * (1. + df ** 2) * dt)     # Eq 5.52
        b11  = -2.-df*b12                                           # Eq 5.51
        b21  = -b12
        b22  =  b11
        
        c12  = - a12                                                # Eq 5.54
        c11  = 1.-df*c12                                            # Eq 5.53
        c21  = -c12
        c22  =  c11
        
        viyp = viy[i] - vix[i]*byc / bxc                            # Eq 5.56
        vizp = viz[i] - vix[i]*bzc / bxc                            # Eq 5.57
                                                 
        df3  = hxs * den[i] / (1. + df * df)                        # Intermediaries
        df4  = 1. / (dt*bxc)
        
        d1   = - df3*(viyp + df*vizp - df4*(az[i] - df*ay[i]))      # Eq 5.55a
        d2   = - df3*(vizp - df*viyp + df4*(ay[i] + df*az[i]))      # Eq 5.55b

        if i == 1:
            #  Solve for E[1], F[1], G[1]
            ddi    = 1. / (b11*b22 - b12*b21)

            e11[1] = -ddi * ( b22 * c11 - b12 * c21)                # Eq 5.67
            e12[1] = -ddi * ( b22 * c12 - b12 * c22)
            e21[1] = -ddi * (-b21 * c11 + b11 * c21)
            e22[1] = -ddi * (-b21 * c12 + b11 * c22)

            f1[1]  =  ddi * ( b22 * d1  - b12 * d2 )                # Eq 5.68
            f2[1]  =  ddi * (-b21 * d1  + b11 * d2 )

            g11[1] = -ddi * ( b22 * a11 - b12 * a21)                # Eq 5.69
            g12[1] = -ddi * ( b22 * a12 - b12 * a22)
            g21[1] = -ddi * (-b21 * a11 + b11 * a21)
            g22[1] = -ddi * (-b21 * a12 + b11 * a22)
        else:
            # Solve for E(I), F(I), G(I) (60-62); make EC, FC, GC arrays (74-76)
            h11 = a11 * e11[i - 1] + a12 * e21[i - 1] + b11
            h12 = a11 * e12[i - 1] + a12 * e22[i - 1] + b12
            h21 = a21 * e11[i - 1] + a22 * e21[i - 1] + b21
            h22 = a21 * e12[i - 1] + a22 * e22[i - 1] + b22
            hdi = 1. / (h11 * h22 - h12 * h21)

            e11[i] = -hdi * ( h22 * c11 - h12 * c21)
            e12[i] = -hdi * ( h22 * c12 - h12 * c22)
            e21[i] = -hdi * (-h21 * c11 + h11 * c21)
            e22[i] = -hdi * (-h21 * c12 + h11 * c22)

            fd1   = d1 - a11 * f1[i - 1] - a12 * f2[i - 1]
            fd2   = d2 - a21 * f1[i - 1] - a22 * f2[i - 1]

            f1[i] = hdi * ( h22 * fd1 - h12 * fd2)
            f2[i] = hdi * (-h21 * fd1 + h11 * fd2)

            gd11 = a11 * g11[i - 1] + a12 * g21[i - 1]
            gd12 = a11 * g12[i - 1] + a12 * g22[i - 1]
            gd21 = a21 * g11[i - 1] + a22 * g21[i - 1]
            gd22 = a21 * g12[i - 1] + a22 * g22[i - 1]

            g11[i] = -hdi * ( h22 * gd11 - h12 * gd21)
            g12[i] = -hdi * ( h22 * gd12 - h12 * gd22)
            g21[i] = -hdi * (-h21 * gd11 + h11 * gd21)
            g22[i] = -hdi * (-h21 * gd12 + h11 * gd22)

            fc1 = fc1 + ec11 * f1[i - 1] + ec12 * f2[i - 1]
            fc2 = fc2 + ec21 * f1[i - 1] + ec22 * f2[i - 1]

            gc11 = gc11 + ec11 * g11[i - 1] + ec12 * g21[i - 1]
            gc12 = gc12 + ec11 * g12[i - 1] + ec12 * g22[i - 1]
            gc21 = gc21 + ec21 * g11[i - 1] + ec22 * g21[i - 1]
            gc22 = gc22 + ec21 * g12[i - 1] + ec22 * g22[i - 1]

            tc11 = ec11 * e11[i - 1] + ec12 * e21[i - 1]
            tc12 = ec11 * e12[i - 1] + ec12 * e22[i - 1]
            tc21 = ec21 * e11[i - 1] + ec22 * e21[i - 1]
            tc22 = ec21 * e12[i - 1] + ec22 * e22[i - 1]

            ec11 = tc11
            ec12 = tc12
            ec21 = tc21
            ec22 = tc22

    # Solve for X(NX) (77-78)
    en11 = e11[nx - 1] + g11[nx - 1]
    en12 = e12[nx - 1] + g12[nx - 1]
    en21 = e21[nx - 1] + g21[nx - 1]
    en22 = e22[nx - 1] + g22[nx - 1]

    ec11 = ec11 + gc11
    ec12 = ec12 + gc12
    ec21 = ec21 + gc21
    ec22 = ec22 + gc22

    h11  = a11 * en11 + a12 * en21 + b11 + c11 * ec11 + c12 * ec21
    h12  = a11 * en12 + a12 * en22 + b12 + c11 * ec12 + c12 * ec22
    h21  = a21 * en11 + a22 * en21 + b21 + c21 * ec11 + c22 * ec21
    h22  = a21 * en12 + a22 * en22 + b22 + c21 * ec12 + c22 * ec22

    hdi  = 1. / (h11 * h22 - h12 * h21)
    p1   = d1 - a11 * f1[nx - 1] - a12 * f2[nx - 1] - c11 * fc1 - c12 * fc2
    p2   = d2 - a21 * f1[nx - 1] - a22 * f2[nx - 1] - c21 * fc1 - c22 * fc2

    ey[nx] = ay[nx]
    ez[nx] = az[nx]
    ay[nx] = hdi * ( h22 * p1 - h12 * p2)
    az[nx] = hdi * (-h21 * p1 + h11 * p2)

    # get all X(I) (58)
    for ii in range(nx - 1, 0, -1): 
        ey[ii] = ay[ii]
        ez[ii] = az[ii]
        ay[ii] = e11[ii]*ay[ii + 1] + e12[ii]*az[ii + 1] + f1[ii] + g11[ii]*ay[nx] + g12[ii]*az[nx]
        az[ii] = e21[ii]*ay[ii + 1] + e22[ii]*az[ii + 1] + f2[ii] + g21[ii]*ay[nx] + g22[ii]*az[nx]

    # Fill ghost cells
    ey[nx + 1] = ay[nx + 1]
    ez[nx + 1] = az[nx + 1]
    ay[nx + 1] = ay[1]
    az[nx + 1] = az[1]
    ey[0]      = ay[0]
    ez[0]      = az[0]
    ay[0]      = ay[nx]
    az[0]      = az[nx]
    
    #  Get Ey Ez (35-36); By Bz (31-32)
    for i in range(1, nx + 1):
        ey[i] = (ey[i] - ay[i]) / dt
        ez[i] = (ez[i] - az[i]) / dt
        by[i] = (az[i - 1] - az[i + 1]) * hxi2 + byc
        bz[i] = (ay[i + 1] - ay[i - 1]) * hxi2 + bzc
        
        if field_zero == 1:
            ey[i] = 0.0
            ez[i] = 0.0
            by[i] = 0.0
            bz[i] = 0.0

    ey[0]      = ey[nx]
    ez[0]      = ez[nx]
    ey[nx + 1] = ey[1]
    ez[nx + 1] = ez[1]
    by[nx + 1] = by[1]
    bz[nx + 1] = bz[1]
    by[0]      = by[nx]
    bz[0]      = bz[nx]

    #################################
    ### ALL ELECTRON AND Ex STUFF ###
    #################################
    for i in range(1, nx + 1):                                     # Calculate del**2 A, electron velocities, drag force
        ajy    = -(ay[i + 1] + ay[i + 1] - 2. * ay[i]) / hxs
        ajz    = -(az[i + 1] + az[i + 1] - 2. * az[i]) / hxs
        vey[i] = -ajy / den[i] + viy[i]
        vez[i] = -ajz / den[i] + viz[i]
        foy[i] = -resis*ajy
        foz[i] = -resis*ajz

    for i in range(1, nx + 1):                                      # Calculate electron temperature and pressure
        if iemod == 0:
            te[i] = te0
        elif iemod == 1:
            te[i] = te0 * (den[i]**gam)
        pe[i]   = te[i]*den[i]

    vey[nx + 1] = vey[1]                                            # Fill ghost cells
    vez[nx + 1] = vez[1]
    te[ nx + 1] = te[1]
    pe[ nx + 1] = pe[1]
    
    vey[0]      = vey[nx]
    vez[0]      = vez[nx]
    te[0]       = te[ nx]
    pe[0]       = pe[ nx]

    # Calculate Ex
    for i in range(1, nx + 1):
        ex[i] = vez[i] * by[i] - vey[i] * bz[i]
        ex[i] = ex[i] - hxi2 * (pe[i + 1] - pe[i - 1]) / den[i]
        
        if field_zero == 1:
            ex[i] = 0.0

    ex[0]      = ex[nx]
    ex[nx + 1] = ex[1]
    return by, bz, ex, ey, ez


if __name__ == '__main__':
    save_path = 'E://runs//winske//'
    np.random.seed(21)
    
    plot         = False
    field_zero   = 0
    steady_state = 1
    
    c      = 3e10                       # Speed of light in cm/s
    ntimes = 1001                       # Number of timesteps
    dtwci  = 0.05                       # Timestep in inverse gyrofrequency
    nx     = 128                        # Number of cells
    xmax   = 128.                       # System length in c/wpi
    wpiwci = 10000.                     # Ratio of plasma and gyro frequencies
    nsp    = 2                          # Number of ion species

    nspec  = np.array([5120,5120])      # Number of macroparticles per species
    wspec  = np.array([1.,1.])          # Mass of each species (in multiples of proton mass)
    dnspec = np.array([0.10,0.900])     # Density of each species (total = n0)

    vbspec = np.array([0.90,-0.10])     # Bulk velocity for each species
    btspec = np.array([10.,1.])         # Species plasma beta (parallel?)
    anspec = np.array([5.,1.])          # T_perp/T_parallel (anisotropy) for each species

    if steady_state == 1:    
        vbspec = np.array([0.0, 0.0])   # Bulk velocity for each species
        btspec = np.array([1.0, 1.0])   # Species plasma beta (parallel?)
        anspec = np.array([1.0, 1.0])   # T_perp/T_parallel (anisotropy) for each species

    gam   = 5. / 3.                     # Polytropic factor for adiabatic electrons
    bete  = 1.                          # Electron beta
    resis = 0.                          # Resistivity (usually 0)
    theta = 0.                          # Angle between B0 and x axis
    iemod = 0                           # Electron model (0: Isothermal, 1: Adiabatic)

    npart  = nspec.sum()

    # Define some variables from inputs
    hx   = xmax / float(nx)             # Spatial step
    hxi  = 1. / hx                      # Inverse spatial step
    dt   = wpiwci*dtwci                 # Time step
    thet = theta*1.74533e-2             # Theta in radians
    cth  = np.cos(thet)                 # Cos theta (for magnetic field rotation)
    sth  = np.sin(thet)                 # Sin theta (for magnetic field rotation)
    bxc  = cth/wpiwci                   # Background magnetic field: x component
    byc  = 0.                           # Background magnetic field: y component
    bzc  = sth/wpiwci                   # Background magnetic field: z component
    vye  = 0.                           # Background electric field: y component
    vze  = 0.                           # Background electric field: z component
    te0  = bete/(2.*wpiwci**2)          # Initial electron temperature
    pe0  = te0                          # Initial electron pressure

    # Initialize particle state arrays
    tx0  = np.zeros(nsp)
    vth  = np.zeros(nsp)
    vbx  = np.zeros(nsp)
    vb0  = np.zeros(nsp)
    ans  = np.zeros(nsp)
    pinv = np.zeros(nsp)
    dfac = np.zeros(nsp)
    frac = np.zeros(nsp)
    npi  = np.zeros(nsp)

    # Initialize global arrays
    x  = np.zeros(npart)
    vx = np.zeros(npart)
    vy = np.zeros(npart)
    vz = np.zeros(npart)

    ay  = np.zeros(nx + 2)             # Vector potential: y component
    az  = np.zeros(nx + 2)             # Vector potential: z component
    vey = np.zeros(nx + 2)             # Electron velocity: y
    vez = np.zeros(nx + 2)             # Electron velocity: z

    te  = np.ones(nx + 2) * te0        # Electron temperature
    pe  = np.ones(nx + 2) * pe0        # Electron pressure
    
    foy = np.zeros(nx + 2)             # Force?
    foz = np.zeros(nx + 2)             # Force?
    by  = np.ones(nx + 2) * byc        # Magnetic field: y
    bz  = np.ones(nx + 2) * bzc        # Magnetic field: z
    ex  = np.zeros(nx + 2)             # Electric field: x
    ey  = np.zeros(nx + 2)             # Electric field: y
    ez  = np.zeros(nx + 2)             # Electric field: z

    den = np.zeros(nx + 2)             # Total density (moment)
    vix = np.zeros(nx + 2)             # Average velocities (moment)
    viy = np.zeros(nx + 2)
    viz = np.zeros(nx + 2)

    # Main program
    vmax = 0.0
    for k in range(nsp):
        tx0[k]  = btspec[k] / (2. * wpiwci **2 )    # Get temperature (x? Parallel?) from beta
        vth[k]  = np.sqrt(2. * tx0[k] / wspec[k])   # Thermal velocity
        vbx[k]  = vbspec[k] / wpiwci                # Bulk velocity
        vb0[k]  = np.max((vbx[k], vth[k]))          # Greater of the two
        ans[k]  = np.sqrt(anspec[k])                # Anisotropy velocity factor
        vfac    = np.max((1., ans[k]))              #
        vmax    = np.max((vmax, vfac*vb0[k]))
        pinv[k] = 1. / (nspec[k])                   # Number of species inverse (fraction?)
        dfac[k] = xmax * pinv[k] / hx               # Position increment for particles?
        frac[k] = dnspec[k] / dnspec.sum()          # Density fraction of each species
        npi[k]  = nspec[k]                          # Seems superfluous?

    vplim = 2. * vmax                               # Particle max speed limit? (for Courant condition?)

    acc1 = 0; acc2 = 0                              # Particle accumulators
    for kk in range(nsp):
        isp    = kk

        acc2  += nspec[kk]                          # Increment by number particles in species kk
        for ii in range(acc1, acc2):
            x[ii]     = xmax * pinv[isp] * (npi[isp] - .5)              # Place particles in configuration space

            vmag     = np.sqrt(-np.log(1.-.999999*np.random.rand()))    # Velocity magnitude (Maxwellian distribution)
            th       = 2.*np.pi*np.random.rand()                        # Random angle
            vxa      = vth[isp]*vmag*np.cos(th) + vbx[isp]              # Particle x-velocity plus bulk velocity

            vmag     = np.sqrt(-np.log(1.-.999999*np.random.rand()))    # Velocity magnitude (Maxwellian distribution)
            th       = 2.*np.pi*np.random.rand()                        # Random angle
            vy[ii]   = vth[isp]*ans[isp]*vmag*np.sin(th)                # Particle y-velocity
            vza      = vth[isp]*ans[isp]*vmag*np.cos(th)                # Particle z-velocity

            vx[ii]   = vxa*cth-vza*sth                                  # Rotate vx to simulation space coordinates
            vz[ii]   = vza*cth+vxa*sth                                  # Rotate vz to simulation space coordinates

            npi[isp] = npi[isp] - 1                                     # Count variable?

        acc1 = acc2

    # Run one time step with dt=0 to initialize fields
    dtsav = dt
    dt    = 0.
    den, vix, viy, viz  = trans(by, bz, ex, ey, ez)
    dt    = dtsav
    by, bz, ex, ey, ez  = field(den, vix, viy, viz)
    
    it = 0; t = 0    
    while it <= ntimes:
        plt.ioff()
        
        print('{}'.format(it))
        it += 1
        t  += dtwci

        den, vix, viy, viz = trans(by, bz, ex, ey, ez)
        by, bz, ex, ey, ez = field(den, vix, viy, viz)
                
        if plot == True:
            # E-Field
            fig1 = plt.figure()
            
            ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
            ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
            ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
            
            ax1.set_title(r'Electric Fields ($nV/m$) $\Omega t$ = {}'.format(t))
            ax1.plot(ex * 1e9)
            ax1.set_ylabel('$E_x$')
            ax1.set_ylim(-50, 50)
            
            ax2.plot(ey * 1e9)
            ax2.set_ylabel('$E_y$')
            ax2.set_ylim(-100, 100)
            
            ax3.plot(ez * 1e9)
            ax3.set_ylabel('$E_z$')
            ax3.set_ylim(-100, 100)
            
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(0, 130)
            
            filename = 'Efields_{}.png'.format(it)
            fullpath = save_path + '//E//'+ filename
            plt.savefig(fullpath, facecolor=fig1.get_facecolor(), edgecolor='none')
            plt.close('all')
            
            # B-Field
            fig2 = plt.figure()
            
            ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
            ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
            ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
            
            ax1.set_title(r'Magnetic Fields $\Omega t$ = {}'.format(t))
            ax1.plot(np.sqrt(bxc ** 2 + by ** 2 + bz ** 2) / bxc)
            ax1.set_ylabel('$|B| / B_0$')
            ax1.set_ylim(0, 2)
            
            ax2.plot(by / bxc)
            ax2.set_ylabel('$B_y / B_0$')
            ax2.set_ylim(-2, 2)
            
            ax3.plot(bz / bxc)
            ax3.set_ylabel('$B_z / B_0$')
            ax3.set_ylim(-2, 2)
            
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(0, 130)
            
            filename = 'Bfields_{}.png'.format(it)
            fullpath = save_path + '//B//' + filename
            plt.savefig(fullpath, facecolor=fig2.get_facecolor(), edgecolor='none')
            plt.close('all')


            # Velocity
            fig3 = plt.figure()
            
            ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)
            ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=4)
            ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=4)
            ax4 = plt.subplot2grid((4, 4), (3, 0), colspan=4)

            ax1.set_title(r'Moments (D = {}) $\Omega t$ = {}'.format(den[1:-1].sum(), t))
            ax1.plot(den)
            ax1.set_ylabel('$n_i$')
            ax1.set_ylim(0, 4)
            
            ax2.plot(vix)
            ax2.set_ylabel('$v_x / c$')
            ax2.set_ylim(-1e-3, 1e-3)
            
            ax3.plot(viy)
            ax3.set_ylabel('$v_y / c$')
            ax3.set_ylim(-1e-4, 1e-4)
            
            ax4.plot(viz)
            ax4.set_ylabel('$v_z / c$')
            ax4.set_ylim(-1e-4, 1e-4)
            
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xlim(0, 130)
                
            filename = 'moments_{}.png'.format(it)
            fullpath = save_path + '//M//' + filename
            plt.savefig(fullpath, facecolor=fig3.get_facecolor(), edgecolor='none')
            plt.close('all')
