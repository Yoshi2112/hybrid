;     1-D FDTD cut down from 2-D version by Jamesina Simpson
;
; C Waters, Apr 2020
; Eqns from Taflove and Hagness; Computational EM
; hard source
; for epsr >= 1 and sig >= 0.0
;
; added Fourier Transform calc in time loop
;
Pro fdtd_1d_cpml

 nmax=450         ; total number of time steps
 XSz = 1000
 YSz = 300

; constants
 c = 2.99792458e8           ; speed of light in free space
 muz = 4.0*!dpi*1.0d-7      ; permeability of free space
 epsz = 1.0/(c*c*muz)       ; permittivity of free space
 etaz = sqrt(muz/epsz)      ; impedance of free space

; material properties - set independent of z
 sig = 0.d0

 epsr = 2.d0                ; relative permittivity
 mur = 1.d0                 ; relative permeability

 freq = 5.0e+9              ; center frequency of source excitation
 lambda = (c/epsr)/freq
 omega = 2.0*!dpi*freq

; Grid parameters
 nz = 201               ; number of grid cells in z-direction in non PML section - must be odd
 nzPML_1 = fix(nz/10)   ; number of PML layers in LH
 nzPML_2 = nzPML_1      ; RH PML
 kmax=(nzPML_1 + nz + nzPML_2)  ; tot points for PML + bulk grid, must be odd

; Location of source
 is = kmax/2

 dz = lambda/15.0       ; grid -> number of points per wavelength
 dt = 0.8*dz/c
 print,'freq=',freq/1e6,' MHz'
 print,'wavelength=',lambda,' m'
 print,'dz=',dz,' m'
 print,'dt=',dt,' s'

;  Wave excitation
 rtau=160.0e-12    ; increase for freq inc
 tau=rtau/dt
 delay=3.0*tau
 src_mult = 1.5

; PML vars
 kappa_z_max = 10.0
 ma = 1
 m = 3
 sig_z_max = 0.75*(0.8*(m+1)/(etaz*dz*(mur*epsr)^0.5))    ; eqn 7.67 see Fig 7.4 for 0.75 (sig_max/sig_opt)
 alpha_z_max = 0.1       ; see Fig 7.4

 psi_Exz_1=dblarr(nzPML_1)
 psi_Exz_2=dblarr(nzPML_2)

 psi_Hyz_1=dblarr(nzPML_1-1)
 psi_Hyz_2=dblarr(nzPML_2-1)

 den_ez=dblarr(kmax-1)
 den_hz=dblarr(kmax-1)

 sige_z_PML_1=dblarr(nzPML_1)
 alphae_z_PML_1=dblarr(nzPML_1)
 kappae_z_PML_1=dblarr(nzPML_1)
 be_z_1=dblarr(nzPML_1)
 ce_z_1=dblarr(nzPML_1)

 for k = 0,nzPML_1-1 do begin
; idx based scaling
   sige_z_PML_1(k) = sig_z_max*((nzPML_1-(k+1))/(nzPML_1-1.0))^m                   ; eqn 7.115a, 7.60a,b
   alphae_z_PML_1(k) = alpha_z_max*((k)/(nzPML_1-1.0))^ma                          ; eqn 7.115c, 7.79
   kappae_z_PML_1(k) = 1.0 + (kappa_z_max-1.0)*((nzPML_1-(k+1))/(nzPML_1-1.0))^m   ; eqn 7.115b
; conv in eqn 7.98
   be_z_1(k)=exp(-(sige_z_PML_1(k)/kappae_z_PML_1(k) + alphae_z_PML_1(k))*dt/epsz) ; eqn 7.102, 7.114a (mistake in book)
   denom = kappae_z_PML_1(k)*(sige_z_PML_1(k) + kappae_z_PML_1(k)*alphae_z_PML_1(k))
   if denom eq 0.0 then begin
     ce_z_1(k) = 0.0
   end else ce_z_1(k) = sige_z_PML_1(k)*(be_z_1(k)-1.0)/denom                       ; eqn 7.99, 7.114b
 end

 sigh_z_PML_1=dblarr(nzPML_1-1)
 alphah_z_PML_1=dblarr(nzPML_1-1)
 kappah_z_PML_1=dblarr(nzPML_1-1)
 bh_z_1=dblarr(nzPML_1-1)
 ch_z_1=dblarr(nzPML_1-1)

 for k = 0,nzPML_1-2 do begin
   sigh_z_PML_1(k) = sig_z_max*((nzPML_1-(k+1)-0.5)/(nzPML_1-1.0))^m                  ; eqn 7.119a
   alphah_z_PML_1(k) = alpha_z_max*(((k+1)-0.5)/(nzPML_1-1.0))^ma                     ; eqn 7.119c
   kappah_z_PML_1(k) = 1.0+(kappa_z_max-1.0)*((nzPML_1-(k+1)-0.5)/(nzPML_1 - 1.0))^m  ; eqn 7.119b
   bh_z_1(k) = exp(-(sigh_z_PML_1(k)/kappah_z_PML_1(k) + alphah_z_PML_1(k))*dt/epsz)  ; eqn 7.118a (mistake in book), from eqn 7.102
   denom = kappah_z_PML_1(k)*(sigh_z_PML_1(k) + kappah_z_PML_1(k)*alphah_z_PML_1(k))
   ch_z_1(k) = sigh_z_PML_1(k)*(bh_z_1(k)-1.0)/denom                                  ; eqn 7.118b
 end

 sige_z_PML_2=dblarr(nzPML_2)
 alphae_z_PML_2=dblarr(nzPML_2)
 kappae_z_PML_2=dblarr(nzPML_2)
 be_z_2=dblarr(nzPML_2)
 ce_z_2=dblarr(nzPML_2)

 for k = 0,nzPML_2-1 do begin
   sige_z_PML_2(k) = sig_z_max*((nzPML_2-(k+1))/(nzPML_2-1.0))^m
   alphae_z_PML_2(k) = alpha_z_max*((k)/(nzPML_2-1.0))^ma
   kappae_z_PML_2(k) = 1.0+(kappa_z_max-1.0)*((nzPML_2-(k+1))/(nzPML_2-1.0))^m
   be_z_2(k) = exp(-(sige_z_PML_2(k)/kappae_z_PML_2(k) + alphae_z_PML_2(k))*dt/epsz)

   denom = kappae_z_PML_2(k)*(sige_z_PML_2(k) + kappae_z_PML_2(k)*alphae_z_PML_2(k))
   if denom eq 0.0 then begin
     ce_z_2(k) = 0.0
   end else ce_z_2(k) = sige_z_PML_2(k)*(be_z_2(k)-1.0)/denom                       ; eqn 7.99, 7.114b
 end

 sigh_z_PML_2=dblarr(nzPML_2-1)
 alphah_z_PML_2=dblarr(nzPML_2-1)
 kappah_z_PML_2=dblarr(nzPML_2-1)
 bh_z_2=dblarr(nzPML_2-1)
 ch_z_2=dblarr(nzPML_2-1)

 for k = 0,nzPML_2-2 do begin
   sigh_z_PML_2(k) = sig_z_max*((nzPML_2-(k+1)-0.5)/(nzPML_2-1.0))^m
   alphah_z_PML_2(k) = alpha_z_max*(((k+1)-0.5)/(nzPML_2-1.0))^ma
   kappah_z_PML_2(k) = 1.0+(kappa_z_max-1.0)*((nzPML_2-(k+1)-0.5)/(nzPML_2-1.0))^m
   bh_z_2(k) = exp(-(sigh_z_PML_2(k)/kappah_z_PML_2(k) + alphah_z_PML_2(k))*dt/epsz)
   denom = kappah_z_PML_2(k)*(sigh_z_PML_2(k) + kappah_z_PML_2(k)*alphah_z_PML_2(k))
   ch_z_2(k) = sigh_z_PML_2(k)*(bh_z_2(k)-1.0)/denom
 end

; update coeffs
 da = 1.0
 db = (dt/(muz*mur))
; if sig_star ne 0 then
;  da = (1.0-sig_star*dt/(2.0*muz*mur)) / ((1.0+sig_star*dt/(2.0*muz*mur)))   ; eqn 7.109a
;  db = dt/(muz*mur) / ((1.0+sig_star*dt/(2.0*muz*mur)))                      ; eqn 7.109b
;
; ca = dblarr(kmax-1)   ; comment out to make sig(z) const
; cb = dblarr(kmax-1)

 eaf = dt*sig/(2.d0*epsz*epsr)
 ca = (1.d0-eaf)/(1.0+eaf)              ; Eqn 7.107a, this is a constant if the permittivity is const in the soln space
 cb = (dt/(epsz*epsr))/(1.d0+eaf)       ; Eqn 7.107b

; denominator terms
 for k=0,nzPML_1-2 do begin
   den_hz(k) = 1.0/(kappah_z_PML_1(k)*dz)
 end
 for k=nzPML_1-1,kmax-nzPML_2-1 do begin
   den_hz(k) = 1.0/dz
 end
 kk = nzPML_2-2
 for k = kmax-nzPML_2,kmax-2 do begin
   den_hz(k) = 1.0/(kappah_z_PML_2(kk)*dz)
   kk = kk - 1
 end

 for k=0,nzPML_1-2 do begin
   den_ez(k) = 1.0/(kappae_z_PML_1(k)*dz)
 end
 for k=nzPML_1-1,kmax-nzPML_2-1 do begin
   den_ez(k) = 1.0/dz
 end
 kk = nzPML_2-2
 for k = Kmax-nzPML_2,kmax-2 do begin
   den_ez(k) = 1.0/(kappae_z_PML_2(kk)*dz)
   kk = kk - 1
 end

 Set_Plot,'win'
 Device,decomposed=0
 loadCT,0,/sil
 window,0,xsize=XSz,ysize=YSz,title='Ex'

; soln field arrays
 ex = dblarr(kmax)       ; bulk + 2*PML
 hy = dblarr(kmax-1)

; FT arrays
 maxF = 1.0/(2.0*dt)
 del_f = 1.0/(nmax*dt)
 nfreq = fix(maxF/del_f)
 f_ax = findgen(nfreq)*del_f
 K_a = complexarr(nfreq)
 imag_u=complex(0.0,1.0)
 K_a = exp(-imag_u*2.0*!pi*f_ax*dt)
 Ex_spec = complexarr(nfreq)
 spec_pos = fix(kmax/4)

; *** *** *** *** ***  BEGIN TIME-STEPPING LOOP *** *** *** *** *** ***
 For n=0,nmax-1 do begin

;  Update EX
   For kk=1,kmax-2 do begin
     ex[kk] = ca*ex[kk] + cb*(hy[kk-1]-hy[kk])*den_ez[kk]     ; eqn 7.106 (psi=0 in the bulk)
   end

; PML z dir
   For kk=1,nzPML_1-1 do begin
     psi_Exz_1[kk] = be_z_1[kk]*psi_Exz_1[kk] + ce_z_1[kk]*(hy[kk-1]-hy[kk])/dz  ; eqn 7.100, 7.105b
     ex[kk] = ex[kk] + cb*psi_Exz_1[kk]
   end

   kk=nzPML_2-1
   For k=kmax-nzPML_2,kmax-2 do begin
     psi_Exz_2[kk] = be_z_2[kk]*psi_Exz_2[kk] + ce_z_2[kk]*(hy[k-1]-hy[k])/dz
     ex[k] = ex[k] + cb*psi_Exz_2[kk]
     kk=kk-1      ; move thru (forward) RH PML
   end

; FT of Ex at some location
   for nf=0,nfreq-1 do begin
     Ex_spec(nf) = Ex_spec(nf) + (K_a(nf)^n)*Ex(spec_pos)
   end

;    Update Hy
   For kk=0,kmax-2 do begin
     hy[kk] = da*hy[kk] + db*(ex[kk]-ex[kk+1])*den_hz[kk]   ; eqn 7.108 (with psi=0), note -ve sign
   end

; PML z dir
   For kk=0,nzPML_1-2 do begin
     psi_Hyz_1[kk] = bh_z_1[kk]*psi_Hyz_1[kk] + ch_z_1[kk]*(ex[kk]-ex[kk+1])/dz  ; eqn 7.110a
     hy[kk] = hy[kk] + db*psi_Hyz_1[kk]
   end

   kk=nzPML_2-2
   For k=kmax-nzPML_2,kmax-2 do begin
     psi_Hyz_2[kk] = bh_z_2[kk]*psi_Hyz_2[kk] + ch_z_2[kk]*(ex[k]-ex[k+1])/dz
     hy[k] = hy[k] + db*psi_Hyz_2[kk]
     kk=kk-1
   end

;  add source
   gaus = exp(-((float(n)-delay)^2/tau^2))
   ex(is) = src_mult*sin(omega*(float(n)-delay)*dt)*gaus

   plot,ex,yrange=[-1.2,1.2],psym=2,title='T='+strtrim(string(n),2)
   oplot,[nzPML_1,nzPML_1],[1.,-1]
   oplot,[kmax-nzPML_2,kmax-nzPML_2],[1.,-1]

   wait,0.01

 end     ; *** *** *** *** *** END TIME STEP LOOP *** *** *** *** ***

 Ex_spec(*) = Ex_spec(*)*dt

 window,1,title='FT'
 plot,f_ax,abs(ex_spec),xtitle='Hz'

stop

Print,'Finished'
end
