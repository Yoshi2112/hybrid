; This program calculates the growth rate for the cyclotron instability for compositions of
; singly ionized H, He, and O, including both warm and cold species. It also allows variable
; anisotropy ratios, perpendicular temperatures (ev), and magnetic flux densities.
; NOTE: All values of the growth rate are multiplied by 1.0E9,  unless the amplitudes
; are normalized, and then the maximum value is 1.0.
; Original Pascal code by John C. Samson
;
; Converted to IDL : Colin L. Waters
; July, 2009
;
Pro growth2

 Pth='d:\idl\bfraser\'
; Print,'Enter Name of Output File : '
 out_fname=Pth+'pc1_tst.dat'

; Print,'Normalize the amplitudes [1=Yes, 0=No] : '
; Read,normalize
 normalize=0
; Print,' Frequency/H-cyclotron Freq.? [1=Yes, 0=No] : '
; Read,normal
 normal=0
; Print,'Enter the maximum normalized frequency for the calculation : '
; Read,maxfreq
 maxfreq=0.7
; Print,' Input magnetic flux density (nT) : '
; read,field
 field=300.0

 NPTS = 200
 N = 3
; Index 1 denotes hydrogen ; 2 denotes helium; 3 denotes oxygen etc.
 M=dblarr(N)
 M(0)=1.0     ; Hydrogen
 M(1)=4.0     ; Helium
 M(2)=16.0    ; Oxygen

; Print,' Input densities of cold species (number/cc) [3]'
; read, ndensc(i)
 ndensc=dblarr(N)
 ndensc(0)=196.
 ndensc(1)=22.
 ndensc(2)=2.

; Density of warm species (same order as cold) (number/cc)
 ndensw=dblarr(N)
 ndensw(0)=5.1
 ndensw(1)=0.05
 ndensw(2)=0.13

; Input the perpendicular temperature (ev)
 temperp=dblarr(N)
 temperp(0)=30000.
 temperp(1)=10000.
 temperp(2)=10000.

; Input the temperature anisotropy
 A=dblarr(N)
 A(0)=1.
 A(1)=1.
 A(2)=1.

;
 PMASS = 1.67E-27
 MUNOT = 1.25660E-6
 EVJOULE = 6.242E18
 CHARGE = 1.60E-19

 smallarray= dblArr(N)
 etac=dblArr(N)
 etaw=dblArr(N)
 ratioc=dblArr(N)
 ratiow=dblArr(N)
 alpha=dblArr(N)
 bet=dblarr(N)
 tempar=dblarr(N)
 numer=dblarr(N)
 denom=dblarr(N)

 growth=dblarr(NPTS)
 x=dblarr(NPTS)
 Openw,u1,out_fname,/get_lun
;
 if (maxfreq gt 1.0) then maxfreq=1.0
 step=maxfreq/float(NPTS)
 field=field*1.0E-9                 ; convert to nT
 if (normal eq 1) then cyclotron=1.0 else cyclotron=CHARGE*field/(2.0*!dpi*PMASS)
;
 for i=0,N-1 do begin           ; Loop over 3 species of ions
  ndensc(i)=ndensc(i)*1.0E6
  ndensw(i)=ndensw(i)*1.0E6
  etac(i)=ndensc(i)/ndensw(0)
  etaw(i)=ndensw(i)/ndensw(0)
  temperp(i)=temperp(i)/EVJOULE
  tempar(i)=temperp(i)/(1.0+A(i))
  alpha(i)=sqrt(2.0*tempar(i)/PMASS)
  bet(i)=ndensw(i)*tempar(i)/(field*field/(2.0*MUNOT))
  numer(i)=M(i)*(etac(i)+etaw(i))
 end                      ; {i-loop}
 themax=0.0

 for k=0,NPTS-1 do begin
  x(k)= (k+1)*step
  for i=0,N-1 do denom(i)=1.0-M(i)*x(k)
  sum1=0.0
  prod2=1.0
  for i=0,N-1 do begin
   prod2=prod2*denom(i)
   prod=1.0
   temp=denom(i)
   denom(i)=numer(i)
   for j=0,N-1 do prod=prod*denom(j)
   sum1= sum1+prod
   denom(i)=temp
  end                  ; {i-loop}
  sum2=0.0
  arg4=prod2/(sum1)
;
; Check for stop band.
;
  if (arg4 lt 0.0) AND (x(k) gt 1.0/M(N-1)) then growth(k)=0.0 else begin
   arg3=arg4/(x(k)*x(k))
   for i=0,N-1 do begin
    if (ndensw(i) gt 1.0E-3) then begin
     arg1=(sqrt(!dpi)*etaw(i)/((M(i))^2*alpha(i)))
     arg1=arg1*((A(i)+1.0)*(1.0-M(i)*x(k))-1.0)
     arg2=(-etaw(i)/M(i))*(M(i)*x(k)-1.0)^2/bet(i)*arg3
     if (arg2 gt 200.0) then arg2=200.0
     if (arg2 lt -200.0) then arg2=-200.0
     sum2=sum2+arg1*exp(arg2)
    end
   end            ; {i-loop}
   growth(k)=sum2*arg3/2.0
  end             ; {else}
  if (growth(k) lt 0.0) then growth(k)=0.0
  x(k)=x(k)*cyclotron
  if (growth(k) gt themax) then themax=growth(k)
 end               ; {k-loop}

 if normalize eq 1 then begin
  for k=0, NPTS-1 do growth(k)=growth(k)/themax
 end else for k=0, NPTS-1 do growth(k)=growth(k)*1.0E9
;
 for k=0,NPTS-1 do printf,u1,x(k),'  ',growth(k)
 free_lun,u1
 window,0,xsize=600,ysize=500,title='Growth'
 plot,x,growth,xtitle='Frequency [Hz]', ytitle='Growth Rate'
 Print,'Finished'
end