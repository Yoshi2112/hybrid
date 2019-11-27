;

;I THINK I MADE A BIG MISTAKE
;IDL GOES FROM 0 IN AN ARRAY (AS THE FIRST ENTRY)
;BUT FORTRAN GOES FROM 1
;THIS MIGHT TAKE A BIT OF FIDDLING...
;could maybe just change array definitions to be one cell bigger, might work? including input variable entries


;
;
;  `hhhhhshhhh+hhhhhhhhhh+hhhhhhy+. yhhhhhhs/` -hhhhh:ohhhhhhs/`           `smmmhhh/ .ydmmmy- +hhhhhhs+` shhhhhhhh:
;  `MMMMMdMMMMyMMMMMMMMMMyMMMMMMMMd NMMMMMMMMo :MMMMM+yMMMMMMMMy           +MMMMMMMs mMMMMMMN`sMMMMMMMMh mMMMMMMMM+
;  `MMMMMdMMMMyMMMMMMMMMMyMMMMMMMMM.NMMMMMMMMd :MMMMM+yMMMMMMMMN`          hMMMMMMMs-MMMMMMMM/sMMMMMMMMM`mMMMMMMMM+
;   -MMMdoMMMo./NMMmoMMm:.oMMMsdMMM.-mMMN/NMMm `/MMMy`.yMMM+NMMM.          NMMM+NMMs/MMMmyMMMo.sMMMomMMM--dMMM-dMM+
;    MMMdoMMM/  /MMMNMM/  /MMMsmMMm  mMMm:mMMo  -MMMo  sMMM.hMMM.          NMMN shh//MMMs/MMMo +MMM-yMMM- hMMM-dmh:
;    MMMMNMMM/   +MMMMy   /MMMMMMN+  mMMMMMMy`  -MMMo  sMMM.hMMM.          NMMN ..../MMMo/MMMo +MMM-yMMM- hMMMNNo
;    MMMdoMMM/    dMMM`   /MMMoyMMN. mMMN+MMM:- -MMMo  sMMM.hMMM.          NMMN hNNd/MMMo/MMMo +MMM-yMMM- hMMM-yhs/
;   .MMMd+MMM+`  .hMMM.  `+MMM+yMMM/.mMMm.MMMhN::MMMs``sMMM:mMMM.          NMMM/NMMN/MMMdyMMMo`oMMM/dMMM-.dMMM.dMMs
;  `mMMMMdMMMN/  mMMMMy  +NMMMNMMMM:mMMMMhMMMMMsNMMMN/sNMMMNMMMM`          dMMMMMMMm:MMMMMMMM+oNMMMNMMMM.hMMMMmMMMs
;  `MMMMMdMMMM+  MMMMMd  oMMMMMMMMN.NMMMMdmMMMMoMMMMM+yMMMMMMMMh           sMMMMMMMs`NMMMMMMM.sMMMMMMMMd mMMMMMMMMs
;  `mmmmmhmmmm+  mmmmmy  +mmmmmmmd/ mmmmmh+mNNy:mmmmm/smmmmmmmy.           .hNNNNNy` /mNNNNmo ommmmmmmy- hmmmmmmmmo
;   ..........`  .....`  `......`   ...... .-. `.....``......`               .-:-.    `-::-`  `......`   .........`
;
;                                                                                                                                              ;
; Jackson Clarke,
; University of Newcastle
; Semester 2 (Aug-Nov), 2009
;
;	I have tried to keep in accordance with the code by Winske and Omidi in
;	Computer Space Plasma Physics: Simulation Techniques and Software edited by
;	Matsumoto and Omura (see http://www.terrapub.co.jp/e-library/cspp/index.html).
;	By this I mean I have kept variable names more or less the same and the
;	algorithm has not been changed in any major way.
;	I have changed the names of the fields and total velocities to capitals
;	(e.g. ex->Ex) to avoid confusion with matrices used for the explicit field solution.
;	I have also changed the way the individual particle position and speed is stored,
;	since the way in the Winske code was unnecessarily tricky (probably due to an
;	inheritance in converting from a more detailed code).
;
;


pro hybrid

	;>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	;
	; DEFINE INPUT VARIABLES
	;

	t		= 0				;i added this, NOT IN WINSKE

	;general
	ntimes	= 2001			;time steps
	dtwci	= 0.05 			;dt in wci units
	nx		= 128			;no of computational cells (not including two ghost cells)
	xmax	= 128.			;system length in c/wpi units
	wpiwci	= 10000. 		;ratio of wpi/wci (?? i think plasma freq to cyclotron freq?)
	resis	= 0. 			;resistivity eta term
	theta	= 0. 			;angle between B0 and x axis
	;ions
	nsp		= 2
	nspec	= [5120,5120] 	;no of simulation particles of each species
	vbspec	= [9.00,-1.00]	;velocity for each species in alfven speed units (va^2=Bo^2/(4*pi*no*mo) see pg 120)
	dnspec	= [0.10,0.900] ;total density of each species
	btspec	= [100.,1.]		;plasma beta for each species
	anspec	= [1.,21.] 		;anisotropy: T-perp/T-parallel for each species (?? see original code and pg 121?)
	wspec	= [1.,1.]		;mass of each species in proton mass units
	;electrons
	iemod	= 0				;electron model (0 for Te constant, 1 for adiabatic)
	bete	= 1. 			;electron beta
	;plotting
	nskip	= 2				;1/fraction of ions plotted
	npltp	= 200 			;time step increment for particle plots
	npltf	= 200 			;time step increment for field plots
	nplth	= 1000 			;time step increment for history plots
	nwrtf	= 20 			;time step increment for history writes
	nwrth	= 40			;time step increment for field writes

	;<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

	;this just fixes the numbering problem i had of 1-> instead of 0-> for input values
	nspec	= [0.,nspec(0),nspec(1)]
	vbspec	= [0.,vbspec(0),vbspec(1)]
	dnspec	= [0.,dnspec(0),dnspec(1)]
	btspec	= [0.,btspec(0),btspec(1)]
	anspec	= [0.,anspec(0),anspec(1)]
	wspec	= [0.,wspec(0),wspec(1)]

	;initialise the run
	init,		Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
				den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
				Vey,Vez,te,pe,										$;electrons
				firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
	     		dtwci,vbspec,dnspec,btspec,dfac,frac,				$
	     		bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
	     		t,nspec,te0,wspec,nskip,vplim,ntimes,				$
	     		npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
	     		vxhist,tphist,txhist,ldec,nplth,diagno,				$
	     		resis,theta,wpiwci,vb0,tx0,lastp

    set_plot,'win'
    device,decomposed=0
    loadct,23,/silent

	;MAIN LOOP
	for it=1,ntimes do begin
		it		= it+1
		t		= t+dtwci		;THERES SOMETHING STRANGE ABOUT THE DT AND DTWCI HERE??
		trans,		Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
					den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
					Vey,Vez,te,pe,										$;electrons
					firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
		     		dtwci,vbspec,dnspec,btspec,dfac,frac,				$
		     		bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
		     		t,nspec,te0,wspec,nskip,vplim,ntimes,				$
		     		npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
		     		vxhist,tphist,txhist,ldec,nplth,diagno,				$
		     		resis,theta,wpiwci,vb0,tx0,lastp
		field,		Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
					den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
					Vey,Vez,te,pe,										$;electrons
					firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
		     		dtwci,vbspec,dnspec,btspec,dfac,frac,				$
		     		bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
		     		t,nspec,te0,wspec,nskip,vplim,ntimes,				$
		     		npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
		     		vxhist,tphist,txhist,ldec,nplth,diagno,				$
		     		resis,theta,wpiwci,vb0,tx0,lastp
		diagno	= 2
		diagnos,	Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
					den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
					Vey,Vez,te,pe,										$;electrons
					firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
		     		dtwci,vbspec,dnspec,btspec,dfac,frac,				$
		     		bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
		     		t,nspec,te0,wspec,nskip,vplim,ntimes,				$
		     		npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
		     		vxhist,tphist,txhist,ldec,nplth,diagno,				$
		     		resis,theta,wpiwci,vb0,tx0,lastp

	endfor

	end



;>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
;
; SUBROUTINE: init
;
; Initialisation routine.
;

pro init, 		Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
				den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
				Vey,Vez,te,pe,										$;electrons
				firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
     			dtwci,vbspec,dnspec,btspec,dfac,frac,				$
     			bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
     			t,nspec,te0,wspec,nskip,vplim,ntimes,				$
     			npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
     			vxhist,tphist,txhist,ldec,nplth,diagno,				$
     			resis,theta,wpiwci,vb0,tx0,lastp

	;define these from the input variables
	nx1		= nx+1
	nx2		= nx+2
	nc 		= nx2			;no of cells
	ns		= nsp			;no of species of ions
	nb		= max(nspec)	;maximum of no. of sim particles out of all species

	;create arrays:
	;x grid evenly spaced
	xgrid	= xmax*findgen(nc+1)/(nc+1)
	;fields
	Ex		= dblarr(nc+1) 			;E fields
	Ey		= dblarr(nc+1)
	Ez		= dblarr(nc+1)
	By		= dblarr(nc+1)			;B fields
	Bz		= dblarr(nc+1)
	Ay		= dblarr(nc+1)			;A fields (vector magnetic potential)
	Az		= dblarr(nc+1)
	foy		= dblarr(nc+1)			;drag force (-resis*[del^2 A])
	foz		= dblarr(nc+1)
	;ions
	den		= dblarr(nc+1)			;density of ALL ions
	Vix		= dblarr(nc+1)			;velocities of ALL ions
	Viy		= dblarr(nc+1)
	Viz		= dblarr(nc+1)
	dns		= dblarr(nc+1,ns+1)		;density of each ion SPECIES
	vxs		= dblarr(nc+1,ns+1)		;velocities of each ion SPECIES
	vys		= dblarr(nc+1,ns+1)
	vzs		= dblarr(nc+1,ns+1)
	x		= dblarr(nb+1,nsp+1)		;ion positions
	vx		= dblarr(nb+1,nsp+1)		;ion velocities
	vy		= dblarr(nb+1,nsp+1)
	vz		= dblarr(nb+1,nsp+1)
	;under here has to do with initialising the ions
	tx0		= dblarr(nsp+1)
	vth		= dblarr(nsp+1)
	vbx		= dblarr(nsp+1)
	vb0		= dblarr(nsp+1)
	ans		= dblarr(nsp+1)
	vfac	= dblarr(nsp+1)
	vmax	= dblarr(nsp+1)
	pinv	= dblarr(nsp+1)
	dfac	= dblarr(nsp+1)
	frac	= dblarr(nsp+1)
	npi		= dblarr(nsp+1)
	;electrons
	Vey		= dblarr(nc+1)			;drift velocity of electron fluid
	Vez		= dblarr(nc+1)
	te		= dblarr(nc+1)			;temperature of electron fluid
	pe		= dblarr(nc+1)			;pressure of electron fluid

	;define some more stuff:
	;grid and dt
	hx		= xmax/nx				;cell size
	hxi		= 1./hx					;hx inverse
	dt		= wpiwci*dtwci			;this give dt in wpi units
	;fields
	thet	= theta*!dpi/180.		;theta in radians
	cth		= cos(thet)				;cos theta
	sth		= sin(thet)				;sin theta
	Bxc		= cth/wpiwci			;constant B field in x
	Byc		= 0.					;constant B field in y
	Bzc		= sth/wpiwci			;constant B field in z
	;electrons
	vye		= 0.					;electron fluid velocities
	vze		= 0.
	te0		= bete/(2.*wpiwci^2)	;the Te0 in the temperature model (5.24)
	pe0		= te0					;pressure term
	;ions
	;I NEED TO LOOK AT EVERYTHING BELOW THIS IN MORE DETAIL
	dnadd	= 0.					;dnadd and the next loop sums all the total densities of the species
	for k=1,nsp do dnadd = dnadd+dnspec(k)
	vmax	= 0.
	for k=1,nsp do begin
		tx0(k)	= btspec(k)/(2.*wpiwci^2)	;Tx0 for ions
	    vth(k)	= sqrt(2.*tx0(k)/wspec(k))
	    vbx(k)	= vbspec(k)/wpiwci
	    vb0(k)	= max([vbx(k),vth(k)])
	    ans(k)	= sqrt(anspec(k))
	    vfac	= max([1.,ans(k)])
	    vmax	= max([vmax,vfac*vb0(k)])
	    pinv(k)	= 1./(nspec(k))
	    dfac(k)	= xmax*pinv(k)/hx
	    frac(k)	= dnspec(k)/dnadd
	    npi(k)	= nspec(k)
	endfor
	;this initialises particles
	;(not sure the theory behind it!)
    for k=1,nsp do begin
	    for p=1,nspec(k) do begin
		    x(p,k)	= xmax*pinv(k)*(npi(k)-.5)
		    vmag	= sqrt(-alog(1.-randomu(s,/double)))
		    th		= 2*!dpi*randomu(s,/double)
		    vxa		= vth(k)*vmag*cos(th)+vbx(k)
		    vmag	= sqrt(-alog(1.-randomu(s,/double)))
		    th		= 2*!dpi*randomu(s,/double)
		    vy(p,k)	= vth(k)*ans(k)*vmag*sin(th)
		    vza		= vth(k)*ans(k)*vmag*cos(th)
		    vx(p,k)	= vxa*cth-vza*sth
		    vz(p,k)	= vza*cth+vxa*sth
		    npi(k)	= npi(k)-1
	    endfor
	endfor
	;now i need to put values in some arrays
	for i=1,nx2 do begin
		te(i)	= te0
		pe(i)	= pe0
		by(i)	= byc
		bz(i)	= bzc
	endfor

	;now we run one time step with dt=0 to initialise the fields
	;diagno tells diagnos which subroutine we're comign from
	dtsav	= dt
	dt		= 0.
	trans,		Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
				den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
				Vey,Vez,te,pe,										$;electrons
				firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
     			dtwci,vbspec,dnspec,btspec,dfac,frac,				$
     			bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
     			t,nspec,te0,wspec,nskip,vplim,ntimes,				$
     			npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
     			vxhist,tphist,txhist,ldec,nplth,diagno,				$
     			resis,theta,wpiwci,vb0,tx0,lastp
	;set dt back to 0.
	dt		= dtsav
	field,		Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
				den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
				Vey,Vez,te,pe,										$;electrons
				firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
     			dtwci,vbspec,dnspec,btspec,dfac,frac,				$
     			bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
     			t,nspec,te0,wspec,nskip,vplim,ntimes,				$
     			npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
     			vxhist,tphist,txhist,ldec,nplth,diagno,				$
     			resis,theta,wpiwci,vb0,tx0,lastp
	diagno	= 2
	diagnos,	Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
				den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
				Vey,Vez,te,pe,										$;electrons
				firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
     			dtwci,vbspec,dnspec,btspec,dfac,frac,				$
     			bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
     			t,nspec,te0,wspec,nskip,vplim,ntimes,				$
     			npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
     			vxhist,tphist,txhist,ldec,nplth,diagno,				$
     			resis,theta,wpiwci,vb0,tx0,lastp

return
end

;
;<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



;>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
;
;
; SUBROUTINE: trans
;
; collects the densities and the currents
;

;variables go here, copy and paste these to main program
pro trans,		Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
				den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
				Vey,Vez,te,pe,										$;electrons
				firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
     			dtwci,vbspec,dnspec,btspec,dfac,frac,				$
     			bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
     			t,nspec,te0,wspec,nskip,vplim,ntimes,				$
     			npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
     			vxhist,tphist,txhist,ldec,nplth,diagno,				$
     			resis,theta,wpiwci,vb0,tx0,lastp

	;first zero the source arrays.
	for i=1,nx2 do begin	;loop over each grid point
		den(i)	= 0				;zeroing for all ions
		Vix(i)	= 0
		Viy(i)	= 0
		Viz(i)	= 0
		for k=1,nsp do begin;loop over eaach species
			dns(i,k)	= 1e-20	;??? in winske code they make this just very small 1e-20, why?
			vxs(i,k)	= 0		;zeroing for each species of ion
			vys(i,k)	= 0
			vzs(i,k)	= 0
		endfor
	endfor

	;now call the particle moving and diagnostics subroutines
	parmov,			Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
					den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
					Vey,Vez,te,pe,										$;electrons
					firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
     				dtwci,vbspec,dnspec,btspec,dfac,frac,				$
     				bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
     				t,nspec,te0,wspec,nskip,vplim,ntimes,				$
     				npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
     				vxhist,tphist,txhist,ldec,nplth,diagno,				$
     				resis,theta,wpiwci,vb0,tx0,lastp
	diagno	= 1
	diagnos,		Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
					den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
					Vey,Vez,te,pe,										$;electrons
					firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
     				dtwci,vbspec,dnspec,btspec,dfac,frac,				$
     				bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
     				t,nspec,te0,wspec,nskip,vplim,ntimes,				$
     				npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
     				vxhist,tphist,txhist,ldec,nplth,diagno,				$
     				resis,theta,wpiwci,vb0,tx0,lastp

	;put ghost cell contribution back into the real domain via
	;periodic BC's
	for k=1,nsp do begin
		dns(2,k)	= dns(2,k)+dns(nx2,k)
	    vxs(2,k)	= vxs(2,k)+vxs(nx2,k)
	    vys(2,k)	= vys(2,k)+vys(nx2,k)
	    vzs(2,k)	= vzs(2,k)+vzs(nx2,k)
	    dns(nx1,k)	= dns(nx1,k)+dns(1,k)
	    vxs(nx1,k)	= vxs(nx1,k)+vxs(1,k)
	    vys(nx1,k)	= vys(nx1,k)+vys(1,k)
	    vzs(nx1,k)	= vzs(nx1,k)+vzs(1,k)
    endfor

	;normalise densities and velocities
	for k=1,nsp do begin
		for i=2,nx1 do begin
			vxs(i,k)	= vxs(i,k)/dns(i,k)
			vys(i,k)	= vys(i,k)/dns(i,k)
			vzs(i,k)	= vzs(i,k)/dns(i,k)
			dns(i,k)	= dns(i,k)*dfac(k)
		endfor
    endfor

	;set ghost cell values via
	;periodic BC's
	for k=1,nsp do begin
		dns(1,k)	= dns(nx1,k)
	    vxs(1,k)	= vxs(nx1,k)
	    vys(1,k)	= vys(nx1,k)
	    vzs(1,k)	= vzs(nx1,k)
	    dns(nx2,k)	= dns(2,k)
	    vxs(nx2,k)	= vxs(2,k)
	    vys(nx2,k)	= vys(2,k)
	    vzs(nx2,k)	= vzs(2,k)
    endfor

	;calculate total ion density and velocities
	for i=1,nx2 do begin
		for k=1,nsp do begin
			den(i)	= den(i)+dns(i,k)*frac(k)
		endfor
		if den(i) lt 0.05 then den(i)=0.05		;?? i'm not sure about this conditon? it's in winske don't know why
	endfor

	;now calculate the Vi's
	for i=1,nx2 do begin
		for k=1,nsp do begin
			Vix(i)	= Vix(i)+dns(i,k)*frac(k)*vxs(i,k)/den(i)
			Viy(i)	= Viy(i)+dns(i,k)*frac(k)*vys(i,k)/den(i)
			Viz(i)	= Viz(i)+dns(i,k)*frac(k)*vzs(i,k)/den(i)
		endfor
	endfor

;can call a smooth routine here if necessary

return
end

;
;<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



;>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
;
;
; SUBROUTINE: parmov
;
; write here what it does...
; move the particles
;

;variables go here, copy and paste these to main program
pro parmov,		Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
				den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
				Vey,Vez,te,pe,										$;electrons
				firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
     			dtwci,vbspec,dnspec,btspec,dfac,frac,				$
     			bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
     			t,nspec,te0,wspec,nskip,vplim,ntimes,				$
     			npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
     			vxhist,tphist,txhist,ldec,nplth,diagno,				$
     			resis,theta,wpiwci,vb0,tx0,lastp

	;alright, in the winske code they have a table called pbuf, which has height four times the number
	;of each species simulation ion (each ion has the same no), and width four columns. they make equivalent
	;the distance x(i) and speed vx,vy,vz(i) for each simulated particle to the four columns.
	;now, when they store values for each particle they store them every fourth row (this seems
	;to be why the table pbuf is this high). i can find no real reason for them having done this...
	;perhaps it was a remnant from a more advanced code that they had and converted to the 1d code.
	;anyway, point is i've changed it so it's simpler (hopefully!).

	;i have x,vx,vy,vz intuitive so that e.g. x(p,k) corresponds to the position of the p'th ion in the k'th species
	;vax,vay,vaz do not have to be remembered for each individual species and are just are used to store certain values

	;loop over all species
	for k=1,nsp do begin
		dta		= dt/wspec(k)	;h on pg110 after (5.13)
		dta2	= dta/2.
		;loop over all particles in the species
		for p=1,nspec(k) do begin
			rx		= x(p,k)*hxi+1.5
			i		= FIX(rx+0.5)	;eq(5.16)
			i1		= i+1
			fx		= rx-i			;W(I+1) eq(5.17)
			fxc		= 1.-fx			;W(I)   eq(5.18)
			;now we apply equation 5.15 and the E'=E-e*resis*J to E and B
			;(but CHECK THIS: in foy,fox apparently we have here -e*resis*J=-resis*[del^2 A])
			;these are the fields felt by the particle (they are weighted according to their position)
			;since they are not really the fields I have suppressed capitals on e and b
			exa		= Ex(i)*fxc+Ex(i1)*fx
      		eya		= Ey(i)*fxc+Ey(i1)*fx+foy(i)*fxc+foy(i1)*fx
      		eza		= Ez(i)*fxc+Ez(i1)*fx+foz(i)*fxc+foz(i1)*fx
      		bya		= By(i)*fxc+By(i1)*fx
      		bza		= Bz(i)*fxc+Bz(i1)*fx
      		;now define f and g and the vxa's to make things easier for later
      		f		= 1.-dta*dta2*(bxc^2+bya^2+bza^2)
      		g		= dta2*(vx(p,k)*bxc+vy(p,k)*bya+vz(p,k)*bza)
      		vxa		= vx(p,k)+dta2*exa
		    vya		= vy(p,k)+dta2*eya
		    vza		= vz(p,k)+dta2*eza
		    ;this employs eq (5.13) to find the velocities at N+1/2 time step
		    vx(p,k)	= f*vx(p,k)+dta*(exa+vya*bza-vza*bya+g*bxc)
		    vy(p,k)	= f*vy(p,k)+dta*(eya+vza*bxc-vxa*bza+g*bya)
		    vz(p,k)	= f*vz(p,k)+dta*(eza+vxa*bya-vya*bxc+g*bza)
		    ;now find the particle position at N+1 using the velocity at N+1/2 (leapfrog stylin')
		    dx		= vx(p,k)*dt
		    x(p,k)	= x(p,k)+dx
		    ;now we need to make sure and particles outside the simulation region (in ghost cells) are put back in
		    ;periodic BC's
		    if x(p,k) ge xmax then x(p,k)=x(p,k)-xmax
		    if x(p,k) le 0.   then x(p,k)=x(p,k)+xmax
		    ;now we push an extra half time step in v so that
		    ;the densities and currents are collected at the full time step N+1
		    rx		= x(p,k)*hxi+1.5000000000001
		    i		= FIX(rx+0.5)
		    i1		= i+1
		    fx		= rx-i
		    fxc		= 1.-fx
		    exa		= Ex(i)*fxc+Ex(i1)*fx
		    eya		= Ey(i)*fxc+Ey(i1)*fx+foy(i)*fxc+foy(i1)*fx
		    eza		= Ez(i)*fxc+Ez(i1)*fx+foz(i)*fxc+foz(i1)*fx
		    bya		= By(i)*fxc+By(i1)*fx
		    bza		= Bz(i)*fxc+Bz(i1)*fx
		    ;above is as before but now we push just using lorentz force (with a time step dt/2):
		    ;dv=[dt/(2*m)]*e(E+[v x B])=[h/2]*(E+[v x B]) then v(N+1/2)+dv is v(N+1)
		    vxa		= vx(p,k)+dta2*(exa+vy(p,k)*bza-vz(p,k)*bya)
		    vya		= vy(p,k)+dta2*(eya+vz(p,k)*bxc-vx(p,k)*bza)
		    vza		= vz(p,k)+dta2*(eza+vx(p,k)*bya-vy(p,k)*bxc)
		    ;then below sums up (over the loop) the velocities and densities of every ion in the species at each position i,i+1
		    vxs(i,k)	= vxs(i,k)+vxa*fxc
		    vxs(i1,k)	= vxs(i1,k)+vxa*fx
		    vys(i,k)	= vys(i,k)+vya*fxc
		    vys(i1,k)	= vys(i1,k)+vya*fx
		    vzs(i,k)	= vzs(i,k)+vza*fxc
		    vzs(i1,k)	= vzs(i1,k)+vza*fx
		    dns(i,k)	= dns(i,k)+fxc
		    dns(i1,k)	= dns(i1,k)+fx
		endfor
	endfor

return
end

;
;<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



;>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
;
; SUBROUTINE: field
;
; write here what it does...
; calculates the fields from time step N to time step N+1

;variables go here, copy and paste these to main program
pro field,		Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,						$;fields
				den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
				Vey,Vez,te,pe,										$;electrons
				firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
     			dtwci,vbspec,dnspec,btspec,dfac,frac,				$
     			bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
     			t,nspec,te0,wspec,nskip,vplim,ntimes,				$
     			npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
     			vxhist,tphist,txhist,ldec,nplth,diagno,				$
     			resis,theta,wpiwci,vb0,tx0,lastp


	gam 	= 2./3.			; gamma for temperature model (5.24)

	hx2		= hx/2.			; hx / 2
	dtx		= dt/(2.*hx)	; dt / ( 2 hx )
	hxs		= hx*hx			; hx^2

	;here set initial Ec, Fc, Gc (5.67-69) matrices so that the loop to produce them works
	;I keep them in lower case so as not to confuse with the E field
	ec11	= 1		;ec the 2x2 identity matrix
	ec12	= 0
	ec21	= 0
	ec22	= 1
	fc1		= 0		;fc the zero vector
	fc2		= 0
	gc11	= 0		;gc zero matrix
	gc12	= 0
	gc21	= 0
	gc22	= 0

	;need to set up e(i),f(i),g(i) matrices/vectors
	e11		= dblarr(nc+1)
	e12		= dblarr(nc+1)
	e21		= dblarr(nc+1)
	e22		= dblarr(nc+1)
	f1		= dblarr(nc+1)
	f2		= dblarr(nc+1)
	g11		= dblarr(nc+1)
	g12		= dblarr(nc+1)
	g21		= dblarr(nc+1)
	g22		= dblarr(nc+1)

	;now define the A, B, C, D arrays (5.48-57)
	;keep them in lower case again so as not to confuse with A potential and B field
	;first define some constants, looping over all cells (inner grid pts)
	for i=2,nx1 do begin

		df		= den(i)*resis/bxc			;(5.43)
		df1		= den(i)/(1.+df*df)			;<-| these will just help
		df2		= hx2*vix(i)*df1/bxc		;  | in defining the A,B,C,D
		df3		= hxs*df1					;  | matrices
		df4		= 1./(dt*bxc)				;<-|
		;now we can define the matrix entries for A,B,C by 5.49-54
		a12		= -df2
		a21		= -a12
		a11		= 1.-df*a12
		a22		= a11
		b12		= df3*df4
		b21		= -b12
		b11		= -2.-df*b12
		b22		= b11
		c12		= -a12
		c21		= -c12
		c11		= 1.-df*c12
		c22		= c11
		;now define V'iy,V'iz by 5.56-57
		viyy	= Viy(i)-Vix(i)*byc/bxc
		vizz	= Viz(i)-Vix(i)*bzc/bxc
		;and finally define D by 5.55
		d1		= -df3*(viyy+df*vizz-df4*(az(i)-df*ay(i)))
		d2		= -df3*(vizz-df*viyy+df4*(ay(i)+df*az(i)))

		;basically now we want to solve for e(2),f(2),g(2) by 5.67-69
		;the following if statement just says do this if i is 2, then
		;skip to the end of the loop, but don't if i>2.
		;loop10 and loop20 are just markers
		if i gt 2 then goto, loop10
		ddi		= 1./(b11*b22-b12*b21)			;b determinant inverse
      	e11(2)	= -ddi*(b22*c11-b12*c21)		;e(2)
      	e12(2)	= -ddi*(b22*c12-b12*c22)
      	e21(2)	= -ddi*(-b21*c11+b11*c21)
      	e22(2)	= -ddi*(-b21*c12+b11*c22)
      	f1(2)	=  ddi*(b22*d1-b12*d2)			;f(2)
      	f2(2)	=  ddi*(-b21*d1+b11*d2)
     	g11(2)	= -ddi*(b22*a11-b12*a21)		;g(2)
      	g12(2)	= -ddi*(b22*a12-b12*a22)
      	g21(2)	= -ddi*(-b21*a11+b11*a21)
      	g22(2)	= -ddi*(-b21*a12+b11*a22)
		goto, loop20

		loop10:
		;so now we want to solve for all e(I),f(I),g(I) using 5.60-62
		;first we define some quantities in the equations
		;the following h is the 2x2 matrix [a(I).e(I-1)+b(I)] in 5.60-62
		h11		= a11*e11(i-1)+a12*e21(i-1)+b11
      	h12		= a11*e12(i-1)+a12*e22(i-1)+b12
      	h21		= a21*e11(i-1)+a22*e21(i-1)+b21
      	h22		= a21*e12(i-1)+a22*e22(i-1)+b22
      	hdi		= 1./(h11*h22-h12*h21)			;h determinant inverse
      	;the following fd is the vector [d(I)-a(I).f(I-1)] in 5.61
      	fd1		= d1-a11*f1(i-1)-a12*f2(i-1)
      	fd2		= d2-a21*f1(i-1)-a22*f2(i-1)
      	;the following gd is the 2x2 matrix [a(I).g(I-1)] in 5.62
      	gd11	= a11*g11(i-1)+a12*g21(i-1)
      	gd12	= a11*g12(i-1)+a12*g22(i-1)
      	gd21	= a21*g11(i-1)+a22*g21(i-1)
      	gd22	= a21*g12(i-1)+a22*g22(i-1)
      	;now here are the equations 5.61-62
      	e11(i)	= -hdi*( h22*c11 -h12*c21)
      	e12(i)	= -hdi*( h22*c12 -h12*c22)
      	e21(i)	= -hdi*(-h21*c11 +h11*c21)
      	e22(i)	= -hdi*(-h21*c12 +h11*c22)
      	f1(i)	=  hdi*( h22*fd1 -h12*fd2)
      	f2(i)	=  hdi*(-h21*fd1 +h11*fd2)
      	g11(i)	= -hdi*( h22*gd11-h12*gd21)
      	g12(i)	= -hdi*( h22*gd12-h12*gd22)
      	g21(i)	= -hdi*(-h21*gd11+h11*gd21)
      	g22(i)	= -hdi*(-h21*gd12+h11*gd22)

      	;now here's the code that will generate ec,fc,gc (5.72-74) by the end of the loop
      	;in Winske there's a 'tc' that i've omitted since i don't think it's necessary
   		fc1		= fc1+ec11*f1(i-1)+ec12*f2(i-1)
      	fc2		= fc2+ec21*f1(i-1)+ec22*f2(i-1)
      	gc11	= gc11+ec11*g11(i-1)+ec12*g21(i-1)
      	gc12	= gc12+ec11*g12(i-1)+ec12*g22(i-1)
      	gc21	= gc21+ec21*g11(i-1)+ec22*g21(i-1)
      	gc22	= gc22+ec21*g12(i-1)+ec22*g22(i-1)
      	ec11	= ec11*e11(i-1)+ec12*e21(i-1)
      	ec12	= ec11*e12(i-1)+ec12*e22(i-1)
      	ec21	= ec21*e11(i-1)+ec22*e21(i-1)
      	ec22	= ec21*e12(i-1)+ec22*e22(i-1)

		loop20:

	endfor

	;now that we have all the e(I),f(I),g(I) and have constructed ec,fc,gc
	;we can solve for X(NX1) by 5.76. note that X(I) is just the vector magnetic
	;potential vector A(y,z) at position I
	;first we define some quantities in the equation 5.76
	en11	= e11(nx)+g11(nx)		;e(nx)+g(nx)
    en12	= e12(nx)+g12(nx)
    en21	= e21(nx)+g21(nx)
    en22	= e22(nx)+g22(nx)
    ec11	= ec11+gc11				;ec+gc (we store it back in ec)
    ec12	= ec12+gc12
    ec21	= ec21+gc21
    ec22	= ec22+gc22
    h11		= a11*en11+a12*en21+b11+c11*ec11+c12*ec21	;a(nx1).[(e(nx)+g(nx)]+b(nx1)+c(nx1).[ec+gc]
    h12		= a11*en12+a12*en22+b12+c11*ec12+c12*ec22
    h21		= a21*en11+a22*en21+b21+c21*ec11+c22*ec21
    h22		= a21*en12+a22*en22+b22+c21*ec12+c22*ec22
    hdi		= 1./(h11*h22-h12*h21)						;determinant inverse of above
    p1		= d1-a11*f1(nx)-a12*f2(nx)-c11*fc1-c12*fc2	;d(nx1)-a(nx1).f(nx)-c(nx1).fc
    p2		= d2-a21*f1(nx)-a22*f2(nx)-c21*fc1-c22*fc2
    ;now we can solve for X(nx1)
    Ey(nx1)	= ay(nx1)				;here we STORE THE Ay,Az AT TIME STEP N IN Ey,Ez.
    Ez(nx1)	= az(nx1)
    Ay(nx1)	= hdi*(h22*p1-h12*p2)	;and this is the Ay,Az AT TIME STEP N+1
    Az(nx1)	= hdi*(-h21*p1+h11*p2)

    ;now we can get all the X(I)'s using 5.58
    ;loop over all from I=nx->2
    for ii=2,nx do begin
		i		= nx2-ii
		Ey(i)	= Ay(i)		;again store A at time step N in E
		Ez(i)	= Az(i)
		Ay(i)	= e11(i)*Ay(i+1)+e12(i)*Az(i+1)+f1(i)+g11(i)*Ay(nx1)+g12(i)*Az(nx1)		;A at time step N+1
		Az(i)	= e21(i)*Ay(i+1)+e22(i)*Az(i+1)+f2(i)+g21(i)*Ay(nx1)+g22(i)*Az(nx1)
    endfor

    ;apply periodic BC's
    Ey(nx2)	= Ay(nx2)
    Ez(nx2)	= Az(nx2)
    Ay(nx2)	= Ay(2)
    Az(nx2)	= Az(2)
    Ey(1)	= Ay(1)
    Ez(1)	= Az(1)
    Ay(1)	= Ay(nx1)
    Az(1)	= Az(nx1)

	;now we have all the X(I)'s (actually the Ay,Az vector potentials)
    ;so now can get Ey,Ez (5.35-36) and By,Bz (5.31-32)
    for i = 2,nx1 do begin
    	Ey(i)	= (Ey(i)-Ay(i))/dt		;note equation is of this form because A at time step N
		Ez(i)	= (Ez(i)-Az(i))/dt		;was stored in the Ey,Ez vector
		By(i)	= (Az(i-1)-Az(i+1))*hxi/2+Byc
		Bz(i)	= (Ay(i+1)-Ay(i-1))*hxi/2+Bzc
	endfor

    ;apply periodic BC's
    Ey(1)	= Ey(nx1)
    Ez(1)	= Ez(nx1)
    Ey(nx2)	= Ey(2)
    Ez(nx2)	= Ez(2)
    By(nx2)	= By(2)
    Bz(nx2)	= Bz(2)
    By(1)	= By(nx1)
    Bz(1)	= Bz(nx1)

	;find del^2 of A, electron velocities (Ve), the drag force, electron temp and pressure inside 5.77,78
	for i=2,nx1 do begin
		Ajy		= -(Ay(i+1)+Ay(i-1)-2.*Ay(i))/hxs	;del^2 A, central difference
      	Ajz		= -(Az(i+1)+Az(i-1)-2.*Az(i))/hxs
      	Vey(i)	= -Ajy/(den(i))+Viy(i)				;electron velocity (5.78)
      	Vez(i)	= -Ajz/(den(i))+Viz(i)
      	foy(i)	= -resis*Ajy						;drag force definition
      	foz(i)	= -resis*Ajz
      	te(i)	= te0*(den(i)^gam)					;this depends on temperature model for the electrons
      	if iemod eq 0 then te(i) = te0				;iemod switches between 5.23 for 0 and 5.24 for 1
		pe(i)	= te(i)*den(i)						;pressure term (5.5)
	endfor
	;apply periodic BC's
	vey(nx2)=vey(2)
    vez(nx2)=vez(2)
    vey(1)=vey(nx1)
    vez(1)=vez(nx1)
    te(nx2)=te(2)
    pe(nx2)=pe(2)
    te(1)=te(nx1)
    pe(1)=pe(nx1)

    ;now we can find Ex using 5.77,78
    for i=2,nx1 do begin
      	Ex(i)	= Vez(i)*By(i)-Vey(i)*Bz(i)-(hxi/2)*(pe(i+1)-pe(i-1))/den(i)
    endfor
    ;apply periodic BC's
    Ex(1)	= Ex(nx1)
    Ex(nx2)	= Ex(2)

return
end

;
;<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


;>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
;
; SUBROUTINE: diagnos
;
; write here what it does...
; does plots and such

;variables go here, copy and paste these to main program
pro diagnos,		Ex,Ey,Ez,By,Bz,Ay,Az,foy,foz,					$;fields
				den,Vix,Viy,Viz,dns,vxs,vys,vzs,x,vx,vy,vz,			$;ions
				Vey,Vez,te,pe,										$;electrons
				firstp,nx,nx1,nx2,hx,hxi,xmax,dt,xgrid,				$;parameters (might be able to delete some of these)
     			dtwci,vbspec,dnspec,btspec,dfac,frac,				$
     			bxc,byc,l3,l4,nb,nc,ns,bete,anspec,nsp,bzc,it,		$
     			t,nspec,te0,wspec,nskip,vplim,ntimes,				$
     			npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist,ehist,	$
     			vxhist,tphist,txhist,ldec,nplth,diagno,				$
     			resis,theta,wpiwci,vb0,tx0,lastp

if diagno eq 1 then begin

	window,0,xsize=1900,ysize=1000,xpos=1280
	!p.multi	= [0,2,3,0,1]
	!p.charsize	= 1.5
	;!y.range	= [-0.1e-3,0.1e-3]
	!x.range	= [0,xmax]
	!x.style	= 1
	!y.style	= 1
	plot,  x(*,1),vx(*,1), psym=3,color=249, yrange=[-0.002,0.002], title='v_x'
	oplot, x(*,2),vx(*,2), psym=3,color=150
	oplot, x(2500:2506,1),vx(2500:2506,1), psym=1,color=212
	oplot, x(2500:2506,2),vx(2500:2506,2), psym=1,color=128
	plot,  x(*,1),vy(*,1), psym=3,color=249, yrange=[-0.002,0.002], title='v_y'
	oplot, x(*,2),vy(*,2), psym=3,color=150
	oplot, x(2500:2506,1),vy(2500:2506,1), psym=1,color=212
	oplot, x(2500:2506,2),vy(2500:2506,2), psym=1,color=128
;	plot,  x(*,1),vz(*,1), psym=3,color=249, yrange=[-0.001,0.001], title='v_z'
;	oplot, x(*,2),vz(*,2), psym=3,color=150
;	oplot, x(2500:2506,1),vz(2500:2506,1), psym=1,color=212
;	oplot, x(2500:2506,2),vz(2500:2506,2), psym=1,color=128

endif

if diagno eq 2 then begin

	plot,  xgrid[1:nc],dns(1:nc,1)		, color=249, yrange=[0,3.], title = 'Normalised Density'
	oplot, xgrid[1:nc],dns(1:nc,2)		, color=150
	plot,  xgrid[1:nc],By[1:nc]				, color=150, title = 'B_y'
	plot,  xgrid[1:nc],Bz[1:nc]				, color=150, title = 'B_z'
	plot,  xgrid[1:nc],(atan(Bz/By))[1:nc]	, color=150, yrange=[-!dpi/2,!dpi/2], title = 'phi = arctan(B_z/B_y)', $
		subtitle='Beam: red, Cold plasma: green, Time: '+Strtrim(t,2)+' wci units'
	Writefile=1
	if Writefile eq 1 then begin
	;filename='C:\Documents and Settings\Chris\My Documents\All University Stuff\2009\2nd Semester\PHYS3330\Data2\picE100'+Strtrim(t*1.0,2)+'B.png'
	filename='C:\Data\Hybrid_Jackson\'+Strtrim(t,2)+'.png'
	image=tvrd(0,0,!d.x_size,!d.y_size,true=1)
	write_png,filename,image,r,g,b
	;print,'File written to', filename
	endif

endif

return
end

;
;<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<