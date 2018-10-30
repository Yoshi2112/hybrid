c================== beginning of hybrid1.f ===============================
c
c manual.h
c
c The hybrid code package consists of 4 files:
c
c 1. manual.h: a simple manual summarizing the contents of these files
c and their use.
c
c 2. h1.f: the 1-D hybrid code, as discussed in the lecture notes
c
c 3. p5.f: plotting package fortran routines
c
c 4. p4.c: rest of the plotting package in c
c
c &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
c
c manual.f
c
c &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
c
c The manual is divided into two parts: (I) deals with the hybrid  code,
c and (II) with the plotting package.
c
c Both should compile and run on most systems with a minimum
c number of changes. The system dependent parts of the plotting
c package will be discussed in II.
c
c I. hybrid code: h1.f
c
c h1.f is a 1-D hybrid (particle ion, massless fluid electron)
c electromagnetic code. The version here has periodic boundary
c conditions, although these can be easily modified as discussed
c in the lecture notes. The code uses an implicit method to
c solve the field equations that is acceptable for most
c applications. A basic set of diagnostics is also included
c which involve graphics via the enclosed plotting package. In
c order to facilitate understanding of the structure of the code,
c no attempt has been made to optimize the code for speed, storage,
c etc.
c
c h1.f consists of the following subroutines:
c
c main:  opens the files, reads the data, runs the main loop
c
c init: initializes the particles and fields
c
c trans: collects the density and currents
c
c parmov: moves the ions
c
c field: solves the electromagnetic field and electron equations
c
c diagnos: plots and prints the output
c
c smooth: a smoothing routine
c
c endrun: writes message on premature exit
c
c angle: computes phase angle
c
c rft2, rfi2, rtran2, fft2: fourier transform routines
c
c
c Parameters to initialize the run are put into a namelist
c called DATUM in a file INPUT, of the form:
c
c  $DATUM variable1=value1,variable2=value2,...$
c (where the first $ is in column 2)
c
c The run produces plots in a PLOT file (may be system dependent),
c written output in an OUTPUT file, and magnetic field arrays at various
c times in files BYS, BZS that may be further postprocessed.
c
c The namelist parameters are:
c
c nx=   no. of cells (usually 2**N for fourier transform diagnostics)
c xmax=   system length  in c/wpi
c dtwci=   time step in wci**-1
c theta=   angle between external Bo and x axis
c wpiwci=  ratio of wpi to wci
c resis=   resistivity (usually 0)
c iemod=   electron model (0 for Te=const, 1 for adiabatic)
c bete=   electron beta
c ntimes=  no. of time steps
c
c ion parameters (several species):
c nsp: no. of ion species
c dnspec: density of each species (no=total)
c vbspec: velocity/alfven speed for each species (va**2=bo**2/4 pi no mo)
c wspec: mass/proton mass for each species
c btspec: plasma beta for each species (beta-j=8 pi no t-j /bo**2)
c anspec: T-perp/T-parallel for each species
c nspec: no. of simulation particles for each species
c
c plotting parameters:
c nskip=   1/fraction ions plotted
c npltp= time step increment for particle plots
c npltf= time step increment for field plots
c nplth= time step increment for history plots
c nwrth= time step increment for history writes
c nwrtf= time step increment for field writes
c
c
c II. The plotting package consists of four parts
c
c A. basic routines that are system dependent
c
c B. the high level routines (which are used in diagnos)
c
c C. intermediate level routines
c
c D. other routines in c (in p4.c) that involve bit manipulation
c
c
c A. The basic routines are:
c
c 1. begplt--initializes the plot routine and makes a metafile.
c
c 2. endplt--clears buffers and closes plot file
c
c 3. adv--advances the frame
c
c 4. movabs--moves pointer to new position
c
c 5. drwabs--draws a line from the current position to the specified position
c
c 6. pntabs--draws a point at the specified position.
c
c 7. seeloc--checks pointer location
c
c The version enclosed has the following system (LANL) dependent routines
c in it:
c
c gplot, gdone, gpage, gmova2, glina2, gmrka2, gwind2
c
c which you will have to replace.
c
c gplot: opens the plot file
c
c gdone: closes the plot file
c
c gpage: advances the frame
c
c gmova2: moves pointer to ix,iy
c
c glina2:  draws line from current position to ix,iy
c
c gmrka2: plots point at ix,iy
c
c gwind2: converts plotting coordinates to (LANL) system coordinates.
c
c
c If your system has these lowest level routines, the conversion is very easy.
c
c It is likely that you will have all of these, except gwind2, in which case
c you have to put the conversion from the plotting package to your
c system in each of the routines: movabs, pntabs, drwabs separately.
c
c the plotting package has integer coordinates:0 .le. ix .le. 1023,
c 0 .le. iy .le. 780 with 0,0 in the lower left corner.
c
c so you would need to modify a routine like movabs as follows:
c
c subroutine movabs(ix,iy)
c common...
c x=f1(ix)
c y=f2(iy)
c call "move"(x,y)
c ixpos=ix
c iypos=iy
c return
c end
c
c where f1, f2 is the linear conversion from my coordinate system (ix,iy)
c to yours (x,y).
c and "move" is your routine to move the pointer to your coordinates x,y
c
c 2. the highest level routines are lplot and pplotc; the arguments
c are described in some detail below.
c
c lplot:
c       subroutine lplot (imx,imy,npts,x,y,inc,iop,ltitle,ntitle,lxname
c      1 ,nxname,lyname,nyname)
c
c
c      lplot plots the values in x and y and connects them with a line.
c  it draws a box around the plot with scaling and places labels on the
c  left and bottom axes.
c       imx indicates the location and length of the x-axis on a frame.
c  imx =a              x-coordinate of the origin           length of the x-axis
c    1                   left-hand side of page                 full page
c    2                   left-hand side of page                 half page
c    3                   middle of page                         half page
c
c   imy indicates the location and length of the y-axis on a frame.
c  imy =               y-coordinate of the origin           length of the y-axis
c    1                    bottom of page                        full page
c    2                    middle of page                        half page
c    3                    bottom of page                        half page
c    4                    top third of page                     third of page
c    5                    middle third of page                  third of page
c c   6                    bottom third of page                  third of page
c  ___________
c  a imx, imy < 0  gives exact scaling; > 0  gives automatic scaling.
c
c
c       npts.  the absolute value of npts is the number of elements in y to be
c             plotted.  an npts < 0  draws a curve onto a frame previously set
c             up by a call to lplot with iop < 0.
c       x is the table of abcissa values to be plotted.
c       y is the table of ordinate values to be plotted.
c       inc.  the absolute value of inc is the spacing between each successive
c             x and y element to be plotted.
c             inc < 0  x(1) = xmin and x(2) = dx
c       iop, the scaling for the x- and y-axes
c
c  ____iop__                  ________significance_______
c  = 1             linear x-axis and linear y-axis
c  = 2             linear x, log y
c  = 3             log x-axis, linear y-axis
c  = 4             log x-axis, log y-axis
c  < 0             scales and draws frame
c
c       ltitle is the the graph title.
c       ntitle is the number of characters in ltitle, maximum of 16.
c       lxname is the x-axis label.
c       nxname is the number of characters in lxname, maximum of 16.
c       lyname is the y-axis label.
c       nyname is the number of characters in lyname, maximum of 16.
c              nyname < 0  scales and tick marks for the plot are not drawn.
c
c pplotc:
c       subroutine pplotc (imx,imy,npts,x,y,inc,z,zmin,zmax)
c
c       pplot plots values in x and y.  each point  is  represented  by  a
c  plotting dot, and adjacent points are not connected.
c       pplotc  plots  the  point  only  if  its z value is less than zmax and
c  greater than zmin.
c       both subroutines assume that the frame, scale, and labels for this imx and
c  imy  plot  have  been  generated  by  lplot  with iop = -1.  only linear-linear
c  scaling is allowed.  npts is the number of elements in x, y, and z.
c  both  routines  have been optimized to plot many particles as dots.  point plot
c  arguments are summarized as follows.
c
c      imx and imy are the same as in lplot above
c       npts is the number of elements in the arrays x, y, and z.
c       x is the table of abcissa values.
c       y is the table of ordinate values.
c       inc.  the absolute value of inc is the spacing value between each
c             successive x and y element.
c             inc < 0 x(1) = xmin and x(2) = dx
c       z is a function of x and y.
c       zmin is the smallest value to be plotted.
c       zmax is the largest value to be plotted.
c
c C. intermediate level routines (generally not used directly by user)
c
c 1. dlch (dlcv): draws large characters horizontally (vertically)
c
c 2. dlglg (dlnln, dlnlg, dlgln) graws grids (linear or log)
c
c 3. sblin (sblog): prints scale on bottom axis
c
c 4. sllin: prints scale on left axis
c
c 5. maxv (minv): finds max (min) of array
c
c 6. ascl: determines numerical scale for grid
c
c 7. convrt: converts real no. to grid coordinates
c
c 8. dga: determines frame coordinates for grid
c
c 9. gxa: draws horizontal axis
c
c 10. gya: draws vertical axis
c
c 11. maxm (minm): finds max (min) of array and its index
c
c 12. lbltop (lblbot): puts lable on top (bottom) of frame
c
c 13. frame9: defines grid area, grid, labels, etc.
c
c************************************************************************
c
c
c  input files for test runs
c
c
c
c  $datum ntimes=2001, dtwci=0.05, nx=128, xmax=256.,
c  npltp=200, npltf=200, nplth=1000, nwrtf=20, nwrth=40,
c  wpiwci=10000., nsp=2, nspec=5120,5120, vbspec=9.85,-0.15,
c  dnspec=0.015,0.985, btspec=1.,1., anspec=1.,1., wspec=1.,1.,
c  bete=1., resis=0., theta=0., iemod=0, nskip=2,
c  label='1-d hybrid test 1: resonant instability'
c  $
c
c
c  $datum ntimes=2001, dtwci=0.025, nx=128, xmax=128.,
c  npltp=200, npltf=200, nplth=1000, nwrtf=20, nwrth=40,
c  wpiwci=10000., nsp=2, nspec=5120,5120, vbspec=9.00,-1.00,
c  dnspec=0.10,0.900, btspec=1.,1., anspec=1.,1., wspec=1.,1.,
c  bete=1., resis=0., theta=0., iemod=0, nskip=2,
c  label='1-d hybrid test 2: nonresnt instability'
c  $
c
c
c  $datum ntimes=1001, dtwci=0.05, nx=128, xmax=128.,
c  npltp=200, npltf=200, nplth=1000, nwrtf=20, nwrth=40,
c  wpiwci=10000., nsp=2, nspec=5120,5120, vbspec=0.90,-0.10,
c  dnspec=0.10,0.900, btspec=10.,1., anspec=5.,1., wspec=1.,1.,
c  bete=1., resis=0., theta=0., iemod=0, nskip=2,
c  label='1-d hybrid test 3: anistrpy instability'
c  $
c
c
c*********************************************************************
c   h1.f (1-d hybrid code....version 4/28/91)
c
      program hybrid
c*************
      parameter (ncc=258, nbb=12800, nss=2)
      common pbuf(4,nbb)
      common /fields/ ex(ncc), ey(ncc), ez(ncc), by(ncc), bz(ncc),
     1 ay(ncc), az(ncc), foy(ncc), foz(ncc)
      common /ions/ den(ncc),vix(ncc),viy(ncc),viz(ncc),dns(ncc,nss),
     1 vxs(ncc,nss), vys(ncc,nss), vzs(ncc,nss)
      common /etrons/ vey(ncc), vez(ncc), te(ncc), pe(ncc)
      dimension x(1), vx(1), vy(1), vz(1)
      equivalence (x(1),pbuf(1,1)),(vx(1),pbuf(2,1)),(vy(1),pbuf(3,1)),
     1 (vz(1),pbuf(4,1))
      common /params/ firstp, nx, nx1, nx2, hx, hxi, xmax, dt,
     1 dtwci,vbspec(nss),dnspec(nss),btspec(nss),dfac(nss),frac(nss),
     2 bxc, byc, l3, l4, nb, nc, ns, bete, anspec(nss), nsp, bzc, it,
     3 t, nspec(nss), te0, wspec(nss), nskip, vplim, ntimes,
     4 npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist(501),ehist(501),
     5 vxhist(501,nss), tphist(501,nss), txhist(501,nss), ldec, nplth,
     6 resis, theta, wpiwci, vb0(nss), tx0(nss), lastp
c************
      character label*64
      data it /0/, t /0./
      data nx /128/, xmax /256./, dtwci /.05/
      data nspec /3200,3200/, wspec /1.,1./
      data vbspec /9.85,-.15/, dnspec /0.015,0.985/bete /1./
      data btspec /1.,1./anspec /1.,1./, nsp /2/
      data theta /00./, wpiwci /10000./
      data nskip /1/, ldec /0/, nplth /2000/
      data nwrtf /20/, nwrth /40/, npltp /400/, npltf /400/
      data label /' 1-D hybrid code test                    '/
      data  resis /0.e-6/, iemod /0/
      data ntimes /4/
      namelist /datum/ ntimes,xmax,dtwci,nskip,npltp,vbspec,
     1 dnspec,btspec,bete,anspec,nspec,nsp,npltf,nwrth,nwrtf,
     2 iemod,resis,nx,wspec,label,theta,wpiwci,nplth
      nc=ncc
      nb=nbb
      ns=nss
      nbbb=nbb
c  open plot file and read namelist
      open (8,file='input',form='formatted',status='old',access=
     1  'sequential')
      open (9,file='output',form='formatted',access='sequential')
      open (3,file='bys',form='unformatted',access='sequential')
      open (4,file='bzs',form='unformatted',access='sequential')
      call begplt('        ')
      read (8,datum)
      call lbltop (label,64)
      write (9,160) label
c  check dimensions
      npart=0
      do 20 k=1,nsp
   20 npart=npart+nspec(k)
      if (nsp.gt.ns) call endrun ('species')
      if (nx+2.gt.nc) call endrun ('dimen')
      if (npart.gt.nbbb) call endrun ('parts')
c  initialize the run
      call init
      write (9,170)
c  main time loop
   30 if (it.lt.ntimes) go to 40
      go to 50
   40 it=it+1
      t=t+dtwci
      call trans
      call field
      call diag2
      go to 30
   50 continue
c  close plot file and exit
      call endplt
      call exit
c
  160 format (64a)
  170 format ('    initialization completed')
      end
      subroutine init
c*************
      parameter (ncc=258, nbb=12800, nss=2)
      common pbuf(4,nbb)
      common /fields/ ex(ncc), ey(ncc), ez(ncc), by(ncc), bz(ncc),
     1 ay(ncc), az(ncc), foy(ncc), foz(ncc)
      common /ions/ den(ncc),vix(ncc),viy(ncc),viz(ncc),dns(ncc,nss),
     1 vxs(ncc,nss), vys(ncc,nss), vzs(ncc,nss)
      common /etrons/ vey(ncc), vez(ncc), te(ncc), pe(ncc)
      dimension x(1), vx(1), vy(1), vz(1)
      equivalence (x(1),pbuf(1,1)),(vx(1),pbuf(2,1)),(vy(1),pbuf(3,1)),
     1 (vz(1),pbuf(4,1))
      common /params/ firstp, nx, nx1, nx2, hx, hxi, xmax, dt,
     1 dtwci,vbspec(nss),dnspec(nss),btspec(nss),dfac(nss),frac(nss),
     2 bxc, byc, l3, l4, nb, nc, ns, bete, anspec(nss), nsp, bzc, it,
     3 t, nspec(nss), te0, wspec(nss), nskip, vplim, ntimes,
     4 npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist(501),ehist(501),
     5 vxhist(501,nss), tphist(501,nss), txhist(501,nss), ldec, nplth,
     6 resis, theta, wpiwci, vb0(nss), tx0(nss), lastp
c************
      dimension pinv(nss), vbx(nss), npi(nss), ans(nss), vth(nss)
c  write out input variables
      write(9,100)
      write(9,101)nx
      write(9,102)xmax
      write(9,103)dtwci
      write(9,104)theta
      write(9,105)wpiwci
      write(9,106)resis
      write(9,107)iemod
      write(9,108)bete
      write(9,109)
      write(9,110)(is,dnspec(is),vbspec(is),btspec(is),anspec(is),
     1 wspec(is),nspec(is),is=1,nsp)
      write(9,111)nskip
      write(9,112)npltp
      write(9,113)npltf
      write(9,114)nplth
      write(9,115)nwrth
      write(9,116)nwrtf
      write(9,117)ntimes
c  define some variables from inputs
      nx1=nx+1
      nx2=nx+2
      hx=xmax/float(nx)
      hxi=1./hx
      dt=wpiwci*dtwci
      thet=theta*1.74533e-2
      cth=cos(thet)
      sth=sin(thet)
      bxc=cth/wpiwci
      byc=0.
      bzc=sth/wpiwci
      vye=0.
      vze=0.
      te0=bete/(2.*wpiwci**2)
      pe0=te0
      dnadd=0.
      do 10 k=1,nsp
   10 dnadd=dnadd+dnspec(k)
      vmax=0.
      do 20 k=1,nsp
      tx0(k)=btspec(k)/(2.*wpiwci**2)
      vth(k)=sqrt(2.*tx0(k)/wspec(k))
      vbx(k)=vbspec(k)/wpiwci
      vb0(k)=amax1(vbx(k),vth(k))
      ans(k)=sqrt(anspec(k))
      vfac=amax1(1.,ans(k))
      vmax=amax1(vmax,vfac*vb0(k))
      pinv(k)=1./(nspec(k))
      dfac(k)=xmax*pinv(k)/hx
      frac(k)=dnspec(k)/dnadd
      npi(k)=nspec(k)
   20 continue
      vplim=2.*vmax
c  initialize particles
      l4=0
      do 50 kk=1,nsp
      isp=kk
      l3=1+l4
      l4=4*nspec(kk)+l4
      do 40 l=l3,l4,4
      x(l)=xmax*pinv(isp)*(npi(isp)-.5)
      vmag=sqrt(-alog(1.-.999999*rand(0)))
      th=6.28131853*rand(0)
      vxa=vth(isp)*vmag*cos(th)+vbx(isp)
      vmag=sqrt(-alog(1.-.999999*rand(0)))
      th=6.28131853*rand(0)
      vy(l)=vth(isp)*ans(isp)*vmag*sin(th)
      vza=vth(isp)*ans(isp)*vmag*cos(th)
      vx(l)=vxa*cth-vza*sth
      vz(l)=vza*cth+vxa*sth
      npi(isp)=npi(isp)-1
   40 continue
   50 continue
c  initialize field arrays
      do 60 i=1,nx2
      ay(i)=0.
      az(i)=0.
      vey(i)=0.
      vez(i)=0.
      te(i)=te0
      pe(i)=pe0
      by(i)=byc
      bz(i)=bzc
      ex(i)=0.
      ey(i)=0.
      ez(i)=0.
      foy(i)=0.
      foz(i)=0.
   60 continue
      write (9,70)
c  run one time step with dt=0 to initialize fields, diagnostics
      dtsav=dt
      dt=0.
      call trans
      write (9,80)
      dt=dtsav
      call field
      write (9,90)
      call diag2
      return
c
   70 format (' init call to  trans  ')
   80 format ('  init call to field  ')
   90 format ('  init call to diagnos')
  100 format(' namelist input parameters/values:      ')
  101 format('       nx=   no. of cells               ',i15)
  102 format('     xmax=   system length              ',f15.1)
  103 format('    dtwci=   time step                  ',f15.3)
  104 format('    theta=   angle x-Bo                 ',f15.1)
  105 format('   wpiwci=   wpi/wci                    ',f15.1)
  106 format('    resis=   resistivity                ',e15.5)
  107 format('    iemod=   electron model             ',i15)
  108 format('     bete=   electron beta              ',f15.3)
  109 format('   species','   density','     vb/va','      beta',
     1 '   anstrpy','     mi/mp',' #partcles')
  110 format(i10,f10.3,f10.2,f10.3,f10.2,f10.1,i10)
  111 format('    nskip=   1/fraction ions plotted    ',i15)
  112 format('    npltp=   increment: particle plots  ',i15)
  113 format('    npltf=   increment: field plots     ',i15)
  114 format('    nplth=   increment: history plots   ',i15)
  115 format('    nwrth=   increment: history writes  ',i15)
  116 format('    nwrtf=   increment: field writes    ',i15)
  117 format('    ntimes   no. of time steps          ',i15)
      end
      subroutine trans
c*************
      parameter (ncc=258, nbb=12800, nss=2)
      common pbuf(4,nbb)
      common /fields/ ex(ncc), ey(ncc), ez(ncc), by(ncc), bz(ncc),
     1 ay(ncc), az(ncc), foy(ncc), foz(ncc)
      common /ions/ den(ncc),vix(ncc),viy(ncc),viz(ncc),dns(ncc,nss),
     1 vxs(ncc,nss), vys(ncc,nss), vzs(ncc,nss)
      common /etrons/ vey(ncc), vez(ncc), te(ncc), pe(ncc)
      dimension x(1), vx(1), vy(1), vz(1)
      equivalence (x(1),pbuf(1,1)),(vx(1),pbuf(2,1)),(vy(1),pbuf(3,1)),
     1 (vz(1),pbuf(4,1))
      common /params/ firstp, nx, nx1, nx2, hx, hxi, xmax, dt,
     1 dtwci,vbspec(nss),dnspec(nss),btspec(nss),dfac(nss),frac(nss),
     2 bxc, byc, l3, l4, nb, nc, ns, bete, anspec(nss), nsp, bzc, it,
     3 t, nspec(nss), te0, wspec(nss), nskip, vplim, ntimes,
     4 npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist(501),ehist(501),
     5 vxhist(501,nss), tphist(501,nss), txhist(501,nss), ldec, nplth,
     6 resis, theta, wpiwci, vb0(nss), tx0(nss), lastp
c************
c  zero source arrays
      do 10 i=1,nx2
      den(i)=0.
      vix(i)=0.
      viy(i)=0.
      viz(i)=0.
   10 continue
      do 20 k=1,nsp
      do 20 i=1,nx2
      dns(i,k)=1.e-20
      vxs(i,k)=0.
      vys(i,k)=0.
      vzs(i,k)=0.
   20 continue
c move particles and do particle diagnostics
      call parmov
      call diag1
c put ghost cell contributions into real domain
      do 50 k=1,nsp
      dns(2,k)=dns(2,k)+dns(nx2,k)
      vxs(2,k)=vxs(2,k)+vxs(nx2,k)
      vys(2,k)=vys(2,k)+vys(nx2,k)
      vzs(2,k)=vzs(2,k)+vzs(nx2,k)
      dns(nx1,k)=dns(nx1,k)+dns(1,k)
      vxs(nx1,k)=vxs(nx1,k)+vxs(1,k)
      vys(nx1,k)=vys(nx1,k)+vys(1,k)
      vzs(nx1,k)=vzs(nx1,k)+vzs(1,k)
   50 continue
c normalize species densities and velocities
      do 60 k=1,nsp
      do 60 i=2,nx1
      vxs(i,k)=vxs(i,k)/dns(i,k)
      vys(i,k)=vys(i,k)/dns(i,k)
      vzs(i,k)=vzs(i,k)/dns(i,k)
      dns(i,k)=dns(i,k)*dfac(k)
   60 continue
c set ghost cell values
      do 70 k=1,nsp
      dns(1,k)=dns(nx1,k)
      vxs(1,k)=vxs(nx1,k)
      vys(1,k)=vys(nx1,k)
      vzs(1,k)=vzs(nx1,k)
      dns(nx2,k)=dns(2,k)
      vxs(nx2,k)=vxs(2,k)
      vys(nx2,k)=vys(2,k)
      vzs(nx2,k)=vzs(2,k)
   70 continue
c calculate total ion density and velocities
      do 80 i=1,nx2
      do 81 k=1,nsp
   81 den(i)=den(i)+dns(i,k)*frac(k)
      if (den(i).lt.0.05) den(i)=0.05
   80 continue
      do 90 k=1,nsp
      do 90 i=1,nx2
      vix(i)=vix(i)+dns(i,k)*frac(k)*vxs(i,k)/den(i)
      viy(i)=viy(i)+dns(i,k)*frac(k)*vys(i,k)/den(i)
      viz(i)=viz(i)+dns(i,k)*frac(k)*vzs(i,k)/den(i)
   90 continue
c  can smooth source terms here if desired
c      call smooth(den,nx2)
c      call smooth(vix,nx2)
c      call smooth(viy,nx2)
c      call smooth(viz,nx2)
      return
      end
      subroutine parmov
c*************
      parameter (ncc=258, nbb=12800, nss=2)
      common pbuf(4,nbb)
      common /fields/ ex(ncc), ey(ncc), ez(ncc), by(ncc), bz(ncc),
     1 ay(ncc), az(ncc), foy(ncc), foz(ncc)
      common /ions/ den(ncc),vix(ncc),viy(ncc),viz(ncc),dns(ncc,nss),
     1 vxs(ncc,nss), vys(ncc,nss), vzs(ncc,nss)
      common /etrons/ vey(ncc), vez(ncc), te(ncc), pe(ncc)
      dimension x(1), vx(1), vy(1), vz(1)
      equivalence (x(1),pbuf(1,1)),(vx(1),pbuf(2,1)),(vy(1),pbuf(3,1)),
     1 (vz(1),pbuf(4,1))
      common /params/ firstp, nx, nx1, nx2, hx, hxi, xmax, dt,
     1 dtwci,vbspec(nss),dnspec(nss),btspec(nss),dfac(nss),frac(nss),
     2 bxc, byc, l3, l4, nb, nc, ns, bete, anspec(nss), nsp, bzc, it,
     3 t, nspec(nss), te0, wspec(nss), nskip, vplim, ntimes,
     4 npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist(501),ehist(501),
     5 vxhist(501,nss), tphist(501,nss), txhist(501,nss), ldec, nplth,
     6 resis, theta, wpiwci, vb0(nss), tx0(nss), lastp
c************
      l4=0
      do 30 kk=1,nsp
      isp=kk
      l3=1+l4
      l4=4*nspec(kk)+l4
      dta=dt/wspec(isp)
      dta2=.5*dta
      do 30 l=l3,l4,4
      rx=x(l)*hxi+1.50000000001
      i=rx
      fx=rx-i
      fxc=1.-fx
      i1=i+1
      exa=ex(i)*fxc+ex(i1)*fx
      eya=ey(i)*fxc+ey(i1)*fx+foy(i)*fxc+foy(i1)*fx
      eza=ez(i)*fxc+ez(i1)*fx+foz(i)*fxc+foz(i1)*fx
      bya=by(i)*fxc+by(i1)*fx
      bza=bz(i)*fxc+bz(i1)*fx
      f=1.-dta*dta2*(bxc**2+bya**2+bza**2)
      g=dta2*(vx(l)*bxc+vy(l)*bya+vz(l)*bza)
      vxa=vx(l)+dta2*exa
      vya=vy(l)+dta2*eya
      vza=vz(l)+dta2*eza
      vx(l)=f*vx(l)+dta*(exa+vya*bza-vza*bya+g*bxc)
      vy(l)=f*vy(l)+dta*(eya+vza*bxc-vxa*bza+g*bya)
      vz(l)=f*vz(l)+dta*(eza+vxa*bya-vya*bxc+g*bza)
      dx=vx(l)*dt
      x(l)=x(l)+dx
c check to see particle is still in simulation region
      if (x(l).lt.xmax) go to 10
      x(l)=x(l)-xmax
   10 if (x(l).gt.0.) go to 20
      x(l)=x(l)+xmax
c push extra half time step to collect density and current
   20 rx=x(l)*hxi+1.5000000000001
      i=rx
      fx=rx-i
      fxc=1.-fx
      i1=i+1
      exa=ex(i)*fxc+ex(i1)*fx
      eya=ey(i)*fxc+ey(i1)*fx+foy(i)*fxc+foy(i1)*fx
      eza=ez(i)*fxc+ez(i1)*fx+foz(i)*fxc+foz(i1)*fx
      bya=by(i)*fxc+by(i1)*fx
      bza=bz(i)*fxc+bz(i1)*fx
      vxa=vx(l)+dta2*(exa+vy(l)*bza-vz(l)*bya)
      vya=vy(l)+dta2*(eya+vz(l)*bxc-vx(l)*bza)
      vza=vz(l)+dta2*(eza+vx(l)*bya-vy(l)*bxc)
      vxs(i,isp)=vxs(i,isp)+vxa*fxc
      vxs(i1,isp)=vxs(i1,isp)+vxa*fx
      vys(i,isp)=vys(i,isp)+vya*fxc
      vys(i1,isp)=vys(i1,isp)+vya*fx
      vzs(i,isp)=vzs(i,isp)+vza*fxc
      vzs(i1,isp)=vzs(i1,isp)+vza*fx
      dns(i,isp)=dns(i,isp)+fxc
      dns(i1,isp)=dns(i1,isp)+fx
   30 continue
      return
      end
      subroutine field
c*************
      parameter (ncc=258, nbb=12800, nss=2)
      common pbuf(4,nbb)
      common /fields/ ex(ncc), ey(ncc), ez(ncc), by(ncc), bz(ncc),
     1 ay(ncc), az(ncc), foy(ncc), foz(ncc)
      common /ions/ den(ncc),vix(ncc),viy(ncc),viz(ncc),dns(ncc,nss),
     1 vxs(ncc,nss), vys(ncc,nss), vzs(ncc,nss)
      common /etrons/ vey(ncc), vez(ncc), te(ncc), pe(ncc)
      dimension x(1), vx(1), vy(1), vz(1)
      equivalence (x(1),pbuf(1,1)),(vx(1),pbuf(2,1)),(vy(1),pbuf(3,1)),
     1 (vz(1),pbuf(4,1))
      common /params/ firstp, nx, nx1, nx2, hx, hxi, xmax, dt,
     1 dtwci,vbspec(nss),dnspec(nss),btspec(nss),dfac(nss),frac(nss),
     2 bxc, byc, l3, l4, nb, nc, ns, bete, anspec(nss), nsp, bzc, it,
     3 t, nspec(nss), te0, wspec(nss), nskip, vplim, ntimes,
     4 npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist(501),ehist(501),
     5 vxhist(501,nss), tphist(501,nss), txhist(501,nss), ldec, nplth,
     6 resis, theta, wpiwci, vb0(nss), tx0(nss), lastp
c************
      dimension vex(1)
      equivalence (vex(1),vix(1))
      dimension e11(ncc), e12(ncc), e21(ncc), e22(ncc), f1(ncc), f2(ncc)
     1 , g11(ncc), g12(ncc), g21(ncc), g22(ncc)
      data gam /.666667/
      hx2=.5*hx
      hxi2=.5*hxi
      dtx=dt*hxi2
      hxs=hx*hx
      dxt=hxs/dt
      ec11=1.
      ec12=0.
      ec21=0.
      ec22=1.
      fc1=0.
      fc2=0.
      gc11=0.
      gc12=0.
      gc21=0.
      gc22=0.
c  set up A B C D arrays (eqns. 49-57)
      do 20 i=2,nx1
      df=den(i)*resis/bxc
      df1=den(i)/(1.+df*df)
      df2=hx2*vix(i)*df1/bxc
      df3=hxs*df1
      df4=1./(dt*bxc)
      a12=-df2
      a21=-a12
      a11=1.-df*a12
      a22=a11
      b12=df3*df4
      b21=-b12
      b11=-2.-df*b12
      b22=b11
      c12=-a12
      c21=-c12
      c11=1.-df*c12
      c22=c11
      viyy=viy(i)-vix(i)*byc/bxc
      vizz=viz(i)-vix(i)*bzc/bxc
      d1=-df3*(viyy+df*vizz-df4*(az(i)-df*ay(i)))
      d2=-df3*(vizz-df*viyy+df4*(ay(i)+df*az(i)))
      if (i.gt.2) go to 10
c  solve for E(2), F(2), G(2) (eqns. 67-69)
      ddi=1./(b11*b22-b12*b21)
      e11(2)=-ddi*(b22*c11-b12*c21)
      e12(2)=-ddi*(b22*c12-b12*c22)
      e21(2)=-ddi*(-b21*c11+b11*c21)
      e22(2)=-ddi*(-b21*c12+b11*c22)
      f1(2)=ddi*(b22*d1-b12*d2)
      f2(2)=ddi*(-b21*d1+b11*d2)
      g11(2)=-ddi*(b22*a11-b12*a21)
      g12(2)=-ddi*(b22*a12-b12*a22)
      g21(2)=-ddi*(-b21*a11+b11*a21)
      g22(2)=-ddi*(-b21*a12+b11*a22)
      go to 20
c solve for E(I), F(I), G(I) (60-62); make EC, FC, GC arrays (74-76)
   10 h11=a11*e11(i-1)+a12*e21(i-1)+b11
      h12=a11*e12(i-1)+a12*e22(i-1)+b12
      h21=a21*e11(i-1)+a22*e21(i-1)+b21
      h22=a21*e12(i-1)+a22*e22(i-1)+b22
      hdi=1./(h11*h22-h12*h21)
      e11(i)=-hdi*(h22*c11-h12*c21)
      e12(i)=-hdi*(h22*c12-h12*c22)
      e21(i)=-hdi*(-h21*c11+h11*c21)
      e22(i)=-hdi*(-h21*c12+h11*c22)
      fd1=d1-a11*f1(i-1)-a12*f2(i-1)
      fd2=d2-a21*f1(i-1)-a22*f2(i-1)
      f1(i)=hdi*(h22*fd1-h12*fd2)
      f2(i)=hdi*(-h21*fd1+h11*fd2)
      gd11=a11*g11(i-1)+a12*g21(i-1)
      gd12=a11*g12(i-1)+a12*g22(i-1)
      gd21=a21*g11(i-1)+a22*g21(i-1)
      gd22=a21*g12(i-1)+a22*g22(i-1)
      g11(i)=-hdi*(h22*gd11-h12*gd21)
      g12(i)=-hdi*(h22*gd12-h12*gd22)
      g21(i)=-hdi*(-h21*gd11+h11*gd21)
      g22(i)=-hdi*(-h21*gd12+h11*gd22)
      fc1=fc1+ec11*f1(i-1)+ec12*f2(i-1)
      fc2=fc2+ec21*f1(i-1)+ec22*f2(i-1)
      gc11=gc11+ec11*g11(i-1)+ec12*g21(i-1)
      gc12=gc12+ec11*g12(i-1)+ec12*g22(i-1)
      gc21=gc21+ec21*g11(i-1)+ec22*g21(i-1)
      gc22=gc22+ec21*g12(i-1)+ec22*g22(i-1)
      tc11=ec11*e11(i-1)+ec12*e21(i-1)
      tc12=ec11*e12(i-1)+ec12*e22(i-1)
      tc21=ec21*e11(i-1)+ec22*e21(i-1)
      tc22=ec21*e12(i-1)+ec22*e22(i-1)
      ec11=tc11
      ec12=tc12
      ec21=tc21
      ec22=tc22
   20 continue
c solve for X(NX1) (77-78)
      en11=e11(nx)+g11(nx)
      en12=e12(nx)+g12(nx)
      en21=e21(nx)+g21(nx)
      en22=e22(nx)+g22(nx)
      ec11=ec11+gc11
      ec12=ec12+gc12
      ec21=ec21+gc21
      ec22=ec22+gc22
      h11=a11*en11+a12*en21+b11+c11*ec11+c12*ec21
      h12=a11*en12+a12*en22+b12+c11*ec12+c12*ec22
      h21=a21*en11+a22*en21+b21+c21*ec11+c22*ec21
      h22=a21*en12+a22*en22+b22+c21*ec12+c22*ec22
      hdi=1./(h11*h22-h12*h21)
      p1=d1-a11*f1(nx)-a12*f2(nx)-c11*fc1-c12*fc2
      p2=d2-a21*f1(nx)-a22*f2(nx)-c21*fc1-c22*fc2
      ey(nx1)=ay(nx1)
      ez(nx1)=az(nx1)
      ay(nx1)=hdi*(h22*p1-h12*p2)
      az(nx1)=hdi*(-h21*p1+h11*p2)
c get all X(I) (58)
      do 30 ii=2,nx
      i=nx2-ii
      ey(i)=ay(i)
      ez(i)=az(i)
      ay(i)=e11(i)*ay(i+1)+e12(i)*az(i+1)+f1(i)+g11(i)*ay(nx1)+g12(i)*az
     1 (nx1)
      az(i)=e21(i)*ay(i+1)+e22(i)*az(i+1)+f2(i)+g21(i)*ay(nx1)+g22(i)*az
     1 (nx1)
   30 continue
      ey(nx2)=ay(nx2)
      ez(nx2)=az(nx2)
      ay(nx2)=ay(2)
      az(nx2)=az(2)
      ey(1)=ay(1)
      ez(1)=az(1)
      ay(1)=ay(nx1)
      az(1)=az(nx1)
c  get Ey Ez (35-36); By Bz (31-32)
      do 40 i=2,nx1
      ey(i)=(ey(i)-ay(i))/dt
      ez(i)=(ez(i)-az(i))/dt
      by(i)=(az(i-1)-az(i+1))*hxi2+byc
   40 bz(i)=(ay(i+1)-ay(i-1))*hxi2+bzc
      ey(1)=ey(nx1)
      ez(1)=ez(nx1)
      ey(nx2)=ey(2)
      ez(nx2)=ez(2)
      by(nx2)=by(2)
      bz(nx2)=bz(2)
      by(1)=by(nx1)
      bz(1)=bz(nx1)
c calculate del**2 A, electron velocities, drag force
      do 50 i=2,nx1
      ajy=-(ay(i+1)+ay(i-1)-2.*ay(i))/hxs
      ajz=-(az(i+1)+az(i-1)-2.*az(i))/hxs
      vey(i)=-ajy/(den(i))+viy(i)
      vez(i)=-ajz/(den(i))+viz(i)
      foy(i)=-resis*ajy
      foz(i)=-resis*ajz
   50 continue
c calculate electron temperature and pressure
      do 60 i=2,nx1
      te(i)=te0*(den(i)**gam)
      if (iemod.eq.0) te(i)=te0
   60 pe(i)=te(i)*den(i)
      vey(nx2)=vey(2)
      vez(nx2)=vez(2)
      te(nx2)=te(2)
      pe(nx2)=pe(2)
      vey(1)=vey(nx1)
      vez(1)=vez(nx1)
      te(1)=te(nx1)
      pe(1)=pe(nx1)
c calculate Ex
      do 70 i=2,nx1
      ex(i)=vez(i)*by(i)-vey(i)*bz(i)
      ex(i)=ex(i)-hxi2*(pe(i+1)-pe(i-1))/den(i)
   70 continue
      ex(1)=ex(nx1)
      ex(nx2)=ex(2)
      return
      end
      subroutine diagnos
c*************
      parameter (ncc=258, nbb=12800, nss=2)
      common pbuf(4,nbb)
      common /fields/ ex(ncc), ey(ncc), ez(ncc), by(ncc), bz(ncc),
     1 ay(ncc), az(ncc), foy(ncc), foz(ncc)
      common /ions/ den(ncc),vix(ncc),viy(ncc),viz(ncc),dns(ncc,nss),
     1 vxs(ncc,nss), vys(ncc,nss), vzs(ncc,nss)
      common /etrons/ vey(ncc), vez(ncc), te(ncc), pe(ncc)
      dimension x(1), vx(1), vy(1), vz(1)
      equivalence (x(1),pbuf(1,1)),(vx(1),pbuf(2,1)),(vy(1),pbuf(3,1)),
     1 (vz(1),pbuf(4,1))
      common /params/ firstp, nx, nx1, nx2, hx, hxi, xmax, dt,
     1 dtwci,vbspec(nss),dnspec(nss),btspec(nss),dfac(nss),frac(nss),
     2 bxc, byc, l3, l4, nb, nc, ns, bete, anspec(nss), nsp, bzc, it,
     3 t, nspec(nss), te0, wspec(nss), nskip, vplim, ntimes,
     4 npltp,npltf,nwrth,nwrtf,cth,sth,iemod,bhist(501),ehist(501),
     5 vxhist(501,nss), tphist(501,nss), txhist(501,nss), ldec, nplth,
     6 resis, theta, wpiwci, vb0(nss), tx0(nss), lastp
c************
      character  blabl*16
      character labp*8,labb*40
      dimension xsc(2), vsc(2), tc(2)
      dimension vx1(nss),vx2(nss),vy1(nss),vy2(nss),vz1(nss),vz2(nss),
     1 tx1(nss),ty1(nss),tz1(nss),fna(nss),labb(nss),wpar(nss)
      dimension temp1(ncc), temp2(ncc), temp3(ncc)
      data labb /'    beam','    ions'/
      entry diag1
      idsw=1
      go to 10
      entry diag2
      idsw=2
   10 write (blabl,240) it,t
      call lblbot (blabl,16)
      if (idsw.eq.2) go to 80
c   particle diagnostics in this part of diagnos
      if (nwrth.le.0) go to 50
      if (mod(it,nwrth).ne.0) go to 50
c   collect particle moments
      do 20 j=1,nsp
      fna(j)=1.e-20
      tx1(j)=0.
      vx1(j)=0.
      ty1(j)=0.
      vy1(j)=0.
      tz1(j)=0.
      vz1(j)=0.
      vx2(j)=0.
      vy2(j)=0.
      vz2(j)=0.
   20 continue
      l4=0
      do 40 kk=1,nsp
      isp=kk
      l3=1+l4
      l4=4*nspec(kk)+l4
      do 40 l=l3,l4,4
      fna(isp)=fna(isp)+1.
      vxa=vx(l)*cth+vz(l)*sth
      vya=vy(l)
      vza=vz(l)*cth-vx(l)*sth
      vx1(isp)=vx1(isp)+vxa
      vx2(isp)=vx2(isp)+vxa*vxa
      vy1(isp)=vy1(isp)+vya
      vy2(isp)=vy2(isp)+vya*vya
      vz1(isp)=vz1(isp)+vza
      vz2(isp)=vz2(isp)+vza*vza
   40 continue
   50 if (npltp.le.0) go to 70
      if (mod(it,npltp).ne.0) go to 70
c   particle plots---each species on separate page
      l4=0
      do 30 kk=1,nsp
      isp=kk
      l3=1+l4
      l4=4*nspec(kk)+l4
      vsc(1)=-vplim
      vsc(2)=vplim
      lskip=nskip*4
      xsc(1)=0.
      xsc(2)=xmax
      labp=labb(isp)
      call lplot (-1,-4,2,xsc,vsc,1,-1,labp,8,1hx,1,2hvx,2)
      call lplot (-1,-5,2,xsc,vsc,1,-1,labp,8,1hx,1,2hvy,2)
      call lplot (-1,-6,2,xsc,vsc,1,-1,labp,8,1hx,1,2hvz,2)
      npts=1+(l4-l3)/lskip
      call pplot (1,4,npts,x(l3),vx(l3),lskip)
      call pplot (1,5,npts,x(l3),vy(l3),lskip)
      call pplot (1,6,npts,x(l3),vz(l3),lskip)
   30 continue
   70 continue
      return
   80 continue
c go through this part of diagnos after sources/fields known
      if (nwrtf.le.0) go to 90
      if (mod(it,nwrtf).ne.0) go to 90
c    write out by bz  into files bys bzs
      write (3) (by(ij),ij=1,nx)
      write (4) (bz(ij),ij=1,nx)
   90 continue
      if (npltf.le.0) go to 150
      if (mod(it,npltf).ne.0) go to 150
c   plot out densities and b fields
      xsc(1)=0.
      xsc(2)=xmax/float(nx1)
      call lplot (-1,-4,nx2,xsc,den,-1,1,1h ,1,1hx,1,3hden,3)
      call lplot (-1,-5,nx2,xsc,dns(1,1),-1,1,1h ,1,1hx,1,3hdnb,3)
      call lplot (-1,-6,nx2,xsc,dns(1,2),-1,1,1h ,1,1hx,1,3hdni,3)
      do 100 i=1,nx2
      temp1(i)=by(i)
  100 temp2(i)=bz(i)
      call smooth (temp1,nx2)
      call smooth (temp2,nx2)
      do 110 i=1,nx2
  110 temp3(i)=angle(temp1(i),temp2(i))
      do 120 i=1,nx2
      temp1(i)=by(i)*wpiwci
  120 temp2(i)=bz(i)*wpiwci
      call lplot (-1,-4,nx2,xsc,temp1,-1,1,1h ,1,1hx,1,2hby,2)
      call lplot (-1,-5,nx2,xsc,temp2,-1,1,1h ,1,1hx,1,2hbz,2)
      call lplot (-1,-6,nx2,xsc,temp3,-1,1,1h ,1,1hx,1,3hphi,3)
c  call fourier transform routine to get modes
      call rft2 (temp1,nx,1)
      nx4=nx/4
      facn=4./(float(nx)**2)
      do 130 i=1,nx4
  130 temp1(i)=facn*(temp1(2*i+1)**2+temp1(2*i+2)**2)
      do 140 i=1,nx2
      temp2(i)=by(i)*wpiwci
  140 temp3(i)=wpiwci*sqrt(by(i)**2+bz(i)**2)
      call lplot (-1,-4,nx2,xsc,temp2,-1,1,1h ,1,1hx,1,2hby,2)
      call lplot (-1,-5,nx2,xsc,temp3,-1,1,1h ,1,1hx,1,1hb,1)
      xsc(1)=1.
      xsc(2)=1.
      call hplot (-1,-6,nx4,xsc,temp1,-1,2,1h ,1,1hk,1,6hbyk**2,6)
  150 if (nwrth.le.0) go to 210
      if (mod(it,nwrth).ne.0) go to 210
c       compute and write particle moments
      do 160 j=1,nsp
      vx1(j)=vx1(j)/fna(j)
      vx2(j)=vx2(j)/fna(j)
      tx1(j)=(vx2(j)-vx1(j)**2)*wspec(j)
      vy1(j)=vy1(j)/fna(j)
      vy2(j)=vy2(j)/fna(j)
      ty1(j)=(vy2(j)-vy1(j)**2)*wspec(j)
      vz1(j)=vz1(j)/fna(j)
      vz2(j)=vz2(j)/fna(j)
  160 tz1(j)=(vz2(j)-vz1(j)**2)*wspec(j)
      do 170 j=1,nsp
      vx1(j)=vx1(j)/vb0(j)
      tx1(j)=tx1(j)/tx0(j)
      vy1(j)=vy1(j)/vb0(j)
      ty1(j)=ty1(j)/tx0(j)
      vz1(j)=vz1(j)/vb0(j)
      tz1(j)=tz1(j)/tx0(j)
  170 continue
      write (9,250) it
      write (9,260)
      write (9,270) (vx1(j),vy1(j),vz1(j),tx1(j),ty1(j),tz1(j),j=1,nsp)
c compute and write out energies
      ldec=ldec+1
      if (ldec.ge.501) ldec=501
      sumb=0.
      sume=0.
      do 180 i=2,nx1
      sumb=sumb+by(i)**2+bz(i)**2
      sume=sume+ex(i)**2+ey(i)**2+ez(i)**2
  180 continue
      wbfl=.5*sumb/nx
      wefl=.5*sume/nx
      wcfl=.5*(bxc**2)
      wfl=wbfl+wefl+wcfl
      wptot=0.
      do 190 k=1,nsp
      wpar(k)=.5*(vx2(k)+vy2(k)+vz2(k))*wspec(k)*frac(k)
      wptot=wptot+wpar(k)
  190 continue
      wtot=wptot+wfl
      write (9,280) wbfl,wefl,wcfl,wfl
      write (9,290) wptot,(wpar(k),k=1,2)
      write (9,300) wtot
c  save field energies and ion moments for plots
      bhist(ldec)=2.*wbfl*(wpiwci**2)
      ehist(ldec)=2.*wefl*(wpiwci**2)
      do 200 k=1,nsp
      vxhist(ldec,k)=vx1(k)
      txhist(ldec,k)=tx1(k)
  200 tphist(ldec,k)=.5*(ty1(k)+tz1(k))
  210 if (nplth.le.0) go to 230
      if (mod(it,nplth).ne.0) go to 230
c  plot out field energies and ion moments
      if (it.eq.0) go to 230
      tc(1)=0.
      tc(2)=dtwci*nwrth
      call lplot (-2,-4,ldec,tc,bhist,-1,1,1h ,1,1ht,1,4hb**2,4)
      call lplot (-3,-4,ldec,tc,bhist,-1,2,1h ,1,1ht,1,4hb**2,4)
      call lplot (-2,-5,ldec,tc,ehist,-1,1,1h ,1,1ht,1,4he**2,4)
      call lplot (-3,-5,ldec,tc,ehist,-1,2,1h ,1,1ht,1,4he**2,4)
      do 220 k=1,nsp
      kl=3-mod(k,2)
      call lplot (-kl,-4,ldec,tc,vxhist(1,k),-1,1,1h ,1,1ht,1,2hvx,2)
      call lplot (-kl,-5,ldec,tc,txhist(1,k),-1,1,1h ,1,1ht,1,2htx,2)
      call lplot (-kl,-6,ldec,tc,tphist(1,k),-1,1,1h ,1,1ht,1,2htp,2)
  220 continue
  230 return
c
  240 format ('it=',i4,' t=',f6.2)
  250 format ('     it=',i8)
  260 format ('     vx   ','     vy   ','     vz   ','     tx   ',
     1 '     ty   ','     tz   ')
  270 format (6f10.5)
  280 format ('  wbfl=',e13.5,'  wefl=',e13.5,'  wcfl=',e13.5,'   wfl=',
     1 f13.5)
  290 format (' wptot=',e13.5,' wp(1)=',e13.5,' wp(2)=',e13.5)
  300 format ('  wtot=',e13.5)
      end
      subroutine smooth (a,nx2)
c   simple smoothing routine for arrays
      dimension a(1)
      nx1=nx2-1
      nx=nx1-1
      aold=a(1)
      do 10 j=2,nx1
      asave=a(j)
      a(j)=.25*aold+.5*asave+.25*a(j+1)
   10 aold=asave
      a(nx2)=a(2)
      a(1)=a(nx1)
      return
      end
      subroutine endrun (h)
      character h*8
c   writes out message if run terminates
      write (6,10) h
      write (9,10) h
      call exit
      call endplt
c
   10 format (///,' run terminated-----',a8)
      end
      function angle (ay,az)
c  calculates phase angle...tan phi=az/ay
      data pi /3.14159265/
      if (abs(ay).lt.1.e-15) ay=sign(1.e-15,ay)
      angle=atan(az/ay)+pi*(1.-.5*ay/abs(ay))
      return
      end
      subroutine rft2 (data1,nr,kr)
c does transform of array data1, puts transforms in same array
c nr is no. of points--works reliably only if power of 2
c original array must have dimension .ge. nr+2
c kr is increment between points
c  f= sum over m  A(m)cos (2 pi m x) - B(m) sin (2 pi m x)
c A(m) B(m) multiplied by nr/2; except m=0 and m=nr/2 where factor is nr.
c coefficients stored in data1 as: A(0),B(0),A(1),B(1),...A(nr),B(nr)
      dimension data1(1)
      call fft2 (data1(1),data1(kr+1),nr/2,-(kr+kr))
      call rtran2 (data1,nr,kr,1)
      return
      end
      subroutine rfi2 (data1,nr,kr)
c  inverse fourier transform
      dimension data1(1)
      call rtran2 (data1,nr,kr,-1)
      mr=nr*kr
      fni=2./nr
      do 10 i=1,mr,kr
   10 data1(i)=data1(i)*fni
      call fft2 (data1(1),data1(kr+1),nr/2,(kr+kr))
      return
      end
      subroutine rtran2 (data1,nr,kr,ktran)
      dimension data1(1)
      ks=2*kr
      n=nr/2
      nmax=n*ks+2
      kmax=nmax/2
      theta=1.5707963268/n
      dc=2.*sin(theta)**2
      ds=sin(2.*theta)
      ws=0.
      if (ktran) 10,10,20
   10 wc=-1.0
      ds=-ds
      go to 30
   20 wc=1.0
      data1(nmax-1)=data1(1)
      data1(nmax-1+kr)=data1(kr+1)
   30 do 40 k=1,kmax,ks
      nk=nmax-k
      sumr=.5*(data1(k)+data1(nk))
      sumi=.5*(data1(k+kr)+data1(nk+kr))
      difr=.5*(data1(k)-data1(nk))
      difr=-difr
      difi=.5*(data1(k+kr)-data1(nk+kr))
      tr=wc*sumi+ws*difr
      ti=ws*sumi-wc*difr
      ti=-ti
      data1(k)=sumr+tr
      data1(k+kr)=difi+ti
      data1(nk)=sumr-tr
      data1(nk+kr)=-difi+ti
      wca=wc-dc*wc-ds*ws
      ws=ws+ds*wc-dc*ws
      wc=wca
   40 continue
      return
      end
      subroutine fft2 (data,datai,n,inc)
      dimension data(1), datai(1)
      ktran=isign(-1,inc)
      ks=iabs(inc)
      ip0=ks
      ip3=ip0*n
      irev=1
      do 50 i=1,ip3,ip0
      if (i-irev) 10,20,20
   10 tempr=data(i)
      tempi=datai(i)
      data(i)=data(irev)
      datai(i)=datai(irev)
      data(irev)=tempr
      datai(irev)=tempi
   20 ibit=ip3/2
   30 if (irev-ibit) 50,50,40
   40 irev=irev-ibit
      ibit=ibit/2
      if (ibit-ip0) 50,30,30
   50 irev=irev+ibit
      ip1=ip0
      theta=float(ktran)*3.1415926536
   60 if (ip1-ip3) 70,100,100
   70 ip2=ip1*2
      sinth=sin(.5*theta)
      wstpr=-2.*sinth*sinth
      wstpi=sin(theta)
      wr=1.
      wi=0.
      do 90 i1=1,ip1,ip0
      do 80 i3=i1,ip3,ip2
      j0=i3
      j1=j0+ip1
      tempr=wr*data(j1)-wi*datai(j1)
      tempi=wr*datai(j1)+wi*data(j1)
      data(j1)=data(j0)-tempr
      datai(j1)=datai(j0)-tempi
      data(j0)=data(j0)+tempr
   80 datai(j0)=datai(j0)+tempi
      tempr=wr
      wr=wr*wstpr-wi*wstpi+wr
   90 wi=wi*wstpr+tempr*wstpi+wi
      ip1=ip2
      theta=.5*theta
      go to 60
  100 return
      end
c====================== end of hybrid1.f ===============================

c=================== beginning of libhyb1.f ===============================
c***************************
c  p5.f
c
c  fortran source for plotting package
c
c***************************
c
      subroutine begplt(ititle)
      character*(*) ititle
      nchar=len(ititle)
c     call gplot(1hu,ititle,nchar)
c     call gwind2(-1.0,1024.0,-1.0,780.0)
      call plots
      return
      end
      subroutine endplt
c     call gdone
      call plot(0.,0.,999)
      return
      end
      subroutine movabs(ix,iy)
      common /pltpos/ixpos,iypos
c      call gmova2(float(ix),float(iy))
      call plot(ix*0.0336,iy*0.0336,3)
      ixpos=ix
      iypos=iy
      return
      end
      subroutine drwabs(ix,iy)
      common /pltpos/ixpos,iypos
c     call glina2(float(ix),float(iy))
      call plot(ix*0.0336,iy*0.0336,2)
      ixpos=ix
      iypos=iy
      return
      end
      subroutine drv (ix1,iy1,ix2,iy2)
      common /pltpos/kbeamx,kbeamy
      if (ix1.ne.kbeamx) go to 10
      if (iy1.eq.kbeamy) go to 30
   10 if (ix2.ne.kbeamx) go to 20
      if (iy2.ne.kbeamy) go to 20
      call drwabs (ix1,iy1)
      return
   20 call movabs (ix1,iy1)
   30 call drwabs (ix2,iy2)
      return
      end
      subroutine pntabs(ix,iy)
      common /pltpos/ixpos,iypos
c     call gmrka2(float(ix),float(iy))
      call plot(ix*0.0336,iy*0.0336,3)
      call plot(ix*0.0336+0.1,iy*0.0336,2)
c     print*,ix,iy
c     read*,iii
c     print*, 'pntabs is called'
      ixpos=ix
      iypos=iy
      return
      end
      subroutine adv (n)
      do 10 i=1,n
c  10 call gpage
   10 call chart
      return
      end
      subroutine seeloc(ix,iy)
      common /pltpos/ixpos,iypos
      call where(x,y,f)
      ix=x*29.762
      iy=y*29.762
      return
      end
      function alog11 (arg)
      if (arg.lt.1.e-20) go to 10
      alog11=alog(arg)/alog(10.)
      return
   10 alog11=-20.
      return
      end
      subroutine dlch(ix,iy,kchr,string,insize)
      character*(*) string
c      print*,'in dlch','|',string,'|',len(string)
      if(len(string).eq.0) return
      call symbol(ix*0.0336,iy*0.0336,0.3,
     $     string,0.,len(string))
      return
      entry dlcv(ix,iy,kchr,string,insize)
C      print*,'in dlch',string,len(string)
      if(len(string).eq.0) return
      call symbol(ix*0.0336,iy*0.0336,0.3,
     $     string,90.,len(string))
      return
      end
      subroutine dlglg (nx)
      common /cje07/ ixl, ixr, iyt, iyb, xl, xr, yt, yb
      dimension xy(4), ixy(4)
      equivalence (xy,xl), (ixy,ixl)
      dimension alg(8)
      character mess*20, head1(2)*10, head2(2)*20
      data (alg(k),k=1,8) /.301029996,.477121255,.602059991,
     1 .698970004,.778151250,.845098040,.903089987, .954242509/
      data mess /'decades exceed 25'/
      data head1 /'  xl=xr.  ','  yt=yb.  '/
      data head2 /'  no of x ','  no of y '/
c      print *,"dlglg"
      itype=2
      iex=2
   10 i1=3*itype-2
      i2=itype+1
      x1=xy(i1)
      x2=xy(i2)
      if (x1.ne.x2) go to 20
      x2=x2+.01
   20 xmin=amin1(x1,x2)
      xmax=amax1(x1,x2)
      xmin=amin1(aint(xmin),sign(aint(abs(xmin)+.999),xmin))
      xmax=amax1(aint(xmax),sign(aint(abs(xmax)+.999),xmax))
      x1=xmin
      x2=xmax
      ny=abs(x1-x2)
      if (ny.le.25) go to 30
      call dlch (500,500,10,head2(itype),1)
      call dlch (500,520,17,mess,1)
      return
   30 if (ny.ne.0) go to 40
      ytt=x1+1.
      if (x2.lt.x1) ytt=x1-1.
      ny=1
      x1=ytt
   40 if (xy(i2).lt.xy(i1)) goto 60
   50 irev=1
      xy(i1)=x1
      xy(i2)=x2
      go to 70
   60 irev=2
      xy(i1)=x2
      xy(i2)=x1
   70 isl=(ixy(i2)-ixy(i1))/ny
      iyc=ixy(i1)
      do 150 i=1,ny
      do 120 k=1,8
      go to (80,90), irev
   80 icy=alg(k)*isl+iyc
      go to (100,110), itype
   90 icy=(1.-alg(k))*isl+iyc
      go to (100,110), itype
  100 call gya (iyt-15,iyt,icy)
      call gya (iyb,iyb+15,icy)
      go to 120
  110 call gxa (ixl,ixl+15,icy)
      call gxa (ixr-15,ixr,icy)
  120 continue
      iyc=ixy(i1)+(i*(ixy(i2)-ixy(i1)))/ny
      go to (130,140), itype
  130 call gya (iyt,iyt-25,iyc)
      call gya (iyb+25,iyb,iyc)
      go to 150
  140 call gxa (ixl,ixl+25,iyc)
      call gxa (ixr-25,ixr,iyc)
  150 continue
      go to (160,170), iex
  160 return
      entry dlgln (nx)
      call dlnln (0,nx)
  170 itype=1
      iex=1
      go to 10
      entry dlnlg (nx)
      call dlnln (nx,0)
      itype=2
      iex=1
      go to 10
      end
c----------------------------------------
c
      subroutine dlnln (nx,ny)
      common /cje07/ ixl, ixr, iyt, iyb, xl, xr, yt, yb
c      print*,"dlnln"
      call gya (iyt,iyb,ixl)
      call gxa (ixl,ixr,iyb)
      if (nx) 60,30,10
   10 nxs=min(nx,128)
      dx=float(ixr-ixl)/nxs
      nxs=nxs-1
      call gya (iyt,iyb,ixr)
      iiyb=iyb+20
      iiyt=iyt-20
      do 20 i=1,nxs
      ixs=ixl+i*dx
      call gya (iyb,iiyb,ixs)
   20 call gya (iyt,iiyt,ixs)
   30 if (ny) 60,60,40
   40 nys=min(ny,128)
      dy=float(iyt-iyb)/nys
      iixr=ixr-20
      iixl=ixl+20
      nys=nys-1
      call gxa (ixl,ixr,iyt)
      do 50 i=1,nys
      iys=iyb+i*dy
      call gxa (ixl,iixl,iys)
   50 call gxa (ixr,iixr,iys)
   60 return
      end
c----------------------------------------
c
      subroutine sblin (nnx,nk)
      common /cje07/ ixl, ixr, iyt, iyb, xl, xr, yt, yb
      character out*8
c	print*,"sblin"
      nc=5
      t=amax1(abs(xl),abs(xr))
      if (abs(t).le.1.e-15) t=1.e-15
      ks=int(alog11(t)+300.)-300
c
      fact=10.**(-ks)
      xll=xl*fact
      xrr=xr*fact
c
c      encode (5,40,out) xll
      write(out,40) xll
      iyl=iyb-22
      ixcc=ixl-21
      call dlch (ixcc,iyl,nc,out,0)
      if (nnx.le.0) return
      nx=min(10,nnx)
      ixm=ixl
      dx=(xrr-xll)/nx
      ddx=float(ixr-ixl)/nx
      do 10 i=1,nx
      xc=xll+i*dx
      ixm=ixl+i*ddx
      ixmm=ixl+i*ddx-35
c      encode (5,40,out) xc
      write(out,40) xc
      call dlch (ixmm,iyl,nc,out,0)
   10 continue
      if (ks.eq.0) return
      if (ks.gt.-10) go to 20
      i=3
c      encode (5,50,out) ks
      write(out,50) ks
      go to 30
   20 i=2
c      encode (5,60,out) ks
      write(out,60) ks
   30 icen=ixr-85
      call dlch (icen,iyb-41,1,'*',0)
      call dlch (icen,iyb-41,3,' 10',0)
      call seeloc (ixx,iyy)
      call dlch (ixx,iyy+8,i,out,0)
      return
c
   40 format (f5.2)
   50 format (i3)
   60 format (i2)
      end
c----------------------------------------
c
      subroutine sblog
      common /cje07/ ixl, ixr, iyt, iyb, xl, xr, yt, yb
      dimension xy(4), ixy(4)
      equivalence (xy,xl), (ixy,ixl)
      character out*10,ten*10
c      data ten /2h10/
c	print *,"sblog"
      ten='10'
      iy=iyb
      iydel=-23
      iydl=8
      ix=ixl
      ixdel=-16
      ixdl=3
      i1=1
      i2=2
      go to 10
      entry sllog
c sawada for titan
      ten='10'
      ix=ixl
      ixdel=-47
      ixdl=3
      iy=iyb
      iydel=-2
      iydl=8
      i1=4
      i2=3
   10 ixyv=xy(i1)
      nx=amin1(abs(xy(i1)-xy(i2)),25.)
c      encode (5,50,out) ixyv
      write(out,50) ixyv
      ixc=ix+ixdel
      iyc=iy+iydel
      ixx=ixc+ixdl
      iyx=iyc+iydl
      call dlch (ixc,iyc,2,ten,0)
      call dlch (ixx,iyx,4,out,0)
      if (nx.eq.0) return
      idxyv=isign(1,ifix(xy(i2)-xy(i1)))
      do 40 i=1,nx
      ixyv=ixyv+idxyv
c      encode (5,50,out) ixyv
      write(out,50) ixyv
      if (i1.eq.1) go to 20
      iyc=iy+iydel+(i*(ixy(i2)-ixy(i1)))/nx
      iyi=iyc+4
      iyx=iyc+iydl
      go to 30
   20 ixc=ix+ixdel+(i*(ixy(i2)-ixy(i1)))/nx
      ixi=ixc+16
      ixx=ixc+ixdl
   30 call dlch (ixc,iyc,2,ten,0)
      call dlch (ixx,iyx,4,out,0)
   40 continue
      return
c
   50 format (i4)
      end
c----------------------------------------
c
      subroutine sllin (nny,nk)
      common /cje07/ ixl, ixr, iyt, iyb, xl, xr, yt, yb
      character out*8
c      print*,"sllin"
      nc=5
      t=amax1(abs(yb),abs(yt))
      if (abs(t).le.1.e-15) t=1.e-15
      ks=int(alog11(t)+300.)-300
c
      fact=10.**(-ks)
      ytt=yt*fact
      ybt=yb*fact
c
c      encode (5,40,out) ybt
      write(out,40) ybt
      ixt=ixl-55
      if (ixt.lt.0) ixt=0
      iycc=iyb-2
      call dlch (ixt,iycc,nc,out,0)
      if (nny.le.0) return
      ny=min(10,nny)
      iyc=iyb
      dy=(ytt-ybt)/ny
      ddy=float(iyt-iyb)/ny
      do 10 i=1,ny
      yc=ybt+i*dy
      iyc=iyb+i*ddy
      iycc=iyb+i*ddy-6
c      encode (5,40,out) yc
      write(out,40) yc
      call dlch (ixt,iycc,nc,out,0)
   10 continue
      if (ks.eq.0) return
      if (ks.gt.-10) go to 20
      i=3
c      encode (5,50,out) ks
      write(out,50) ks
      go to 30
   20 i=2
c      encode (5,60,out) ks
      write(out,60) ks
   30 call dlch (ixt-15,iycc+17,1,'*',0)
      call dlch (ixt-15,iycc+17,3,' 10',0)
      call seeloc (ixx,iyy)
      call dlch (ixx,iyy+8,i,out,0)
      return
c
   40 format (f5.2)
   50 format (i3)
   60 format (i2)
      end
c----------------------------------------
c
      subroutine maxv (x,ix,n,i,y)
      dimension x(200)
c
      assign 30 to i1
   10 i2=1
   20 y=x(1)
      i=1
      if (i2.eq.0) y=abs(y)
      nn=(n-1)*ix+1
      do 60 ii=1,nn,ix
      z=x(ii)
      if (i2.eq.0) z=abs(z)
      go to i1, (30,40)
   30 if (z-y) 60,60,50
   40 if (z-y) 50,60,60
   50 y=z
      i=(ii-1)/ix+1
   60 continue
      ii=ix*(i-1)+1
      y=x(ii)
      return
      entry minv (x,ix,n,i,y)
c
      assign 40 to i1
      go to 10
      entry maxav (x,ix,n,i,y)
c
      assign 30 to i1
      i2=0
      go to 20
c
      entry minav (x,ix,n,i,y)
      i2=0
      go to 20
      end
c----------------------------------------
c
      subroutine ascl (m,z1,z2,major,minor,kf)
      zmin=z1
      zmax=z2
      if (zmax-zmin) 10,10,20
   10 major=0
      minor=0
      kf=0
      go to 460
   20 if (m) 10,10,30
   30 if (m-20) 40,40,10
   40 fm=m
      if (zmax) 50,120,50
   50 if (zmin) 60,120,60
   60 zbar=zmax/zmin
      if (abs(zbar)-1000.) 80,70,70
   70 zmin=0.
   80 if (abs(zbar)-.001) 90,90,100
   90 zmax=0.
      go to 120
  100 if (abs(zbar-1.)-.000005*fm) 110,110,120
  110 zbar=(zmax+zmin)/2.
      z=.0000026*fm*abs(zbar)
      zmax=zbar+z
      zmin=zbar-z
      go to 130
  120 if (zmax-zmin.eq.fm) go to 130
      zmax=zmax-.000001*abs(zmax)
      zmin=zmin+.000001*abs(zmin)
  130 p=(zmax-zmin)/fm
      iflag=0
      tenk=1.
      k=0
      if (p-1.) 140,150,150
  140 iflag=1
      p=1./p
  150 if (p-10000.) 170,160,160
  160 p=p/10000.
      tenk=tenk*10000.
      k=k+4
      go to 150
  170 if (p-10.) 190,180,180
  180 p=p/10.
      tenk=tenk*10.
      k=k+1
      go to 170
  190 if (iflag) 200,210,200
  200 p=10./p
      tenk=.1/tenk
      k=-k-1
  210 if (p-2.) 220,230,240
  220 p=1.
      go to 260
  230 p=2.
      nm=4
      go to 270
  240 if (p-5.) 230,250,250
  250 p=5.
  260 nm=5
  270 dz=p*tenk
      n1=zmin/dz
      fn=n1
      z=fn*dz
      if (z-zmin) 290,290,280
  280 z=z-dz
      n1=n1-1
  290 zmin=z
      z1=zmin
      n2=zmax/dz
      fn=n2
      z=fn*dz
      if (z-zmax) 300,310,310
  300 n2=n2+1
      z=z+dz
  310 zmax=z
      z2=zmax
      major=n2-n1
      minor=nm*major
      if (k) 320,450,340
  320 if (k+5) 340,330,330
  330 k=-k
      go to 450
  340 if (abs(zmax)-abs(zmin)) 350,350,360
  350 z=abs(zmin)
      go to 370
  360 z=abs(zmax)
  370 z=z/tenk
      j=0
  380 if (z-10.) 400,390,390
  390 z=z/10.
      j=j+1
      go to 380
  400 if (k) 430,410,410
  410 if (j+k-5) 420,420,430
  420 k=0
      go to 450
  430 k=10+j
      if (k-11) 440,450,450
  440 k=11
  450 kf=k
  460 return
      end
c----------------------------------------
c
      subroutine convrt (z,iz,z1,z2,iz1,iz2)
      f=z2-z1
      if (f.ne.0) f=(iz2-iz1)/f
      iz=min(max(iz1+ifix((z-z1)*f),min(iz1,iz2)),max(iz1,iz2))
      return
      end
c----------------------------------------
c
      subroutine dga (ix1,ix2,iy1,iy2,x1,x2,y1,y2)
      common /cje07/ ixl, ixr, iyt, iyb, xl, xr, yt, yb
      ixl=min(max(min(ix1,ix2),0),1023)
      iyb=min(max(min(iy1,iy2),0),1023)
      ixr=min(max(max(ix1,ix2),0),1023)
      iyt=min(max(max(iy1,iy2),0),1023)
      xl=x1
      xr=x2
      yt=y1
      yb=y2
      return
      end
c----------------------------------------
c
c----------------------------------------
c
c----------------------------------------
c
c----------------------------------------
c
c----------------------------------------
c
c----------------------------------------
c
      subroutine frame (ixl,ixr,iyt,iyb)
c  omura    call cfrm
changed 16-may-76 stb
      call drv (ixl,iyt,ixr,iyt)
      call drwabs (ixw,iyb)
      call drwabs (ixl,iyb)
      call drwabs (ixl,iyt)
      return
      end
c----------------------------------------
c
      subroutine gxa (ix1,ix2,iy)
changed 16-may-76 stb
      call drv (ix1,iy,ix2,iy)
      return
      end
c----------------------------------------
c
      subroutine gya (iy1,iy2,ix)
changed 16-may-76
      call drv (ix,iy1,ix,iy2)
      return
      end
c----------------------------------------
c
      subroutine maxm (a,ia,n,m,i,j,b)
      dimension a(ia,m)
      assign 60 to ii
   10 b=a(1,1)
   20 i=1
      j=1
      do 50 k=1,n
      do 50 l=1,m
      go to ii, (60,80,100,110)
   60 s=a(k,l)
   70 if (s-b) 40,40,30
   80 s=a(k,l)
   90 if (s-b) 30,40,40
  100 s=abs(a(k,l))
      go to 70
  110 s=abs(a(k,l))
      go to 90
   30 b=s
      i=k
      j=l
   40 continue
   50 continue
      b=a(i,j)
      return
      entry minim (a,ia,n,m,i,j,b)
      assign 80 to ii
      go to 10
      entry maxam (a,ia,n,m,i,j,b)
      assign 100 to ii
  120 b=abs(a(1,1))
      go to 20
      entry minam (a,ia,n,m,i,j,b)
      assign 110 to ii
      go to 120
      end
      subroutine lplot (mx,my,npts,x,y,inc,iop,nam,ncn,xtl,ncx,ytl,ncy)
c
      character*(*) nam,xtl,ytl
      dimension nam(1), xtl(1), ytl(1)
c  if inc.lt.0,  x(1)=xmin and x(2)=hx
c  abs(iop)   1..lin-lin  2..lin-log  3..log-lin  4..log-log
c    if(iop.lt.0)  scale and frame only
c   abs(npts) number of points  if negative,  add on to previous plot
c  abs(mx) and abs(my) are quadrants  1 is full width or height
c         2 is first half and 3 is second half
c  if quadrant indicator is plus, rescale for roundness.  if not, dont.
c
      common /cje07/ ixl, ixr, iyt, iyb, xl, xr, yt, yb
      dimension x(*), y(*)
      integer hsw
      data hsw /0/
      data nbad /0/
      nbad=nbad+1
c      if(nbad.lt.50) write(9,999) nam(1),ncn,xtl(1),ncx,ytl(1),ncy
c      endfile 9
c  999 format(1x,a8,o24,a8,o24,a8,o24)
   10 iopl=iabs(iop)-1
      logx=iopl.and.2
      logy=iopl.and.1
      inca=iabs(inc)
      ntot=iabs(npts)
c
      if (npts.lt.0) go to 40
      call maxv (y,inca,ntot,idum,ymx)
      call minv (y,inca,ntot,idum,ymn)
      if (inc.lt.0) go to 20
      call maxv (x,inca,ntot,idum,xmx)
      call minv (x,inca,ntot,idum,xmn)
      go to 30
   20 xmn=x(1)
      xmx=x(1)+(ntot-1)*x(2)
   30 call frame9 (mx,my,iopl+1,xmn,xmx,ymn,ymx,
     $      nam(1),ncn,xtl(1),ncx,ytl(1),ncy)
      if (iop.gt.0) go to 50
      return
c
   40 call fram9 (mx,my)
   50 xfac=(ixr-ixl)/(xr-xl)
      yfac=(iyt-iyb)/(yt-yb)
      hx=0.
      if (inc.lt.0) hx=x(2)
      xr=x(1)
      xj=xr
      if (logx.eq.0) go to 60
      xj=alog11(xj)
   60 ix1=min(ixr,max(ixl,ixl+ifix((xj-xl)*xfac)))
      yj=y(1)
      if (logy.eq.0) go to 70
      yj=alog11(yj)
   70 iy1=min(iyt,max(iyb,iyb+ifix((yj-yb)*yfac)))
      if (hsw.ne.0) call gya (iyb,iy1,ix1)
      j=1
      if (hsw.ne.0) ntot=ntot-1
c omura     call ccrv
      do 120 i=2,ntot
      j=j+inca
      xr=xr+hx
      if (inc.gt.0) xr=x(j)
      xj=xr
      if (logx.eq.0) go to 80
      xj=alog11(xj)
   80 ix=min(ixr,max(ixl,ixl+ifix((xj-xl)*xfac)))
      yj=y(j)
      if (logy.eq.0) go to 90
      yj=alog11(yj)
   90 iy=min(iyt,max(iyb,iyb+ifix((yj-yb)*yfac)))
      if (hsw.ne.0) go to 100
      if(ix.ne.ix1)go to 92
      if(ix.eq.ixr)go to 110
      if(ix.eq.ixl)go to 110
   92 if(iy.ne.iy1)go to 94
      if(iy.eq.iyt)go to 110
      if(iy.eq.iyb)go to 110
   94 continue
      call drv (ix1,iy1,ix,iy)
      go to 110
  100 if(ix.ne.ix1)go to 102
      if(ix.eq.ixr)go to 104
      if(ix.eq.ixl)go to 104
  102 call drv (ix1,iy1,ix,iy1)
  104 if(iy.ne.iy1)go to 106
      if(iy.eq.iyt)go to 110
      if(iy.eq.iyb)go to 110
  106 continue
      call drv (ix,iy1,ix,iy)
  110 ix1=ix
  120 iy1=iy
      if (hsw.eq.0) return
      hsw=0
      ix=ix1+float(ix1-ixl)/float(ntot-1)
      call drv (ix1,iy1,ix,iy1)
      call drv (ix,iy1,ix,iyb)
      return
      entry hplot (mx,my,npts,x,y,inc,iop,nam,ncn,xtl,ncx,ytl,ncy)
      hsw=1
      go to 10
      end
      subroutine pplotc (mx,my,npts,x,y,inc,z,zmin,zmax)
c
      common /cje07/ ixl, ixr, iyt, iyb, xl, xr, yt, yb
      dimension x(*), y(*), z(*)
c
      call fram9 (mx,my)
c omura      call cpnt
      xfac=(ixr-ixl)/(xr-xl)
      yfac=(iyt-iyb)/(yt-yb)
      jlast=npts*inc
      do 20 j=1,jlast,inc
      if (z(j).lt.zmin) go to 20
      if (z(j).gt.zmax) go to 20
      ix=max(ixl,min(ixr,ixl+ifix((x(j)-xl)*xfac)))
      iy=max(iyb,min(iyt,iyb+ifix((y(j)-yb)*yfac)))
      call pntabs (ix,iy)
c     print*,x(j),y(j),ix,iy
c     read*, iii
   20 continue
      return
      end
c
      subroutine pplot(mx,my,npts,x,y,inc)
c
      common /cje07/ ixl, ixr, iyt, iyb, xl, xr, yt, yb
      dimension x(*), y(*)
      isw=1
   10 call fram9 (mx,my)
      xfac=(ixr-ixl)/(xr-xl)
      yfac=(iyt-iyb)/(yt-yb)
      jlast=npts*inc
   30 do 40 j=1,jlast,inc
      ix=max(ixl,min(ixr,ixl+ifix((x(j)-xl)*xfac)))
      iy=max(iyb,min(iyt,iyb+ifix((y(j)-yb)*yfac)))
c      print*,x(j),y(j),ix,iy
c      read*, iii
   40 call pntabs (ix,iy)
      return
      end
      subroutine lbltop (labelt,nct)
      character*(*) labelt
      character labtop*80, labbot*40
      common/lhead/labtop,labbot,mct,mcb
      mct=min(nct,80)
      labtop = labelt
c      write(labtop,10) labelt
c   10 format(a80)
      return
      entry lblbot (labelt,nct)
      mcb=min(nct,40)
      labbot = labelt
c      write(labbot,20) labelt
c   20 format(a40)
      return
      end
      subroutine frame9 (mx,my,iop,xlnn,xrnn,ybnn,ytnn,nam,ncn,xtl,ncx
     1 ,ytl,ncy)
c
      character*(*) nam,xtl,ytl
      dimension nam(1),xtl(1),ytl(1)
      dimension ixl(3), ixr(3), iyb(6), iyt(6), jxl(18), jxr(18), jyb(18
     1 ), jyt(18), xl(18), xr(18), yb(18), yt(18)
      dimension kdates(3)
      character labtop*80,labbot*40
      common/lhead/labtop,labbot,mct,mcb
      equivalence (kx,kvx), (ky,kvy)
      integer asw
      data asw /0/
      data ixl /90,90,590/, ixr /990,490,990/, iyt /722,722,362,722,482,
     1 242/, iyb /77,437,77,557,317,77/
      data initf /0/, mct /1/, mcb /1/
c
      data nbad /0/
c omura      call cfrm
      nbad=nbad+1
c      if(nbad.lt.50) write(9,999) nam(1),ncn,xtl(1),ncx,ytl(1),ncy
c      endfile 9
c  999 format(1x,a8,o24,a8,o24,a8,o24)
      noscal=0
      if (initf.ne.0) go to 10
c      call date (kdates(1))
c      call time (kdates(2))
      initf=1
   10 mxp=iabs(mx)
      myp=iabs(my)
      m=myp+6*mxp-6
      if (m.gt.18) stop 765
      iopl=iabs(iop)-1
      logx=iopl.and.2
      logy=iopl.and.1
      xln=xlnn
      xrn=xrnn
      if (xln.eq.xrn) xrn=xln+1.
      if (logx.eq.0) go to 20
      if (xrn.le.0) xrn=1.0
      xln=amax1(xln,1.e-10*xrn)
      xln=alog11(xln)
      xrn=alog11(xrn)
   20 ybn=ybnn
      ytn=ytnn
      if (ybn.eq.ytn) ytn=ybn+1.
      if (logy.eq.0) go to 30
      if (ytn.le.0) ytn=1.
      ybn=amax1(ybn,1.e-10*ytn)
      ybn=alog11(ybn)
      ytn=alog11(ytn)
   30 continue
      xl(m)=xln
      jxl(m)=ixl(mxp)
      xr(m)=xrn
      jxr(m)=ixr(mxp)
      yb(m)=ybn
      jyb(m)=iyb(myp)
      yt(m)=ytn
      jyt(m)=iyt(myp)
      if (asw.eq.0) go to 40
      if (asw.eq.-1) go to 60
      asw=-1
      go to 50
   40 if (mxp.eq.3) go to 60
      if (myp.eq.3) go to 60
      if (myp.ge.5) go to 60
   50 call adv (1)
      iix=500-6*mct
      call dlch (iix,766,mct,labtop,1)
      iix=600-9*mcb
      call dlch (iix,0,mcb,labbot,2)
c      call dlch (35,20,8,kdates(1),0)
c      call dlch (35,0,8,kdates(2),0)
c
   60 if (xr(m).le.xl(m)) xr(m)=xl(m)+1.
      if (yt(m).le.yb(m)) yt(m)=yb(m)+1.
      kx=4
      lx=1
      if (mx.gt.0) call ascl (2,xl(m),xr(m),kx,kdum,lx)
      ky=4
      ly=1
      if (my.gt.0) call ascl (2,yb(m),yt(m),ky,kdum,ly)
      call dga (jxl(m),jxr(m),jyt(m),jyb(m),xl(m),xr(m),yt(m),yb(m))
      call drv (jxl(m),jyt(m),jxr(m),jyt(m))
      call drv (jxr(m),jyt(m),jxr(m),jyb(m))
      call drv (jxr(m),jyb(m),jxl(m),jyb(m))
      call drv (jxl(m),jyb(m),jxl(m),jyt(m))
      if (noscal.eq.1) go to 110
      iopl=iabs(iop)
      go to (70,80,90,100), iopl
   70 call dlnln (kvx,kvy)
      call sllin (ky,ly)
      call sblin (kx,lx)
      go to 110
   80 call dlnlg (kvx)
      call sblin (kx,lx)
      call sllog
      go to 110
   90 call dlgln (kvy)
      call sblog
      call sllin (ky,ly)
      go to 110
  100 call dlglg
      call sblog
      call sllog
  110 lytit=(jyb(m)+jyt(m))/2-5*ncy
      lxtit=(jxl(m)+jxr(m))/2-5*ncx
      lxnam=(jxl(m)+jxr(m))/2-5*ncn
c     print*,'frame9', ncx,xtl,len(xtl(1))
c     print*,'frame9', ncy,ytl,len(ytl(1))
c     print*,'frame9', ncn,nam,len(nam(1))
      call dlch (lxtit,jyb(m)-43,ncx,xtl,1)
      call dlcv (jxl(m)-60,lytit,ncy,ytl,1)
      call dlch (lxnam,jyt(m)+8,ncn,nam,1)
      return
c
      entry fram9 (mx,my)
      m=iabs(my)+6*iabs(mx)-6
      if (m.gt.18) stop 766
      call dga (jxl(m),jxr(m),jyt(m),jyb(m),xl(m),xr(m),yt(m),yb(m))
      return
c
      entry fram9n (mx,my,iop,xlnn,xrnn,ybnn,ytnn,nam,ncn,xtl,ncx
     1 ,ytl,ncy)
      noscal=1
      go to 10
c
      entry setadv (mx,my,iop,xlnn,xrnn,ybnn,ytnn,nam,ncn,xtl,ncx
     1 ,ytl,ncy)
      asw=mx
      return
      end
c=================== end of libhyb1.f ===============================
