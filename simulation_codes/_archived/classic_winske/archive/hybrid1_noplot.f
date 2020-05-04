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
c      call begplt('        ')
      read (8,datum)
c      call lbltop (label,64)
      write (9,160) label
c  check dimensions
      npart=0
      do 20 k=1,nsp
   20 npart=npart+nspec(k)
c      if (nsp.gt.ns) call endrun ('species')
c      if (nx+2.gt.nc) call endrun ('dimen')
c      if (npart.gt.nbbb) call endrun ('parts')
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
c      call diag2
      go to 30
   50 continue
c  close plot file and exit
c      call endplt
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
c      call diag2
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
c      call diag1
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
c====================== end of hybrid1.f ===============================
