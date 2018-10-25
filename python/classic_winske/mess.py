
ntimes=1001
dtwci=0.05
nx=128
xmax=128.
npltp=200
npltf=200
nplth=1000
nwrtf=20
nwrth=40
wpiwci=10000.
nsp=2
nspec=[5120,5120]
vbspec=[0.90,-0.10]
dnspec=[0.10,0.900]
btspec=[10.,1.]
anspec=[5.,1.]
wspec=[1.,1.]

bete=1.
resis=0.
theta=0.
iemod=0
nskip=2

for k in range(nsp):
    npart = npart + nspec(k)

def main():
    init()
    while it < ntimes:
        it=it+1
        t=t+dtwci
        
        trans()
        field()
        diag2()


def init():
    # define some variables from inputs
    nx1  = nx + 1
    nx2  = nx + 2
    hx   = xmax / float(nx)
    hxi  = 1. / hx
    dt   = wpiwci*dtwci
    thet = theta*1.74533e-2
    cth  = cos(thet)
    sth  = sin(thet)
    bxc  = cth/wpiwci
    byc  = 0.
    bzc  = sth/wpiwci
    vye  = 0.
    vze  = 0.
    te0  = bete/(2.*wpiwci**2)
    pe0  = te0
    
    dnadd= 0.
    for k in range(nsp):
        dnadd = dnadd + dnspec(k)
        
    vmax  = 0.
    for k in range(nsp):
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

    vplim=2.*vmax
    
    #initialize particles
    l4=0
    for k in range(nsp):
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

# initialize field arrays
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