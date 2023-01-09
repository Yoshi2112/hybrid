! Module to define global variables (e.g. constants)
! May also load include subroutines to load from text
module globals
implicit none
! Constants
integer, parameter :: Np=100, Nt=1000
real, parameter :: mi=1.673e-27, qi=1.602e-19, vi=1e6
real, dimension(3) :: B0=0, E0=0
real :: Bt, wL, rL, TL
! runFile parameters
character(10) :: driveLetter
character(40) :: savePath
character(300):: runDescription
real :: dampLengthFraction, dampMult, resisMult, simTime, dxm, gperiodRes, freqRes, pdumpFreq, fdumpFreq
integer :: runNumber, seed, NX, tempLogical
logical :: saveParticles, saveFields, uniformB, particlePeriodic, particleReflect, particleReinit
logical :: fieldPeriodic, noWaves, smoothSources, quietStart, dampElectric, ie
! plasmaFile parameters
!...
integer :: Nj
!speciesLabels
!speciesColour
!speciesTemp
!speciesDist
!speciesPPC
!speciesMass
!speciesCharge
!speciesDrift
!speciesDensity
!speciesAnisotropy
!speciesEnergyPerp
!ElecTemp
!plasmaBetaFlag
!L_value
!Beq
!Bxmax
contains
subroutine init()
implicit none
	! Magnetic field
	B0(1) = -200e-9
	Bt = SQRT(B0(1)**2 + B0(2)**2 + B0(3)**2)

	! Calculate gyrofrequency, gyroperiod, Larmor radius, timestep
	wL = qi * Bt / mi
	rL = vi / wL
	TL = (2*3.141593)/wL
end subroutine init
!********************
!** Subroutine :: Load run parameters from runFile
!** This file contains 27 lines, with the last line
!** being a continuous string of indeterminate length
!** All values except the last one start at column 25
subroutine load_run(runFile)
implicit none
integer :: ii
character(260) :: runFile
character(300) :: buf
	write(*, *) "Loading run parameters from file ", runFile
	open(unit=10, file=runFile)
	do ii = 1, 27
		read(10, '(a)') buf
		if (ii < 27) then
			buf = TRIM(ADJUSTL(buf(25:)))
		else
			buf = TRIM(ADJUSTL(buf))
		end if
		
		select case (ii)
		case (1)
			driveLetter = buf
			write(*, *) driveLetter
		case (2)
			savePath = buf
			write(*, *) savePath
		case (3)
			if (buf == '-') then
				runNumber = -1
			else
				read(buf, '(i2)') runNumber
			end if
			write(*, *) runNumber
		case (4)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				saveParticles = .false.
			else
				saveParticles = .true.
			end if
			write(*, *) saveParticles
		case (5)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				saveFields = .false.
			else
				saveFields = .true.
			end if
			write(*, *) saveFields
		case (6)
			read(buf, '(i10)') seed
			write(*, *) seed
		case (7)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				uniformB = .false.
			else
				uniformB = .true.
			end if
			write(*, *) uniformB
		case (8)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				particlePeriodic = .false.
			else
				particlePeriodic = .true.
			end if
			write(*, *) particlePeriodic
		case (9)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				particleReflect = .false.
			else
				particleReflect = .true.
			end if
			write(*, *) particleReflect
		case (10)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				particleReinit = .false.
			else
				particleReinit = .true.
			end if
			write(*, *) particleReinit
		case (11)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				fieldPeriodic = .false.
			else
				fieldPeriodic = .true.
			end if
			write(*, *) fieldPeriodic
		case (12)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				noWaves = .false.
			else
				noWaves = .true.
			end if
			write(*, *) noWaves
		case (13)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				smoothSources = .false.
			else
				smoothSources = .true.
			end if
			write(*, *) smoothSources
		case (14)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				quietStart = .false.
			else
				quietStart = .true.
			end if
			write(*, *) quietStart
		case (15)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				dampElectric = .false.
			else
				dampElectric = .true.
			end if
			write(*, *) dampElectric
		case (16)
			read(buf, '(f6.4)') dampLengthFraction
			write(*, *) dampLengthFraction
		case (17)
			read(buf, '(f6.4)') dampMult
			write(*, *) dampMult
		case (18)
			read(buf, '(f6.4)') resisMult
			write(*, *) resisMult
		case (19)
			read(buf, '(i5)') NX
			write(*, *) NX
		case (20)
			read(buf, '(f7.1)') simTime
			write(*, *) simTime
		case (21)
			read(buf, '(f4.2)') dxm
			write(*, *) dxm
		case (22)
			read(buf, '(i2)') tempLogical
			if (tempLogical == 0) then
				ie = .false.
			else
				ie = .true.
			end if
			write(*, *) ie
		case (23)
			read(buf, '(f5.3)') gperiodRes
			write(*, *) gperiodRes
		case (24)
			read(buf, '(f5.3)') freqRes
			write(*, *) freqRes
		case (25)
			read(buf, '(f5.3)') pdumpFreq
			write(*, *) pdumpFreq
		case (26)
			read(buf, '(f5.3)') fdumpFreq
			write(*, *) fdumpFreq
		case (27)
			runDescription = buf
			write(*, *) runDescription
		end select
	end do
	close(10)
	return
end subroutine load_run
!********************
!** Subroutine :: Load plasma parameters from plasmaFile
subroutine load_plasma(plasmaFile)
	implicit none
	integer :: ii
	character(260) :: plasmaFile
	character(300) :: buf
	!*****************
	write(*, *) "Loading run parameters from file ", plasmaFile
	open(unit=20, file=plasmaFile)
	! Read number of species
	read(20, '(a)') buf
	write(*, *) buf
	! Count number of characters
	! Species encountered when string goes from space to character
	do ii = 1, LEN(buf)	
		! If ii is space and ii + 1 is not, Nj += 1
		if ((buf(ii:ii) == ' ') .and. (buf(ii+1:ii+1) .ne. ' ')) then
			write(*, *) buf(ii:ii), buf(ii+1:ii+1)
			Nj = Nj + 1
		end if
	end do
	! Weird because apparently there's a comma in buf
	Nj = Nj - 1
	write(*, *) "Number of species ", Nj
end subroutine load_plasma
end module globals