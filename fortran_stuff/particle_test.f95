! Simple example program to push particles in a constant field
! To Do:
!   -- Separate position, velocity updates into functions
!   -- Load values from a file
!   -- Save values to a file
program particle_test
	use globals
	!use particleRoutines
	! Declare variables, constants
	implicit none
	
	integer :: ii
	real :: dt
	real, dimension(Np, 3) :: x=0, v=0 
	character(260) :: runFile='C:\Users\Yoshi\Documents\GitHub\hybrid\simulation_codes\run_inputs\_run_params.run'
	character(260) :: plasmaFile='C:\Users\Yoshi\Documents\GitHub\hybrid\simulation_codes\run_inputs\_plasma_params.plasma'
	
	! Initialize global parameters
	call init()
	call load_run(runFile)
	call load_plasma(plasmaFile)

	! Set initial values
	!dt = 0.01 / wL
	!v(:, 3) = vi !m/s
	!x(:, 2) = rL !m

	!write(*, *) "Initial Position (km)   ", x(1, :)*1e-3
	!write(*, *) "Initial Velocity (km/s) ", v(1, :)*1e-3
	!write(*, *)
	!write(*, *) "Larmor Radius (km) ", rL*1e-3
	!write(*, *) "Gyroperiod (s)     ", TL

	! Update values in a loop and output results
	!write(*, *) "Doing loop"
	!do ii = 1, Nt
	!	call updateVelocityBoris(x, v, dt)
	!	call updatePosition(x, v, dt)
	!	!radius = SQRT(x(1, 2)**2 + x(1, 3)**2)
	!	!write(*, *) x*1e-3, radius*1e-3
	!end do

end program particle_test