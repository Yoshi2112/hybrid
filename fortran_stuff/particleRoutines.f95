!************************************************
!** Module for dealing with particles
!** This module will eventually contain
!** -- Position update
!** -- Velocity update
!** -- Particle weighting
!** -- Particle boundary condition subroutines
!** -- Particle injection subroutines
!************************************************
module particleRoutines
implicit none
	! Global variable declarations go here
contains	
!***************
! Subroutine to advance the position of particles based on their velocity
subroutine updatePosition(x, v, dt)
use globals
implicit none
real, dimension(Np, 3), intent(inout) :: x
real, dimension(Np, 3), intent(in) :: v
real, intent(in) :: dt
integer :: ii
	do ii = 1, Np
		x(ii, :) = x(ii, :) + v(ii, :)*dt
	end do
	return
end subroutine updatePosition
!***************
! Subroutine to update velocity of particle based on the Boris-Buneman algorithm and the strength of a uniform magnetic field B0 and electric field E0
subroutine updateVelocityBoris(x, v, dt)
use globals
implicit none
real, dimension(Np, 3), intent(inout) :: v
real, dimension(Np, 3), intent(in) :: x
real, intent(in) :: dt
real, dimension(3) :: S, T, v_prime
real :: qmi
integer :: ii
	! Save number of operations
	qmi = 0.5 * dt * qi / mi
	do ii = 1, Np
		! v -> v_minus
		v(ii, :) = v(ii, :) + qmi * E0

		! Boris variables
		T = qmi * B0
		S = 2.*T / (1. + T(1)*T(1) + T(2)*T(2) + T(3)*T(3))

		! Calculate v_prime from cross product
		v_prime(1) = v(ii, 1) + v(ii, 2)*T(3) - v(ii, 3)*T(2)
		v_prime(2) = v(ii, 2) + v(ii, 3)*T(1) - v(ii, 1)*T(3)
		v_prime(3) = v(ii, 3) + v(ii, 1)*T(2) - v(ii, 2)*T(1)
		
		! v_minus -> v_plus
		v(ii, 1) = v(ii, 1) + v_prime(2)*S(3) - v_prime(3)*S(2)
		v(ii, 2) = v(ii, 2) + v_prime(3)*S(1) - v_prime(1)*S(3)
		v(ii, 3) = v(ii, 3) + v_prime(1)*S(2) - v_prime(2)*S(1)

		! v_plus -> v (updated)
		v(ii, :) = v(ii, :) + qmi * E0
	end do
	return
end subroutine updateVelocityBoris
end module particleRoutines