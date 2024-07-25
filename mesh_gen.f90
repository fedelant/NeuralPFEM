!!***********************************************  
subroutine Tassellation(node_array, vel_array, free_array, bound_array, out_array, edges, step, ar_freq, n_points)
use fluid_module

integer :: n_points, i1, j1, z1, index, step, ar_freq
integer, dimension(n_points) :: free_array
integer, dimension(n_points) :: bound_array
real, dimension(n_points, 2) :: node_array, vel_array
integer, dimension(n_points*3, 3) :: out_array
integer, dimension(n_points*3*9, 2) :: edges

!!f2py input-output declaration
!f2py intent(in) steps
!f2py intent(in) ar_freq
!f2py intent(in) n_points
!f2py intent(in) node_array
!f2py intent(out) node_array
!f2py intent(in) vel_array
!f2py intent(out) vel_array
!f2py intent(in) free_array
!f2py intent(in) bound_array
!f2py intent(out) free_array
!f2py intent(in) out_array
!f2py intent(out) out_array
!f2py intent(in) edges
!f2py intent(out) edges

!!---------------------------------------------------
!!This subroutine performs the remeshing process with
!!1) The Delaunay triangulation
!!2) The Alpha-shape method
!!----------------------------------------------

!!Fill data structures with array input data
call fill_struct(node_array, vel_array, free_array, bound_array, n_points)

!!The Delaunay triangulation
call mesh_generation(npoints)

call alpha_shape()
if (mod(step,ar_freq) == 0) then
    call AddRemNodesLocal()
end if

!!Fill output arrays
out_array = 0
index = 1
do i1=1,size(elem_array) / 3
    out_array(i1, 1) = elem_array(i1, 1) -1
    out_array(i1, 2) = elem_array(i1, 2) -1
    out_array(i1, 3) = elem_array(i1, 3) -1
    do j1=1,3
       do z1=1,3
        !!if (j1/=z1 .and. (node(out_array(i1, j1) + 1)%bound /= 1 &
          !!  .or. node(out_array(i1, z1) + 1)%bound /= 1)) then
          edges(index, 1) = out_array(i1, j1) 
          edges(index, 2) = out_array(i1, z1) 
          index = index + 1
        !!end if
       end do
    end do
end do

free_array = 0

do i1=1,n_points
    node_array(i1, 1) = node(i1)%coord(1)
    node_array(i1,2) = node(i1)%coord(2) 
    vel_array(i1, 1) = unn(2*i1-1)
    vel_array(i1, 2) = unn(2*i1)
    if (node(i1)%free_surf == 1) then
        free_array(i1) = 1
    end if
end do

!!Deallocate data structures
deallocate(node)
deallocate(unn)
deallocate(elem_array)
deallocate(elements)
return
end subroutine
!!***********************************************

!!The Delaunay triangulation
!!***********************************************  
subroutine mesh_generation(n)
!
  use fluid_module
  implicit none

  integer :: n
  integer, parameter :: ncc_max = 0
  integer, parameter :: nrow = 9

  real ( kind = 8 ) a
  real ( kind = 8 ) areap
  real ( kind = 8 ) armax
  real ( kind = 8 ) ds(n)
  real ( kind = 8 ) dsq
  integer ier
  integer io1
  integer io2
  integer k
  integer ksum
  integer, dimension ( ncc_max ) :: lcc = 10 
  integer lct(ncc_max)
  integer lend(n)
  integer, parameter :: lin = 1
  integer list(6*n)
  integer lnew
  integer, parameter :: lplt = 3
  integer lptr(6*n)
  integer ltri(nrow,2*n)
  integer lw
  integer n0
  integer na
  integer nb
  integer :: ncc = 0
  integer nearnd
  integer nn
  integer nodes(2*n)
  integer nt
  logical numbr
  real ( kind = 8 ), parameter :: pltsiz = 7.5D+00
  logical prntx
  character ( len = 80 ) title
  real ( kind = 8 ), parameter :: tol = 0.001D+00
  real ( kind = 8 ) wx1
  real ( kind = 8 ) wx2
  real ( kind = 8 ) wy1
  real ( kind = 8 ) wy2
  integer :: i1,j1
  real(8), allocatable :: x(:)
  real(8), allocatable :: y(:)
  real(8) :: x1, x2, x3, y1, y2, y3
  real(8) :: xc, yc, cr, sa, ar

  allocate(x(n))
  allocate(y(n))

  !!Assign the coordinates of the mesh nodes 
  do i1=1,n
     x(i1)=node(i1)%coord(1)*1000
     y(i1)=node(i1)%coord(2)*1000
  end do

  
  if ( n < 3 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'TRIPACK_PRB - Fatal error!'
    write ( *, '(a)' ) '  N must be at least 3!'
    stop
  end if
  
  !!Create the Delaunay triangulation (TRMESH), and test
  !!for errors (refer to TRMTST below).  NODES and DS are
  !!used as work space.
  call trmesh ( n, x, y, list, lptr, lend, lnew, nodes, nodes(n+1), ds, ier )

  if ( ier == -2 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'TRIPACK_PRB - Fatal error!'
    write ( *, '(a)' ) '  Error in TRMESH:'
    write ( *, '(a)' ) '  The first three nodes are collinear.'
    stop
  else if ( ier == -4 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'TRIPACK_PRB - Fatal error!'
    write ( *, '(a)' ) '  Error in TRMESH:'
    write ( *, '(a)' ) '  Invalid triangulation.'
    stop
  else if ( 0 < ier ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'TRIPACK_PRB - Fatal error!'
    write ( *, '(a)' ) '  Error in TRMESH:'
    write ( *, '(a)' ) '  Duplicate nodes encountered.'  
    stop
  end if
  
  lw = 2*n

  if ( ier /= 0 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'TRIPACK_PRB - Fatal error!'
    write ( *, '(a,i6)' ) '  Error in ADDCST, IER = ', ier
    stop
  end if

  if ( lw == 0 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'TRIPACK_PRB: Note'
    write ( *, '(a)' ) '  Subroutine EDGE was not tested, because'
    write ( *, '(a)' ) '  no edges were swapped by ADDCST.'
  end if

  prntx = .true.
  call trlist ( ncc, lcc, n, list, lptr, lend, nrow, nt, ltri, lct, ier )

  !!Total number of the elements created by the Delaunay triangulation
  number_elements = nt 
  allocate(list_elements(number_elements))

  do i1=1,nt
    list_elements(i1)%nodes =  ltri(1:3,i1)
    x1=node( list_elements(i1)%nodes(1) )%coord(1)
    y1=node( list_elements(i1)%nodes(1) )%coord(2)
    x2=node( list_elements(i1)%nodes(2) )%coord(1)
    y2=node( list_elements(i1)%nodes(2) )%coord(2)
    x3=node( list_elements(i1)%nodes(3) )%coord(1)
    y3=node( list_elements(i1)%nodes(3) )%coord(2)
    call circum ( x1, y1, x2, y2, x3, y3, .false., xc, yc, cr, sa, ar )
    list_elements(i1)%circumcenter(1) = xc
    list_elements(i1)%circumcenter(2) = yc
  end do

  deallocate(x,y)

  return

end subroutine
!!***********************************************  

!!The alpha shape    
!!***********************************************************************************
subroutine alpha_shape

!!-----------------------------------------------------------------------------------
!!This subroutine performs the Alpha-shape method, which removes unphysical elements.
!!-----------------------------------------------------------------------------------

use fluid_module
implicit none

integer :: i1,j1,n(3),count,control,mm,k1,r1
real(8) :: hh, x(3), y(3), jac
real(8), allocatable :: circle_radius(:)
integer(2), allocatable :: check(:)
integer :: CountInt, CountFree, CountBound, CountEuler

allocate(h(number_elements))
allocate(circle_radius(number_elements))

if (problem_type == 3) then 
    do i1=1,npoints
        if(node(i1)%bed == 2) then
            node(i1)%bound = 0
            node(i1)%robin = 0
            node(i1)%int   = 1
        end if
    end do
endif
    
!!1) Compute the elements characteristic length of each element

!Loop over the elements
do i1=1,number_elements
  n(1)=list_elements(i1)%nodes(1)
  n(2)=list_elements(i1)%nodes(2)
  n(3)=list_elements(i1)%nodes(3)

  do j1=1,3
    x(j1)=node(n(j1))%coord(1)
    y(j1)=node(n(j1))%coord(2)
  end do
    
  !!Compute the radius of the circumcircle r=sqrt( (x1-xc)^2-(y1-yc)^2)
  circle_radius(i1)=((node(n(1))%coord(1)-list_elements(i1)%circumcenter(1))**2 + &
  (node(n(1))%coord(2)-list_elements(i1)%circumcenter(2))**2)**.5

  !!Minimum between the 3 sides
  h(i1)=min(((node(n(1))%coord(1)-node(n(2))%coord(1))**2+(node(n(1))%coord(2)-node(n(2))%coord(2))**2)**.5,&
     ((node(n(1))%coord(1)-node(n(3))%coord(1))**2+(node(n(1))%coord(2)-node(n(3))%coord(2))**2)**.5,&
     ((node(n(3))%coord(1)-node(n(2))%coord(1))**2+(node(n(3))%coord(2)-node(n(2))%coord(2))**2)**.5)
end do

!!Characteristic mesh size: take as the mean value of the minimum sides of all the elements
hh=sum(h)/number_elements

if(it_time==1 .and. benchmark/=3 )then
    el_size=hh
end if

control=0
count=0

!!Remark: 
!!1) "number_elements" is the total number of triangles obtained from the Delaunay triangulation before the Alpha-shape;
!!2) "list_elements" contains all triangles information  (unphysical elements are included).

!!Loop over the all elements
do i1=1,number_elements
    
    do j1=1,3
       n(j1)=list_elements(i1)%nodes(j1)

       x(j1)=node(n(j1))%coord(1)
       y(j1)=node(n(j1))%coord(2)
    end do
    
    CountInt=0
    CountFree=0
    CountBound=0
    CountEuler=0
	
	do j1=1,3
	   n(j1)=list_elements(i1)%nodes(j1)

	   x(j1)=node(n(j1))%coord(1)
	   y(j1)=node(n(j1))%coord(2)
       
        
        !!Count the number of nodes inside the fluid bulk within the element
        if(node(n(j1))%int==1)then        
            CountInt=CountInt+1
        end if
        
        !!Count the number of nodes on the free surface within the element
        if(node(n(j1))%free_surf==1)then        
            CountFree=CountFree+1
        end if
        
        !!Count the number of bounded nodes within the element
        if(node(n(j1))%bound==1)then        
            CountBound=CountBound+1
        end if
        
        !!Count the number of eulerian nodes within the element
        if(node(n(j1))%euler==1)then        
            CountEuler=CountEuler+1
        end if
        
    end do
     
     call triangle_area(x(1),y(1),x(2),y(2),x(3),y(3),jac)

    if(jac/=0.d0)then
            
        !!Slip elements
        if(node(n(1))%slip==1 .and. node(n(2))%slip==1 .and. node(n(3))%slip==1)then
                list_elements(i1)%area=0.d0
            
        !!Box elements
        elseif(node(n(1))%box==1 .or. node(n(2))%box==1 .or. node(n(3))%box==1)then
                list_elements(i1)%area=0.d0
                    
        !!!Bed elements
        !elseif(node(n(1))%bed==1 .and. node(n(2))%bed==1 .and. node(n(3))%bed==1)then
        !        list_elements(i1)%area=0.d0
                    
        !!3-nodes internal
        else if (CountInt == 3) then
            call alphainner(hh,circle_radius(i1),control,i1,count)  
            
        !3-nodes bound
        else if (CountBound == 3) then
            list_elements(i1)%area=0.d0        
            
        !!3-nodes free-surface     
        else if (CountFree == 3) then
            call alphasurf(hh,circle_radius(i1),control,i1,count)  
                
        !1-nodes bound, 1-node free-surf
        else if (CountBound >= 1 .and. CountFree >= 1) then
            call alphabound(hh,circle_radius(i1),control,i1,count)
            
        else    
        !!all other cases    
            call alphageneral(hh,circle_radius(i1),control,i1,count)
        end if
            
    end if
end do

!!Remarks: 
!!At this stage the Alpha-shape method has been applied to the convex hull created by the Delaunay triangulation:
!!1) distorted elements will have area index=0;
!!2) undistorted elements will have area index  =1;
!!3) "elements" contains the remaining triangles after the application of the Alpha-shape. It has to be allocated 
!!    whenever a Delaunay triangulation is performed.

!!number of the remaining (undistorted) elements after Alpha-Shape
nelement=count
allocate(elem_array(nelement,3))
allocate(elements(nelement))

do i1=1,npoints
    node(i1)%free_surf=0
    node(i1)%int=0
end do

count=0
do i1=1,number_elements
    if (list_elements(i1)%area==1.d0)then
    
        !!undistorted elements: save it 
        count=count+1
        elements(count)%nodes=list_elements(i1)%nodes
        elements(count)%circumcenter=list_elements(i1)%circumcenter 
        elem_array(count,1)=list_elements(i1)%nodes(1)
        elem_array(count,2)=list_elements(i1)%nodes(2)
        elem_array(count,3)=list_elements(i1)%nodes(3)
     else 
     
        !!distorted elements: nodes not on the boundaries will be on the free surface 
        do j1=1,3
            if ( node(list_elements(i1)%nodes(j1))%bound==0 )then
                node(list_elements(i1)%nodes(j1))%free_surf=1
            end if
        end do
    end if
end do

!!A node will be internal if it is neither on the boundaries nor on the free surface
do i1=1,npoints
    node(i1)%alone=1
    node(i1)%neighb=0
    if (node(i1)%free_surf==0 .and. node(i1)%bound==0) then
        node(i1)%int=1
    end if   
end do

!!When a node belongs to an element, it will not be alone.
do k1=1,nelement
    n=elements(k1)%nodes
    node(n(1))%alone=0
    node(n(2))%alone=0
    node(n(3))%alone=0
    
    !!Find neighbouring nodes 
    do i1=1,3
     do j1=1,3
        do r1=2,16
          if (node(n(i1))%neighb(r1)==n(j1) .or. n(i1)==n(j1)) then
            exit
          elseif (node(n(i1))%neighb(r1)==0) then
             node(n(i1))%neighb(r1)=n(j1)
             node(n(i1))%neighb(1)=node(n(i1))%neighb(1)+1
             exit
          end if
        end do
      end do
    end do
    
end do

!!An alone node will be considered as belonging to the free surface
do i1=1,npoints
    if(node(i1)%alone==1) then
        node(i1)%free_surf=1
    end if
end do


deallocate(list_elements)
deallocate(circle_radius)
deallocate(h)


end subroutine alpha_shape
    
    
    
    
!!***************************************************************************  
subroutine alphainner(hhh,Rad,control,i1,count)

!!---------------------------------------------------------------------------
!!This subroutine applies the Alpha-shape to an element inside the fluid bulk
!!---------------------------------------------------------------------------


use fluid_module
implicit none

real(8) :: Rad,hhh
integer :: count,control,i1, j1

if (control==0)then
    if (Rad<alpha_inner*hhh)then
        !!Undistorted element to be kept
        count=count+1
        list_elements(i1)%area=1.d0
    else
        !!Distorted element to be removed
        list_elements(i1)%area=0.d0 
    end if
end if   
    
return
end subroutine alphainner
!!*************************************************************************  

    
    
    
!!*************************************************************************  
subroutine alphasurf(hhh,Rad,control,i1,count)

!!-------------------------------------------------------------------------
!!This subroutine applies the Alpha-shape to an element at the free surface
!!-------------------------------------------------------------------------

use fluid_module
implicit none

real(8) :: Rad,hhh
integer :: count,control,i1,j1

if (control==0)then
    if (Rad<alpha_surf*hhh)then
        !!Undistorted element to be kept
        count=count+1
        list_elements(i1)%area=1.d0
    else
        !!Distorted element to be removed
        list_elements(i1)%area=0.d0 
    end if
end if

return
end subroutine alphasurf
!!*************************************************************************   
    
    
    
    
!!*************************************************************************  
subroutine alphabound(hhh,Rad,control,i1,count)
  
!!---------------------------------------------------------------------
!!This subroutine applies the Alpha-shape to an element at the boundary
!!---------------------------------------------------------------------

use fluid_module
implicit none

real(8) :: Rad,hhh
integer :: count,control,i1,j1

if (control==0)then
    if (Rad<alpha_bound*hhh)then
        !!Undistorted element to be kept
        count=count+1
        list_elements(i1)%area=1.d0
    else
        !!Distorted element to be removed
        list_elements(i1)%area=0.d0 
    end if
end if

return
end subroutine alphabound
!!*************************************************************************   


  
    
!!*************************************************************************    
subroutine alphageneral(hhh,Rad,control,i1,count)

!!------------------------------------------------------------
!!This subroutine applies the Alpha-shape to a general element
!!not included in the previous cases.
!!------------------------------------------------------------

use fluid_module
implicit none

real(8) :: Rad,hhh
integer :: count,control,i1,j1

if (control==0)then
    if (Rad<alpha_gen*hhh )then
        !!Undistorted element to be kept
        count=count+1
        list_elements(i1)%area=1.d0
    else
        !!Distorted element to be removed
        list_elements(i1)%area=0.d0 
    end if
end if

return
end subroutine alphageneral
!!*************************************************************************      

!!Utilities
!!*************************************************************************
function store ( x )

  implicit none

  real ( kind = 8 ) store
  real ( kind = 8 ) x
  real ( kind = 8 ) y

  common /stcom/ y

  y = x
  store = y

  return
end


function indxcc ( ncc, lcc, n, list, lend )
  implicit none

  integer n

  integer i
  integer ifrst
  integer ilast
  integer indxcc
  integer lcc(*)
  integer lend(n)
  integer list(*)
  integer lp
  integer n0
  integer ncc
  integer nst
  integer nxt

  indxcc = 0

  if ( ncc < 1 ) then
    return
  end if
!
!  Set N0 to the boundary node with smallest index.
!
  n0 = 0

  do

    n0 = n0 + 1
    lp = lend(n0)

    if ( list(lp) <= 0 ) then
      exit
    end if

  end do
!
!  Search in reverse order for the constraint I, if any, that
!  contains N0.  IFRST and ILAST index the first and last
!  nodes in constraint I.
!
  i = ncc
  ilast = n

  do

    ifrst = lcc(i)

    if ( ifrst <= n0 ) then
      exit
    end if

    if ( i == 1 ) then
      return
    end if

    i = i - 1
    ilast = ifrst - 1

  end do
!
!  N0 is in constraint I which indexes an exterior constraint
!  curve iff the clockwise-ordered sequence of boundary
!  node indexes beginning with N0 is increasing and bounded
!  above by ILAST.
!
  nst = n0

  do

    nxt = -list(lp)

    if ( nxt == nst ) then
      exit
    end if

    if ( nxt <= n0  .or. ilast < nxt ) then
      return
    end if

    n0 = nxt
    lp = lend(n0)

  end do
!
!  Constraint I contains the boundary node sequence as a subset.
!
  indxcc = i

  return
end
    
function jrand ( n, ix, iy, iz )

  implicit none

  integer ix
  integer iy
  integer iz
  integer jrand
  integer n
  real ( kind = 8 ) u
  real ( kind = 8 ) x

  ix = mod ( 171 * ix, 30269 )
  iy = mod ( 172 * iy, 30307 )
  iz = mod ( 170 * iz, 30323 )

  x = ( real ( ix, kind = 8 ) / 30269.0D+00 ) &
    + ( real ( iy, kind = 8 ) / 30307.0D+00 ) &
    + ( real ( iz, kind = 8 ) / 30323.0D+00 )

  u = x - int ( x )
  jrand = real ( n, kind = 8 ) * u + 1.0D+00

  return
end

    
function left ( x1, y1, x2, y2, x0, y0 )

  implicit none

  real ( kind = 8 ) dx1
  real ( kind = 8 ) dx2
  real ( kind = 8 ) dy1
  real ( kind = 8 ) dy2
  logical left
  real ( kind = 8 ) x0
  real ( kind = 8 ) x1
  real ( kind = 8 ) x2
  real ( kind = 8 ) y0
  real ( kind = 8 ) y1
  real ( kind = 8 ) y2

  dx1 = x2 - x1
  dy1 = y2 - y1
  dx2 = x0 - x1
  dy2 = y0 - y1
!
!  If the sign of the vector cross product of N1->N2 and
!  N1->N0 is positive, then sin(A) > 0, where A is the
!  angle between the vectors, and thus A is in the range
!  (0,180) degrees.
!
  left = dx1 * dy2 >= dx2 * dy1

  return
end
    
function crtri ( ncc, lcc, i1, i2, i3 )

  implicit none

  logical crtri
  integer i
  integer i1
  integer i2
  integer i3
  integer imax
  integer imin
  integer lcc(*)
  integer ncc

  imax = max ( i1, i2, i3 )
!
!  Find the index I of the constraint containing IMAX.
!
  i = ncc + 1

  do

    i = i - 1

    if ( i <= 0 ) then
      crtri = .false.
      return
    end if

    if ( lcc(i) <= imax ) then
      exit
    end if

  end do

  imin = min ( i1, i2, i3 )
!
!  P lies in a constraint region iff I1, I2, and I3 are nodes
!  of the same constraint (LCC(I) <= IMIN), and (IMIN,IMAX)
!  is (I1,I3), (I2,I1), or (I3,I2).
!
  crtri = lcc(i) <= imin .and. ( &
    ( imin == i1 .and. imax == i3 ) .or.  &
    ( imin == i2 .and. imax == i1 ) .or.  &
    ( imin == i3 .and. imax == i2 ) )

  return
end
    
function swptst ( in1, in2, io1, io2, x, y )

  implicit none

  real ( kind = 8 ) cos1
  real ( kind = 8 ) cos2
  real ( kind = 8 ) dx11
  real ( kind = 8 ) dx12
  real ( kind = 8 ) dx21
  real ( kind = 8 ) dx22
  real ( kind = 8 ) dy11
  real ( kind = 8 ) dy12
  real ( kind = 8 ) dy21
  real ( kind = 8 ) dy22
  integer in1
  integer in2
  integer io1
  integer io2
  real ( kind = 8 ) sin1
  real ( kind = 8 ) sin12
  real ( kind = 8 ) sin2
  logical swptst
  real ( kind = 8 ) swtol
  real ( kind = 8 ) x(*)
  real ( kind = 8 ) y(*)
!
!  Tolerance stored by TRMESH or TRMSHR.
!
  common /swpcom/ swtol
!
!  Compute the vectors containing the angles T1 and T2.
!
  dx11 = x(io1) - x(in1)
  dx12 = x(io2) - x(in1)
  dx22 = x(io2) - x(in2)
  dx21 = x(io1) - x(in2)

  dy11 = y(io1) - y(in1)
  dy12 = y(io2) - y(in1)
  dy22 = y(io2) - y(in2)
  dy21 = y(io1) - y(in2)
!
!  Compute inner products.
!
  cos1 = dx11 * dx12 + dy11 * dy12
  cos2 = dx22 * dx21 + dy22 * dy21
!
!  The diagonals should be swapped iff 180 < (T1+T2)
!  degrees.  The following two tests ensure numerical
!  stability:  the decision must be FALSE when both
!  angles are close to 0, and TRUE when both angles
!  are close to 180 degrees.
!
  if ( 0.0D+00 <= cos1 .and. 0.0D+00 <= cos2 ) then
    swptst = .false.
    return
  end if

  if ( cos1 < 0.0D+00 .and. cos2 < 0.0D+00 ) then
    swptst = .true.
    return
  end if
!
!  Compute vector cross products (Z-components).
!
  sin1 = dx11 * dy12 - dx12 * dy11
  sin2 = dx22 * dy21 - dx21 * dy22
  sin12 = sin1 * cos2 + cos1 * sin2

  if ( -swtol <= sin12 ) then
    swptst = .false.
  else
    swptst = .true.
  end if

  return
end

function lstptr ( lpl, nb, list, lptr )
!
  implicit none

  integer list(*)
  integer lp
  integer lpl
  integer lptr(*)
  integer lstptr
  integer nb
  integer nd

  lp = lptr(lpl)

  do

    nd = list(lp)

    if ( nd == nb ) then
      exit
    end if

    lp = lptr(lp)

    if ( lp == lpl ) then
      exit
    end if

  end do

  lstptr = lp

  return
end

subroutine triangle_area(x1,y1,x2,y2,x3,y3,area)

!!-----------------------------------------------------
!! Compute the area of the triangle
!! Input variables : coordinates x,y of the three nodes
!! output variables: area
!!-----------------------------------------------------

implicit none
real(8) :: x1,y1,x2,y2,x3,y3,area

area=abs((x1-x3)*(y2-y3)-(x2-x3)*(y1-y3))/2

return
end subroutine triangle_area

subroutine fill_struct(node_array, vel_array, free_array, bound_array, n)
!
  use fluid_module
  
  integer :: n
  real, dimension(n, 2) :: node_array, vel_array
  integer, dimension(n) :: free_array
  integer, dimension(n) :: bound_array
  
  alpha_surf = 2
  alpha_inner = 1.2
  alpha_bound = 1.2
  alpha_gen = 1.2
  
  npoints = n
  allocate(node(npoints))
  allocate(unn(2*npoints))
  
  do i = 1,n
    node(i)%number = i
    node(i)%coord(1) = node_array(i,1)
    node(i)%coord(2) = node_array(i,2)
    node(i)%bound = bound_array(i)
    node(i)%free_surf = free_array(i)
    if (bound_array(i) == 0 .and. free_array(i) == 0) then
        node(i)%int = 1
    end if
    unn(2*i-1)=vel_array(i,1)
    unn(2*i)=vel_array(i,2)
  end do
  
  return
  
end subroutine

subroutine trmesh ( n, x, y, list, lptr, lend, lnew, near, next, dist, ier )

  implicit none

  integer n

  real ( kind = 8 ) d
  real ( kind = 8 ) d1
  real ( kind = 8 ) d2
  real ( kind = 8 ) d3
  real ( kind = 8 ) dist(n)
  real ( kind = 8 ) eps
  integer i
  integer i0
  integer ier
  integer j
  integer k
  integer km1
  integer lcc(1)
  logical left
  integer lend(n)
  integer list(*)
  integer lnew
  integer lp
  integer lpl
  integer lptr(*)
  integer ncc
  integer near(n)
  integer next(n)
  integer nexti
  integer nn
  real ( kind = 8 ) swtol
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) y(n)

  common /swpcom/ swtol

  nn = n

  if ( nn < 3 ) then
    ier = -1
    return
  end if

!  Compute a tolerance for function SWPTST:  SWTOL = 10*
!  (machine precision)
!
  eps = epsilon ( eps )

  swtol = eps * 20.0D+00
!
!  Store the first triangle in the linked list.
!
  if ( .not. left ( x(1), y(1), x(2), y(2), x(3), y(3) ) ) then
!
!  The initial triangle is (3,2,1) = (2,1,3) = (1,3,2).
!
    list(1) = 3
    lptr(1) = 2
    list(2) = -2
    lptr(2) = 1
    lend(1) = 2

    list(3) = 1
    lptr(3) = 4
    list(4) = -3
    lptr(4) = 3
    lend(2) = 4

    list(5) = 2
    lptr(5) = 6
    list(6) = -1
    lptr(6) = 5
    lend(3) = 6

  else if ( .not. left(x(2),y(2),x(1),y(1),x(3),y(3)) ) then
!
!  The initial triangle is (1,2,3).
!
    list(1) = 2
    lptr(1) = 2
    list(2) = -3
    lptr(2) = 1
    lend(1) = 2

    list(3) = 3
    lptr(3) = 4
    list(4) = -1
    lptr(4) = 3
    lend(2) = 4

    list(5) = 1
    lptr(5) = 6
    list(6) = -2
    lptr(6) = 5
    lend(3) = 6

  else
!
!  The first three nodes are collinear.
!
    ier = -2
    return
  end if
!
!  Initialize LNEW and test for N = 3.
!
  lnew = 7
  if (nn == 3) then
    ier = 0
    return
  end if
  
  near(1) = 0
  near(2) = 0
  near(3) = 0

  do k = nn, 4, -1

    d1 = ( x(k) - x(1) )**2 + ( y(k) - y(1) )**2
    d2 = ( x(k) - x(2) )**2 + ( y(k) - y(2) )**2
    d3 = ( x(k) - x(3) )**2 + ( y(k) - y(3) )**2

    if ( d1 <= d2  .and.  d1 <= d3 ) then
      near(k) = 1
      dist(k) = d1
      next(k) = near(1)
      near(1) = k
    else if (d2 <= d1  .and.  d2 <= d3) then
      near(k) = 2
      dist(k) = d2
      next(k) = near(2)
      near(2) = k
    else
      near(k) = 3
      dist(k) = d3
      next(k) = near(3)
      near(3) = k
    end if

  end do

  ncc = 0

  do k = 4, nn
    km1 = k-1
    call addnod ( k, x(k), y(k), near(k), ncc, lcc, km1, x, y, &
      list, lptr, lend, lnew, ier )

    if ( ier /= 0 ) then
      return
   
    end if
!
!  Remove K from the set of unprocessed nodes associated with NEAR(K).
!
    i = near(k)

    if (near(i) == k) then

      near(i) = next(k)

    else

      i = near(i)

      do

        i0 = i
        i = next(i0)
        if (i == k) then
          exit
        end if

      end do

      next(i0) = next(k)

    end if

    near(k) = 0
!
!  Loop on neighbors J of node K.
!
    lpl = lend(k)
    lp = lpl

4   continue

    lp = lptr(lp)
    j = abs ( list(lp) )
!
!  Loop on elements I in the sequence of unprocessed nodes
!  associated with J:  K is a candidate for replacing J
!  as the nearest triangulation node to I.  The next value
!  of I in the sequence, NEXT(I), must be saved before I
!  is moved because it is altered by adding I to K's set.
!
    i = near(j)

5   continue

    if ( i == 0 ) go to 6
    nexti = next(i)
!
!  Test for the distance from I to K less than the distance
!  from I to J.
!
    d = (x(k)-x(i))**2 + (y(k)-y(i))**2
!
!  Replace J by K as the nearest triangulation node to I:
!  update NEAR(I) and DIST(I), and remove I from J's set
!  of unprocessed nodes and add it to K's set.
!
    if ( d < dist(i) ) then
      near(i) = k
      dist(i) = d
      if (i == near(j)) then
        near(j) = nexti
      else
        next(i0) = nexti
      end if
      next(i) = near(k)
      near(k) = i
    else
      i0 = i
    end if
!
!  Bottom of loop on I.
!
    i = nexti
    go to 5
!
!  Bottom of loop on neighbors J.
!
6   continue

    if ( lp /= lpl ) then
      go to 4
    end if

  end do

  return
    end
    
subroutine trlist ( ncc, lcc, n, list, lptr, lend, nrow, nt, ltri, lct, ier )

!*****************************************************************************80
!
!! TRLIST converts a triangulation to triangle list form.
!
!  Discussion:
!
!    This subroutine converts a triangulation data structure
!    from the linked list created by subroutine TRMESH or
!    TRMSHR to a triangle list.
!
!  Modified:
!
!    16 June 2007
!
!  Author:
!
!    Robert Renka,
!    Department of Computer Science,
!    University of North Texas,
!    renka@cs.unt.edu
!
!  Reference:
!
!    Robert Renka,
!    Algorithm 751: TRIPACK,
!    A Constrained Two-Dimensional Delaunay Triangulation Package,
!    ACM Transactions on Mathematical Software,
!    Volume 22, Number 1, 1996.
!
!  Parameters:
!
!    Input, integer NCC, the number of constraints.  NCC >= 0.
!
!    Input, integer LCC(*), list of constraint curve starting indexes (or
!    dummy array of length 1 if NCC = 0).  Refer to subroutine ADDCST.
!
!    Input, integer N, the number of nodes in the triangulation.  N >= 3.
!
!    Input, integer LIST(*), LPTR(*), LEND(N), linked list data structure
!    defining the triangulation.  Refer to subroutine TRMESH.
!
!    Input, integer NROW, the number of rows (entries per triangle)
!    reserved for the triangle list LTRI.  The value must be 6 if only
!    the vertex indexes and neighboring triangle indexes are to be
!    stored, or 9 if arc indexes are also to be assigned and stored.
!    Refer to LTRI.
!
!    Input, integer LTRI(NROW*NT), where NT is at most 2N-5.  (A sufficient
!    length is 12 * N if NROW=6 or 18*N if NROW=9.)
!
!    Output, integer NT, the number of triangles in the triangulation unless
!    IER /= 0, in which case NT = 0.  NT = 2N - NB- 2, where NB is the number
!    of boundary nodes.
!
!    Output, integer LTRI(NROW,NT), whose J-th column contains the vertex nodal
!    indexes (first three rows), neighboring triangle indexes (second three
!    rows), and, if NROW = 9, arc indexes (last three rows) associated with
!    triangle J for J = 1,...,NT.  The vertices are ordered counterclockwise
!    with the first vertex taken to be the one with smallest index.  Thus,
!    LTRI(2,J) and LTRI(3,J) are larger than LTRI(1,J) and index adjacent
!    neighbors of node LTRI(1,J).  For I = 1,2,3, LTRI(I+3,J) and LTRI(I+6,J)
!    index the triangle and arc, respectively, which are opposite (not shared
!    by) node LTRI(I,J), with LTRI(I+3,J) = 0 if LTRI(I+6,J) indexes a boundary
!    arc.  Vertex indexes range from 1 to N, triangle indexes from 0 to NT,
!    and, if included, arc indexes from 1 to NA = NT+N-1.  The triangles are
!    ordered on first (smallest) vertex indexes, except that the sets of
!    constraint triangle (triangles contained in the closure of a constraint
!    region) follow the non-constraint triangles.
!
!    Output, integer LCT(NCC), containing the triangle index of the first
!    triangle of constraint J in LCT(J).  Thus, the number of non-constraint
!    triangles is LCT(1)-1, and constraint J contains LCT(J+1)-LCT(J)
!    triangles, where LCT(NCC+1) = NT+1.
!
!    Output, integer IER = Error indicator.
!    0, if no errors were encountered.
!    1, if NCC, N, NROW, or an LCC entry is outside its valid range on input.
!    2, if the triangulation data structure (LIST,LPTR,LEND) is invalid.
!
!  Local Parameters:
!
!    ARCS = TRUE iff arc indexes are to be stored.
!    KA,KT = Numbers of currently stored arcs and triangles.
!    N1ST = Starting index for the loop on nodes (N1ST = 1 on
!           pass 1, and N1ST = LCC1 on pass 2).
!    NM2 = Upper bound on candidates for N1.
!    PASS2 = TRUE iff constraint triangles are to be stored.
!
  implicit none

  integer n
  integer nrow

  logical arcs
  logical cstri
  integer i
  integer i1
  integer i2
  integer i3
  integer ier
  integer isv
  integer j
  integer jlast
  integer ka
  integer kn
  integer kt
  integer l
  integer lcc(*)
  integer lcc1
  integer lct(*)
  integer lend(n)
  integer list(*)
  integer lp
  integer lp2
  integer lpl
  integer lpln1
  integer lptr(*)
  integer ltri(nrow,*)
  integer n1
  integer n1st
  integer n2
  integer n3
  integer ncc
  integer nm2
  integer nn
  integer nt
  logical pass2
!
!  Test for invalid input parameters and store the index
!  LCC1 of the first constraint node (if any).
!
  nn = n

  if ( ncc < 0 .or. ( nrow /= 6  .and. nrow /= 9 ) ) then
    nt = 0
    ier = 1
    return
  end if

  lcc1 = nn+1

  if (ncc == 0) then

    if ( nn < 3 ) then
      nt = 0
      ier = 1
      return
    end if

  else

    do i = ncc, 1, -1
      if ( lcc1 - lcc(i) < 3 ) then
        nt = 0
        ier = 1
        return
      end if
      lcc1 = lcc(i)
    end do

    if ( lcc1 < 1 ) then
      nt = 0
      ier = 1
      return
    end if

  end if
!
!  Initialize parameters for loop on triangles KT = (N1,N2,
!  N3), where N1 < N2 and N1 < N3.  This requires two
!  passes through the nodes with all non-constraint
!  triangles stored on the first pass, and the constraint
!  triangles stored on the second.
!
  arcs = nrow == 9
  ka = 0
  kt = 0
  n1st = 1
  nm2 = nn - 2
  pass2 = .false.
!
!  Loop on nodes N1:
!  J = constraint containing N1,
!  JLAST = last node in constraint J.
!
2 continue

  j = 0
  jlast = lcc1 - 1

  do n1 = n1st, nm2

    if ( jlast < n1 ) then
!
!  N1 is the first node in constraint J+1.  Update J and
!  JLAST, and store the first constraint triangle index
!  if in pass 2.
!
      j = j + 1

      if ( j < ncc ) then
        jlast = lcc(j+1) - 1
      else
        jlast = nn
      end if

      if ( pass2 ) then
        lct(j) = kt + 1
      end if

    end if
!
!  Loop on pairs of adjacent neighbors (N2,N3).  LPLN1 points
!  to the last neighbor of N1, and LP2 points to N2.
!
    lpln1 = lend(n1)
    lp2 = lpln1

    3 continue

      lp2 = lptr(lp2)
      n2 = list(lp2)
      lp = lptr(lp2)
      n3 = abs ( list(lp) )

      if ( n2 < n1 .or. n3 < n1 ) then
        go to 10
      end if
!
!  (N1,N2,N3) is a constraint triangle iff the three nodes
!  are in the same constraint and N2 < N3.  Bypass con-
!  straint triangles on pass 1 and non-constraint triangles
!  on pass 2.
!
      cstri = n1 >= lcc1  .and.  n2 < n3  .and. n3 <= jlast

      if ( ( cstri  .and.  .not. pass2 )  .or. &
          ( .not. cstri  .and.  pass2 ) ) then
        go to 10
      end if
!
!  Add a new triangle KT = (N1,N2,N3).
!
      kt = kt + 1
      ltri(1,kt) = n1
      ltri(2,kt) = n2
      ltri(3,kt) = n3
!
!  Loop on triangle sides (I1,I2) with neighboring triangles
!  KN = (I1,I2,I3).
!
      do i = 1,3

        if ( i == 1 ) then
          i1 = n3
          i2 = n2
        else if ( i == 2 ) then
          i1 = n1
          i2 = n3
        else
          i1 = n2
          i2 = n1
        end if
!
!  Set I3 to the neighbor of I1 which follows I2 unless
!  I2->I1 is a boundary arc.
!
        lpl = lend(i1)
        lp = lptr(lpl)

4       continue

          if (list(lp) == i2) then
            go to 5
          end if

          lp = lptr(lp)

          if ( lp /= lpl ) then
            go to 4
          end if
!
!  I2 is the last neighbor of I1 unless the data structure
!  is invalid.  Bypass the search for a neighboring
!  triangle if I2->I1 is a boundary arc.
!
        if ( abs ( list(lp) ) /= i2 ) then
          go to 13
        end if

        kn = 0

        if (list(lp) < 0) then
          go to 8
        end if
!
!  I2->I1 is not a boundary arc, and LP points to I2 as
!  a neighbor of I1.
!
5   continue

        lp = lptr(lp)
        i3 = abs ( list(lp) )
!
!  Find L such that LTRI(L,KN) = I3 (not used if KN > KT),
!  and permute the vertex indexes of KN so that I1 is
!  smallest.
!
        if ( i1 < i2  .and.  i1 < i3 ) then
          l = 3
        else if (i2 < i3) then
          l = 2
          isv = i1
          i1 = i2
          i2 = i3
          i3 = isv
        else
          l = 1
          isv = i1
          i1 = i3
          i3 = i2
          i2 = isv
        end if
!
!  Test for KN > KT (triangle index not yet assigned).
!
        if ( i1 > n1  .and.  .not. pass2 ) then
          go to 9
        end if
!
!  Find KN, if it exists, by searching the triangle list in
!  reverse order.
!
        do kn = kt-1,1,-1
          if ( ltri(1,kn) == i1  .and.  ltri(2,kn) == &
              i2 .and. ltri(3,kn) == i3 ) then
            go to 7
          end if
        end do

        go to 9
!
!  Store KT as a neighbor of KN.
!
7       continue

        ltri(l+3,kn) = kt
!
!  Store KN as a neighbor of KT, and add a new arc KA.
!
8       continue

        ltri(i+3,kt) = kn

        if (arcs) then
          ka = ka + 1
          ltri(i+6,kt) = ka
          if ( kn /= 0 ) then
            ltri(l+6,kn) = ka
          end if
        end if

9       continue

    end do
!
!  Bottom of loop on triangles.
!
10  continue

    if ( lp2 /= lpln1 ) then
      go to 3
    end if

  end do
!
!  Bottom of loop on nodes.
!
  if ( .not. pass2 .and. 0 < ncc ) then
    pass2 = .true.
    n1st = lcc1
    go to 2
  end if
!
!  No errors encountered.
!
  nt = kt
  ier = 0
  return
!
!  Invalid triangulation data structure:  I1 is a neighbor of
!  I2, but I2 is not a neighbor of I1.
!
   13 continue

  nt = 0
  ier = 2

  return
end
    
subroutine circum ( x1, y1, x2, y2, x3, y3, ratio, xc, yc, cr, sa, ar )

!*****************************************************************************80
!
!! CIRCUM determines the circumcenter (and more) of a triangle.
!
!  Discussion:
!
!    Given three vertices defining a triangle, this routine
!    returns the circumcenter, circumradius, signed
!    triangle area, and, optionally, the aspect ratio of the
!    triangle.
!
!  Modified:
!
!    16 June 2007
!
!  Author:
!
!    Robert Renka,
!    Department of Computer Science,
!    University of North Texas,
!    renka@cs.unt.edu
!
!  Reference:
!
!    Robert Renka,
!    Algorithm 751: TRIPACK,
!    A Constrained Two-Dimensional Delaunay Triangulation Package,
!    ACM Transactions on Mathematical Software,
!    Volume 22, Number 1, 1996.
!
!  Parameters:
!
!    Input, real ( kind = 8 ) X1, Y1, X2, Y2, X3, Y3, the coordinates of
!    the vertices.
!
!    Input, logical RATIO, is TRUE if and only if the aspect ratio is
!    to be computed.
!
!    Output, real ( kind = 8 ) XC, YC, coordinates of the circumcenter (center
!    of the circle defined by the three points) unless SA = 0, in which XC
!    and YC are not altered.
!
!    Output, real ( kind = 8 ) CR, the circumradius (radius of the circle
!    defined by the three points) unless SA = 0 (infinite radius), in which
!    case CR is not altered.
!
!    Output, real ( kind = 8 ) SA, the signed triangle area with positive value
!    if and only if the vertices are specified in counterclockwise order:
!    (X3,Y3) is strictly to the left of the directed line from (X1,Y1)
!    toward (X2,Y2).
!
!    Output, real ( kind = 8 ) AR, the aspect ratio r/CR, where r is the
!    radius of the inscribed circle, unless RATIO = FALSE, in which case AR
!    is not altered.  AR is in the range 0 to 0.5, with value 0 iff SA = 0 and
!    value 0.5 iff the vertices define an equilateral triangle.
!
  implicit none

  real ( kind = 8 ) ar
  real ( kind = 8 ) cr
  real ( kind = 8 ) ds(3)
  real ( kind = 8 ) fx
  real ( kind = 8 ) fy
  logical ratio
  real ( kind = 8 ) sa
  real ( kind = 8 ) u(3)
  real ( kind = 8 ) v(3)
  real ( kind = 8 ) x1
  real ( kind = 8 ) x2
  real ( kind = 8 ) x3
  real ( kind = 8 ) xc
  real ( kind = 8 ) y1
  real ( kind = 8 ) y2
  real ( kind = 8 ) y3
  real ( kind = 8 ) yc
!
!  Set U(K) and V(K) to the x and y components, respectively,
!  of the directed edge opposite vertex K.
!
  u(1) = x3 - x2
  u(2) = x1 - x3
  u(3) = x2 - x1
  v(1) = y3 - y2
  v(2) = y1 - y3
  v(3) = y2 - y1
!
!  Set SA to the signed triangle area.
!
  sa = ( u(1) * v(2) - u(2) * v(1) ) / 2.0D+00

  if ( sa == 0.0D+00 ) then
    if ( ratio ) then
      ar = 0.0D+00
    end if
    return
  end if
!
!  Set DS(K) to the squared distance from the origin to vertex K.
!
  ds(1) = x1 * x1 + y1 * y1
  ds(2) = x2 * x2 + y2 * y2
  ds(3) = x3 * x3 + y3 * y3
!
!  Compute factors of XC and YC.
!
  fx = - dot_product ( ds(1:3), v(1:3) )
  fy =   dot_product ( ds(1:3), u(1:3) )

  xc = fx / ( 4.0D+00 * sa )
  yc = fy / ( 4.0D+00 * sa )
  cr = sqrt ( ( xc - x1 )**2 + ( yc - y1 )**2 )

  if ( .not. ratio ) then
    return
  end if
!
!  Compute the squared edge lengths and aspect ratio.
!
  ds(1:3) = u(1:3)**2 + v(1:3)**2

  ar = 2.0D+00 * abs ( sa ) / &
       ( ( sqrt ( ds(1) ) + sqrt ( ds(2) ) + sqrt ( ds(3) ) ) * cr )

  return
end
    

subroutine addnod ( k, xk, yk, ist, ncc, lcc, n, x, y, list, lptr, lend, lnew, &
  ier )

  implicit none

  logical crtri
  integer i
  integer i1
  integer i2
  integer i3
  integer ibk
  integer ier
  integer in1
  integer indxcc
  integer io1
  integer io2
  integer ist
  integer k
  integer kk
  integer l
  integer lcc(*)
  integer lccip1
  integer lend(*)
  integer list(*)
  integer lnew
  integer lp
  integer lpf
  integer lpo1
  integer lptr(*)
  integer lstptr
  integer n
  integer ncc
  integer nm1
  logical swptst
  real ( kind = 8 ) x(*)
  real ( kind = 8 ) xk
  real ( kind = 8 ) y(*)
  real ( kind = 8 ) yk

  kk = k
!
!  Test for an invalid input parameter.
!
  if ( kk < 1  .or.  ist < 1  .or.  n < ist &
      .or.  ncc < 0  .or.  n < 3 ) then
    ier = -1
    return
  end if

  lccip1 = n + 1

  do i = ncc, 1, -1
    if ( lccip1-lcc(i) < 3 ) then
      ier = -1
      return
    end if
    lccip1 = lcc(i)
  end do

  if ( lccip1 < kk ) then
    ier = -1
    return
  end if
!
!  Find a triangle (I1,I2,I3) containing K or the rightmost
!  (I1) and leftmost (I2) visible boundary nodes as viewed from node K.
!
  call trfind ( ist, xk, yk, n, x, y, list, lptr, lend, i1, i2, i3 )
!
!  Test for collinear nodes, duplicate nodes, and K lying in
!  a constraint region.
!
  if ( i1 == 0 ) then
    ier = -2
    return
  end if

  if ( i3 /= 0 ) then

    l = i1
    if ( xk == x(l)  .and.  yk == y(l) ) then
      ier = l
      return
    end if

    l = i2
    if ( xk == x(l)  .and.  yk == y(l) ) then
      ier = l
      return
    end if

    l = i3
    if ( xk == x(l)  .and.  yk == y(l) ) then
      ier = l
      return
    end if

    if ( 0 < ncc .and.  crtri(ncc,lcc,i1,i2,i3) ) then
      ier = -3
      return
    end if

  else
!
!  K is outside the convex hull of the nodes and lies in a
!  constraint region iff an exterior constraint curve is present.
!
    if ( 0 < ncc .and. indxcc(ncc,lcc,n,list,lend) /= 0 ) then
      ier = -3
      return
    end if

  end if
!
!  No errors encountered.
!
  ier = 0
  nm1 = n
  n = n + 1

  if (kk < n) then
!
!  Open a slot for K in X, Y, and LEND, and increment all
!  nodal indexes which are greater than or equal to K.
!
!  Note that LIST, LPTR, and LNEW are not yet updated with
!  either the neighbors of K or the edges terminating on K.
!
    do ibk = nm1, kk, -1
      x(ibk+1) = x(ibk)
      y(ibk+1) = y(ibk)
      lend(ibk+1) = lend(ibk)
    end do

    do i = 1, ncc
      lcc(i) = lcc(i) + 1
    end do

    l = lnew - 1

    do i = 1, l

      if ( kk <= list(i) ) then
        list(i) = list(i) + 1
      end if

      if ( list(i) <= -kk ) then
        list(i) = list(i) - 1
      end if

    end do

    if ( kk <= i1 ) then
      i1 = i1 + 1
    end if

    if ( kk <= i2 ) then
      i2 = i2 + 1
    end if

    if ( kk <= i3 ) then
      i3 = i3 + 1
    end if

  end if
!
!  Insert K into X and Y, and update LIST, LPTR, LEND, and
!  LNEW with the arcs containing node K.
!
  x(kk) = xk
  y(kk) = yk

  if ( i3 == 0 ) then
    call bdyadd ( kk, i1, i2, list, lptr, lend, lnew )
  else
    call intadd ( kk, i1, i2, i3, list, lptr, lend, lnew )
  end if
!
!  Initialize variables for optimization of the triangulation.
!
  lp = lend(kk)
  lpf = lptr(lp)
  io2 = list(lpf)
  lpo1 = lptr(lpf)
  io1 = abs ( list(lpo1) )
!
!  Begin loop:  find the node opposite K.
!
  do

    lp = lstptr ( lend(io1), io2, list, lptr )

    if ( 0 <= list(lp) ) then

      lp = lptr(lp)
      in1 = abs ( list(lp) )
!
!  Swap test:  if a swap occurs, two new arcs are
!  opposite K and must be tested.
!
      if ( .not. crtri ( ncc, lcc, io1, io2, in1 ) ) then

        if ( swptst(in1,kk,io1,io2,x,y) ) then

          call swap ( in1, kk, io1, io2, list, lptr, lend, lpo1 )

          if ( lpo1 == 0 ) then
            ier = -4
            exit
          end if

          io1 = in1

          cycle

        end if

      end if

    end if
!
!  No swap occurred.  Test for termination and reset IO2 and IO1.
!
    if ( lpo1 == lpf .or. list(lpo1) < 0 ) then
      exit
    end if

    io2 = io1
    lpo1 = lptr(lpo1)
    io1 = abs ( list(lpo1) )

  end do

  return
end


subroutine trfind ( nst, px, py, n, x, y, list, lptr, lend, i1, i2, i3 )

!*****************************************************************************80
!
!! TRFIND locates a point relative to a triangulation.
!
!  Discussion:
!
!    This subroutine locates a point P relative to a triangu-
!    lation created by subroutine TRMESH or TRMSHR.  If P is
!    contained in a triangle, the three vertex indexes are
!    returned.  Otherwise, the indexes of the rightmost and
!    leftmost visible boundary nodes are returned.
!
!  Modified:
!
!    16 June 2007
!
!  Author:
!
!    Robert Renka,
!    Department of Computer Science,
!    University of North Texas,
!    renka@cs.unt.edu
!
!  Reference:
!
!    Robert Renka,
!    Algorithm 751: TRIPACK,
!    A Constrained Two-Dimensional Delaunay Triangulation Package,
!    ACM Transactions on Mathematical Software,
!    Volume 22, Number 1, 1996.
!
!  Parameters:
!
!    Input, integer NST, the index of a node at which TRFIND begins the
!    search.  Search time depends on the proximity of this node to P.
!
!    Input, real ( kind = 8 ) PX, PY, the coordinates of the point P to be
!    located.
!
!    Input, integer N, the number of nodes in the triangulation.  3 <= N.
!
!    Input, real ( kind = 8 ) X(N), Y(N), the coordinates of the nodes in
!    the triangulation.
!
!    Input, integer LIST(*), LPTR(*), LEND(N), the data structure defining
!    the triangulation.  Refer to subroutine TRMESH.
!
!    Output, integer I1, I2, I3, nodal indexes, in counterclockwise order,
!    of the vertices of a triangle containing P if P is contained in a
!    triangle.  If P is not in the convex hull of the nodes, I1 indexes
!    the rightmost visible boundary node, I2 indexes the leftmost visible
!    boundary node, and I3 = 0.  Rightmost and leftmost are defined from
!    the perspective of P, and a pair of points are visible from each
!    other if and only if the line segment joining them intersects no
!    triangulation arc.  If P and all of the nodes lie on a common line,
!    then I1 = I2 = I3 = 0 on output.
!
!  Local parameters:
!
!    B1,B2 =    Unnormalized barycentric coordinates of P with respect
!               to (N1,N2,N3)
!    IX,IY,IZ = Integer seeds for JRAND
!    LP =       LIST pointer
!    N0,N1,N2 = Nodes in counterclockwise order defining a
!               cone (with vertex N0) containing P
!    N1S,N2S =  Saved values of N1 and N2
!    N3,N4 =    Nodes opposite N1->N2 and N2->N1, respectively
!    NB =       Index of a boundary node -- first neighbor of
!               NF or last neighbor of NL in the boundary traversal loops
!    NF,NL =    First and last neighbors of N0, or first
!               (rightmost) and last (leftmost) nodes
!               visible from P when P is exterior to the triangulation
!    NP,NPP =   Indexes of boundary nodes used in the boundary traversal loops
!    XA,XB,XC = Dummy arguments for FRWRD
!    YA,YB,YC = Dummy arguments for FRWRD
!    XP,YP =    Local variables containing the components of P
!
  implicit none

  integer n

  real ( kind = 8 ) b1
  real ( kind = 8 ) b2
  logical frwrd
  integer i1
  integer i2
  integer i3
  integer, save :: ix = 1
  integer, save :: iy = 2
  integer, save :: iz = 3
  integer jrand
  logical left
  integer lend(n)
  integer list(*)
  integer lp
  integer lptr(*)
  integer lstptr
  integer n0
  integer n1
  integer n1s
  integer n2
  integer n2s
  integer n3
  integer n4
  integer nb
  integer nf
  integer nl
  integer np
  integer npp
  integer nst
  real ( kind = 8 ) px
  real ( kind = 8 ) py
  real ( kind = 8 ) store
  real ( kind = 8 ) x(n)
  real ( kind = 8 ) xa
  real ( kind = 8 ) xb
  real ( kind = 8 ) xc
  real ( kind = 8 ) xp
  real ( kind = 8 ) y(n)
  real ( kind = 8 ) ya
  real ( kind = 8 ) yb
  real ( kind = 8 ) yc
  real ( kind = 8 ) yp
!
!  Statement function:
!
!  FRWRD = TRUE iff C is forward of A->B iff <A->B,A->C> >= 0.
!
  frwrd(xa,ya,xb,yb,xc,yc) = (xb-xa)*(xc-xa) + (yb-ya)*(yc-ya) >= 0.0D+00
!
!  Initialize variables.
!
  xp = px
  yp = py
  n0 = nst

  if ( n0 < 1  .or.  n < n0 ) then
    n0 = jrand ( n, ix, iy, iz )
  end if
!
!  Set NF and NL to the first and last neighbors of N0, and
!  initialize N1 = NF.
!
1 continue

  lp = lend(n0)
  nl = list(lp)
  lp = lptr(lp)
  nf = list(lp)
  n1 = nf
!
!  Find a pair of adjacent neighbors N1,N2 of N0 that define
!  a wedge containing P:  P LEFT N0->N1 and P RIGHT N0->N2.
!
  if ( 0 < nl ) then
    go to 2
  end if
!
!   N0 is a boundary node.  Test for P exterior.
!
  nl = -nl

  if ( .not. left ( x(n0), y(n0), x(nf), y(nf), xp, yp ) ) then
    nl = n0
    go to 9
  end if

  if ( .not. left(x(nl),y(nl),x(n0),y(n0),xp,yp) ) then
    nb = nf
    nf = n0
    np = nl
    npp = n0
    go to 11
  end if

  go to 3
!
!  N0 is an interior node.  Find N1.
!
2 continue

    do

      if ( left(x(n0),y(n0),x(n1),y(n1),xp,yp) ) then
        exit
      end if

      lp = lptr(lp)
      n1 = list(lp)

      if ( n1 == nl ) then
        go to 6
      end if

    end do
!
!  P is to the left of edge N0->N1.  Initialize N2 to the
!  next neighbor of N0.
!
3 continue

    lp = lptr(lp)
    n2 = abs ( list(lp) )

    if ( .not. left(x(n0),y(n0),x(n2),y(n2),xp,yp) ) then
      go to 7
    end if

    n1 = n2
    if ( n1 /= nl ) then
      go to 3
    end if

  if ( .not. left(x(n0),y(n0),x(nf),y(nf),xp,yp) ) then
    go to 6
  end if

  if (xp == x(n0) .and. yp == y(n0)) then
    go to 5
  end if
!
!  P is left of or on edges N0->NB for all neighbors NB of N0.
!  All points are collinear iff P is left of NB->N0 for
!  all neighbors NB of N0.  Search the neighbors of N0.
!  NOTE: N1 = NL and LP points to NL.
!
4   continue

    if ( .not. left(x(n1),y(n1),x(n0),y(n0),xp,yp) ) then
      go to 5
    end if

    lp = lptr(lp)
    n1 = abs ( list(lp) )

    if ( n1 == nl ) then
      i1 = 0
      i2 = 0
      i3 = 0
      return
    end if

    go to 4
!
!  P is to the right of N1->N0, or P=N0.  Set N0 to N1 and start over.
!
5 continue

  n0 = n1
  go to 1
!
!  P is between edges N0->N1 and N0->NF.
!
6 continue

  n2 = nf
!
!  P is contained in the wedge defined by line segments
!  N0->N1 and N0->N2, where N1 is adjacent to N2.  Set
!  N3 to the node opposite N1->N2, and save N1 and N2 to
!  test for cycling.
!
7 continue

  n3 = n0
  n1s = n1
  n2s = n2
!
!  Top of edge hopping loop.  Test for termination.
!
8 continue

  if ( left ( x(n1), y(n1), x(n2), y(n2), xp, yp ) ) then
!
!  P LEFT N1->N2 and hence P is in (N1,N2,N3) unless an
!  error resulted from floating point inaccuracy and
!  collinearity.  Compute the unnormalized barycentric
!  coordinates of P with respect to (N1,N2,N3).
!
    b1 = (x(n3)-x(n2))*(yp-y(n2)) - (xp-x(n2))*(y(n3)-y(n2))
    b2 = (x(n1)-x(n3))*(yp-y(n3)) - (xp-x(n3))*(y(n1)-y(n3))

    if ( store ( b1 + 1.0D+00 ) >= 1.0D+00  .and. &
         store ( b2 + 1.0D+00 ) >= 1.0D+00 ) then
      go to 16
    end if
!
!  Restart with N0 randomly selected.
!
    n0 = jrand ( n, ix, iy, iz )
    go to 1

  end if
!
!  Set N4 to the neighbor of N2 which follows N1 (node
!  opposite N2->N1) unless N1->N2 is a boundary edge.
!
  lp = lstptr(lend(n2),n1,list,lptr)

  if ( list(lp) < 0 ) then
    nf = n2
    nl = n1
    go to 9
  end if

  lp = lptr(lp)
  n4 = abs ( list(lp) )
!
!  Select the new edge N1->N2 which intersects the line
!  segment N0-P, and set N3 to the node opposite N1->N2.
!
  if ( left(x(n0),y(n0),x(n4),y(n4),xp,yp) ) then
    n3 = n1
    n1 = n4
    n2s = n2
    if (n1 /= n1s  .and.  n1 /= n0) go to 8
  else
    n3 = n2
    n2 = n4
    n1s = n1
    if ( n2 /= n2s  .and.  n2 /= n0 ) then
      go to 8
    end if
  end if
!
!  The starting node N0 or edge N1-N2 was encountered
!  again, implying a cycle (infinite loop).  Restart
!  with N0 randomly selected.
!
  n0 = jrand ( n, ix, iy, iz )
  go to 1
!
!  Boundary traversal loops.  NL->NF is a boundary edge and
!  P RIGHT NL->NF.  Save NL and NF.

9 continue

  np = nl
  npp = nf
!
!  Find the first (rightmost) visible boundary node NF.  NB
!  is set to the first neighbor of NF, and NP is the last neighbor.
!
10 continue

  lp = lend(nf)
  lp = lptr(lp)
  nb = list(lp)

  if ( .not. left(x(nf),y(nf),x(nb),y(nb),xp,yp) ) then
    go to 12
  end if
!
!  P LEFT NF->NB and thus NB is not visible unless an error
!  resulted from floating point inaccuracy and collinear-
!  ity of the 4 points NP, NF, NB, and P.
!
11 continue

  if ( frwrd(x(nf),y(nf),x(np),y(np),xp,yp)  .or. &
       frwrd(x(nf),y(nf),x(np),y(np),x(nb),y(nb)) ) then
    i1 = nf
    go to 13
  end if
!
!  Bottom of loop.
!
12 continue

  np = nf
  nf = nb
  go to 10
!
!  Find the last (leftmost) visible boundary node NL.  NB
!  is set to the last neighbor of NL, and NPP is the first
!  neighbor.
!
13 continue

  lp = lend(nl)
  nb = -list(lp)

  if ( .not. left(x(nb),y(nb),x(nl),y(nl),xp,yp) ) then
    go to 14
  end if
!
!  P LEFT NB->NL and thus NB is not visible unless an error
!  resulted from floating point inaccuracy and collinear-
!  ity of the 4 points P, NB, NL, and NPP.
!
  if ( frwrd(x(nl),y(nl),x(npp),y(npp),xp,yp)  .or. &
       frwrd(x(nl),y(nl),x(npp),y(npp),x(nb),y(nb)) ) then
    go to 15
  end if
!
!  Bottom of loop.
!
14 continue

  npp = nl
  nl = nb
  go to 13
!
!  NL is the leftmost visible boundary node.
!
15 continue

  i2 = nl
  i3 = 0
  return
!
!  P is in the triangle (N1,N2,N3).
!
16 continue

  i1 = n1
  i2 = n2
  i3 = n3

  return
end

subroutine bdyadd ( kk, i1, i2, list, lptr, lend, lnew )

!*****************************************************************************80
!
!! BDYADD adds a boundary node to a triangulation.
!
!  Discussion:
!
!    This subroutine adds a boundary node to a triangulation
!    of a set of points in the plane.  The data structure is
!    updated with the insertion of node KK, but no optimization
!    is performed.
!
!  Modified:
!
!    16 June 2007
!
!  Author:
!
!    Robert Renka,
!    Department of Computer Science,
!    University of North Texas,
!    renka@cs.unt.edu
!
!  Reference:
!
!    Robert Renka,
!    Algorithm 751: TRIPACK,
!    A Constrained Two-Dimensional Delaunay Triangulation Package,
!    ACM Transactions on Mathematical Software,
!    Volume 22, Number 1, 1996.
!
!  Parameters:
!
!    Input, integer KK, the index of a node to be connected to the sequence
!    of all visible boundary nodes.  1 <= KK and
!    KK must not be equal to I1 or I2.
!
!    Input, integer I1, the first (rightmost as viewed from KK) boundary
!    node in the triangulation which is visible from
!    node KK (the line segment KK-I1 intersects no arcs.
!
!    Input, integer I2, the last (leftmost) boundary node which is visible
!    from node KK.  I1 and I2 may be determined by subroutine TRFIND.
!
!    Input/output, integer LIST(*), LPTR(*), LEND(N), LNEW.  The
!    triangulation data structure created by TRMESH or TRMSHR.
!    On input, nodes I1 and I2 must be included in the triangulation.
!    On output, the data structure has been updated with the addition
!    of node KK.  Node KK is connected to I1, I2, and all boundary
!    nodes in between.
!
  implicit none

  integer i1
  integer i2
  integer k
  integer kk
  integer lend(*)
  integer list(*)
  integer lnew
  integer lp
  integer lptr(*)
  integer lsav
  integer n1
  integer n2
  integer next
  integer nsav

  k = kk
  n1 = i1
  n2 = i2
!
!  Add K as the last neighbor of N1.
!
  lp = lend(n1)
  lsav = lptr(lp)
  lptr(lp) = lnew
  list(lnew) = -k
  lptr(lnew) = lsav
  lend(n1) = lnew
  lnew = lnew + 1
  next = -list(lp)
  list(lp) = next
  nsav = next
!
!  Loop on the remaining boundary nodes between N1 and N2,
!  adding K as the first neighbor.
!
  do

    lp = lend(next)

    call insert ( k, lp, list, lptr, lnew )

    if ( next == n2 ) then
      exit
    end if

    next = -list(lp)
    list(lp) = next

  end do
!
!  Add the boundary nodes between N1 and N2 as neighbors
!  of node K.
!
  lsav = lnew
  list(lnew) = n1
  lptr(lnew) = lnew + 1
  lnew = lnew + 1
  next = nsav

  do

    if ( next == n2 ) then
      exit
    end if

    list(lnew) = next
    lptr(lnew) = lnew + 1
    lnew = lnew + 1
    lp = lend(next)
    next = list(lp)

  end do

  list(lnew) = -n2
  lptr(lnew) = lsav
  lend(k) = lnew
  lnew = lnew + 1

  return
end
    
subroutine intadd ( kk, i1, i2, i3, list, lptr, lend, lnew )

!*****************************************************************************80
!
!! INTADD adds an interior point to a triangulation.
!
!  Discussion:
!
!    This subroutine adds an interior node to a triangulation
!    of a set of points in the plane.  The data structure is
!    updated with the insertion of node KK into the triangle
!    whose vertices are I1, I2, and I3.  No optimization of the
!    triangulation is performed.
!
!  Modified:
!
!    16 June 2007
!
!  Author:
!
!    Robert Renka,
!    Department of Computer Science,
!    University of North Texas,
!    renka@cs.unt.edu
!
!  Reference:
!
!    Robert Renka,
!    Algorithm 751: TRIPACK,
!    A Constrained Two-Dimensional Delaunay Triangulation Package,
!    ACM Transactions on Mathematical Software,
!    Volume 22, Number 1, 1996.
!
!  Parameters:
!
!    Input, integer KK, the index of the node to be inserted.  1 <= KK
!    and KK must not be equal to I1, I2, or I3.
!
!    Input, integer I1, I2, I3, indexes of the counterclockwise-ordered
!    sequence of vertices of a triangle which contains node KK.
!
!    Input/output, integer LIST(*), LPTR(*), LEND(N), LNEW, the data
!    structure defining the triangulation.  Refer to subroutine TRMESH.
!    Triangle (I1,I2,I3) must be included in the triangulation.
!    On output, updated with the addition of node KK.  KK
!    will be connected to nodes I1, I2, and I3.
!
  implicit none

  integer i1
  integer i2
  integer i3
  integer k
  integer kk
  integer lend(*)
  integer list(*)
  integer lnew
  integer lp
  integer lptr(*)
  integer lstptr
  integer n1
  integer n2
  integer n3

  k = kk
!
!  Initialization.
!
  n1 = i1
  n2 = i2
  n3 = i3
!
!  Add K as a neighbor of I1, I2, and I3.
!
  lp = lstptr(lend(n1),n2,list,lptr)
  call insert (k,lp,list,lptr,lnew)
  lp = lstptr(lend(n2),n3,list,lptr)
  call insert (k,lp,list,lptr,lnew)
  lp = lstptr(lend(n3),n1,list,lptr)
  call insert (k,lp,list,lptr,lnew)
!
!  Add I1, I2, and I3 as neighbors of K.
!
  list(lnew) = n1
  list(lnew+1) = n2
  list(lnew+2) = n3
  lptr(lnew) = lnew + 1
  lptr(lnew+1) = lnew + 2
  lptr(lnew+2) = lnew
  lend(k) = lnew + 2
  lnew = lnew + 3

  return
end
    
subroutine swap ( in1, in2, io1, io2, list, lptr, lend, lp21 )

!*****************************************************************************80
!
!! SWAP adjusts a triangulation by swapping a diagonal arc.
!
!  Discussion:
!
!    Given a triangulation of a set of points on the unit
!    sphere, this subroutine replaces a diagonal arc in a
!    strictly convex quadrilateral (defined by a pair of adja-
!    cent triangles) with the other diagonal.  Equivalently, a
!    pair of adjacent triangles is replaced by another pair
!    having the same union.
!
!  Modified:
!
!    16 June 2007
!
!  Author:
!
!    Robert Renka,
!    Department of Computer Science,
!    University of North Texas,
!    renka@cs.unt.edu
!
!  Reference:
!
!    Robert Renka,
!    Algorithm 751: TRIPACK,
!    A Constrained Two-Dimensional Delaunay Triangulation Package,
!    ACM Transactions on Mathematical Software,
!    Volume 22, Number 1, 1996.
!
!  Parameters:
!
!    Input, integer IN1, IN2, IO1, IO2, the nodal indexes of the vertices of
!    the quadrilateral.  IO1-IO2 is replaced by IN1-IN2.  (IO1,IO2,IN1)
!    and (IO2,IO1,IN2) must be triangles on input.
!
!    Input/output, integer LIST(*), LPTR(*), LEND(N), the data structure
!    defining the triangulation.  Refer to subroutine TRMESH.  On output,
!    updated with the swap; triangles (IO1,IO2,IN1) and (IO2,IO1,IN2) are
!    replaced by (IN1,IN2,IO2) and (IN2,IN1,IO1) unless LP21 = 0.
!
!    Output, integer LP21, the index of IN1 as a neighbor of IN2 after the
!    swap is performed unless IN1 and IN2 are adjacent on input, in which
!    case LP21 = 0.
!
!  Local parameters:
!
!    LP, LPH, LPSAV = LIST pointers
!
  implicit none

  integer in1
  integer in2
  integer io1
  integer io2
  integer lend(*)
  integer list(*)
  integer lp
  integer lp21
  integer lph
  integer lpsav
  integer lptr(*)
  integer lstptr
!
!  Test for IN1 and IN2 adjacent.
!
  lp = lstptr(lend(in1),in2,list,lptr)

  if ( abs ( list(lp) ) == in2 ) then
    lp21 = 0
    return
  end if
!
!  Delete IO2 as a neighbor of IO1.
!
  lp = lstptr(lend(io1),in2,list,lptr)
  lph = lptr(lp)
  lptr(lp) = lptr(lph)
!
!  If IO2 is the last neighbor of IO1, make IN2 the last neighbor.
!
  if ( lend(io1) == lph ) then
    lend(io1) = lp
  end if
!
!  Insert IN2 as a neighbor of IN1 following IO1
!  using the hole created above.
!
  lp = lstptr(lend(in1),io1,list,lptr)
  lpsav = lptr(lp)
  lptr(lp) = lph
  list(lph) = in2
  lptr(lph) = lpsav
!
!  Delete IO1 as a neighbor of IO2.
!
  lp = lstptr(lend(io2),in1,list,lptr)
  lph = lptr(lp)
  lptr(lp) = lptr(lph)
!
!  If IO1 is the last neighbor of IO2, make IN1 the last neighbor.
!
  if ( lend(io2) == lph ) then
    lend(io2) = lp
  end if
!
!  Insert IN1 as a neighbor of IN2 following IO2.
!
  lp = lstptr(lend(in2),io2,list,lptr)
  lpsav = lptr(lp)
  lptr(lp) = lph
  list(lph) = in1
  lptr(lph) = lpsav
  lp21 = lph

  return
end
    
subroutine insert ( k, lp, list, lptr, lnew )

!*****************************************************************************80
!
!! INSERT inserts K as a neighbor of N1.
!
!  Discussion:
!
!    This subroutine inserts K as a neighbor of N1 following
!    N2, where LP is the LIST pointer of N2 as a neighbor of
!    N1.  Note that, if N2 is the last neighbor of N1, K will
!    become the first neighbor (even if N1 is a boundary node).
!
!  Modified:
!
!    16 June 2007
!
!  Author:
!
!    Robert Renka,
!    Department of Computer Science,
!    University of North Texas,
!    renka@cs.unt.edu
!
!  Reference:
!
!    Robert Renka,
!    Algorithm 751: TRIPACK,
!    A Constrained Two-Dimensional Delaunay Triangulation Package,
!    ACM Transactions on Mathematical Software,
!    Volume 22, Number 1, 1996.
!
!  Parameters:
!
!    Input, integer K, the index of the node to be inserted.
!
!    Input, integer LP, the LIST pointer of N2 as a neighbor of N1.
!
!    Input/output, integer LIST(*), LPTR(*), LNEW, the data structure
!    defining the triangulation.  Refer to subroutine TRMESH.  On output,
!    the data structure has been updated to include node K.
!
  implicit none

  integer k
  integer list(*)
  integer lnew
  integer lp
  integer lptr(*)
  integer lsav

  lsav = lptr(lp)
  lptr(lp) = lnew
  list(lnew) = k
  lptr(lnew) = lsav
  lnew = lnew + 1

  return
end
    
subroutine AddRemNodesLocal
use fluid_module
use fluid_module_dt
use solution_module
use sediment_module
implicit none

integer :: i1, j1, n(3), k1, t1, remove,p1, elem_scelto,m1,check_addrem,count_nodes, add_counter,s1,q1, check_error, confirm, skip
real(8) :: xc, yc, radius, gamma_min,area, l1,l2,l3, rad_in
real(8) :: L(3), dL(3,2), jac,jac1,jac2,jac3
real(8) :: xl, yl, x(3), y(3),ar, gamma_temp(3),Xg,Yg
integer, allocatable :: nl(:), elem_concorr(:), elem_concorr_temp(:)
real(8), allocatable :: dist(:) , move_nodes(:)

area=0.d0
elem_scelto=0
area_m=0.d0
check_addrem=0
count_nodes=0
add_counter=0
allocate(move_nodes(npoints))
move_nodes=0
do i1=1,nelement
    n(1)=elements(i1)%nodes(1)
    n(2)=elements(i1)%nodes(2)
    n(3)=elements(i1)%nodes(3)

    call triangle_area(node(n(1))%coord(1),node(n(1))%coord(2),node(n(2))%coord(1), &
    node(n(2))%coord(2),node(n(3))%coord(1),node(n(3))%coord(2),area)

    area_m=area_m+area 
end do
area_m=area_m/nelement
radius= sqrt(2*area_m)/2

do k1=1,nelement
    n=elements(k1)%nodes
    
    skip=0
    if(node(n(1))%euler==1 .or. node(n(2))%euler==1 .or. node(n(3))%euler==1) then
        skip=1
    end if
    
    if (problem_type == 3) then
        if(node(n(1))%bed==1 .or. node(n(2))%bed==1 .or. node(n(3))%bed==1) then
            skip=1
        end if
    endif 
    
    
    if (skip==0) then 
        do i1=1,3
            x(i1)=node(n(i1))%coord(1)
            y(i1)=node(n(i1))%coord(2)
        end do
        call triangle_area(x(1),y(1),x(2),y(2),x(3),y(3),jac)
        
        l1= sqrt( (x(1)-x(2))**2+(y(1)-y(2))**2 )
        l2= sqrt( (x(1)-x(3))**2+(y(1)-y(3))**2 )
        l3= sqrt( (x(3)-x(2))**2+(y(3)-y(2))**2 )

        rad_in=jac/(l1+l2+l3) 
        if (2*rad_in<0.5d0*radius .or. 2*rad_in>1.5d0*radius)then
            
            if (l1<l3 .and. l2<l3 .and. node(n(1))%free_surf==0 .and. node(n(1))%bound==0 .and. node(n(1))%euler==0)then
                
                confirm=1
                do i1=1,add_counter
                    if(move_nodes(i1)==n(1))then
                        confirm=0
                    end if
                end do
                if(confirm==1)then
                    add_counter=add_counter+1
                    move_nodes(add_counter)=n(1)
                end if
           
            elseif (l1<l2 .and. l3<l2 .and. node(n(2))%free_surf==0 .and. node(n(2))%bound==0.and. node(n(2))%euler==0)then
                
                confirm=1
                do i1=1,add_counter
                    if(move_nodes(i1)==n(2))then
                        confirm=0
                    end if
                end do
                
                if(confirm==1)then
                    add_counter=add_counter+1
                    move_nodes(add_counter)=n(2)
                end if
            
                
            elseif (l2<l1 .and. l3<l1 .and. node(n(3))%free_surf==0 .and. node(n(3))%bound==0 .and. node(n(3))%euler==0)then
                
                confirm=1
                do i1=1,add_counter
                    if(move_nodes(i1)==n(3))then
                        confirm=0
                    end if
                end do
                
                if(confirm==1)then
                    add_counter=add_counter+1
                    move_nodes(add_counter)=n(3)
                end if
                
                elseif(node(n(1))%free_surf==0 .and. node(n(1))%bound==0 .and. node(n(1))%euler==0)then
        
                confirm=1
                do i1=1,add_counter
                    if(move_nodes(i1)==n(1))then
                        confirm=0
                    end if
                end do
                
                if(confirm==1)then
                    add_counter=add_counter+1
                    move_nodes(add_counter)=n(1)
                end if
                
                
            elseif(node(n(2))%free_surf==0 .and. node(n(2))%bound==0 .and. node(n(2))%euler==0)then
                                       
                confirm=1
                do i1=1,add_counter
                    if(move_nodes(i1)==n(2))then
                        confirm=0
                    end if
                end do
                
                if(confirm==1)then
                    add_counter=add_counter+1
                    move_nodes(add_counter)=n(2)  
                end if  
                
                
            elseif(node(n(3))%free_surf==0 .and. node(n(3))%bound==0 .and. node(n(3))%euler==0)then
                
                confirm=1
                do i1=1,add_counter
                    if(move_nodes(i1)==n(3))then
                        confirm=0
                    end if
                end do
                if(confirm==1)then
                    add_counter=add_counter+1
                    move_nodes(add_counter)=n(3)
                end if
                
            end if
        end if
    end if
end do
     
    
do s1=1,add_counter
    i1=move_nodes(s1) 
    xc=node(i1)%coord(1)
    yc=node(i1)%coord(2)
    q1=node(i1)%neighb(1)                                         
    allocate (nl(q1))  
    check_error=0
    do j1=2,q1+1           
        nl(j1-1) = node(i1)%neighb(j1)
        if(node(i1)%neighb(j1)<0)then
            check_error=1
            write(*,*)'Error'
            pause
        end if
    end do
    if(check_error==0)then
        do j1=1,q1
            xl=node(nl(j1))%coord(1)
            yl=node(nl(j1))%coord(2) 
        end do
        check_addrem=1
        count_nodes=count_nodes+1
        elem_scelto=0                                                       
        p1=0
        
        allocate(elem_concorr_temp(500))
        elem_concorr_temp=0
        do k1=1,nelement
            do m1=1,3
                if (elements(k1)%nodes(m1)==i1) then
                    p1=p1+1     
                    elem_concorr_temp(p1)=k1
                end if
            end do
        end do
        allocate(elem_concorr(p1))
        do k1=1,p1
            elem_concorr(k1)=elem_concorr_temp(k1)
        end do
        deallocate (elem_concorr_temp)
        Xg=0.d0
        Yg=0.d0
        do k1=1,q1
            Xg=Xg+(node(nl(k1))%coord(1))/q1
            Yg=Yg+(node(nl(k1))%coord(2))/q1
        end do
        do  k1=1,p1
            n=elements(elem_concorr(k1))%nodes
            do j1=1,3
                x(j1)=node(n(j1))%coord(1)
                y(j1)=node(n(j1))%coord(2)
            end do
                 
            call triangle_area(x(1),y(1),x(2),y(2),x(3),y(3),jac) 
            call triangle_area(Xg,Yg,x(2),y(2),x(3),y(3),jac1)
            call triangle_area(x(1),y(1),Xg,Yg,x(3),y(3),jac2)
            call triangle_area(x(1),y(1),x(2),y(2),Xg,Yg,jac3)
                 
            if (jac1+jac2+jac3<=jac) then          
                elem_scelto=elem_concorr(k1)        
            end if
        end do
        if (elem_scelto==0)then
            
        else
            n=elements(elem_scelto)%nodes
            do j1=1,3
                x(j1)=node(n(j1))%coord(1)
                y(j1)=node(n(j1))%coord(2)
            end do
            call shape_function(x(1),y(1),x(2),y(2),x(3),y(3),Xg,Yg,L,dL)
            
            node(i1)%coord(1)=Xg
            node(i1)%coord(2)=Yg
            
            unn(2*i1-1) = L(1)*unn(2*n(1)-1)+L(2)*unn(2*n(2)-1)+L(3)*unn(2*n(3)-1)
            unn(2*i1)   = L(1)*unn(2*n(1))+L(2)*unn(2*n(2))+L(3)*unn(2*n(3))
            
        end if    
        deallocate(elem_concorr)
     end if
     deallocate(nl)
end do 

    end subroutine AddRemNodesLocal
    
!!*********************************************** 
subroutine shape_function(x1,y1,x2,y2,x3,y3,x,y,L,dL)

!!------------------------------------------------------------------------------------------------
!! Compute the value shape functions (L) and their derivatives (dL) for the element in point (x,y)
!! Input variables : x1,y1,x2,y2,x3,y3,x,y
!! Output variables: L,dL
!!------------------------------------------------------------------------------------------------

implicit none

real(8) :: x1,y1,x2,y2,x3,y3,area,x,y
real(8) :: area1,area2,area3,j,L(3),dL(3,2)

call triangle_area(x1,y1,x2,y2,x3,y3,area)
call triangle_area(x,y,x2,y2,x3,y3,area1)
call triangle_area(x1,y1,x,y,x3,y3,area2)
call triangle_area(x1,y1,x2,y2,x,y,area3)

j=1/(2*area)

L(1)=area1/area
L(2)=area2/area
L(3)=area3/area

dL(1,1)=j*(y2-y3)
dL(1,2)=-j*(x2-x3)
dL(2,1)=-j*(y1-y3)
dL(2,2)=j*(x1-x3)
dL(3,1)=j*(y1-y2)
dL(3,2)=j*(x2-x1)

return
end subroutine shape_function    
!***********************************************************
    
    
