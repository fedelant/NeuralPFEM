!!!!!!!!!!!!!!!!!!!!!!!!!!
!!      FLUID ONLY      !!
!!!!!!!!!!!!!!!!!!!!!!!!!! 
    
!!*******************************************************************  
module fluid_module

!!------------------------------------------------------------
!! Variables shared between all analysis type (fluid,3DCP,FSI)
!!------------------------------------------------------------

integer :: problem_type !fluid=0, 3DCP=1, FSI=2, FluidSedMix=3
integer :: npoints, ntank, n_dbc, nelement,number_elements
real(8) :: el_size

real(8), allocatable :: gamma(:)
real(8), allocatable :: visc(:)
real(8), allocatable :: h(:)
integer(8), allocatable :: count_strain(:)
real(8),parameter :: PI=3.14159265358979

integer, dimension(:,:), allocatable:: elem_array

!!Material parameters
integer :: MatType          !!Type of material (0=Newtonian, 1=Bingham, 2=Frictional)
real(8) :: p0, p02          !!Initial pressure
real(8) :: K0,KK            !!Bulk modulus
real(8) :: rho_l0,mu,tau0,shear_mod, eta_p,cons_param,power_ind    !!Initial density, viscosity and tangential stress
real(8) :: c_lbar,gamma_l   !!Speed of dilatational wave, specific heat ratio

!!Frequencies
real(8) :: fps, fpsRemesh, fpsAddRem    !!output, remesh and AddRem frequencies

!!Time variables 
real(8) :: t,t_in,t_fin                     !!Current, initial, final time
real(8) :: deltat,deltat0,deltat_previous,deltat_previous2                   !!time step
real(8) :: deltatOut,deltatRemesh           !!Output and remeshing step
real(8) :: deltatAddRem,deltatAddRemGlob    !!Local and global step
real(8) :: Tfilter,weight, Tratio 

!!Counters
integer :: it_time,it_time_restart                          !!n. of time steps
integer :: n_remesh, out_time,OutParCont    !!n. of remeshing, output and parallelized output
real(8) :: t_output,t_remesh                !!Output and Remeshing time
real(8) :: t_AddRem,t_AddRemGlob            !!Local and global AddRem time

!!Mesh check parameters
real(8) :: area_m,dx_drop,dy_drop   !!elements mean area
real(8) :: alpha_surf,alpha_inner,alpha_bound,alpha_gen
real(8) :: rapp_rr =1.d0
logical :: check_mesh=.true.
logical :: made_mesh=.false.

!!Slip parameters
real(8) :: h_slip, phi_bulk, phi_basal

!!Analysis options
integer :: SlipON, ConvON   !!Include slip condition
integer :: EulerON          !!Include Eulerian formulation
integer :: GravityON        !!Include Gravity
integer :: RestartON        !!Restart analysis from an user-specified point  
integer :: TagAllocate=1    !!Allocate auxiliary matrices for parallelization OPENMP

!!Variables for OpenMP parallelization 
integer :: threads_number  !!Number of thread
real(8),dimension(:,:), allocatable :: OutCoord, OutUnn,OutP, OutRhonn, OutInsph, OutStress
real(8),dimension(:,:), allocatable :: OutViscDt, OutMuEff,OutCFL,OutMatTime,OutElMatTime,OutYield,OutVisc,OutElTensor
real(8),dimension(:,:), allocatable :: Mrho_loc, KDV_loc, StabP_loc
real(8),dimension(:,:), allocatable :: FUG_loc, Mvel_loc, KpConvP_loc
integer(4),dimension(:,:), allocatable :: OutConnec, OutCosim, OutEul, OutThreshold, OutJustAdd
integer(4),dimension(:,:), allocatable :: OutFree, OutBound, OutContact,OutLayer,OutBox
integer(4),allocatable :: OutElement(:),outPoint(:)
real(8),dimension(:,:), allocatable :: OutStrainRate, OutStrainEl, OutStrainPl, OutStrain

!! VARIABLES FOR MONITORING THE COMPUTATIONAL TIME 
!!Total analysis time
integer :: time_in(8),time_out(8)
real(8) :: delta_time, delta_time0

!!Output time
integer :: time_output_in(8),time_output_out(8)
real(8) :: delta_time_output
real(8) :: total_output_time = 0.d0

!!triangulation time
integer :: time_tri_in(8),time_tri_out(8)
real(8) :: delta_time_tri
real(8) :: total_tri_time = 0.d0

!!Solver time
integer :: time_sol_in(8),time_sol_out(8)
real(8) :: delta_time_sol
real(8) :: total_sol_time = 0.d0

!!AddRem time
integer :: time_AddRem_in(8),time_AddRem_out(8)
real(8) :: delta_time_AddRem
real(8) :: total_AddRem_time = 0.d0

!!Benchmark variables and arrays
integer :: n_rows_data_set, row_count = 1
integer, parameter :: benchmark = 0
real(8) , allocatable :: x_data_set(:), y_data_set(:), diff(:)
real(8), allocatable :: unn(:)
!!STRUCTURE DATA 

type Snode
 integer :: number
 real(8) :: coord(2)
 integer(2) :: bound=0
 integer(2) :: alone=0
 integer(2) :: free_surf=0
 integer(2) :: move_node=0
 integer(2) :: int = 0
 integer    :: neighb(500)
 integer(2) :: ntri=0
 integer(2) :: euler=0
 integer(2) :: slip=0
 integer(2) :: state=0
 integer(2) :: box=0
 integer(2) :: contact=0
 integer(2) :: layer =1
 integer(2) :: update=0         !need to specify next layers, in update_geom
 integer(2) :: bed=0            !Bed node that is erodible
 integer(2) :: robin=0          !Node that has imposed on the Robin boundary condition
 integer(2) :: just_add
 real(8) :: mat_time        = 0.d0
 real(8) :: strain_el(3)    = 0.d0
 real(8) :: str_rate_normP  = 0.d0
 real(8) :: dstrain_el(3)   = 0.d0
 integer(2) :: count_strain = 0
 real(8) :: bdc(2)
end type Snode

type Selement
 integer :: nodes(3)=0
 integer :: euler !1 if at least one eulerian node belongs to the element
 integer :: threshold
 real(8) :: circumcenter(2) = 0.d0
 real(8) :: InRad           = 0.d0
 real(8) :: viscosity       = 0.d0 ! it is the same of mu eff
 real(8) :: mu_eff          = 0.d0
 real(8) :: dt_cfl          = 0.d0
 real(8) :: dt_visc         = 0.d0
 real(8) :: strain_rate     = 0.d0
 real(8) :: strain_el       = 0.d0
 real(8) :: strain          = 0.d0
 real(8) :: stress(3)       = 0.d0
 real(8) :: el_p            = 0.d0
 real(8) :: el_rho          = 0.d0
 real(8) :: area            = 0.d0
 real(8) :: yield           = 0.d0
 real(8) :: el_mat_time     = 0.d0
 real(8) :: visc            = 0.d0
end type Selement

type output_var_type
 logical :: binary = .true.
 logical :: vel    = .true.
 logical :: press  = .true.
 logical :: int_en = .true.
 logical :: dens   = .false.
 logical :: stress = .true.
 logical :: visc   = .false.
 logical :: free_s = .false.
 logical :: bound  = .true.
 logical :: int_p  = .false.
 logical :: info   = .false.
 logical :: state  = .true.
 logical :: conc   = .false.
 logical :: bed    = .false.
 logical :: robin  = .false.
end type output_var_type

type(Snode), allocatable :: node(:)              !!list of nodes 
type(Selement), allocatable :: list_elements(:)  !!Elements after the triangulation (including distorted el).
type(Selement), allocatable :: elements(:)       !!Remaining undistorted elements, after alpha-shape.
type(output_var_type) :: output_var              !!Setting requested variables in ouput


    end module fluid_module
!!******************************************************************* 
    
    
    
!!*******************************************************************  
module fluid_module_dt

!!-------------------------------------------------------------------
!!quantities needed in fluid_solution_dt
!!-------------------------------------------------------------------

real(8) :: ff(2)
real(8) :: mu_eff, deltat_visc_min

real(8), allocatable :: FU(:),FUG(:)
real(8), allocatable :: Mrho(:),Mvel(:)
real(8), allocatable :: KDV(:), StabP(:), KpConvP(:)
    end module fluid_module_dt
!!*******************************************************************  

    
    
!!*******************************************************************  
module solution_module

!!-------------------------------------------------------------------
!!solution variables
!!-------------------------------------------------------------------

real(8), allocatable :: p(:), f(:), p_m(:)
real(8), allocatable :: rhonn(:)
real(8), allocatable :: ff0(:),  epsylon(:,:)

    end module solution_module   
!!*******************************************************************
    

!!*******************************************************************  
    
!!!!!!!!!!!!!!!!!!!!!!!!!!
!!   Sediment Module    !!
!!!!!!!!!!!!!!!!!!!!!!!!!! 
    
module sediment_module

!!-------------------------------------------------------------------
!!Sediment variables, parameters and structures
!!-------------------------------------------------------------------

!Sediment model parameters
real(8) :: diff_s,sigma_c   !!Sediment diffusivity and Schmidt number
real(8) :: ws, expon        !!Sediment velocity, hindered settling exponent

!Robin condition parameters
real(8) :: gamma_r, c_star  !!Reflectivity coefficient, equilibrium concentration at the boundary

!Sediment properties and parameters
real(8) :: Dm, rho_s        !!Mean dimension of the sediment, sediment density
real(8), parameter :: von_k = 0.41d0  !Von-karman constant

!!Stabilization for the sediment equation
integer :: StabON           

!!Sediment variables for OpenMP parallelization 
real(8), dimension(:,:), allocatable :: OutC
real(8), dimension(:,:), allocatable :: FUGsed_loc, Msed_loc, KcConv_loc

!!Quantities for fluid_solution_dt
real(8), allocatable :: FU_sed(:), Msed(:), KcConv(:), con(:)

end module sediment_module

!!*******************************************************************  
    
!!!!!!!!!!!!!!!!!!!!!!!!!!
!! 3D CONCRETE PRINTING !!
!!!!!!!!!!!!!!!!!!!!!!!!!! 
    
!!*******************************************************************  
module module_3DCP  

!!-------------------------------------------------------------------
!!quantities needed specifically for 3DCP
!!-------------------------------------------------------------------

integer ::  n_eul_bc, ntot, ngen_tot !n of mix el., n\B0 of fictitious nodes to allocate, ntot of allocated nodes, n\B0 of added nodes at time t
real(8) :: h_nozzle, v_print, v_flow, tau0_static !height of the nozzle w.r.t. previous layer, time to complete 1 layer, printing vel., n\B0 of current layer
real(8) :: area_0,v_mesh(2)

logical :: continuos
integer :: n_corners,nlayer,n_segment
real(8), allocatable :: xxx(:), yyy(:), zzz(:)
real(8) :: t_previous, t_layer

real(8) :: t_box
integer ::eul_first,eul_last

integer::check_adaptive,count_addFS
integer, allocatable :: moving_nodes(:),id_del(:)
real(8), allocatable :: moving_dist(:)
real(8) :: scale_el2,scale_el3,scale_r2,scale_fb,scale_surf_max,scale_surf_min
integer :: scale_r3


integer ::up,to_be_moved_counter2,del_check,poss
integer,allocatable :: to_be_moved2(:)
real(8) :: ElViscoMax
real(8) :: Rthix,Athix,Tchange,Amu

    end module module_3DCP
!!*******************************************************************
    
    
    
    
    
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  FLUID-STRUCTURE INTERACTION !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
    
!!*******************************************************************  
module module_FSI 

!!-------------------------------------------------------------------
!!quantities needed specifically for FSI
!!-------------------------------------------------------------------

!add HERE FSI quantities please...

end module module_FSI
 !!*******************************************************************  
