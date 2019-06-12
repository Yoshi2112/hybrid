# =============================================================================
# def ax_add_run_params(ax):
#     font    = 'monospace'
#     top     = 1.07
#     left    = 0.78
#     h_space = 0.04
#     
#     ## Simulation Parameters ##
#     
#     ## Particle Parameters ##
#     ax1.text(0.00, top - 0.02, '$B_0 = $variable'     , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(0.00, top - 0.05, '$n_0 = $variable' % n0, transform=ax1.transAxes, fontsize=10, fontname=font)
#     
#     ax1.text(left + 0.06,  top, 'Cold'    , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 0.099, top, 'Warm'    , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 0.143, top, '$A_i$'   , transform=ax1.transAxes, fontsize=10, fontname=font)
#     
#     ax1.text(left + 0.192, top, r'$\beta_{\parallel}$'                    , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 4.2*h_space, top - 0.02, '{:>7.2f}'.format(betapar[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 4.2*h_space, top - 0.04, '{:>7.2f}'.format(betapar[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 4.2*h_space, top - 0.06, '{:>7.2f}'.format(betapar[2]), transform=ax1.transAxes, fontsize=10, fontname=font)
# 
#     ax1.text(left + 0.49*h_space, top - 0.02, ' H+:'                    , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 1*h_space, top - 0.02, '{:>7.3f}'.format(H_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 2*h_space, top - 0.02, '{:>7.3f}'.format(H_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 3*h_space, top - 0.02, '{:>7.2f}'.format(A[0]),      transform=ax1.transAxes, fontsize=10, fontname=font)
#     
#     ax1.text(left + 0.49*h_space, top - 0.04, 'He+:'                     , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 1*h_space, top - 0.04, '{:>7.3f}'.format(He_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 2*h_space, top - 0.04, '{:>7.3f}'.format(He_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 3*h_space, top - 0.04, '{:>7.2f}'.format(A[1])      , transform=ax1.transAxes, fontsize=10, fontname=font)
#     
#     ax1.text(left + 0.49*h_space, top - 0.06, ' O+:'                    , transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 1*h_space, top - 0.06, '{:>7.3f}'.format(O_frac[0]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 2*h_space, top - 0.06, '{:>7.3f}'.format(O_frac[1]), transform=ax1.transAxes, fontsize=10, fontname=font)
#     ax1.text(left + 3*h_space, top - 0.06, '{:>7.2f}'.format(A[2])     , transform=ax1.transAxes, fontsize=10, fontname=font)
#     return
# =============================================================================

# =============================================================================
# def diagnostic_multiplot(qq):
#     plt.ioff()
# 
#     fig_size = 4, 7                                                             # Set figure grid dimensions
#     fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
#     fig.patch.set_facecolor('w')                                                # Set figure face color
# 
#     va        = B0 / np.sqrt(mu0*ne*mp)                                         # Alfven speed: Assuming pure proton plasma
# 
#     pos       = position / dx                                                   # Cell particle position
#     vel       = velocity / va                                                   # Normalized velocity
# 
#     den_norm  = dns / density                                                   # Normalize density for each species to initial values
#     qdens_norm= q_dns / (density*charge).sum()                                  # Normalized change density
#      
# #----- Velocity (x, y) Plots: Hot Species
#     ax_vx   = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=3)
#     ax_vy   = plt.subplot2grid(fig_size, (2, 0), rowspan=2, colspan=3)
# 
#     for jj in range(Nj):
#         ax_vx.scatter(pos[idx_bounds[jj, 0]: idx_bounds[jj, 1]], vel[0, idx_bounds[jj, 0]: idx_bounds[jj, 1]], s=3, c=temp_color[jj], lw=0, label=species_lbl[jj])
#         ax_vy.scatter(pos[idx_bounds[jj, 0]: idx_bounds[jj, 1]], vel[1, idx_bounds[jj, 0]: idx_bounds[jj, 1]], s=3, c=temp_color[jj], lw=0)
# 
#     ax_vx.legend()
#     ax_vx.set_title(r'Particle velocities vs. Position (x)')
#     ax_vy.set_xlabel(r'Cell', labelpad=10)
# 
#     ax_vx.set_ylabel(r'$\frac{v_x}{c}$', rotation=90)
#     ax_vy.set_ylabel(r'$\frac{v_y}{c}$', rotation=90)
# 
#     plt.setp(ax_vx.get_xticklabels(), visible=False)
#     ax_vx.set_yticks(ax_vx.get_yticks()[1:])
# 
#     for ax in [ax_vy, ax_vx]:
#         ax.set_xlim(0, NX)
#         ax.set_ylim(-10, 10)
# 
# #----- Density Plot
#     ax_den = plt.subplot2grid((fig_size), (0, 3), colspan=3)                     # Initialize axes
#     
#     ax_den.plot(qdens_norm, color='green')                                       # Create overlayed plots for densities of each species
# 
#     for jj in range(Nj):
#         ax_den.plot(den_norm, color=temp_color[jj])
#         
#     ax_den.set_title('Normalized Densities and Fields')                          # Axes title (For all, since density plot is on top
#     ax_den.set_ylabel(r'$\frac{n_i}{n_0}$', fontsize=14, rotation=0, labelpad=5) # Axis (y) label for this specific axes
#     ax_den.set_ylim(0, 2)
#     
# #----- E-field (Ex) Plot
#     ax_Ex = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_den)
# 
#     ax_Ex.plot(E[:, 0], color='red', label=r'$E_x$')
#     ax_Ex.plot(E[:, 1], color='cyan', label=r'$E_x$')
#     ax_Ex.plot(E[:, 2], color='black', label=r'$E_x$')
# 
#     ax_Ex.set_xlim(0, NX)
# 
#     #ax_Jx.set_yticks(np.arange(-200e-5, 201e-5, 50e-5))
#     #ax_Jx.set_yticklabels(np.arange(-150, 201, 50))
#     ax_Ex.set_ylabel(r'$E$', labelpad=25, rotation=0, fontsize=14)
# 
# #----- Magnetic Field (By) and Magnitude (|B|) Plots
#     ax_By = plt.subplot2grid((fig_size), (2, 3), colspan=3, sharex=ax_den)
#     ax_B  = plt.subplot2grid((fig_size), (3, 3), colspan=3, sharex=ax_den)
# 
#     mag_B  = (np.sqrt(B[:-1, 0] ** 2 + B[:-1, 1] ** 2 + B[:-1, 2] ** 2)) / B0
#     B_norm = B[:-1, :] / B0                                                           
# 
#     ax_B.plot(mag_B, color='g')                                                        # Create axes plots
#     ax_By.plot(B_norm[:, 1], color='g') 
#     ax_By.plot(B_norm[:, 2], color='b') 
# 
#     ax_B.set_xlim(0,  NX)                                                               # Set x limit
#     ax_By.set_xlim(0, NX)
# 
#     ax_B.set_ylim(0, 2)                                                                 # Set y limit
#     ax_By.set_ylim(-1, 1)
# 
#     ax_B.set_ylabel( r'$|B|$', rotation=0, labelpad=20, fontsize=14)                    # Set labels
#     ax_By.set_ylabel(r'$\frac{B_{y,z}}{B_0}$', rotation=0, labelpad=10, fontsize=14)
#     ax_B.set_xlabel('Cell Number')                                                      # Set x-axis label for group (since |B| is on bottom)
# 
#     for ax in [ax_den, ax_Ex, ax_By]:
#         plt.setp(ax.get_xticklabels(), visible=False)
#         ax.set_yticks(ax.get_yticks()[1:])
# 
#     for ax in [ax_den, ax_Ex, ax_By, ax_B]:
#         qrt = NX / (4.)
#         ax.set_xticks(np.arange(0, NX + qrt, qrt))
#         ax.grid()
# 
# #----- Plot Adjustments
#     plt.tight_layout(pad=1.0, w_pad=1.8)
#     fig.subplots_adjust(hspace=0)
# 
#     filename = 'diag%05d.png' % ii
#     path     = anal_dir + '/diagnostic_plot/'
#     
#     if os.path.exists(path) == False:                                   # Create data directory
#         os.makedirs(path)
# 
#     fullpath = path + filename
#     plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
#     print('Plot saved'.format(ii))
#     plt.close('all')
#     return
# =============================================================================

# =============================================================================
# def examine_run_parameters(to_file=False):
#     '''
#     Diagnostic information to compare runs at a glance. Values include
#     
#     cellpart, Nj, B0, ne, NX, Te0, max_rev, ie, theta, dxm
#     
#     number of files
#     '''
#     import tabulate
#     global run_num
#     
#     run_params = ['cellpart', 'Nj', 'B0', 'ne', 'NX', 'Te0', 'max_rev', 'ie', 'theta', 'dxm', 'num_particle_steps', 'num_field_steps']
#     run_dict = {'run_num' : []}
#     for param in run_params:
#         run_dict[param] = []
#     
#     for run_num in range(num_runs):
#         manage_dirs(create_new=False)
#         load_header()                                           # Load simulation parameters
#         load_particles()                                        # Load particle parameters
#         
#         run_dict['run_num'].append(run_num)
#         for param in run_params:
#             run_dict[param].append(globals()[param])
#         
#     if to_file == True:
#         txt_path  = base_dir + 'QL_simulation_parameters.txt'
#         run_file = open(txt_path, 'w')
#     else:
#         run_file = None
#         
#     print('Simulation parameters for runs in series \'{}\':'.format(series), file=run_file)
#     print((tabulate.tabulate(run_dict, headers="keys")), file=run_file)
#     return
# =============================================================================