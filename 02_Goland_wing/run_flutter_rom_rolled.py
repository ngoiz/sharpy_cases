'''
Goland wing flutter study.

For testing purposes, the wing can be built at a random roll attitude.
Aeroelastic integration is achieved upon projection of the UVLM inputs/outputs
over the GEBM degrees of freedom.
'''

import time
import numpy as np
import scipy as sc
import os

import sharpy.sharpy_main
# import sharpy.solvers.modal as modal
import sharpy.utils.h5utils as h5

import sharpy.linear.src.libsparse as libsp
import sharpy.linear.src.libss as libss
import sharpy.linear.src.librom as librom
import sharpy.linear.src.lin_aeroelastic as lin_aeroelastic

import cases.templates.flying_wings as flying_wings
# from IPython import embed


### Parametrisation, numerics & flying properties
# time.sleep(3600*18)
Nsurf=2
N=60
M=4  #  4
Mstar_fact=18

Nmodes=4
integr_order=2
RemovePred=False#True
UseSparse=True

Uinf0=1. # ROM point
UinfVec=np.linspace(30.,230.,201)
Alpha0Deg=0.0
Roll0Deg=60.
RollNodes=False#True

figsfold='./figs_rom_direct_Izfixed/modes/'
os.system('mkdir -p %s'%figsfold)
resfold='./res_rom_direct_Izfixed/'
os.system('mkdir -p %s'%resfold)
os.system('mkdir -p %s/balsys/'%resfold)
os.system('mkdir -p %s/eigs/'%resfold)


def analyse_eigs(eigs,dlti=True):
	'''
	Analyse eigenvalues of discrete or continuous-time LTI system. If dlt=True,
	the natural frequency is normalsed by the 1/time-step (i.e. divide fn by
	time-step to obtain)
	'''
	if dlti:
		fn=0.5*np.angle(eigs)/np.pi
	else:
		fn=np.abs(eigs.imag)
	return fn


# sharpy setting
route_main=os.path.abspath('.') + '/cases/'
case_main='goland_r%.2d_rnodes%s_Nsurf%.2dM%.2dN%.2dwk%.2d'\
								 %(int(Roll0Deg),RollNodes,Nsurf,M,N,Mstar_fact)
case_main_lin=case_main+'_int%.1drp%s_Nm%.3d'\
										%(integr_order,RemovePred,Nmodes)

# ------------------------------------------------ Build solution at UinfVec[0]
# This is used for creating the reference (normalised) UVLM


### Reference geometrically-exact solution (linearisation & scaling point)
u_inf=Uinf0
case_here=case_main+'_a%.4d_uinf%.4d'%( int(np.round(100*Alpha0Deg)), 10*u_inf)
route_here=route_main+'a%.4d/'%int(np.round(100*Alpha0Deg))
os.system('mkdir -p %s'%(route_here,))

# Build wing model
ws=flying_wings.Goland(M=M,N=N,Mstar_fact=Mstar_fact,
						n_surfaces=Nsurf,
						alpha=Alpha0Deg,
						roll=Roll0Deg,
						RollNodes=RollNodes,
					    u_inf=u_inf,
					    route=route_here,
					    case_name=case_here)

ws.clean_test_files()
ws.update_derived_params()
ws.generate_fem_file()
ws.generate_aero_file()

# solution flow
ws.set_default_config_dict()
ws.config['SHARPy']['flow']=[
	'BeamLoader', 'AerogridLoader', 'StaticUvlm', 'Modal', 'SaveData']
ws.config['SaveData']={'folder': route_here}
ws.config['LinearUvlm'] = {	'dt': ws.dt,
						    'integr_order':integr_order,
							'density': ws.rho,
							'remove_predictor': RemovePred,
							'use_sparse': UseSparse,
						 	'ScalingDict':{ 'length'  : 0.5*ws.c_ref,
						 				  	'speed'   : u_inf,
						 				  	'density' : ws.rho}}
ws.config['Modal']['NumLambda']= 3*Nmodes
ws.config['Modal']['keep_linear_matrices']='on'
ws.config['Modal']['use_undamped_modes']=True
ws.config.write()

# solve
data=sharpy.sharpy_main.main(['', route_here+case_here+'.solver.txt'])


### Allocate aeroelastic class
# This is done out of convenience to produce SS models of GEBM and UVLM
Sol=lin_aeroelastic.LinAeroEla(data)
gebm=Sol.lingebm_str

### get gain gebm<->uvlm
Sol.get_gebm2uvlm_gains()

# str -> aero
Kas=np.block([[Sol.Kdisp[:,:-10],np.zeros((3*Sol.linuvlm.Kzeta,gebm.num_dof)) ],
			  [ Sol.Kvel_disp[:,:-10], Sol.Kvel_vel[:,:-10]] ])
# aero -> str
Ksa=Sol.Kforces[:-10,:]
assert np.max(np.abs(Sol.Kvel_disp[:,:-10]))<1e-8,\
	'This coupling procedure is only valid if velocities do not depend on displacements'

### Prepare structural solver and modes
gebm.dlti=True
gebm.newmark_damp=5e-3
gebm.modal=True
gebm.proj_modes='undamped'
gebm.Nmodes=Nmodes
gebm.discr_method='newmark'

# combine modes in sym/anti-sym
Ntot=len(gebm.freq_natural)
Usym=np.zeros_like(gebm.U)
for cc in range(Ntot//2):
	jj01=2*cc
	jj02=2*cc+1
	Usym[:,jj01]=1./np.sqrt(2)*(gebm.U[:,jj01]+gebm.U[:,jj02])
	Usym[:,jj02]=1./np.sqrt(2)*(gebm.U[:,jj01]-gebm.U[:,jj02])
gebm.U=Usym

# remove anti-sym modes
Nnodes_free=N
# get z disp dog of wings
iivec01=[6*nn+2 for nn in range(Nnodes_free//2)]			# wing 01
iivec02=[6*nn+2 for nn in range(Nnodes_free//2,Nnodes_free)]# wing 02
tol=1e-10

jjvec=[]
for cc in range(Ntot//2):
	FoundSym=False
	for ii in range(2):
		jj=2*cc+ii
		# print('---- checking mode %.3d' %jj)

		# get max amplitude point wing 01
		iimax01=np.argmax(np.abs(Usym[iivec01,jj]))
		zmax01=Usym[iivec01,jj][iimax01]
		# get max amplitude point wing 02
		iimax02=np.argmax(np.abs(Usym[iivec02,jj]))
		zmax02=Usym[iivec02,jj][iimax02]

		# check if mode is sym
		zdiff=np.abs(zmax01-zmax02)
		if zdiff<np.abs(zmax01+zmax02):
			# print('mode symmetric (difference 01-02)=%.3e' %zdiff )
			jjvec.append(jj)
			if FoundSym==True:
				raise NameError('Sym mode already found...')
			FoundSym=True
		else:
			pass
gebm.U=Usym[:,jjvec]
gebm.freq_natural=gebm.freq_natural[jjvec]
freq0=gebm.freq_natural.copy()


### assemble, project and scale UVLM solution
Sol.linuvlm.remove_predictor=RemovePred
Sol.linuvlm.assemble_ss()

# scale
print('Scaling of UVLM eq.s started...')
Sol.linuvlm.nondimss()
Lref0=Sol.linuvlm.ScalingFacts['length']
Uref0=Sol.linuvlm.ScalingFacts['speed']
Fref0=Sol.linuvlm.ScalingFacts['force']
tref0=Sol.linuvlm.ScalingFacts['time']
print('\t\tdone in %.2f sec' %Sol.linuvlm.cpu_summary['nondim'])


# remove gust input
print('removing gust input started...')
t0=time.time()
Sol.linuvlm.SS.B=libsp.csc_matrix(Sol.linuvlm.SS.B[:,:6*Sol.linuvlm.Kzeta])
Sol.linuvlm.SS.D=Sol.linuvlm.SS.D[:,:6*Sol.linuvlm.Kzeta]
print('\t\tdone in %.2f sec' %(time.time()-t0))

# project
print('projection of UVLM eq.s started...')
t0=time.time()
Kin =libsp.dot(Kas, sc.linalg.block_diag(gebm.U[:,:gebm.Nmodes],
					 									gebm.U[:,:gebm.Nmodes]))
Kout=libsp.dot(gebm.U[:,:gebm.Nmodes].T,Ksa)
Sol.linuvlm.SS.addGain(Kin, where='in')
Sol.linuvlm.SS.addGain(Kout, where='out')
print('\t\tdone in %.2f sec' %(time.time()-t0))

Kin,Kout=None,None
Kas,Ksa=None,None
Sol.Kforces=None
Sol.Kdisp=None
Sol.Kvel_disp=None


########################################################################### ROM

cpubal=time.time()
kmin= 512#(2*gebm.Nmodes)*(2**6) 	# speed-up and better accuracy
tolSVD=1e-8
print('Balanced realisation started (with A sparsity)...')
print('kmin: %s tolSVD: %s'%(kmin,tolSVD))
gv,T,Ti,rc,ro=librom.balreal_iter(Sol.linuvlm.SS.A,Sol.linuvlm.SS.B,Sol.linuvlm.SS.C,
				lowrank=True,tolSmith=1e-10,tolSVD=tolSVD,kmin=kmin,tolAbs=False,Print=True)
cpubal=time.time()-cpubal
print('\t\tdone in %.2f sec'%cpubal )


### define balaned system
Ab=libsp.dot(Ti,libsp.dot(Sol.linuvlm.SS.A,T))
Bb=libsp.dot(Ti,Sol.linuvlm.SS.B)
Cb=libsp.dot(Sol.linuvlm.SS.C,T)
SSb=libss.ss(Ab,Bb,Cb,Sol.linuvlm.SS.D,dt=Sol.linuvlm.SS.dt)
Nxbal=Ab.shape[0]


### tune ROM
# ROM is tuned under the assumption that the balanced system freq. response
# is exact.
ds=Sol.linuvlm.SS.dt
fs=1./ds
fn=fs/2.
ks=2.*np.pi*fs
kn=2.*np.pi*fn
# build freq. range
kmin,kmax,Nk=0.001,.5,30
kv=np.linspace(kmin,kmax,Nk)

Yfull=libss.freqresp(Sol.linuvlm.SS,kv,dlti=True)
Yb=libss.freqresp(SSb,kv,dlti=True)
Ybmax=np.max(np.abs(Yb))
erb=np.max(np.abs(Yb-Yfull))
print('Freq response error of balanced system: %.2e' %erb)
SSrom=librom.tune_rom(SSb,kv,tol=1e-6*Ybmax,gv=gv,method='realisation',convergence='all')

params=h5.ReadInto('params')
params.Nmodes=gebm.Nmodes
params.freq0=freq0.copy()
params.Nxfull=Sol.linuvlm.SS.A.shape[0]
params.cputime=cpubal
params.er_abs=erb
params.T=T
params.Ti=Ti

BalRes=h5.ReadInto('BalRes')
BalRes.ScalingFacts=Sol.linuvlm.ScalingFacts
BalRes.SSb=SSb
BalRes.params=params

scaling=h5.ReadInto('scaling')
scaling.rhoref0=Sol.linuvlm.ScalingFacts['density']
scaling.Lref0=Lref0
scaling.Uref0=Uref0
scaling.Fref0=Fref0
scaling.tref0=tref0

h5name=resfold+'/balsys/ssb_%s.h5'%(case_main_lin,)
os.system('rm -f %s' %h5name )
h5.saveh5('.',h5name,*(params,SSb,scaling),ClassesToSave=(h5.ReadInto, libss.ss) )
# hdfile=h5py.File(h5name,'w')
# savedata.add_as_grp( BalRes, hdfile,
# 					 grpname='balres',
#         			 ClassesToSave=(libss.ss,h5utils.ReadInto,) ,
#                      compress_float=False )
# hdfile.close()
del params, BalRes
T,Ti=None,None

import matplotlib.pyplot as plt
color1 = np.array([0, 1, 0])
color2 = np.array([0, 0, 1])

Nu=len(UinfVec)
EigsCont=[]
for uu in range(Nu):
	u_inf=UinfVec[uu]

	### define dimensional quantities
	Uref=u_inf
	qinf=.5*ws.rho*Uref**2
	tref=Lref0/Uref
	fref=qinf*Lref0**2
	# scale_in=np.array( gebm.Nmodes*[Lref0] + gebm.Nmodes*[Uref] )
	# scale_out=qinf*Lref0**2

	### update gebm
	gebm.dt=Sol.linuvlm.SS.dt # normalised time
	gebm.freq_natural=freq0.copy() * tref
	gebm.inout_coords='modes'
	gebm.assemble()
	SSstr=gebm.SSdisc

	# gebm modal -> uvlm modal
	Tas=np.eye(2*gebm.Nmodes)/Lref0
	# uvlm modal -> gebm modal
	Tsa=np.diag( (fref*tref**2) * np.ones(gebm.Nmodes,) )

	### feedback connection
	SS=libss.couple( ss01=SSrom,ss02=SSstr,K12=Tas,K21=Tsa )

	### only for eigenvalue conversion!
	dt_new=ws.c_ref/M/u_inf
	assert np.abs(dt_new - tref*Sol.linuvlm.SS.dt)<1e-14, 'dimensional time-scaling not correct!'


	### aeroelastic eigenvalues
	eigs,eigvecs=np.linalg.eig(SS.A)
	eigmag=np.abs(eigs)
	order=np.argsort(eigmag)[::-1]
	eigs=eigs[order]
	eigvecs=eigvecs[:,order]
	eigmag=eigmag[order]
	eigmax=np.max(eigmag)
	Nunst=np.sum(eigmag>1.)
	fn=analyse_eigs(eigs,dlti=True)/dt_new
	eigscont=np.log(eigs)/dt_new

	# uvlm and gebm
	eigs_gebm=np.linalg.eigvals(SSstr.A)
	order=np.argsort(eigs_gebm)[::-1]
	eigs_gebm=eigs_gebm[order]
	fn_gebm=analyse_eigs(eigs_gebm)/dt_new
	eigmax_gebm=np.max(np.abs(eigs_gebm))

	print('DLTI\tu: %.2f m/2\tmax.eig.: %.6f\tmax.eig.gebm: %.6f'\
								 					%(u_inf,eigmax,eigmax_gebm))
	# print('\tGEBM nat. freq. (Hz):'+len(fn_gebm)*'\t%.2f' %tuple(fn_gebm))
	print('\tN unstab.: %.3d' %(Nunst,))
	print('\tAela freq. unst. (Hz):'+Nunst*'\t%.2f'%tuple(fn[:Nunst]) )

	eigs_cont = np.log(eigs) / dt_new
	color = color1 * (uu/(Nu-1)) + color2 * (1 - uu/(Nu-1))
	plt.scatter(eigs_cont.real, eigs_cont.imag, c=color, s=2)

	### save eigen-analysis
	Neigvec=20
	stab=h5.ReadInto('stab')
	stab.eigs=eigs
	stab.eigvecs=eigvecs[:,:Neigvec]
	stab.eigscont=eigscont
	stab.fn=fn
	stab.fn_gebm=fn_gebm
	stab.u_inf=u_inf

	h5name=resfold+'/eigs/eigs_%sU%.5d.h5'%(case_main_lin,int(u_inf*100))
	os.system('rm -f %s' %h5name )
	h5.saveh5('.',h5name,*(stab,))
	# modal.write_modes_vtk(	data, gebm.U, NumLambda=min(10,Nmodes),
	# 						filename_root=figsfold+'mode_'+case_main_lin,

	#                         rot_max_deg=15.,perc_max=0.15)

plt.xlim([-10, 10])
plt.ylim([0, 300])
plt.show()

EigsCont=np.array(EigsCont)
print(SSrom.states)
