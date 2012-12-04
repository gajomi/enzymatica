from enzymatica import *
import pylab as pylab

def full_time_series(sigma,calib,initial_conditions,time,k,phi):
    rate_fun = partial(mm_rate,k)
    Z = [reaction_time_series(rate_fun, z0, time) for z0 in Z0]
    return [turbidity_from(z,partial(basic_susceptibility,phi,z[0,:]),calib) for z in Z]

def write_test_data(filename,sigma,calib,initial_conditions,time,k,phi):
    """Generates a test data set from specified arguments"""
    Rho_exact = full_time_series(sigma,calib,initial_conditions,time,k,phi)
    Rho_data = [rho+sigma*np.random.randn(*rho.shape) for rho in Rho_exact]
    with open(filename,'w') as f:
        f.write('>Test data generated assuming MM reaction\n')
        f.write('k_exact = '+str(k)+'\n phi_exact = ' + str(phi) + '\n\n')
        f.write('>Turbidity error (std)\n')
        f.write(str(sigma)+'\n\n')
        f.write('>Turbidity Calibration\n')
        f.write(str(calib[0])+' '+str(calib[1])+'\n\n')
        f.write('>Initial Conditions (substrate concentration,enzyme concentration)\n')
        f.write(' '.join(str(s0) for s0 in S0)+'\n')
        f.write(' '.join(str(e0) for e0 in E0)+'\n\n')
        f.write('>Time series\n')
        for row in zip(time,*Rho_data):
            f.write(' '.join(str(x) for x in row)+'\n')
    print 'Test written to: '+filename

#sampling times, turbidity calibration and turbidity measurement error                                            
N_max = 1000
T_end = 30
t_interval = .1
time = [t_interval*n for n in takewhile(lambda k: t_interval*k < T_end, range(N_max))]
calibration = (1.0,0.1)
sigma = .002

#initial conditions                                                                                               
S0 = [1.0, 1.0, 1.0, 1.0, 1.0]
E0 = [2.0, 4.0, 8.0, 16.0, 32.0]
Z0 = [[s0, e0, 0.0, 0.0] for s0,e0 in zip(S0,E0)]

#general range and guesses for reaction rate constants and turbidity shape parameters
k_box = [(.5, 1.5),(.5,1.5),(.1,.4)]
phi0 = (2,2)

#reaction rate constants and turbidity shape parameters                                                           
K = [(1.0,1.1,.2),(.5,2.1,.5)]
Phi = [(2,2),(5,1),(1,5)]
indexed_params = [((i+1)*(j+1),k,phi) for i,k in zip(range(len(K)),K) for j,phi in zip(range(len(Phi)),Phi)]

K_est = []
Phi_est = []
for index,k,phi in indexed_params:
    datafile = 'test_data'+str(index)+'.data'
    write_test_data('test_data'+str(index)+'.data',sigma,calibration,Z0,time,k,phi)
    meta,calib,sigma,initial_conditions,time,turbidity = parse_turbidity_data(datafile)
    args = (sigma,calib,initial_conditions,time,turbidity,phi0)
    sol_data = infer_ml_parameters_given(*args,k_bounds = k_box)
    sol,sol_cov = sol_data
    k_est = sol[0:3]
    phi_est = sol[3:]
    K_est.append(k_est)
    Phi_est.append(phi_est)
    
    turbidity_est = np.array(full_time_series(sigma,calib,initial_conditions,time,k,phi))
    
    fraction = np.linspace(0,1,100)
    susceptibility_est = basic_susceptibility(phi_est,[1],np.array([fraction for i in range(4)]).T)
    susceptibility_true = basic_susceptibility(phi,[1],np.array([fraction for i in range(4)]).T)

    
    pylab.plot(time,turbidity.T,'o')
    pylab.plot(time,turbidity_est.T,'-')
    pylab.xlabel('time')
    pylab.ylabel('turbidity')
    pylab.savefig('turbidity'+str(index)+'.png')
    
    pylab.clf()
    pylab.plot(fraction,susceptibility_true)
    pylab.plot(fraction,susceptibility_est)
    pylab.xlabel('fraction bonds cleaved')
    pylab.ylabel('probability of lysis')
    pylab.savefig('susceptibility'+str(index)+'.png')
    pylab.clf()

for k_est,phi_est,i,k,phi in zip(K_est,Phi_est,*zip(*indexed_params)):
    print ' '
    print i
    print ''.join('{0:.3g}'.format(k_i)+' ' for k_i in k)
    print ''.join('{0:.3g}'.format(k_i)+' ' for k_i in k_est)
    print ''.join('{0:.3g}'.format(phi_i)+' ' for phi_i in phi)
    print ''.join('{0:.3g}'.format(phi_i)+' ' for phi_i in phi_est)

