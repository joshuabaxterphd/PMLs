import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def signal(time, pulse_width, pulse_delay, omega0, amplitude = 1.0):
    return amplitude * np.exp(-((time - pulse_delay)/pulse_width) ** 2) * np.sin(omega0 * time)

eps0 = 8.8541878128e-12 # permittivity of free space
mu0 = 1.256637062e-6 # permeability of free space
c0 = 2.99792458e8 # speed of light in vacuum
imp0 = np.sqrt(mu0/eps0) # impedance of free space

simulation_size = 20e-6
step_size = 5e-9 # dy
N_space_cells = int(simulation_size/step_size) # jmax
print(f"there are {N_space_cells} FDTD cells")

dt = step_size/c0
simulation_time = 1e-12
N_time_steps = int(simulation_time/dt)
print(f"there are {N_time_steps} FDTD time steps")

Ex = np.zeros(N_space_cells)
Hz = np.zeros(N_space_cells)
eps = np.ones(N_space_cells) * 1
sigma = np.zeros(N_space_cells)
sigma_h = np.zeros(N_space_cells)
refractive_index = np.sqrt(eps)

pml_size = 10
m = 3
pml_cond_e = - (m + 1) * np.log(1e-10) * c0 * eps0 / (2 * pml_size * step_size)

sigma[-pml_size:] = pml_cond_e * (np.arange(pml_size)/pml_size) ** m
sigma_h[-pml_size:] = pml_cond_e * mu0 / (eps0 * eps[-pml_size:]) * ((np.arange(pml_size) + 0.5)/pml_size) ** m

sigma[:pml_size] = pml_cond_e * np.flip((np.arange(pml_size) + 1)/pml_size) ** m
sigma_h[:pml_size] = pml_cond_e * mu0 / (eps0 * eps[:pml_size]) * np.flip((np.arange(pml_size) + 0.5)/pml_size) ** m


# plt.plot(sigma)
# plt.show()

h_coeff = dt/(mu0 * step_size)

denominator = eps0 * eps/dt + sigma/2
e_coeff_1 = (eps0 * eps/dt - sigma/2) / denominator
e_coeff_2 = 1.0 / (step_size * denominator)

denominator_h = mu0/dt + sigma_h/2
h_coeff_1 = (mu0/dt - sigma_h/2) / denominator_h
h_coeff_2 = 1.0 / (step_size * denominator_h)

c = c0 / refractive_index[0]
c_ = c0 / refractive_index[-1]
a = (c * dt - step_size) / (c * dt + step_size)
a_ = (c_ * dt - step_size) / (c_ * dt + step_size)
pml_size = 0

# set up pulse stuff
center_wavelength = 1550e-9
omega0 = 2 * np.pi * c0 / center_wavelength
pulse_width = 10e-15
pulse_delay = 4 * pulse_width

# time = np.linspace(0,simulation_time,N_time_steps)
# pulse = signal(time,pulse_width,pulse_delay,omega0)
# plt.plot(time,pulse)
# plt.show()

j_source = N_space_cells//2 + pml_size
t_offset = refractive_index[j_source] * step_size / (2 * c0)
Z = imp0 / refractive_index[j_source]

E_movie = []

# set up Fourier monitor
jT = N_space_cells - pml_size - 5
jR = j_source - 5
jR = N_space_cells - pml_size - 5

ER = np.zeros(N_time_steps)
ET = np.zeros(N_time_steps)

# FDTD algorithm
for n in range(N_time_steps):
    Hz_prev = Hz.copy()
    Ex_prev = Ex.copy()

    # update magnetic field at n+1/2
    Hz[:N_space_cells-1] = (h_coeff_1[:N_space_cells-1] * Hz_prev[:N_space_cells-1]
                            + h_coeff_2[:N_space_cells-1] * (Ex[1:] - Ex[0:N_space_cells-1]))

    # add magnetic field source
    Hz[j_source-1] = Hz[j_source-1] + signal((n + 0.5)*dt + t_offset, pulse_width, pulse_delay, omega0) / Z
    # Hz[j_source-1] = Hz[j_source-1] - signal((n + 0.5)*dt - t_offset, pulse_width, pulse_delay, omega0) / Z

    # update electric field at n+1
    Ex[1:N_space_cells-1] = (e_coeff_1[1:N_space_cells-1] * Ex_prev[1:N_space_cells-1]
                             + e_coeff_2[1:N_space_cells-1] * (Hz[1:N_space_cells-1] - Hz[:N_space_cells-2]))

    # add electric field source
    Ex[j_source] = Ex[j_source] + signal((n + 1)*dt, pulse_width, pulse_delay, omega0)

    # store reflection and transmission data
    ET[n] = Ex[jT]
    ER[n] = Ex[jR]

    if n % 25 == 0:
        print(n)
        print(np.min(Ex),np.max(Ex))
        E_movie.append(Ex.copy())


# plt.plot(ET)
plt.semilogy(ER)
plt.show()

wavelengths = np.linspace(center_wavelength - 100e-9,center_wavelength + 100e-9, 100)
omegas = 2 * np.pi * c0 /wavelengths
time = np.arange(N_time_steps) * dt

def Discrete_Fourier_Transform(field, time, omega):
    N_freq = omega.shape[0]
    field_omega = np.zeros(N_freq, dtype= 'complex128')
    for w in range(N_freq):
        field_omega[w] = np.sum(field * np.exp(1j * omega[w] * time))
    return field_omega

def thin_film_TR(n1,n2,n3,wavelengths,thickness):
    r12 = (n1 - n2)/(n1 + n2)
    r23 = (n2 - n3)/(n2 + n3)
    t12 = 2 * n1 / (n1 + n2)
    t23 = 2 * n2 / (n2 + n3)
    beta = 2 * np.pi * n2 * thickness / wavelengths
    r = (r12 + r23 * np.exp(-2 * 1j * beta))/(1.0 + r12 * r23 * np.exp(-2 * 1j * beta))
    t = (t12 * t23 * np.exp(-1j * beta))/(1.0 + r12 * r23 * np.exp(-2 * 1j * beta))
    return (n3/n1) * np.abs(t) ** 2, np.abs(r) ** 2


pulse = signal(time,pulse_width,pulse_delay,omega0)

ET_FT = Discrete_Fourier_Transform(ET,time,omegas)
ER_FT = Discrete_Fourier_Transform(ER,time,omegas)
pulse_FT = Discrete_Fourier_Transform(pulse,time,omegas)

R = np.abs(ER_FT) ** 2 / np.abs(pulse_FT) ** 2
T = np.abs(ET_FT) ** 2 / np.abs(pulse_FT) ** 2 * refractive_index[jT] / refractive_index[j_source]

# T_, R_ = thin_film_TR(1.0,np.sqrt(12.11),np.sqrt(4), wavelengths, thin_film_thickness)

plt.plot(wavelengths, R, 'blue')
# plt.plot(wavelengths, R_, 'x', color = 'blue')
# plt.plot(wavelengths, T, 'red')
# plt.plot(wavelengths, T_, 'x', color = 'red')
# plt.plot(wavelengths,1-R-T,"violet")
plt.xlabel("wavelengths (m)")
plt.ylabel("Spectrum")
# plt.legend(["Reflectance","R-analytic", "Transmittance","T-analytic","Absorption"])
plt.legend(["Reflectance", "Transmittance"])
plt.show()


frames = [] # for storing the generated images
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for i in range(len(E_movie)):
    im, = ax.plot(E_movie[i],color = 'red')
    frames.append([im])
ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True,
                                repeat_delay=1000)
plt.show()
