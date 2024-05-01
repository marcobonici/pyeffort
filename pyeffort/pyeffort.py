from juliacall import Main as jl
import numpy as np

jl.seval("using Effort")
jl.seval("using SimpleChains")
jl.seval("using BSON")
jl.seval("using Static")
SimpleChain = jl.seval("SimpleChains.SimpleChain")
static = jl.seval("Static.static")
TurboDense = jl.seval("SimpleChains.TurboDense")
tanh = jl.seval("SimpleChains.tanh")
identity = jl.seval("SimpleChains.identity")

effort_compute_Pl = jl.seval('Effort.get_Pℓ')
#effort_compute_Xil = jl.seval('Effort.get_Xiℓ')
#effort_compute_Xil = jl.seval('Effort.get_Xiℓ')
effort_compute_f_z = jl.seval('Effort._f_z')
create_bin_edges = jl.seval('Effort.create_bin_edges')
get_stoch_terms_binned_efficient = jl.seval('Effort.get_stoch_terms_binned_efficient')
load_emu_jl = jl.seval('BSON.load')
SimpleChainsEmulator = jl.seval("Effort.SimpleChainsEmulator")
P11Emulator = jl.seval("Effort.P11Emulator")
PloopEmulator = jl.seval("Effort.PloopEmulator")
PctEmulator = jl.seval("Effort.PctEmulator")
PlEmulator = jl.seval("Effort.PℓEmulator")
get_stoch_terms = jl.seval("Effort.get_stoch_terms")



def load_component(component, l, folder, k_grid, sky):
    nk = len(k_grid)
    if component == "11":
        k_number = (nk)*3
    elif component == "loop":
        k_number = (nk)*12
    elif component == "ct":
        k_number = (nk)*6
    else:
        print("You didn't choose a viable component!")

    mlpd = SimpleChain(
      static(6),
      TurboDense(tanh, 64),
      TurboDense(tanh, 64),
      TurboDense(tanh, 64),
      TurboDense(tanh, 64),
      TurboDense(tanh, 64),
      TurboDense(identity, k_number)
    )

    weights = np.load(folder+"weights_P_"+component+"_lcdm_l_"+str(l)+"_sky_"+str(sky)+".npy")
    outMinMax = np.load(folder+"outMinMax_P_"+component+"_lcdm_l_"+str(l)+"_sky_"+str(sky)+".npy")
    inMinMax = np.load(folder+"inMinMax_lcdm_l_"+str(l)+"_sky_"+str(sky)+".npy")

    sc_emu = SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

    if component == "11":
        comp_emu = P11Emulator(TrainedEmulator = sc_emu, kgrid = k_grid,
               InMinMax=inMinMax, OutMinMax = outMinMax)
    elif component == "loop":
        comp_emu = PloopEmulator(TrainedEmulator = sc_emu, kgrid = k_grid,
               InMinMax=inMinMax, OutMinMax = outMinMax)
    elif component == "ct":
        comp_emu = PctEmulator(TrainedEmulator = sc_emu, kgrid = k_grid,
               InMinMax=inMinMax, OutMinMax = outMinMax)
    else:
        print("You didn't choose a viable component!")

    return comp_emu

def load_multipole(l, folder, k_grid, sky):
    P11 = load_component("11", l, folder, k_grid, sky)
    Ploop = load_component("loop", l, folder, k_grid, sky)
    Pct = load_component("ct", l, folder, k_grid, sky)
    emulator = PlEmulator(P11 = P11, Ploop = Ploop, Pct = Pct)
    return emulator

def compute_Pl(*args):
    my_list = [elem for elem in args]
    if len(my_list) == 3:
        for i in range(len(args)-1):
            my_list[i] = jl.collect(my_list[i])
    else:
         for i in range(len(args)-2):
            my_list[i] = jl.collect(my_list[i])
    Pl = effort_compute_Pl(*my_list)
    return np.array(Pl)

def compute_fz(z, Omm, w0, wa):
    return effort_compute_f_z(z, Omm, w0, wa)
