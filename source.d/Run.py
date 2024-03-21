from SROM import SROMAeroS

sROM = SROMAeroS('../input0.d')
alpha = sROM.optimize()
sol, mean_rom = sROM.postprocessing(alpha, 256)
################################################################################
