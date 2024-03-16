from SROM import SROMAeroS
#from DerivedExample import SROMAeroS1

#sROM = SROMAeroS1('../CF_Input4')
#sROM = SROMAeroS1('../CF_Input1')
sROM = SROMAeroS('../CF_Input7')
alpha = sROM.optimize()
#sol, mean_rom2 = sROM.postprocessing(alpha, 256)
#sROM.plot(sol, sROM.mean_ref, mean_rom2, 0, 0, 0.95, 0, 'plot_.pdf')
################################################################################
