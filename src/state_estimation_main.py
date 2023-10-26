from src.simulation_without_grad import *

fit_parameter = json.load(open(current_dir + result_dir + "/fitting_result.json"))
garment_E[None] = fit_parameter['E']
garment_nu[None] = fit_parameter['nu']
contact_stiffness[None] = fit_parameter['contact_stiffness']
shearing_stiffness[None] = fit_parameter['shearing_stiffness']
print("Initializing objects...")
initialize_objects()
print("Initialized objects")
animation(operator_update_interval * (n_actions - 1))
