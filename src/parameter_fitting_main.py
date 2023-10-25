from src.parameter_fitting_functions import *

garment_E[None] = initial_garment_E
garment_nu[None] = initial_garment_nu
contact_stiffness[None] = initial_contact_stiffness
shearing_stiffness[None] = initial_shearing_stiffness

print("Initializing objects...")
initialize_objects()
print("Initialized objects")

time0 = time.time()
parameters_learning(True, True, 35)
time1 = time.time()

result_file = open(result_file_path)
result = json.load(result_file)
result['E'] = garment_E[None]
result['nu'] = garment_nu[None]
result['contact_stiffness'] = contact_stiffness[None]
result['shearing_stiffness'] = shearing_stiffness[None]
with open(result_file_path, 'w') as f:
    json.dump(result, f, indent=1)
