from multiprocessing import freeze_support
from tabulate import tabulate
from tools.configloader import ConfigLoader
from patient.patientgroup import PatientGroup
from config import config
from data import run_model
from tools.visualis import Visualize 

from manager.data import DataManager 
from tools.stats import print_group_stats
import tabulate
import os 
import psutil
  
print('process started with pid:', os.getpid())
 
# def print_process_info():
#     psUtilInfo = psutil.Process(os.getpid())
#     cpuPercentage = int(psUtilInfo.get_cpu_percent())
#     memoryInfo, _vms = psUtilInfo.get_memory_info()
# print_process_info()

# Importing patient groups
patient_groups = [
    PatientGroup(config.basepath.format(groupid=1)),
    PatientGroup(config.basepath.format(groupid=2)),
    PatientGroup(config.basepath.format(groupid=3)),
    PatientGroup(config.basepath.format(groupid=4)),
]

# print_group_stats(patient_groups)

if config.show_visualization:
    vv = Visualize(patient_groups,catagory = [1,2,3], patients=[1,2,3,4,5,6,7,8], exercises=['RF','AF'], bones=["thorax_r_x_ext", "thorax_r_y_ax", "thorax_r_z_lat",
                      "clavicula_r_y_pro", "clavicula_r_z_ele", "clavicula_r_x_ax"])
    vv.visualise(mode='exercise')
    vv.visualise(mode = 'idle') 

exit()
# Loading a list of configurations 
configloader = ConfigLoader(patient_groups, 'model_evaulation.json')

if __name__ == "__main__":
    freeze_support()
    configloader.clear_evaluation_result()
    # TODO: Calculate time between runs
    while configloader.next_config():
        print('Updating exercises based on config... ')
        configloader.update_exercises()

        dm = DataManager(*patient_groups)
        pipeline = dm.generate_pipeline()
        print("Based on the above configuration the following "
              "pipeline is created %s " % pipeline)
        dm.send_through(*pipeline)

        train, test = dm.create_split()
        print('Running the model....')
        score = run_model(train, test) 
        configloader.update_table(score)
    
    print('Finished running all configurations')
    configloader.print_table()

    # sorted_scores = list(sorted(scores, key=lambda x: x[0]))
    # print(tabulate(sorted_scores))

#   Accuracy       MCC    LogLoss      RSME     RMSLE  default
# ----------  --------  ---------  --------  --------  ---------
#   0.889488  0.752936    3.16644  0.894728  0.272647  True
#   0.731449  0.642798    3.45956  0.755038  0.224504  True