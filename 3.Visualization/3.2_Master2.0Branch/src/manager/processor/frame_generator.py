import numpy as np
from manager.processor.processor_interface import ProcessorInterface


class GenerateFrameProcessor(ProcessorInterface):
    def handle(self):
        np_combination_array = np.empty((0, len(self.config.columns) * self.config.frames_counts * 5))
        for exercise_combination in self.data:
            if exercise_combination[0].patientgroup in []:
                print('w0t')
            # if exercise_combination[0].patientgroup in ['2', '3']:
                # # Creating 5 empty array's
                # data_array = [np.array([]) for _ in range(len(exercise_combination))]
                #
                # for exercise_id in range(len(exercise_combination)):
                #     for exercise_frame in exercise_combination[exercise_id].np_frames:
                #         # Losse oefening [30]
                #         exercise_flat = exercise_frame.reshape(1, len(self.config.columns) * self.config.frames_counts)
                #         data_array[exercise_id] = np.append(data_array[exercise_id], exercise_flat[0])
                #
                # for data in data_array:
                #     np_combination_array = np.vstack([np_combination_array, data])
            else:
                data = np.array([])
                for exercise in exercise_combination:
                    # Getting 5 frames from exercise 
                    exercise_flat = exercise.np_data.reshape(1, len(self.config.columns) * self.config.frames_counts)
                    data = np.append(data, exercise_flat[0])

                np_combination_array = np.vstack([np_combination_array, data])

        return np_combination_array
