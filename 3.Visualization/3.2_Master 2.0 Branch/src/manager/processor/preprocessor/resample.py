from manager.processor.processor_interface import ProcessorInterface
from tools.resample import resample_excercise


class ResamplePreProcessor(ProcessorInterface):
    def handle(self):
        for exercise in self.data:
            exercise.dataframe = resample_excercise(exercise.dataframe, self.config.frames_counts)

        return self.data
