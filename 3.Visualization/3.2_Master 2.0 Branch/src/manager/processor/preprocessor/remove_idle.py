from manager.processor.processor_interface import ProcessorInterface
from tools.remove_idle import RemoveIdle


class RemoveIdlePreProcessor(ProcessorInterface):
    def handle(self):
        for exercise in self.data:
            exercise.idle = RemoveIdle(exercise)
            exercise.dataframe = exercise.idle.df

        return self.data
