from manager.processor.processor_interface import ProcessorInterface


class InitialProcessor(ProcessorInterface):
    def handle(self):
        for exercise in self.data:
            exercise.update_config()

        return self.data
