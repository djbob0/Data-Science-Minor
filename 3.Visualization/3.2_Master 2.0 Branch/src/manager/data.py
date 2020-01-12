from manager.processor import *
from manager.train_test import TrainTestManager
import copy

from config import config 


class DataManager:
    def __init__(self, *args):
        """
        :param args: categories to load
        :param config: Machine learning configuration
        """
        ProcessorRules(config())  # Verify the given configuration

        self.categories = args
        self.m_train_test = TrainTestManager(self.categories)

    def generate_pipeline(self) -> list:
        """
        Generate a processing pipeline
        based on the given configuration

        :return: processor pipeline
        :rtype: list
        """
        pipeline = [InitialProcessor]
        if config().occupied_space:
            pipeline.append(OccupiedSpaceProcessor)
            return pipeline

        if config().remove_idle:
            pipeline.append(RemoveIdlePreProcessor)

        if config().resample_exercise:
            pipeline.append(ResamplePreProcessor)

        pipeline.append(GenerateCombinationsProcessor)

        if config().frame_generator:
            pipeline.append(GenerateFrameProcessor)
        else:
            pipeline.append(DataFinalizationProcessor)

        return pipeline

    def send_through(self, *args):
        """
        Send the patient's exercises through the
        Specified processors. Results will be written
        Into <pat obj>.processed

        :param args: Processors
        """
        for cat in self.categories:
            for pat in cat:
                pat.processed = copy.deepcopy(pat.get_exercises())
                for processor in args:
                    pat.processed = (processor(pat.processed, config())).handle()

    """
    Facades to remove some clutter
    of the data manager
    """
    def create_split(self):
        return self.m_train_test.create_split()

    def create_percentage_based_split(self):
        return self.m_train_test.create_percentage_based_split()
