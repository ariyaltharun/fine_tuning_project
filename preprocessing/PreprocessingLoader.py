from PreprocessingFactory import ImagePreprocessor

class PreprocessingLoader:
    def __init__(self, model_checkpoint):
        self.preprocessor = ImagePreprocessor(model_checkpoint)

    def get_preprocessor(self):
        return self.preprocessor
