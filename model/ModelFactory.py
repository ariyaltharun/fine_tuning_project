from .ModelLoader import ModelLoader

class ModelFactory:
  @staticmethod
  def create(model_type, peft_params=None):
    """
    Creates a model instance based on the specified type and PEFT parameters.

    Args:
      model_type: str, type of model to create (e.g., "bert", "vision_transformer").
      peft_params: dict, optional parameters for PEFT (if applicable).

    Returns:
      A Transformers model instance.
    """

    if model_type == "vision_transformer":
      return ModelLoader.load_tokenizer(model_type)
    elif model_type == "bert":
      return  ModelLoader.load_tokenizer(model_type)
    else:
      raise ValueError(f"Unknown model type: {model_type}")