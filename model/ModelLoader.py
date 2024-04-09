from transformers import AutoTokenizer
from transformers import AutoModelForImageClassification
import dataset
class ModelLoader:
  @staticmethod
  def load_tokenizer(model_type):
    """
    Loads a tokenizer corresponding to the specified model type.

    Args:
      model_type: str, type of model (e.g., "bert", "vision_transformer").

    Returns:
      A Transformers tokenizer instance.
    """

    if model_type == "vision_transformer":
      try:
        dataset_loader = dataset.DatasetFactory.get_dataset(model_type)
        dataset = dataset_loader.get_dataset()
      except ValueError as e:
        print(f"Error: {e}")

      model_name = "google/vit-base-patch16-224-in21k"
      labels = dataset.features["label"].names
      label2id, id2label = dict(), dict()
      for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label
    
      model = AutoModelForImageClassification.from_pretrained(
            model_name,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
      return model
    
    else:
      raise ValueError(f"Unknown model type: {model_type}")



# from transformers import AutoImageProcessor
# image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)