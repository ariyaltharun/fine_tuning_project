from dataset import DatasetFactory
from PreprocessingLoader import PreprocessingLoader

def main():
    try:
        dataset_loader = DatasetFactory.get_dataset("vision_transformer")
        dataset = dataset_loader.get_dataset()
        
        splits = dataset.train_test_split(test_size=0.1)
        train_ds = splits["train"]
        val_ds = splits["test"]

        model_checkpoint = "google/vit-base-patch16-224-in21k"
        preprocessing_loader = PreprocessingLoader(model_checkpoint)
        preprocessor = preprocessing_loader.get_preprocessor()
        train_ds.set_transform(preprocessor.preprocess_train)
        val_ds.set_transform(preprocessor.preprocess_val)
    except ValueError as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
