# dataset/main.py
from dataset import DatasetFactory

def main():
    # Example of how to use the DatasetFactory to get an image dataset
    try:
        dataset_loader = DatasetFactory.get_dataset("vision_transformer")
        dataset = dataset_loader.get_dataset()
        print(dataset)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
