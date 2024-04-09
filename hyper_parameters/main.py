from HyperParameterFactory import HyperParameterFactory

def main():
    # Example of how to get hyper-parameters for a specific tuning method
    try:
        hyper_params = HyperParameterFactory.get_hyper_parameters("LoRA")
        print(hyper_params)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
