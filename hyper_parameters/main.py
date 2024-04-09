from HyperParameterFactory import HyperParameterFactory

def main():
    # Example of how to get hyper-parameters for a specific tuning method
    try:
        peft_hyper_params = HyperParameterFactory.get_peft_parameters("LoRA",params = [])
        general_hyper_params = HyperParameterFactory.get_general_parameters(params = None)
        print(peft_hyper_params)
        print(general_hyper_params)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
