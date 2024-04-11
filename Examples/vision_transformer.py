import sys
sys.path.append('S:/SiDhU/Codes/HP/actual_repo/fine_tuning_project/')
import dataset
import model
import hyper_parameters
import train
import preprocessing
import peft_techniques
model_name = "vision_transformer"
model_checkpoint = "google/vit-base-patch16-224-in21k"
data = dataset.DatasetFactory.get_dataset(model_name)
parameter = {
            "r": 16,
            "lora_alpha": 16,
            "target_modules": ["query", "value"], 
            "lora_dropout": 0.1,
            "bias": "none",
        }
gen_params = {
            "batch_size": 128,
            "learning_rate": 0.01,
            "num_train_epochs": 2,
        }

peft_params = hyper_parameters.HyperParameterFactory.get_peft_parameters("lora",parameter)
general_params = hyper_parameters.HyperParameterFactory.get_general_parameters(gen_params)
print(peft_params)
print(general_params)
vision_model = model.ModelFactory.create(model_name, data)
processor = preprocessing.PreprocessingFactory(model_checkpoint)
train_ds, val_ds = processor.get_image_preprocessor(data)

lora_model = peft_techniques.PeftFactory.create_peft_method("LoRA", peft_params)

args = hyper_parameters.HyperParameterFactory.get_general_parameters(general_params)

trainer = train.TrainFactory.get_trainer(model,args,train_ds,val_ds,processor)
train_results = trainer.train()
validation_results = trainer.evaluate(val_ds)


