class LoRAHyperparameters:
    def _init_(self):
        # Essential LoRA configuration
        self.task_type = "SEQ_CLS"  # Task type for adapter configuration
        self.r = 4  # Number of adapters for each layer
        self.lora_alpha = 32  # Hidden dimension of adapters
        self.lora_dropout = 0.01  # Dropout applied within adapters
        self.target_modules = ['query']  # Model layers to add adapters to

        # General model hyperparameters
        self.learning_rate = 2e-5
        self.batch_size = 16
        self.num_epochs = 3

        # Weight decay for regularization
        self.weight_decay = 0.01

        # Additional options for flexibility
        self.evaluation_strategy = "epoch"
        self.save_strategy = "epoch"
        self.load_best_model_at_end = True

    def create_training_args(self, output_dir):
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            load_best_model_at_end=self.load_best_model_at_end,
        )