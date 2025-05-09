# Model Training Configuration
config = {
    'image_size': 224,
    'batch_size': 4,
    'num_epochs': 100,
    'learning_rate': 0.00001,
    'optimizer': 'Adam',
    'weight_decay': 1e-5,
    'scheduler': 'StepLR',
    'step_size': 10,
    'gamma': 0.5,
    'num_classes': 2,
    'device': 'cuda'  # or 'cpu'

}
