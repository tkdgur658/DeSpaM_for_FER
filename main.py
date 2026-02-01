from experiment import run_experiment
DATASETS = ['rafdb'] 
MODELS = [  
    'ProposedNet',
]
ITERATIONS = [1, 5]  
EPOCHS = 100
results = run_experiment(DATASETS, MODELS, ITERATIONS, EPOCHS)