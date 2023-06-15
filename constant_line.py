import wandb

if __name__ == "__main__":
    hyperparameters = {
        'Mode': 'Standard C',
        'Weakly Error': 0,
        'Data Set': 'AG_NEWS',
        'Train Set': 32000,
        'Sampling Method': 'Random',
        'Init Sample Size': 32000,
        'N-Sample': 0,
        'Reset Model': 'False',
        'AL Iterations': '-',
        'Patience': 3,
        'P. Label Conf.': 1,
        'Delta Change': 0
    }
    with wandb.init(project='active-learning-plus', config=hyperparameters):
        strong_labels = 320
        for i in range(9):
            print(f'Iteration: {i}')
            wandb.log({'test accuracy': 0.9034, 'Strong Labels': strong_labels})
            print(strong_labels)
            strong_labels += 160
