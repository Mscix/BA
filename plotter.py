import wandb
import matplotlib.pyplot as plt


def average_loss_per_iteration():
    pass

def colored_chart_per_iteration():
    pass


def standard_chart(x=None, y=None, x_label='X-Axis', y_label='Y-Axis', title='Chart'):
    if x and y:
        pass
    elif not x and y:
        x = [i for i in range(len(y))]
    else:
        raise ValueError('Wrong values, can not plot.')
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    wandb.log({title: wandb.Image(plt)})



