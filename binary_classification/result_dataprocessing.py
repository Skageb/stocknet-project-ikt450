from utils import plot_lines
import json

def generate_training_plot_from_file(path:str):
    '''Produce a loss graph of a provided experiment result file'''
    with open(path, 'r') as f:
        json_obj = json.load(f)
    loss = json_obj['Report from Training']['loss_across_epochs']
    eval_accuracy = json_obj['Report from Training']['eval_accuracy_per_epoch']
    train_accuracy = json_obj['Report from Training']['train_accuracy_per_epoch']

    loss_plot = plot_lines([loss], 
                           x_label='Epochs', 
                           y_label=json_obj['Config']['loss_func'], 
                           title=f'Loss Plot, Dataset: {json_obj["Dataclass"]} Model: {json_obj["Model"]}', 
                           path=path.replace('.json', '_loss.jpg'))
    
    accuracy_plot = plot_lines([eval_accuracy, train_accuracy],
                                x_label='Epochs', 
                                y_label='Accuracy', 
                                title=f'Accuracy Plot, Dataset: {json_obj["Dataclass"]} Model: {json_obj["Model"]}', 
                                path=path.replace('.json', '_accuracy.jpg'),
                                legends=['Eval Accuracy', 'Train Accuracy'])
    




if __name__ == '__main__':
    experiment_path = '/home/skage/projects/ikt450_deep-neural-networks/stocknet-project-ikt450/skage/results/model_RNN_simple_dataset_TweetXPriceY/0001.json'
    generate_training_plot_from_file(experiment_path)