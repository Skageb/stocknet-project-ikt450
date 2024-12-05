import json
import os
import matplotlib.pyplot as plt
import numpy as np
import inspect

def log_results(experiment_name:str, log_obj:dict):
    '''experiment_name is the name the file will be stored with. Suggested as f"model_{model_class}_dataset_{dataset_class}". The name gets "_{id}.json" appended'''
    root = 'results/'
    result_dir = os.path.join(root, experiment_name)
    os.makedirs(result_dir)

    #Create new id with 4 digits incrementally
    dir_ids = [int(path.split(".")[-2].split('_')[-1]) for path in os.listdir(result_dir)]
    new_id = str(max(dir_ids)) + 1 if len(dir_ids) > 0 else '0'     #Increment max id by 1 or set to 0 if no id present
    id = '0'*(4-len(new_id)) + new_id    #Make id 4 digits

    target_file = f'{result_dir}_{id}.json'
    with open(target_file, 'w') as f:
        json.dump(log_obj)

def write_log_to_file(experiment_name:str, log_obj:dict):
    '''experiment_name is the name the file will be stored with. Suggested as f"model_{model_class}_dataset_{dataset_class}". The name gets "_{id}.json" appended'''
    root = 'results/'
    result_dir = os.path.join(root, experiment_name)
    
    if not result_dir.split('/')[-1] in os.listdir(root):
        os.makedirs(result_dir)

    #Create new id with 4 digits incrementally
    dir_ids = [int(path.split(".")[-2].split('_')[0]) for path in os.listdir(result_dir)]
    new_id = str(max(dir_ids)+ 1) if len(dir_ids) > 0 else '0'     #Increment max id by 1 or set to 0 if no id present
    id = '0'*(4-len(new_id)) + new_id    #Make id 4 digits

    target_file = os.path.join(result_dir, f'{id}.json')
    with open(target_file, 'w') as f:
        json.dump(log_obj, f, indent=4)
    
    return target_file


def log_config(log_object, config):
    config_to_log = {}
    for key, value in vars(config).items():
        #print(key, inspect.isclass(value), inspect.isfunction(value))
        if inspect.isclass(value) or inspect.isfunction(value): #or isinstance(value, types.FunctionType):  # Check if it's a class instance
            config_to_log[key] = value.__name__  # Log the class name
            #print(config_to_log[key])
        elif isinstance(value, np.ndarray):
            config_to_log[key] = value.tolist()
        else:
            config_to_log[key] = value  # Log the value directly for primitive types
    log_object['Config'] = config_to_log
    #Rearrange dict so config comes after dataset and model
    log_object = {k: log_object[k] for k in list(log_object.keys())[:2] + ['Config'] + list(log_object.keys())[2:-1]}
    return log_object


def logable_config(config):
    config_to_log = {}
    for key, value in vars(config).items():
        #print(key, inspect.isclass(value), inspect.isfunction(value))
        if inspect.isclass(value) or inspect.isfunction(value): #or isinstance(value, types.FunctionType):  # Check if it's a class instance
            config_to_log[key] = value.__name__  # Log the class name
            #print(config_to_log[key])
        elif isinstance(value, np.ndarray):
            config_to_log[key] = value.tolist()
        else:
            config_to_log[key] = value  # Log the value directly for primitive types
    return config_to_log


def plot_lines(y_values, x_values=None, x_label='', y_label='', title='', legends=None, path=None):
    """
    Plots line graphs from one or more arrays. (This function was generated with GPT-o1-preview)

    Parameters:
    ----------
    y_values : array-like or list of array-like
        The y-values to plot. Can be a single array or a list of arrays for multiple lines.
    x_values : array-like or list of array-like, optional
        The x-values corresponding to y-values. If None, defaults to indices starting from 1.
        If provided, should match the length of y_values.
    x_label : str, optional
        Label for the x-axis.
    y_label : str, optional
        Label for the y-axis.
    title : str, optional
        Title of the plot.
    legends : list of str, optional
        Legend labels for each line. If None, defaults to 'Line 1', 'Line 2', etc.
    path : str, optional
        File path to save the plot image. If provided, the plot is saved to this path.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.

    Example:
    -------
    # Plot a single line
    y = [1, 2, 3, 4, 5]
    fig = plot_lines(y, y_label='Value', title='Single Line Plot')

    # Plot multiple lines with custom x-values and legends
    y1 = [1, 2, 3, 4, 5]
    y2 = [2, 3, 4, 5, 6]
    x = [0, 1, 2, 3, 4]
    fig = plot_lines([y1, y2], x_values=[x, x], legends=['Dataset 1', 'Dataset 2'], 
                     x_label='Time', y_label='Value', title='Multiple Lines')
    """
    import matplotlib.pyplot as plt

    # Ensure y_values is a list
    if not isinstance(y_values, list):
        y_values = [y_values]

    # Handle x_values
    if x_values is None:
        x_values = [range(1, len(y)+1) for y in y_values]
    else:
        if not isinstance(x_values, list):
            x_values = [x_values]
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have the same number of elements.")
        # Check that each x and y pair have the same length
        for x, y in zip(x_values, y_values):
            if len(x) != len(y):
                raise ValueError("Each x and y pair must have the same length.")

    # Handle legends
    if legends is None:
        legends = [f'Line {i+1}' for i in range(len(y_values))]
    else:
        if len(legends) != len(y_values):
            raise ValueError("Length of legends must match number of lines.")

    # Create the plot
    fig, ax = plt.subplots()
    for x, y, legend in zip(x_values, y_values, legends):
        ax.plot(x, y, label=legend)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if legends:
        ax.legend()
    ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plot if path is provided
    if path:
        plt.savefig(path)

    return fig


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


if __name__ == "__main__":
    # Plot test 1
    y = np.random.rand(10)

    # Plotting
    fig = plot_lines(y, y_label='Random Value', title='Random Values Over Time')
    plt.show()

    #Plot test 2
    # Sample data
    y1 = np.random.rand(10)
    y2 = np.random.rand(10)
    x = np.linspace(0, 9, 10)

    # Plotting
    fig = plot_lines(
        y_values=[y1, y2],
        x_values=[x, x],
        x_label='Time',
        y_label='Value',
        title='Comparison of Two Random Datasets',
        legends=['Dataset 1', 'Dataset 2']
    )
    plt.show()

    #Plot test 3
    import numpy as np

    # Sample data
    y = np.sin(np.linspace(0, 2 * np.pi, 100))

    # Plotting and saving
    fig = plot_lines(y, x_label='Angle [rad]', y_label='sin(x)', title='Sine Wave', path='sine_wave.png')


