import datetime
import os

def get_result_directory_name(timestamp: str,
                              model_arch_name: str,
                              batch_size,
                              optimizer_args: dict,
                              dataset_args: dict):
    # Model and Dataset first
    dirname = f'{timestamp}_model_{model_arch_name.rsplit(sep=".", maxsplit=1)[1]}'
    dirname += f'_dataset_{str(dataset_args["name"]).rsplit(".", maxsplit=1)[1].split("_")[0]}'

    # Training Hyperparams - Batch Size, Optimizer
    dirname += f'_bs_{batch_size}'
    for key, value in optimizer_args.items():
        dirname += f'_{key}_{value}'

    return dirname


def make_results_dir(arguments):
    """
    Create a timestamped output directory with training details
    :return: The output directory path.
    """
    optimizer_args = arguments['optimizer_args']
    outdir = arguments.get("outdir")
    model_arch_name = arguments.get('model_args').get('model_arch_name')

    train_data_args = arguments.get('train_data_args')
    batch_size = train_data_args.get("batch_size", 1)

    dirname = get_result_directory_name(timestamp = datetime.datetime.now().isoformat(),
                                        batch_size=batch_size,
                                        model_arch_name=model_arch_name,
                                        optimizer_args=optimizer_args,
                                        dataset_args=arguments['dataset_args'])

    outdir = os.path.join(outdir, dirname)
    os.makedirs(outdir, exist_ok=True)
    print("Output directory: %s", outdir)
    return outdir
