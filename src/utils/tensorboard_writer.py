import os

from torch.utils.tensorboard import SummaryWriter

from src.utils import logger


class TensorboardWriter:

    def __init__(self, outdir):
        assert (os.path.isdir(outdir))
        self.outdir = outdir
        self.writer = SummaryWriter(self.outdir, flush_secs=10)

    def save_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def save_scalars(self, main_tag, scalars_dict, global_step=None):
        self.writer.add_scalars(main_tag, scalars_dict, global_step)

    def save_image(self, tag, image, global_step=None, dataformats='CHW'):
        self.writer.add_image(tag=tag,
                              img_tensor=image,
                              global_step=global_step,
                              dataformats=dataformats)

    def save_figure(self, tag, figure, global_step=None, close=False):
        self.writer.add_figure(tag=tag,
                               figure=figure,
                               global_step=global_step,
                               close=close)

    def save_graph(self, model, inputs_to_model=None):
        """
        Saves graph to the tensorboard. Ideally call once.
        :param model: The torch.nn.Module object
        :param inputs_to_model: tensor or a list of tensor(batch) will also be showed.
        :return: None
        """
        try:
            self.writer.add_graph(model, inputs_to_model)
        except Exception as e:
            print('Check this for fix: https://github.com/lanpa/tensorboardX/issues/389#issuecomment-475879228')

    def flush(self):
        """
        If you need to flush all data immediately.
        """
        self.writer._get_file_writer().flush()

    def close(self):
        """
        To be called in the end
        """
        self.writer.close()


tensorboard_writer = None


def initialize_tensorboard(outdir) -> TensorboardWriter:
    global tensorboard_writer
    if tensorboard_writer is None:
        logger.info(f'Initializing Tensorboard Writer in {outdir}')
        tensorboard_writer = TensorboardWriter(outdir)
    return tensorboard_writer


def close_tensorboard():
    global tensorboard_writer
    tensorboard_writer.close()
    tensorboard_writer = None


if __name__ == '__main__':
    import torch
    import torchvision
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import datasets, transforms

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('../data/', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model = torchvision.models.resnet50(False)
    # Have ResNet model take in grayscale rather than RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    images, labels = next(iter(trainloader))

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)

    writer.add_text('lstm', 'This is an lstm', 0)
    writer.add_scalar('loss', torch.tensor(2.0), global_step=0)
    writer.add_scalar('loss', torch.tensor(1.0), global_step=0.1)
    writer.close()

