import importlib
from datetime import datetime
from torchvision.utils import make_grid

from EasyPytorch.Vsualization import Logger


class TensorBoard:
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir="../Log", logger=None, enabled=True):
        self.writer = None
        self.selected_module = ""

        if enabled:
            if logger is None:
                logger = Logger(log_dir)
            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    self.selected_module = module
                    break
                except ImportError:
                    succeeded = False

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                          "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                          "version >= 1.1 to use 'torch.Utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    # def __getattr__(self, name):
    #     """
    #     If visualization is configured to use:
    #         return add_data() methods of tensorboard with additional information (step, tag) added.
    #     Otherwise:
    #         return a blank function handle that does nothing
    #     """
    #     if name in self.tb_writer_ftns:
    #         add_data = getattr(self.writer, name, None)
    #     else:
    #         # default action for returning methods defined in this class, set_step() for instance.
    #         try:
    #             attr = object.__getattr__(name)
    #         except AttributeError:
    #             raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
    #         return attr

    def add_scalar(self, key, value):
        self.writer.add_scalar(key, value, self.step)

    def add_image(self, name, images):
        self.writer.add_image(name, images)

    def add_histogram(self, name, p, bins='auto'):
        self.writer.add_histogram(name, p, bins=bins)

    def close(self):
        self.writer.close()