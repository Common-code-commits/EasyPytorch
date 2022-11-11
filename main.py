from os import walk



from os.path import join


from Config import Config





class Trainer:

    def __init__(self):
        self.config = Config()
        config = self.config.train_loader.args
        self.label = Trainer.get_label(config.data_dir)
        self.train_loader = self.get_dataloader()
        self.test_loader = self.get_dataloader(test=True)

        for data in self.test_loader:
            print(data)



    def get_dataloader(self, test=False):

        pass


if __name__ == "__main__":
    trainer = Trainer()
