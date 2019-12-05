import numpy as np

class MyGenerator(Sequence):
    """Custom generator"""

    def __init__(self, data_x, batch_size=8):
        """construction   

        :param data_x: numpy x_train/val(uint8)
        :param batch_size: Batch size
        """

        self.data_x = data_x
        self.length = self.data_x.shape[0]
        self.batch_size = batch_size
        self.num_batches_per_epoch = int((self.length - 1) / batch_size) + 1


    def __getitem__(self, idx):
        """Get batch data   

        :param idx: Index of batch  

        :return imgs: numpy array of images 
        :return labels: numpy array of label  
        """

        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > self.length:
            end_pos = self.length
        imgs = (self.data_x[start_pos : end_pos] / 255).astype('float32')
        return imgs, imgs


    def __len__(self):
        """Batch length"""

        return self.num_batches_per_epoch


    def on_epoch_end(self):
        """Task when end of epoch"""
        pass
