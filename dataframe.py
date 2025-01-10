from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class Dataframe(Dataset):
    def __init__(
        self,
        df,
        label_to_id,
        train=False,
        text_field="descripcion",
        label_field="tipo",
        folder_field="Real",

    ):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.train = train
        self.text_field = text_field
        self.label_field = label_field
        self.folder_field = folder_field

    @property
    def get_number_labels(self):
        return len(self.label_to_id)
    
    def __getitem__(self, index):
        text = str(self.df.at[index, self.text_field])
            
        label = self.label_to_id[self.df.at[index, self.label_field]]

        return text, label

    def __len__(self):
        return self.df.shape[0]