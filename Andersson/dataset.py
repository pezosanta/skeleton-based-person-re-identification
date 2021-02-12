import torch
from torch.utils.data import Dataset, DataLoader
import glob
import json
import numpy as np



database_path = "../../../datasets/Andersson/kinect_gait_json_dataset/"
train_database_path = database_path + "train/"
val_database_path = database_path + "val/"
test_database_path = database_path + "test/"

train_json_path = glob.glob(train_database_path + "*.json")
val_json_path = glob.glob(val_database_path + "*.json")
test_json_path = glob.glob(test_database_path + "*.json")



class Andersson_dataset(Dataset):
    def __init__(self, mode, window_size=3):
        self.mode = mode    
        self.window_size = window_size
        self.keypoints_num = 20
        self.track_num = 170
        self.device = torch.device('cuda:0')

        if self.mode == 'train':
            self.json_path = train_json_path
        elif self.mode == 'val':
            self.json_path = val_json_path
        elif self.mode == 'test':
            self.json_path = test_json_path

        self.numpy_windows = None
        self.numpy_one_hot_masks = None

        self._create_dataset()

    

    def _create_dataset(self):
        #print(len(self.json_path))
        # TODO Change range if RAM has been upgraded
        for i, json_file in enumerate(self.json_path[50:150]):
            print(json_file)
            with open(json_file) as f:
                current_dictionary = json.load(f)
            
            current_annotations = current_dictionary["annotations"]
            current_track_id = int(current_annotations[0]["track_id"])

            current_numpy_windows, current_numpy_one_hot_mask = self._create_sliding_window(current_annotations, current_track_id)

            if self.numpy_windows is None and self.numpy_one_hot_masks is None:
                self.numpy_windows = current_numpy_windows
                self.numpy_one_hot_masks = current_numpy_one_hot_mask
            else:
                self.numpy_windows = np.concatenate((self.numpy_windows, current_numpy_windows), axis=0)
                self.numpy_one_hot_masks = np.concatenate((self.numpy_one_hot_masks, current_numpy_one_hot_mask), axis=0)

            #print(self.numpy_windows.shape)
            #print(self.numpy_one_hot_masks.shape)

    
    def _create_sliding_window(self, annotations, track_id):
        # Sorting the list of dictionaries by x["id"] (ascending order)
        annotations.sort(key=lambda x: x["id"])

        # Min-Max normalizing 
        xs = np.array([annotation["keypoints"][0::3] for annotation in annotations]).flatten() / 1920
        ys = np.array([annotation["keypoints"][1::3] for annotation in annotations]).flatten() / 1080
        
        # Concatenating xs and ys in the following order: xs[0], ys[0], xs[1], ys[1], ...
        xys = np.ravel([xs,ys],'F')

        # Creating an array of sliding numpy_windows with shape of [num_frames, window_size, self.keypoints_num*2]
        # Using self.keypoints*2 due to the fact that a keypoint has 2 (x,y) coordinates
        numpy_windows = np.array([xys[i:(i+self.window_size*self.keypoints_num*2)] for i in range(0, (len(xys)-self.window_size*self.keypoints_num*2+1), (self.keypoints_num*2))])
        numpy_windows = numpy_windows.reshape([-1, self.window_size, self.keypoints_num*2])

        numpy_one_hot_mask = np.zeros((numpy_windows.shape[0], self.track_num), dtype=int)
        numpy_one_hot_mask[:, track_id] = 1

        return numpy_windows, numpy_one_hot_mask

    

    def _transform(self, current_numpy_window, current_numpy_one_hot_mask):
                      
        tensor_current_window = torch.as_tensor(current_numpy_window, dtype=torch.float32).to(device=self.device)
        
        tensor_one_hot_mask = torch.as_tensor(current_numpy_one_hot_mask, dtype=torch.float32).to(device=self.device)
        #tensor_one_hot_mask = torch.as_tensor(current_numpy_one_hot_mask, dtype=torch.long).to(device=self.device)               

        return tensor_current_window, tensor_one_hot_mask

    

    def __getitem__(self, index):
        current_numpy_window = self.numpy_windows[index]
        current_numpy_one_hot_mask = self.numpy_one_hot_masks[index]

        tensor_current_window, tensor_one_hot_mask = self._transform(current_numpy_window, current_numpy_one_hot_mask)
        
        return tensor_current_window, tensor_one_hot_mask

    

    def __len__(self): 
        return self.numpy_windows.shape[0]


if __name__=="__main__":
    ds = Andersson_dataset(mode="train", window_size=80)
    print(len(ds))
    
