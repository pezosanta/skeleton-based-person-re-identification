import torch
from torch.utils.data import Dataset, DataLoader
import glob
import json
import numpy as np



database_path = "../../datasets/Andersson/kinect_gait_json_dataset/"
train_database_path = database_path + "train/"
val_database_path = database_path + "val/"
test_database_path = database_path + "test/"

train_json_path = glob.glob(train_database_path + "*.json")
val_json_path = glob.glob(val_database_path + "*.json")
test_json_path = glob.glob(test_database_path + "*.json")



class Andersson_windows_dataset(Dataset):
    def __init__(self, mode, window_size=3):
        self.mode = mode    
        self.window_size = window_size
        self.keypoints_num = 20

        if self.mode == 'train':
            self.json_path = train_json_path
        elif self.mode == 'val':
            self.json_path = val_json_path
        elif self.mode == 'test':
            self.json_path = test_json_path

    
    def _create_sliding_window(self, annotations):
        annotations.sort(key=lambda x: x["id"])

        # Min-Max normalizing 
        xs = np.array([annotation["keypoints"][0::3] for annotation in annotations]).flatten()# / 1920
        ys = np.array([annotation["keypoints"][1::3] for annotation in annotations]).flatten()# / 1080

        # Concatenating xs and ys in the following order: xs[0], ys[0], xs[1], ys[1], ...
        xys = np.ravel([xs,ys],'F')

        # Creating an array of sliding windows with shape of [num_frames, window_size, self.keypoints_num*2]
        # Using self.keypoints*2 due to the fact that a keypoint has 2 (x,y) coordinates
        windows = np.array([xys[i:(i+self.window_size*self.keypoints_num*2)] for i in range(0, (len(xys)-self.window_size*self.keypoints_num*2+1), (self.keypoints_num*2))])
        windows = windows.reshape([-1, self.window_size, self.keypoints_num*2])

        '''
        windows = []
        for i in range(0, (len(xs)-self.window_size*self.keypoints_num+1), self.keypoints_num):
            window = [xs[i:(i+self.window_size*self.keypoints_num)], ys[i:(i+self.window_size*self.keypoints_num)]]
            windows.append(window)
        '''
        
        return windows        


    def __getitem__(self, index):
        current_json_file = self.json_path[index]

        with open(current_json_file) as f:
            current_dictionary = json.load(f)

        annotations = current_dictionary["annotations"]

        track_id = annotations[0]["track_id"]
        numpy_windows = self._create_sliding_window(annotations=annotations)
        
        return numpy_windows, track_id
        
                
    def __len__(self): 
        return len(self.json_path)


class Andersson_iterating_dataset(Dataset):
    def __init__(self, windows, track_id): 

        self.windows = windows[0]
        self.track_id = int(track_id)
        self.track_num = 170
        self.device = torch.device('cuda:0')
        #self.device = torch.device('cpu')


    def _transform(self, numpy_current_window, track_id):
                      
        tensor_current_window = torch.as_tensor(numpy_current_window, dtype=torch.float32).to(device=self.device)
        
        numpy_one_hot_mask = np.zeros(self.track_num, dtype=int)
        numpy_one_hot_mask[track_id] = 1
        #tensor_one_hot_mask = torch.as_tensor(numpy_one_hot_mask, dtype=torch.float32).to(device=self.device)
        tensor_one_hot_mask = torch.as_tensor(numpy_one_hot_mask, dtype=torch.long).to(device=self.device)               

        return tensor_current_window, tensor_one_hot_mask


    def __getitem__(self, index):
        numpy_current_window = self.windows[index]

        tensor_current_window, tensor_one_hot_mask = self._transform(numpy_current_window, self.track_id)
        
        return tensor_current_window, tensor_one_hot_mask
                
    def __len__(self): 
        return self.windows.shape[0]

if __name__ == "__main__":

    dataset = Andersson_windows_dataset(mode='test', window_size=10)
    numpy_windows, track_id = dataset[1]
    print(len(test_json_path))
    print(numpy_windows.shape)
    print(track_id)
    print(numpy_windows[1])

    loader = DataLoader(dataset, batch_size = 1, shuffle = True)
    it = iter(loader)
    windows, track_id = next(it)
    #print(windows[0].shape)
    #print(track_id)
    #print(int(track_id))

    dataset2 = Andersson_iterating_dataset(windows=windows, track_id=track_id)
    tensor_window, tensor_label = dataset2[0]

    loader = DataLoader(dataset2, batch_size = 64, shuffle = True)
    print(len(loader))

    from model import LSTM

    model = LSTM(input_size=40, hidden_size=256, num_classes=170, n_layers=16).to(device=torch.device('cuda:0'))

    it = iter(loader)
    tensor_window, tensor_label = next(it)
    output = model(tensor_window)
    tensor_window, tensor_label = next(it)
    output = model(tensor_window)




