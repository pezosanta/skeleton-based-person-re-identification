import torch
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
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

train_json_path = natsorted(train_json_path)#[0:54]
val_json_path = natsorted(val_json_path)#[0:18]
test_json_path = natsorted(test_json_path)



class Andersson_dataset(Dataset):
    def __init__(self, mode, window_size=3, log_reg=False):
        self.mode = mode
        self.log_reg = log_reg
        self.step_size = 10   
        self.window_size = window_size
        self.keypoints_num = 39#20
        self.track_num = 170
        self.device = torch.device('cuda:0')

        self.tree_structure = [ 10,1,0,1,3,5,7,9,7,5,3,1,
                                2,4,6,8,6,4,2,1,10,11,13,
                                15,17,19,17,15,13,11,12,14,
                                16,18,16,14,12,11,10 ]

        if self.mode == 'train':
            self.json_path = train_json_path
        elif self.mode == 'val':
            self.json_path = val_json_path
        elif self.mode == 'test':
            self.json_path = test_json_path

        self.numpy_windows = None
        self.numpy_one_hot_masks = None

        self._create_dataset()

        self.length = self._get_length()

    

    def _create_dataset(self):
        print(len(self.json_path))
      
        for i, json_file in enumerate(self.json_path):
            print(i)

            with open(json_file) as f:
                current_dictionary = json.load(f)
            
            current_annotations = current_dictionary["annotations"]
            current_track_id = int(current_annotations[0]["track_id"])
            
            current_numpy_windows, current_numpy_one_hot_mask = self._create_sliding_window(current_annotations, current_track_id)

            if self.mode == "train":
                if i < (len(self.json_path)/4):
                    if self.numpy_windows is None and self.numpy_one_hot_masks is None:
                        self.numpy_windows = [current_numpy_windows]
                        self.numpy_one_hot_masks = [current_numpy_one_hot_mask]
                    else:
                        self.numpy_windows[0] = np.concatenate((self.numpy_windows[0], current_numpy_windows), axis=0)
                        self.numpy_one_hot_masks[0] = np.concatenate((self.numpy_one_hot_masks[0], current_numpy_one_hot_mask), axis=0)
                elif (len(self.json_path)/4) <= i < (2*len(self.json_path)/4):
                    if len(self.numpy_windows) == 1 and len(self.numpy_one_hot_masks) == 1:
                        self.numpy_windows.append(current_numpy_windows)
                        self.numpy_one_hot_masks.append(current_numpy_one_hot_mask)
                    else:
                        self.numpy_windows[1] = np.concatenate((self.numpy_windows[1], current_numpy_windows), axis=0)
                        self.numpy_one_hot_masks[1] = np.concatenate((self.numpy_one_hot_masks[1], current_numpy_one_hot_mask), axis=0)
                elif (2*len(self.json_path)/4) <= i < (3*len(self.json_path)/4):
                    if len(self.numpy_windows) == 2 and len(self.numpy_one_hot_masks) == 2:
                        self.numpy_windows.append(current_numpy_windows)
                        self.numpy_one_hot_masks.append(current_numpy_one_hot_mask)
                    else:
                        self.numpy_windows[2] = np.concatenate((self.numpy_windows[2], current_numpy_windows), axis=0)
                        self.numpy_one_hot_masks[2] = np.concatenate((self.numpy_one_hot_masks[2], current_numpy_one_hot_mask), axis=0)
                else:
                    if len(self.numpy_windows) == 3 and len(self.numpy_one_hot_masks) == 3:
                        self.numpy_windows.append(current_numpy_windows)
                        self.numpy_one_hot_masks.append(current_numpy_one_hot_mask)
                    else:
                        self.numpy_windows[3] = np.concatenate((self.numpy_windows[3], current_numpy_windows), axis=0)
                        self.numpy_one_hot_masks[3] = np.concatenate((self.numpy_one_hot_masks[3], current_numpy_one_hot_mask), axis=0)

            else:
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
        xs = np.array([np.array(annotation["keypoints"][0::3])[self.tree_structure] for annotation in annotations]).flatten() / 1920
        ys = np.array([np.array(annotation["keypoints"][1::3])[self.tree_structure] for annotation in annotations]).flatten() / 1080

        # Concatenating xs and ys in the following order: xs[0], ys[0], xs[1], ys[1], ...
        xys = np.ravel([xs,ys],'F')

        # Creating an array of sliding numpy_windows with shape of [num_frames, window_size, self.keypoints_num*2]
        # Using self.keypoints*2 due to the fact that a keypoint has 2 (x,y) coordinates
        numpy_windows = np.array([xys[i:(i+self.window_size*self.keypoints_num*2)] for i in range(0, (len(xys)-self.window_size*self.keypoints_num*2+1), (self.keypoints_num*2))])
        numpy_windows = numpy_windows.reshape([-1, self.window_size, self.keypoints_num*2])
        
        if self.log_reg:
            numpy_windows_mean = np.mean(numpy_windows, axis=1)
            numpy_windows_var = np.var(numpy_windows, axis=1)
        
            numpy_windows = np.stack((numpy_windows_mean, numpy_windows_var), axis=2)
            
        numpy_one_hot_mask = np.zeros((numpy_windows.shape[0], self.track_num), dtype=int)
        numpy_one_hot_mask[:, track_id] = 1
        
        return numpy_windows, numpy_one_hot_mask

    

    def _transform(self, current_numpy_window, current_numpy_one_hot_mask):
                      
        tensor_current_window = torch.as_tensor(current_numpy_window, dtype=torch.float32).to(device=self.device)
        
        tensor_one_hot_mask = torch.as_tensor(current_numpy_one_hot_mask, dtype=torch.float32).to(device=self.device)
        #tensor_one_hot_mask = torch.as_tensor(current_numpy_one_hot_mask, dtype=torch.long).to(device=self.device)               

        return tensor_current_window, tensor_one_hot_mask


    
    def _get_length(self):
        if self.mode == "train":
            length = 0
            for windows in self.numpy_windows:
                length += len(windows)
            return length
        else:
            return(len(self.numpy_windows))

    

    def __getitem__(self, index):        
        if self.mode == "train":
            if index < len(self.numpy_windows[0]):
                #print(f"array: 0 | index: {index}")
                current_numpy_window = self.numpy_windows[0][index]
                current_numpy_one_hot_mask = self.numpy_one_hot_masks[0][index]
            elif index < (len(self.numpy_windows[0]) + len(self.numpy_windows[1])):
                #print(f"array: 1 | index: {index-len(self.numpy_windows[0])}")
                current_numpy_window = self.numpy_windows[1][index-len(self.numpy_windows[0])]
                current_numpy_one_hot_mask = self.numpy_one_hot_masks[1][index-len(self.numpy_one_hot_masks[0])]
            elif index < (len(self.numpy_windows[0]) + len(self.numpy_windows[1]) + len(self.numpy_windows[2])):
                #print(f"array: 2 | index: {index-len(self.numpy_windows[0])-len(self.numpy_windows[1])}")
                current_numpy_window = self.numpy_windows[2][index-len(self.numpy_windows[0])-len(self.numpy_windows[1])]
                current_numpy_one_hot_mask = self.numpy_one_hot_masks[2][index-len(self.numpy_one_hot_masks[0])-len(self.numpy_one_hot_masks[1])]
            else:
                #print(f"array: 3 | index: {index-len(self.numpy_one_hot_masks[0])-len(self.numpy_one_hot_masks[1])-len(self.numpy_one_hot_masks[2])}")
                current_numpy_window = self.numpy_windows[3][index-len(self.numpy_windows[0])-len(self.numpy_windows[1])-len(self.numpy_windows[2])]
                current_numpy_one_hot_mask = self.numpy_one_hot_masks[3][index-len(self.numpy_one_hot_masks[0])-len(self.numpy_one_hot_masks[1])-len(self.numpy_one_hot_masks[2])]
        else:
            current_numpy_window = self.numpy_windows[index]
            current_numpy_one_hot_mask = self.numpy_one_hot_masks[index]

        tensor_current_window, tensor_one_hot_mask = self._transform(current_numpy_window, current_numpy_one_hot_mask)
        
        return tensor_current_window, tensor_one_hot_mask

    

    def __len__(self):
        return self.length



if __name__=="__main__":
    #ds = Andersson_dataset(mode="test", window_size=40, log_reg=True)
    
    
    
    




