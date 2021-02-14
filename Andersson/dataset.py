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

        self.length = self._get_length()

    

    def _create_dataset(self):
        print(len(self.json_path))
        # TODO Change range if RAM has been upgraded
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
                #elif (3*len(self.json_path)/4) <= i < (len(self.json_path)):
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
    print("\n\n")
    print(train_json_path)
    print(val_json_path)

    '''
    ds = Andersson_dataset(mode="train", window_size=40)
    #print(len(ds))

    print(len(ds.numpy_windows))
    
    for windows in ds.numpy_windows:
        print(len(windows))
    
    for masks in ds.numpy_one_hot_masks:
        print(len(masks))

    print(len(ds))

    print("\n\n")
    x=ds[0]
    x=ds[61882]

    print("\n\n")
    x=ds[61883]
    x=ds[61883+64018]

    print("\n\n")
    x=ds[61883+64019]
    x=ds[61883+64019+62151]

    print("\n\n")
    x=ds[61883+64019+62152]
    x=ds[61883+64019+62152+59958]

    print("\n\n")
    x=ds[61883+64019+62152+59959]
    '''




    
    
    




