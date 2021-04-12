import torch
from torch.utils.data import Dataset, DataLoader


import glob
import json
import numpy as np
from natsort import natsorted
from tqdm import tqdm



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



class SupervisedDataset(Dataset):
    def __init__(self, mode, window_size=3, log_reg=False):
        self.mode = mode
        self.log_reg = log_reg
     
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

        # Creating an array of sliding numpy_windows with shape of [num_frames, self.window_size, self.keypoints_num*2]
        # The window shifting step is self.keypoints_num*2 (1 frame)
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



class SiameseDataset(Dataset):
    def __init__(self, mode, window_size, num_test_person, use_tree_structure=True) -> None:
        self.mode = mode
        self.num_test_person = num_test_person
        self.device = torch.device('cuda:0')

        self.window_size = window_size
        self.skip_window_size = self.window_size // 4

        self.tree_structure = [ 10,1,0,1,3,5,7,9,7,5,3,1,
                                2,4,6,8,6,4,2,1,10,11,13,
                                15,17,19,17,15,13,11,12,14,
                                16,18,16,14,12,11,10 ]

        if use_tree_structure:
            self.keypoints_num = len(self.tree_structure)       # 39
        else:
            self.keypoints_num= 20
        

        if self.mode == 'train':
            self.json_path = train_json_path
        elif self.mode == 'val':
            self.json_path = val_json_path
        elif self.mode == 'test':
            self.json_path = test_json_path

        self.base_dataset = self._create_base_dataset()
        self.min_windows_num = self.get_min_windows_num(need_print=False)        
        
        self.pair_dataset = self._create_pair_dataset()
    


    def __len__(self):
        return self.pair_dataset.shape[0]


    
    def __getitem__(self, index):
        person1_id, person1_window_id, person2_id, person2_window_id, label = self.pair_dataset[index]
        
        person1_window = self.base_dataset[person1_id][person1_window_id]
        person2_window = self.base_dataset[person2_id][person2_window_id]
        
        tensor_person1_window, tensor_person2_window, tensor_label = self._transform(person1_window=person1_window, person2_window=person2_window, label=label)
        
        return tensor_person1_window, tensor_person2_window, tensor_label
    


    def _transform(self, person1_window, person2_window, label):
                      
        tensor_person1_window = torch.as_tensor(person1_window, dtype=torch.float32).to(device=self.device)
        tensor_person2_window = torch.as_tensor(person2_window, dtype=torch.float32).to(device=self.device)

        tensor_label = torch.as_tensor(label, dtype=torch.float32).to(device=self.device)
        #tensor_one_hot_mask = torch.as_tensor(current_numpy_one_hot_mask, dtype=torch.long).to(device=self.device)               

        return tensor_person1_window, tensor_person2_window, tensor_label



    def _create_base_dataset(self):
        base_dataset = np.array([None]*170)
      
        for i, json_file in enumerate(tqdm(self.json_path, desc="Creating windows for each person", unit=" person")):
            with open(json_file) as f:
                current_dictionary = json.load(f)
            
            current_annotations = current_dictionary["annotations"]
            current_track_id = int(current_annotations[0]["track_id"])
            
            current_numpy_windows = self._create_sliding_windows(current_annotations)

            if base_dataset[current_track_id] is None:
                base_dataset[current_track_id] = current_numpy_windows
            else:
                base_dataset[current_track_id] = np.concatenate((base_dataset[current_track_id], current_numpy_windows), axis=0)
        
        base_dataset = self._clean_base_dataset(base_dataset)

        return base_dataset



    def _calculate_window_idxs(self, full_size, window_size, step_size):
        range_list = []
        for i in range(0, full_size, step_size):
            if (i+window_size) <= full_size:
                range_list.append([i, i+window_size])
        
        return range_list



    def _create_sliding_windows(self, annotations):
        # Sorting the list of dictionaries by x["id"] (ascending order)
        annotations.sort(key=lambda x: x["id"])
        
        # Min-Max normalizing 
        xs = np.array([np.array(annotation["keypoints"][0::3])[self.tree_structure] for annotation in annotations]).flatten() / 1920
        ys = np.array([np.array(annotation["keypoints"][1::3])[self.tree_structure] for annotation in annotations]).flatten() / 1080
        
        # Concatenating xs and ys in the following order: xs[0], ys[0], xs[1], ys[1], ...
        xys = np.ravel([xs,ys],'F')
        
        # Calculating how many self.window_size sized windows (with self.skip_window_size shifts) the video contains 
        #num_windows = xys.shape[0] // (self.window_size*self.keypoints_num*2)
        windows = self._calculate_window_idxs(full_size=xys.shape[0], window_size=self.window_size*self.keypoints_num*2, step_size=self.skip_window_size*self.keypoints_num*2)
        
        # Creating an array of sliding numpy_windows with shape of [num_frames, self.window_size, self.keypoints_num*2]
        # The window shifting step is self.keypoints_num*2 (1 frame)
        # Using self.keypoints*2 due to the fact that a keypoint has 2 (x,y) coordinates
        #numpy_windows = np.array([xys[i:(i+self.window_size*self.keypoints_num*2)] for i in range(0, (len(xys)-self.window_size*self.keypoints_num*2+1), (self.keypoints_num*2))])
        
        # Creating an array of sliding numpy_windows with shape of [num_windows, self.window_size, self.keypoints_num*2]
        # The window shifting step is self.window_size*self.keypoints_num*2 (self.window_size frame)
        # Using self.keypoints*2 due to the fact that a keypoint has 2 (x,y) coordinates
        #numpy_windows = np.array([xys[i:(i+self.window_size*self.keypoints_num*2)] for i in range(0, (num_windows*self.window_size*self.keypoints_num*2), (self.window_size*self.keypoints_num*2))])
        numpy_windows = np.array([xys[idxs[0]:idxs[1]] for idxs in windows])
        
        numpy_windows = numpy_windows.reshape([-1, self.window_size, self.keypoints_num*2])
        
        return numpy_windows



    def _clean_base_dataset(self, base_dataset):
        # Cleaning up Nonetype members of the base dataset
        not_none_indices = [windows is not None for windows in base_dataset]
        base_dataset = base_dataset[not_none_indices]

        # Excluding num_test_person person from the training dataset
        base_dataset = base_dataset[0:-self.num_test_person]

        return base_dataset
    


    def get_min_windows_num(self, need_print=True):
        min = 100000
        min_id = 170
        for id, windows in enumerate(self.base_dataset):
            if windows is None:
                if need_print:
                    print(f"{id}: None")
            else:
                if need_print:
                    print(f"{id}: {windows.shape}")
                if windows.shape[0] < min:
                    min = windows.shape[0]
                    min_id = id
        
        if need_print:
            print(f"Min # of windows: {min_id}: {min}")

        return min



    def n_over_k(self, n, k=2):
        def factorial(n):
            return 1 if (n==1 or n==0) else n * factorial(n - 1)

        return factorial(n)/(factorial(k)*factorial(n-k))

    

    def _create_pair_dataset(self):
        np.random.seed(42)

        # Initializing random generator for random permutation
        rng = np.random.default_rng(seed=42)

        # Random shuffle (permute) the order of the windows for each person
        for person_windows in self.base_dataset:
            rng.shuffle(person_windows, axis=0)
        
        # Creating [(self.min_windows_num!)/((2!)*((self.min_windows_num-2)!)) + self.min_windows_num] 
        #   number of positive pairs for each person
        # The "+ self.min_windows_num" are pairs that contain the same windows
        # With (self.min_windows_num = 32), the shape of the positive pair array for each person: (527, 2, 40, 78)
        #self.min_windows_num = 32
        num_pairs_per_person = int(self.n_over_k(n=self.min_windows_num) + self.min_windows_num)

        pos_pairs = np.zeros(shape=(len(self.base_dataset), num_pairs_per_person, 5), dtype=np.int)#40, 78))
        neg_pairs = np.zeros(shape=(len(self.base_dataset), num_pairs_per_person, 5), dtype=np.int)
        for curr_person_idx in tqdm(range(len(self.base_dataset)), unit=" person", desc="Creating pairs for each person"):
            counter = 0

            for curr_window_idx in range((self.min_windows_num)):
                for other_window_idx in range(curr_window_idx, self.min_windows_num):
                    pos_pairs[curr_person_idx, counter, :] = [curr_person_idx, curr_window_idx, curr_person_idx, other_window_idx, 1]
                
                    
                    other_person_idx = (curr_person_idx + np.random.choice(np.arange(1, len(self.base_dataset)//2), 1)) % len(self.base_dataset)
                    neg_pairs[curr_person_idx, counter, :] = [curr_person_idx, curr_window_idx, other_person_idx, other_window_idx, 0]

                    counter += 1        

        pair_dataset = np.stack((pos_pairs, neg_pairs), axis=0).reshape(-1, 5)
        print(f"Created pair dataset containing {pair_dataset.shape[0]//2} positive and {pair_dataset.shape[0]//2} negative pairs by concatenating person pairs ({num_pairs_per_person} pairs/person, {self.min_windows_num} windows/person).")


        return pair_dataset





if __name__=="__main__":
    

    #ds = Andersson_dataset(mode="test", window_size=40, log_reg=True)
    
    ds = SiameseDataset(mode="train", window_size=40, num_test_person=20)
    

    val_ds = SiameseDataset(mode="val", window_size=40, num_test_person=20)
    
    print(len(ds))
    print(ds.base_dataset.shape)

    print(len(val_ds))
    print(val_ds.base_dataset.shape)

    x1, x2, y = val_ds[50]
    print(x1.shape, x1.dtype, x2.shape, x2.dtype, y, y.shape)
    
   



   




