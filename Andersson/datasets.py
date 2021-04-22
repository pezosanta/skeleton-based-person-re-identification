import torch
from torch.utils.data import Dataset, DataLoader

import glob
import json
import yaml
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from enum import Enum



class DatasetType(Enum):
    TRAINING = 'training'
    VALIDATION = 'validation'
    TEST = 'test'



class TrainingMode(Enum):
    SUPERVISED = 0
    SUPERVISED_LOGREG = 1
    SIAMESE = 2



class BaseDataset():
    def __init__(self, dataset_type: DatasetType, num_exclude: int, window_size: int, shift_size: int, mode: TrainingMode = TrainingMode.SUPERVISED, use_tree_structure: bool = True):
        self.dataset_type = dataset_type
        self.mode = mode

        self.window_size = window_size
        self.shift_size = shift_size
        self.num_exclude = num_exclude

        self.tree_structure = [ 10,1,0,1,3,5,7,9,7,5,3,1,
                                2,4,6,8,6,4,2,1,10,11,13,
                                15,17,19,17,15,13,11,12,14,
                                16,18,16,14,12,11,10 ]
        self.keypoints_num = len(self.tree_structure) if use_tree_structure else 20
        self.features_num = 2*self.keypoints_num if self.mode != TrainingMode.SUPERVISED_LOGREG else 2

        self.json_paths = self._load_database_path()

        self.num_person_id = 170
        self.base_dataset, self.base_labels, self.base_windows_num = self._create_base_dataset()



    def _load_database_path(self):
        paths = yaml.safe_load(open('datasets.yml').read())

        database_path = paths['base_path'] + paths[self.dataset_type.value]
        json_paths = natsorted(glob.glob(database_path + "*.json"))

        return json_paths
    


    def _create_base_dataset(self):
        base_dataset = np.array([None]*self.num_person_id)
        base_labels = np.array([None]*self.num_person_id)
        base_windows_num = np.zeros(self.num_person_id, dtype=int)
      
        for i, json_file in enumerate(tqdm(self.json_paths, desc="Creating windows for each person", unit=" person")):
            with open(json_file) as f:
                current_dictionary = json.load(f)
            
            current_annotations = current_dictionary["annotations"]
            current_track_id = int(current_annotations[0]["track_id"])
            
            current_numpy_windows, current_numpy_labels = self._create_sliding_windows(annotations=current_annotations, track_id=current_track_id)

            if base_dataset[current_track_id] is None:
                base_dataset[current_track_id] = current_numpy_windows
                base_labels[current_track_id] = current_numpy_labels
            else:
                base_dataset[current_track_id] = np.concatenate((base_dataset[current_track_id], current_numpy_windows), axis=0)
                base_labels[current_track_id] = np.concatenate((base_labels[current_track_id], current_numpy_labels), axis=0)
                
            base_windows_num[current_track_id] += current_numpy_windows.shape[0]
        
        base_dataset, base_labels, base_windows_num = self._clean_base_dataset(base_dataset=base_dataset, base_labels=base_labels, base_windows_num=base_windows_num)

        if self.mode != TrainingMode.SIAMESE:
            base_dataset, base_labels = self._concatenate_windows(base_dataset=base_dataset, base_labels=base_labels, base_windows_num=base_windows_num)

        return base_dataset, base_labels, base_windows_num

    

    def _calculate_window_idxs(self, full_size, window_size, step_size):
        range_list = []
        for i in range(0, full_size, step_size):
            if (i+window_size) <= full_size:
                range_list.append([i, i+window_size])
        
        return range_list



    def _create_sliding_windows(self, annotations, track_id):
        # Sorting the list of dictionaries by x["id"] (ascending order)
        annotations.sort(key=lambda x: x["id"])
        
        # Min-Max normalizing 
        xs = np.array([np.array(annotation["keypoints"][0::3])[self.tree_structure] for annotation in annotations]).flatten() / 1920
        ys = np.array([np.array(annotation["keypoints"][1::3])[self.tree_structure] for annotation in annotations]).flatten() / 1080
        
        # Concatenating xs and ys in the following order: xs[0], ys[0], xs[1], ys[1], ...
        xys = np.ravel([xs,ys],'F')
        
        # Calculating how many self.window_size sized windows (with self.skip_window_size shifts) the video contains 
        #num_windows = xys.shape[0] // (self.window_size*self.keypoints_num*2)
        windows = self._calculate_window_idxs(full_size=xys.shape[0], window_size=self.window_size*self.keypoints_num*2, step_size=self.shift_size*self.keypoints_num*2)
        
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

        if self.mode == TrainingMode.SUPERVISED_LOGREG:
            numpy_windows_mean = np.mean(numpy_windows, axis=1)
            numpy_windows_var = np.var(numpy_windows, axis=1)
        
            numpy_windows = np.stack((numpy_windows_mean, numpy_windows_var), axis=2)
        
        numpy_labels = np.zeros((numpy_windows.shape[0]), dtype=int)
        numpy_labels[:] = track_id
        
        return numpy_windows, numpy_labels


    
    def _clean_base_dataset(self, base_dataset, base_labels, base_windows_num):
        # Cleaning up Nonetype members of the base dataset
        not_none_indices = [windows is not None for windows in base_dataset]

        base_dataset = base_dataset[not_none_indices]
        base_labels = base_labels[not_none_indices]
        base_windows_num = base_windows_num[not_none_indices]

        # Making person ids continous from 0 to num_persons
        for i, labels in enumerate(base_labels):
            labels[:] = i

        # Excluding num_test_person person from the training dataset
        if self.num_exclude != 0:
            base_dataset = base_dataset[0:-self.num_exclude]
            base_labels = base_labels[0:-self.num_exclude]
            base_windows_num = base_windows_num[0:-self.num_exclude]

        return base_dataset, base_labels, base_windows_num

    

    def _concatenate_windows(self, base_dataset, base_labels, base_windows_num):
        seq_length = 2*self.keypoints_num if self.mode == TrainingMode.SUPERVISED_LOGREG else self.window_size

        concat_dataset = np.zeros((int(base_windows_num.sum()), seq_length, self.features_num))
        concat_labels = np.zeros(base_windows_num.sum(), dtype=int)

        curr_idx = 0
        for i in tqdm(range(base_dataset.shape[0]), desc="Concatenating windows, labels, #windows of each person", unit=" person"):
            concat_dataset[curr_idx:curr_idx+base_dataset[i].shape[0], :, :] = base_dataset[i]
            concat_labels[curr_idx:curr_idx+base_labels[i].shape[0]] = base_labels[i]

            curr_idx += base_dataset[i].shape[0]

        return concat_dataset, concat_labels



    def get(self):
        return self.base_dataset, self.base_labels, self.base_windows_num






class SupervisedDataset(Dataset):
    def __init__(self, dataset_type: DatasetType, num_exclude: int, window_size: int, shift_size: int, mode: TrainingMode = TrainingMode.SUPERVISED, use_tree_structure: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dataset, self.labels, self.windows_num = BaseDataset(dataset_type=dataset_type, num_exclude=num_exclude, window_size=window_size, shift_size=shift_size, mode=mode, use_tree_structure=use_tree_structure).get()

    
    
    def __getitem__(self, index):
        current_numpy_window = self.dataset[index]
        current_numpy_label = self.labels[index]

        current_tensor_window, current_tensor_label = self._transform(current_numpy_window=current_numpy_window, current_numpy_label=current_numpy_label)

        return current_tensor_window, current_tensor_label



    def __len__(self):
        return self.dataset.shape[0]



    def _transform(self, current_numpy_window, current_numpy_label):
        current_tensor_window = torch.as_tensor(current_numpy_window, dtype=torch.float32).to(device=self.device)
        current_tensor_label = torch.as_tensor(current_numpy_label, dtype=torch.long).to(device=self.device)              

        return current_tensor_window, current_tensor_label

    

    def get_class_weights(self):
        tensor_weights = torch.reciprocal(torch.tensor(self.windows_num, dtype=torch.float32))

        return tensor_weights






class SiameseDataset(Dataset):
    def __init__(self, dataset_type: DatasetType, num_exclude: int, window_size: int, shift_size: int, mode: TrainingMode = TrainingMode.SIAMESE, need_pair_dataset: bool = True, use_tree_structure: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.base_dataset, _, _ = BaseDataset(dataset_type=dataset_type, num_exclude=num_exclude, window_size=window_size, shift_size=shift_size, mode=mode, use_tree_structure=use_tree_structure).get()

        self.min_windows_num = self.get_min_windows_num(need_print=False)        
        
        self.need_pair_dataset = need_pair_dataset

        if self.need_pair_dataset:
            self.pair_dataset = self._create_pair_dataset()
    


    def __len__(self):
        if self.need_pair_dataset:
            return self.pair_dataset.shape[0]
        else:
            return len(self.base_dataset)


    
    def __getitem__(self, index):
        if self.need_pair_dataset:
            person1_id, person1_window_id, person2_id, person2_window_id, label = self.pair_dataset[index]
            
            person1_window = self.base_dataset[person1_id][person1_window_id]
            person2_window = self.base_dataset[person2_id][person2_window_id]
            
            tensor_person1_window, tensor_person2_window, tensor_label = self._transform(person1_window=person1_window, person2_window=person2_window, label=label)
            
            return tensor_person1_window, tensor_person2_window, tensor_label
        else:
            return None
    


    def _transform(self, person1_window, person2_window, label):
                      
        tensor_person1_window = torch.as_tensor(person1_window, dtype=torch.float32).to(device=self.device)
        tensor_person2_window = torch.as_tensor(person2_window, dtype=torch.float32).to(device=self.device)

        tensor_label = torch.as_tensor(label, dtype=torch.float32).to(device=self.device)      

        return tensor_person1_window, tensor_person2_window, tensor_label



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
    
    
    #ds = SupervisedDataset(dataset_type=DatasetType.TRAINING, num_exclude=0, window_size=40, shift_size=1, mode=TrainingMode.SUPERVISED_LOGREG)
    ds = SiameseDataset(dataset_type=DatasetType.VALIDATION, num_exclude=0, window_size=40, shift_size=10, mode=TrainingMode.SIAMESE, need_pair_dataset=False)

    loader = DataLoader(dataset=ds, batch_size=10, shuffle=True)

    it = iter(loader)
    window1, window2, label = next(it)

    print(window1.shape, window2.shape, label)

    #print(ds.windows_num.shape)
    #print(ds.windows_num[60:100])
    #print(ds.get_class_weights()[60:100])
    #print(ds.get_class_weights()[60])


    '''
    for i, labels in enumerate(ds.base_labels):
        labels[:] = i

    for i in range(len(ds.base_dataset)):
        print(ds.base_labels[i][0])
    '''

    '''
    ds = SiameseDataset(dataset='train', need_pair_dataset=False, window_size=40, num_test_person=20)
    val_ds = SiameseDataset(dataset="val", need_pair_dataset=False, window_size=40, num_test_person=20)
    test_ds = SiameseDataset(dataset='test', need_pair_dataset=False, window_size=40)
    
    print(len(ds))
    print(ds.base_dataset.shape, ds.base_dataset[0].shape, ds.base_dataset[10].shape)

    print(len(val_ds))
    print(val_ds.base_dataset.shape, val_ds.base_dataset[0].shape, val_ds.base_dataset[10].shape)

    print(len(test_ds))
    print(test_ds.base_dataset.shape, test_ds.base_dataset[0].shape, test_ds.base_dataset[10].shape)
    '''
   



   




