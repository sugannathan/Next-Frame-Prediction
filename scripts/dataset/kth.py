from torch.utils.data import Dataset
import numpy as np
import cv2, torch, os
class KTH(Dataset):
    
    categories = ["boxing", "handclapping", "handwaving","jogging", "running", "walking"]

    def __init__(self, directory, download, train: bool, frames_per_item = 20, transform=None):
        self.__transform = transform
        self.__directory = directory
        self.__frames_per_item = frames_per_item
        self.__download=download
        
        if train:
            self.train = True
            self.persons = ['11', '12', '13', '14', '15', '16', '17', '18','19', '20', '21', '23', '24', '25', '01', '04']
        else:
            self.train = False
            self.persons = ['22', '02', '03', '05', '06', '07', '08', '09', '10']

        self.prepare_data()

    def prepare_data(self):
        self.__data = self.__parse_sequence_file(self.__directory, self.__frames_per_item)

        if(self.__download):
            self.__process_video_frames(self.__data, self.__directory + '/data', self.__transform)
            
        idx = self.get_indices_for_persons(self.persons)
        # These indexes correspond to the indexes of the .pt files that we generated 
        self.data_idx = idx

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):

        idx_to_load = self.data_idx[idx]
        full_path_of_file_to_load = os.path.join(self.__directory, "data", f"{str(idx_to_load)}.pt")

        item_data = torch.load(full_path_of_file_to_load)
        item_data=self.__transform(item_data)
        item_data=torch.swapaxes(item_data, 0, 1)

        return (item_data[:10,:,:,:], item_data[10:,:,:,:])


    def __parse_sequence_file(self, base_dir, num_frames):
        # parse file for example : "person01_boxing_d1		frames	1-95, 96-185, 186-245, 246-360"
        data = []
        with open(base_dir + '/sequences.txt', 'r') as sequence_file:
            for sequence in sequence_file:
                split_1 = sequence.split('frames')   
                if len(split_1) > 1:
                    label_desc = split_1[0].split('_')[1]   
                    label = self.categories.index(label_desc)
                    person_num = split_1[0][6:8]
                    filepath = base_dir + '/' + label_desc + '/' + split_1[0].strip() + '_uncomp.avi' 
    
                    for overall_start, overall_end in [tuple(split_2.split('-')) for split_2 in split_1[1].strip().split(',')]:
                        for i in range(int(overall_start) , int(overall_end) - num_frames, 5):
                            end = i + num_frames
                            data.append({'video_filepath': filepath, 'start':i, 'end':end, 'label':label, 
                                         'label_desc' : label_desc, 'person':person_num})
        return data
    
    def __process_video_frames(self, data, base_dir, transform):
        if not os.path.isdir(base_dir) : os.mkdir(base_dir)
        for i, data_item in enumerate(data):
            print(data_item['video_filepath'])
            vid = cv2.VideoCapture(data_item['video_filepath'])
            frames = []
            for frame_num in range(data_item['start'], data_item['end']):

                vid.set(1, frame_num)
                _, frame = vid.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append([frame])

            vid.release()
            frames = torch.from_numpy(np.moveaxis( np.array(frames), 1,0)) / 255
            torch.save(frames, base_dir + '/' + str(i) + '.pt')


    def get_indices_for_persons(self, person_ids):
        return [ i for i, x in enumerate(self.__data) if x['person'] in person_ids]