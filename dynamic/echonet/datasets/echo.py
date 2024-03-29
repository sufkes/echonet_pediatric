"""EchoNet-Dynamic Dataset."""

import pathlib
import os
import collections

import numpy as np
import skimage.draw
import torch.utils.data
import echonet

class Echo(torch.utils.data.Dataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {"train", "val", "test", "external_test"}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None,

                 #### Added by Steven Ufkes:
                 file_list_path=None, # path to FileList.csv
                 load_tracings=True, # whether to load VolumeTracings.csv
                 volume_tracings_path=None, # path to VolumeTracings.csv
                 file_path_col='FilePath', # Column in FileList.csv to read AVI file paths from.
                 subject_name_col='Subject', # Column in FileList.csv to read subject IDs from.
                 split_col='Split', # Column in FileList.csv to assign splits from.
                 rotate180=False # Whether to rotate the image 180 degrees; try because it looks like our images are rotated relative to the EchoNet Dynamic dataset.
                 ):
                 
        if root is None:
            root = echonet.config.DATA_DIR
        self.folder = pathlib.Path(root)

        self.split = split
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.rotate180 = rotate180

        #### Things added by Steven Ufkes
        if file_list_path is None: # Use hard-coded value as before if no argument provided.
            self.file_list_path = os.path.join(self.folder, 'FileList.csv')
        else:
            self.file_list_path = file_list_path

        if volume_tracings_path is None:
            self.volume_tracings_path = os.path.join(self.folder, 'VolumeTracings.csv')
        else:
            self.volume_tracings_path = volume_tracings_path
        ####

        #self.fnames, self.outcome = [], []
        self.fnames, self.fpaths, self.outcome = [], [], [] # Steven Ufkes: Store subject names in fnames, as was essentially done before (and is expected by other scripts). Now also save file paths in the new fpaths attribute.

        if split == "external_test":
            # Steven Ufkes: looks like "external tests" were set up in a different way than the rest.
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
#            with open(self.folder / "FileList.csv") as f:
            with open(self.file_list_path, 'r') as f: # Steven Ufkes: Replaced hardcoded file list name in previous line.
                # Steven Ufkes: Probaly best to just convert this to Pandas but leave as is for now.
                self.header = f.readline().strip().split(",")
                #filenameIndex = self.header.index("FileName")
                #splitIndex = self.header.index("Split")
                subjectNameIndex = self.header.index(subject_name_col)
                filePathIndex = self.header.index(file_path_col) # Steven Ufkes
                splitIndex = self.header.index(split_col) # Steven Ufkes

                for line in f:
                    lineSplit = line.strip().split(',')

                    #### Steven Ufkes: Replace section below.
                    subjectName = lineSplit[subjectNameIndex]
                    filePath = lineSplit[filePathIndex]
                    fileMode = lineSplit[splitIndex].lower()
                    if (split in ['all', fileMode]) and os.path.exists(filePath):
                        self.fnames.append(subjectName)
                        self.fpaths.append(filePath)
                        self.outcome.append(lineSplit) # store list containing the value of each column for this row; used in conjunction with self.header to retrieve values.
                    ####

                    #fileName = lineSplit[filenameIndex] # Steven Ufkes: Original line that gets the file names without the AVI extension (in their dataset).
                    #fileMode = lineSplit[splitIndex].lower()
                    #if split in ["all", fileMode] and os.path.exists(self.folder / "Videos" / fileName): # Steven Ufkes: self.folder has type <class 'pathlib.PosixPath'> which apparently supports division by strings.
                    # Steven Ufkes: Line above doesn't seem to work properly even on the EchoNet dataset. Maybe they were using Windows.
                        #self.fnames.append(fileName)
                        #self.outcome.append(lineSplit)

            # Steven I think frames and trace are related to VolumeTracings, but leave these here for now.
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            #print('')
            #print('self.fnames before removing ones with (less than 2 frames?)')
            #print(self.fnames)
            #if load_VolumeTracings: # Steven Ufkes: Add an option to skip loading the volume tracings if only running video.py.
            if load_tracings: # Steven Ufkes: Only load tracings if option is True.
                print("Warning: Major changes were made to the clipwise EF prediction, but these changes were not made to the segmentation section.")
                #with open(self.folder / "VolumeTracings.csv") as f:
                with open(self.volume_tracings_path, 'r') as f: # Steven Ufkes: I didn't modify the rest of this section in the same way as the previous section, since we aren't using it now.
                    header = f.readline().strip().split(",")
                    assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                    for line in f:
                        filename, x1, y1, x2, y2, frame = line.strip().split(',')
                        x1 =  float(x1)
                        y1 = float(y1)
                        x2 = float(x2)
                        y2 = float(y2)
                        frame = int(frame)
                        if frame not in self.trace[filename]:
                            self.frames[filename].append(frame)
                        self.trace[filename][frame].append((x1, y1, x2, y2))
                for filename in self.frames:
                    for frame in self.frames[filename]:
                        self.trace[filename][frame] = np.array(self.trace[filename][frame])
                #print('')
                #print('self.frames:')
                #print (self.frames)
                #keep = [len(self.frames[os.path.splitext(f)[0]]) >= 2 for f in self.fnames] # Steven Ufkes: Doesn't work for their dataset due to file extension inconsistency.
                keep = [len(self.frames[f]) >= 2 for f in self.fnames]
                self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
                #print('')
                #print('self.fnames after removing ones with (less than 2 frames?)')
                #print(self.fnames)
                self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "external_test":
            video_path = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "clinical_test":
            video_path = os.path.join(self.folder, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            #video = os.path.join(self.folder, "Videos", self.fnames[index])
            video_path = os.path.join(self.fpaths[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video_path).astype(np.float32)

        # Rotate video 180 degrees.
        if self.rotate180:
            video = np.rot90(video, k=2, axes=(len(video.shape)-2, len(video.shape)-1)) # rotate by 180 degrees in plane of last two dimensions (height and width).

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = os.path.splitext(self.fnames[index])[0]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "clinical_test" or self.split == "external_test":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select random clips
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        ## Steve: Padding section below was not set up to deal with clips>1; try to do that. How about let's pad each clip differently too? Why not.
        if self.pad is not None:
            if self.clips == 1:
                ####### ORIGINAL
                # Add padding of zeros (mean color of videos)
                # Crop of original size is taken out
                # (Used as augmentation)
                c, l, h, w = video.shape
                temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
                temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
                i, j = np.random.randint(0, 2 * self.pad, 2)
                video = temp[:, :, i:(i + h), j:(j + w)]
                ####### END OF ORIGINAL
            else:
                num_clips, c, l, h, w = video.shape
                for clip_index in range(num_clips):
                    video_clip = video[clip_index, :, :, :, :]
                    temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
                    temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video_clip  # pylint: disable=E1130
                    i, j = np.random.randint(0, 2 * self.pad, 2)
                    video[clip_index, :, :, :, :] = temp[:, :, i:(i + h), j:(j + w)]
        return video, target

    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
