# Lint as: python3
""""Contains classes and methods to load the datasets"""
import numpy as np
import torch
import torchaudio
import os

import audio_preprocess_lib

"""
Load dataset paths, all in the same list
"""
# General params
base_path = "/nas/public/exchange/bdsound_bss"
audio_folder = "simulations"
beamspace_folder = "filter_beamspace"
datasets_names = ['dsNoFit_s01_i04_4mic_u_26mm', 'dsNoFit_s01_i04_3mic_u_20mm',
                  'dsNoFit_s01_i04_4mic_u_30mm', 'dsNoFit_s01_i04_3mic_u_30mm', 'dsNoFit_s01_i04_4mic_u_20mm']

#datasets_names = ['dsNoFit_s01_i04_4mic_u_26mm']
val_split = 0.2 # Percentage of total dataset used for validation
ref_mic_idx = 2
n_frames = 50

# Store beamspaces in torch PARAMETERDICT dictionary, in order to load it when training the network
beamspace_torch_dict = torch.nn.ParameterDict({
    'dsNoFit_s01_i04_4mic_u_26mm': torch.from_numpy(np.load(os.path.join(base_path, beamspace_folder, '4mic_u_26mm.npy'))),
    'dsNoFit_s01_i04_3mic_u_20mm': torch.from_numpy(np.load(os.path.join(base_path, beamspace_folder, '3mic_u_20mm.npy'))),
    'dsNoFit_s01_i04_4mic_u_30mm': torch.from_numpy(np.load(os.path.join(base_path, beamspace_folder, '4mic_u_30mm.npy'))),
    'dsNoFit_s01_i04_3mic_u_30mm': torch.from_numpy(np.load(os.path.join(base_path, beamspace_folder, '3mic_u_30mm.npy'))),
    'dsNoFit_s01_i04_4mic_u_20mm': torch.from_numpy(np.load(os.path.join(base_path, beamspace_folder, '4mic_u_20mm.npy'))),
    'etsi_s01_i04_text_backup': torch.from_numpy(np.load(os.path.join(base_path, beamspace_folder, '4mic_u_26mm.npy'))),
    'etsi_s01_i04_text_unseen3': torch.from_numpy(np.load(os.path.join(base_path, beamspace_folder, '3mic_u_52mm.npy'))),
    'etsi_s01_i04_text_unseen4': torch.from_numpy(np.load(os.path.join(base_path, beamspace_folder, '4mic_u_52mm.npy'))),
})

# Data elements
mix_list = []  # Contains paths to mixture audio files
target_list = []  # Contains paths to target audio files
beamspace_list = []  # Contains paths to beamspace files

for i in range(len(datasets_names)):

    # Load target and mix files in current dataset
    curr_target = os.listdir(os.path.join(base_path, audio_folder, datasets_names[i], 'audio/target'))
    curr_mix = os.listdir(os.path.join(base_path, audio_folder, datasets_names[i], 'audio/mix'))

    # Repeat beamspace (easier)
    curr_beamspace = [datasets_names[i] for s in curr_mix]
    # Sort lists --> to have 1to1 correspondance
    curr_target.sort()
    curr_mix.sort()

    # Append full path, otherwise no way to distinguish audio files
    curr_target = [os.path.join(base_path, audio_folder, datasets_names[i], 'audio/target') + '/' + s for s in curr_target]
    curr_mix = [os.path.join(base_path, audio_folder, datasets_names[i], 'audio/mix') + '/' + s for s in curr_mix]

    # Concatenate lists
    target_list = target_list + curr_target
    mix_list = mix_list + curr_mix
    beamspace_list = beamspace_list + curr_beamspace

"""
Create pytorch dataset
"""
#device = "cuda" if torch.cuda.is_available() else "cpu"

# Separate train and val datasets
val_length = int(0.2*len(target_list))
train_length = len(target_list) - val_length
np.random.seed(seed=42)
idx_val = np.random.choice(np.arange(len(target_list)), size=val_length, replace=False)

# Val data
target_list_val = np.array(target_list)[idx_val].tolist()
mix_list_val = np.array(mix_list)[idx_val].tolist()
beamspace_list_val = np.array(beamspace_list)[idx_val].tolist()

# train data
target_list_train = np.delete(np.array(target_list), idx_val).tolist()
mix_list_train= np.delete(np.array(mix_list), idx_val).tolist()
beamspace_list_train = np.delete(np.array(beamspace_list), idx_val).tolist()


# Define Dataset class
class BeamspaceDataset(torch.utils.data.Dataset):
    def __init__(self, mix_paths, target_paths, beamspace_paths):
        self.mix_paths = mix_paths
        self.target_paths = target_paths
        self.beamspace_paths = beamspace_paths

    def __len__(self):
        return len(self.mix_paths)

    def __getitem__(self, idx):
        with torch.no_grad():
            mix_audio, sr = torchaudio.load(self.mix_paths[idx])
            target_audio, sr = torchaudio.load(self.target_paths[idx])
            beamspace_matrix = beamspace_torch_dict[self.beamspace_paths[idx]]

            #mix_audio, target_audio, beamspace_matrix = mix_audio.to(device), target_audio.to(device), beamspace_matrix.to(device)
            # Compute STFT and beamspace
            mix_stft = audio_preprocess_lib.compute_multichannel_stft_torch(mix_audio)
            trgt_stft = audio_preprocess_lib.compute_multichannel_stft_torch(target_audio)
            beamspace = audio_preprocess_lib.compute_beamspace(mix_stft, beamspace_matrix)

            # Split complex and real part
            mix_stft_ref = torch.cat(
                (torch.real(mix_stft[:, :, ref_mic_idx]).unsqueeze(-1),
                 torch.imag(mix_stft[:, :, ref_mic_idx]).unsqueeze(-1))
                , dim=2)
            trgt_stft_ref = torch.cat(
                (torch.real(trgt_stft[:, :, ref_mic_idx]).unsqueeze(-1),
                 torch.imag(trgt_stft[:, :, ref_mic_idx]).unsqueeze(-1)),
                dim=2)
            beamspace_input = torch.cat((
                torch.real(beamspace).unsqueeze(-1),
                torch.imag(beamspace).unsqueeze(-1),
            ), dim=3)

            # Slice frame
            idx_slice = torch.randint(low=n_frames // 2, high=mix_stft_ref.shape[0] - n_frames // 2 + 1, size=(1, 1))
            mix_stft_ref = mix_stft_ref[idx_slice-n_frames//2:idx_slice+n_frames//2]
            beamspace_input = beamspace_input[idx_slice-n_frames//2:idx_slice+n_frames//2]
            trgt_stft_ref = trgt_stft_ref[idx_slice-n_frames//2:idx_slice+n_frames//2]
            return mix_stft_ref, trgt_stft_ref, beamspace_input


train_dataset, val_dataset = BeamspaceDataset(mix_list_train, target_list_train, beamspace_list_train), BeamspaceDataset(mix_list_val, target_list_val, beamspace_list_val)
print(f'Splitting into Train ({train_dataset.__len__()} tracks) and Validation ({val_dataset.__len__()} tracks) sets.')


for a, b, c in train_dataset:
    break