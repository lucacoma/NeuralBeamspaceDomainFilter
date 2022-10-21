"""
Test network on real measurements and store them in folder
"""
import torchaudio
import torch
import data_lib
import audio_preprocess_lib
import numpy as np
import os
from network_lib import BFM, RRM
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']='0'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_tracks(idx, mix_list, trgt_list, beamspace_matrix, noise_list, model_BFM, model_RRM,
                mix_save_path, noise_save_path, trgt_save_path, est_save_path, device):
    # for idx in tqdm(range(len(mix_list))):
    # def save_audio_tracks(idx):
    # Load audio tracks

    mix_audio, sr = torchaudio.load(mix_list[idx])
    target_audio, sr = torchaudio.load(trgt_list[idx])
    noise_audio, sr = torchaudio.load(noise_list[idx])
    # Compute STFT and beamspace

    mix_stft = audio_preprocess_lib.compute_multichannel_stft_torch(mix_audio)
    trgt_stft = audio_preprocess_lib.compute_multichannel_stft_torch(target_audio)
    beamspace = audio_preprocess_lib.compute_beamspace(mix_stft, beamspace_matrix)

    noise_stft = audio_preprocess_lib.compute_multichannel_stft_torch(noise_audio)
    # Split complex and real part

    mix_stft_ref = torch.cat(
        (torch.real(mix_stft[:, :, data_lib.ref_mic_idx]).unsqueeze(-1),
         torch.imag(mix_stft[:, :, data_lib.ref_mic_idx]).unsqueeze(-1))
        , dim=2)
    trgt_stft_ref = torch.cat(
        (torch.real(trgt_stft[:, :, data_lib.ref_mic_idx]).unsqueeze(-1),
         torch.imag(trgt_stft[:, :, data_lib.ref_mic_idx]).unsqueeze(-1)),
        dim=2)
    beamspace_input = torch.cat((
        torch.real(beamspace).unsqueeze(-1),
        torch.imag(beamspace).unsqueeze(-1),
    ), dim=3)
    noise_stft_ref = torch.cat(
        (torch.real(noise_stft[:, :, data_lib.ref_mic_idx]).unsqueeze(-1),
         torch.imag(noise_stft[:, :, data_lib.ref_mic_idx]).unsqueeze(-1)),
        dim=2)

    # Modify STFT in order to of the right size to be input to network
    # N.B. --> we cut STFT from frame_length//2 to len(STFT) -frame_length//2 +1 since it is then easier to divide in frames
    # N.B. --> we reshape stft in size (N_Frames, frame_length, time, freq)
    frame_length = 50
    N_FRAMES = len(mix_stft_ref) // frame_length
    T = len(mix_stft_ref)
    F = mix_stft_ref.shape[1]

    X_mix = torch.reshape(mix_stft_ref[:N_FRAMES * frame_length], (N_FRAMES, frame_length, F, 2))
    X_trgt = torch.reshape(trgt_stft_ref[:N_FRAMES * frame_length], (N_FRAMES, frame_length, F, 2))
    X_beamspace = torch.reshape(beamspace_input[:N_FRAMES * frame_length], (N_FRAMES, frame_length, F, 5, 2))

    X_noise = torch.reshape(noise_stft_ref[:N_FRAMES * frame_length], (N_FRAMES, frame_length, F, 2))

    X_bfm, X_embed, en_list = model_BFM(X_beamspace.to(device), X_mix.to(device))
    X_rrm = model_RRM(X_embed.to(device), X_mix.to(device), en_list)
    X_bfm_rrm = X_bfm + X_rrm

    X_bfm_rrm = X_bfm_rrm.detach().cpu().numpy()
    X_mix = reshape_track(X_mix)
    X_trgt = reshape_track(X_trgt)
    X_est = reshape_track(X_bfm_rrm)
    X_noise = reshape_track(X_noise)

    # Now let's compute ISTFT

    x_mix = audio_preprocess_lib.compute_multichannel_istft(X_mix)
    x_trgt = audio_preprocess_lib.compute_multichannel_istft(X_trgt)
    x_est = audio_preprocess_lib.compute_multichannel_istft(X_est)
    x_noise = audio_preprocess_lib.compute_multichannel_istft(X_noise)

    # Filename of audio files

    mix_filename = mix_list[idx].split('/')[-1]
    trgt_filename = trgt_list[idx].split('/')[-1]
    est_filename = 'estimated_' + trgt_list[idx].split('/')[-1].split('_')[-1]
    noise_filename = noise_list[idx].split('/')[-1]

    # Save audiofiles

    torchaudio.save(
        filepath=os.path.join(mix_save_path, mix_filename), src=torch.Tensor(x_mix).unsqueeze(0), sample_rate=sr)
    torchaudio.save(
        filepath=os.path.join(trgt_save_path, trgt_filename), src=torch.Tensor(x_trgt).unsqueeze(0), sample_rate=sr)
    torchaudio.save(
        filepath=os.path.join(est_save_path, est_filename), src=torch.Tensor(x_est).unsqueeze(0), sample_rate=sr)
    torchaudio.save(
        filepath=os.path.join(noise_save_path, noise_filename), src=torch.Tensor(x_noise).unsqueeze(0), sample_rate=sr)


def reshape_track(track):
    """
    Reshape track in order to be able to do istft desired
    :param track:
    :return:
    """
    # Reshape
    track = np.reshape(track,(track.shape[0]*track.shape[1],track.shape[2],track.shape[3]))
    # Make it complex
    track = track[:, :, 0] + 1j*track[:, :, 1]
    return track

def main():
    # Paths
    measurements_paths = '/nas/public/exchange/bdsound_bss/measurements/'
    dataset_names = ['etsi_s01_i04_text_backup', 'etsi_s01_i04_text_unseen3', 'etsi_s01_i04_text_unseen4']
    # dataset_names = ['etsi_s01_i04_text_backup']

    data_save_base_path = '/nas/public/exchange/bdsound_bss/baseline' # Path where we will save data
    if not os.path.exists(data_save_base_path): os.makedirs(data_save_base_path)

    # Cycle through datasets
    model_bfm_path = "models/model_BFM_20220928-112846.pth"
    model_rrm_path = "models/model_RRM_20220928-112846.pth"
    # Load Model
    device_type = 'cuda' #'cuda'
    device = torch.device(device_type)
    if device_type =='cpu':
        model_BFM = BFM()
        model_RRM = RRM()
        model_BFM.load_state_dict(torch.load(model_bfm_path, map_location=device))
        model_RRM.load_state_dict(torch.load(model_rrm_path, map_location=device))

    if device_type == "cuda":
        model_BFM = torch.nn.DataParallel(BFM())
        model_RRM = torch.nn.DataParallel(RRM())
        model_BFM.load_state_dict(torch.load(model_bfm_path))
        model_RRM.load_state_dict(torch.load(model_rrm_path))
        model_BFM.to(device)
        model_RRM.to(device)

    print('params BFM')
    print(str(count_parameters(model_BFM)))
    print('params RRM')
    print(str(count_parameters(model_RRM)))

    """
    model_BFM.eval()
    model_RRM.eval()
    for i in range(len(dataset_names)):
        print('Testing and saving tracks from dataset '+dataset_names[i])
        # Retrieve paths
        curr_dataset = dataset_names[i]

        # Handle paths where we'll save stuff at the end
        dataset_save_path = os.path.join(data_save_base_path, curr_dataset)
        mix_save_path = os.path.join(dataset_save_path, 'audio', 'mix')
        trgt_save_path = os.path.join(dataset_save_path, 'audio', 'target')
        est_save_path = os.path.join(dataset_save_path, 'audio', 'estimated')

        noise_save_path = os.path.join(dataset_save_path, 'audio', 'noise')

        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path)
            os.makedirs(mix_save_path)
            os.makedirs(trgt_save_path)
            os.makedirs(est_save_path)
            os.makedirs(noise_save_path)

        # Handle paths from where we'll get the data
        mix_base_path = os.path.join(measurements_paths, dataset_names[i], 'audio', 'mix')
        mix_track_names = os.listdir(mix_base_path)
        mix_list = [os.path.join(mix_base_path, mix_track_names[s]) for s in range(0, len(mix_track_names))]
        mix_list.sort()

        trgt_base_path = os.path.join(measurements_paths, dataset_names[i], 'audio', 'target')
        trgt_track_names = os.listdir(trgt_base_path)
        trgt_list = [os.path.join(trgt_base_path, trgt_track_names[s]) for s in range(0, len(trgt_track_names))]
        trgt_list.sort()

        noise_base_path = os.path.join(measurements_paths, dataset_names[i], 'audio', 'noise')
        noise_track_names = os.listdir(noise_base_path)
        noise_list = [os.path.join(noise_base_path, noise_track_names[s]) for s in range(0, len(noise_track_names))]
        noise_list.sort()

        beamspace_matrix = data_lib.beamspace_torch_dict[curr_dataset]

        N_tracks = len(mix_list)

        for i in tqdm(range(N_tracks)):
            save_tracks(
                i, mix_list, trgt_list, beamspace_matrix,
                noise_list, model_BFM, model_RRM,
                mix_save_path, noise_save_path,
                trgt_save_path, est_save_path, device
            )
    """
if __name__ =="__main__":
    main()