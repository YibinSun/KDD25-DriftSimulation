from CTGANs import train_ctgan

origins = {
    # 'alb_avg_4hr' : './original_data/NZEP_datasets/ALB0331/ALB0331_avg_4hr.csv',
    # 'wil_avg_24hr' : './original_data/NZEP_datasets/WIL0331/WIL0331_avg_24hr.csv',
    # 'ham_avg_6hr' : './original_data/NZEP_datasets/HAM0331/HAM0331_avg_6hr.csv',
    # 'sdn_avg_30min' : './original_data/NZEP_datasets/SDN0331/SDN0331_avg_0.5hr.csv',
    #
    # 'abalone' : './original_data/abalone.arff',
    # 'bike' : './original_data/bike.arff',
    # 'diamonds' : './original_data/diamonds.arff',
    'House8L' : './original_data/House8L.arff',
    'superconductivity' : './original_data/superconductivity.arff',

}

for key in origins:
    model = train_ctgan(origins[key], epochs=300, batch_size=500, generator_lr=1e-3, discriminator_lr=1e-3, save_path=f'./trained_models/ctgan_{key}.pkl')

# model1 = train_ctgan(data_path1, epochs=300, batch_size=500, generator_lr=1e-3, discriminator_lr=1e-3, save_path=f'./trained_models/ctgan_{data_path1.split("/")[-1].split(".")[0]}.pkl')
# model2 = train_ctgan(data_path2, epochs=300, batch_size=500, generator_lr=1e-3, discriminator_lr=1e-3, save_path=f'./trained_models/ctgan_{data_path2.split("/")[-1].split(".")[0]}.pkl')