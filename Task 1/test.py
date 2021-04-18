from train import Compressor

def compute(csv_path,model_path):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  df = pd.read_csv(csv_path)
  scaled_df = pd.DataFrame(scaler.transform(df))

  dataset = SAPData(scaled_df)
  SAPdataloader = DataLoader(dataset,batch_size = 512, shuffle = False)

  model = Compressor()
  model.load_state_dict(torch.load(model_path))
  
  with torch.no_grad():
    arr_decoded = torch.empty((0,1,3)).to(device)
    arr_encoded = torch.empty((0,1,3)).to(device)

    for data in SAPdataloader:
      encoded,decoded = model(data)

      encoded = encoded.squeeze(1)
      decoded = decoded.unsqueeze(1)
      arr_decoded = torch.vstack((arr_decoded,decoded))
      arr_encoded = torch.vstack((arr_encoded,encoded))


  arr_decoded = arr_decoded.squeeze(1).cpu().detach().numpy()
  decoded_df = pd.DataFrame(arr_decoded)
  new_decoded_df = scaler.inverse_transform(decoded_df)

  arr_encoded = arr_encoded.squeeze(1).cpu().detach().numpy()
  encoded_df = pd.DataFrame(arr_encoded)
  new_encoded_df = scaler.inverse_transform(encoded_df)

  enc = new_encoded_df.to_csv('Encoded.csv')
  dec = new_decoded_df.to_csv('Decoded.csv')

  return new_decoded_df, new_encoded_df


def compute_error(csv_path,csv_path_1):

  df = pd.read_csv(csv_path)
  df1 = pd.read_csv(csv_path_1)

  df1.columns=['ch1','ch2','ch3']

  df1['mag1'] = np.sqrt(df1['ch1']**2 + df1['ch2']**2 + df1['ch3']**2)
  df1['mag2'] = np.sqrt(df['ch1']**2 + df['ch2']**2 + df['ch3']**2)

  df1['diff'] = np.abs(df1['mag1']**2 - df1['mag2']**2)
  df1['org'] = df1['mag2']**2

  df1['final'] = df1['diff'] / df1['org']

  return (df1['final'].mean())