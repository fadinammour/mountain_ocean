import requests

path_folder = '/Users/fnammour/Documents/Administrative/CV/private_sector/Ideta/technical_test/mountain_ocean/_static/img/'

path_img = path_folder + 'sample_file.jpg'

resp = requests.post("http://127.0.0.1:5000/predict",
                     files={"file": open(path_img,'rb')})

print(resp.json())