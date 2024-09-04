import yaml
import zipfile
from pathlib import Path

def unzip(source, destination):
    zip_ref = zipfile.ZipFile(source)
    zip_ref.extractall(destination)
    zip_ref.close()


if __name__ == '__main__':
    
    params_file = 'params.yaml'
    with open(params_file) as f:
        yaml_file = yaml.safe_load(f)

    source = yaml_file['data_unzip']['source']
    destination = yaml_file['data_unzip']['destination']

    unzip(source, destination)

    print('Unzipped the data..')