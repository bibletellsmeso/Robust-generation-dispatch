import pandas as pd

def text_to_csv(input_txt_file, output_csv_file):
    with open(input_txt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [line.strip().split(',') for line in lines]
    df = pd.DataFrame(data)
    df.to_csv(output_csv_file, index=False, header=False)

text_to_csv('/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/data/20250122_Hawaii_weather.txt', '/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/data/20250122_Hawaii_weather.csv')