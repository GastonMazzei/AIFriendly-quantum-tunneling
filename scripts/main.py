from datasets_generator import *

df = main(20000, False,500)

print(df.head)

df.to_csv('local-data.csv')
