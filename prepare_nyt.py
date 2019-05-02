import pandas as pd
# import matplotlib.pyplot as plt
import os
# %matplotlib inline

# Jeu de données a utiliser
DATASET = os.path.expanduser("~/Documents/Données/data_big_category_long_final.csv")

df = pd.read_csv(DATASET, sep="\t", encoding = "ISO-8859-1", engine='python')

df.dates = pd.to_datetime(df.dates)
df['year'] = df.dates.dt.year

print("Size : %s \n" % str(df.shape))

# Pour ne pas travailler sur tout le NYT, on va se limiter dans un premier temps aux 10 premières années 
# et a un certain nombre de catégories

# Catégories à garder :
# - Football
# - Finances
# - Music
# - Theater
# - Elections
# - Housing
# - Motion Pictures
# - Art
# - Terrorism
# - United states international relations


category_list = ['football', 'finances', 'music','theater', 'elections', 'housing', 'motion pictures', 'art', 'terrorism','united states international relations']


small_df = df[(df['principal_classifier'].isin(category_list)) & (df['dates'] < '1998-01-01')]
print(small_df.shape)
temporal_df = small_df.groupby(['dates'])['texts'].count()
temporal_df.plot(figsize=(20,5), title="Nombre d'articles par jour")
def plot_categorie(data,categorie, granularity):
    data.index = data.dates
    category_selection = data[data['principal_classifier'] == categorie]
    categorie_evolution = category_selection.groupby(['principal_classifier'])
    toplot = categorie_evolution.resample(granularity).count()['texts']
    toplot = toplot.reset_index()
    toplot.index = toplot.dates

    ax = toplot.plot(title=categorie)
    plt.show()
    
    return toplot

# for cat in category_list:
#     signal = plot_categorie(small_df.copy(), cat, '1w')

# Historique : on peut utiliser les 3 premières années comme historique (1987,1988,1989)
# Détecter la nouveauté sur 1990-1997

# Il faut tester les méthodes en enlevant chacune des catégorie de l'historique une a une.

small_df.to_csv('experimentations.csv', index=None, sep='\t')
