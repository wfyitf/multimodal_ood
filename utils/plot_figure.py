import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})
plt.rcParams['figure.dpi'] = 150 

save_path = 'figures/ood_detection/ood_detection_cosine'

def plot_cosine(df_table, save_fig = False, save_name = None, save_format = 'png', dpi = 300):
    # Create figure and axes for a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Data for histograms
    columns = ['image_score', 'image_score_max', 'dialogue_score', 'dialogue_score_max'] #'dialogue_score_segment', 'dialogue_score_segment_max']
    titles = ['Image Scores Distribution', 'Image Max Scores Distribution',
            'Dialogue Sum Scores Distribution', 'Dialogue Max Scores Distribution']
            #'Dialogue Segment Sum Scores Distribution', 'Dialogue Segment Max Scores Distribution']
    x_labels = ['Image Score', 'Image Max Score', 'Dialogue Score', 'Dialogue Max Score'] # 'Dialogue Segment Score', 'Dialogue Segment Max Score']

    # Loop through to plot each histogram in its subplot
    for i, ax in enumerate(axs.flatten()):
        ood_scores = df_table[df_table['OOD'] == 0][columns[i]]
        non_ood_scores = df_table[df_table['OOD'] == 1][columns[i]]
        sns.histplot(non_ood_scores, bins=80, alpha=0.5, label='ID', kde=True, color='blue', ax=ax, stat="density")
        sns.histplot(ood_scores, bins=80, alpha=0.5, label='OOD', kde=True, color='red', ax=ax, stat="density")
        ax.legend(loc='upper right')
        ax.set_title(titles[i])
        ax.set_xlabel(x_labels[i])
        ax.set_ylabel('Density')

    plt.tight_layout()

    if save_fig:
        save_path = save_path + '/' + save_name + '.' + save_format
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()