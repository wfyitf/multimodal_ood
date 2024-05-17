import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})
plt.rcParams['figure.dpi'] = 150 

save_path = '../figures/example'

def plot_cosine(df_table, save_fig = False, save_format = 'png', dpi = 300):
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
        save_path_final = save_path + '/cosine_origin.' + save_format
        plt.savefig(save_path_final, dpi=dpi, bbox_inches='tight')

    plt.show()


def plot_score_distribution(df_test, score_type, type, fpr, mode, save_fig = False, save_format = 'png', dpi = 300, include_fpr = True, show_plot = True):
    
    if mode != 'overall':
        ood_scores = df_test[df_test['OOD'] == 0][f'{score_type}_{type}_{mode}']
        non_ood_scores = df_test[df_test['OOD'] == 1][f'{score_type}_{type}_{mode}']
    else:
        ood_scores = df_test[df_test['OOD'] == 0][f'{score_type}_{mode}_simialrity_{type}']
        non_ood_scores = df_test[df_test['OOD'] == 1][f'{score_type}_{mode}_simialrity_{type}']

    sns.histplot(non_ood_scores, bins=80, alpha=0.5, label='ID', kde=True, color='blue',stat="density")
    sns.histplot(ood_scores, bins=80, alpha=0.5, label='OOD', kde=True, color='red', stat="density")
    
    if score_type == 'cosine':
        score_name = 'Cosine'
    elif score_type == "mp":
        score_name = "Probability"
    elif score_type == "msp":
        score_name = "Softmax Probability"
    elif score_type == "entropy":
        score_name = "Entropy"
    elif score_type == "logits":
        score_name = "Logits"
    

    plt.title(f'{mode.title()} {score_name} Distribution')
    plt.xlabel(f'{mode.title()} {type.title()} Score')
    plt.ylabel('Probability Density')
    hist_id, bins_id = np.histogram(non_ood_scores, bins=80, density=True)
    cumulative_id = np.cumsum(hist_id * np.diff(bins_id))
    threshold_index_id = np.where(cumulative_id >= (1 - fpr/100))[0][0]
    threshold_value_id = bins_id[threshold_index_id]
    hist_ood, bins_ood = np.histogram(ood_scores, bins=80, density=True)
    cumulative_ood = np.cumsum(hist_ood * np.diff(bins_ood))
    threshold_index_ood = np.searchsorted(bins_ood, threshold_value_id) - 1
    cumulative_probability_ood = cumulative_ood[threshold_index_ood]

    if include_fpr:
        plt.axvline(x=threshold_value_id, color='green', linestyle='--', label=f'{fpr}\% Recall at {threshold_value_id:.2f}')
        plt.axvline(x=threshold_value_id, color='purple', linestyle=':', label=f'FPR{fpr} with {(1 - cumulative_probability_ood):.2f}')

    plt.legend()
    plt.show()
    if save_fig:
        save_path_final = save_path + f'/score_distribution_{score_type}_{type}_{mode}.' + save_format
        plt.savefig(save_path_final, dpi=dpi, bbox_inches='tight')

    if show_plot:
        plt.show()


def plot_kde_joint_distribution(df_test, score_type, type, mode, save_fig = False, save_format = 'png', dpi = 300, show_plot = True):

    fig, ax = plt.subplots()
    sns.kdeplot(
        x=df_test[df_test['OOD'] == 0][f'{score_type}_{type}_{mode}'],
        y=2*df_test[df_test['OOD'] == 0]['image_text_similarity'],
        cmap='Reds',
        fill=True,
        thresh=0,
        levels=10,
        alpha=1,
        ax=ax
    )
    sns.kdeplot(
        x=df_test[df_test['OOD'] == 1][f'{score_type}_{type}_{mode}'],
        y=2*df_test[df_test['OOD'] == 1]['image_text_similarity'],
        cmap='Blues',
        fill=True,
        thresh=0,
        levels=10,
        alpha=0.6,
        ax=ax
    )
    ax.set_xlabel(f'$s_T(x_T, y)$', fontsize = 30)
    ax.set_ylabel(f'$s(x_I, x_T)$', fontsize = 30)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)

    if save_fig:
        save_path_final = save_path + f'/kde_joint_distribution_{score_type}_{type}_{mode}.' + save_format
        plt.savefig(save_path_final, dpi=dpi, bbox_inches='tight')
    
    if show_plot:
        plt.show()


def plot_rough_distribution(df_test, field_name, save_fig = False, save_format = 'png', dpi = 300, show_plot = True):

    ood_scores = df_test[df_test['OOD'] == 0][field_name]
    non_ood_scores = df_test[df_test['OOD'] == 1][field_name]

    fig, ax = plt.subplots(figsize=(10, 2))
    sns.histplot(
        non_ood_scores, 
        bins=80, 
        alpha=0.5, 
        label='ID', 
        kde=False, 
        color='blue', 
        stat="density",
        ax=ax
    )

    sns.histplot(
        ood_scores, 
        bins=80, 
        alpha=0.5, 
        label='OOD', 
        kde=False, 
        color='red', 
        stat="density",
        ax=ax
    )

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(axis='y', which='both', left=False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xlabel('')
    ax.set_ylabel('')

    if save_fig:
        save_path_final = save_path + f'/rough_distribution_{field_name}.' + save_format
        plt.savefig(save_path_final, dpi=dpi, bbox_inches='tight')

    if show_plot:
        plt.show()


def plot_joint_distribution(df_test, score_type, type, mode, id = True, color = 'blue', save_fig = False, save_format = 'png', dpi = 300, show_plot = True):
    sns.jointplot(x=f'{score_type}_{type}_{mode}', y=f'image_text_similarity', data=df_test[df_test['OOD'] == id], kind='hex', color=color)
    if id:
        OOD = "ID"
    else:
        OOD = "OOD"
    plt.suptitle(f'Joint Distribution of Image Score and Image-Text Similarity For {OOD} Data', y=1.02)
    plt.xlabel(f'{mode.title()} {type.title()} Score')
    plt.ylabel('Image-Text Similarity')
    if save_fig:
        save_path_final = save_path + f'/joint_distribution_{score_type}_{type}_{mode}.' + save_format
        plt.savefig(save_path_final, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_example_label(save_fig = False, save_format = 'png', dpi = 300, show_plot = True):
    categories = ['Dog', 'Cat', 'Car', 'Kitchen', 'Bedroom']
    ind_bars = [0.1, 0.9, 0.1, 0.2, 0.3]
    ood_bars_1 = [0.1, 0.1, 0.2, 0.1, 0.1]
    ood_bars_2 = [0.15, 0.10, 0.15, 0.13, 0.18]
    # ind_bars = [0.3, 0.8, 0.1, 0.25, 0.3]
    # ood_bars_1 = [0.2, 0.75, 0.05, 0.2, 0.7]
    # ood_bars_2 = [0.1, 0.1, 0.2, 0.1, 0.1]
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)


    rects1 = ax.bar(x - width, ind_bars, width, label='Example 1', color='blue', alpha=0.8, edgecolor='darkblue')
    rects2 = ax.bar(x, ood_bars_1, width, label='Example 2', color='red', alpha=0.8, edgecolor='darkred')
    rects3 = ax.bar(x + width, ood_bars_2, width, label='Example 3', color='lightcoral', alpha=0.8, edgecolor='darkred')

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=30)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    max_ind_index = ind_bars.index(max(ind_bars))
    max_ood_1_index = ood_bars_1.index(max(ood_bars_1))
    max_ood_2_index = ood_bars_2.index(max(ood_bars_2))

    rects1[max_ind_index].set_edgecolor('green')
    rects1[max_ind_index].set_linewidth(5)

    rects2[max_ood_1_index].set_edgecolor('green')
    rects2[max_ood_1_index].set_linewidth(5)

    rects3[max_ood_2_index].set_edgecolor('green')
    rects3[max_ood_2_index].set_linewidth(5)

    ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    if save_fig:
        save_path_final = save_path + '/example_label.' + save_format
        plt.savefig(save_path_final, dpi=dpi, bbox_inches='tight')

    plt.show()