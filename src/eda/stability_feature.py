import os
import warnings

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings(action='ignore', category=FutureWarning)
plt.ioff()

def line_plot(data: pl.DataFrame, col: str) -> plt.figure:
    if data.select(pl.col(col).is_null().mean()).item() == 1.:
        data = data.with_columns(
            pl.lit(0).alias(col)
        )
        suffix: str = '  is all empty'
    else:
        suffix: str = ''
        
    plt.figure(figsize=(12,8))
    fig = sns.lineplot(
        data=data, 
        x="date_decision", y=col
    )
    plt.title(col + suffix)
    return fig

def save_multiple_line_plot(dataset: pl.DataFrame, dataset_name: str, save_path: str) -> None:
    assert isinstance(dataset, pl.DataFrame)
    
    save_path_file = os.path.join(
        save_path, dataset_name + '.pdf'
    )

    predictor_ = [col for col in dataset.columns if dataset_name in col]
    
    data_selected = dataset.group_by('date_decision').agg(
        pl.col(col).mean()
        for col in predictor_
    )

    with PdfPages(save_path_file) as export_pdf:
        
        for col in predictor_:
            fig = line_plot(data_selected, col=col)
            export_pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
